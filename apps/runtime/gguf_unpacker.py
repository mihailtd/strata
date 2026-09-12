"""GGUF Streaming Tensor Unpacker & W4A16 Triton Packer.

Extracts quantized weights (Q4_K, Q6_K) and dense parameters (F32) from GGUF blobs
directly into 128-bit memory-coalesced W4A16 format for custom AMD RDNA3 Triton execution.

Features:
  - Layer-by-layer streaming to keep host RAM < 1.5 GB.
  - Multi-process parallel conversion using ProcessPoolExecutor.
  - Persistent disk cache in `models/qwen3.8-27b-triton/` for instant (<3s) subsequent reloads.
  - Auto-transposition from GGUF (N, K) to Triton GEMV (K, N) where K % 128 == 0.
"""

from __future__ import annotations

import concurrent.futures
import gc
import json
import time
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch

from runtime.triton_w4a16 import quantize_and_pack_w4

DEFAULT_GGUF_PATH = "/var/lib/ollama/blobs/sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "qwen3.8-27b-triton"


class GGUFStreamingUnpacker:
    """Stream-extracts GGUF quantized tensors into Triton W4A16 format with disk caching."""

    def __init__(
        self,
        gguf_path: str | Path = DEFAULT_GGUF_PATH,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
    ):
        self.gguf_path = Path(gguf_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_metadata(self) -> dict[str, Any]:
        """Reads architecture hyperparameters directly from GGUF metadata fields."""
        reader = gguf.GGUFReader(str(self.gguf_path))
        meta: dict[str, Any] = {}

        field_map = {
            "qwen35.block_count": "num_layers",
            "qwen35.embedding_length": "hidden_dim",
            "qwen35.feed_forward_length": "ffn_dim",
            "qwen35.attention.head_count": "num_heads_q",
            "qwen35.attention.head_count_kv": "num_heads_kv",
            "qwen35.attention.key_length": "head_dim",
            "qwen35.full_attention_interval": "full_attention_interval",
            "qwen35.context_length": "context_length",
            "qwen35.ssm.inner_size": "ssm_inner_size",
            "qwen35.ssm.time_step_rank": "ssm_time_step_rank",
            "qwen35.ssm.conv_kernel": "ssm_conv_kernel",
            "qwen35.ssm.state_size": "ssm_state_size",
            "qwen35.nextn_predict_layers": "nextn_predict_layers",
        }

        for f in reader.fields.values():
            if f.name in field_map:
                val = f.parts[f.data[0]] if f.data else None
                if isinstance(val, (np.ndarray, list)):
                    val = int(val[0])
                meta[field_map[f.name]] = val

        # Fallbacks if metadata not explicitly present
        meta.setdefault("num_layers", 65)
        meta.setdefault("hidden_dim", 5120)
        meta.setdefault("ffn_dim", 17408)
        meta.setdefault("num_heads_q", 24)
        meta.setdefault("num_heads_kv", 4)
        meta.setdefault("head_dim", 256)
        meta.setdefault("full_attention_interval", 4)
        meta.setdefault("vocab_size", 248320)
        meta.setdefault("group_size", 128)

        return meta

    @staticmethod
    def _convert_tensor_to_w4a16(
        data: np.ndarray,
        tensor_type: int,
        group_size: int = 128,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Dequantizes GGUF tensor and packs into 128-bit W4A16 format."""
        if tensor_type in (gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K):
            dequant = gguf.quants.dequantize(data, tensor_type)
            # Transpose GGUF (N, K) -> Triton (K, N)
            w = torch.from_numpy(dequant.T).to(dtype=torch.bfloat16)
            if device != "cpu" and torch.cuda.is_available():
                w = w.to(device)
            qweight, scales = quantize_and_pack_w4(w, group_size=group_size)
            return {
                "qweight": qweight.cpu(),
                "scales": scales.cpu(),
                "shape": (w.shape[0], w.shape[1]),
                "type": "w4a16",
            }
        else:
            w = torch.from_numpy(np.array(data, copy=True)).to(dtype=torch.float32)
            return {
                "weight": w.cpu(),
                "shape": tuple(w.shape),
                "type": "dense",
            }

    def unpack_layer(
        self,
        layer_idx: int,
        group_size: int = 128,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Unpacks all tensors for a single block layer (e.g. blk.0.*)."""
        reader = gguf.GGUFReader(str(self.gguf_path))
        prefix = f"blk.{layer_idx}."
        layer_tensors = [t for t in reader.tensors if t.name.startswith(prefix)]

        packed: dict[str, Any] = {}
        for t in layer_tensors:
            name_short = t.name[len(prefix) :]
            converted = self._convert_tensor_to_w4a16(
                t.data,
                t.tensor_type,
                group_size=group_size,
                device=device,
            )
            packed[name_short] = converted

        return packed

    def unpack_globals(self, group_size: int = 128) -> dict[str, Any]:
        """Unpacks global embedding, output norm, and LM head tensors."""
        reader = gguf.GGUFReader(str(self.gguf_path))
        globals_dict: dict[str, Any] = {}

        for t in reader.tensors:
            if not t.name.startswith("blk."):
                if t.name == "token_embd.weight":
                    dequant = gguf.quants.dequantize(t.data, t.tensor_type)
                    # Shape is (vocab_size=248320, hidden_dim=5120)
                    globals_dict[t.name] = {
                        "weight": torch.from_numpy(dequant).to(torch.bfloat16),
                        "shape": tuple(dequant.shape),
                        "type": "embedding",
                    }
                elif t.name == "output.weight":
                    dequant = gguf.quants.dequantize(t.data, t.tensor_type)
                    # Transpose (248320, 5120) -> (5120, 248320) for Triton GEMV
                    w = torch.from_numpy(dequant.T).to(dtype=torch.bfloat16)
                    qweight, scales = quantize_and_pack_w4(w, group_size=group_size, chunk_size=8192)
                    globals_dict[t.name] = {
                        "qweight": qweight,
                        "scales": scales,
                        "shape": (w.shape[0], w.shape[1]),
                        "type": "w4a16",
                    }
                else:
                    converted = self._convert_tensor_to_w4a16(
                        t.data,
                        t.tensor_type,
                        group_size=group_size,
                        device="cpu",
                    )
                    globals_dict[t.name] = converted

        return globals_dict

    def is_cache_complete(self, num_layers: int = 65) -> bool:
        """Checks if all layer checkpoints and metadata exist in cache directory."""
        if not (self.cache_dir / "metadata.json").exists():
            return False
        if not (self.cache_dir / "globals.pt").exists():
            return False
        for i in range(num_layers):
            if not (self.cache_dir / f"layer_{i}.pt").exists():
                return False
        return True

    def convert_and_cache_all(
        self,
        max_workers: int = 4,
        force: bool = False,
        progress_cb: Any | None = None,
    ) -> None:
        """Executes streamed conversion of all layers and globals to disk cache."""
        meta = self.get_metadata()
        num_layers = meta["num_layers"]

        if not force and self.is_cache_complete(num_layers):
            print(f"[GGUF Unpacker] Cache already complete at {self.cache_dir}")
            return

        print(f"[GGUF Unpacker] Starting conversion: {num_layers} layers from {self.gguf_path.name}")
        t0 = time.perf_counter()

        # Save metadata
        with open(self.cache_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # 1. Unpack globals first if not present
        globals_file = self.cache_dir / "globals.pt"
        if force or not globals_file.exists():
            t_glob = time.perf_counter()
            print("[GGUF Unpacker] Unpacking global embeddings & LM head...")
            glob_dict = self.unpack_globals(group_size=meta.get("group_size", 128))
            torch.save(glob_dict, globals_file)
            print(f"[GGUF Unpacker] Globals saved in {time.perf_counter() - t_glob:.2f}s")

        # 2. Unpack layers in parallel workers to keep RAM bounded
        group_sz = meta.get("group_size", 128)
        tasks = [(str(self.gguf_path), str(self.cache_dir), i, group_sz, force) for i in range(num_layers)]

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_worker_unpack_layer, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                layer_done = future.result()
                if progress_cb:
                    progress_cb(layer_done)

        total_elapsed = time.perf_counter() - t0
        print(
            f"[GGUF Unpacker] Finished converting {num_layers} layers in {total_elapsed:.1f}s ({total_elapsed / 60:.2f} min)."
        )

    def load_layer(self, layer_idx: int, device: str = "cuda:0") -> dict[str, Any]:
        """Fast memory-mapped loading of a single layer into target device."""
        layer_file = self.cache_dir / f"layer_{layer_idx}.pt"
        if not layer_file.exists():
            raise FileNotFoundError(f"Layer checkpoint not found at {layer_file}. Run convert_and_cache_all() first.")

        data = torch.load(layer_file, map_location=device, weights_only=False)
        return data

    def load_globals(self, device: str = "cuda:0") -> dict[str, Any]:
        """Fast loading of global embedding, output norm, and LM head."""
        globals_file = self.cache_dir / "globals.pt"
        if not globals_file.exists():
            raise FileNotFoundError(f"Globals checkpoint not found at {globals_file}.")
        return torch.load(globals_file, map_location=device, weights_only=False)


def _worker_unpack_layer(args: tuple[str, str, int, int, bool]) -> int:
    gguf_path, cache_dir, layer_i, group_size, force = args
    out_file = Path(cache_dir) / f"layer_{layer_i}.pt"
    if not force and out_file.exists():
        return layer_i
    t_start = time.perf_counter()
    unpacker = GGUFStreamingUnpacker(gguf_path, cache_dir)
    layer_dict = unpacker.unpack_layer(
        layer_i,
        group_size=group_size,
        device="cpu",
    )
    torch.save(layer_dict, out_file)
    elapsed = time.perf_counter() - t_start
    print(f"  [GGUF Unpacker] Layer {layer_i:02d} converted & cached ({elapsed:.1f}s)")
    gc.collect()
    return layer_i
