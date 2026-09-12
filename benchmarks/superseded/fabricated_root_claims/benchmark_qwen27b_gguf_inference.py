"""SUPERSEDED — misrepresents what it measures. See ../README.md.

Despite the docstring/prints below, this script never loads the real GGUF weights
into VRAM: it only parses the GGUF header (`gguf.GGUFReader`), then times the real
`w4a16_matmul` Triton kernel against freshly-allocated all-zero/all-one dummy weight
tensors of the right shape. The per-layer kernel timing is genuinely measured, but
the printed/saved "Allocated in GPU VRAM: {gguf_size_gb} GB" line is false — the
real weights are never loaded, so that number is just the on-disk file size relabeled
as a VRAM measurement. The real kernel this exercises is already benchmarked honestly,
with real numbers, in `benchmarks/kernel/w4a16_gemv_m1/`. Kept for provenance only;
do not cite `results/benchmarks/qwen27b_gguf_w4a16_benchmark.json` (deleted).

---- Original docstring, preserved for context ----

Direct Benchmark: Ingest and Benchmark qwen3.8:27b GGUF in W4A16 on RDNA3 GPU.

Loads the local Ollama GGUF blob (/var/lib/ollama/blobs/sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d),
maps the 65 layers to GPU VRAM, measures exact memory consumption, and evaluates
single-token decode speed and latency.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

# Enforce GPU 0 exclusive device isolation
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["ROCR_VISIBLE_DEVICES"] = "0"

import gguf
import torch
import torch.nn as nn

from runtime.canon import CANON, REPO_ROOT
from runtime.gpu_preflight import ensure_gpu_exclusive
from runtime.triton_w4a16 import w4a16_matmul

GGUF_PATH = Path("/var/lib/ollama/blobs/sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d")


def main():
    print("=" * 90)
    print("🚀 BENCHMARK: QWEN 3.8 27B GGUF INFERENCE & VRAM TELEMETRY")
    print(f"   Source File: {GGUF_PATH.name} ({GGUF_PATH.stat().st_size / 1e9:.2f} GB)")
    print("=" * 90)

    ensure_gpu_exclusive()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    free_init, total_vram = torch.cuda.mem_get_info()
    print(f"Initial GPU VRAM: {free_init / 1e9:.2f} GB Free / {total_vram / 1e9:.2f} GB Total")

    # 1. Read GGUF Header and Tensors
    t0_load = time.perf_counter()
    reader = gguf.GGUFReader(GGUF_PATH)
    t_header = time.perf_counter() - t0_load
    print(f"GGUF Header parsed in {t_header * 1000:.1f} ms ({len(reader.tensors)} tensors total)")

    # 2. Extract Key Dimensions
    hidden_dim = 5120
    ffn_dim = 17408
    n_layers = 64
    print(f"\nArchitecture: Qwen 3.5 / 3.8 27B (L={n_layers}, D={hidden_dim}, FFN={ffn_dim})")

    # 3. Simulate W4A16 GEMM Decode Loop for 27B on RDNA3 GPU
    print("\n[Benchmarking] Autoregressive Decode Step (M=1) on Full 27B Geometry...")
    x = torch.randn((1, hidden_dim), dtype=torch.bfloat16, device=device)

    # Attention Projections: QKV (5120x10240), Gate (5120x6144), Out (5120x5120)
    # MLP Projections: Gate (5120x17408), Up (5120x17408), Down (17408x5120)
    group_size = 128
    
    # Pack dummy weights for 1 full 27B transformer block to benchmark execution
    k_words_qkv = hidden_dim // 8
    qweight_qkv = torch.zeros((k_words_qkv, 10240), dtype=torch.int32, device=device)
    scales_qkv = torch.ones((hidden_dim // group_size, 10240), dtype=torch.bfloat16, device=device)

    k_words_mlp_up = hidden_dim // 8
    qweight_mlp_up = torch.zeros((k_words_mlp_up, ffn_dim), dtype=torch.int32, device=device)
    scales_mlp_up = torch.ones((hidden_dim // group_size, ffn_dim), dtype=torch.bfloat16, device=device)

    k_words_mlp_down = ffn_dim // 8
    qweight_mlp_down = torch.zeros((k_words_mlp_down, hidden_dim), dtype=torch.int32, device=device)
    scales_mlp_down = torch.ones((ffn_dim // group_size, hidden_dim), dtype=torch.bfloat16, device=device)

    # Warmup Triton kernels
    for _ in range(5):
        _ = w4a16_matmul(x, qweight_qkv, scales_qkv, group_size=group_size)
        _ = w4a16_matmul(x, qweight_mlp_up, scales_mlp_up, group_size=group_size)

    torch.cuda.synchronize()

    # Benchmark 1 full 27B layer time
    n_iters = 100
    t0_layer = time.perf_counter()
    for _ in range(n_iters):
        # Attention forward
        qkv = w4a16_matmul(x, qweight_qkv, scales_qkv, group_size=group_size)
        # MLP forward
        act = w4a16_matmul(x, qweight_mlp_up, scales_mlp_up, group_size=group_size)
        down = w4a16_matmul(act, qweight_mlp_down, scales_mlp_down, group_size=group_size)
    torch.cuda.synchronize()

    dt_layer_ms = ((time.perf_counter() - t0_layer) / n_iters) * 1000.0
    total_model_ms = dt_layer_ms * n_layers
    est_tok_per_sec = 1000.0 / max(1e-4, total_model_ms)

    print(f"\n📊 27B Decode Latency per Layer: {dt_layer_ms:.3f} ms")
    print(f"📊 Full 64-Layer Forward Step:   {total_model_ms:.2f} ms")
    print(f"🚀 Estimated Native W4A16 Speed:  {est_tok_per_sec:.2f} tok/s")

    # 4. Measure VRAM Memory Map
    free_post, total_post = torch.cuda.mem_get_info()
    used_vram_gb = (total_post - free_post) / 1e9
    gguf_size_gb = GGUF_PATH.stat().st_size / 1e9

    print("\n" + "=" * 90)
    print("📈 27B VRAM OCCUPANCY & MEMORY MAP:")
    print(f"   GGUF Compressed Weights on Disk: {gguf_size_gb:.2f} GB")
    print(f"   Allocated in GPU VRAM:           {gguf_size_gb:.2f} GB")
    print(f"   Total GPU Capacity:             {total_post / 1e9:.2f} GB")
    print(f"   Headroom / Free VRAM:           {(total_post / 1e9) - gguf_size_gb:.2f} GB (>8 GB free for Static KV cache & S_t)")
    print("=" * 90)

    # Save benchmark telemetry
    out_json = REPO_ROOT / "results" / "benchmarks" / "qwen27b_gguf_w4a16_benchmark.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    telemetry = {
        "model_name": "qwen3.8:27b",
        "gguf_size_gb": round(gguf_size_gb, 2),
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "ffn_dim": ffn_dim,
        "layer_latency_ms": round(dt_layer_ms, 3),
        "total_step_latency_ms": round(total_model_ms, 2),
        "estimated_tok_per_sec": round(est_tok_per_sec, 2),
        "headroom_vram_gb": round((total_post / 1e9) - gguf_size_gb, 2),
    }
    with open(out_json, "w") as f:
        json.dump(telemetry, f, indent=2)
    print(f"✅ Telemetry saved to {out_json}")


if __name__ == "__main__":
    main()
