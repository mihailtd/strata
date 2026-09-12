"""W4A16 (INT4 Weight, BF16 Activation) Model Loader and Layer Engine for 27B/32B Scaling on 24 GB VRAM.

Replaces standard nn.Linear layers with W4A16Linear modules that execute:
  1. Base GEMM: dequantized directly in GPU registers feeding RDNA3 WMMA hardware tiles.
  2. Fused Dynamic LoRA: in-register low-rank accumulation (Out = X @ dequant(W) + alpha * (X @ A) @ B).
  3. Layer-by-layer streamed loading & quantization to prevent host/GPU memory spikes.

Vendored into runtime-triton for self-sufficiency -- see native_27b_engine.py's docstring in this same directory for why. Do not re-link to apps/runtime.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from triton_w4a16 import (
    fused_w4a16_lora_matmul,
    quantize_and_pack_w4,
    w4a16_matmul,
)


class W4A16Linear(nn.Module):
    """Drop-in replacement for nn.Linear executing RDNA3 register-level W4A16 GEMM with fused LoRA."""

    in_features: int
    out_features: int
    group_size: int
    qweight: torch.Tensor
    scales: torch.Tensor
    bias: torch.Tensor | None
    lora_a: torch.Tensor | None
    lora_b: torch.Tensor | None
    lora_alpha: float
    max_lora_rank: int
    static_lora_a: torch.Tensor | None
    static_lora_b: torch.Tensor | None
    has_active_lora: bool

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        return super().__call__(x, out=out)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        group_size: int = 128,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        assert in_features % 8 == 0, f"in_features ({in_features}) must be divisible by 8"
        assert in_features % group_size == 0, (
            f"in_features ({in_features}) must be divisible by group_size ({group_size})"
        )

        # Packed INT4 weights: (K // 8, N) in int32
        k_words = in_features // 8
        n_groups = in_features // group_size

        self.register_buffer(
            "qweight",
            torch.zeros((k_words, out_features), dtype=torch.int32, device=device),
        )
        self.register_buffer(
            "scales",
            torch.ones((n_groups, out_features), dtype=torch.bfloat16, device=device),
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros((out_features,), dtype=torch.bfloat16, device=device),
            )
        else:
            self.bias = None

        # Dynamic LoRA Adapter Branch (in GPU memory / L2 cache)
        self.lora_a: torch.Tensor | None = None  # Shape: (in_features, r) in bfloat16
        self.lora_b: torch.Tensor | None = None  # Shape: (r, out_features) in bfloat16
        self.lora_alpha: float = 1.0

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
        device: torch.device | None = None,
    ) -> W4A16Linear:
        """Quantizes an existing unquantized nn.Linear into a W4A16Linear module."""
        w = linear.weight.detach().to(dtype=torch.bfloat16)
        if device is not None:
            w = w.to(device)

        # Transpose to (K, N) where K=in_features, N=out_features
        w_kn = w.t().contiguous()
        qw, scales = quantize_and_pack_w4(w_kn, group_size=group_size)

        mod = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            group_size=group_size,
            device=device or w.device,
        )
        mod.qweight.copy_(qw)
        mod.scales.copy_(scales)
        if linear.bias is not None and mod.bias is not None:
            mod.bias.copy_(linear.bias.detach().to(dtype=torch.bfloat16, device=device or w.device))

        return mod

    @classmethod
    def from_packed(
        cls,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None = None,
        group_size: int = 128,
        device: torch.device | None = None,
    ) -> W4A16Linear:
        """Instantiates W4A16Linear directly from pre-quantized qweight and scales."""
        k_words, out_features = qweight.shape
        in_features = k_words * 8
        target_device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        mod = cls(
            in_features=in_features,
            out_features=out_features,
            bias=bias is not None,
            group_size=group_size,
            device=target_device,
        )
        mod.qweight.copy_(qweight.to(device=target_device))
        mod.scales.copy_(scales.to(device=target_device))
        if bias is not None and mod.bias is not None:
            mod.bias.copy_(bias.to(device=target_device))
        return mod

    def init_static_lora_buffer(self, max_rank: int = 16) -> None:
        """Preallocates fixed-address GPU memory buffers for LoRA adapter weights.
        Enables permanent validity of HIP graphs across dynamic adapter hot-swaps.
        """
        self.max_lora_rank = max_rank
        target_device = self.qweight.device
        self.static_lora_a = torch.zeros((self.in_features, max_rank), dtype=torch.bfloat16, device=target_device)
        self.static_lora_b = torch.zeros((max_rank, self.out_features), dtype=torch.bfloat16, device=target_device)
        self.lora_a = self.static_lora_a
        self.lora_b = self.static_lora_b
        self.lora_alpha = 1.0
        self.has_active_lora = False

    def set_lora_adapter(
        self,
        lora_a: torch.Tensor | None,
        lora_b: torch.Tensor | None,
        alpha: float = 1.0,
    ) -> None:
        """Sets or clears the active LoRA adapter branch for this linear module."""
        if lora_a is not None and lora_b is not None:
            assert lora_a.shape[0] == self.in_features, f"LoRA A input dim {lora_a.shape[0]} != {self.in_features}"
            assert lora_b.shape[1] == self.out_features, f"LoRA B output dim {lora_b.shape[1]} != {self.out_features}"
            assert lora_a.shape[1] == lora_b.shape[0], f"Rank mismatch: A is {lora_a.shape}, B is {lora_b.shape}"
            r = lora_a.shape[1]
            if (
                hasattr(self, "static_lora_a")
                and self.static_lora_a is not None
                and self.static_lora_b is not None
                and r <= self.max_lora_rank
            ):
                self.static_lora_a.zero_()
                self.static_lora_b.zero_()
                self.static_lora_a[:, :r].copy_(lora_a.to(dtype=torch.bfloat16, device=self.qweight.device))
                # Fold alpha directly into lora_b so kernel launch scalar remains constant (1.0)
                self.static_lora_b[:r, :].copy_((lora_b * alpha).to(dtype=torch.bfloat16, device=self.qweight.device))
                self.lora_a = self.static_lora_a
                self.lora_b = self.static_lora_b
                self.lora_alpha = 1.0
                self.has_active_lora = True
            else:
                self.lora_a = lora_a.to(dtype=torch.bfloat16, device=self.qweight.device)
                self.lora_b = lora_b.to(dtype=torch.bfloat16, device=self.qweight.device)
                self.lora_alpha = alpha
                self.has_active_lora = True
        else:
            self.clear_lora()

    def clear_lora(self) -> None:
        """Disengages active LoRA branch."""
        if hasattr(self, "static_lora_a") and self.static_lora_a is not None and self.static_lora_b is not None:
            self.static_lora_a.zero_()
            self.static_lora_b.zero_()
            self.lora_alpha = 1.0
            self.has_active_lora = False
        else:
            self.lora_a = None
            self.lora_b = None
            self.lora_alpha = 1.0
            self.has_active_lora = False

    def forward(self, x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """Executes fused W4A16 matrix multiplication."""
        x_bf16 = x.to(torch.bfloat16) if x.dtype != torch.bfloat16 else x

        if hasattr(self, "static_lora_a") and self.static_lora_a is not None:
            # Static buffer path: permanently fixed GPU pointers for zero-recapture HIP graphs
            out_res = fused_w4a16_lora_matmul(
                x_bf16,
                self.qweight,
                self.scales,
                self.static_lora_a,
                self.static_lora_b,
                out=out,
                alpha=1.0,
                group_size=self.group_size,
            )
        elif self.lora_a is not None and self.lora_b is not None:
            out_res = fused_w4a16_lora_matmul(
                x_bf16,
                self.qweight,
                self.scales,
                self.lora_a,
                self.lora_b,
                out=out,
                alpha=self.lora_alpha,
                group_size=self.group_size,
            )
        else:
            out_res = w4a16_matmul(
                x_bf16,
                self.qweight,
                self.scales,
                out=out,
                group_size=self.group_size,
            )

        if self.bias is not None:
            out_res = out_res + self.bias
        return out_res

    @property
    def weight_bytes(self) -> int:
        """Returns total bytes used by this module's packed weights."""
        return (
            (self.qweight.numel() * 4)
            + (self.scales.numel() * 2)
            + (self.bias.numel() * 2 if self.bias is not None else 0)
        )


@dataclass
class ModelQuantizationSummary:
    """Summary of model quantization and VRAM savings."""

    total_linear_layers: int
    unquantized_weight_gb: float
    quantized_weight_gb: float
    compression_ratio: float
    vram_saved_gb: float
    quantization_time_s: float


class W4A16ModelLoader:
    """Utilities for converting, loading, and managing 27B/32B models in W4A16 on 24 GB VRAM."""

    @staticmethod
    def replace_linear_modules(
        module: nn.Module,
        group_size: int = 128,
        skip_modules: list[str] | None = None,
        verbose: bool = False,
    ) -> ModelQuantizationSummary:
        """Recursively replaces all nn.Linear modules with W4A16Linear modules layer-by-layer."""
        skip_modules = skip_modules or ["lm_head"]  # Keep lm_head in bfloat16 for logit fidelity
        t0 = time.perf_counter()

        unquant_bytes = 0
        quant_bytes = 0
        layer_count = 0

        for name, child in list(module.named_children()):
            if any(skip in name for skip in skip_modules):
                if verbose:
                    print(f"  [W4A16 Loader] Skipping module: {name}")
                continue

            if isinstance(child, nn.Linear):
                # Calculate original unquantized bytes (bf16: 2 bytes/param)
                orig_bytes = child.weight.numel() * 2
                if child.bias is not None:
                    orig_bytes += child.bias.numel() * 2

                # Convert to W4A16
                w4_mod = W4A16Linear.from_linear(child, group_size=group_size, device=child.weight.device)
                setattr(module, name, w4_mod)

                unquant_bytes += orig_bytes
                quant_bytes += w4_mod.weight_bytes
                layer_count += 1

                # Clean up original weight to free memory immediately
                del child
            else:
                sub_summary = W4A16ModelLoader.replace_linear_modules(
                    child, group_size=group_size, skip_modules=skip_modules, verbose=verbose
                )
                unquant_bytes += int(sub_summary.unquantized_weight_gb * 1e9)
                quant_bytes += int(sub_summary.quantized_weight_gb * 1e9)
                layer_count += sub_summary.total_linear_layers

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        dt = time.perf_counter() - t0
        unquant_gb = unquant_bytes / 1e9
        quant_gb = quant_bytes / 1e9
        ratio = unquant_gb / max(1e-6, quant_gb)
        saved_gb = unquant_gb - quant_gb

        return ModelQuantizationSummary(
            total_linear_layers=layer_count,
            unquantized_weight_gb=unquant_gb,
            quantized_weight_gb=quant_gb,
            compression_ratio=ratio,
            vram_saved_gb=saved_gb,
            quantization_time_s=dt,
        )

    @staticmethod
    def calculate_27b_32b_memory_map(
        num_layers: int = 64,
        hidden_size: int = 5120,
        intermediate_size: int = 27392,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        max_seq_len: int = 4096,
    ) -> dict[str, Any]:
        """Calculates exact VRAM requirements for 27B/32B model in W4A16 vs BF16."""
        # 1. Weights per layer:
        # Self-attention: Q (5120x5120), K (5120x1024), V (5120x1024), O (5120x5120) -> 62.9M params
        # MLP: Gate (5120x27392), Up (5120x27392), Down (27392x5120) -> 420.7M params
        # Total params per layer = ~483.6M params
        qkv_o_params = (
            (hidden_size * hidden_size) + (2 * hidden_size * (num_kv_heads * head_dim)) + (hidden_size * hidden_size)
        )
        mlp_params = (2 * hidden_size * intermediate_size) + (intermediate_size * hidden_size)
        layer_params = qkv_o_params + mlp_params
        total_model_params = layer_params * num_layers

        # In BF16 (2 bytes/param)
        bf16_weight_gb = (total_model_params * 2) / 1e9

        # In W4A16 (0.5 bytes/param + 16-bit scales per 128 group)
        w4_weight_gb = (total_model_params * 0.5 + (total_model_params / 128) * 2) / 1e9

        # Static KV Cache: 64 layers * 2 (K, V) * 8 heads * 4096 seq * 128 dim * 2 bytes (bf16)
        kv_cache_gb = (num_layers * 2 * num_kv_heads * max_seq_len * head_dim * 2) / 1e9

        # Activations & CUDA Graph static replay buffers
        cuda_graph_overhead_gb = 1.2

        total_w4_footprint_gb = w4_weight_gb + kv_cache_gb + cuda_graph_overhead_gb

        return {
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "total_params_billion": total_model_params / 1e9,
            "bf16_weight_gb": bf16_weight_gb,
            "w4a16_weight_gb": w4_weight_gb,
            "static_kv_cache_gb": kv_cache_gb,
            "cuda_graph_buffers_gb": cuda_graph_overhead_gb,
            "total_w4_footprint_gb": total_w4_footprint_gb,
            "fits_in_24gb_vram": total_w4_footprint_gb <= 23.0,
            "headroom_vram_gb": max(0.0, 24.0 - total_w4_footprint_gb),
        }
