"""Empirical Benchmark for RDNA3 Fused W4A16 + Dynamic LoRA Kernels.

Evaluates performance across realistic LLM layer dimensions and batch/token regimes:
- M in [1, 2, 4, 16, 64, 256, 1024]
- Real Qwen3.5-4B projection shapes: (2560, 2560), (2560, 7680), (2560, 6912), (6912, 2560)
- Arms:
  * Arm A: PyTorch BF16 (torch.matmul)
  * Arm B: Triton BF16 WMMA (triton_wmma_matmul)
  * Arm C: Triton W4A16 GEMM (w4a16_matmul)
  * Arm D: Triton Fused W4A16 + LoRA (fused_w4a16_lora_matmul)
  * Arm E: PyTorch BF16 + separate LoRA
- Audit-compliant methodology: Warmup, 5 repeats per cell, alternating arm order, median + IQR.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from runtime.canon import CANON, REPO_ROOT
from runtime.triton_w4a16 import (
    fused_w4a16_lora_matmul,
    quantize_and_pack_w4,
    unpack_and_dequantize_w4,
    w4a16_matmul,
)
from runtime.triton_wmma import triton_wmma_matmul


def compute_median_iqr(values: list[float]) -> tuple[float, float, float]:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    median = sorted_vals[n // 2]
    q1 = sorted_vals[max(0, n // 4)]
    q3 = sorted_vals[min(n - 1, 3 * n // 4)]
    return median, q1, q3


def run_w4a16_benchmark(
    shapes: list[tuple[int, int, int, str]] | None = None,
    rank: int = 8,
    alpha: float = 16.0,
    group_size: int = 128,
    warmup: int = 10,
    iters: int = 30,
    n_repeats: int = 5,
    device: str = "cuda:0",
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm / CUDA GPU is required for W4A16 Triton benchmark")

    if shapes is None:
        # (M, K, N, description)
        shapes = [
            (1, 2560, 2560, "Single-token decode (M=1, K=2560, N=2560)"),
            (2, 2560, 2560, "MTP Speculative verify K=2 (M=2, K=2560, N=2560)"),
            (4, 2560, 2560, "MTP Speculative verify K=4 (M=4, K=2560, N=2560)"),
            (16, 2560, 2560, "Micro-batch B=16 decode (M=16, K=2560, N=2560)"),
            (64, 2560, 7680, "Prompt prefill QKV proj (M=64, K=2560, N=7680)"),
            (128, 2560, 6912, "Medium prefill Gate/Up proj (M=128, K=2560, N=6912)"),
            (256, 6912, 2560, "Prefill Down proj (M=256, K=6912, N=2560)"),
            (1024, 2560, 2560, "Serving batch / long prefill (M=1024, K=2560, N=2560)"),
        ]

    torch.manual_seed(42)
    device_name = torch.cuda.get_device_name(0)

    print("=" * 110)
    print(f" RDNA3 FUSED W4A16 + DYNAMIC LoRA BENCHMARK ({device_name})")
    print(f" Config: group_size={group_size}, rank={rank}, alpha={alpha}, n_repeats={n_repeats}, iters={iters}")
    print("=" * 110)

    results_table = []

    for M, K, N, desc in shapes:
        print(f"\n[Benchmarking] {desc}...")

        x = torch.randn((M, K), dtype=torch.bfloat16, device=device)
        w = torch.randn((K, N), dtype=torch.bfloat16, device=device)
        lora_a = torch.randn((K, rank), dtype=torch.bfloat16, device=device)
        lora_b = torch.randn((rank, N), dtype=torch.bfloat16, device=device)

        # Quantize base weight
        qweight, scales = quantize_and_pack_w4(w, group_size=group_size)

        # Numerical verification
        dequant_ref = unpack_and_dequantize_w4(qweight, scales, group_size=group_size)
        ref_base = torch.matmul(x, dequant_ref)
        ref_fused = ref_base + alpha * torch.matmul(torch.matmul(x, lora_a), lora_b)

        w4_out = w4a16_matmul(x, qweight, scales, group_size=group_size)
        fused_out = fused_w4a16_lora_matmul(
            x, qweight, scales, lora_a=lora_a, lora_b=lora_b, alpha=alpha, group_size=group_size
        )

        base_cos_sim = torch.cosine_similarity(ref_base.flatten().float(), w4_out.flatten().float(), dim=0).item()
        fused_cos_sim = torch.cosine_similarity(ref_fused.flatten().float(), fused_out.flatten().float(), dim=0).item()

        # Arm timing definitions
        def run_arm_a(): # PyTorch BF16
            for _ in range(warmup):
                _ = torch.matmul(x, w)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = torch.matmul(x, w)
            torch.cuda.synchronize()
            return ((time.perf_counter() - t0) / iters) * 1000.0

        def run_arm_b(): # Triton BF16 WMMA
            for _ in range(warmup):
                _ = triton_wmma_matmul(x, w)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = triton_wmma_matmul(x, w)
            torch.cuda.synchronize()
            return ((time.perf_counter() - t0) / iters) * 1000.0

        def run_arm_c(): # Triton W4A16 GEMM
            for _ in range(warmup):
                _ = w4a16_matmul(x, qweight, scales, group_size=group_size)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = w4a16_matmul(x, qweight, scales, group_size=group_size)
            torch.cuda.synchronize()
            return ((time.perf_counter() - t0) / iters) * 1000.0

        def run_arm_d(): # Triton Fused W4A16 + LoRA
            for _ in range(warmup):
                _ = fused_w4a16_lora_matmul(
                    x, qweight, scales, lora_a=lora_a, lora_b=lora_b, alpha=alpha, group_size=group_size
                )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = fused_w4a16_lora_matmul(
                    x, qweight, scales, lora_a=lora_a, lora_b=lora_b, alpha=alpha, group_size=group_size
                )
            torch.cuda.synchronize()
            return ((time.perf_counter() - t0) / iters) * 1000.0

        def run_arm_e(): # PyTorch BF16 + Separate LoRA
            for _ in range(warmup):
                _ = torch.matmul(x, w) + alpha * torch.matmul(torch.matmul(x, lora_a), lora_b)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = torch.matmul(x, w) + alpha * torch.matmul(torch.matmul(x, lora_a), lora_b)
            torch.cuda.synchronize()
            return ((time.perf_counter() - t0) / iters) * 1000.0

        arm_fns = [("a", run_arm_a), ("b", run_arm_b), ("c", run_arm_c), ("d", run_arm_d), ("e", run_arm_e)]
        collected_times: dict[str, list[float]] = {k: [] for k, _ in arm_fns}

        # Alternating arm order across repeats
        for rep in range(n_repeats):
            # Rotate starting arm
            shift = rep % len(arm_fns)
            current_order = arm_fns[shift:] + arm_fns[:shift]
            if rep % 2 == 1:
                current_order = list(reversed(current_order))

            for arm_key, fn in current_order:
                collected_times[arm_key].append(fn())

        # Medians and IQRs
        a_med, a_q1, a_q3 = compute_median_iqr(collected_times["a"])
        b_med, b_q1, b_q3 = compute_median_iqr(collected_times["b"])
        c_med, c_q1, c_q3 = compute_median_iqr(collected_times["c"])
        d_med, d_q1, d_q3 = compute_median_iqr(collected_times["d"])
        e_med, e_q1, e_q3 = compute_median_iqr(collected_times["e"])

        # Speedups
        w4_vs_torch_speedup = a_med / max(1e-5, c_med)
        fused_vs_torch_lora_speedup = e_med / max(1e-5, d_med)

        # Memory footprint calculations
        bf16_mb = (K * N * 2) / 1e6
        w4_mb = (qweight.numel() * 4 + scales.numel() * 2) / 1e6
        vram_reduction = bf16_mb / w4_mb

        # TFLOPS (2 * M * K * N)
        total_ops = 2.0 * M * K * N
        w4_tflops = (total_ops / (c_med * 1e-3)) / 1e12

        row = {
            "M": M,
            "K": K,
            "N": N,
            "desc": desc,
            "bf16_mb": round(bf16_mb, 2),
            "w4_mb": round(w4_mb, 2),
            "vram_reduction": round(vram_reduction, 2),
            "torch_bf16_ms": round(a_med, 4),
            "triton_bf16_ms": round(b_med, 4),
            "triton_w4a16_ms": round(c_med, 4),
            "fused_w4a16_lora_ms": round(d_med, 4),
            "torch_bf16_lora_ms": round(e_med, 4),
            "w4_vs_torch_speedup": round(w4_vs_torch_speedup, 2),
            "fused_vs_torch_lora_speedup": round(fused_vs_torch_lora_speedup, 2),
            "w4_tflops": round(w4_tflops, 2),
            "base_cos_sim": round(base_cos_sim, 6),
            "fused_cos_sim": round(fused_cos_sim, 6),
            "iqr_torch_bf16": [round(a_q1, 4), round(a_q3, 4)],
            "iqr_triton_w4a16": [round(c_q1, 4), round(c_q3, 4)],
            "iqr_fused_lora": [round(d_q1, 4), round(d_q3, 4)],
        }
        results_table.append(row)

        print(
            f"  M={M:4d} | PyTorch BF16: {a_med:6.4f} ms | W4A16: {c_med:6.4f} ms ({w4_vs_torch_speedup:4.2f}x) | "
            f"Fused W4+LoRA: {d_med:6.4f} ms vs PyTorch+LoRA: {e_med:6.4f} ms ({fused_vs_torch_lora_speedup:4.2f}x) | "
            f"VRAM: {bf16_mb:.1f}MB -> {w4_mb:.1f}MB ({vram_reduction:.2f}x)"
        )

    out_data = {
        "device": device_name,
        "target_arch": "gfx1100",
        "canon": CANON.stamp(),
        "config": {
            "group_size": group_size,
            "rank": rank,
            "alpha": alpha,
            "n_repeats": n_repeats,
            "iters": iters,
        },
        "results": results_table,
    }

    out_file = REPO_ROOT / "results/benchmarks/w4a16_fused_perf.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)

    print("\n" + "=" * 110)
    print(f"[Persisted] Benchmark telemetry saved to {out_file}")
    print("=" * 110)
    return out_data


def main():
    parser = argparse.ArgumentParser(description="RDNA3 Fused W4A16 + Dynamic LoRA Benchmark")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", type=float, default=16.0, help="LoRA scaling alpha")
    parser.add_argument("--group_size", type=int, default=128, help="Quantization group size")
    parser.add_argument("--n_repeats", type=int, default=5, help="Alternating repeat passes")
    parser.add_argument("--iters", type=int, default=30, help="Timing iterations per pass")
    args = parser.parse_args()

    run_w4a16_benchmark(
        rank=args.rank,
        alpha=args.alpha,
        group_size=args.group_size,
        n_repeats=args.n_repeats,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
