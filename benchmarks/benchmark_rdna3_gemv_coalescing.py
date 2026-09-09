"""Synthetic Benchmark: RDNA3 W4A16 GEMV Memory Coalescing & Tile Tuning.

Sweeps block tile sizes (BLOCK_N, BLOCK_K), warp allocations (num_warps=2, 4, 8),
and pipeline stages (num_stages=1, 2, 3) on AMD Radeon RX 7900 XTX to maximize
effective memory bandwidth (GB/s) during single-token decode (M=1).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import triton
import triton.language as tl

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime.triton_w4a16 import _w4a16_gemv_m1_kernel, quantize_and_pack_w4


def benchmark_gemv_config(
    x: torch.Tensor,
    qw: torch.Tensor,
    scales: torch.Tensor,
    out: torch.Tensor,
    N: int,
    K: int,
    block_n: int,
    block_k: int,
    num_warps: int,
    num_stages: int,
    group_size: int = 128,
    num_trials: int = 100,
) -> float:
    """Runs GEMV with specific compile parameters and returns median latency in ms."""
    grid = (triton.cdiv(N, block_n),)
    device = x.device

    # Warmup
    for _ in range(5):
        _w4a16_gemv_m1_kernel[grid](
            x, qw, scales, out,
            N, K,
            x.stride(1),
            qw.stride(0), qw.stride(1),
            scales.stride(0),
            out.stride(1),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_SIZE=group_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_trials):
        _w4a16_gemv_m1_kernel[grid](
            x, qw, scales, out,
            N, K,
            x.stride(1),
            qw.stride(0), qw.stride(1),
            scales.stride(0),
            out.stride(1),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_SIZE=group_size,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    end_event.record()
    torch.cuda.synchronize(device)

    return start_event.elapsed_time(end_event) / num_trials


def sweep_configurations(device: torch.device) -> Dict[str, Any]:
    # Test FFN Down matrix (K=13824, N=5120) and Gate matrix (K=5120, N=13824)
    shapes = [
        ("FFN_GateUp", 5120, 13824),
        ("FFN_Down", 13824, 5120),
        ("SSM_InProj", 5120, 10240),
    ]

    candidate_configs = [
        # (block_n, block_k, num_warps, num_stages)
        (128, 64, 4, 2),   # Current baseline
        (128, 64, 8, 2),
        (256, 64, 8, 2),
        (64, 128, 4, 2),
        (128, 128, 4, 2),
        (128, 128, 8, 2),
        (64, 64, 4, 2),
        (128, 64, 4, 1),
    ]

    best_overall = {}

    for shape_name, K, N in shapes:
        print(f"\n--- Sweeping {shape_name}: K={K}, N={N} ---")
        w = torch.randn(K, N, dtype=torch.bfloat16, device=device)
        qw, scales = quantize_and_pack_w4(w)
        x = torch.randn(1, K, dtype=torch.bfloat16, device=device)
        out = torch.empty((1, N), dtype=torch.bfloat16, device=device)

        weight_bytes = (qw.numel() * 4) + (scales.numel() * 2)

        best_ms = float("inf")
        best_cfg = None

        for bn, bk, nw, ns in candidate_configs:
            try:
                ms = benchmark_gemv_config(x, qw, scales, out, N, K, bn, bk, nw, ns)
                bandwidth_gbs = (weight_bytes / (ms / 1000.0)) / (1024**3)
                print(f"  BN={bn:3d}, BK={bk:3d}, Warps={nw}, Stages={ns} | Latency: {ms:6.3f} ms | Bandwidth: {bandwidth_gbs:6.1f} GB/s")
                if ms < best_ms:
                    best_ms = ms
                    best_cfg = (bn, bk, nw, ns, bandwidth_gbs)
            except Exception as e:
                print(f"  BN={bn:3d}, BK={bk:3d}, Warps={nw}, Stages={ns} | Failed: {e}")

        best_overall[shape_name] = {
            "K": K,
            "N": N,
            "best_config": {
                "BLOCK_N": best_cfg[0],
                "BLOCK_K": best_cfg[1],
                "num_warps": best_cfg[2],
                "num_stages": best_cfg[3],
                "latency_ms": round(best_ms, 3),
                "bandwidth_gbs": round(best_cfg[4], 1),
            },
        }
        print(f"  >>> Winner for {shape_name}: {best_cfg[0]}x{best_cfg[1]} (w={best_cfg[2]}, s={best_cfg[3]}) -> {best_ms:.3f} ms ({best_cfg[4]:.1f} GB/s)")

    return best_overall


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA/ROCm not available!")
        return

    device = torch.device("cuda:0")
    print("=" * 80)
    print("      SYNTHETIC BENCHMARK: RDNA3 W4A16 GEMV MEMORY BANDWIDTH TUNING     ")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(device)}")

    results = sweep_configurations(device)

    out_dir = Path("results/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "rdna3_gemv_tuning_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark tuning results to: {out_file}")


if __name__ == "__main__":
    main()
