"""Synthetic Benchmark: Batched Prompt Prefill (M = S) vs Sequential Eager Token Loop.

Evaluates the wall-clock latency and memory bus traffic reduction of executing
a multi-token prompt (S=16, 32, 64, 128) in a single batched GEMM pass vs 
sequential token-by-token GEMV passes on AMD Radeon RX 7900 XTX.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime.triton_w4a16 import quantize_and_pack_w4, w4a16_matmul
from src.runtime.native_27b_engine import RMSNorm, PreallocatedKVCache, apply_rotary_emb


def benchmark_linear_batched_vs_sequential(
    in_features: int = 5120,
    out_features: int = 13824,
    seq_lens: List[int] = [16, 32, 64, 128],
    device: torch.device = torch.device("cuda:0"),
    num_trials: int = 20,
) -> List[Dict[str, Any]]:
    """Compares single-pass GEMM (M=S) against sequential GEMV (M=1 executed S times)."""
    print(f"\n--- Benchmarking W4A16 Linear ({in_features} -> {out_features}) ---")
    
    w = torch.randn(in_features, out_features, dtype=torch.bfloat16, device=device)
    qw, scales = quantize_and_pack_w4(w)
    weight_bytes = (qw.numel() * 4) + (scales.numel() * 2)

    results = []

    for s in seq_lens:
        x_batched = torch.randn(1, s, in_features, dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(3):
            _ = w4a16_matmul(x_batched, qw, scales)
            for i in range(min(s, 4)):
                _ = w4a16_matmul(x_batched[:, i : i + 1, :], qw, scales)
        torch.cuda.synchronize(device)

        # 1. Batched Single-Pass GEMM
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for _ in range(num_trials):
            _ = w4a16_matmul(x_batched, qw, scales)
        end_event.record()
        torch.cuda.synchronize(device)
        batched_ms = start_event.elapsed_time(end_event) / num_trials

        # 2. Sequential Token-by-Token Loop
        start_event.record()
        for _ in range(num_trials):
            for i in range(s):
                _ = w4a16_matmul(x_batched[:, i : i + 1, :], qw, scales)
        end_event.record()
        torch.cuda.synchronize(device)
        sequential_ms = start_event.elapsed_time(end_event) / num_trials

        speedup = sequential_ms / max(1e-5, batched_ms)
        bytes_transferred_seq_mb = (weight_bytes * s) / (1024 * 1024)
        bytes_transferred_batch_mb = weight_bytes / (1024 * 1024)

        print(
            f"  SeqLen {s:3d}: Batched = {batched_ms:6.2f} ms | Sequential = {sequential_ms:6.2f} ms | "
            f"Speedup = {speedup:5.1f}x | VRAM Read: {bytes_transferred_batch_mb:.1f} MB vs {bytes_transferred_seq_mb:.1f} MB"
        )

        results.append({
            "seq_len": s,
            "batched_ms": round(batched_ms, 3),
            "sequential_ms": round(sequential_ms, 3),
            "speedup": round(speedup, 2),
            "vram_read_reduction_ratio": s,
        })

    return results


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA/ROCm not available!")
        return

    device = torch.device("cuda:0")
    print("=" * 80)
    print("      SYNTHETIC BENCHMARK: BATCHED PREFILL VS SEQUENTIAL EAGER LOOP     ")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(device)}")

    results = benchmark_linear_batched_vs_sequential(device=device)

    out_dir = Path("results/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "batched_prefill_synthetic_benchmark.json"
    with open(out_file, "w") as f:
        json.dump({"benchmark": "batched_prefill_synthetic", "results": results}, f, indent=2)
    print(f"\nSaved synthetic benchmark results to: {out_file}")


if __name__ == "__main__":
    main()
