"""Benchmark Factor-Based VRAM Residency vs Dense Model Duplication.

Audits memory efficiency, compression ratios, and multi-expert standby capacity
for low-rank factor residency (FoldableExpert) on Qwen 3.5 4B.

Usage:
    uv run --env-file .env python scripts/runtime/memory/factor_residency/benchmark_factor_residency.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.novel_peft import FoldableExpert  # noqa: E402

ADAPTER_PATHS = {
    "financial_planning": REPO_ROOT / "results/adapters/m2_financial_r8a128",
    "postgresql": REPO_ROOT / "results/adapters/m2_postgresql_r8a128",
    "astral": REPO_ROOT / "results/adapters/m2_astral_r8a128",
}

# Qwen 3.5 4B Architecture Constants
BASE_PARAM_COUNT = 4_260_000_000  # ~4.26B parameters
DENSE_BYTES_BF16 = BASE_PARAM_COUNT * 2  # 8.52 GB in bfloat16
GPU_TOTAL_VRAM_GB = 24.0  # AMD RX 7900 XTX 24GB VRAM


def main():
    parser = argparse.ArgumentParser(description="Factor-Based VRAM Residency Benchmark")
    parser.add_argument("--out", default="results/factor_residency_benchmark.json")
    args = parser.parse_args()

    print("=" * 80)
    print(" ⭐ Factor-Based VRAM Residency vs Dense Duplication Benchmark")
    print("=" * 80)

    # 1. Load low-rank factor experts and measure exact memory and disk load time
    experts: dict[str, FoldableExpert] = {}
    load_times_ms: dict[str, float] = {}

    for name, path in ADAPTER_PATHS.items():
        if not path.exists():
            print(f"  [skip] {name} adapter not found at {path}")
            continue

        t0 = time.perf_counter()
        exp = FoldableExpert.from_dir(path, name=name)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        experts[name] = exp
        load_times_ms[name] = elapsed_ms

    if not experts:
        raise RuntimeError("No adapters found to benchmark!")

    # Calculate exact factor sizes
    first_exp = next(iter(experts.values()))
    factor_bytes = sum(
        u.numel() * u.element_size() + v.numel() * v.element_size() for u, v in first_exp.factors.values()
    )
    factor_mb = factor_bytes / (1024 * 1024)
    dense_gb = DENSE_BYTES_BF16 / (1024 * 1024 * 1024)
    compression_ratio = DENSE_BYTES_BF16 / factor_bytes

    print("\n--- 1. Memory Compression: Dense Model vs Low-Rank Factor Expert ---")
    print(f"  Dense Model Size (Qwen 3.5 4B in bf16) : {dense_gb:6.2f} GB ({DENSE_BYTES_BF16 / 1e9:.2f} GB)")
    print(f"  Single Factor Expert Size (Rank 8)     : {factor_mb:6.2f} MB ({factor_bytes / 1e6:.2f} MB)")
    print(f"  Memory Compression Ratio per Expert    : {compression_ratio:6.1f}x reduction")
    print(f"  Mean Factor Load Latency from Disk     : {sum(load_times_ms.values()) / len(load_times_ms):6.2f} ms")

    # 2. Multi-Expert Fleet Capacity Simulation
    print("\n--- 2. Multi-Expert Fleet VRAM Footprint Scaling ---")
    fleet_sizes = [1, 2, 3, 5, 10, 20, 50, 100]
    scaling_table = []

    print(f"  {'Fleet Size':<12s} {'Dense VRAM':<16s} {'Factor VRAM':<16s} {'VRAM Saved':<14s} {'Status (24GB GPU)'}")
    print("  " + "─" * 76)

    for n in fleet_sizes:
        dense_req_gb = n * dense_gb
        # Factor VRAM = 1 Base Model (8.52 GB) + Pristine Buffer (5.12 GB active slots) + N * factor_mb
        factor_req_gb = (DENSE_BYTES_BF16 + 5.12 * 1e9 + n * factor_bytes) / (1024 * 1024 * 1024)
        saved_gb = dense_req_gb - factor_req_gb
        status = "FITS ✅" if factor_req_gb <= GPU_TOTAL_VRAM_GB * 0.9 else "OOM ❌"
        dense_status = "FITS" if dense_req_gb <= GPU_TOTAL_VRAM_GB * 0.9 else "OOM ❌"

        scaling_table.append({
            "fleet_size": n,
            "dense_vram_gb": round(dense_req_gb, 2),
            "factor_vram_gb": round(factor_req_gb, 2),
            "vram_saved_gb": round(saved_gb, 2),
            "factor_fits": status == "FITS ✅",
            "dense_fits": dense_status == "FITS",
        })

        print(
            f"  {n:<12d} {dense_req_gb:6.2f} GB ({dense_status:<5s}) {factor_req_gb:6.2f} GB        "
            f"{saved_gb:6.2f} GB      {status}"
        )

    # 3. Maximum Theoretical Concurrent Capacity on 24GB GPU
    available_for_factors_gb = (GPU_TOTAL_VRAM_GB * 0.9) - (dense_gb + 5.12)  # Base + pristine buffer
    max_resident_experts = int((available_for_factors_gb * 1024) / factor_mb)

    print("\n--- 3. Fleet Capacity Limit on AMD RX 7900 XTX (24GB VRAM) ---")
    print("  Max Dense Models Deployable Simultaneously : 2 models (17.04 GB, 3rd model OOMs)")
    print(f"  Max Low-Rank Factor Experts in Standby    : {max_resident_experts} specialized domain experts!")

    # 4. Save results to JSON
    report = {
        "model_name": "Qwen/Qwen3.5-4B",
        "dense_model_bytes": DENSE_BYTES_BF16,
        "dense_model_gb": dense_gb,
        "factor_expert_bytes": factor_bytes,
        "factor_expert_mb": factor_mb,
        "compression_ratio": compression_ratio,
        "mean_load_time_ms": sum(load_times_ms.values()) / len(load_times_ms),
        "scaling_table": scaling_table,
        "max_concurrent_experts_24gb": max_resident_experts,
    }

    out_file = REPO_ROOT / args.out
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2))
    print(f"\nSaved factor residency benchmark to {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
