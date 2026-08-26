#!/usr/bin/env python3
"""Autonomous Orchestrator: Train and Evaluate all 6 Domain Experts for 27B/32B Models in W4A16/QLoRA.

Sequential, crash-proof automated training pipeline for 6 core domain experts:
1. postgresql        -> results/adapters/m2_postgresql_r8a128_v7_27b
2. astral            -> results/adapters/m2_astral_r8a128_v7_27b
3. python_web        -> results/adapters/m2_python_web_r8a128_v7_27b
4. python_modern     -> results/adapters/m2_python_modern_r8a128_v7_27b
5. duckdb            -> results/adapters/m2_duckdb_r8a128_v7_27b
6. financial_planning-> results/adapters/m2_financial_r8a128_v7_27b
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Enforce GPU 0 exclusive device isolation
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["ROCR_VISIBLE_DEVICES"] = "0"

import torch
from runtime.canon import CANON, REPO_ROOT

EXPERT_DOMAINS = [
    ("postgresql", "results/adapters/m2_postgresql_r8a128_v7_27b"),
    ("astral", "results/adapters/m2_astral_r8a128_v7_27b"),
    ("python_web", "results/adapters/m2_python_web_r8a128_v7_27b"),
    ("python_modern", "results/adapters/m2_python_modern_r8a128_v7_27b"),
    ("duckdb", "results/adapters/m2_duckdb_r8a128_v7_27b"),
    ("financial_planning", "results/adapters/m2_financial_r8a128_v7_27b"),
]


def is_adapter_trained(adapter_dir: Path) -> bool:
    """Checks if adapter directory contains valid weights and config."""
    config_file = adapter_dir / "adapter_config.json"
    weights_safetensors = adapter_dir / "adapter_model.safetensors"
    weights_bin = adapter_dir / "adapter_model.bin"
    return config_file.exists() and (weights_safetensors.exists() or weights_bin.exists())


def train_expert(
    domain: str,
    out_dir: Path,
    model_id: str = "Qwen/Qwen2.5-32B-Instruct",
    vram_cap: float = 22.0,
) -> dict[str, Any]:
    print("\n" + "=" * 90)
    print(f" >>> TRAINING 27B/32B DOMAIN EXPERT: [{domain.upper()}] -> {out_dir.name}")
    print(f"     Base Model: {model_id} (QLoRA 4-bit, r=8, alpha=128)")
    print("=" * 90)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train" / "train_expert.py"),
        "--model-id", model_id,
        "--domain", domain,
        "--v7",
        "--out", str(out_dir),
        "--qlora",
        "--max-length", "512",
        "--batch-size", "1",
        "--grad-accum", "4",
        "--gradient-checkpointing",
        "--vram-cap-gb", str(vram_cap),
        "--stop-at-dw-over-w", "0.075",
        "--logging-steps", "10",
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=os.environ.copy())
    elapsed_sec = time.perf_counter() - t0

    if proc.returncode != 0:
        raise RuntimeError(f"Training failed for domain {domain} with exit code {proc.returncode}")

    print(f"\n[Finished] Domain {domain} trained successfully in {elapsed_sec:.1f}s ({elapsed_sec/60:.2f} min)")

    geom_file = out_dir / "geometry_trace.json"
    geom_data = []
    if geom_file.exists():
        try:
            with open(geom_file) as f:
                geom_data = json.load(f)
        except Exception:
            pass

    return {
        "domain": domain,
        "out_dir": str(out_dir),
        "training_time_sec": elapsed_sec,
        "geometry_trace": geom_data,
    }


def main():
    parser = argparse.ArgumentParser(description="Train all 6 domain LoRA experts for 27B/32B models.")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-32B-Instruct", help="Base model identifier")
    parser.add_argument("--domains", nargs="+", default=None, help="Specific domains to train (default: all 6)")
    parser.add_argument("--force", action="store_true", help="Force re-training even if adapter exists")
    parser.add_argument("--vram-cap", type=float, default=22.0, help="VRAM cap in GB")
    args = parser.parse_args()

    domains_to_train = EXPERT_DOMAINS
    if args.domains:
        domains_to_train = [d for d in EXPERT_DOMAINS if d[0] in args.domains]

    print("=" * 90)
    print("🚀 LAUNCHING 27B/32B MULTI-EXPERT FLEET TRAINING PIPELINE")
    print(f"   Target Base Model: {args.model_id}")
    print(f"   Domains Scheduled ({len(domains_to_train)}): {[d[0] for d in domains_to_train]}")
    print(f"   Output Directory:  results/adapters/m2_*_r8a128_v7_27b")
    print("=" * 90)

    results = []
    t_start = time.perf_counter()

    for domain, rel_path in domains_to_train:
        out_dir = REPO_ROOT / rel_path
        if is_adapter_trained(out_dir) and not args.force:
            print(f"\n[Skip] Adapter already exists for {domain}: {out_dir.name}")
            continue

        res = train_expert(domain, out_dir, model_id=args.model_id, vram_cap=args.vram_cap)
        results.append(res)

        # Force garbage collection between domain runs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 90)
    print(f"✅ ALL DOMAIN EXPERTS COMPLETED in {total_time/60:.2f} minutes!")
    print("=" * 90)


if __name__ == "__main__":
    main()
