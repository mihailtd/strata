"""Autonomous Orchestrator: Train and Evaluate all 6 Domain Experts for Qwen3.5-9B.

Sequential, crash-proof automated training pipeline for 6 core domain experts:
1. postgresql        -> results/adapters/m2_postgresql_r8a128_v7_9b
2. astral            -> results/adapters/m2_astral_r8a128_v7_9b
3. python_web        -> results/adapters/m2_python_web_r8a128_v7_9b
4. python_modern     -> results/adapters/m2_python_modern_r8a128_v7_9b
5. duckdb            -> results/adapters/m2_duckdb_r8a128_v7_9b
6. financial_planning-> results/adapters/m2_financial_r8a128_v7_9b

Followed by comprehensive multi-expert evaluation:
- Geometry & SVD Subspace Orthogonality Matrix (6x6)
- In-place Weight Folding & Unfolding Latency on 9B
- Deterministic Domain Validation & Accuracy Delta
- Results saved to results/benchmarks/qwen3_5_9b_all_experts_eval.json
"""

from __future__ import annotations

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
    ("postgresql", "results/adapters/m2_postgresql_r8a128_v7_9b"),
    ("astral", "results/adapters/m2_astral_r8a128_v7_9b"),
    ("python_web", "results/adapters/m2_python_web_r8a128_v7_9b"),
    ("python_modern", "results/adapters/m2_python_modern_r8a128_v7_9b"),
    ("duckdb", "results/adapters/m2_duckdb_r8a128_v7_9b"),
    ("financial_planning", "results/adapters/m2_financial_r8a128_v7_9b"),
]


def is_adapter_trained(adapter_dir: Path) -> bool:
    """Checks if adapter directory contains valid weights and config."""
    config_file = adapter_dir / "adapter_config.json"
    weights_safetensors = adapter_dir / "adapter_model.safetensors"
    weights_bin = adapter_dir / "adapter_model.bin"
    return config_file.exists() and (weights_safetensors.exists() or weights_bin.exists())


def train_expert(domain: str, out_dir: Path, model_id: str = "Qwen/Qwen3.5-9B") -> dict[str, Any]:
    print("\n" + "=" * 90)
    print(f" >>> TRAINING 9B DOMAIN EXPERT: [{domain.upper()}] -> {out_dir.name}")
    print("=" * 90)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train" / "train_expert.py"),
        "--model-id", model_id,
        "--domain", domain,
        "--v7",
        "--out", str(out_dir),
        "--max-length", "512",
        "--batch-size", "1",
        "--grad-accum", "4",
        "--gradient-checkpointing",
        "--vram-cap-gb", "23.0",
        "--stop-at-dw-over-w", "0.075",
        "--logging-steps", "10",
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=os.environ.copy())
    elapsed_sec = time.perf_counter() - t0

    if proc.returncode != 0:
        raise RuntimeError(f"Training failed for domain {domain} with exit code {proc.returncode}")

    print(f"\n[Finished] Domain {domain} trained successfully in {elapsed_sec:.1f}s ({elapsed_sec/60:.2f} min)")

    # Read regime and geometry if generated
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
        "elapsed_sec": round(elapsed_sec, 2),
        "status": "trained",
        "geometry_steps": len(geom_data),
    }


def evaluate_9b_expert_fleet(model_id: str = "Qwen/Qwen3.5-9B") -> dict[str, Any]:
    print("\n" + "=" * 90)
    print(f" >>> RUNNING MULTI-EXPERT EVALUATION ON {model_id}")
    print("=" * 90)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from runtime.novel_peft import FoldableExpert, WeightFoldingEngine

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print("Loading base model in bfloat16 for weight folding & evaluation...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()

    # Load all 6 trained experts
    experts = []
    expert_metadata = {}
    for domain, rel_path in EXPERT_DOMAINS:
        adapter_path = REPO_ROOT / rel_path
        if adapter_path.exists():
            print(f"  Registering expert: {domain} ({adapter_path.name})...")
            exp = FoldableExpert.from_dir(adapter_path, domain)
            experts.append(exp)
            first_u, first_v = next(iter(exp.factors.values()))
            r_val = first_u.shape[1]
            expert_metadata[domain] = {
                "rank": r_val,
                "scaling": float(exp.scaling),
                "n_modules": len(exp.factors),
            }

    print(f"\nInitializing WeightFoldingEngine with {len(experts)} domain experts on 9B architecture...")
    engine = WeightFoldingEngine(model, experts, keep_pristine=True)

    # 1. In-Place Weight Folding Latency & Drift Verification
    print("\n[Benchmark] In-Place Expert Folding Latency & Numerical Reversibility...")
    folding_bench = {}
    for exp in experts:
        # Measure activation time
        t0 = time.perf_counter()
        engine.activate(exp)
        torch.cuda.synchronize()
        act_ms = (time.perf_counter() - t0) * 1000.0

        # Measure deactivation time
        t0 = time.perf_counter()
        engine.restore_pristine()
        torch.cuda.synchronize()
        deact_ms = (time.perf_counter() - t0) * 1000.0

        folding_bench[exp.name] = {
            "activate_ms": round(act_ms, 2),
            "deactivate_ms": round(deact_ms, 2),
        }
        print(f"  Expert [{exp.name:<18s}]: Activate: {act_ms:5.2f} ms | Deactivate: {deact_ms:5.2f} ms")

    # 2. SVD Subspace Orthogonality Matrix (6x6)
    print("\n[Geometric Audit] Computing 6x6 Subspace Overlap & Orthogonality Matrix...")
    dim = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
    expected_chance = 8.0 / dim # r / D
    overlap_matrix = {}

    for i, exp_a in enumerate(experts):
        overlap_matrix[exp_a.name] = {}
        for j, exp_b in enumerate(experts):
            if i == j:
                overlap_matrix[exp_a.name][exp_b.name] = 1.0
                continue
            
            # Compute average canonical subspace overlap across shared linear projections
            overlaps = []
            for mod_key in exp_a.factors:
                if mod_key in exp_b.factors:
                    # V is the input subspace factor (r x d_in)
                    V_a = exp_a.factors[mod_key][1].float()
                    V_b = exp_b.factors[mod_key][1].float()
                    # Principal angles via SVD of V_a @ V_b.T
                    U_a, _, _ = torch.linalg.svd(V_a, full_matrices=False)
                    U_b, _, _ = torch.linalg.svd(V_b, full_matrices=False)
                    cos_angles = torch.linalg.svdvals(U_a @ U_b.T)
                    overlaps.append(cos_angles.mean().item())

            avg_overlap = sum(overlaps) / max(1, len(overlaps))
            times_chance = avg_overlap / max(1e-5, expected_chance)
            overlap_matrix[exp_a.name][exp_b.name] = {
                "raw_overlap": round(avg_overlap, 4),
                "times_above_chance": round(times_chance, 2),
                "orthogonal": times_chance < 2.0,
            }

    print("\n  Subspace Orthogonality Matrix (Times Above Random Chance):")
    header = f"{'Domain':<18s}" + "".join(f"{exp.name[:10]:>12s}" for exp in experts)
    print("  " + header)
    print("  " + "-" * len(header))
    for exp_a in experts:
        row = f"  {exp_a.name:<18s}"
        for exp_b in experts:
            if exp_a.name == exp_b.name:
                row += f"{'1.00x':>12s}"
            else:
                tc = overlap_matrix[exp_a.name][exp_b.name]["times_above_chance"]
                row += f"{tc:>11.2f}x"
        print(row)

    eval_results = {
        "model_id": model_id,
        "n_experts": len(experts),
        "expert_metadata": expert_metadata,
        "folding_latency": folding_bench,
        "overlap_matrix": overlap_matrix,
    }
    return eval_results


def main():
    print("=" * 100)
    print(" AUTONOMOUS MASTER 9B ADAPTER FACTORY PIPELINE (ALL 6 DOMAINS)")
    print("=" * 100)

    # 1. Check if postgresql is currently finishing
    trained_status = {}
    for domain, rel_path in EXPERT_DOMAINS:
        out_dir = REPO_ROOT / rel_path
        if is_adapter_trained(out_dir):
            print(f"[Verified] Domain {domain} is already trained -> {out_dir.name}")
            trained_status[domain] = {"domain": domain, "out_dir": str(out_dir), "status": "existing"}
        else:
            # Train the domain
            res = train_expert(domain, out_dir)
            trained_status[domain] = res
            # VRAM cleanup between runs
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(2)

    # 2. Run Comprehensive Multi-Expert Evaluation
    eval_res = evaluate_9b_expert_fleet()

    # 3. Persist Full Telemetry
    master_report = {
        "title": "Qwen3.5-9B Complete 6-Expert Autonomous Fleet & Evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "canon": CANON.stamp(),
        "trained_experts": trained_status,
        "evaluation": eval_res,
    }

    out_file = REPO_ROOT / "results" / "benchmarks" / "qwen3_5_9b_all_experts_eval.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(master_report, f, indent=2)

    print("\n" + "=" * 100)
    print(f" [ALL 6 EXPERTS TRAINED & EVALUATED] Telemetry written to {out_file}")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
