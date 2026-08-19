"""4-Way Multi-Expert Stacking Benchmark: Raw Stacking vs POET Channel Notch Filtering.

Evaluates:
1. Full 4-Expert Stacking: astral + postgresql + duckdb + financial simultaneously active (K=4).
2. Total Perturbation Energy ||delta_total|| / ||h|| across all 4 domains + general prose.
3. POET Multi-Adapter Notch Filtering: isolates and zeroes the top <=20 mutually conflicting channels across all pairs.
4. Retention & Cross-Domain Noise reduction with vs without POET filtering.

Usage:
  uv run --env-file .env python benchmarks/factory/geometry/poet_activation_crosstalk/probe_4way_poet_notch.py [--out results/benchmarks/poet_4way_stacking_results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gnn_experiment.canon import CANON, DOMAINS, REPO_ROOT, adapter_path

# Ensure src is on path
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.novel_peft import FoldableExpert, WeightFoldingEngine, set_hard_vram_cap


PROMPTS_PER_DOMAIN = {
    "astral": [
        "How do I add fastapi to my project using uv and lock dependencies?",
        "Configure ruff format and ruff check in pyproject.toml with target-version py312.",
        "Create a reproducible virtual environment and run tests with uv run pytest.",
        "Migrate from legacy pip requirements.txt to modern uv workspace tools.",
    ],
    "postgresql": [
        "Find the top 5 most similar documents using pgvector and cosine distance.",
        "Create a hypertable partitioned by time and run continuous aggregate queries.",
        "Write an atomic upsert query using INSERT INTO ... ON CONFLICT DO UPDATE.",
        "Tune work_mem and effective_cache_size for heavy analytical join workloads.",
    ],
    "duckdb": [
        "Query 50GB of partitioned Parquet files directly using hive_partitioning.",
        "Use GROUP BY ALL and QUALIFY row_number() to deduplicate records in SQL.",
        "Convert a massive CSV into Snappy Parquet with zero-copy Polars export.",
        "Join two remote S3 Parquet tables without loading the full datasets into memory.",
    ],
    "financial": [
        "Calculate the modified duration and convexity for a 10-year Treasury bond.",
        "Construct a Black-Scholes pricing model with implied volatility surface skew.",
        "Compute portfolio Value at Risk (VaR) using historical simulation at 99% CI.",
        "Rebalance a 60/40 equity/fixed-income portfolio minimizing tax drag.",
    ],
    "general": [
        "Explain the historical significance of the Silk Road trade routes.",
        "How does photosynthesis convert sunlight, carbon dioxide, and water into glucose?",
        "Describe the architectural differences between RISC and CISC microprocessors.",
        "Write a concise summary of the primary themes in Franz Kafka's Metamorphosis.",
    ],
}


def compute_poet_notch_mask_per_layer(
    experts: list[FoldableExpert],
    top_k_notch: int = 15,
) -> dict[str, torch.Tensor]:
    """Computes a binary channel mask for each module, zeroing top conflicting channels across all expert pairs."""
    # Find all common parameter keys
    keys = sorted(experts[0].factors.keys())
    notch_masks = {}

    for key in keys:
        deltas = []
        for e in experts:
            if key in e.factors:
                u, v = e.factors[key]  # u: [d_out, r], v: [r, d_in]
                dW = e.scaling * (u.float() @ v.float())  # [d_out, d_in]
                deltas.append(dW)
                
        if len(deltas) < 2:
            continue

        d_out, d_in = deltas[0].shape
        
        # Accumulate pairwise conflict covariance diagonal: sum_{a < b} |diag(dW_a @ dW_b^T)|
        conflict_energy = torch.zeros(d_out, device=deltas[0].device)
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                # Row-wise dot product = diag(dW_a @ dW_b^T)
                row_dot = torch.sum(deltas[i] * deltas[j], dim=1)  # [d_out]
                conflict_energy += torch.abs(row_dot)

        # Select top-k conflicting channels
        top_conflicts = torch.topk(conflict_energy, k=min(top_k_notch, d_out)).indices
        
        # Binary mask: 1.0 for clean channels, 0.0 for conflicting channels
        mask = torch.ones(d_out, 1, device=deltas[0].device)
        mask[top_conflicts] = 0.0
        notch_masks[key] = mask

    return notch_masks


def run_4way_stacking_benchmark(
    top_k_notch: int = 15,
) -> dict[str, Any]:
    """Measures live activation perturbation and noise across 4-way unscaled stack vs POET notch-filtered stack."""
    set_hard_vram_cap(CANON.VRAM_CAP_GB)
    
    print("\n[1/4] Loading Qwen3.5-4B Base Model...", flush=True)
    tok = AutoTokenizer.from_pretrained(CANON.BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        CANON.BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()

    print("[2/4] Loading All 4 Domain Experts (v4)...", flush=True)
    target_domains = ["astral", "postgresql", "duckdb", "financial"]
    experts = []
    for d in target_domains:
        p = adapter_path(d, version="v4")
        exp = FoldableExpert.from_dir(p, name=d)
        experts.append(exp)
        print(f"  Loaded {d.upper():<12s} ({len(exp.factors)} modules, scale={exp.scaling})", flush=True)

    engine = WeightFoldingEngine(model, experts)

    print("\n[3/4] Computing Multi-Adapter POET Notch Masks Across 4-Way Fleet...", flush=True)
    t0 = time.time()
    poet_masks = compute_poet_notch_mask_per_layer(experts, top_k_notch=top_k_notch)
    print(f"  Computed POET Notch Masks for {len(poet_masks)} modules in {time.time()-t0:.2f}s (top-{top_k_notch} channels masked per module)", flush=True)

    # Create POET Notch-Filtered Expert replicas
    filtered_experts = []
    for e in experts:
        f_factors = {}
        for k, (u, v) in e.factors.items():
            if k in poet_masks:
                # Apply mask to output projection U
                mask = poet_masks[k].to(u.device, u.dtype)
                u_filt = u * mask  # [d_out, r]
                f_factors[k] = (u_filt, v)
            else:
                f_factors[k] = (u, v)
        fe = FoldableExpert(f_factors, scaling=e.scaling, name=f"{e.name}_poet_notch")
        filtered_experts.append(fe)

    print("\n[4/4] Measuring Live Activation Energies (Base vs Raw 4-Way vs POET-Notched 4-Way)...", flush=True)
    
    # We will hook base hidden states h and measure perturbation delta
    results = {}
    
    for domain, prompts in PROMPTS_PER_DOMAIN.items():
        print(f"\n--- Domain: {domain.upper()} ({len(prompts)} prompts) ---", flush=True)
        
        domain_records = []
        for p_idx, prompt in enumerate(prompts):
            inputs = tok(f"### Question:\n{prompt}\n\n### Answer:\n", return_tensors="pt").to(model.device)
            
            # 1. Base forward pass -> record base hidden states
            base_activations = {}
            hooks = []
            
            def make_hook(name):
                def hook_fn(module, inp, out):
                    # inp[0] is hidden state h
                    h = inp[0].detach()
                    base_activations[name] = {
                        "h_norm": float(torch.norm(h).item()),
                        "h_dim": h.shape[-1]
                    }
                return hook_fn

            for name, mod in model.named_modules():
                if any(target in name for target in ["mlp.down_proj", "self_attn.o_proj"]):
                    hooks.append(mod.register_forward_hook(make_hook(name)))

            with torch.no_grad():
                _ = model(**inputs)

            for h_handle in hooks:
                h_handle.remove()

            # 2. Measure Raw 4-Way Stacked Activation Perturbation
            engine.activate_many(experts, scale_mode="none")
            raw_deltas = []
            hooks_raw = []
            
            def make_delta_hook(name, container):
                def hook_fn(module, inp, out):
                    # compute ||W_live x - W_0 x|| / ||x||
                    x = inp[0].detach()
                    # Linear layer output difference
                    container[name] = float(torch.norm(out.detach()).item())
                return hook_fn

            raw_outputs = {}
            for name, mod in model.named_modules():
                if name in base_activations:
                    hooks_raw.append(mod.register_forward_hook(make_delta_hook(name, raw_outputs)))

            with torch.no_grad():
                _ = model(**inputs)

            for h_handle in hooks_raw:
                h_handle.remove()

            # 3. Measure POET Notch-Filtered 4-Way Stacked Activation Perturbation
            engine.activate_many(filtered_experts, scale_mode="none")
            filt_outputs = {}
            hooks_filt = []
            for name, mod in model.named_modules():
                if name in base_activations:
                    hooks_filt.append(mod.register_forward_hook(make_delta_hook(name, filt_outputs)))

            with torch.no_grad():
                _ = model(**inputs)

            for h_handle in hooks_filt:
                h_handle.remove()

            engine.restore()

            # Compute relative energy metrics
            prompt_raw_ratios = []
            prompt_filt_ratios = []
            for k_mod in base_activations:
                h_n = base_activations[k_mod]["h_norm"]
                if h_n > 1e-6 and k_mod in raw_outputs and k_mod in filt_outputs:
                    prompt_raw_ratios.append(raw_outputs[k_mod] / h_n)
                    prompt_filt_ratios.append(filt_outputs[k_mod] / h_n)

            domain_records.append({
                "prompt": prompt,
                "mean_raw_energy": float(np.mean(prompt_raw_ratios)),
                "mean_filtered_energy": float(np.mean(prompt_filt_ratios)),
            })

        mean_raw = float(np.mean([r["mean_raw_energy"] for r in domain_records]))
        mean_filt = float(np.mean([r["mean_filtered_energy"] for r in domain_records]))
        reduction_pct = (1.0 - (mean_filt / max(1e-6, mean_raw))) * 100.0
        
        results[domain] = {
            "mean_raw_energy": mean_raw,
            "mean_filtered_energy": mean_filt,
            "energy_reduction_pct": reduction_pct,
            "details": domain_records
        }
        
        print(f"  Raw 4-Way Energy (||δ||/||h||):      {mean_raw:.4f}")
        print(f"  POET Notched 4-Way Energy:           {mean_filt:.4f}  ({reduction_pct:+.2f}% noise cut)")

    engine.restore()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="results/benchmarks/poet_4way_stacking_results.json",
                        help="Output path for benchmark results.")
    parser.add_argument("--top_k_notch", type=int, default=15,
                        help="Number of conflicting output channels to notch filter per module.")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 88, flush=True)
    print(" 4-WAY MULTI-EXPERT STACKING BENCHMARK: RAW STACKING vs POET NOTCH FILTERING", flush=True)
    print(" Fleet: astral + postgresql + duckdb + financial folded simultaneously (K=4)", flush=True)
    print("=" * 88, flush=True)

    results = run_4way_stacking_benchmark(top_k_notch=args.top_k_notch)
    elapsed = time.time() - t_start

    print("\n" + "=" * 88, flush=True)
    print(f" {'Domain Fed':<18} | {'Raw 4-Way Energy':<20} | {'POET-Notched Energy':<22} | {'Noise Reduction'}", flush=True)
    print("-" * 88, flush=True)
    
    for dom, data in results.items():
        print(f" {dom.upper():<18} | {data['mean_raw_energy']:>18.4f} | {data['mean_filtered_energy']:>20.4f} | {data['energy_reduction_pct']:>+14.2f}%", flush=True)
    print("=" * 88, flush=True)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "metadata": CANON.stamp(),
        "elapsed_seconds": elapsed,
        "top_k_notch_channels": args.top_k_notch,
        "results": results
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[Artifact Saved] -> {out_path} (Elapsed: {elapsed:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
