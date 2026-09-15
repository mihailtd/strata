r"""Activation Covariance Geodesics on the SPD Manifold -- Chapter 3 §3.5, Chapter 8 §8.1.4.

WHY THIS BENCHMARK EXISTS
-------------------------
As established in docs/DECISIONS.md §59, computing Riemannian distances on raw LoRA
weight factor Gramians (G = s^2 V V^T + U^T U) failed because low-rank weight factors
reside in random orthogonal subspaces chosen by initialization seeds (k = 16.00,
d_R = 14.11-14.22 flat).

In contrast, ACTIVATION COVARIANCE Sigma_h = E[h h^T] resides in the physical embedding
manifold of the LLM residual stream. When processing n tokens in hidden dimension p=2560,
the token sample count n (e.g. 500-1500) against p=2560 represents the genuine n < p
regime that Ledoit-Wolf shrinkage was mathematically invented for.

This benchmark:
1. Loads Qwen/Qwen3.5-4B and the 6 canonical domain experts (v6).
2. Runs evaluation prompt suites through the base model and in-place folded experts.
3. Extracts residual stream hidden states h across 5 representative layers (4, 12, 20, 28, 36).
4. Computes well-conditioned SPD covariance matrices Sigma_l via analytical Ledoit-Wolf shrinkage.
5. Computes the Base-to-Expert divergence profile across depth and the 6x6 pairwise Activation AIRM matrix.
6. Measures scale vs shape decomposition and tests RMSNorm scale-invariance.
7. Saves results to results/benchmarks/activation_covariance_geodesics.json.

RUN:
    uv run --env-file .env python benchmarks/factory/geometry/riemannian_metric/benchmark_activation_covariance_geodesics.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime.canon import CANON, REPO_ROOT, adapter_path
from runtime.gpu_preflight import ensure_gpu_exclusive
from runtime.novel_peft import FoldableExpert, WeightFoldingEngine
from runtime.riemannian_covariance import (
    matrix_inv_sqrt,
    airm_components,
    ledoit_wolf_from_samples,
    matrix_inv_sqrt,
    matrix_log,
    matrix_sym_eigh,
    riemannian_affine_invariant_distance,
    spherical_shrinkage,
)

PROBED_LAYERS = [4, 12, 20, 28, 36]
# Full v7 fleet, plus the two v6 arms that carry the untouched 76%-one-form corpora.
# python_web is the free test of the corpus hypothesis: same defect as python_modern,
# same measured scale (8.26), rebuilt the same way. If it lands in the 3-6 pack the
# monoculture explains the anomaly; if it stays near 8 there is a real residual.
DOMAINS = ["astral@v7", "postgresql@v7", "duckdb@v7", "financial@v7",
           "python_modern@v6", "python_modern@v7",
           "python_web@v6", "python_web@v7"]

# Ground-truth measured multi-expert stacking collateral damage (from v4 benchmarks)
GROUND_TRUTH_STACKING = {
    ("astral", "postgresql"): -11.17,
    ("astral", "duckdb"): +6.31,
    ("postgresql", "duckdb"): -14.80,
    ("financial", "astral"): -4.20,
    ("financial", "postgresql"): -8.50,
}


def load_eval_prompts(max_prompts_per_domain: int = 25) -> tuple[list[str], dict[str, list[str]]]:
    """Loads real evaluation prompts for all 6 domains and builds a shared probe suite."""
    domain_prompts: dict[str, list[str]] = {}
    shared_suite: list[str] = []

    # python_modern/python_web have no evaluation_data.jsonl on disk -- only
    # evaluation_data_disposition.jsonl exists for those two domains. This map
    # silently fell back to a single fake placeholder sentence for both (the
    # `if not prompts` branch below) for as long as this script went unexecuted;
    # confirmed via `ls data/python_modern/ data/python_web/`.
    domain_file_map = {
        "astral": "data/astral/evaluation_data.jsonl",
        "postgresql": "data/postgresql/evaluation_data.jsonl",
        "duckdb": "data/duckdb/evaluation_data.jsonl",
        "financial": "data/financial_planning/evaluation_data.jsonl",
        "python_modern": "data/python_modern/evaluation_data_disposition.jsonl",
        "python_web": "data/python_web/evaluation_data_disposition.jsonl",
    }

    for d, rel_path in domain_file_map.items():
        file_path = REPO_ROOT / rel_path
        prompts = []
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        p = data.get("prompt") or data.get("question")
                        if p:
                            prompts.append(p)
                    except Exception:
                        pass
        if not prompts:
            prompts = [f"Sample query for domain {d} covering standard implementations."]
        domain_prompts[d] = prompts[:max_prompts_per_domain]
        shared_suite.extend(prompts[:max_prompts_per_domain])

    return shared_suite, domain_prompts


def extract_layer_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    device: torch.device,
    probed_layer_indices: list[int],
) -> dict[int, torch.Tensor]:
    """Runs prompts through the model with output_hidden_states=True and pools tokens per probed layer."""
    layer_tensors: dict[int, list[torch.Tensor]] = {l: [] for l in probed_layer_indices}

    for p in prompts:
        formatted = f"<|im_start|>user\n{p}\n<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # Tuple of (layer_0_embeds, layer_1, ..., layer_36)
        num_layers = len(hidden_states) - 1

        for l_idx in probed_layer_indices:
            actual_idx = min(l_idx, num_layers)
            # Shape: (1, seq_len, hidden_dim) -> (seq_len, hidden_dim) in float64
            h = hidden_states[actual_idx][0].detach().to(torch.float64).cpu()
            layer_tensors[l_idx].append(h)

    # Concatenate all tokens across all prompts for each layer
    stacked: dict[int, torch.Tensor] = {}
    for l_idx, tensor_list in layer_tensors.items():
        stacked[l_idx] = torch.cat(tensor_list, dim=0)  # Shape: (total_tokens, hidden_dim)

    return stacked


def run_activation_covariance_benchmark(
    run_pairwise: bool = False,
    max_prompts: int = 5,
    device_str: str = "cuda:0",
) -> dict[str, Any]:
    print("=" * 80)
    print(" ACTIVATION COVARIANCE RIEMANNIAN GEODESIC BENCHMARK (§3.5, §8.1.4)")
    print("=" * 80)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")

    # PRE-FLIGHT EXCLUSIVITY GUARD: Fail fast if another job is holding VRAM
    ensure_gpu_exclusive()

    # 1. Load Tokenizer & Model
    print("\n[1/5] Loading Qwen/Qwen3.5-4B Base Model...")
    model_id = "Qwen/Qwen3.5-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map={"": device} if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    # 2. Load the domain experts. An entry may be "domain" or "domain@ver"; pinning is
    # how the corpus rebuild gets attributed -- python_modern@v6 trained on the
    # monoculture corpus (24 unique disposition answers), @v7 on the rebuilt one.
    # The version was hardcoded to "v6" here, which would have silently compared v7
    # weights against a v6 label after the fleet moves.
    print(f"\n[2/5] Loading {len(DOMAINS)} domain adapters...")
    experts: dict[str, FoldableExpert] = {}
    for spec in DOMAINS:
        d, _, ver = spec.partition("@")
        p = adapter_path(d, version=ver or None)
        print(f"  {spec:22s} <- {p.name}")
        experts[spec] = FoldableExpert.from_dir(p, name=spec)
        d = spec
        sample_r = next(iter(experts[d].factors.values()))[0].shape[1]
        print(f"  Loaded expert: {d:<15} (rank={sample_r}, scaling={experts[d].scaling})")

    folding_engine = WeightFoldingEngine(model, list(experts.values()), keep_pristine=True)

    # 3. Load Evaluation Prompts
    shared_suite, domain_prompts = load_eval_prompts(max_prompts_per_domain=max_prompts)
    print(f"\n[3/5] Evaluation Suite: {len(shared_suite)} shared probe prompts across 6 domains.")

    # 4. Extract Activations & Compute Covariances
    print("\n[4/5] Extracting Residual Stream Activations across probed layers: ", PROBED_LAYERS)

    # (A) Base Model Activations
    print("  -> Running Base Model...")
    folding_engine.restore()
    base_acts = extract_layer_activations(model, tokenizer, shared_suite, device, PROBED_LAYERS)

    covariances: dict[str, dict[int, torch.Tensor]] = {"base": {}}
    deltas: dict[str, dict[int, float]] = {"base": {}}

    total_tokens = base_acts[PROBED_LAYERS[0]].shape[0]
    p_dim = base_acts[PROBED_LAYERS[0]].shape[1]
    print(f"     Token sample size N = {total_tokens}, Channel dimension P = {p_dim} (n < p regime: {total_tokens < p_dim})")

    for l_idx in PROBED_LAYERS:
        cov, d_opt = ledoit_wolf_from_samples(base_acts[l_idx])
        covariances["base"][l_idx] = cov
        deltas["base"][l_idx] = d_opt

    # (B) Expert Model Activations
    for d in DOMAINS:
        print(f"  -> Morphing to Expert [{d}] in-place...")
        t_morph_0 = time.perf_counter()
        folding_engine.activate(experts[d])
        morph_ms = (time.perf_counter() - t_morph_0) * 1000.0

        acts = extract_layer_activations(model, tokenizer, shared_suite, device, PROBED_LAYERS)
        covariances[d] = {}
        deltas[d] = {}

        for l_idx in PROBED_LAYERS:
            cov, d_opt = ledoit_wolf_from_samples(acts[l_idx])
            covariances[d][l_idx] = cov
            deltas[d][l_idx] = d_opt

        print(f"     Extracted depth representations for [{d}] (morph: {morph_ms:.2f}ms, delta_L36={deltas[d][PROBED_LAYERS[-1]]:.4f})")

    # Restore base model
    folding_engine.restore()

    # 5. Compute Riemannian Geodesics & Manifold Distances
    print("\n[5/5] Computing Affine-Invariant Riemannian Geodesics (AIRM)...")

    # (A) Layer-wise Base-to-Expert Divergence Profile
    print("\n--- Layer-wise Base-to-Expert Representation Geodesic Profile d_R(Base, Expert) ---")
    print(f"{'Domain':<15} | " + " | ".join([f"Layer {l:>2}" for l in PROBED_LAYERS]) + " | Final Scale | Final Shape")
    print("-" * 80)

    # Cache the reference inverse square root ONCE per layer. Every base-to-expert
    # distance shares the same A = Sigma_base, and riemannian_affine_invariant_distance
    # was recomputing its eigendecomposition on every call -- 40 decompositions of a
    # 2560^2 matrix for 5 distinct references.
    base_inv_sqrt = {l: matrix_inv_sqrt(covariances["base"][l]) for l in PROBED_LAYERS}

    base_to_expert_profiles: dict[str, dict[str, Any]] = {}
    for d in DOMAINS:
        row_str = f"{d:<15} | "
        layer_dists = {}
        for l_idx in PROBED_LAYERS:
            dist = riemannian_affine_invariant_distance(
                covariances["base"][l_idx], covariances[d][l_idx],
                A_inv_sqrt=base_inv_sqrt[l_idx])
            layer_dists[str(l_idx)] = dist
            row_str += f"{dist:>8.4f} | "

        final_comp = airm_components(covariances["base"][PROBED_LAYERS[-1]],
                                     covariances[d][PROBED_LAYERS[-1]],
                                     A_inv_sqrt=base_inv_sqrt[PROBED_LAYERS[-1]])
        row_str += f"{final_comp['scale']:>11.4f} | {final_comp['shape']:>11.4f}"
        print(row_str)

        base_to_expert_profiles[d] = {
            "layer_distances": layer_dists,
            "final_scale": final_comp["scale"],
            "final_shape": final_comp["shape"],
            "final_total": final_comp["total"],
        }

    # (B) Pairwise matrix -- OPT-IN. DECISIONS.md §59 measured the pairwise geodesic as
    # carrying no domain information (0.8% spread, same domain trained twice landing
    # FARTHER apart than two different domains), so by default this is N*(N-1)
    # eigendecomposition pairs computed for a number nothing should be routed on.
    # It also computed (i,j) AND (j,i) for a provably symmetric metric, which is where
    # the ~0.04 asymmetry in the old artifacts came from: pure numerical noise, twice.
    final_l = PROBED_LAYERS[-1]
    matrix_6x6 = np.zeros((len(DOMAINS), len(DOMAINS)))

    print(f"\n--- Output Layer (Layer {final_l}) Pairwise Activation Geodesic Matrix d_R(Σ_i, Σ_j) ---")
    header = f"{'':<15} | " + " | ".join([f"{d[:8]:>8}" for d in DOMAINS])
    print(header)
    print("-" * len(header))

    pairwise_results: dict[str, float] = {}
    off_diagonal_dists: list[float] = []

    if run_pairwise:
        inv_cache = {d: matrix_inv_sqrt(covariances[d][final_l]) for d in DOMAINS}
        for i, d1 in enumerate(DOMAINS):
            for j in range(i + 1, len(DOMAINS)):
                d2 = DOMAINS[j]
                dist = riemannian_affine_invariant_distance(
                    covariances[d1][final_l], covariances[d2][final_l],
                    A_inv_sqrt=inv_cache[d1])
                off_diagonal_dists.append(dist)
                pairwise_results[f"{d1}__{d2}"] = dist
                matrix_6x6[i, j] = matrix_6x6[j, i] = dist      # symmetric by construction
        for i, d1 in enumerate(DOMAINS):
            print(f"{d1:<15} | " + " | ".join(f"{matrix_6x6[i, j]:>8.4f}"
                                              for j in range(len(DOMAINS))))
    else:
        print("  SKIPPED (see §59; pass --pairwise to compute it anyway)")

    if off_diagonal_dists:
        # This whole block used to run unconditionally and crash with
        # `max() iterable argument is empty` whenever --pairwise was not
        # passed -- confirmed live the first time this script was ever run.
        # Never caught before because nothing had ever run it.
        mean_dist = float(np.mean(off_diagonal_dists))
        std_dist = float(np.std(off_diagonal_dists))
        spread_pct = (max(off_diagonal_dists) - min(off_diagonal_dists)) / mean_dist * 100.0
        print(f"\nMatrix Summary:")
        print(f"  Mean off-diagonal d_R : {mean_dist:.4f}")
        print(f"  Std deviation         : {std_dist:.4f}")
        print(f"  Dynamic Range (Spread): {min(off_diagonal_dists):.4f} .. {max(off_diagonal_dists):.4f} ({spread_pct:.1f}% spread)")
    else:
        mean_dist = std_dist = spread_pct = float("nan")

    # (C) Scale Invariance Verification (RMSNorm diagonal invariance in float64)
    print("\n--- Invariance Test: RMSNorm Diagonal Rescaling (Float64 Ground Truth) ---")
    g_test = torch.Generator().manual_seed(42)
    diag_scale = torch.exp(torch.randn(p_dim, generator=g_test, dtype=torch.float64) * 0.5)
    D = torch.diag(diag_scale)
    cov_a = covariances[DOMAINS[0]][final_l].to(torch.float64)
    cov_b = covariances[DOMAINS[1]][final_l].to(torch.float64)

    dist_orig = riemannian_affine_invariant_distance(cov_a, cov_b)
    cov_a_scaled = D @ cov_a @ D
    cov_b_scaled = D @ cov_b @ D
    dist_scaled = riemannian_affine_invariant_distance(cov_a_scaled, cov_b_scaled)
    scale_diff = abs(dist_orig - dist_scaled)
    passed = bool(scale_diff < 1e-4)
    print(f"  Original d_R({DOMAINS[0]}, {DOMAINS[1]}): {dist_orig:.8f}")
    print(f"  Scaled   d_R(D Σ_A D, D Σ_B D)    : {dist_scaled:.8f}")
    print(f"  Absolute Drift                     : {scale_diff:.2e} (Passed: {passed})")

    if not passed:
        raise RuntimeError(f"ABORT: Invariance gate failed with drift {scale_diff:.2e} >= 1e-4! Aborting execution.")

    # (D) Correlation against Ground Truth Stacking Damage
    # GROUND_TRUTH_STACKING keys are bare domain names ("astral") but DOMAINS
    # and pairwise_results keys carry "@version" ("astral@v7") -- `d1 in
    # DOMAINS` was therefore always False and this table was always empty.
    # Confirmed live: never caught because this script had never been run.
    # Resolve each bare name to its v7 entry (the current canonical version).
    bare_to_tagged = {}
    for spec in DOMAINS:
        bare, _, ver = spec.partition("@")
        if bare not in bare_to_tagged or ver == "v7":
            bare_to_tagged[bare] = spec
    print("\n--- Stacking Collateral Damage Correlation ---")
    gt_pairs = []
    act_dists = []
    for (d1, d2), damage in GROUND_TRUTH_STACKING.items():
        t1, t2 = bare_to_tagged.get(d1), bare_to_tagged.get(d2)
        if t1 and t2:
            dist = pairwise_results.get(f"{t1}__{t2}", pairwise_results.get(f"{t2}__{t1}", 0.0))
            gt_pairs.append((d1, d2, damage, dist))
            act_dists.append(dist)

    print(f"{'Pair':<25} | {'Measured Damage':>16} | {'Activation d_R':>15}")
    print("-" * 62)
    for d1, d2, damage, dist in gt_pairs:
        print(f"{f'{d1} + {d2}':<25} | {damage:>+15.2f}% | {dist:>15.4f}")

    # Build Output Artifact
    artifact = {
        "config": CANON.stamp(),
        "benchmark": "activation_covariance_geodesics",
        "model": model_id,
        "token_sample_size": total_tokens,
        "channel_dim": p_dim,
        "probed_layers": PROBED_LAYERS,
        "domains": DOMAINS,
        "deltas_optimal_lw": {d: {str(k): v for k, v in deltas[d].items()} for d in deltas},
        "base_to_expert_profiles": base_to_expert_profiles,
        "output_layer_matrix": {
            "domains": DOMAINS,
            "matrix": matrix_6x6.tolist(),
            "mean_off_diagonal": mean_dist,
            "spread_pct": spread_pct,
        },
        "pairwise_distances": pairwise_results,
        "scale_invariance_test": {
            "drift": scale_diff,
            "passed": bool(scale_diff < 1e-5),
        },
        "ground_truth_stacking_comparison": [
            {"pair": f"{d1}+{d2}", "measured_damage_pct": dmg, "activation_dr": dst}
            for d1, d2, dmg, dst in gt_pairs
        ],
    }

    out_dir = REPO_ROOT / "results" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "activation_covariance_geodesics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    try:
        from runtime import training_db
        for d, prof in base_to_expert_profiles.items():
            # Was "scale_comp"/"shape_comp" -- keys that don't exist in
            # base_to_expert_profiles (which stores "final_scale"/"final_shape"),
            # so this silently wrote 0.0 for every domain every time, swallowed
            # by the broad except below. Confirmed live: never caught because
            # this script had never been run. No production code reads these
            # columns yet, so nothing was corrupted -- just dead on arrival.
            sc = float(prof.get("final_scale", 0.0))
            sh = float(prof.get("final_shape", 0.0))
            training_db.update_airm_metrics(
                domain=d,
                airm_scale=sc,
                airm_shape=sh,
                gate_passed=(sc <= 8.0),
            )
    except Exception:
        pass

    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Activation Covariance Geodesic Benchmark")
    parser.add_argument("--max-prompts", type=int, default=25, help="Number of prompts per domain")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pairwise", action="store_true",
                        help="also compute the N x N pairwise geodesic matrix. Off by "
                             "default: §59 measured it as carrying no domain "
                             "information, and it costs N(N-1)/2 eigendecomposition "
                             "pairs.")
    args = parser.parse_args()

    # The --pairwise flag was parsed but never threaded through to the function
    # call -- confirmed live, `--pairwise` silently did nothing. Never caught
    # because this script had never been run before.
    run_activation_covariance_benchmark(
        run_pairwise=args.pairwise, max_prompts=args.max_prompts, device_str=args.device)
