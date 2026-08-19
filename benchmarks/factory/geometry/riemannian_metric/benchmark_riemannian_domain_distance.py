r"""Benchmark: Log-Covariance Metric & Ledoit-Wolf Shrinkage Geodesic Distance Matrix.

References:
- Chapter 3: §3.3 (Norms and Proximity of Matrices), §3.5 (Log of a Covariance Matrix)
- Chapter 8: §8.1.4 (Ledoit-Wolf Optimal Shrinkage Estimators)

Evaluates:
1. True geometric distance between domain adapters on the Riemannian Manifold of SPD operators.
2. Compares Naive Frobenius / Euclidean distance vs Cosine distance vs Affine-Invariant Riemannian (AIRM) vs Log-Euclidean (LERM).
3. Constructs the 6x6 Geodesic Distance Matrix across all v6 domain experts:
   astral, postgresql, duckdb, financial, python_modern, python_web.

Usage:
    CUDA_VISIBLE_DEVICES="" uv run python benchmarks/factory/geometry/riemannian_metric/benchmark_riemannian_domain_distance.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from gnn_experiment.canon import CANON, REPO_ROOT, adapter_path
from gnn_experiment.novel_peft import FoldableExpert
from gnn_experiment.riemannian_covariance import (
    compute_adapter_gramian,
    ledoit_wolf_shrinkage,
    log_euclidean_distance,
    matrix_log,
    riemannian_affine_invariant_distance,
)

torch.set_num_threads(2)

DOMAINS = ["astral", "postgresql", "duckdb", "financial", "python_modern", "python_web"]


def load_all_v6_experts() -> dict[str, FoldableExpert]:
    """Loads all 6 canonical v6 FoldableExpert instances."""
    experts = {}
    for d in DOMAINS:
        path = adapter_path(d, version="v6")
        experts[d] = FoldableExpert.from_dir(path, name=d)
    return experts


def compute_layer_geodesic_matrices(
    experts: dict[str, FoldableExpert],
    target_module_suffix: str = "down_proj",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict[str, Any]:
    """Computes pairwise distance matrices across all domain experts for a specified module type."""
    domains = list(experts.keys())
    N = len(domains)

    # Accumulate matrices
    mat_euclidean = np.zeros((N, N))
    mat_cosine = np.zeros((N, N))
    mat_airm = np.zeros((N, N))
    mat_log_euclidean = np.zeros((N, N))

    # Find shared module paths matching suffix (e.g. 'down_proj.weight')
    first_expert = experts[domains[0]]
    matching_keys = [k for k in first_expert.factors.keys() if f"{target_module_suffix}.weight" in k or k.endswith(target_module_suffix)]

    evaluated_layers = 0

    for key in matching_keys:
        # Check all experts have this key
        if not all(key in experts[d].factors for d in domains):
            continue

        evaluated_layers += 1

        # Extract Gramians and factor tensors for each domain at this layer
        gramians_lw = []
        u_list = []
        v_list = []
        scalings = []
        norms_sq = []

        for d in domains:
            u, v = experts[d].factors[key]
            s = experts[d].scaling
            u = u.to(device).float()
            v = v.to(device).float()
            u_list.append(u)
            v_list.append(v)
            scalings.append(s)

            # Compute Ledoit-Wolf shrunk subspace Gramian in R^{r x r}
            G = (s**2) * (v @ v.T) + (u.T @ u)
            G_lw, _ = ledoit_wolf_shrinkage(G)
            G_lw = G_lw + 1e-5 * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
            gramians_lw.append(G_lw)

            # Precompute squared Frobenius norm via low-rank trace: tr((U^T U) (V V^T))
            norm_sq = (s**2) * torch.trace((u.T @ u) @ (v @ v.T)).item()
            norms_sq.append(max(0.0, float(norm_sq)))

        # Compute pairwise distances using fast rank-r traces
        for i in range(N):
            for j in range(i, N):
                G_i, G_j = gramians_lw[i], gramians_lw[j]

                # Fast Low-Rank Inner Product: <W_i, W_j> = s_i * s_j * tr((U_i^T U_j) (V_j V_i^T))
                u_i, v_i, s_i = u_list[i], v_list[i], scalings[i]
                u_j, v_j, s_j = u_list[j], v_list[j], scalings[j]

                inner = s_i * s_j * torch.trace((u_i.T @ u_j) @ (v_j @ v_i.T)).item()
                norm_i_sq = norms_sq[i]
                norm_j_sq = norms_sq[j]

                # 1. Frobenius Euclidean Distance: sqrt(||W_i||^2 + ||W_j||^2 - 2<W_i, W_j>)
                dist_sq = max(0.0, norm_i_sq + norm_j_sq - 2.0 * inner)
                d_euc = float(np.sqrt(dist_sq))

                # 2. Cosine Distance: 1 - <W_i, W_j> / (||W_i|| * ||W_j||)
                denom = max(1e-9, np.sqrt(norm_i_sq * norm_j_sq))
                cos = float(inner / denom)
                d_cos = 1.0 - cos

                # 3. Affine-Invariant Riemannian Distance (AIRM)
                d_airm = riemannian_affine_invariant_distance(G_i, G_j)

                # 4. Log-Euclidean Distance (LERM)
                d_le = log_euclidean_distance(G_i, G_j)

                # Accumulate
                mat_euclidean[i, j] += d_euc
                mat_euclidean[j, i] += d_euc
                mat_cosine[i, j] += d_cos
                mat_cosine[j, i] += d_cos
                mat_airm[i, j] += d_airm
                mat_airm[j, i] += d_airm
                mat_log_euclidean[i, j] += d_le
                mat_log_euclidean[j, i] += d_le

    # Average across layers
    if evaluated_layers > 0:
        mat_euclidean /= evaluated_layers
        mat_cosine /= evaluated_layers
        mat_airm /= evaluated_layers
        mat_log_euclidean /= evaluated_layers

    return {
        "domains": domains,
        "evaluated_layers": evaluated_layers,
        "euclidean": mat_euclidean.tolist(),
        "cosine": mat_cosine.tolist(),
        "riemannian_airm": mat_airm.tolist(),
        "log_euclidean": mat_log_euclidean.tolist(),
    }




def run_benchmark() -> dict[str, Any]:
    print("=" * 95, flush=True)
    print(" LOG-COVARIANCE & LEDOIT-WOLF RIEMANNIAN GEODESIC BENCHMARK", flush=True)
    print(" Manifold Geometry of Positive-Definite Operators (Ch 3 §3.5 & Ch 8 §8.1.4)", flush=True)
    print("=" * 95, flush=True)

    t0 = time.time()
    print("\n[1/3] Loading all 6 canonical v6 domain experts...", flush=True)
    experts = load_all_v6_experts()
    print(f"  Loaded {len(experts)} experts in {time.time()-t0:.2f}s", flush=True)

    modules_to_evaluate = ["down_proj", "gate_proj", "q_proj", "v_proj"]
    module_results = {}

    print("\n[2/3] Computing Riemannian Manifold Geodesic Distance Matrices...", flush=True)
    for mod in modules_to_evaluate:
        t_mod = time.time()
        res = compute_layer_geodesic_matrices(experts, target_module_suffix=mod)
        module_results[mod] = res
        print(f"  Evaluated module [{mod:<10s}] across {res['evaluated_layers']} layers in {time.time()-t_mod:.2f}s", flush=True)


    # Aggregate global mean Riemannian and Log-Euclidean distance matrices
    N = len(DOMAINS)
    global_airm = np.zeros((N, N))
    global_lerm = np.zeros((N, N))
    global_euc = np.zeros((N, N))
    global_cos = np.zeros((N, N))

    for mod in modules_to_evaluate:
        global_airm += np.array(module_results[mod]["riemannian_airm"])
        global_lerm += np.array(module_results[mod]["log_euclidean"])
        global_euc += np.array(module_results[mod]["euclidean"])
        global_cos += np.array(module_results[mod]["cosine"])

    global_airm /= len(modules_to_evaluate)
    global_lerm /= len(modules_to_evaluate)
    global_euc /= len(modules_to_evaluate)
    global_cos /= len(modules_to_evaluate)

    # 3. Print Distance Matrices
    print("\n" + "=" * 95, flush=True)
    print(" RIEMANNIAN AFFINE-INVARIANT GEODESIC DISTANCE MATRIX (d_R)", flush=True)
    print(" Scale-Invariant Distance on the Manifold of Positive Definite Operators", flush=True)
    print(" " + "─" * 93, flush=True)
    header = f" {'Domain':<16}" + "".join(f"{d[:8]:>12}" for d in DOMAINS)
    print(header, flush=True)
    print(" " + "─" * 93, flush=True)
    for i, d in enumerate(DOMAINS):
        row = f" {d:<16}" + "".join(f"{global_airm[i, j]:12.4f}" for j in range(N))
        print(row, flush=True)
    print("=" * 95, flush=True)

    print("\n" + "=" * 95, flush=True)
    print(" LOG-EUCLIDEAN RIEMANNIAN DISTANCE MATRIX (d_LE)", flush=True)
    print(" " + "─" * 93, flush=True)
    print(header, flush=True)
    print(" " + "─" * 93, flush=True)
    for i, d in enumerate(DOMAINS):
        row = f" {d:<16}" + "".join(f"{global_lerm[i, j]:12.4f}" for j in range(N))
        print(row, flush=True)
    print("=" * 95, flush=True)

    # Find Nearest Neighbors for each domain
    domain_affinities = {}
    print("\n[3/3] Domain Semantic Affinities (Manifold Nearest Neighbors):", flush=True)
    for i, d in enumerate(DOMAINS):
        dists = [(DOMAINS[j], float(global_airm[i, j])) for j in range(N) if i != j]
        dists.sort(key=lambda x: x[1])
        domain_affinities[d] = {
            "closest_domain": dists[0][0],
            "closest_distance": dists[0][1],
            "furthest_domain": dists[-1][0],
            "furthest_distance": dists[-1][1],
            "all_neighbors_sorted": dists,
        }
        print(f"  Domain [{d:<16s}] → Nearest: {dists[0][0]:<14s} (d_R={dists[0][1]:.3f}) | Furthest: {dists[-1][0]:<14s} (d_R={dists[-1][1]:.3f})", flush=True)

    return {
        "metadata": CANON.stamp(),
        "domains": DOMAINS,
        "evaluated_modules": modules_to_evaluate,
        "global_matrices": {
            "riemannian_airm": global_airm.tolist(),
            "log_euclidean": global_lerm.tolist(),
            "euclidean": global_euc.tolist(),
            "cosine": global_cos.tolist(),
        },
        "module_matrices": module_results,
        "domain_affinities": domain_affinities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="results/benchmarks/riemannian_domain_geodesics.json",
        help="Output JSON artifact path.",
    )
    args = parser.parse_args()

    t_start = time.time()
    results = run_benchmark()
    elapsed = time.time() - t_start

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results | {"elapsed_seconds": elapsed}, indent=2))
    print(f"\n[Artifact] → {out_path}  (elapsed: {elapsed:.2f}s)\n", flush=True)


if __name__ == "__main__":
    main()
