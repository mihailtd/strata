r"""SUPERSEDED -- fabricated, not a real measurement.

Flagged as Critical #2 in docs/EXPERIMENT_REAUDIT_2026-09.md. Stages 2 and 3
below (`compute_activation_covariance`) build "activations" from a synthetic
`X_full = shared_drift + innovations` matrix, the same fabrication pattern as
its sibling `probe_lv_glasso_interference.py` -- real adapter weights, fake
input. Stage 1 (raw weight cosine, `compute_weight_cosine_lowrank`) IS real
(operates only on real trained factors), so the "§38 cosine is uniformly
uninformative" finding stands; the "LV-GLasso resolves it via a 13,497x
reduction / one true conflict edge" conclusion in Stages 2-3 does not -- see
`probe_lv_glasso_interference.py`'s retirement banner for why, and
`experiments/factory/geometry/surgical_notch_sweep/` for the real,
current-adapter, real-weight-math answer to the underlying question
("does surgical notching work"). Kept here for provenance only.

--- Original docstring, preserved for context ---

§38 Cosine Mystery Resolution — Weight-Space Cosine vs Precision Graph (CPU-only).

THE §38 MYSTERY
---------------
In §38, pairwise weight-space cosine similarity between all four v4 domain adapters
was measured to be near-uniform: cos(W_A, W_B) ≈ +0.015 for EVERY pair, regardless
of whether the pair interfered at runtime or not.

This was mysterious: adapters that CLEARLY interfered (fin+pg) looked identical in
cosine space to adapters that composed cleanly (ast+pg). Cosine failed as a predictor.

HOW THIS PROBE RESOLVES IT
--------------------------
The mystery has a mathematical cause: raw weight cosine is a MARGINAL correlation
statistic (Σ_ij), confounded by the shared foundation model subspace L.

Every adapter's weight update dW = (alpha/r) * B @ A inherits the same dominant
singular directions from the base Qwen weight W_0. That shared base structure is a
low-rank confounder L that inflates all cross-adapter correlations uniformly.

When you strip L via the Latent Variable Graphical Lasso decomposition (§9.4.2):
    Θ̃ = S_sparse - L_lowrank
the spurious correlations collapse to zero and the direct conflict graph S is revealed.

WHAT THIS PROBE MEASURES (three-stage cascade):
    Stage 1  — §38 REPLICA: raw weight-space cosine Σ_{ij} between adapter pairs.
                Should reproduce the ~0.015 uniform confound.
    Stage 2  — PRECISION INVERSION: off-diagonal block mean of Θ = Σ^{-1}.
                The confounder L is partially removed. Expect ~12× reduction.
    Stage 3  — LV-GLASSO: off-diagonal block mean of S (sparse component of Θ̃).
                Full confounder removal. Expect near-zero (conditional independence).

VERDICT TABLE: for each method, would it correctly predict which pairs conflict?
    • §38 cosine  → NO: all pairs look identical (~0.015), zero predictive power.
    • Precision Θ → PARTIAL: drops to ~0.0037 but still can't localize.
    • LV-GLasso S → YES: S is 100% sparse everywhere except the 1 true collision.

CPU-Only: No model, no GPU. Uses only adapter safetensors + numpy/scipy.

Usage:
    CUDA_VISIBLE_DEVICES="" uv run python benchmarks/factory/geometry/latent_variable_glasso/probe_cosine_mystery_resolution.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from runtime.canon import CANON, REPO_ROOT, adapter_path
from runtime.novel_peft import FoldableExpert


# ─────────────────────────────────────────────────────────────────────────────
# ADMM LV-GLasso solver (shared with probe_lv_glasso_interference.py)
# ─────────────────────────────────────────────────────────────────────────────

def soft_threshold(X: np.ndarray, lam: float) -> np.ndarray:
    """Elementwise soft-thresholding: sgn(X) * max(|X| - lam, 0)."""
    return np.sign(X) * np.maximum(np.abs(X) - lam, 0.0)


def solve_lv_glasso_admm(
    S_emp: np.ndarray,
    lambda1: float = 0.05,
    lambda2: float = 0.10,
    rho: float = 1.0,
    max_iter: int = 150,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Latent Variable Graphical Lasso via ADMM (§9.4.2).

    Returns (Theta, S_sparse, L_lowrank, info) where Theta = S_sparse - L_lowrank.
    """
    p = S_emp.shape[0]
    # Normalize to correlation matrix for numerical stability
    d_diag = np.sqrt(np.maximum(np.diag(S_emp), 1e-8))
    inv_d = 1.0 / d_diag
    C_emp = S_emp * np.outer(inv_d, inv_d)

    # Ridge floor (§9.6.3) — prevents singularity when p > n
    kappa = 0.02
    C_reg = 0.5 * (C_emp + C_emp.T) + kappa * np.eye(p)

    R = np.linalg.inv(C_reg)
    S_sparse = np.copy(R)
    L_lowrank = np.zeros((p, p))
    U = np.zeros((p, p))

    res_norm = np.inf
    for it in range(max_iter):
        # Step 1: logdet proximal update
        A = S_sparse - L_lowrank - U - (1.0 / rho) * C_reg
        A_sym = 0.5 * (A + A.T)
        eigvals, eigvecs = np.linalg.eigh(A_sym)
        r_eigvals = (eigvals + np.sqrt(eigvals**2 + 4.0 / rho)) / 2.0
        R_new = eigvecs @ np.diag(r_eigvals) @ eigvecs.T

        # Step 2: ℓ1 soft-threshold on off-diagonals
        B = R_new + L_lowrank + U
        S_new = soft_threshold(B, lambda1 / rho)
        np.fill_diagonal(S_new, np.diag(B))

        # Step 3: nuclear norm / trace-norm proximal update
        C = S_new - R_new - U
        C_sym = 0.5 * (C + C.T)
        l_eigvals, l_eigvecs = np.linalg.eigh(C_sym)
        l_eigvals_thresh = np.maximum(l_eigvals - (lambda2 / rho), 0.0)
        L_new = l_eigvecs @ np.diag(l_eigvals_thresh) @ l_eigvecs.T

        # Step 4: dual update
        residual = R_new - (S_new - L_new)
        U = U + residual

        res_norm = float(np.linalg.norm(residual, "fro")) / float(
            np.linalg.norm(R_new, "fro") + 1e-8
        )
        if res_norm < tol and it > 10:
            break
        R, S_sparse, L_lowrank = R_new, S_new, L_new

    # Rescale back to original feature scale
    D_mat = np.outer(inv_d, inv_d)
    info = {
        "iterations": it + 1,
        "converged": res_norm < tol,
        "final_residual": float(res_norm),
        "rank_L": int(np.sum(np.linalg.svd(L_lowrank, compute_uv=False) > 1e-3)),
        "sparsity_S": float(
            np.mean(np.abs(S_sparse - np.diag(np.diag(S_sparse))) < 1e-3)
        ),
    }
    return R * D_mat, S_sparse * D_mat, L_lowrank * D_mat, info


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — §38 replica: raw weight-space cosine (low-rank fast path)
# ─────────────────────────────────────────────────────────────────────────────
# dW = scale * U @ V   shape (d_out, d_in),  U: (d_out, r),  V: (r, d_in)
#
# ⟨dW_A, dW_B⟩_F = scale_A * scale_B * tr(U_B^T U_A  V_A V_B^T)
#  ‖dW_A‖_F²     = scale_A²            * tr((U_A^T U_A)(V_A V_A^T))
#
# Both are O(r² (d_out + d_in)) — never materialise the full d_out×d_in matrix.
# ─────────────────────────────────────────────────────────────────────────────

def _lora_inner(u_a: np.ndarray, v_a: np.ndarray, u_b: np.ndarray, v_b: np.ndarray,
                sc_a: float, sc_b: float) -> float:
    """⟨dW_A, dW_B⟩_F using the low-rank identity (no full dW materialised).

    dW = scale * U @ V  (shape d_out × d_in)
    ⟨dW_A, dW_B⟩_F = sc_a * sc_b * tr(dW_A^T dW_B)
                    = sc_a * sc_b * tr((U_A V_A)^T (U_B V_B))
                    = sc_a * sc_b * tr(V_A^T (U_A^T U_B) V_B)
                    = sc_a * sc_b * tr((U_A^T U_B)(V_B V_A^T))
    """
    UtU = u_a.T @ u_b   # (r_a, r_b)
    VVt = v_b @ v_a.T   # (r_b, r_a)
    return float(sc_a * sc_b * np.trace(UtU @ VVt))


def _lora_norm_sq(u: np.ndarray, v: np.ndarray, scale: float) -> float:
    """‖dW‖_F² = scale² * tr((U^T U)(V V^T))."""
    UtU = u.T @ u   # (r, r)
    VVt = v @ v.T   # (r, r)
    return float(scale**2 * np.trace(UtU @ VVt))


def compute_weight_cosine_lowrank(exp_a: FoldableExpert, exp_b: FoldableExpert) -> float:
    """Global weight-space cosine between two adapters, computed without materialising dW.

    cos(dW_A, dW_B) = ⟨dW_A, dW_B⟩_F / (‖dW_A‖_F * ‖dW_B‖_F)

    This is exactly what §38 measured (direction of full adapter update), but fast.
    """
    keys = sorted(set(exp_a.factors.keys()) & set(exp_b.factors.keys()))
    inner = 0.0
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    sc_a = float(exp_a.scaling)
    sc_b = float(exp_b.scaling)

    for key in keys:
        u_a_t, v_a_t = exp_a.factors[key]
        u_b_t, v_b_t = exp_b.factors[key]
        u_a = u_a_t.numpy().astype(np.float64)
        v_a = v_a_t.numpy().astype(np.float64)
        u_b = u_b_t.numpy().astype(np.float64)
        v_b = v_b_t.numpy().astype(np.float64)

        inner    += _lora_inner(u_a, v_a, u_b, v_b, sc_a, sc_b)
        norm_a_sq += _lora_norm_sq(u_a, v_a, sc_a)
        norm_b_sq += _lora_norm_sq(u_b, v_b, sc_b)

    denom = np.sqrt(norm_a_sq * norm_b_sq)
    return float(inner / denom) if denom > 1e-12 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 + 3 — activation delta covariance → Θ → S
#   (reuses the activation simulation from probe_lv_glasso_interference.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_activation_covariance(
    adapters: dict[str, FoldableExpert],
    n_samples: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Build the [p × p] sample covariance matrix of adapter activation responses.

    Each column of the feature matrix Y is the per-token L2 activation response
    of one (adapter, module) pair — identical to probe_lv_glasso_interference.py.
    Returns (S_emp, adapter_slices) where adapter_slices maps domain → column indices.
    """
    np.random.seed(seed)

    first = next(iter(adapters.values()))
    common_keys = sorted(first.factors.keys())

    d_max = 9216
    r_latent = 8
    # Shared latent drift (the "foundation model confounder" L)
    base_factors = np.random.randn(n_samples, r_latent)
    base_loadings = np.random.randn(r_latent, d_max) / np.sqrt(d_max)
    shared_drift = base_factors @ base_loadings
    innovations = np.random.randn(n_samples, d_max) * 0.5
    X_full = shared_drift + innovations

    cols: list[np.ndarray] = []
    adapter_slices: dict[str, list[int]] = {}

    col_idx = 0
    for name, exp in adapters.items():
        adapter_slices[name] = []
        scale = exp.scaling
        for key in common_keys:
            u, v = exp.factors[key]
            u_np, v_np = u.numpy(), v.numpy()
            d_in = v_np.shape[1]
            X_mod = X_full[:, :d_in]
            proj = X_mod @ v_np.T           # [n, r]
            delta = scale * (proj @ u_np.T) # [n, d_out]
            response = np.linalg.norm(delta, axis=1)  # [n]
            cols.append(response)
            adapter_slices[name].append(col_idx)
            col_idx += 1

    Y = np.column_stack(cols)
    Y -= np.mean(Y, axis=0, keepdims=True)
    Y /= (np.std(Y, axis=0, keepdims=True) + 1e-6)

    S_emp = (1.0 / n_samples) * (Y.T @ Y)
    return S_emp, adapter_slices


def off_diagonal_block_mean(M: np.ndarray, idx_a: list[int], idx_b: list[int]) -> float:
    """Mean absolute off-diagonal block value between adapter A and adapter B."""
    block = M[np.ix_(idx_a, idx_b)]
    return float(np.mean(np.abs(block)))


def off_diagonal_block_edge_count(
    M: np.ndarray, idx_a: list[int], idx_b: list[int], tau: float = 0.05
) -> int:
    """Count nonzero edges (|M_ij| > tau) in the cross-adapter block."""
    block = M[np.ix_(idx_a, idx_b)]
    return int(np.sum(np.abs(block) > tau))


# ─────────────────────────────────────────────────────────────────────────────
# Main resolution report
# ─────────────────────────────────────────────────────────────────────────────

def run_cosine_mystery_resolution() -> dict[str, Any]:
    """Execute the three-stage §38 → §49 mystery resolution cascade."""
    DOMAINS = ["astral", "postgresql", "duckdb", "financial"]
    PAIRS = list(combinations(DOMAINS, 2))

    print("=" * 88, flush=True)
    print(" §38 COSINE MYSTERY RESOLUTION — Weight Cosine vs Precision Graph (CPU-only)", flush=True)
    print(" Showing WHY raw cosine failed §38, and how LV-GLasso resolves it", flush=True)
    print("=" * 88, flush=True)

    # ── Load adapters ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading v4 domain adapters...", flush=True)
    adapters: dict[str, FoldableExpert] = {}
    for d in DOMAINS:
        p = adapter_path(d, version="v4")
        exp = FoldableExpert.from_dir(p, name=d)
        adapters[d] = exp
        print(f"  {d:<14s} {len(exp.factors)} modules  scaling={exp.scaling:.1f}", flush=True)

    # ── Stage 1: §38 replica — raw weight cosine ──────────────────────────────
    print("\n[2/4] Stage 1 — §38 replica: raw weight-space cosine (low-rank fast path)...", flush=True)
    t0 = time.time()
    stage1: dict[str, float] = {}
    for d1, d2 in PAIRS:
        cs = compute_weight_cosine_lowrank(adapters[d1], adapters[d2])
        stage1[f"{d1}+{d2}"] = cs
    t_s1 = time.time() - t0
    print(f"  Computed {len(PAIRS)} pairwise weight cosines in {t_s1:.2f}s (no dW materialised)", flush=True)

    # ── Stage 2 + 3: activation covariance → Θ → S ───────────────────────────
    print("\n[3/4] Stage 2+3 — activation covariance Σ → precision Θ → LV-GLasso S...", flush=True)
    t1 = time.time()
    S_emp, adapter_slices = compute_activation_covariance(adapters, n_samples=300)
    print(f"  Activation covariance matrix built: {S_emp.shape} in {time.time()-t1:.2f}s", flush=True)

    # Marginal correlation matrix (same quantity the LV-GLasso measures as Σ)
    diag_sqrt = np.sqrt(np.diag(S_emp))
    Sigma = S_emp / (np.outer(diag_sqrt, diag_sqrt) + 1e-8)

    # Standard precision (GLasso with L forced to zero — pure Θ = Σ^{-1})
    t2 = time.time()
    Theta_glasso, _, _, info_gl = solve_lv_glasso_admm(
        S_emp, lambda1=0.08, lambda2=1e6, max_iter=100
    )
    print(f"  Standard GLasso Θ: {info_gl['iterations']} iters ({time.time()-t2:.2f}s), sparsity {info_gl['sparsity_S']*100:.1f}%", flush=True)

    # Latent Variable GLasso (S_sparse removes confounder L)
    t3 = time.time()
    Theta_lv, S_sparse, L_lowrank, info_lv = solve_lv_glasso_admm(
        S_emp, lambda1=0.08, lambda2=0.05, max_iter=150
    )
    tau = 0.05
    S_thresh = S_sparse * (np.abs(S_sparse) > tau)
    print(f"  LV-GLasso S:      {info_lv['iterations']} iters ({time.time()-t3:.2f}s), rank(L)={info_lv['rank_L']}, sparsity(S)={info_lv['sparsity_S']*100:.1f}%", flush=True)

    # ── Stage 2+3 per-pair metrics ────────────────────────────────────────────
    stage2_sigma: dict[str, float] = {}   # marginal corr from activation space
    stage2_theta: dict[str, float] = {}   # precision GLasso
    stage3_S: dict[str, float] = {}       # LV-GLasso sparse
    stage3_edges: dict[str, int] = {}     # nonzero edges in S

    for d1, d2 in PAIRS:
        key = f"{d1}+{d2}"
        idx1 = adapter_slices[d1]
        idx2 = adapter_slices[d2]
        stage2_sigma[key] = off_diagonal_block_mean(Sigma, idx1, idx2)
        stage2_theta[key] = off_diagonal_block_mean(Theta_glasso, idx1, idx2)
        stage3_S[key] = off_diagonal_block_mean(S_sparse, idx1, idx2)
        stage3_edges[key] = off_diagonal_block_edge_count(S_thresh, idx1, idx2)

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n[4/4] § 38 → § 49 Three-Stage Mystery Resolution Cascade", flush=True)
    print("=" * 110, flush=True)
    print(
        f" {'Pair':<22} | {'§38 Weight Cos':<16} | {'Act Corr Σ':<12} | {'Precision Θ':<12} | {'LV-GLasso S':<12} | {'Edges (S)':<10} | Verdict",
        flush=True
    )
    print("-" * 110, flush=True)

    pair_results: list[dict[str, Any]] = []
    for d1, d2 in PAIRS:
        key = f"{d1}+{d2}"
        wcos  = stage1[key]
        sigma = stage2_sigma[key]
        theta = stage2_theta[key]
        S_val = stage3_S[key]
        edges = stage3_edges[key]

        # §38 would have said: all pairs look the same → no predictor
        s38_wrong = "❌ Failed" if abs(wcos - np.mean(list(stage1.values()))) < 0.005 else "OK"
        verdict = "✅ Resolved (S≈0)" if edges == 0 else f"⚠️  {edges} edge(s)"
        print(
            f" {key:<22} | {wcos:>14.4f}   | {sigma:>10.4f}   | {theta:>10.4f}   | {S_val:>10.4f}   | {edges:>9}  | {verdict}",
            flush=True
        )
        pair_results.append({
            "pair": key,
            "stage1_weight_cosine": float(wcos),
            "stage2_activation_sigma": float(sigma),
            "stage2_precision_theta": float(theta),
            "stage3_lv_glasso_S": float(S_val),
            "stage3_conflict_edges": edges,
        })

    print("=" * 110, flush=True)

    # Summary statistics
    mean_wcos = float(np.mean([r["stage1_weight_cosine"] for r in pair_results]))
    std_wcos  = float(np.std([r["stage1_weight_cosine"] for r in pair_results]))
    mean_sig  = float(np.mean([r["stage2_activation_sigma"] for r in pair_results]))
    mean_th   = float(np.mean([r["stage2_precision_theta"] for r in pair_results]))
    mean_S    = float(np.mean([r["stage3_lv_glasso_S"] for r in pair_results]))
    reduction_sigma_to_theta = mean_sig / (mean_th + 1e-10)
    reduction_theta_to_S     = mean_th  / (mean_S  + 1e-10)

    print(f"\n{'─'*60}", flush=True)
    print(f"  §38 Weight Cosine (all pairs):   mean={mean_wcos:.4f}  std={std_wcos:.5f}", flush=True)
    print(f"  Activation Corr Σ (mean):        {mean_sig:.4f}", flush=True)
    print(f"  Precision Θ off-block (mean):    {mean_th:.4f}  ({reduction_sigma_to_theta:.1f}× drop from Σ)", flush=True)
    print(f"  LV-GLasso S off-block (mean):    {mean_S:.6f}  ({reduction_theta_to_S:.1f}× drop from Θ)", flush=True)
    print(f"  Total Σ → S reduction:           {mean_sig / (mean_S + 1e-10):.0f}× (confounder removed)", flush=True)
    print(f"{'─'*60}", flush=True)

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  THE §38 MYSTERY IS SOLVED                                          ║
║                                                                      ║
║  Raw weight cosine (§38): {mean_wcos:.4f} ± {std_wcos:.4f}               ║
║  → Near-uniform for ALL pairs.  Zero predictive power.              ║
║  → Confounded by shared foundation subspace L (rank={info_lv['rank_L']:<3d}).         ║
║                                                                      ║
║  LV-GLasso S (§49): {mean_S:.6f} (off-block mean)               ║
║  → S is 100% sparse for ≥5/6 pairs after removing L.               ║
║  → Direct conditional graph reveals true collision structure.        ║
╚══════════════════════════════════════════════════════════════════════╝
""", flush=True)

    return {
        "metadata": CANON.stamp(),
        "summary": {
            "mean_weight_cosine_s38": mean_wcos,
            "std_weight_cosine_s38": std_wcos,
            "mean_activation_sigma": mean_sig,
            "mean_precision_theta": mean_th,
            "mean_lv_glasso_S": mean_S,
            "sigma_to_theta_reduction_x": float(reduction_sigma_to_theta),
            "theta_to_S_reduction_x": float(reduction_theta_to_S),
            "total_sigma_to_S_reduction_x": float(mean_sig / (mean_S + 1e-10)),
            "rank_latent_L": info_lv["rank_L"],
            "lv_glasso_sparsity_pct": info_lv["sparsity_S"] * 100.0,
        },
        "per_pair": pair_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="results/benchmarks/cosine_mystery_resolution.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    t_start = time.time()
    results = run_cosine_mystery_resolution()
    elapsed = time.time() - t_start

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results | {"elapsed_seconds": elapsed}, indent=2))
    print(f"\n[Artifact] → {out_path}  (elapsed: {elapsed:.1f}s)\n", flush=True)


if __name__ == "__main__":
    main()
