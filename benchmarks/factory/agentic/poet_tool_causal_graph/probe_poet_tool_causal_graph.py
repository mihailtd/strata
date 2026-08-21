"""POET + NOTEARS Continuous Causal DAG Learning for Agent Tool Graphs.

Implements and evaluates:
1. Structural Equation Models for Agent Tool Workflows (Ch 10 §10.1 & §10.3)
2. POET Confounder Factor Filtering: Sigma = L_global_task + Sigma_causal (Ch 7 §7.3.1)
3. Continuous Acyclicity Optimization via NOTEARS (h(W) = Tr(exp(W o W)) - d = 0) (Ch 10 §10.3)
4. Evaluation of Causal Discovery (TPR, FDR, SHD) across sample sizes N in [100, 500, 2000].

Usage:
  uv run python benchmarks/factory/agentic/poet_tool_causal_graph/probe_poet_tool_causal_graph.py [--out results/benchmarks/poet_tool_causal_graph.json]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg as sla
from scipy.optimize import minimize

from runtime.canon import CANON, REPO_ROOT

TOOL_NAMES = [
    "uv_init",        # 0
    "uv_add",         # 1
    "edit_code",      # 2
    "uv_lock",        # 3
    "run_pytest",     # 4
    "sqlglot_parse",  # 5
    "duckdb_query",   # 6
    "export_parquet", # 7
    "git_commit",     # 8
    "push_repo",      # 9
]


def build_ground_truth_dag() -> np.ndarray:
    """Builds ground-truth 10-node causal DAG adjacency matrix W_true."""
    d = len(TOOL_NAMES)
    W = np.zeros((d, d), dtype=np.float64)
    
    # Tool Causal Workflow Transitions
    W[0, 1] = 0.85  # uv_init -> uv_add
    W[1, 2] = 0.90  # uv_add -> edit_code
    W[2, 3] = 0.75  # edit_code -> uv_lock
    W[2, 4] = 0.80  # edit_code -> run_pytest
    W[3, 4] = 0.70  # uv_lock -> run_pytest
    W[5, 6] = 0.85  # sqlglot_parse -> duckdb_query
    W[6, 7] = 0.80  # duckdb_query -> export_parquet
    W[4, 8] = 0.95  # run_pytest -> git_commit
    W[7, 8] = 0.65  # export_parquet -> git_commit
    W[8, 9] = 0.90  # git_commit -> push_repo
    
    return W


def simulate_tool_execution_data(
    W_true: np.ndarray,
    n_samples: int = 500,
    confounder_strength: float = 0.75,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulates linear Structural Equation Model X = X @ W + Z + Confounder (Ch 10 §10.1)."""
    np.random.seed(seed)
    d = W_true.shape[0]
    
    # Exogenous noise Z ~ N(0, 1)
    Z = np.random.randn(n_samples, d)
    
    # Generate structural DAG data via (I - W^T)^{-1}
    I_minus_W = np.eye(d) - W_true
    I_minus_W_inv = np.linalg.inv(I_minus_W)
    X_pure = Z @ I_minus_W_inv  # [N, d]
    
    # Global task complexity confounder (common pervasive factor L in Ch 7)
    # Tasks with high complexity invoke all tools more frequently
    global_task_load = np.random.randn(n_samples, 1)
    confounder_loadings = np.random.uniform(0.5, 1.0, size=(1, d))
    L_confounder = confounder_strength * (global_task_load @ confounder_loadings)
    
    X_observed = X_pure + L_confounder
    return X_observed, X_pure


def notears_linear(
    X: np.ndarray,
    lambda1: float = 0.05,
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e+16,
    w_threshold: float = 0.3,
) -> np.ndarray:
    """Continuous optimization for structure learning using NOTEARS (Zheng et al., 2018; Book Ch 10 §10.3).
    
    min 1/(2N) ||X - X W||_F^2 + lambda1 ||W||_1
    s.t. h(W) = Tr(exp(W o W)) - d = 0
    """
    n, d = X.shape
    
    def _h(W: np.ndarray) -> tuple[float, np.ndarray]:
        """Acyclicity constraint and its gradient."""
        M = W * W
        E = sla.expm(M)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    def _loss(W: np.ndarray) -> tuple[float, np.ndarray]:
        """Least squares loss and gradient."""
        R = X - X @ W
        loss = 0.5 / float(n) * np.sum(R ** 2)
        G_loss = -1.0 / float(n) * (X.T @ R)
        return loss, G_loss

    def _func(w_vec: np.ndarray, rho: float, alpha: float) -> tuple[float, np.ndarray]:
        W = w_vec.reshape(d, d)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * np.sum(np.abs(W))
        G_obj = G_loss + (rho * h + alpha) * G_h + lambda1 * np.sign(W)
        return obj, G_obj.flatten()

    w_est = np.zeros(d * d)
    rho, alpha, h = 1.0, 0.0, np.inf
    bnds = [(0, 0) if i == j else (None, None) for i in range(d) for j in range(d)]
    
    for _ in range(max_iter):
        w_new = minimize(
            fun=lambda w: _func(w, rho, alpha)[0],
            x0=w_est,
            jac=lambda w: _func(w, rho, alpha)[1],
            bounds=bnds,
            method="L-BFGS-B",
            options={"maxiter": 150}
        ).x
        
        W_new = w_new.reshape(d, d)
        h_new, _ = _h(W_new)
        if h_new > 0.25 * h:
            rho *= 10
        else:
            w_est = w_new
            h = h_new
            alpha += rho * h_new
            if h <= h_tol or rho >= rho_max:
                break

    W_est = w_est.reshape(d, d)
    # Threshold small spurious edges
    W_est[np.abs(W_est) < w_threshold] = 0.0
    return W_est


def poet_filter_confounders(X: np.ndarray, rank: int = 1) -> np.ndarray:
    """Removes pervasive global task confounders via POET low-rank factor decomposition (Ch 7 §7.3.1)."""
    # Centered X
    X_c = X - np.mean(X, axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(X_c, full_matrices=False)
    
    # Leading rank-1 component represents global task activity
    L_factor = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
    
    # Residual matrix contains the isolated causal tool transitions
    X_filtered = X_c - L_factor
    return X_filtered


def count_accuracy(B_true: np.ndarray, B_est: np.ndarray) -> dict[str, float]:
    """Computes TPR, FDR, and Structural Hamming Distance (SHD) for DAG discovery."""
    d = B_true.shape[0]
    B_true_bin = (B_true != 0).astype(int)
    B_est_bin = (B_est != 0).astype(int)
    
    tp = np.sum((B_true_bin == 1) & (B_est_bin == 1))
    fp = np.sum((B_true_bin == 0) & (B_est_bin == 1))
    fn = np.sum((B_true_bin == 1) & (B_est_bin == 0))
    
    total_true = np.sum(B_true_bin)
    total_est = np.sum(B_est_bin)
    
    tpr = float(tp / total_true) if total_true > 0 else 0.0
    fdr = float(fp / total_est) if total_est > 0 else 0.0
    
    # Structural Hamming Distance (edge insertions, deletions, reversals)
    shd = float(np.sum(np.abs(B_true_bin - B_est_bin)))
    
    return {
        "true_edges": int(total_true),
        "predicted_edges": int(total_est),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "tpr": tpr,
        "fdr": fdr,
        "shd": shd
    }


def run_tool_causal_graph_benchmark() -> dict[str, Any]:
    """Evaluates DAG structure learning with and without POET confounder filtering."""
    W_true = build_ground_truth_dag()
    sample_sizes = [100, 300, 1000]
    results_by_n = {}
    
    for n in sample_sizes:
        X_obs, _ = simulate_tool_execution_data(W_true, n_samples=n, confounder_strength=0.80, seed=42 + n)
        
        # 1. Naive Correlation / Thresholding baseline
        corr = np.corrcoef(X_obs, rowvar=False)
        np.fill_diagonal(corr, 0.0)
        W_naive = (np.abs(corr) > 0.40).astype(float)
        acc_naive = count_accuracy(W_true, W_naive)
        
        # 2. Standard NOTEARS (Unfiltered)
        W_notears_raw = notears_linear(X_obs, lambda1=0.05, w_threshold=0.25)
        acc_raw = count_accuracy(W_true, W_notears_raw)
        
        # 3. POET-Filtered NOTEARS (Ours)
        X_poet = poet_filter_confounders(X_obs, rank=1)
        W_notears_poet = notears_linear(X_poet, lambda1=0.05, w_threshold=0.25)
        acc_poet = count_accuracy(W_true, W_notears_poet)
        
        results_by_n[f"N_{n}"] = {
            "sample_size": n,
            "naive_correlation": acc_naive,
            "raw_notears": acc_raw,
            "poet_filtered_notears": acc_poet
        }
        
    return results_by_n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="results/benchmarks/poet_tool_causal_graph.json",
                        help="Output path for benchmark artifact.")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 80, flush=True)
    print(" POET + NOTEARS AGENT TOOL CAUSAL GRAPH DISCOVERY BENCHMARK", flush=True)
    print(" Book Ref: Ch 10 §10.3 (Continuous DAG Optimization & Structural Equation Models)", flush=True)
    print("=" * 80, flush=True)

    results = run_tool_causal_graph_benchmark()
    elapsed = time.time() - t_start

    print("\n  " + "-" * 74, flush=True)
    print(f"  {'Method':<26} | {'Sample Size':<12} | {'TPR (Recall)':<12} | {'FDR':<10} | {'SHD Error'}", flush=True)
    print("  " + "-" * 74, flush=True)
    
    for n_key, data in results.items():
        n_val = data["sample_size"]
        m_naive = data["naive_correlation"]
        m_raw = data["raw_notears"]
        m_poet = data["poet_filtered_notears"]
        
        print(f"  {'Naive Correlation':<26} | N = {n_val:<8d} | {m_naive['tpr']:>10.1%}   | {m_naive['fdr']:>8.1%} | {m_naive['shd']:>6.0f}", flush=True)
        print(f"  {'Standard NOTEARS (Raw)':<26} | N = {n_val:<8d} | {m_raw['tpr']:>10.1%}   | {m_raw['fdr']:>8.1%} | {m_raw['shd']:>6.0f}", flush=True)
        print(f"  {'POET-Filtered NOTEARS':<26} | N = {n_val:<8d} | {m_poet['tpr']:>10.1%}   | {m_poet['fdr']:>8.1%} | {m_poet['shd']:>6.0f}", flush=True)
        print("  " + "-" * 74, flush=True)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_payload = {
        "metadata": CANON.stamp(),
        "elapsed_seconds": elapsed,
        "benchmark": "poet_tool_causal_graph",
        "results": results
    }
    out_path.write_text(json.dumps(final_payload, indent=2))
    print(f"\n[Artifact Saved] -> {out_path} (Elapsed: {elapsed:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
