"""POET (Principal Orthogonal ComplEment Thresholding) Low-Rank + Sparse Decomposition Benchmark.

Implements and evaluates:
1. Low-Rank + Sparse Decomposition (Ch 7 §7.3.1)
2. Optimal Nearest Kronecker Decomposition (Van Loan-Pitsianis algorithm)
3. POET Adaptive & Hard Thresholding on Residuals (Ch 7 §7.3.3)
4. Storage Footprint vs Reconstruction Error Curve across all 4 v4 expert adapters.

Usage:
  uv run python benchmarks/factory/geometry/poet_decomposition/probe_poet_decomposition.py [--out results/benchmarks/poet_decomposition_benchmark.json]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from runtime.canon import CANON, DOMAINS, REPO_ROOT, adapter_path


# ---------------------------------------------------------------------------
# Mathematical Operators
# ---------------------------------------------------------------------------

def nearest_kronecker_product(M: torch.Tensor, m1: int, n1: int, niter: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes optimal G1 (m1 x n1) and G2 (m2 x n2) minimizing ||M - G1 (x) G2||_F.
    
    Uses power iteration on the rearranged tensor R in O(m*n) time.
    """
    m, n = M.shape
    assert m % m1 == 0 and n % n1 == 0, f"Cannot factor ({m}, {n}) by ({m1}, {n1})"
    m2, n2 = m // m1, n // n1
    
    # Rearrange M into (m1*n1, m2*n2)
    R = M.reshape(m1, m2, n1, n2).permute(0, 2, 1, 3).reshape(m1 * n1, m2 * n2)
    
    # Power iteration for rank-1 SVD of rearranged matrix
    v = torch.randn(R.shape[1], 1, dtype=R.dtype, device=R.device)
    v = v / (torch.norm(v) + 1e-12)
    for _ in range(niter):
        u = R @ v
        u = u / (torch.norm(u) + 1e-12)
        v = R.T @ u
        v = v / (torch.norm(v) + 1e-12)
    sigma = (u.T @ R @ v).squeeze()
    
    G1 = torch.sqrt(torch.abs(sigma) + 1e-12) * u.reshape(m1, n1)
    G2 = torch.sign(sigma) * torch.sqrt(torch.abs(sigma) + 1e-12) * v.reshape(m2, n2)
    return G1, G2


def factor_dimensions(m: int, n: int) -> tuple[int, int]:
    """Finds balanced (m1, n1) factor dimensions for Kronecker product."""
    if m == 2560 and n == 2560:
        return 64, 64
    elif m == 9216 and n == 2560:
        return 64, 64
    elif m == 2560 and n == 9216:
        return 64, 64
    else:
        def best_divisor(val: int, target: int = 64) -> int:
            divs = [d for d in range(1, val + 1) if val % d == 0 and d <= 128]
            return min(divs, key=lambda x: abs(x - target)) if divs else 1
        return best_divisor(m), best_divisor(n)


def apply_poet_thresholding(
    R: torch.Tensor,
    method: str = "hard",
    sparsity: float = 0.05,
    delta: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Applies POET thresholding (Ch 7 §7.3.3) to residual matrix R."""
    flat_abs = torch.abs(R).flatten()
    total_elements = flat_abs.numel()
    
    if sparsity <= 0.0:
        return torch.zeros_like(R), {"sparsity": 0.0, "cutoff": 0.0}
    
    if delta is None:
        if total_elements > 100_000:
            sub = flat_abs[::50]
            cutoff = float(torch.quantile(sub, 1.0 - sparsity).item())
        else:
            k = max(1, int(sparsity * total_elements))
            cutoff = float(torch.kthvalue(flat_abs, total_elements - k + 1).values.item())
    else:
        cutoff = delta

    if method == "hard":
        S = torch.where(torch.abs(R) >= cutoff, R, torch.zeros_like(R))
    elif method == "soft":
        S = torch.sign(R) * torch.clamp(torch.abs(R) - cutoff, min=0.0)
    elif method == "adaptive":
        row_norm = torch.sqrt(torch.norm(R, dim=1, keepdim=True) / np.sqrt(R.shape[1]) + 1e-12)
        col_norm = torch.sqrt(torch.norm(R, dim=0, keepdim=True) / np.sqrt(R.shape[0]) + 1e-12)
        adaptive_cutoff = cutoff * (row_norm * col_norm)
        S = torch.where(torch.abs(R) >= adaptive_cutoff, R, torch.zeros_like(R))
    else:
        raise ValueError(f"Unknown thresholding method: {method}")

    actual_sparsity = float(torch.count_nonzero(S).item() / total_elements)
    return S, {"sparsity": actual_sparsity, "cutoff": cutoff}


def compute_compression_size_mb(
    m: int,
    n: int,
    decomp_type: str,
    rank: int = 8,
    m1: int = 64,
    n1: int = 64,
    num_sparse_entries: int = 0,
) -> float:
    """Calculates exact serialized footprint in Megabytes (FP16 weights + indices)."""
    bytes_per_fp16 = 2
    bytes_per_index = 2  # 16-bit CSR row/col index
    
    if decomp_type == "lora":
        total_bytes = (m * rank + rank * n) * bytes_per_fp16
    elif decomp_type == "pure_kronecker":
        m2, n2 = m // m1, n // n1
        total_bytes = (m1 * n1 + m2 * n2) * bytes_per_fp16
    elif decomp_type == "svd_truncated":
        total_bytes = (m * rank + rank * n) * bytes_per_fp16
    elif decomp_type == "poet_kronecker":
        m2, n2 = m // m1, n // n1
        base_bytes = (m1 * n1 + m2 * n2) * bytes_per_fp16
        sparse_bytes = num_sparse_entries * (bytes_per_fp16 + 2 * bytes_per_index)
        total_bytes = base_bytes + sparse_bytes
    elif decomp_type == "poet_svd":
        base_bytes = (m * rank + rank * n) * bytes_per_fp16
        sparse_bytes = num_sparse_entries * (bytes_per_fp16 + 2 * bytes_per_index)
        total_bytes = base_bytes + sparse_bytes
    else:
        total_bytes = (m * n) * bytes_per_fp16

    return total_bytes / (1024 * 1024)


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------

def evaluate_layer_fast(
    A: torch.Tensor,
    B: torch.Tensor,
    alpha: float,
    sparsity_grid: list[float],
) -> dict[str, Any]:
    """Evaluates all POET decompositions for a single layer."""
    m, r = B.shape
    _, n = A.shape
    
    Delta_W = (alpha / float(r)) * (B @ A)
    norm_orig = float(torch.norm(Delta_W).item())
    if norm_orig < 1e-9:
        return {}

    m1, n1 = factor_dimensions(m, n)
    lora_mb = compute_compression_size_mb(m, n, "lora", rank=r)
    evals = {}
    
    # 1. Pure Kronecker
    G1, G2 = nearest_kronecker_product(Delta_W, m1, n1)
    L_kron = torch.kron(G1, G2)
    R_kron = Delta_W - L_kron
    err_pure_kron = float(torch.norm(R_kron).item() / norm_orig)
    kron_mb = compute_compression_size_mb(m, n, "pure_kronecker", m1=m1, n1=n1)
    
    evals["pure_kronecker"] = {
        "rel_err": err_pure_kron,
        "size_mb": kron_mb,
        "compression_ratio": lora_mb / kron_mb
    }
    
    # Residuals
    flat_abs_k = torch.abs(R_kron).flatten()
    total_elements = flat_abs_k.numel()
    sub_k = flat_abs_k[::max(1, total_elements // 10000)]
    flat_R2_k = flat_abs_k ** 2
    sum_total_R2_k = float(torch.sum(flat_R2_k).item())
    
    row_norm_k = torch.sqrt(torch.norm(R_kron, dim=1, keepdim=True) / np.sqrt(n) + 1e-12)
    col_norm_k = torch.sqrt(torch.norm(R_kron, dim=0, keepdim=True) / np.sqrt(m) + 1e-12)
    R_norm_k = torch.abs(R_kron) / (row_norm_k * col_norm_k)
    flat_R_norm_k = R_norm_k.flatten()
    sub_norm_k = flat_R_norm_k[::max(1, total_elements // 10000)]

    for sp in sparsity_grid:
        cutoff_hard = float(torch.quantile(sub_k, 1.0 - sp).item())
        mask_keep_hard = (flat_abs_k >= cutoff_hard)
        num_nz_hard = int(torch.sum(mask_keep_hard).item())
        err_hard = float(np.sqrt(max(0.0, sum_total_R2_k - float(torch.sum(flat_R2_k[mask_keep_hard]).item()))) / norm_orig)
        mb_hard = compute_compression_size_mb(m, n, "poet_kronecker", m1=m1, n1=n1, num_sparse_entries=num_nz_hard)
        
        k_hard = f"poet_kron_hard_sp{int(sp*100):02d}"
        evals[k_hard] = {
            "rel_err": err_hard,
            "actual_sparsity": float(num_nz_hard / total_elements),
            "size_mb": mb_hard,
            "compression_ratio": lora_mb / max(1e-6, mb_hard)
        }
        
        cutoff_soft = cutoff_hard
        err_soft = float(torch.sqrt(torch.sum(torch.clamp(flat_R2_k, max=cutoff_soft**2))).item() / norm_orig)
        mb_soft = mb_hard
        k_soft = f"poet_kron_soft_sp{int(sp*100):02d}"
        evals[k_soft] = {
            "rel_err": err_soft,
            "actual_sparsity": float(num_nz_hard / total_elements),
            "size_mb": mb_soft,
            "compression_ratio": lora_mb / max(1e-6, mb_soft)
        }

        cutoff_adapt = float(torch.quantile(sub_norm_k, 1.0 - sp).item())
        mask_keep_adapt = (flat_R_norm_k >= cutoff_adapt)
        num_nz_adapt = int(torch.sum(mask_keep_adapt).item())
        err_adapt = float(np.sqrt(max(0.0, sum_total_R2_k - float(torch.sum(flat_R2_k[mask_keep_adapt]).item()))) / norm_orig)
        mb_adapt = compute_compression_size_mb(m, n, "poet_kronecker", m1=m1, n1=n1, num_sparse_entries=num_nz_adapt)
        k_adapt = f"poet_kron_adaptive_sp{int(sp*100):02d}"
        evals[k_adapt] = {
            "rel_err": err_adapt,
            "actual_sparsity": float(num_nz_adapt / total_elements),
            "size_mb": mb_adapt,
            "compression_ratio": lora_mb / max(1e-6, mb_adapt)
        }

    # 3. Low-Rank SVD (via fast QR)
    Qb, Rb = torch.linalg.qr(B)
    Qa, Ra = torch.linalg.qr(A.T)
    M_small = Rb @ Ra.T
    U_s, S_s, Vh_s = torch.linalg.svd(M_small)
    U_svd = Qb @ U_s
    S_svd = (alpha / float(r)) * S_s
    Vh_svd = Vh_s @ Qa.T

    for r_k in [1, 2, 4]:
        L_svd = U_svd[:, :r_k] @ torch.diag(S_svd[:r_k]) @ Vh_svd[:r_k, :]
        R_svd = Delta_W - L_svd
        err_svd = float(torch.norm(R_svd).item() / norm_orig)
        svd_mb = compute_compression_size_mb(m, n, "svd_truncated", rank=r_k)
        evals[f"svd_rank{r_k}"] = {"rel_err": err_svd, "size_mb": svd_mb}
        
        flat_abs_svd = torch.abs(R_svd).flatten()
        sub_svd = flat_abs_svd[::max(1, total_elements // 10000)]
        cutoff_svd = float(torch.quantile(sub_svd, 0.98).item())
        mask_svd = (flat_abs_svd >= cutoff_svd)
        num_nz_svd = int(torch.sum(mask_svd).item())
        sum_R2_svd = float(torch.sum(flat_abs_svd**2).item())
        err_svd_poet = float(np.sqrt(max(0.0, sum_R2_svd - float(torch.sum((flat_abs_svd**2)[mask_svd]).item()))) / norm_orig)
        svd_poet_mb = compute_compression_size_mb(m, n, "poet_svd", rank=r_k, num_sparse_entries=num_nz_svd)
        evals[f"poet_svd_r{r_k}_sp02"] = {"rel_err": err_svd_poet, "size_mb": svd_poet_mb}

    return {
        "shape": [m, n],
        "norm": norm_orig,
        "lora_mb": lora_mb,
        "evals": evals
    }


def benchmark_poet_on_adapter(adapter_dir: Path) -> dict[str, Any]:
    """Runs POET decomposition across all layers of a single adapter."""
    weights_path = adapter_dir / "adapter_model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    weights = load_file(str(weights_path))
    
    layer_modules: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for k, tensor in weights.items():
        if "lora_A" in k:
            base_key = k.replace(".lora_A.weight", "")
            layer_modules[base_key]["A"] = tensor.float()
        elif "lora_B" in k:
            base_key = k.replace(".lora_B.weight", "")
            layer_modules[base_key]["B"] = tensor.float()

    sparsity_grid = [0.01, 0.02, 0.05, 0.10]
    total_lora_mb = 0.0
    agg_errors: dict[str, list[float]] = defaultdict(list)
    agg_sizes: dict[str, list[float]] = defaultdict(list)
    results_by_layer = []

    module_keys = sorted(layer_modules.keys())
    for mod_idx, mod_key in enumerate(module_keys):
        A = layer_modules[mod_key]["A"]
        B = layer_modules[mod_key]["B"]
        
        res = evaluate_layer_fast(A, B, CANON.LORA_ALPHA, sparsity_grid)
        if not res:
            continue
            
        total_lora_mb += res["lora_mb"]
        results_by_layer.append({
            "module": mod_key,
            "shape": res["shape"],
            "norm": res["norm"],
            "lora_mb": res["lora_mb"],
            "poet_evals": res["evals"]
        })
        
        for method_name, m_data in res["evals"].items():
            agg_errors[method_name].append(m_data["rel_err"])
            agg_sizes[method_name].append(m_data["size_mb"])

    summary_methods = {}
    for method_name, errors in agg_errors.items():
        mean_err = float(np.mean(errors))
        p95_err = float(np.percentile(errors, 95))
        total_mb = float(np.sum(agg_sizes[method_name]))
        summary_methods[method_name] = {
            "mean_rel_err": mean_err,
            "p95_rel_err": p95_err,
            "total_size_mb": total_mb,
            "compression_ratio": total_lora_mb / max(1e-6, total_mb)
        }

    return {
        "adapter_name": adapter_dir.name,
        "total_lora_mb": total_lora_mb,
        "num_layers_evaluated": len(results_by_layer),
        "summary": summary_methods,
        "layers": results_by_layer
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="results/benchmarks/poet_decomposition_benchmark.json",
                        help="Output path for benchmark JSON artifact.")
    parser.add_argument("--domains", nargs="+", default=["astral", "postgresql", "duckdb", "financial"],
                        help="Domains to evaluate.")
    args = parser.parse_args()

    t_start = time.time()
    all_results = {}
    
    print("=" * 80, flush=True)
    print(" POET (LOW-RANK + SPARSE) ADAPTER DECOMPOSITION BENCHMARK (CPU-ONLY)", flush=True)
    print(f" Target Version: v4 | Goldilocks alpha={CANON.LORA_ALPHA}, rank={CANON.LORA_RANK}", flush=True)
    print("=" * 80, flush=True)

    for domain in args.domains:
        try:
            path = adapter_path(domain, version="v4")
        except Exception as e:
            print(f"  [skip] {domain}: {e}", flush=True)
            continue

        print(f"\nEvaluating Domain: {domain.upper()} ({path.name})...", flush=True)
        t0 = time.time()
        res = benchmark_poet_on_adapter(path)
        all_results[domain] = res
        print(f"  Done in {time.time()-t0:.2f}s | {res['num_layers_evaluated']} modules | Dense LoRA: {res['total_lora_mb']:.1f} MB", flush=True)
        
        print("\n  " + "-" * 74, flush=True)
        print(f"  {'Decomposition Method':<28} | {'Mean Rel Err':<12} | {'Total Size':<10} | {'Compression'}", flush=True)
        print("  " + "-" * 74, flush=True)
        
        key_showcase = [
            ("pure_kronecker", "Pure Kronecker (r=1)"),
            ("poet_kron_hard_sp01", "POET Kron + 1% Sparse"),
            ("poet_kron_hard_sp02", "POET Kron + 2% Sparse"),
            ("poet_kron_hard_sp05", "POET Kron + 5% Sparse"),
            ("poet_kron_adaptive_sp05", "POET Kron + 5% Adaptive"),
            ("svd_rank2", "SVD Truncated (Rank 2)"),
            ("poet_svd_r2_sp02", "POET SVD (Rank 2 + 2% Sp)"),
            ("svd_rank4", "SVD Truncated (Rank 4)"),
        ]
        
        for k_id, label in key_showcase:
            if k_id in res["summary"]:
                s = res["summary"][k_id]
                print(f"  {label:<28} | {s['mean_rel_err']:>10.2%}   | {s['total_size_mb']:>7.1f} MB | {s['compression_ratio']:>6.1f}x", flush=True)
        print("  " + "-" * 74, flush=True)

    elapsed = time.time() - t_start
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    final_artifact = {
        "metadata": CANON.stamp(),
        "elapsed_seconds": elapsed,
        "domains": all_results
    }
    
    out_path.write_text(json.dumps(final_artifact, indent=2))
    print(f"\n[Artifact Saved] -> {out_path} (Total Elapsed: {elapsed:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
