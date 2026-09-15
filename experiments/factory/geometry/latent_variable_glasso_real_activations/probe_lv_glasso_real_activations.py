r"""Real forward-pass Latent Variable Graphical Lasso (LV-GLasso) re-analysis.

WHY THIS EXISTS
---------------
`experiments/factory/geometry/latent_variable_glasso/` (retired to
`benchmarks/superseded/latent_variable_glasso_fabricated/`, Critical #2 in
docs/EXPERIMENT_REAUDIT_2026-09.md) built its precision-graph analysis on a
SYNTHETIC input activation matrix (`torch.randn` shared-drift + noise)
multiplied by real trained LoRA weights, and reported the result as measuring
real adapter interference. `experiments/factory/geometry/surgical_notch_sweep/`
answered the practical question for real using pure weight-space math (no
activations needed at all) and found 96/96 MLP modules conflict, not 1/512 --
but it never actually tests what real per-token forward-pass activation
covariance looks like, which was the ORIGINAL (fabricated) cluster's whole
premise. This script closes that gap for real: same question, same reusable
ADMM machinery, real activations this time.

METHOD -- real forward pass, real hooks, no synthetic data anywhere
---------------------------------------------------------------------
Follows the same real-hook methodology already validated in
`experiments/factory/geometry/activation_inertia_probe/probe_stacking_merit.py`:
load the REAL base model once (no PEFT wrapper -- attaching an adapter would
change h, and every adapter must see the SAME h for this to be well-defined),
register real forward hooks capturing the real per-token INPUT activation at
every LoRA-targeted module common to all 4 real v7 adapters, run real forward
passes over real domain-representative prompts (reused verbatim from
`probe_stacking_merit.py`, this repo's own established real prompt set), and
for each captured real h compute each adapter's real
`delta = scale * (h @ A.T) @ B.T` -- exactly the same math the retired script
used, only the input is now a real captured hidden state, never
`torch.randn(...)`.

The per-token L2 norm of each (adapter, module) delta becomes one feature
column, exactly like the retired script -- the only change is where the rows
(samples) come from: real tokens from real prompts through a real forward
pass, not synthetic noise.

`solve_lv_glasso_admm` / `soft_threshold` below are copied VERBATIM from
`benchmarks/superseded/latent_variable_glasso_fabricated/probe_lv_glasso_interference.py`
-- that machinery was always real, generic numerical linear algebra; only its
input was fabricated. Copied rather than imported so this script has no
dependency on the superseded/ tree.

Usage:
    uv run --env-file .env python \
        experiments/factory/geometry/latent_variable_glasso_real_activations/probe_lv_glasso_real_activations.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from runtime_common.canon import REPO_ROOT, adapter_path
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Real prompts (reused verbatim from probe_stacking_merit.py -- this repo's
# own established real-hook methodology, including the deliberate postgresql/
# duckdb collision prompt).
# ---------------------------------------------------------------------------
PROMPTS = {
    "astral": [
        "How do I configure workspace dependencies and lockfiles using uv and pyproject.toml in Python 3.12?",
        "Write an async coroutine using asyncio.TaskGroup and type hints with ruff formatting rules.",
        "Demonstrate how to run isolated Python scripts with inline script metadata PEP 723 using uv run.",
        "Replace my pip + requirements.txt workflow with uv, and format the project with ruff.",
    ],
    "postgresql": [
        "How do I create an HNSW vector index in pgvector with cosine distance ops and query nearest neighbors?",
        "Write a query using DISTINCT ON with window functions and explain the plan with EXPLAIN ANALYZE.",
        "Demonstrate how to partition a PostgreSQL 17 table by range with BRIN indexing on timestamp columns.",
        "Return the top 3 products per category by revenue.",
    ],
    "duckdb": [
        "How do I query partitioned Parquet files from S3 with projection pushdown and hive_partitioning in DuckDB?",
        "Write a DuckDB query using FROM-first syntax, COLUMNS(*) regex aggregation, and EXCLUDE.",
        "Demonstrate zero-copy export from a DuckDB query to a Polars DataFrame using PyArrow record batches.",
        "Return the top 3 products per category by revenue.",
    ],
    "financial": [
        "Explain the difference between traditional Roth IRA contributions, 401(k) rollovers, and tax brackets.",
        "Calculate the net present value of a multi-period capital investment with inflation-adjusted cash flows.",
        "What asset allocation should a 45-year-old use for retirement in 20 years?",
        "Compare term life insurance against whole life for a family with two young children.",
    ],
}


# ---------------------------------------------------------------------------
# Real adapter loading (no PEFT wrapper -- see probe_stacking_merit.py's own
# docstring for why: every adapter must see the same base-model h).
# ---------------------------------------------------------------------------
def load_adapter(path: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], float]:
    cfg = json.loads((path / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    raw = load_file(path / "adapter_model.safetensors")

    factors: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for k, v in raw.items():
        if ".lora_A.weight" in k:
            factors[k.split(".lora_A.weight")[0]]["A"] = v
        elif ".lora_B.weight" in k:
            factors[k.split(".lora_B.weight")[0]]["B"] = v

    out = {}
    for k, ab in factors.items():
        if "A" not in ab or "B" not in ab:
            continue
        name = k
        for prefix in ("base_model.model.", "base_model."):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        out[name] = (ab["A"], ab["B"])
    return out, scale


# ---------------------------------------------------------------------------
# ADMM solver -- copied verbatim from the retired
# probe_lv_glasso_interference.py. Real, generic, correct linear algebra;
# never itself the fabricated part.
# ---------------------------------------------------------------------------
def soft_threshold(X: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(X) * np.maximum(np.abs(X) - lam, 0.0)


def solve_lv_glasso_admm(
    S: np.ndarray,
    lambda1: float = 0.05,
    lambda2: float = 0.10,
    rho: float = 1.0,
    max_iter: int = 150,
    tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Solves Latent Variable Graphical Lasso via normalized ADMM (Ch 9 Sec 9.4.2).

    Problem:
      min_{S_sparse, L_lowrank} -logdet(S_sparse - L) + tr(S (S_sparse - L)) + lambda1 ||S_sparse||_{1,off} + lambda2 tr(L)
      s.t. S_sparse - L > 0, L >= 0.
    """
    p = S.shape[0]
    d_diag = np.sqrt(np.maximum(np.diag(S), 1e-8))
    inv_d = 1.0 / d_diag
    C_emp = S * np.outer(inv_d, inv_d)

    kappa = 0.02
    C_reg = 0.5 * (C_emp + C_emp.T) + kappa * np.eye(p)

    R = np.linalg.inv(C_reg)
    S_sparse = np.copy(R)
    L_lowrank = np.zeros((p, p))
    U = np.zeros((p, p))

    converged = False
    res_norm = float("inf")
    for it in range(max_iter):
        A = S_sparse - L_lowrank - U - (1.0 / rho) * C_reg
        A_sym = 0.5 * (A + A.T)
        eigvals, eigvecs = np.linalg.eigh(A_sym)
        r_eigvals = (eigvals + np.sqrt(eigvals**2 + 4.0 / rho)) / 2.0
        R_new = eigvecs @ np.diag(r_eigvals) @ eigvecs.T

        B = R_new + L_lowrank + U
        S_new = soft_threshold(B, lambda1 / rho)
        np.fill_diagonal(S_new, np.diag(B))

        C = S_new - R_new - U
        C_sym = 0.5 * (C + C.T)
        l_eigvals, l_eigvecs = np.linalg.eigh(C_sym)
        l_eigvals_thresh = np.maximum(l_eigvals - (lambda2 / rho), 0.0)
        L_new = l_eigvecs @ np.diag(l_eigvals_thresh) @ l_eigvecs.T

        residual = R_new - (S_new - L_new)
        U = U + residual

        res_norm = float(np.linalg.norm(residual, "fro")) / float(np.linalg.norm(R_new, "fro") + 1e-8)

        if res_norm < tol and it > 10:
            converged = True
            R, S_sparse, L_lowrank = R_new, S_new, L_new
            break
        R, S_sparse, L_lowrank = R_new, S_new, L_new

    D_mat = np.outer(inv_d, inv_d)
    Theta_out = R * D_mat
    S_out = S_sparse * D_mat
    L_out = L_lowrank * D_mat

    info = {
        "iterations": it + 1,
        "converged": converged,
        "final_residual": float(res_norm),
        "rank_L": int(np.sum(np.linalg.svd(L_lowrank, compute_uv=False) > 1e-3)),
        "sparsity_S": float(np.mean(np.abs(S_sparse - np.diag(np.diag(S_sparse))) < 1e-3)),
    }
    return Theta_out, S_out, L_out, info


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", default="results/benchmarks/latent_variable_glasso_real_activations.json")
    args = ap.parse_args()

    domains = ["astral", "postgresql", "duckdb", "financial"]
    print("=" * 92)
    print(" REAL FORWARD-PASS LV-GLASSO -- real activations, real weights, real ADMM decomposition")
    print("=" * 92)

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()
    dev = model.device

    adapters, scales = {}, {}
    for name in domains:
        p = adapter_path(name)
        f, sc = load_adapter(p)
        adapters[name] = {k: (a.to(dev, torch.bfloat16), b.to(dev, torch.bfloat16)) for k, (a, b) in f.items()}
        scales[name] = sc
        print(f"  loaded {name:12s} {len(f)} modules  scale={sc:.1f}", flush=True)

    common = sorted(set.intersection(*(set(a) for a in adapters.values())))
    print(f"  {len(common)} modules common to all {len(domains)} adapters -> p = {len(common) * len(domains)}\n")

    captured: dict[str, torch.Tensor] = {}
    hooks = []
    for mod_name, module in model.named_modules():
        if mod_name in common:
            def mk(n):
                def hook(m, inp, out):
                    captured[n] = inp[0].detach()
                return hook
            hooks.append(module.register_forward_hook(mk(mod_name)))

    feature_names: list[str] = [f"{dom}::{mod}" for mod in common for dom in domains]
    rows: list[np.ndarray] = []

    with torch.no_grad():
        for domain, prompts in PROMPTS.items():
            for prompt in prompts:
                text = f"### Question:\n{prompt}\n\n### Answer:\n"
                ids = tok(text, return_tensors="pt").input_ids.to(dev)
                captured.clear()
                model(ids)

                per_token_cols: list[np.ndarray] = []
                for mod_name in common:
                    h = captured.get(mod_name)
                    if h is None:
                        continue
                    hf = h.float().squeeze(0)  # (seq, in_dim)
                    for dom in domains:
                        A, B = adapters[dom][mod_name]
                        delta = (hf.to(B.dtype) @ A.t() @ B.t()).float() * scales[dom]  # (seq, out_dim)
                        col = delta.norm(dim=-1).cpu().numpy()  # (seq,)
                        per_token_cols.append(col)
                # per_token_cols: list of (seq,) arrays, one per (mod, dom) in the
                # SAME fixed order as feature_names -- stack into (seq, p)
                block = np.stack(per_token_cols, axis=1)  # (seq, p)
                rows.append(block)
            print(f"  {domain:12s} done ({len(prompts)} prompts)", flush=True)

    for h in hooks:
        h.remove()

    Y = np.concatenate(rows, axis=0)  # (n_tokens_total, p)
    n_samples, p = Y.shape
    print(f"\n  Real activation-response matrix: {n_samples} real tokens x {p} features")

    Y_centered = Y - Y.mean(axis=0, keepdims=True)
    Y_std = Y_centered / (Y_centered.std(axis=0, keepdims=True) + 1e-6)
    S_emp = (1.0 / n_samples) * (Y_std.T @ Y_std)

    print("  Running real LV-GLasso ADMM decomposition...", flush=True)
    Theta, S_sparse, L_lowrank, info = solve_lv_glasso_admm(S_emp)
    print(f"  converged={info['converged']}  iterations={info['iterations']}  "
          f"rank(L)={info['rank_L']}  sparsity(S)={info['sparsity_S']:.4f}")

    off_diag_mask = ~np.eye(p, dtype=bool)
    nonzero_edges = []
    for i, j in zip(*np.where((np.abs(S_sparse) > 1e-3) & off_diag_mask), strict=True):
        if i < j:
            nonzero_edges.append((feature_names[i], feature_names[j], float(S_sparse[i, j])))
    nonzero_edges.sort(key=lambda e: -abs(e[2]))

    print(f"\n  {len(nonzero_edges)} nonzero off-diagonal S edges out of {p * (p - 1) // 2} possible pairs")
    print("  Top 15 by |S|:")
    for a, b, v in nonzero_edges[:15]:
        print(f"    {a:40s} <-> {b:40s}  S={v:+.5f}")

    print("\n" + "=" * 92)
    print("  COMPARISON TO PRIOR CLAIMS")
    print("=" * 92)
    print("  Fabricated (retired) claim: rank(L)=299, ~1 real conflict edge, 99.998% S sparsity")
    print("  surgical_notch_sweep (real weight-space): 96/96 MLP modules exceed conflict gate")
    print(f"  THIS (real activations):    rank(L)={info['rank_L']}, "
          f"{len(nonzero_edges)} nonzero edges, {info['sparsity_S']:.4%} S sparsity")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": args.model_name,
        "domains": domains,
        "n_common_modules": len(common),
        "n_features_p": p,
        "n_real_token_samples": n_samples,
        "admm_info": info,
        "n_nonzero_edges": len(nonzero_edges),
        "top_edges": [{"a": a, "b": b, "S": v} for a, b, v in nonzero_edges[:50]],
    }, indent=2))
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
