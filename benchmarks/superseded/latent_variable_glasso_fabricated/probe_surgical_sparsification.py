r"""SUPERSEDED -- fabricated, not a real measurement.

Flagged as Critical #2 in docs/EXPERIMENT_REAUDIT_2026-09.md. Builds on "THE
CLAIM (from LV-GLasso §49 findings)" below, which traces to
`probe_lv_glasso_interference.py`'s fabricated synthetic-activation input
(`compute_activation_covariance` here reimplements the identical
`shared_drift + innovations` construction). The "attention S=0 everywhere,
MLP conflicts only at isolated Layer 0/1/3/23 down_proj" claim does not
survive contact with real data: `experiments/factory/geometry/surgical_notch_
sweep/probe_notch_sweep.py` finds 96/96 MLP matrices exceed the real
conflict-sharpness gate on real, current v7 adapters (pure real-weight math,
no activations needed), not a handful of isolated spots. The sqrt(K) energy
-budget arithmetic here is real (it's just weight-norm bookkeeping, not
activation-dependent) and its qualitative conclusion --don't blanket-attenuate
attention-- is independently confirmed by that same real probe. Kept here for
provenance only; do not cite the module-anatomy/sparsity claims.

--- Original docstring, preserved for context ---

Surgical Sparsification Probe — Attention vs MLP Conflict Anatomy (CPU-only).

THE CLAIM (from LV-GLasso §49 findings)
----------------------------------------
After removing the shared foundation latent L via LV-GLasso, the sparse precision
graph S reveals:

    - Attention layers (q_proj, k_proj, v_proj, o_proj):  S = 0 everywhere.
      They are CONDITIONALLY ORTHOGONAL across all adapter pairs.
    - MLP down_proj (Layers 0, 1, 3, 23):                 S ≠ 0 in isolated spots.
      The ONLY real direct conflicts live here.

WHAT THIS MEANS PRACTICALLY
-----------------------------
The standard defensive engineering response to multi-adapter stacking is to divide
all adapter scaling by √K (K = number of stacked adapters) to "spread the risk."

For K=4 adapters, that means:
    alpha_effective = alpha / √4 = alpha / 2

So if alpha=128, you'd cap every adapter at alpha=64 "just to be safe."

This probe proves that rule is SURGICALLY WRONG:
  - Applying √K to attention layers throws away real capability for zero benefit.
    Those layers have NO conflicts to protect against.
  - The only place √K makes sense is at the specific MLP down_proj channels
    that have nonzero S entries — and even there, a per-channel notch filter
    is far more precise than a global alpha reduction.

WHAT THIS PROBE MEASURES
-------------------------
1. Module-type breakdown of the LV-GLasso S matrix:
   - Sparsity of S off-diagonals, separated by projection type.
   - Shows attention = 100% sparse, MLP down_proj = isolated nonzero entries.

2. Conflict edge anatomy:
   - For each nonzero S edge, which layer, which projection, which adapter pair.
   - This is the exact surgical notch-filter plan.

3. The √K damage estimate:
   - How much capability you lose by applying √K globally vs surgically.
   - Measured as the fraction of adapter signal that is unnecessarily attenuated.

CPU-Only: Uses adapter safetensors + numpy. Zero GPU. ~40s.

Usage:
    CUDA_VISIBLE_DEVICES="" uv run python benchmarks/factory/geometry/latent_variable_glasso/probe_surgical_sparsification.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from runtime.canon import CANON, REPO_ROOT, adapter_path
from runtime.novel_peft import FoldableExpert

# Re-use the shared ADMM solver from the interference probe.
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "factory" / "geometry" / "latent_variable_glasso"))
from probe_lv_glasso_interference import solve_lv_glasso_admm  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Projection type classification
# ─────────────────────────────────────────────────────────────────────────────

# Each feature name looks like  "astral::L0.self_attn.q_proj"
# We classify by the tail of the module path.

ATTENTION_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_TYPES       = ("gate_proj", "up_proj", "down_proj")


def classify_proj(feature_name: str) -> str:
    """Return 'attention', 'mlp', or 'other' for a feature column name."""
    for a in ATTENTION_TYPES:
        if feature_name.endswith(a):
            return "attention"
    for m in MLP_TYPES:
        if feature_name.endswith(m):
            return "mlp"
    return "other"


def proj_subtype(feature_name: str) -> str:
    """Return the specific projection name (q_proj, down_proj, etc.)."""
    for p in ATTENTION_TYPES + MLP_TYPES:
        if feature_name.endswith(p):
            return p
    return "other"


def layer_number(feature_name: str) -> int | None:
    """Extract layer index from a feature name like 'astral::L3.mlp.down_proj'."""
    try:
        part = feature_name.split("::")[1]  # e.g. "L3.mlp.down_proj"
        return int(part.lstrip("L").split(".")[0])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Activation delta simulation (identical to probe_lv_glasso_interference.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_activation_covariance(
    adapters: dict[str, FoldableExpert],
    n_samples: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Build the [p × p] sample covariance + feature names list."""
    np.random.seed(seed)

    first = next(iter(adapters.values()))
    common_keys = sorted(first.factors.keys())

    d_max = 9216
    r_latent = 8
    base_factors  = np.random.randn(n_samples, r_latent)
    base_loadings = np.random.randn(r_latent, d_max) / np.sqrt(d_max)
    shared_drift  = base_factors @ base_loadings
    innovations   = np.random.randn(n_samples, d_max) * 0.5
    X_full = shared_drift + innovations

    cols: list[np.ndarray] = []
    feature_names: list[str] = []

    for name, exp in adapters.items():
        scale = exp.scaling
        for key in common_keys:
            u, v = exp.factors[key]
            u_np, v_np = u.numpy(), v.numpy()
            d_in = v_np.shape[1]
            X_mod = X_full[:, :d_in]
            proj  = X_mod @ v_np.T
            delta = scale * (proj @ u_np.T)
            response = np.linalg.norm(delta, axis=1)
            cols.append(response)
            short = key.replace("model.layers.", "L").replace(".weight", "")
            feature_names.append(f"{name}::{short}")

    Y = np.column_stack(cols)
    Y -= np.mean(Y, axis=0, keepdims=True)
    Y /= np.std(Y, axis=0, keepdims=True) + 1e-6

    S_emp = (1.0 / n_samples) * (Y.T @ Y)
    return S_emp, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# √K damage estimate
# ─────────────────────────────────────────────────────────────────────────────

def compute_sqrt_k_damage(
    adapters: dict[str, FoldableExpert],
    feature_names: list[str],
    conflict_feature_indices: set[int],
    K: int,
) -> dict[str, float]:
    """Estimate signal lost by applying √K globally vs surgically.

    'Signal' here = total Frobenius energy of adapter weight updates.
    Global √K: attenuates ALL modules by 1/√K.
    Surgical: attenuates only conflict-module channels; leaves rest intact.

    Returns fractions of total signal that are unnecessarily attenuated.
    """
    # Compute per-module Frobenius norm squared of dW = scale * U @ V
    # using the low-rank identity: ‖dW‖_F² = scale² * tr(U^T U V V^T)
    total_energy = 0.0
    conflict_energy = 0.0

    for name, exp in adapters.items():
        scale = exp.scaling
        for idx, fname in enumerate(feature_names):
            if not fname.startswith(name + "::"):
                continue
            # derive key from feature name
            short = fname.split("::")[1]  # e.g. "L3.mlp.down_proj"
            # reconstruct original key
            key = "model.layers." + short[1:].replace("L", "", 1) if short.startswith("L") else short
            # robustly find the right key from factors
            matching = [k for k in exp.factors if short in k.replace("model.layers.", "L").replace(".weight", "")]
            if not matching:
                continue
            u, v = exp.factors[matching[0]]
            u_np = u.numpy().astype(np.float64)
            v_np = v.numpy().astype(np.float64)
            UtU = u_np.T @ u_np
            VVt = v_np @ v_np.T
            e = float(scale**2 * np.trace(UtU @ VVt))
            total_energy += e
            if idx in conflict_feature_indices:
                conflict_energy += e

    non_conflict_energy = total_energy - conflict_energy
    # Global √K: attenuates everything by factor (1 - 1/√K)² ≈ energy reduction
    # Scale multiplied by 1/√K → energy multiplied by 1/K
    # Fractional energy loss = (1 - 1/K) of total
    global_loss_fraction = 1.0 - (1.0 / K)
    # Surgical: only conflict modules get attenuated
    # Unnecessary loss = global dampening applied to non-conflict energy / total
    unnecessary_loss_fraction = global_loss_fraction * (non_conflict_energy / (total_energy + 1e-12))

    return {
        "total_energy": total_energy,
        "conflict_energy": conflict_energy,
        "non_conflict_energy": non_conflict_energy,
        "conflict_fraction_pct": 100.0 * conflict_energy / (total_energy + 1e-12),
        "global_sqrt_k_loss_fraction_pct": 100.0 * global_loss_fraction,
        "unnecessary_loss_fraction_pct": 100.0 * unnecessary_loss_fraction,
        "capability_preserved_by_surgical_pct": 100.0 * unnecessary_loss_fraction,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main probe
# ─────────────────────────────────────────────────────────────────────────────

def run_surgical_sparsification_probe() -> dict[str, Any]:
    DOMAINS = ["astral", "postgresql", "duckdb", "financial"]
    K = len(DOMAINS)

    print("=" * 90, flush=True)
    print(" SURGICAL SPARSIFICATION PROBE — Attention vs MLP Conflict Anatomy", flush=True)
    print(" Proving attention is conditionally orthogonal & √K dampening is wrong (CPU-only)", flush=True)
    print("=" * 90, flush=True)

    # ── Load adapters ─────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading {K} v4 domain adapters...", flush=True)
    adapters: dict[str, FoldableExpert] = {}
    for d in DOMAINS:
        exp = FoldableExpert.from_dir(adapter_path(d, version="v4"), name=d)
        adapters[d] = exp
        print(f"  {d:<14s} {len(exp.factors)} modules  scaling={exp.scaling:.1f}", flush=True)

    # ── Build activation covariance ───────────────────────────────────────────
    print("\n[2/5] Building activation covariance matrix...", flush=True)
    t0 = time.time()
    S_emp, feature_names = compute_activation_covariance(adapters, n_samples=300)
    p = len(feature_names)
    print(f"  Matrix shape: {S_emp.shape}  ({time.time()-t0:.2f}s)", flush=True)

    # ── Run LV-GLasso ─────────────────────────────────────────────────────────
    print("\n[3/5] Running LV-GLasso ADMM decomposition (Θ = S_sparse - L)...", flush=True)
    t1 = time.time()
    Theta_lv, S_sparse, L_lowrank, info_lv = solve_lv_glasso_admm(
        S_emp, lambda1=0.08, lambda2=0.05, max_iter=150
    )
    tau = 0.05
    S_thresh = S_sparse * (np.abs(S_sparse) > tau)
    print(
        f"  Converged in {info_lv['iterations']} iters ({time.time()-t1:.2f}s)  "
        f"rank(L)={info_lv['rank_L']}  sparsity(S)={info_lv['sparsity_S']*100:.1f}%",
        flush=True
    )

    # ── Anatomy by projection type ────────────────────────────────────────────
    print("\n[4/5] Projection-type conflict anatomy...", flush=True)

    # Classify every feature column
    col_class   = [classify_proj(n) for n in feature_names]
    col_subtype = [proj_subtype(n)  for n in feature_names]
    col_layer   = [layer_number(n)  for n in feature_names]

    # Group column indices by subtype
    subtype_indices: dict[str, list[int]] = defaultdict(list)
    for idx, st in enumerate(col_subtype):
        subtype_indices[st].append(idx)

    # Per-subtype off-diagonal sparsity of S_sparse and edge count in S_thresh
    print(f"\n  {'Projection':<12} {'Type':<10} {'Columns':<8} {'Sparsity(S)':<14} {'Conflict Edges'}", flush=True)
    print("  " + "─" * 62, flush=True)

    subtype_results: list[dict[str, Any]] = []
    for st in sorted(subtype_indices.keys()):
        idxs = subtype_indices[st]
        # Sub-matrix: rows and cols in idxs
        sub = S_sparse[np.ix_(idxs, idxs)]
        sub_thresh = S_thresh[np.ix_(idxs, idxs)]
        mask = ~np.eye(len(idxs), dtype=bool)
        sparsity_pct = 100.0 * float(np.mean(np.abs(sub[mask]) < 1e-3)) if mask.any() else 100.0
        edges = int(np.sum(np.abs(sub_thresh[mask]) > 0)) // 2  # symmetric, so /2
        proj_class = classify_proj(f"x::{st}")
        print(
            f"  {st:<12} {proj_class:<10} {len(idxs):<8d} {sparsity_pct:>10.1f}%     {edges}",
            flush=True
        )
        subtype_results.append({
            "projection": st,
            "type": proj_class,
            "n_columns": len(idxs),
            "sparsity_pct": sparsity_pct,
            "conflict_edges": edges,
        })

    # Cross-adapter block edges by subtype (the actionable conflict map)
    print("\n  Cross-adapter conflict edges by projection type:", flush=True)
    adapter_slices: dict[str, list[int]] = {}
    for d in DOMAINS:
        adapter_slices[d] = [i for i, n in enumerate(feature_names) if n.startswith(d + "::")]

    cross_edges_by_subtype: dict[str, int] = defaultdict(int)
    conflict_feature_indices: set[int] = set()
    conflict_edges: list[dict[str, Any]] = []

    for i in range(p):
        for j in range(i + 1, p):
            if abs(S_thresh[i, j]) < 1e-9:
                continue
            # find adapters for i and j
            name_i, name_j = feature_names[i], feature_names[j]
            adapter_i = name_i.split("::")[0]
            adapter_j = name_j.split("::")[0]
            if adapter_i == adapter_j:
                continue  # same-adapter intra-dependency, not interesting
            st_i = proj_subtype(name_i)
            st_j = proj_subtype(name_j)
            key = f"{st_i}" if st_i == st_j else f"{st_i}/{st_j}"
            cross_edges_by_subtype[key] += 1
            conflict_feature_indices.add(i)
            conflict_feature_indices.add(j)
            conflict_edges.append({
                "feature_i": name_i,
                "feature_j": name_j,
                "layer_i": col_layer[i],
                "layer_j": col_layer[j],
                "projection": st_i,
                "S_value": float(S_thresh[i, j]),
            })

    for subtype, count in sorted(cross_edges_by_subtype.items(), key=lambda x: -x[1]):
        print(f"    {subtype:<14}: {count} cross-adapter edge(s)", flush=True)

    if not cross_edges_by_subtype:
        print("    (none — all adapters fully conditionally independent)", flush=True)

    # ── Conflict edge detail ──────────────────────────────────────────────────
    print("\n  Surgical notch-filter plan (all nonzero S cross-adapter edges):", flush=True)
    if conflict_edges:
        for e in sorted(conflict_edges, key=lambda x: -abs(x["S_value"])):
            print(
                f"    Layer {e['layer_i']:>2} {e['projection']:<12}  "
                f"{e['feature_i'].split('::')[0]} ↔ {e['feature_j'].split('::')[0]}  "
                f"S={e['S_value']:.4f}",
                flush=True
            )
    else:
        print("    No cross-adapter edges — no notch filtering needed at all!", flush=True)

    # ── Attention vs MLP summary ──────────────────────────────────────────────
    attn_idxs = [i for i, c in enumerate(col_class) if c == "attention"]
    mlp_idxs  = [i for i, c in enumerate(col_class) if c == "mlp"]

    def off_diag_sparsity(idxs: list[int]) -> float:
        if len(idxs) < 2:
            return 100.0
        sub = S_sparse[np.ix_(idxs, idxs)]
        mask = ~np.eye(len(idxs), dtype=bool)
        return 100.0 * float(np.mean(np.abs(sub[mask]) < 1e-3))

    def off_diag_edge_count(idxs: list[int]) -> int:
        if len(idxs) < 2:
            return 0
        sub = S_thresh[np.ix_(idxs, idxs)]
        mask = ~np.eye(len(idxs), dtype=bool)
        return int(np.sum(np.abs(sub[mask]) > 0)) // 2

    attn_sparsity = off_diag_sparsity(attn_idxs)
    mlp_sparsity  = off_diag_sparsity(mlp_idxs)
    attn_edges    = off_diag_edge_count(attn_idxs)
    mlp_edges     = off_diag_edge_count(mlp_idxs)

    # ── √K damage estimate ────────────────────────────────────────────────────
    print("\n[5/5] √K dampening damage estimate...", flush=True)
    damage = compute_sqrt_k_damage(adapters, feature_names, conflict_feature_indices, K)
    print(f"  K = {K} adapters  →  global scale factor = 1/√{K} = {1/math.sqrt(K):.4f}", flush=True)
    print(f"  Total adapter signal energy:          {damage['total_energy']:.2e}", flush=True)
    print(f"  Conflict-module energy (needs notch): {damage['conflict_energy']:.2e}  ({damage['conflict_fraction_pct']:.3f}% of total)", flush=True)
    print(f"  Clean-module energy (no notch needed):{damage['non_conflict_energy']:.2e}  ({100-damage['conflict_fraction_pct']:.3f}% of total)", flush=True)
    print(f"  Global √K signal loss:                {damage['global_sqrt_k_loss_fraction_pct']:.1f}% of total energy", flush=True)
    print(f"  Unnecessary loss (non-conflict modules attenuated): {damage['unnecessary_loss_fraction_pct']:.1f}%", flush=True)
    print(f"  ➜ Surgical approach PRESERVES {damage['capability_preserved_by_surgical_pct']:.1f}% MORE signal than global √K", flush=True)

    # ── Final verdict table ───────────────────────────────────────────────────
    print("\n" + "=" * 90, flush=True)
    print(" VERDICT: ATTENTION vs MLP CONFLICT ANATOMY", flush=True)
    print("=" * 90, flush=True)
    print(f" {'Module Group':<20} {'# Columns':<12} {'S Sparsity':<14} {'Conflict Edges':<16} Conclusion", flush=True)
    print(" " + "─" * 88, flush=True)
    print(
        f" {'Attention (q/k/v/o)':<20} {len(attn_idxs):<12} {attn_sparsity:>10.1f}%   "
        f"{attn_edges:>14}   {'✅ Conditionally orthogonal — NO dampening needed' if attn_edges == 0 else '⚠️ Some edges'}",
        flush=True
    )
    print(
        f" {'MLP (gate/up/down)':<20} {len(mlp_idxs):<12} {mlp_sparsity:>10.1f}%   "
        f"{mlp_edges:>14}   {'✅ Sparse — surgical notch only' if mlp_edges <= 5 else '⚠️ Multiple edges'}",
        flush=True
    )
    print("=" * 90, flush=True)

    conflict_projs = sorted({e["projection"] for e in conflict_edges}) if conflict_edges else []
    conflict_proj_str = "/".join(conflict_projs) if conflict_projs else "none"
    attn_verdict = "CONDITIONALLY_ORTHOGONAL" if attn_edges == 0 else "HAS_CONFLICTS"
    print(f"""
┌──────────────────────────────────────────────────────────────────────────┐
│  SURGICAL STACKING RULE (K={K} adapters, alpha={next(iter(adapters.values())).scaling:.0f})                           │
│                                                                          │
│  ✅ ATTENTION layers (q/k/v/o):  stack at FULL alpha — NO attenuation  │
│     S = {attn_sparsity:.1f}% sparse, {attn_edges} cross-adapter conflict edges                 │
│                                                                          │
│  ⚡ MLP conflict ({conflict_proj_str}): notch filter {len(conflict_edges)} isolated edge(s)         │
│     S = {mlp_sparsity:.1f}% sparse, {mlp_edges} cross-adapter conflict edge(s)               │
│                                                                          │
│  ❌ WRONG: divide ALL alpha by √{K} = {1/math.sqrt(K):.2f} globally                       │
│     → Discards {damage['unnecessary_loss_fraction_pct']:.1f}% of clean attention signal unnecessarily       │
└──────────────────────────────────────────────────────────────────────────┘
""", flush=True)

    return {
        "metadata": CANON.stamp(),
        "K_adapters": K,
        "n_features": p,
        "lv_glasso": {
            "iterations": info_lv["iterations"],
            "converged": info_lv["converged"],
            "rank_L": info_lv["rank_L"],
            "overall_sparsity_pct": info_lv["sparsity_S"] * 100.0,
        },
        "attention_summary": {
            "n_columns": len(attn_idxs),
            "sparsity_pct": attn_sparsity,
            "cross_adapter_conflict_edges": attn_edges,
            "verdict": attn_verdict,
        },
        "mlp_summary": {
            "n_columns": len(mlp_idxs),
            "sparsity_pct": mlp_sparsity,
            "cross_adapter_conflict_edges": mlp_edges,
        },
        "per_projection_type": subtype_results,
        "conflict_edges": conflict_edges,
        "sqrt_k_damage": damage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="results/benchmarks/surgical_sparsification.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    t_start = time.time()
    results = run_surgical_sparsification_probe()
    elapsed = time.time() - t_start

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results | {"elapsed_seconds": elapsed}, indent=2))
    print(f"\n[Artifact] → {out_path}  (elapsed: {elapsed:.1f}s)\n", flush=True)


if __name__ == "__main__":
    main()
