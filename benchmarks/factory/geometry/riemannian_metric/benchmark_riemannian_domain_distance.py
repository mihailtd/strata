r"""Geodesic distance between domain experts on the SPD manifold -- Ch.3 §3.5, Ch.8 §8.1.4.

WHAT THIS REPLACES
------------------
The first version of this benchmark reduced each adapter to G = s^2 VV^T + U^T U
in its OWN rank-8 basis and fed pairs of those to d_R. It produced a 6x6 matrix
whose entire off-diagonal spread was 0.2600 .. 0.2837 -- every pair within 9% of
every other -- and it was written up as CONFIRMED in DECISIONS.md §55.

That matrix was noise. Rotating one adapter's rank basis (U -> U R, V -> R^T V)
leaves dW bit-identical -- the adapter is the same object -- yet swung the reported
d_R over 0.2768 .. 0.3305 on the astral/postgres pair alone. The basis artefact was
2.3x the whole reported signal. d_LE agreeing with d_R to four decimals was the
tell: that happens when both matrices are dominated by the same regulariser.

THE FIX
-------
Compare the two adapters INSIDE ONE SHARED BASIS. For dW = s U V the output-side
operator Sigma = dW dW^T is a function of the adapter alone. Project both into an
orthonormal basis Q of span(range dW_a U range dW_b), k <= 2r dimensions, then
measure. AIRM is congruence-invariant so Q's arbitrary orientation cancels; the
relative ORIENTATION of the two subspaces -- the thing that decides whether two
experts stack -- is now what the number responds to.

Guarded at startup by a self-test that ABORTS on a rank-basis rotation moving
d_R, so this cannot silently regress.

WHAT IS AND IS NOT ESTABLISHED
------------------------------
delta is chosen, not estimated: a weight Gramian has no sample count, so the
Ledoit-Wolf optimum is not identifiable here (see riemannian_covariance.py). The
benchmark therefore sweeps delta and reports whether the pair ORDERING survives.
An ordering that moves with delta is an artefact of the regulariser.

The consistency arm checks d_R against measured stacking outcomes. There are FIVE
measured pairs across two runs. Five pairs cannot validate a predictor -- this
arm can only show a contradiction, never confirm one.

    CUDA_VISIBLE_DEVICES="" uv run python \
        benchmarks/factory/geometry/riemannian_metric/benchmark_riemannian_domain_distance.py
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from runtime.canon import CANON, REPO_ROOT, adapter_path
from runtime.novel_peft import FoldableExpert
from runtime.riemannian_covariance import (
    airm_components, joint_subspace_operators, log_euclidean_distance,
    riemannian_affine_invariant_distance,
)

DOMAINS = ["astral", "postgresql", "duckdb", "financial", "python_modern", "python_web"]
MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DELTA = 0.05
DELTA_SWEEP = [0.02, 0.05, 0.10, 0.20]

# Measured stacking outcomes, for the consistency arm. Both runs are v4 adapters
# at 2048 tokens, so the geometry side of this arm is computed on v4 too -- the
# 6x6 headline matrix is v6, and comparing v6 geometry to v4 behaviour would be
# an uncontrolled mismatch.
GROUND_TRUTH = {
    "stacked_4expert_matrix_2048.json": {
        "solo": {"astral": "Solo Astral", "postgresql": "Solo PostgreSQL",
                 "duckdb": "Solo DuckDB"},
        "pairs": {("astral", "postgresql"): "ast+pg   [SHIP A]",
                  ("astral", "duckdb"): "ast+duck [SHIP B]",
                  ("postgresql", "duckdb"): "pg+duck  [collide]"},
        "base": "Base Model",
        "test_domains": {"financial_planning": "financial", "astral": "astral",
                         "postgresql": "postgresql", "duckdb": "duckdb"},
    },
    "stacked_experts_v4_all_clean.json": {
        "solo": {"astral": "ast", "postgresql": "pg", "financial": "fin"},
        "pairs": {("financial", "astral"): "fin+ast",
                  ("financial", "postgresql"): "fin+pg"},
        "base": "base",
        "test_domains": {"financial_planning": "financial", "astral": "astral",
                         "postgresql": "postgresql"},
    },
}


# ---------------------------------------------------------------------------

def self_test(experts: dict[str, FoldableExpert]) -> None:
    """Abort unless d_R is a function of the ADAPTER, not of its rank basis.

    This is the gate the first version had no way to fail. It costs ~0.2s.
    """
    a, b = experts["astral"], experts["postgresql"]
    key = next(k for k in a.factors if "down_proj" in k)
    ua, va = a.factors[key]
    ub, vb = b.factors[key]
    r = ua.shape[1]

    base = airm_components(*joint_subspace_operators(
        ua, va, a.scaling, ub, vb, b.scaling, delta=DELTA)[:2])["total"]

    vals = []
    for seed in range(4):
        torch.manual_seed(seed)
        R, _ = torch.linalg.qr(torch.randn(r, r, dtype=torch.float64))
        Sa, Sb, _ = joint_subspace_operators(
            ua, va, a.scaling, (ub.double() @ R), (R.T @ vb.double()), b.scaling,
            delta=DELTA)
        vals.append(airm_components(Sa, Sb)["total"])
    drift = max(abs(v - base) for v in vals)

    Sx, Sy, _ = joint_subspace_operators(ua, va, a.scaling, ua, va, a.scaling,
                                         delta=DELTA)
    self_d = airm_components(Sx, Sy)["total"]

    print(f"  [gate] rank-basis rotation drift  {drift:.2e}   (must be < 1e-6)")
    print(f"  [gate] self-distance d_R(X, X)    {self_d:.2e}   (must be < 1e-6)")
    if drift >= 1e-6 or self_d >= 1e-6:
        raise SystemExit(
            "\n  ABORT: d_R is not a function of the adapter. This is the exact\n"
            "  defect that made DECISIONS.md §55 wrong. Do not report these numbers.")
    print("  [gate] PASS -- d_R depends on the adapter, not on its coordinates\n")


def pair_distance(ea: FoldableExpert, eb: FoldableExpert, keys: list[str],
                  delta: float) -> dict:
    """Mean AIRM over the shared modules, split into scale and shape."""
    tot = sc = sh = 0.0
    ks: list[int] = []
    for key in keys:
        ua, va = ea.factors[key]
        ub, vb = eb.factors[key]
        Sa, Sb, k = joint_subspace_operators(ua, va, ea.scaling, ub, vb, eb.scaling,
                                             delta=delta)
        c = airm_components(Sa, Sb)
        tot += c["total"]; sc += c["scale"]; sh += c["shape"]; ks.append(k)
    n = max(1, len(keys))
    return {"d_R": tot / n, "scale": sc / n, "shape": sh / n,
            "mean_k": float(np.mean(ks)) if ks else 0.0, "layers": len(keys)}


def matrix_for(experts: dict[str, FoldableExpert], domains: list[str],
               keys: list[str], delta: float) -> tuple[np.ndarray, dict]:
    n = len(domains)
    M = np.zeros((n, n))
    detail = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = pair_distance(experts[domains[i]], experts[domains[j]], keys, delta)
            M[i, j] = M[j, i] = d["d_R"]
            detail[f"{domains[i]}|{domains[j]}"] = d
    return M, detail


def spearman(a: list[float], b: list[float]) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / den) if den > 1e-12 else 0.0


def print_matrix(title: str, M: np.ndarray, domains: list[str]) -> None:
    print(f"\n {title}")
    print(" " + "-" * (18 + 11 * len(domains)))
    print(f" {'':16s}" + "".join(f"{d[:9]:>11}" for d in domains))
    for i, d in enumerate(domains):
        print(f" {d:16s}" + "".join(
            "          ." if i == j else f"{M[i, j]:11.3f}"
            for j in range(len(domains))))


# ---------------------------------------------------------------------------

def consistency_arm(delta: float) -> dict:
    """Does d_R contradict the stacking we actually measured? (n=5 pairs.)"""
    print("\n" + "=" * 92)
    print(" CONSISTENCY ARM -- d_R on v4 adapters vs MEASURED v4 stacking outcomes")
    print("=" * 92)

    v4 = {}
    for dom in ["astral", "postgresql", "duckdb", "financial"]:
        try:
            v4[dom] = FoldableExpert.from_dir(adapter_path(dom, version="v4"), dom)
        except FileNotFoundError:
            print(f"  no v4 adapter for {dom}; consistency arm incomplete")
    keys = sorted(set.intersection(*[set(e.factors) for e in v4.values()])) if v4 else []
    keys = [k for k in keys if any(m in k for m in MODULES)]

    rows = []
    for fname, spec in GROUND_TRUTH.items():
        path = REPO_ROOT / "results/benchmarks" / fname
        if not path.exists():
            print(f"  missing {fname}, skipped")
            continue
        means = json.loads(path.read_text())["means"]
        for (x, y), arm in spec["pairs"].items():
            if x not in v4 or y not in v4 or arm not in means:
                continue
            own, other = [], []
            for test_key, dom in spec["test_domains"].items():
                stacked = means[arm][test_key]
                if dom in (x, y):
                    solo = means[spec["solo"][dom]][test_key]
                    own.append(stacked - solo)
                else:
                    other.append(stacked - means[spec["base"]][test_key])
            g = pair_distance(v4[x], v4[y], keys, delta)
            rows.append({"pair": f"{x}+{y}", "run": fname.split('.')[0][:22],
                         "d_R": g["d_R"], "shape": g["shape"], "scale": g["scale"],
                         "synergy": float(np.mean(own)) if own else float("nan"),
                         "collateral": float(np.mean(other)) if other else float("nan")})

    if not rows:
        print("  nothing measurable")
        return {"pairs": []}

    print(f"\n {'pair':22s} {'d_R':>8s} {'shape':>8s} {'scale':>8s} "
          f"{'synergy':>9s} {'collateral':>11s}   run")
    print(" " + "-" * 88)
    for r in sorted(rows, key=lambda r: r["d_R"]):
        print(f" {r['pair']:22s} {r['d_R']:8.3f} {r['shape']:8.3f} {r['scale']:8.3f} "
              f"{r['synergy']:+9.2f} {r['collateral']:+11.2f}   {r['run']}")

    dr = [r["d_R"] for r in rows]
    syn = [r["synergy"] for r in rows]
    col = [r["collateral"] for r in rows]
    rs_syn, rs_col = spearman(dr, syn), spearman(dr, col)
    print(f"\n  Spearman  d_R vs synergy     {rs_syn:+.3f}")
    print(f"  Spearman  d_R vs collateral   {rs_col:+.3f}")
    print(f"\n  n={len(rows)} pairs, across two runs. At n={len(rows)} a Spearman of")
    print("  +-1.000 arises by chance often enough to be worthless as evidence.")
    print("  Read this arm ONLY as: does the geometry flatly contradict behaviour?")
    return {"pairs": rows, "spearman_synergy": rs_syn, "spearman_collateral": rs_col,
            "n": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/benchmarks/riemannian_domain_geodesics.json")
    ap.add_argument("--delta", type=float, default=DELTA)
    args = ap.parse_args()

    if torch.cuda.is_available():
        raise SystemExit(
            "  This probe is CPU-only by design (~seconds). A visible GPU means it\n"
            "  would contend with whatever else is on the card. Re-run with\n"
            "  CUDA_VISIBLE_DEVICES=\"\".")

    t0 = time.time()
    print("=" * 92)
    print(" RIEMANNIAN GEODESIC DISTANCE BETWEEN DOMAIN EXPERTS")
    print(" AIRM in a shared subspace -- Ch.3 §3.5, spherical shrinkage Ch.8 §8.1.4")
    print(f" adapters {CANON.ADAPTER_VERSION}   delta={args.delta}   CPU")
    print("=" * 92)

    experts = {d: FoldableExpert.from_dir(adapter_path(d), d) for d in DOMAINS}
    print(f"  loaded {len(experts)} experts in {time.time()-t0:.2f}s\n")
    self_test(experts)

    shared = sorted(set.intersection(*[set(e.factors) for e in experts.values()]))
    keys = [k for k in shared if any(m in k for m in MODULES)]
    print(f"  {len(keys)} weight matrices shared by all {len(DOMAINS)} experts")

    M, detail = matrix_for(experts, DOMAINS, keys, args.delta)
    print_matrix(f"AIRM GEODESIC DISTANCE d_R  ({CANON.ADAPTER_VERSION}, "
                 f"delta={args.delta}, mean over {len(keys)} matrices)", M, DOMAINS)

    off = M[np.triu_indices(len(DOMAINS), 1)]
    print(f"\n  off-diagonal  min {off.min():.3f}  max {off.max():.3f}  "
          f"spread {off.max()-off.min():.3f}  "
          f"({100*(off.max()-off.min())/off.mean():.1f}% of the mean)")

    print("\n  scale vs shape -- is a domain FARTHER, or just BIGGER?")
    print(f"  {'pair':28s} {'d_R':>8s} {'shape':>8s} {'scale':>8s} {'mean k':>8s}")
    print("  " + "-" * 64)
    for k_, v in sorted(detail.items(), key=lambda kv: -kv[1]["d_R"]):
        print(f"  {k_:28s} {v['d_R']:8.3f} {v['shape']:8.3f} {v['scale']:8.3f} "
              f"{v['mean_k']:8.2f}")

    print("\n" + "=" * 92)
    print(f" DELTA SENSITIVITY -- does the pair ORDERING survive the regulariser?")
    print("=" * 92)
    ref = None
    sweep = {}
    for d in DELTA_SWEEP:
        Md, _ = matrix_for(experts, DOMAINS, keys, d)
        o = Md[np.triu_indices(len(DOMAINS), 1)]
        ref = o if ref is None else ref
        rho = spearman(list(ref), list(o))
        sweep[str(d)] = {"min": float(o.min()), "max": float(o.max()),
                         "spearman_vs_first": rho}
        print(f"  delta={d:<5} d_R in [{o.min():7.3f}, {o.max():7.3f}]   "
              f"rank-corr vs delta={DELTA_SWEEP[0]}: {rho:+.3f}")

    cons = consistency_arm(args.delta)

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": CANON.stamp() | {"delta": args.delta, "metric": "AIRM-shared-subspace",
                                   "modules": MODULES, "n_matrices": len(keys)},
        "domains": DOMAINS, "airm": M.tolist(), "pairs": detail,
        "delta_sweep": sweep, "consistency": cons,
        "elapsed_seconds": time.time() - t0,
    }, indent=2))
    print(f"\n  {time.time()-t0:.1f}s   wrote {args.out}")


if __name__ == "__main__":
    main()
