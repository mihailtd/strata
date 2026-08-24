r"""RETIRED 2026-08-24 -- CLIME sparse precision estimation in the n << d regime (Ch.4 §4.4.3).

╔══════════════════════════════════════════════════════════════════════════════╗
║ THIS BENCHMARK IS SUPERSEDED. Kept for provenance. Do not cite as live.       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ It retired its own proposal, on three findings:                               ║
║                                                                               ║
║  1. Graphical lasso BEATS CLIME at n/d <= 1 -- the B=1 decode regime CLIME    ║
║     was proposed for -- while being ~35x faster (1.3 ms vs 48 ms at d=32).    ║
║  2. The real head cross-talk graph is NOT REPRODUCIBLE at decode window        ║
║     sizes: two disjoint 32-token windows each find ~80 edges and share ZERO.  ║
║     That is a property of the activations, so it closes the APPLICATION --    ║
║     no estimator fixes it.                                                    ║
║  3. Column decoupling converts to only 2.41x of a 59.5x parallel bound.       ║
║                                                                               ║
║ What held: the entrywise guarantee ||S*Theta - I||_inf <= lambda, exactly, on ║
║ every feasible column at every lambda tested.                                 ║
║                                                                               ║
║ Read README.md in this directory before proposing any per-token head graph.   ║
║ See docs/DECISIONS.md §66.                                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Original header follows.

Benchmark: CLIME sparse precision estimation in the n << d regime (Ch.4 §4.4.3).

WHAT IS ON TRIAL
----------------
The proposal is to estimate the head-to-head cross-talk graph from activations by
inverting the empirical covariance -- and, because at B = 1 decode (or speculative
M = 2..4) the sample count is far below the feature dimension, to do it with CLIME
rather than a direct inverse: d decoupled linear programs, each with a hard entrywise
feasibility bound, instead of one rank-deficient matrix inversion.

Three claims, measured separately:

  A. THE FAILURE IT AVOIDS. That a direct inverse in the n << d regime produces dense
     spurious cross-talk. Measured against a KNOWN sparse precision matrix: edge
     recovery F1, and the magnitude the naive inverse actually reaches.

  B. THE GUARANTEE. That || Sigma_hat Theta_hat - I ||_inf <= lambda holds. Checked on
     the solved output over feasible columns, not assumed. The companion measurement is
     the feasibility frontier: below some lambda the LP has no solution at all, which is
     a fact about the sample size and belongs in the report.

  C. THE PARALLELISM. That the columns are independent. Measured as thread-pool wall
     time against the serial run, plus the critical-path bound (the slowest single
     column) that a perfectly parallel implementation could reach.

Comparators: graphical lasso (ADMM, likelihood-based, but couples all d^2 entries),
Ledoit-Wolf shrinkage (the repo's existing estimator for activation covariance in the
n < p regime), ridge inverse, and the naive inverse.

lambda is NEVER chosen with the ground truth. It is selected by held-out Gaussian
log-likelihood; the oracle-best F1 is reported alongside, clearly labelled, so the gap
between "what selection achieves" and "what the estimator could achieve" is visible.

Usage (provenance only):
  uv run python benchmarks/superseded/clime_head_crosstalk/benchmark_clime_inversion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from runtime.cpu_bench import enforce_cpu_only, parse_bootstrap_flags  # noqa: E402

# --threads and --gpu-capture must be honoured before numpy/torch load; argparse is
# too late. The estimators stay on the CPU either way -- --gpu-capture only moves the
# base-model forward pass that produces the real activations.
_BOOT = parse_bootstrap_flags(sys.argv)
enforce_cpu_only(threads=_BOOT["threads"], allow_gpu=_BOOT["allow_gpu"])

import argparse  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from runtime.activation_features import as_matrix, capture_real_hidden_states, domain_prompts  # noqa: E402
from runtime.canon import REPO_ROOT  # noqa: E402
from runtime.clime_precision import (  # noqa: E402
    clime_lambda_theoretical,
    clime_precision,
    graphical_lasso_admm,
    max_constraint_violation,
    naive_inverse_precision,
    sparsity,
    support_metrics,
    symmetry_gap,
)
from runtime.cpu_bench import print_header, write_telemetry  # noqa: E402
from runtime.vecchia_precision import dense_nll  # noqa: E402

PROBE_DOMAINS = ["astral", "postgresql", "duckdb", "financial", "python_modern", "python_web"]
LAMBDA_GRID = (0.05, 0.1, 0.2, 0.3, 0.45, 0.6)


def ground_truth_precision(d: int, seed: int = 0) -> np.ndarray:
    """Banded chain plus a few hubs: local structure with a handful of long-range edges.

    Both parts are deliberate. The chain is what a layer-local dependence story predicts;
    the hubs are the long-range edges an estimator that only ever finds neighbours would
    miss, and the recall column is where that shows up.
    """
    rng = np.random.default_rng(seed)
    Theta = np.eye(d)
    for i in range(d - 1):
        Theta[i, i + 1] = Theta[i + 1, i] = 0.35
    for hub in rng.choice(d, size=max(d // 16, 1), replace=False):
        for target in rng.choice(d, size=3, replace=False):
            if target != hub:
                Theta[hub, target] = Theta[target, hub] = 0.25
    # shift onto the SPD cone with a margin, then normalise the diagonal to 1
    w = np.linalg.eigvalsh(Theta)
    Theta += (max(0.0, 0.15 - w.min())) * np.eye(d)
    dg = np.sqrt(np.diag(Theta))
    return Theta / np.outer(dg, dg)


def extended_bic(Theta: np.ndarray, S: np.ndarray, n: int, gamma: float = 0.5) -> float:
    """Extended BIC for Gaussian graphical models (Foygel & Drton 2010). Lower is better.

        eBIC = -n (logdet Theta - tr(S Theta)) + |E| log n + 4 gamma |E| log d

    Included because held-out likelihood turns out to be a poor lambda selector for
    CLIME -- it rewards density -- and a runtime needs SOME rule it can apply without
    the ground truth it does not have.
    """
    d = S.shape[0]
    sign, logdet = np.linalg.slogdet(Theta)
    if sign <= 0:
        return float("inf")
    off = ~np.eye(d, dtype=bool)
    n_edges = float(np.sum(np.abs(Theta[off]) > 1e-4) / 2)
    fit_term = -n * (float(logdet) - float(np.trace(S @ Theta)))
    return fit_term + n_edges * np.log(max(n, 2)) + 4.0 * gamma * n_edges * np.log(max(d, 2))


def fit_all_methods(
    X_train: np.ndarray,
    X_val: np.ndarray,
    lambda_grid: tuple[float, ...],
    n_jobs: int = 1,
) -> list[dict[str, Any]]:
    """Every (method, lambda) pair, with the metrics that do not need ground truth."""
    n, d = X_train.shape
    mean = X_train.mean(axis=0)
    Xc = X_train - mean
    S = (Xc.T @ Xc) / n

    fits: list[dict[str, Any]] = []

    def record(method: str, lam: float, Theta: np.ndarray, seconds: float, extra: dict[str, Any]) -> None:
        Sym = 0.5 * (Theta + Theta.T)
        pd = bool(np.all(np.linalg.eigvalsh(Sym) > 1e-10))
        fits.append(
            {
                "method": method,
                "lambda": lam,
                "Theta": Theta,
                "seconds": seconds,
                "positive_definite": pd,
                "val_nll": dense_nll(Sym, mean, X_val) if pd else float("inf"),
                "ebic": extended_bic(Sym, S, n, gamma=0.5) if pd else float("inf"),
                "sparsity": sparsity(Theta),
                **extra,
            }
        )

    for lam in lambda_grid:
        t0 = time.perf_counter()
        res = clime_precision(S, lam, n_jobs=n_jobs)
        record(
            "clime", lam, res.Theta, time.perf_counter() - t0,
            {
                "feasible_fraction": res.feasible_fraction,
                "constraint_violation_raw": max_constraint_violation(S, res.Theta_raw, np.where(res.feasible)[0]),
                "constraint_violation_sym": max_constraint_violation(S, res.Theta),
                "symmetry_gap": symmetry_gap(res.Theta_raw),
                "critical_path_ms": res.critical_path_seconds * 1000.0,
                "ideal_parallel_speedup": res.ideal_parallel_speedup,
            },
        )

        t0 = time.perf_counter()
        Theta_g, info = graphical_lasso_admm(S, lam=lam)
        record("glasso", lam, Theta_g, time.perf_counter() - t0, {"iterations": info["iterations"]})

        t0 = time.perf_counter()
        Theta_r = naive_inverse_precision(S, ridge=lam)
        record("ridge_inverse", lam, Theta_r, time.perf_counter() - t0, {})

    t0 = time.perf_counter()
    Theta_n = naive_inverse_precision(S, ridge=0.0)
    record("naive_inverse", 0.0, Theta_n, time.perf_counter() - t0, {"max_abs_entry": float(np.abs(Theta_n).max())})

    import torch

    from runtime.riemannian_covariance import ledoit_wolf_from_samples

    t0 = time.perf_counter()
    S_lw, delta = ledoit_wolf_from_samples(torch.from_numpy(Xc).float())
    Theta_lw = np.linalg.inv(S_lw.numpy().astype(np.float64))
    record("ledoit_wolf", float(delta), Theta_lw, time.perf_counter() - t0, {"shrinkage_delta": float(delta)})

    return fits


def score_against_truth(fits: list[dict[str, Any]], Theta_true: np.ndarray, tol: float) -> list[dict[str, Any]]:
    rows = []
    for f in fits:
        m = support_metrics(f["Theta"], Theta_true, tol=tol)
        err = float(np.linalg.norm(f["Theta"] - Theta_true, "fro") / np.linalg.norm(Theta_true, "fro"))
        rows.append(
            {k: v for k, v in f.items() if k != "Theta"}
            | {**m, "relative_frobenius_error": err, "max_abs_entry": float(np.abs(f["Theta"]).max())}
        )
    return rows


def summarise_by_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per method: the lambda selected by held-out likelihood, and the oracle-best F1."""
    out: dict[str, Any] = {}
    for method in sorted({r["method"] for r in rows}):
        sub = [r for r in rows if r["method"] == method]
        usable = [r for r in sub if np.isfinite(r["val_nll"])]
        selected = min(usable, key=lambda r: r["val_nll"]) if usable else None
        by_ebic = [r for r in sub if np.isfinite(r["ebic"])]
        selected_ebic = min(by_ebic, key=lambda r: r["ebic"]) if by_ebic else None
        oracle = max(sub, key=lambda r: r["f1"])
        out[method] = {
            "selected_by_heldout_nll": (
                {k: v for k, v in selected.items() if k != "Theta"} if selected else None
            ),
            "selected_by_ebic": (
                {k: v for k, v in selected_ebic.items() if k != "Theta"} if selected_ebic else None
            ),
            "oracle_best_f1": oracle["f1"],
            "oracle_lambda": oracle["lambda"],
            "any_positive_definite": any(r["positive_definite"] for r in sub),
            "max_abs_entry": float(np.max([r["max_abs_entry"] for r in sub])),
        }
    return out


def regime_sweep(d: int, n_grid: tuple[int, ...], repeats: int, seed: int = 0) -> dict[str, Any]:
    Theta_true = ground_truth_precision(d, seed=seed)
    Sigma = np.linalg.inv(Theta_true)
    results: list[dict[str, Any]] = []

    for n in n_grid:
        per_rep: list[dict[str, Any]] = []
        for rep in range(repeats):
            rng = np.random.default_rng(1000 * seed + 17 * n + rep)
            X = rng.multivariate_normal(np.zeros(d), Sigma, size=n + max(n, 64))
            X_train, X_val = X[:n], X[n:]
            fits = fit_all_methods(X_train, X_val, LAMBDA_GRID)
            per_rep.append(summarise_by_method(score_against_truth(fits, Theta_true, tol=1e-3)))

        merged: dict[str, Any] = {}
        for method in per_rep[0]:
            sel = [r[method]["selected_by_heldout_nll"] for r in per_rep if r[method]["selected_by_heldout_nll"]]
            sel_b = [r[method]["selected_by_ebic"] for r in per_rep if r[method]["selected_by_ebic"]]
            merged[method] = {
                "n_reps_with_pd_solution": len(sel),
                "f1_ebic": float(np.mean([s["f1"] for s in sel_b])) if sel_b else float("nan"),
                "lambda_ebic": float(np.mean([s["lambda"] for s in sel_b])) if sel_b else float("nan"),
                "max_abs_entry": float(np.mean([r[method]["max_abs_entry"] for r in per_rep])),
                "f1_selected": float(np.mean([s["f1"] for s in sel])) if sel else float("nan"),
                "precision_selected": float(np.mean([s["precision"] for s in sel])) if sel else float("nan"),
                "recall_selected": float(np.mean([s["recall"] for s in sel])) if sel else float("nan"),
                "lambda_selected": float(np.mean([s["lambda"] for s in sel])) if sel else float("nan"),
                "sparsity_selected": float(np.mean([s["sparsity"] for s in sel])) if sel else float("nan"),
                "seconds_selected": float(np.mean([s["seconds"] for s in sel])) if sel else float("nan"),
                "oracle_best_f1": float(np.mean([r[method]["oracle_best_f1"] for r in per_rep])),
            }
        results.append({"n": n, "d": d, "n_over_d": n / d, "methods": merged})
    return {"d": d, "n_grid": list(n_grid), "regimes": results,
            "true_edges": int((np.abs(Theta_true) > 1e-3).sum() - d) // 2}


def guarantee_sweep(d: int, n: int, seed: int = 0) -> dict[str, Any]:
    """Claim B: feasibility frontier and the entrywise bound, across lambda."""
    Theta_true = ground_truth_precision(d, seed=seed)
    Sigma = np.linalg.inv(Theta_true)
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(np.zeros(d), Sigma, size=n)
    S = np.cov(X, rowvar=False, bias=True)

    rows = []
    for lam in LAMBDA_GRID:
        res = clime_precision(S, lam)
        cols = np.where(res.feasible)[0]
        viol = max_constraint_violation(S, res.Theta_raw, cols)
        rows.append(
            {
                "lambda": lam,
                "feasible_fraction": res.feasible_fraction,
                "violation_raw_feasible": viol,
                "bound_holds": bool(np.isnan(viol) or viol <= lam + 1e-6),
                "violation_after_symmetrisation": max_constraint_violation(S, res.Theta),
                "symmetry_gap": symmetry_gap(res.Theta_raw),
                "sparsity": sparsity(res.Theta),
                "f1": support_metrics(res.Theta, Theta_true, tol=1e-3)["f1"],
                "total_ms": res.total_seconds * 1000.0,
            }
        )
    return {"d": d, "n": n, "theoretical_lambda": clime_lambda_theoretical(n, d), "rows": rows}


def parallel_sweep(d: int, n: int, lam: float, jobs: tuple[int, ...], seed: int = 0) -> dict[str, Any]:
    """Claim C: do the decoupled columns actually convert into wall-clock time?"""
    Theta_true = ground_truth_precision(d, seed=seed)
    rng = np.random.default_rng(seed + 1)
    X = rng.multivariate_normal(np.zeros(d), np.linalg.inv(Theta_true), size=n)
    S = np.cov(X, rowvar=False, bias=True)

    rows = []
    base = None
    for j in jobs:
        res = clime_precision(S, lam, n_jobs=j)
        if base is None:
            base = res.total_seconds
        rows.append(
            {
                "n_jobs": j,
                "wall_seconds": res.total_seconds,
                "speedup_vs_serial": base / res.total_seconds,
                "critical_path_ms": res.critical_path_seconds * 1000.0,
                "ideal_parallel_speedup": res.ideal_parallel_speedup,
                "median_column_ms": float(np.median(res.column_seconds) * 1000.0),
            }
        )
    return {"d": d, "n": n, "lambda": lam, "rows": rows}


def real_head_arm(feats: dict[str, Any], windows: tuple[int, ...], lam: float, n_jobs: int) -> dict[str, Any]:
    """Real per-head activations, sliced into short decode-sized windows (true n << d).

    No ground truth exists here, so the reported quantity is STABILITY: two disjoint
    windows produce two edge sets, and their Jaccard overlap says whether the graph is
    reproducible or is being redrawn from noise each window.
    """
    H = as_matrix(feats, "head_slice")
    n_tokens, d = H.shape
    rows = []
    for w in windows:
        if 2 * w > n_tokens:
            continue
        rng = np.random.default_rng(w)
        idx = rng.permutation(n_tokens)
        A, B = H[idx[:w]], H[idx[w : 2 * w]]

        edges = []
        stats = []
        for block in (A, B):
            S = np.cov(block, rowvar=False, bias=True)
            res = clime_precision(S, lam, n_jobs=n_jobs)
            off = ~np.eye(d, dtype=bool)
            edges.append((np.abs(res.Theta) > 1e-3) & off)
            stats.append(
                {
                    "feasible_fraction": res.feasible_fraction,
                    "sparsity": sparsity(res.Theta),
                    "total_ms": res.total_seconds * 1000.0,
                }
            )
        inter = float(np.sum(edges[0] & edges[1]))
        union = float(np.sum(edges[0] | edges[1]))
        rows.append(
            {
                "window_tokens": w,
                "d_features": d,
                "n_over_d": w / d,
                "edges_window_a": int(edges[0].sum() // 2),
                "edges_window_b": int(edges[1].sum() // 2),
                "jaccard_stability": (inter / union) if union > 0 else float("nan"),
                "feasible_fraction": float(np.mean([s["feasible_fraction"] for s in stats])),
                "sparsity": float(np.mean([s["sparsity"] for s in stats])),
                "total_ms": float(np.mean([s["total_ms"] for s in stats])),
            }
        )
    return {"lambda": lam, "d_features": d, "n_tokens": n_tokens, "rows": rows,
            "head_names_sample": feats.get("head_names", [])[:8]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "benchmarks" / "clime_precision_inversion.json"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8,
                    help="BLAS/torch thread cap (read before numpy loads, see parse_bootstrap_flags)")
    ap.add_argument("--gpu-capture", action="store_true",
                    help="run the base-model activation capture on the GPU; estimators stay on CPU")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--prompts-per-domain", type=int, default=48)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--real-lambda", type=float, default=0.3)
    ap.add_argument("--no-real-activations", action="store_true")
    ap.add_argument("--cache", default=str(REPO_ROOT / "results" / "cache" / "real_layer_activations.npz"))
    args = ap.parse_args()

    print_header(
        "CLIME SPARSE PRECISION INVERSION -- n << d ACTIVATION CROSS-TALK (Ch.4 §4.4.3)",
        "Decoupled l1 linear programs vs a direct inverse, graphical lasso and Ledoit-Wolf shrinkage",
    )
    payload: dict[str, Any] = {"benchmark": "clime_precision_inversion"}

    # --- Claim A -------------------------------------------------------------
    print(f"\n[1/4] Recovery against a known sparse precision matrix (d={args.d})...", flush=True)
    sweep = regime_sweep(args.d, n_grid=(8, 16, 32, 64, 128, 256), repeats=args.repeats)
    payload["recovery"] = sweep
    print(f"  Ground truth carries {sweep['true_edges']} off-diagonal edges "
          f"out of {args.d * (args.d - 1) // 2} possible.", flush=True)
    print("  " + "-" * 104, flush=True)
    print(f"  {'n':>5} {'n/d':>6} | {'method':>14} | {'F1 (NLL-sel)':>12} | {'F1 (eBIC)':>9} | "
          f"{'precision':>9} | {'recall':>7} | {'oracle F1':>9} | {'max|Theta|':>11}", flush=True)
    print("  " + "-" * 104, flush=True)
    for regime in sweep["regimes"]:
        for method, m in regime["methods"].items():
            def _f(v: float, w: int) -> str:
                return f"{v:>{w}.3f}" if np.isfinite(v) else f"{'not PD':>{w}}"
            print(f"  {regime['n']:>5} {regime['n_over_d']:>6.2f} | {method:>14} | {_f(m['f1_selected'], 12)} | "
                  f"{_f(m['f1_ebic'], 9)} | {_f(m['precision_selected'], 9)} | {_f(m['recall_selected'], 7)} | "
                  f"{m['oracle_best_f1']:>9.3f} | {m['max_abs_entry']:>11.2e}", flush=True)
        print("  " + "-" * 104, flush=True)

    # --- Claim B -------------------------------------------------------------
    print("\n[2/4] Feasibility frontier and the entrywise guarantee...", flush=True)
    guarantee = guarantee_sweep(args.d, n=max(args.d // 2, 8))
    payload["guarantee"] = guarantee
    print(f"  d={guarantee['d']}, n={guarantee['n']} (n/d={guarantee['n']/guarantee['d']:.2f}), "
          f"theory scale lambda={guarantee['theoretical_lambda']:.3f}", flush=True)
    print("  " + "-" * 104, flush=True)
    print(f"  {'lambda':>7} | {'feasible':>9} | {'||S0-I||inf':>12} | {'<= lambda':>9} | "
          f"{'after sym':>10} | {'sym gap':>8} | {'sparsity':>8} | {'F1':>6}", flush=True)
    print("  " + "-" * 104, flush=True)
    for r in guarantee["rows"]:
        viol = "n/a" if np.isnan(r["violation_raw_feasible"]) else f"{r['violation_raw_feasible']:.4f}"
        print(f"  {r['lambda']:>7.3f} | {r['feasible_fraction']*100:>8.0f}% | {viol:>12} | "
              f"{str(r['bound_holds']):>9} | {r['violation_after_symmetrisation']:>10.4f} | "
              f"{r['symmetry_gap']:>8.3f} | {r['sparsity']:>8.3f} | {r['f1']:>6.3f}", flush=True)
    print("  " + "-" * 104, flush=True)

    # --- Claim C -------------------------------------------------------------
    print("\n[3/4] Column decoupling -> thread scaling...", flush=True)
    parallel = parallel_sweep(64, n=32, lam=0.3, jobs=(1, 2, 4, args.jobs) if args.jobs > 4 else (1, 2, 4))
    payload["parallel"] = parallel
    print(f"  {'workers':>8} | {'wall s':>8} | {'speedup':>8} | {'critical path ms':>17} | {'ideal bound':>11}",
          flush=True)
    print("  " + "-" * 70, flush=True)
    for r in parallel["rows"]:
        print(f"  {r['n_jobs']:>8} | {r['wall_seconds']:>8.3f} | {r['speedup_vs_serial']:>7.2f}x | "
              f"{r['critical_path_ms']:>17.2f} | {r['ideal_parallel_speedup']:>10.1f}x", flush=True)

    # --- Real arm ------------------------------------------------------------
    if not args.no_real_activations:
        print("\n[4/4] REAL per-head activations, decode-sized windows...", flush=True)
        prompts = domain_prompts(PROBE_DOMAINS, n_per_domain=args.prompts_per_domain, seed=7, source="training")
        feats = capture_real_hidden_states(
            [p["prompt"] for p in prompts], max_length=args.max_length,
            dtype="bfloat16", threads=args.threads, cache_path=args.cache,
            device="cuda:0" if args.gpu_capture else "cpu",
        )
        real = real_head_arm(
            feats, windows=(8, 16, 32, 64, 128, 256, 512, 1024), lam=args.real_lambda, n_jobs=args.jobs
        )
        real["provenance"] = {
            k: feats[k] for k in ("source", "model_id", "device", "n_prompts", "n_tokens", "head_dim")
            if k in feats
        }
        payload["real_heads"] = real
        print(f"  {real['d_features']} head features from {real['n_tokens']} real token positions "
              f"(lambda={args.real_lambda})", flush=True)
        print("  " + "-" * 90, flush=True)
        print(f"  {'window':>7} | {'n/d':>6} | {'edges A':>8} | {'edges B':>8} | {'Jaccard':>8} | "
              f"{'feasible':>9} | {'ms':>8}", flush=True)
        print("  " + "-" * 90, flush=True)
        for r in real["rows"]:
            print(f"  {r['window_tokens']:>7} | {r['n_over_d']:>6.2f} | {r['edges_window_a']:>8} | "
                  f"{r['edges_window_b']:>8} | {r['jaccard_stability']:>8.3f} | "
                  f"{r['feasible_fraction']*100:>8.0f}% | {r['total_ms']:>8.1f}", flush=True)
        print("  " + "-" * 90, flush=True)

    # --- Verdict -------------------------------------------------------------
    hardest = sweep["regimes"][0]["methods"]
    easiest = sweep["regimes"][-1]["methods"]
    verdict = {
        "naive_inverse_f1_at_lowest_n": hardest["naive_inverse"]["f1_selected"],
        "naive_inverse_max_entry_at_lowest_n": hardest["naive_inverse"]["max_abs_entry"],
        "clime_f1_ebic_at_lowest_n": hardest["clime"]["f1_ebic"],
        "clime_oracle_f1_at_highest_n": easiest["clime"]["oracle_best_f1"],
        "clime_f1_at_lowest_n": hardest["clime"]["f1_selected"],
        "glasso_f1_at_lowest_n": hardest["glasso"]["f1_selected"],
        "clime_f1_at_highest_n": easiest["clime"]["f1_selected"],
        "glasso_f1_at_highest_n": easiest["glasso"]["f1_selected"],
        "guarantee_holds_everywhere": all(r["bound_holds"] for r in guarantee["rows"]),
        "measured_thread_speedup": parallel["rows"][-1]["speedup_vs_serial"],
        "ideal_parallel_bound": parallel["rows"][-1]["ideal_parallel_speedup"],
    }
    payload["verdict"] = verdict

    print("\n" + "=" * 104, flush=True)
    print(" VERDICT", flush=True)
    print("=" * 104, flush=True)
    print(f"  Entrywise guarantee ||S*Theta - I||_inf <= lambda held on every feasible column: "
          f"{verdict['guarantee_holds_everywhere']}", flush=True)
    print(f"  Column decoupling: {verdict['measured_thread_speedup']:.2f}x measured on threads, "
          f"against an embarrassingly-parallel bound of {verdict['ideal_parallel_bound']:.1f}x.", flush=True)
    print(f"  Direct inverse at n/d=0.25 reaches entries of {verdict['naive_inverse_max_entry_at_lowest_n']:.1e} "
          f"and is never positive definite -- the failure CLIME exists to avoid is real and measured.",
          flush=True)
    print("=" * 104, flush=True)

    write_telemetry(args.out, payload, extra_config=vars(args))


if __name__ == "__main__":
    main()
