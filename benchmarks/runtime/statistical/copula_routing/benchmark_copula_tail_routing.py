r"""Benchmark: nonparametric copula tail dependence for dual-expert routing (Ch.3 §3.6).

WHAT IS ON TRIAL
----------------
The proposal is to trigger dual-expert folding from the UPPER TAIL DEPENDENCE lambda_U
between expert activation magnitudes, rather than from a linear similarity score, on the
argument that the prompts needing two experts are exactly the extreme co-activations a
correlation cannot see.

Measured in four parts:

  A. CAN lambda_U BE ESTIMATED AT ALL at the sample sizes a router has? Swept against
     three copulas with lambda_U known in closed form: Gaussian (lambda_U = 0 exactly),
     Student-t, and Gumbel. This is where the method either survives or does not --
     the estimator uses only the k largest observations, and both k too small and k too
     large break it in opposite directions.

  B. A THRESHOLD THAT MEANS SOMETHING. Because the estimator is biased upward on
     tail-INDEPENDENT data, an absolute cutoff like "lambda_U > 0" is not implementable.
     The benchmark calibrates the cutoff against a Gaussian-copula null with matched
     Pearson correlation and then measures detection rate at that calibrated cutoff.

  C. ROUTING QUALITY against a ground-truth stream where it is known which tokens
     genuinely need two experts. Compared with a Pearson-correlation router at MATCHED
     false-activation rate, so the comparison is between dependence measures and not
     between two arbitrary thresholds.

  D. COST. Per-token decision latency, and the wasted fold time a false dual-activation
     buys, priced at the fold latency measured in this repo's own multi-turn artifact.

The real v7 experts are measured too, on the adapter-factor surrogate stream -- that arm
says whether any real expert PAIR shows tail dependence above the calibrated null.

Usage:
  uv run python benchmarks/runtime/statistical/copula_routing/benchmark_copula_tail_routing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "apps"))

from runtime.cpu_bench import enforce_cpu_only, parse_bootstrap_flags  # noqa: E402

# --threads and --gpu-capture must be honoured before numpy/torch load; argparse is too
# late. The estimators stay on the CPU either way -- --gpu-capture only moves the
# base-model forward pass that produces the real per-expert activation magnitudes.
_BOOT = parse_bootstrap_flags(sys.argv)
enforce_cpu_only(threads=_BOOT["threads"], allow_gpu=_BOOT["allow_gpu"])

import argparse  # noqa: E402
import json  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from runtime.activation_features import (  # noqa: E402
    domain_prompts,
    expert_magnitudes,
    load_experts,
    real_expert_response_magnitudes,
)
from runtime.canon import DOMAINS, REPO_ROOT  # noqa: E402
from runtime.copula_routing import (  # noqa: E402
    CopulaTailRouter,
    gaussian_null_cutoff,
    pseudo_observations,
    sample_gaussian_copula,
    sample_gumbel_copula,
    sample_t_copula,
    tail_dependence_matrix,
    theoretical_lambda_u_gumbel,
    theoretical_lambda_u_t,
    upper_tail_dependence,
    upper_tail_dependence_log,
)
from runtime.cpu_bench import median_iqr, print_header, write_telemetry  # noqa: E402

N_GRID = (512, 2048, 8192)
K_FRACTIONS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16)
# thresholds run negative because the calibrated metric is an EXCESS over a null cutoff
THRESHOLD_GRID = tuple(np.round(np.arange(-0.60, 1.0, 0.02), 4))
ROUTER_ARMS = ("pearson", "tail", "tail_calibrated")
# A wasted fold is real work, so the comparison is made at matched waste budgets rather
# than at one arbitrary operating point.
FALSE_DUAL_BUDGETS = (0.005, 0.01, 0.02, 0.05)


def reference_families() -> dict[str, dict[str, Any]]:
    return {
        "gaussian_rho0.7": {"sampler": lambda n, rng: sample_gaussian_copula(n, 0.7, rng),
                            "lambda_u": 0.0, "tail_dependent": False},
        "t_rho0.7_nu4": {"sampler": lambda n, rng: sample_t_copula(n, 0.7, 4.0, rng),
                         "lambda_u": theoretical_lambda_u_t(0.7, 4.0), "tail_dependent": True},
        "gumbel_theta2": {"sampler": lambda n, rng: sample_gumbel_copula(n, 2.0, rng),
                          "lambda_u": theoretical_lambda_u_gumbel(2.0), "tail_dependent": True},
    }


def estimator_sweep(replicates: int, seed: int) -> dict[str, Any]:
    """Claim A: bias, spread, and the separation between tail-dependent and not."""
    families = reference_families()
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    for n in N_GRID:
        for frac in K_FRACTIONS:
            k = max(int(round(frac * n)), 2)
            if k >= n:
                continue
            est: dict[str, dict[str, np.ndarray]] = {}
            for fam, spec in families.items():
                exc, log = [], []
                for _ in range(replicates):
                    U = pseudo_observations(spec["sampler"](n, rng))
                    exc.append(upper_tail_dependence(U, 0, 1, k))
                    log.append(upper_tail_dependence_log(U, 0, 1, k))
                est[fam] = {"exceedance": np.array(exc), "log": np.array(log, dtype=float)}

            null = est["gaussian_rho0.7"]["exceedance"]
            row: dict[str, Any] = {"n": n, "k": k, "k_over_n": k / n, "families": {}}
            for fam, spec in families.items():
                e = est[fam]["exceedance"]
                lg = est[fam]["log"]
                pooled = np.sqrt(0.5 * (e.var() + null.var())) or 1e-12
                row["families"][fam] = {
                    "true_lambda_u": spec["lambda_u"],
                    "exceedance_mean": float(e.mean()),
                    "exceedance_sd": float(e.std()),
                    "exceedance_bias": float(e.mean() - spec["lambda_u"]),
                    "log_mean": float(np.nanmean(lg)),
                    "log_sd": float(np.nanstd(lg)),
                    "separation_from_null": float((e.mean() - null.mean()) / pooled),
                }
            rows.append(row)
    return {"replicates": replicates, "rows": rows,
            "families": {f: {"lambda_u": s["lambda_u"], "tail_dependent": s["tail_dependent"]}
                         for f, s in families.items()}}


def calibration_sweep(replicates: int, seed: int, alpha: float = 0.05) -> dict[str, Any]:
    """Claim B: a cutoff calibrated on a matched tail-independent null, then its power."""
    families = reference_families()
    rng = np.random.default_rng(seed)
    rows = []
    for n in N_GRID:
        k = max(int(round(0.02 * n)), 2)
        null = np.array([
            upper_tail_dependence(pseudo_observations(sample_gaussian_copula(n, 0.7, rng)), 0, 1, k)
            for _ in range(replicates)
        ])
        cutoff = float(np.quantile(null, 1.0 - alpha))
        detect = {}
        for fam, spec in families.items():
            vals = np.array([
                upper_tail_dependence(pseudo_observations(spec["sampler"](n, rng)), 0, 1, k)
                for _ in range(replicates)
            ])
            detect[fam] = {"detection_rate": float(np.mean(vals > cutoff)),
                           "true_lambda_u": spec["lambda_u"],
                           "tail_dependent": spec["tail_dependent"]}
        rows.append({"n": n, "k": k, "alpha": alpha, "null_cutoff": cutoff,
                     "naive_zero_cutoff_false_positive_rate": float(np.mean(null > 0.0)),
                     "detection": detect})
    return {"replicates": replicates, "rows": rows}


def make_token_stream(
    n: int,
    rng: np.random.Generator,
    p_cross: float = 0.06,
    rho_distractor: float = 0.85,
    theta: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A stream built so the two dependence measures must disagree.

    Four experts, and the ground truth is deliberately adversarial to a correlation
    router:

      db(0) + systems(1)   -- near ZERO linear correlation in the bulk, but a p_cross
                              minority of tokens drives both into their joint upper tail
                              together (Gumbel). This is the pair that genuinely needs
                              dual folding, and it is invisible to Pearson.
      python(2) + finance(3) -- strongly correlated (rho = 0.85 Gaussian copula) and
                              therefore the pair a correlation router will co-activate,
                              but a Gaussian copula has lambda_U = 0 exactly: they drift
                              together in the middle and separate in the extreme. Every
                              dual activation of this pair is wasted work.

    Marginals go through exp() so magnitudes are heavily skewed. A rank-based measure is
    unaffected by that; a moment-based one is not.
    """
    from scipy.stats import norm

    U = np.empty((n, 4))
    U[:, 0] = rng.random(n)
    U[:, 1] = rng.random(n)
    cov = np.array([[1.0, rho_distractor], [rho_distractor, 1.0]])
    U[:, 2:] = norm.cdf(rng.multivariate_normal(np.zeros(2), cov, size=n))

    is_cross = rng.random(n) < p_cross
    n_cross = int(is_cross.sum())
    if n_cross:
        keep = np.empty((0, 2))
        while keep.shape[0] < n_cross:
            G = sample_gumbel_copula(8 * n_cross + 128, theta, rng)
            keep = np.vstack([keep, G[(G[:, 0] > 0.9) & (G[:, 1] > 0.9)]])
        U[is_cross, 0] = keep[:n_cross, 0]
        U[is_cross, 1] = keep[:n_cross, 1]
    return np.exp(3.0 * U), is_cross


def routing_sweep(n_fit: int, n_eval: int, seed: int, fold_latency_ms: float) -> dict[str, Any]:
    """Claim C: dual-activation quality at MATCHED false-activation rate."""
    rng = np.random.default_rng(seed)
    H_fit, _ = make_token_stream(n_fit, rng)
    H_eval, is_cross = make_token_stream(n_eval, rng)
    names = ["db_expert", "systems_expert", "python_expert", "finance_expert"]

    curves: dict[str, list[dict[str, Any]]] = {}
    latency: dict[str, dict[str, float]] = {}
    fit_seconds: dict[str, float] = {}

    for metric in ROUTER_ARMS:
        router = CopulaTailRouter(pair_metric=metric, tail_quantile=0.95, seed=seed).fit(H_fit, names)
        fit_seconds[metric] = router.fit_seconds
        curve = []
        for thr in THRESHOLD_GRID:
            router.pair_threshold = float(thr)
            decisions = [router.route(row) for row in H_eval]
            dual = np.array([len(dec.co_activated) > 0 for dec in decisions])
            correct_pair = np.array([
                set(dec.experts) >= {"db_expert", "systems_expert"} for dec in decisions
            ])
            tp = float(np.sum(correct_pair & is_cross))
            fp = float(np.sum(dual & ~is_cross))
            fn = float(np.sum(~correct_pair & is_cross))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            curve.append({
                "threshold": float(thr),
                "recall_on_cross_tokens": recall,
                "precision": precision,
                "f1": (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0,
                "false_dual_rate_on_normal": fp / max(float(np.sum(~is_cross)), 1.0),
                "dual_activation_rate": float(dual.mean()),
            })
        curves[metric] = curve
        latency[metric] = median_iqr([router.route(row).decision_us for row in H_eval[:2000]])

    # matched-cost comparison: highest recall available at <= 5% false dual activations
    matched: dict[str, dict[str, Any]] = {}
    partial_auc = {}
    for metric, curve in curves.items():
        matched[metric] = {}
        for budget in FALSE_DUAL_BUDGETS:
            feasible = [c for c in curve if c["false_dual_rate_on_normal"] <= budget]
            best = max(feasible, key=lambda c: c["recall_on_cross_tokens"]) if feasible else None
            matched[metric][f"{budget:.3f}"] = best
        pts = sorted(
            [(c["false_dual_rate_on_normal"], c["recall_on_cross_tokens"]) for c in curve if
             c["false_dual_rate_on_normal"] <= 0.05]
        )
        partial_auc[metric] = (
            float(np.trapezoid([r for _, r in pts], [f for f, _ in pts]) / 0.05) if len(pts) > 1 else float("nan")
        )

    pair_matrices = {}
    for metric in ROUTER_ARMS:
        r = CopulaTailRouter(pair_metric=metric, seed=seed).fit(H_fit, names)
        pair_matrices[metric] = {
            "db_vs_systems_TRUE_PAIR": float(r.pair_matrix[0, 1]),
            "python_vs_finance_DISTRACTOR": float(r.pair_matrix[2, 3]),
            "db_vs_python": float(r.pair_matrix[0, 2]),
            "ranks_true_pair_first": bool(r.pair_matrix[0, 1] > r.pair_matrix[2, 3]),
            "k": r.k,
        }

    wasted = {
        metric: (
            m["0.050"]["false_dual_rate_on_normal"] * fold_latency_ms if m.get("0.050") else float("nan")
        )
        for metric, m in matched.items()
    }
    return {
        "n_fit": n_fit,
        "n_eval": n_eval,
        "cross_token_fraction": float(is_cross.mean()),
        "curves": curves,
        "matched_by_false_dual_budget": matched,
        "pair_fit_seconds": fit_seconds,
        "pair_matrices": pair_matrices,
        "decision_latency_us": latency,
        "partial_auc_to_5pct_false_dual": partial_auc,
        "fold_latency_ms_used": fold_latency_ms,
        "expected_wasted_fold_ms_per_normal_token": wasted,
    }


def real_expert_arm(
    n_samples: int,
    replicates: int,
    seed: int,
    use_real_tokens: bool,
    prompts_per_domain: int = 40,
    max_length: int = 384,
    cache_path: str | None = None,
) -> dict[str, Any]:
    """Does any real v7 expert PAIR show tail dependence above its matched null?

    Two data sources, and the difference between them is the whole caveat:

      real tokens (--gpu-capture) -- each expert's EXACT response magnitude on the real
        activations a base-model forward produced for real domain text. This is the signal
        a router would actually score. Requires a forward pass, hence the GPU flag.

      surrogate (default)         -- the same factors driven by Gaussian inputs. Kept as a
        control: the only structure it can show is what the factors impose.

    Before any tail question is asked, the routing signal is validated: if the experts'
    magnitudes cannot tell domains apart at all, their co-activation structure is moot.
    """
    domains = [d for d in DOMAINS if d not in ("merged_sql", "merged_all")]
    experts = load_experts(domains)
    validation: dict[str, Any] = {}

    if use_real_tokens:
        prompts = domain_prompts(domains, n_per_domain=prompts_per_domain, seed=11, source="training")
        captured = real_expert_response_magnitudes(
            prompts, experts, max_length=max_length, device="cuda:0", cache_path=cache_path
        )
        M = captured["magnitudes"]
        names = captured["expert_names"]
        source = "real_expert_response"
        provenance = {k: captured[k] for k in ("source", "model_id", "device", "n_prompts", "n_tokens")}

        # --- is the routing signal discriminative at all? ---------------------
        token_domains = np.array(captured["token_domains"])
        U = pseudo_observations(M)
        per_domain = {}
        correct = 0
        scored = 0
        for d in names:
            mask = token_domains == d
            if not mask.any():
                continue
            mean_u = U[mask].mean(axis=0)
            winner = names[int(mean_u.argmax())]
            correct += int(winner == d)
            scored += 1
            per_domain[d] = {
                "mean_pseudo_obs": {n: float(x) for n, x in zip(names, mean_u, strict=True)},
                "argmax": winner,
                "n_tokens": int(mask.sum()),
            }
        top1 = np.array(names)[U.argmax(axis=1)]
        validation = {
            "domain_argmax_correct": correct,
            "domains_scored": scored,
            "per_token_top1_accuracy": float((top1 == token_domains).mean()),
            "chance_accuracy": 1.0 / len(names),
            "per_domain": per_domain,
            "note": "raw magnitudes are NOT discriminative (per-expert scale dominates); "
                    "these are AFTER the router's rank transform, which removes that scale.",
        }
    else:
        M, names = expert_magnitudes(experts, n_samples=n_samples, seed=seed)
        source = "residual_stream_surrogate"
        provenance = {"source": source, "n_samples": n_samples}

    n = M.shape[0]
    k = max(int(round(0.02 * n)), 2)
    lam = tail_dependence_matrix(M, k=k)
    corr = np.corrcoef(M, rowvar=False)

    rng = np.random.default_rng(seed)
    pairs = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            rho = float(np.clip(corr[a, b], -0.95, 0.95))
            cutoff = gaussian_null_cutoff(n, k, rho, replicates, rng)
            pairs.append({
                "pair": f"{names[a]}+{names[b]}",
                "lambda_u_hat": float(lam[a, b]),
                "pearson": float(corr[a, b]),
                "null_cutoff_95": cutoff,
                "excess_over_null": float(lam[a, b]) - cutoff,
                "exceeds_null": bool(lam[a, b] > cutoff),
            })

    # Does calibration actually REORDER the routing table, on real data?
    by_raw = [p["pair"] for p in sorted(pairs, key=lambda p: -p["lambda_u_hat"])]
    by_excess = [p["pair"] for p in sorted(pairs, key=lambda p: -p["excess_over_null"])]
    displaced = [p for p in by_raw[:3] if p not in by_excess[:3]]

    return {
        "source": source,
        "provenance": provenance,
        "n_samples": n,
        "k": k,
        "experts": names,
        "routing_signal_validation": validation,
        "pairs": pairs,
        "n_pairs_above_null": int(sum(p["exceeds_null"] for p in pairs)),
        "top3_by_raw_lambda": by_raw[:3],
        "top3_by_calibrated_excess": by_excess[:3],
        "displaced_by_calibration": displaced,
    }


def median_fold_latency(path: Path) -> float:
    """Price a wasted fold from this repo's own measurement, not from a guess."""
    if not path.exists():
        return float("nan")
    data = json.loads(path.read_text())
    vals = [
        res["fold_latency_ms"]
        for step in data.get("steps", [])
        for res in step.get("results", {}).values()
        if res.get("fold_latency_ms") and res.get("expert") not in (None, "base")
    ]
    return float(np.median(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "benchmarks" / "copula_tail_routing.json"))
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--n-fit", type=int, default=4096)
    ap.add_argument("--n-eval", type=int, default=4096)
    ap.add_argument("--real-samples", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--threads", type=int, default=8,
                    help="BLAS thread cap (read before numpy loads, see parse_bootstrap_flags)")
    ap.add_argument("--no-real-experts", action="store_true")
    ap.add_argument("--gpu-capture", action="store_true",
                    help="score experts on REAL tokens via a base-model forward pass "
                         "(GPU); without it the arm falls back to the Gaussian surrogate")
    ap.add_argument("--prompts-per-domain", type=int, default=40)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--expert-cache",
                    default=str(REPO_ROOT / "results" / "cache" / "real_expert_magnitudes.npz"))
    ap.add_argument("--multi-turn-artifact",
                    default=str(REPO_ROOT / "results" / "benchmarks" / "multi_turn_execution_results_v4.json"))
    args = ap.parse_args()

    print_header(
        "NONPARAMETRIC COPULA TAIL DEPENDENCE -- DUAL-EXPERT ROUTING (Ch.3 §3.6)",
        "lambda_U against copulas with known tails, then routing at matched false-activation cost",
    )
    payload: dict[str, Any] = {"benchmark": "copula_tail_routing"}

    print("\n[1/4] Estimator behaviour against copulas with lambda_U known in closed form.", flush=True)
    est = estimator_sweep(args.replicates, args.seed)
    payload["estimator"] = est
    print("  " + "-" * 106, flush=True)
    print(f"  {'n':>6} {'k':>5} {'k/n':>6} | {'gaussian (0.000)':>22} | {'t nu=4 (0.391)':>22} | "
          f"{'separation':>10}", flush=True)
    print("  " + "-" * 106, flush=True)
    for row in est["rows"]:
        g = row["families"]["gaussian_rho0.7"]
        t = row["families"]["t_rho0.7_nu4"]
        print(f"  {row['n']:>6} {row['k']:>5} {row['k_over_n']:>6.3f} | "
              f"{g['exceedance_mean']:>10.3f} +/- {g['exceedance_sd']:<8.3f} | "
              f"{t['exceedance_mean']:>10.3f} +/- {t['exceedance_sd']:<8.3f} | "
              f"{t['separation_from_null']:>9.2f}d", flush=True)
    print("  " + "-" * 106, flush=True)
    print("  The gaussian column has TRUE lambda_U = 0 exactly. Everything it reports above 0 is bias.",
          flush=True)

    print("\n[2/4] Calibrated cutoff (95th percentile of a matched tail-independent null).", flush=True)
    calib = calibration_sweep(args.replicates, args.seed + 1)
    payload["calibration"] = calib
    print("  " + "-" * 100, flush=True)
    print(f"  {'n':>6} {'k':>5} | {'null cutoff':>11} | {'FPR at cutoff 0':>15} | "
          f"{'detect t':>9} | {'detect gumbel':>13} | {'detect gauss':>12}", flush=True)
    print("  " + "-" * 100, flush=True)
    for r in calib["rows"]:
        print(f"  {r['n']:>6} {r['k']:>5} | {r['null_cutoff']:>11.3f} | "
              f"{r['naive_zero_cutoff_false_positive_rate']*100:>14.0f}% | "
              f"{r['detection']['t_rho0.7_nu4']['detection_rate']*100:>8.0f}% | "
              f"{r['detection']['gumbel_theta2']['detection_rate']*100:>12.0f}% | "
              f"{r['detection']['gaussian_rho0.7']['detection_rate']*100:>11.0f}%", flush=True)
    print("  " + "-" * 100, flush=True)

    fold_ms = median_fold_latency(Path(args.multi_turn_artifact))
    print(f"\n[3/4] Routing on a ground-truth stream (a wasted fold costs {fold_ms:.2f} ms, "
          f"measured in this repo's multi-turn artifact).", flush=True)
    routing = routing_sweep(args.n_fit, args.n_eval, args.seed + 2, fold_ms)
    payload["routing"] = routing
    print(f"  {routing['cross_token_fraction']*100:.1f}% of evaluation tokens genuinely need "
          f"db_expert + systems_expert.", flush=True)
    print("  " + "-" * 104, flush=True)
    print(f"  {'router':>16} | {'true pair':>9} | {'distractor':>10} | {'ranks true 1st':>14} | "
          f"{'pAUC(<=5%)':>10} | {'fit ms':>8}", flush=True)
    print("  " + "-" * 104, flush=True)
    for metric in ROUTER_ARMS:
        pm = routing["pair_matrices"][metric]
        print(f"  {metric:>16} | {pm['db_vs_systems_TRUE_PAIR']:>9.3f} | "
              f"{pm['python_vs_finance_DISTRACTOR']:>10.3f} | {str(pm['ranks_true_pair_first']):>14} | "
              f"{routing['partial_auc_to_5pct_false_dual'][metric]:>10.3f} | "
              f"{routing['pair_fit_seconds'][metric]*1000:>8.1f}", flush=True)
    print("  " + "-" * 104, flush=True)
    print("\n  Recall on cross-domain tokens at matched wasted-fold budgets:", flush=True)
    print("  " + "-" * 104, flush=True)
    header = f"  {'router':>16} | " + " | ".join(f"{'<=' + f'{b*100:g}% waste':>16}" for b in FALSE_DUAL_BUDGETS)
    print(header, flush=True)
    print("  " + "-" * 104, flush=True)
    for metric in ROUTER_ARMS:
        cells = []
        for b in FALSE_DUAL_BUDGETS:
            m = routing["matched_by_false_dual_budget"][metric].get(f"{b:.3f}")
            cells.append(f"{m['recall_on_cross_tokens']*100:>15.1f}%" if m else f"{'infeasible':>16}")
        print(f"  {metric:>16} | " + " | ".join(cells), flush=True)
    print("  " + "-" * 104, flush=True)
    for metric, lat in routing["decision_latency_us"].items():
        print(f"  {metric:>10} routing decision: {lat['median']:.2f} us median "
              f"(IQR {lat['iqr']:.2f}) -- {lat['median']/1000:.4f} ms/token", flush=True)

    if not args.no_real_experts:
        label = "REAL TOKENS" if args.gpu_capture else "Gaussian surrogate"
        print(f"\n[4/4] Real v7 experts, scored on {label}.", flush=True)
        real = real_expert_arm(
            args.real_samples, max(args.replicates // 2, 50), args.seed + 3,
            use_real_tokens=args.gpu_capture,
            prompts_per_domain=args.prompts_per_domain,
            max_length=args.max_length,
            cache_path=args.expert_cache,
        )
        payload["real_experts"] = real

        val = real["routing_signal_validation"]
        if val:
            print(f"  Routing signal check: domain argmax correct "
                  f"{val['domain_argmax_correct']}/{val['domains_scored']}, "
                  f"per-token top-1 {val['per_token_top1_accuracy']*100:.1f}% "
                  f"(chance {val['chance_accuracy']*100:.1f}%)", flush=True)
            print("  " + "-" * 96, flush=True)
            print(f"  {'token domain':>14} | " + " ".join(f"{n[:9]:>10}" for n in real["experts"]) + " | argmax",
                  flush=True)
            print("  " + "-" * 96, flush=True)
            for d in real["experts"]:
                row = val["per_domain"].get(d)
                if not row:
                    continue
                cells = " ".join(f"{row['mean_pseudo_obs'][n]:>10.3f}" for n in real["experts"])
                print(f"  {d[:14]:>14} | {cells} | {row['argmax']}", flush=True)
            print("  " + "-" * 96, flush=True)

        print(f"\n  {len(real['experts'])} experts, {real['n_samples']} {label} samples, k={real['k']}",
              flush=True)
        print("  " + "-" * 96, flush=True)
        print(f"  {'pair':>30} | {'lambda_U':>9} | {'pearson':>8} | {'null 95%':>9} | "
              f"{'excess':>8} | {'above':>5}", flush=True)
        print("  " + "-" * 96, flush=True)
        for p_ in sorted(real["pairs"], key=lambda p_: -p_["excess_over_null"]):
            print(f"  {p_['pair']:>30} | {p_['lambda_u_hat']:>9.3f} | {p_['pearson']:>8.3f} | "
                  f"{p_['null_cutoff_95']:>9.3f} | {p_['excess_over_null']:>+8.3f} | "
                  f"{'YES' if p_['exceeds_null'] else '-':>5}", flush=True)
        print("  " + "-" * 96, flush=True)
        print(f"  {real['n_pairs_above_null']} of {len(real['pairs'])} expert pairs exceed their "
              f"matched tail-independent null.", flush=True)
        if real["displaced_by_calibration"]:
            print(f"  Calibration REORDERS the routing table: {real['displaced_by_calibration']} "
                  f"drops out of the top 3 once its own correlation-matched null is subtracted.",
                  flush=True)

    def _recall(metric: str, budget: float) -> float | None:
        m = routing["matched_by_false_dual_budget"][metric].get(f"{budget:.3f}")
        return m["recall_on_cross_tokens"] if m else None

    payload["verdict"] = {
        "recall_at_1pct_waste": {m: _recall(m, 0.01) for m in ROUTER_ARMS},
        "recall_at_5pct_waste": {m: _recall(m, 0.05) for m in ROUTER_ARMS},
        "decision_latency_us_median": routing["decision_latency_us"]["tail_calibrated"]["median"],
        "partial_auc": routing["partial_auc_to_5pct_false_dual"],
        "gaussian_null_bias_at_2pct_k": min(
            (r for r in est["rows"] if r["n"] == 2048),
            key=lambda r: abs(r["k_over_n"] - 0.02),
        )["families"]["gaussian_rho0.7"]["exceedance_bias"],
        "pair_matrices": routing["pair_matrices"],
    }

    print("\n" + "=" * 106, flush=True)
    print(" VERDICT", flush=True)
    print("=" * 106, flush=True)
    v = payload["verdict"]
    for budget, key in ((1, "recall_at_1pct_waste"), (5, "recall_at_5pct_waste")):
        cells = ", ".join(
            f"{m}: {(v[key][m] or 0)*100:.1f}%" if v[key][m] is not None else f"{m}: infeasible"
            for m in ROUTER_ARMS
        )
        print(f"  Recall on cross-domain tokens at a <={budget}% wasted-fold budget -- {cells}", flush=True)
    print("  Ranks the genuinely tail-dependent pair above the merely-correlated distractor: "
          + ", ".join(f"{m}={v['pair_matrices'][m]['ranks_true_pair_first']}" for m in ROUTER_ARMS), flush=True)
    print(f"  Per-token decision cost {v['decision_latency_us_median']:.2f} us "
          f"({v['decision_latency_us_median']/1000:.4f} ms).", flush=True)
    print(f"  lambda_U estimator bias on a TRULY tail-independent pair (n=2048, k/n=0.02): "
          f"+{v['gaussian_null_bias_at_2pct_k']:.3f} -- absolute cutoffs are not usable, "
          f"calibrated ones are.", flush=True)
    print("=" * 106, flush=True)

    write_telemetry(args.out, payload, extra_config=vars(args))


if __name__ == "__main__":
    main()
