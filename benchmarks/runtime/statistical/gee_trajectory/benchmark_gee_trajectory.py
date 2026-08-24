r"""Benchmark: GEE drift detection on autocorrelated agent trajectories (Ch.6 §6.3).

WHAT IS ON TRIAL
----------------
The proposal is to detect multi-turn reasoning drift with a marginal model that carries
a working correlation -- AR(1) across turns -- and a sandwich variance, instead of
treating per-turn signals as independent draws.

The claim is NOT that AR(1) is the true correlation structure of an agent conversation.
It is that a detector which ignores autocorrelation has the wrong variance and therefore
the wrong false-alarm rate. That is a calibration claim, and calibration is measurable:

  A. SIZE. Under NO drift, how often does each detector fire at a nominal 5%? An honest
     detector fires 5% of the time. This is the whole benchmark -- a drift alarm that
     fires on a normal thinking chain will swap an expert mid-conversation for nothing.

  B. POWER. Once size is correct, how much real drift does each detector catch? Power is
     only comparable between arms that hold their size, so arms that over-reject are
     marked rather than ranked.

  C. THE SMALL-K BOUNDARY. The sandwich is a large-cluster estimator. With a handful of
     live conversations it is biased downward and over-rejects for a completely
     different reason. The sweep over K locates that boundary instead of assuming it
     away, which matters because the real artifact on disk has K = 3.

Ground truth is known by construction in A-C (the data are generated with and without a
slope), which is the only way a false-alarm rate can be measured at all. The real
multi-turn artifact is reported afterwards as a demonstration, with its cluster count
placed against the boundary measured in C.

Usage:
  uv run python benchmarks/runtime/statistical/gee_trajectory/benchmark_gee_trajectory.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

from runtime.cpu_bench import enforce_cpu_only, parse_bootstrap_flags  # noqa: E402

# --threads must be honoured before numpy loads; argparse is too late. These two
# benchmarks never touch the GPU: they are Monte Carlo over small matrices, where a
# kernel launch costs more than the computation it would carry.
_BOOT = parse_bootstrap_flags(sys.argv)
enforce_cpu_only(threads=_BOOT["threads"])

import argparse  # noqa: E402
import json  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from runtime.canon import REPO_ROOT  # noqa: E402
from runtime.cpu_bench import print_header, time_repeats, write_telemetry  # noqa: E402
from runtime.gee_trajectory import DriftMonitor, fit_gee  # noqa: E402

NOMINAL = 0.05

# The four detectors. "naive" is the thing in the field: independence + model-based SE.
ARMS: dict[str, dict[str, Any]] = {
    "naive_independence": {"corr": "independence", "robust": False, "correction": None},
    "independence_sandwich": {"corr": "independence", "robust": True, "correction": None},
    "ar1_sandwich": {"corr": "ar1", "robust": True, "correction": None},
    "ar1_sandwich_df": {"corr": "ar1", "robust": True, "correction": "df"},
    "exchangeable_sandwich": {"corr": "exchangeable", "robust": True, "correction": None},
}


def simulate_trajectories(
    rng: np.random.Generator, K: int, T: int, rho: float, slope: float, sigma: float = 1.0
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """K conversations of T turns, AR(1) within conversation, optional linear drift."""
    y_list, X_list = [], []
    t_idx = np.arange(T, dtype=np.float64)
    t_centred = t_idx - t_idx.mean()
    for _ in range(K):
        e = np.empty(T)
        e[0] = rng.normal(0.0, sigma / np.sqrt(max(1 - rho**2, 1e-6)))
        for t in range(1, T):
            e[t] = rho * e[t - 1] + rng.normal(0.0, sigma)
        y_list.append(0.5 + slope * t_centred + e)
        X_list.append(np.column_stack([np.ones(T), t_centred]))
    return y_list, X_list


def rejection_rate(
    K: int, T: int, rho: float, slope: float, replicates: int, seed: int
) -> dict[str, dict[str, Any]]:
    """Fraction of replicates in which each arm rejects "no drift" at the nominal level."""
    counts = dict.fromkeys(ARMS, 0)
    alphas: list[float] = []
    fit_ms: list[float] = []
    rng = np.random.default_rng(seed)

    for _ in range(replicates):
        y_list, X_list = simulate_trajectories(rng, K, T, rho, slope)
        for name, cfg in ARMS.items():
            fit = fit_gee(y_list, X_list, corr=cfg["corr"], small_sample_correction=cfg["correction"])
            if fit.wald(1, robust=cfg["robust"])["p_value"] < NOMINAL:
                counts[name] += 1
            if name == "ar1_sandwich":
                alphas.append(fit.alpha)
                fit_ms.append(fit.fit_seconds * 1000.0)

    out: dict[str, dict[str, Any]] = {}
    for name, c in counts.items():
        rate = c / replicates
        se = float(np.sqrt(max(rate * (1 - rate), 1e-12) / replicates))
        out[name] = {"rejection_rate": rate, "mc_standard_error": se,
                     "ci95": [max(0.0, rate - 1.96 * se), min(1.0, rate + 1.96 * se)]}
    out["_diagnostics"] = {
        "alpha_hat_mean": float(np.mean(alphas)) if alphas else float("nan"),
        "alpha_true": rho,
        "fit_ms_median": float(np.median(fit_ms)) if fit_ms else float("nan"),
        "replicates": replicates,
    }
    return out


def size_sweep(rhos: tuple[float, ...], K: int, T: int, replicates: int, seed: int) -> dict[str, Any]:
    rows = []
    for rho in rhos:
        res = rejection_rate(K, T, rho, slope=0.0, replicates=replicates, seed=seed + int(rho * 100))
        rows.append({"rho": rho, "K": K, "T": T, **res})
    return {"nominal": NOMINAL, "K": K, "T": T, "rows": rows}


def power_sweep(slopes: tuple[float, ...], rho: float, K: int, T: int,
                replicates: int, seed: int) -> dict[str, Any]:
    rows = []
    for slope in slopes:
        res = rejection_rate(K, T, rho, slope=slope, replicates=replicates, seed=seed + int(slope * 10000))
        rows.append({"slope": slope, **res})
    return {"rho": rho, "K": K, "T": T, "rows": rows}


def cluster_sweep(k_grid: tuple[int, ...], rho: float, T: int, replicates: int, seed: int) -> dict[str, Any]:
    rows = []
    for K in k_grid:
        res = rejection_rate(K, T, rho, slope=0.0, replicates=replicates, seed=seed + K)
        rows.append({"K": K, **res})
    return {"rho": rho, "T": T, "rows": rows}


def latency_sweep(k_grid: tuple[int, ...], t_grid: tuple[int, ...], repeats: int) -> dict[str, Any]:
    """Per-refit cost. A per-turn detector must fit inside the gap between turns."""
    rng = np.random.default_rng(0)
    rows = []
    for K in k_grid:
        for T in t_grid:
            y_list, X_list = simulate_trajectories(rng, K, T, rho=0.6, slope=0.0)
            _, stats = time_repeats(
                lambda y=y_list, X=X_list: fit_gee(y, X, corr="ar1"), repeats=repeats, warmup=2
            )
            rows.append({"K": K, "T": T, "n_obs": K * T, "fit_ms": stats})
    return {"rows": rows}


def real_multi_turn_arm(path: Path) -> dict[str, Any] | None:
    """The 15-step multi-turn execution artifact: composite score per step, per arm."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    steps = data.get("steps", [])
    if not steps:
        return None

    series: dict[str, list[float]] = {}
    for step in steps:
        for arm_name, res in step.get("results", {}).items():
            score = res.get("composite_score")
            if score is not None:
                series.setdefault(arm_name, []).append(float(score))

    monitor = DriftMonitor(corr="ar1", min_turns=5, small_sample_correction="df")
    for arm_name, values in series.items():
        for v in values:
            monitor.observe(arm_name, v)

    verdict = monitor.assess()
    fold_latencies = [
        res["fold_latency_ms"]
        for step in steps
        for res in step.get("results", {}).values()
        if res.get("fold_latency_ms") is not None
    ]
    return {
        "source": str(path.relative_to(REPO_ROOT)),
        "adapter_version": data.get("adapter_version"),
        "arms": {k: {"n_turns": len(v), "mean": float(np.mean(v))} for k, v in series.items()},
        "verdict": verdict,
        "median_fold_latency_ms": float(np.median(fold_latencies)) if fold_latencies else None,
    }


def print_rate_table(rows: list[dict[str, Any]], key: str, key_label: str, target: float | None) -> None:
    print("  " + "-" * 108, flush=True)
    header = f"  {key_label:>8} | " + " | ".join(f"{name:>21}" for name in ARMS)
    print(header, flush=True)
    print("  " + "-" * 108, flush=True)
    for r in rows:
        cells = []
        for name in ARMS:
            rate = r[name]["rejection_rate"]
            se = r[name]["mc_standard_error"]
            flag = ""
            if target is not None:
                inflated = rate > target + 1.96 * max(se, 1e-9) + 0.005
                flag = " !!" if inflated else "   "
            cells.append(f"{rate*100:>17.1f}%{flag}")
        print(f"  {r[key]:>8} | " + " | ".join(cells), flush=True)
    print("  " + "-" * 108, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "benchmarks" / "gee_trajectory_drift.json"))
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--clusters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--threads", type=int, default=8,
                    help="BLAS thread cap (read before numpy loads, see parse_bootstrap_flags)")
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--multi-turn-artifact",
                    default=str(REPO_ROOT / "results" / "benchmarks" / "multi_turn_execution_results_v4.json"))
    args = ap.parse_args()

    print_header(
        "GEE TRAJECTORY DRIFT DETECTION -- AUTOCORRELATED MULTI-TURN SIGNALS (Ch.6 §6.3)",
        f"Working correlation + sandwich variance vs an independence detector "
        f"({args.replicates} Monte Carlo replicates per cell)",
    )
    payload: dict[str, Any] = {"benchmark": "gee_trajectory_drift", "nominal_level": NOMINAL}

    print(f"\n[1/5] FALSE ALARMS under NO drift (K={args.clusters} conversations, T={args.turns} turns).", flush=True)
    print(f"      An honest detector fires {NOMINAL*100:.0f}% of the time. '!!' marks significant inflation.",
          flush=True)
    size = size_sweep((0.0, 0.3, 0.6, 0.9), args.clusters, args.turns, args.replicates, args.seed)
    payload["size"] = size
    print_rate_table(size["rows"], "rho", "AR1 rho", NOMINAL)
    for r in size["rows"]:
        d = r["_diagnostics"]
        print(f"      rho={r['rho']:.1f}: alpha_hat={d['alpha_hat_mean']:.3f} (true {d['alpha_true']:.1f}), "
              f"fit {d['fit_ms_median']:.2f} ms", flush=True)

    print(f"\n[2/5] POWER against real drift (rho=0.6, K={args.clusters}, T={args.turns}).", flush=True)
    power = power_sweep((0.005, 0.01, 0.02, 0.04), 0.6, args.clusters, args.turns, args.replicates, args.seed)
    payload["power"] = power
    print_rate_table(power["rows"], "slope", "slope", None)
    print("      Arms flagged '!!' above do not hold their size; their power is not comparable.", flush=True)

    print("\n[3/5] SMALL-K BOUNDARY: false alarms vs number of live conversations (rho=0.6, no drift).",
          flush=True)
    clusters = cluster_sweep((3, 5, 10, 20, 40, 80), 0.6, args.turns, max(args.replicates // 2, 200), args.seed)
    payload["cluster_boundary"] = clusters
    print_rate_table(clusters["rows"], "K", "K", NOMINAL)

    print("\n[4/5] Refit latency per turn.", flush=True)
    latency = latency_sweep((3, 10, 30), (10, 20, 50), args.repeats)
    payload["latency"] = latency
    print(f"  {'K':>4} | {'T':>4} | {'obs':>6} | {'median ms':>10} | {'IQR ms':>10}", flush=True)
    print("  " + "-" * 52, flush=True)
    for r in latency["rows"]:
        print(f"  {r['K']:>4} | {r['T']:>4} | {r['n_obs']:>6} | {r['fit_ms']['median']:>10.2f} | "
              f"{r['fit_ms']['iqr']:>10.2f}", flush=True)

    print("\n[5/5] Real multi-turn artifact.", flush=True)
    real = real_multi_turn_arm(Path(args.multi_turn_artifact))
    payload["real_multi_turn"] = real
    if real:
        v = real["verdict"]
        print(f"  {real['source']} (adapter {real['adapter_version']}): "
              f"{v['n_clusters']} arms x {v['n_obs'] // max(v['n_clusters'], 1)} turns", flush=True)
        print(f"  slope={v['slope']:+.5f}  alpha_hat={v['alpha_hat']:.3f}  "
              f"robust p={v['p_robust']:.4f}  naive p={v['p_naive']:.4f}  fit {v['fit_ms']:.2f} ms", flush=True)
        boundary = next((r["K"] for r in clusters["rows"]
                         if r["ar1_sandwich_df"]["rejection_rate"] <= NOMINAL + 0.02), None)
        print(f"  K={v['n_clusters']} is BELOW the usable cluster count measured in [3/5] "
              f"(first K holding size: {boundary}). Read this as a demonstration, not a calibrated test.",
              flush=True)
    else:
        print("  artifact not found -- skipped", flush=True)

    naive_worst = max(r["naive_independence"]["rejection_rate"] for r in size["rows"])
    ar1_worst = max(r["ar1_sandwich"]["rejection_rate"] for r in size["rows"])
    payload["verdict"] = {
        "naive_worst_false_alarm_rate": naive_worst,
        "ar1_sandwich_worst_false_alarm_rate": ar1_worst,
        "nominal": NOMINAL,
        "false_alarm_reduction": naive_worst / ar1_worst if ar1_worst > 0 else float("inf"),
    }

    print("\n" + "=" * 108, flush=True)
    print(" VERDICT", flush=True)
    print("=" * 108, flush=True)
    print(f"  Worst-case false alarm rate at nominal {NOMINAL*100:.0f}%: "
          f"independence detector {naive_worst*100:.1f}%, AR(1)+sandwich {ar1_worst*100:.1f}% "
          f"({naive_worst/max(ar1_worst,1e-9):.1f}x fewer false swaps).", flush=True)
    print("=" * 108, flush=True)

    write_telemetry(args.out, payload, extra_config=vars(args))


if __name__ == "__main__":
    main()
