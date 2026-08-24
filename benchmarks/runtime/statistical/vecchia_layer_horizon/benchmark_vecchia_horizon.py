r"""Benchmark: Vecchia horizon windowing for cross-layer activation covariance (Ch.12 §12.2.1).

WHAT IS ON TRIAL
----------------
The proposal is to replace the O(L^3) dense precision estimate over an L-layer stack
with a Vecchia approximation that conditions each layer on its m predecessors, giving a
banded GMRF and O(L m^2) cost -- and, specifically, to set m = 2 on the argument that
residual skip-connections make cross-layer dependence local.

Two separate claims, benchmarked separately, because only one of them is arithmetic:

  A. THE SPEEDUP.  Banded fit/solve versus dense covariance + inverse, swept over
     L = 8..80 (80 = the 70B geometry this runtime targets). Pure complexity; the only
     question is the measured exponent.

  B. THE HORIZON.  Whether m = 2 is enough. Measured as held-out negative
     log-likelihood on REAL per-layer activations captured from a CPU forward pass of
     the base model, plus KL divergence to the dense fit and the fraction of
     off-diagonal precision mass the band actually contains. If the NLL curve has not
     flattened by m = 2, then m = 2 is the wrong horizon no matter how fast it is.

ARMS
----
  real       -- REAL per-layer residual deltas, base model on CPU, real domain prompts.
  surrogate  -- adapter-factor residual-stream surrogate. Kept as a NULL control: its
                Gaussian inputs produce no cross-layer structure, so any horizon result
                that looks the same on both arms is an artifact of the estimator.
  synthetic  -- banded GMRF calibrated to the lag-1 partial correlation measured on the
                real arm. Used ONLY for the L-scaling timings, where real data stops at
                L = 32 but the target stack is L = 80.

GPU: never touched. The base-model forward runs on CPU with device visibility cleared.

Usage:
  uv run python benchmarks/runtime/statistical/vecchia_layer_horizon/benchmark_vecchia_horizon.py
  uv run python .../benchmark_vecchia_horizon.py --no-real-activations   # skip the forward pass
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

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

from runtime.activation_features import (  # noqa: E402
    as_matrix,
    capture_real_hidden_states,
    domain_prompts,
    load_experts,
    residual_stream_features,
)
from runtime.canon import REPO_ROOT  # noqa: E402
from runtime.cpu_bench import print_header, time_repeats, write_telemetry  # noqa: E402
from runtime.vecchia_precision import (  # noqa: E402
    band_mass_fraction,
    banded_solve_lapack,
    conditional_decay_profile,
    dense_nll,
    dense_precision,
    fit_vecchia,
    fit_vecchia_batched,
    gaussian_kl,
    to_banded_storage,
    to_dense_precision,
    vecchia_nll,
    vecchia_quadratic,
)

PROBE_DOMAINS = ["astral", "postgresql", "duckdb", "financial", "python_modern", "python_web"]
M_GRID = (0, 1, 2, 3, 4, 6, 8, 12, 16, 24)
# 32 = Qwen3.5-4B, 80 = the 70B geometry the streaming runtime targets. 128..512 are
# past any real stack and are here only to locate the crossover where O(L^3) bites.
L_GRID = (8, 16, 32, 48, 64, 80, 128, 256, 512)
N_GRID = (64, 1024)


# ---------------------------------------------------------------------------
# Data arms
# ---------------------------------------------------------------------------
def build_real_arm(
    n_per_domain: int, max_length: int, threads: int, cache: Path, device: str = "cpu"
) -> dict[str, Any]:
    prompts = domain_prompts(PROBE_DOMAINS, n_per_domain=n_per_domain, seed=7, source="training")
    t0 = time.perf_counter()
    feats = capture_real_hidden_states(
        [p["prompt"] for p in prompts],
        max_length=max_length,
        dtype="bfloat16",
        threads=threads,
        cache_path=str(cache),
        device=device,
    )
    feats["capture_seconds"] = time.perf_counter() - t0
    feats["n_domains"] = len(PROBE_DOMAINS)
    return feats


def build_surrogate_arm(n_samples: int) -> dict[str, Any]:
    expert = load_experts(["astral"])["astral"]
    return residual_stream_features(expert, n_samples=n_samples, seed=42)


def synthetic_banded(L: int, n: int, lag1: float, seed: int = 0) -> np.ndarray:
    """AR(1)-in-layer-index Gaussian field with a target lag-1 partial correlation."""
    rng = np.random.default_rng(seed)
    phi = float(np.clip(lag1, 0.0, 0.95))
    X = np.zeros((n, L))
    X[:, 0] = rng.standard_normal(n)
    for ell in range(1, L):
        X[:, ell] = phi * X[:, ell - 1] + np.sqrt(1 - phi**2) * rng.standard_normal(n)
    return X


# ---------------------------------------------------------------------------
# Claim B: the horizon
# ---------------------------------------------------------------------------
def horizon_sweep(X: np.ndarray, label: str, split: float = 0.6) -> dict[str, Any]:
    n, L = X.shape
    cut = int(n * split)
    X_tr, X_val = X[:cut], X[cut:]

    Theta_dense, mean_dense = dense_precision(X_tr, ridge=1e-4)
    nll_dense = dense_nll(Theta_dense, mean_dense, X_val)
    Sigma_ref = np.linalg.inv(Theta_dense)
    decay = conditional_decay_profile(Theta_dense, max_lag=min(12, L - 1))

    rows: list[dict[str, Any]] = []
    for m in [m for m in M_GRID if m < L]:
        f = fit_vecchia(X_tr, m=m, ridge=1e-4)
        Theta_m = to_dense_precision(f)
        rows.append(
            {
                "m": m,
                "val_nll": vecchia_nll(f, X_val),
                "train_nll": vecchia_nll(f, X_tr),
                "kl_to_dense": gaussian_kl(Sigma_ref, Theta_m),
                "band_mass_of_dense": band_mass_fraction(Theta_dense, m),
                "stored_parameters": f.nonzeros(),
                "fit_ms": f.fit_seconds * 1000.0,
            }
        )

    nll_m0 = rows[0]["val_nll"]
    gain_total = nll_m0 - nll_dense
    # If the dense fit does not beat the diagonal model on held-out data there is no
    # cross-layer structure to capture, and "% of the gain" would divide by noise.
    dense_beats_independent = gain_total > 1e-9
    for r in rows:
        r["gain_captured"] = ((nll_m0 - r["val_nll"]) / gain_total) if dense_beats_independent else float("nan")

    best = min(r["val_nll"] for r in rows)
    saturating_m = next(r["m"] for r in rows if r["val_nll"] <= best + 0.05)

    return {
        "label": label,
        "n_samples": int(n),
        "n_layers": int(L),
        "n_train": int(cut),
        "n_val": int(n - cut),
        "dense_val_nll": nll_dense,
        "dense_beats_independent": bool(dense_beats_independent),
        "dense_parameters": int(L * (L + 1) / 2),
        "saturating_m_within_0.05_nats": int(saturating_m),
        "partial_correlation_by_lag": {str(k): v for k, v in decay.items()},
        "sweep": rows,
    }


# ---------------------------------------------------------------------------
# Claim A: the speedup
# ---------------------------------------------------------------------------
def scaling_sweep(lag1: float, repeats: int, m: int = 2) -> dict[str, Any]:
    """Fit and apply cost versus stack depth, at two update-window sizes.

    Both axes matter and they pull in opposite directions. The dense arm pays
    O(n L^2) to form the covariance and O(L^3) to invert it, so at a long update
    window the covariance dominates and the cubic term is invisible; at a short window
    (n = 64 tokens, the realistic decode-side case) the inverse dominates and the
    asymptotics are what is actually being paid.
    """
    from scipy.linalg import cho_factor, cho_solve

    rows: list[dict[str, Any]] = []
    for n_samples in N_GRID:
        for L in L_GRID:
            X = synthetic_banded(L, n_samples, lag1, seed=L)
            y = np.random.default_rng(L).standard_normal(L)

            _, dense_fit = time_repeats(lambda X=X: dense_precision(X, ridge=1e-6), repeats=repeats, warmup=2)
            _, vec_fit = time_repeats(
                lambda X=X: fit_vecchia_batched(X, m=m, ridge=1e-6), repeats=repeats, warmup=2
            )
            _, loop_fit = time_repeats(lambda X=X: fit_vecchia(X, m=m, ridge=1e-6), repeats=repeats, warmup=2)

            Theta_dense, mean_dense = dense_precision(X, ridge=1e-6)
            factor = fit_vecchia_batched(X, m=m, ridge=1e-6)
            ab = to_banded_storage(factor)
            cho = cho_factor(Theta_dense)
            yc = y - mean_dense

            _, dense_quad = time_repeats(lambda T=Theta_dense, y=yc: float(y @ (T @ y)),
                                         repeats=repeats * 3, warmup=5)
            _, vec_quad = time_repeats(lambda f=factor, y=y: vecchia_quadratic(f, y),
                                       repeats=repeats * 3, warmup=5)
            _, dense_solve = time_repeats(lambda c=cho, y=y: cho_solve(c, y), repeats=repeats * 3, warmup=5)
            _, band_solve = time_repeats(lambda ab=ab, y=y: banded_solve_lapack(ab, y),
                                         repeats=repeats * 3, warmup=5)

            rows.append(
                {
                    "n_samples": n_samples,
                    "L": L,
                    "dense_fit_ms": dense_fit,
                    "vecchia_fit_ms": vec_fit,
                    "vecchia_loop_fit_ms": loop_fit,
                    "fit_speedup": dense_fit["median"] / vec_fit["median"],
                    "dense_quadratic_us": {k: v * 1000.0 for k, v in dense_quad.items()},
                    "vecchia_quadratic_us": {k: v * 1000.0 for k, v in vec_quad.items()},
                    "quadratic_speedup": dense_quad["median"] / vec_quad["median"],
                    "dense_cho_solve_us": {k: v * 1000.0 for k, v in dense_solve.items()},
                    "banded_solve_us": {k: v * 1000.0 for k, v in band_solve.items()},
                    "solve_speedup": dense_solve["median"] / band_solve["median"],
                }
            )

    exponents: dict[str, float] = {}
    for n_samples in N_GRID:
        sub = [r for r in rows if r["n_samples"] == n_samples]
        logL = np.log([r["L"] for r in sub])
        exponents[f"dense_n{n_samples}"] = float(
            np.polyfit(logL, np.log([r["dense_fit_ms"]["median"] for r in sub]), 1)[0]
        )
        exponents[f"vecchia_n{n_samples}"] = float(
            np.polyfit(logL, np.log([r["vecchia_fit_ms"]["median"] for r in sub]), 1)[0]
        )

    return {
        "m": m,
        "lag1_partial_correlation": lag1,
        "n_grid": list(N_GRID),
        "rows": rows,
        "empirical_exponents": exponents,
    }


# ---------------------------------------------------------------------------
def print_horizon(result: dict[str, Any]) -> None:
    print(f"\n  Arm: {result['label']}   "
          f"(n={result['n_samples']} samples, L={result['n_layers']} layers, "
          f"{result['n_train']} train / {result['n_val']} held out)", flush=True)
    print("  " + "-" * 100, flush=True)
    print(f"  {'m':>3} | {'held-out NLL':>13} | {'gain vs m=0':>11} | {'KL to dense':>11} | "
          f"{'band mass':>9} | {'params':>7} | {'fit ms':>7}", flush=True)
    print("  " + "-" * 100, flush=True)
    for r in result["sweep"]:
        gain = f"{r['gain_captured']*100:>10.1f}%" if np.isfinite(r["gain_captured"]) else f"{'n/a':>11}"
        print(f"  {r['m']:>3} | {r['val_nll']:>13.4f} | {gain} | "
              f"{r['kl_to_dense']:>11.4f} | {r['band_mass_of_dense']*100:>8.1f}% | "
              f"{r['stored_parameters']:>7} | {r['fit_ms']:>7.3f}", flush=True)
    print("  " + "-" * 100, flush=True)
    print(f"  {'dense':>3} | {result['dense_val_nll']:>13.4f} | {100.0:>10.1f}% | "
          f"{0.0:>11.4f} | {100.0:>8.1f}% | {result['dense_parameters']:>7} |", flush=True)
    lags = result["partial_correlation_by_lag"]
    shown = ", ".join(f"lag{k}={v:.3f}" for k, v in list(lags.items())[:6])
    print(f"\n  Mean |partial correlation| by layer separation: {shown}", flush=True)


def print_scaling(result: dict[str, Any]) -> None:
    for n_samples in result["n_grid"]:
        sub = [r for r in result["rows"] if r["n_samples"] == n_samples]
        print(f"\n  Update window n = {n_samples} tokens  (m={result['m']}):", flush=True)
        print("  " + "-" * 108, flush=True)
        print(f"  {'L':>4} | {'dense fit ms':>12} | {'Vecchia fit ms':>14} | {'speedup':>8} | "
              f"{'dense quad us':>13} | {'Vecchia quad us':>15} | {'quad speedup':>12}", flush=True)
        print("  " + "-" * 108, flush=True)
        for r in sub:
            print(f"  {r['L']:>4} | {r['dense_fit_ms']['median']:>12.4f} | "
                  f"{r['vecchia_fit_ms']['median']:>14.4f} | {r['fit_speedup']:>7.2f}x | "
                  f"{r['dense_quadratic_us']['median']:>13.2f} | {r['vecchia_quadratic_us']['median']:>15.2f} | "
                  f"{r['quadratic_speedup']:>11.2f}x", flush=True)
        print("  " + "-" * 108, flush=True)
        print(f"  Empirical exponent  dense: L^{result['empirical_exponents'][f'dense_n{n_samples}']:.2f}   "
              f"Vecchia: L^{result['empirical_exponents'][f'vecchia_n{n_samples}']:.2f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "results" / "benchmarks" / "vecchia_layer_horizon.json"))
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--threads", type=int, default=8,
                    help="BLAS/torch thread cap (read before numpy loads, see parse_bootstrap_flags)")
    ap.add_argument("--gpu-capture", action="store_true",
                    help="run the base-model activation capture on the GPU; estimators stay on CPU")
    ap.add_argument("--prompts-per-domain", type=int, default=48)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--surrogate-samples", type=int, default=1024)
    ap.add_argument("--no-real-activations", action="store_true",
                    help="skip the CPU forward pass and run the surrogate arm only")
    ap.add_argument("--cache", default=str(REPO_ROOT / "results" / "cache" / "real_layer_activations.npz"))
    args = ap.parse_args()

    print_header(
        "VECCHIA HORIZON WINDOWING -- CROSS-LAYER ACTIVATION PRECISION (Ch.12 §12.2.1)",
        "Banded GMRF over the residual stream: is the O(L m^2) win real, and is m=2 enough?",
    )

    payload: dict[str, Any] = {"benchmark": "vecchia_layer_horizon", "arms": {}}

    # --- Claim B, real arm ---------------------------------------------------
    lag1 = 0.3
    if not args.no_real_activations:
        print("\n[1/3] Capturing REAL per-layer activations (base-model forward)...", flush=True)
        device = "cuda:0" if args.gpu_capture else "cpu"
        feats = build_real_arm(args.prompts_per_domain, args.max_length, args.threads, Path(args.cache), device)
        print(f"  {feats['n_tokens']} token positions x {len(feats['layers'])} layers "
              f"from {feats['n_prompts']} prompts "
              f"({'cache' if feats.get('from_cache') else f'{feats.get('capture_seconds', 0):.1f}s forward'})",
              flush=True)
        real = horizon_sweep(
            as_matrix(feats, "layer_delta"),
            f"real activations (Qwen3.5-4B forward on {feats.get('device', 'cpu')})",
        )
        real["provenance"] = {
            k: feats[k] for k in ("source", "model_id", "dtype", "device", "n_prompts", "n_tokens")
            if k in feats
        }
        print_horizon(real)
        payload["arms"]["real"] = real
        lag1 = float(real["partial_correlation_by_lag"].get("1", 0.3))

    # --- Claim B, surrogate null control ------------------------------------
    print("\n[2/3] Surrogate control arm (real v7 factors, Gaussian inputs)...", flush=True)
    sur_feats = build_surrogate_arm(args.surrogate_samples)
    surrogate = horizon_sweep(as_matrix(sur_feats, "layer_delta"), "surrogate control (adapter factors)")
    surrogate["provenance"] = {k: sur_feats[k] for k in ("source", "expert", "d_model")}
    print_horizon(surrogate)
    payload["arms"]["surrogate"] = surrogate

    # --- Claim A, scaling ----------------------------------------------------
    print(f"\n[3/3] Depth scaling on a GMRF calibrated to the measured lag-1 pcorr ({lag1:.3f})...", flush=True)
    scaling = scaling_sweep(lag1=lag1, repeats=args.repeats)
    print_scaling(scaling)
    payload["scaling"] = scaling

    # --- Verdict -------------------------------------------------------------
    def _row(n_samples: int, L: int) -> dict[str, Any]:
        return next(r for r in scaling["rows"] if r["n_samples"] == n_samples and r["L"] == L)

    verdict: dict[str, Any] = {
        "fit_speedup_L80_n64": _row(64, 80)["fit_speedup"],
        "fit_speedup_L80_n1024": _row(1024, 80)["fit_speedup"],
        "fit_speedup_L512_n64": _row(64, 512)["fit_speedup"],
        "quadratic_speedup_L80_n64": _row(64, 80)["quadratic_speedup"],
        "exponents": scaling["empirical_exponents"],
    }
    if "real" in payload["arms"]:
        r = payload["arms"]["real"]
        by_m = {row["m"]: row for row in r["sweep"]}
        verdict["m2_gain_captured"] = by_m[2]["gain_captured"]
        verdict["m2_band_mass"] = by_m[2]["band_mass_of_dense"]
        verdict["saturating_m"] = r["saturating_m_within_0.05_nats"]
        verdict["m2_sufficient"] = bool(r["saturating_m_within_0.05_nats"] <= 2)
    payload["verdict"] = verdict

    print("\n" + "=" * 104, flush=True)
    print(" VERDICT", flush=True)
    print("=" * 104, flush=True)
    print(f"  Refit speedup at the 70B geometry (L=80, m=2): "
          f"{verdict['fit_speedup_L80_n64']:.2f}x at n=64, "
          f"{verdict['fit_speedup_L80_n1024']:.2f}x at n=1024.", flush=True)
    print(f"  At L=512 (past any real stack) it reaches {verdict['fit_speedup_L512_n64']:.2f}x -- "
          f"that is where O(L^3) starts to bite.", flush=True)
    if "m2_sufficient" in verdict:
        print(f"  m=2 captures {verdict['m2_gain_captured']*100:.1f}% of the dense model's held-out "
              f"likelihood gain and {verdict['m2_band_mass']*100:.1f}% of the precision mass.", flush=True)
        print(f"  Held-out NLL saturates (within 0.05 nats) at m={verdict['saturating_m']} -> "
              f"m=2 {'IS' if verdict['m2_sufficient'] else 'IS NOT'} sufficient on real activations.", flush=True)
    print("=" * 104, flush=True)

    write_telemetry(args.out, payload, extra_config=vars(args))


if __name__ == "__main__":
    main()
