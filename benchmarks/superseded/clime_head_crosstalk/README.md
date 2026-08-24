# CLIME Head Cross-Talk Inversion — RETIRED (Ch.4 §4.4.3)

**Retired 2026-08-24, on its own benchmark's evidence.** Ranked P2 in the original
proposal; measured P4 and closed out.

**Read this before proposing any per-token attention-head graph.** The retirement rests on
two independent findings, and the second one closes off the whole *application*, not just
this estimator.

> ⚠️ **Not retired for being wrong.** CLIME does exactly what it claims: its entrywise
> guarantee held on every feasible column at every λ tested. It is retired because a
> standard alternative beats it in the target regime, and because the signal it was meant
> to produce does not exist at the sample sizes the runtime has.

---

## What was proposed

At B = 1 decode (or speculative M = 2..4), the sample count `n` is far below the feature
dimension `d`, so the empirical covariance is rank-deficient and a direct inverse
manufactures dense spurious cross-talk between attention heads. CLIME replaces the
inversion with `d` decoupled ℓ₁ linear programs:

$$\min \|\theta_j\|_1 \quad \text{s.t.} \quad \|\hat\Sigma \theta_j - e_j\|_\infty \le \lambda$$

The proposal's three selling points were: a hard entrywise error bound, columns that
decouple onto parallel hardware, and a clean sparse head graph for routing.

---

## Finding 1 — the failure it avoids is real

Measured, not assumed. At n/d = 0.25 a direct inverse of the sample covariance reaches
entries of **5.8 × 10¹⁶** and is **never positive definite**. It is unusable, not merely
inaccurate. That part of the motivation is sound.

## Finding 2 — but graphical lasso wins the regime CLIME was proposed for

Edge recovery F1 against a known sparse precision matrix (d = 32, 36 true edges), at the λ
selected by extended BIC — **never using the ground truth to pick λ**:

```
┌───────┬────────┬───────────────┬───────────────┬───────────────┐
│   n   │  n/d   │     CLIME     │  graph lasso  │ naive inverse │
├───────┼────────┼───────────────┼───────────────┼───────────────┤
│     8 │  0.25  │  no PD soln   │     0.225     │  not PD       │
│    16 │  0.50  │     0.011     │   **0.380**   │  not PD       │
│    32 │  1.00  │     0.328     │   **0.491**   │  not PD       │
│    64 │  2.00  │     0.522     │   **0.637**   │     0.136     │
│   128 │  4.00  │   **0.922**   │     0.844     │     0.136     │
│   256 │  8.00  │     0.762     │     0.785     │     0.137     │
└───────┴────────┴───────────────┴───────────────┴───────────────┘
```

**At n/d ≤ 1 — the actual B = 1 decode case — graphical lasso wins every cell.** CLIME only
takes the lead at n/d ≥ 4, where the sample-starvation crisis it was proposed to solve has
already passed.

And it costs far more to get there: **~48 ms per d = 32 estimate against glasso's 1.3 ms —
about 35× slower.**

## Finding 3 — the real head graph is not reproducible at decode window sizes

**This is the finding that closes the application.** 128 real head features from 41,535 real
token positions (`Qwen3.5-4B` forward pass). Two *disjoint* windows, same conversation
pool, Jaccard overlap of the recovered edge sets:

| window (tokens) | n/d | edges found in A | edges found in B | **Jaccard overlap** |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.12 | 18 | 14 | **0.000** |
| 32 | 0.25 | 81 | 86 | **0.000** |
| 64 | 0.50 | 79 | 80 | **0.032** |
| 128 | 1.00 | 93 | 101 | **0.176** |
| 256 | 2.00 | 100 | 105 | **0.250** |
| 512 | 4.00 | 110 | 116 | **0.387** |
| 1024 | 8.00 | 133 | 126 | **0.472** |

Two disjoint 32-token windows each "find" ~80 head edges and share **zero** of them. A
router keyed on a per-step head graph would be routing on static. It takes ~512 tokens to
reach even half-agreement — far past any decode-step window.

**This is a property of the activations, not of CLIME.** Any estimator, including the glasso
that beat it, faces the same instability. That is why the whole application is closed, not
just this method.

## Finding 4 — the parallelism does not convert to wall clock

| workers | wall s | speedup | critical path | embarrassingly-parallel bound |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.245 | 1.00× | 5.70 ms | 43.0× |
| 2 | 0.128 | 1.91× | 12.35 ms | 20.5× |
| 4 | 0.160 | **1.53×** | 47.73 ms | 13.2× |

The columns *are* structurally independent. Python threads over HiGHS convert a small
fraction of that into wall clock, and **4 workers can be slower than 2**.

> This particular measurement is **load-sensitive** — it is wall clock under whatever else
> the machine is doing. Across five runs it landed between **1.53× and 2.41×**, against an
> embarrassingly-parallel bound that itself moved between 13× and 60× (the bound is
> total-column-time ÷ slowest-column-time, so contention inflates both terms). The
> conclusion is robust to that spread: a small single-digit fraction of a large bound. The
> structural claim is true; realising it would need processes, or a solver that releases
> the GIL properly.

## What did hold: the guarantee

For the record, since it is the one claim that survived intact (d = 32, n = 16):

| λ | feasible columns | ‖SΘ−I‖∞ | ≤ λ ? | sparsity |
| ---: | ---: | ---: | :---: | ---: |
| 0.05 | 0% | — | ✅ | 1.000 |
| 0.20 | 34% | 0.2000 | ✅ | 0.972 |
| 0.30 | 97% | 0.3000 | ✅ | 0.978 |
| 0.45 | 100% | 0.4500 | ✅ | 1.000 |

The bound holds **exactly** on every feasible column. The companion fact is the frontier:
below λ ≈ 0.2 the LP is *infeasible* at this sample size, and above λ ≈ 0.45 the estimate
collapses to the diagonal. The usable window is narrow and sample-size dependent.

---

## ⛔ Do not retry

* **Do not route or prune attention heads from a per-token or per-step head graph.** Finding
  3 is about the data, not the estimator. Nothing below ~512 tokens of context produces a
  reproducible graph.
* **Do not swap CLIME back in for glasso** at n/d ≤ 1 without new evidence — it lost on
  accuracy *and* cost, on ground truth, with λ chosen honestly by eBIC.
* **If you need a precision matrix over activations**, use
  `graphical_lasso_admm` in `src/runtime/clime_precision.py` (kept live — see below), or
  `ledoit_wolf_from_samples` in `src/runtime/riemannian_covariance.py`.

## What stayed live, and why

`src/runtime/clime_precision.py` was **not** moved here. It holds `graphical_lasso_admm` —
the estimator that *won* this comparison — plus the support/guarantee diagnostics
(`support_metrics`, `max_constraint_violation`, `extended_bic` usage) that any future
precision-matrix work will want. The CLIME entry points (`clime_column`,
`clime_precision`) remain in it as the losing arm of a documented comparison, and its
module docstring carries a pointer here.

Its unit tests stay live in `tests/test_statistical_estimators.py` — including
`test_naive_inverse_blows_up_when_n_is_below_d`, which pins Finding 1.

---

## 🛠️ Reproduce (for provenance only — do not cite these as live results)

```bash
uv run python benchmarks/superseded/clime_head_crosstalk/benchmark_clime_inversion.py --repeats 5
```

Raw telemetry: [`results/benchmarks/clime_precision_inversion.json`](../../../results/benchmarks/clime_precision_inversion.json)

Sibling benchmarks that were **validated**:
[`benchmarks/runtime/statistical/`](../../runtime/statistical/) — GEE drift detection (P1),
copula tail routing (P2), Vecchia horizon windowing (P3, at m = 8).

See `docs/DECISIONS.md` §66.
