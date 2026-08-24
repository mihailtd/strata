# Statistical Estimators for the Activation & Routing Layers

CPU-side estimators over activation summaries, agent-trajectory signals, and expert
co-activation. Four methods from *Regressions in Covariances, Dependencies and Graphs*
(Pourahmadi & Arabpour), each benchmarked against the specific claim that motivated it.

**Three shipped, one retired.** Each active benchmark has its own README with full tables:
[vecchia_layer_horizon/](vecchia_layer_horizon/) · [gee_trajectory/](gee_trajectory/) ·
[copula_routing/](copula_routing/) · retired: [clime_head_crosstalk/](../../superseded/clime_head_crosstalk/)

Every number below was measured on this machine. Where the measurement contradicts the
claim, the contradiction is the headline.

---

## 💡 What these four things are, in plain English

The runtime constantly has to answer four statistical questions, and the naive answer to
each one is wrong in a specific, expensive way:

1. **"Which layers of the model move together?"** Answering it exactly means building and
   inverting a matrix over all 32–80 layers, every time you update. **Vecchia** says: a
   layer mostly only depends on the couple of layers right before it, so only track that
   narrow band and skip the rest of the matrix.

2. **"Which attention heads are actually talking to each other?"** During single-token
   decoding you have far fewer samples than you have heads, and the standard maths
   (inverting a covariance matrix) doesn't just get inaccurate — it explodes, inventing
   connections between unrelated heads. **CLIME** replaces the inversion with thousands of
   small optimisation problems that come with a hard error bound.

3. **"Is this conversation going off the rails?"** Turns inside one conversation are
   related to each other, but the obvious drift detector assumes they're independent. It
   therefore thinks it has far more evidence than it does, and cries wolf. **GEE** models
   the turn-to-turn correlation explicitly and uses a variance estimate that survives
   getting that model wrong.

4. **"Should we load TWO experts for this prompt?"** Ordinary similarity scores measure
   whether two experts trend together *on average*. What actually matters is whether they
   spike together on the rare, weird prompt. Those are different questions, and
   **copulas** answer the second one.

### What the benchmarks found

| Question | Does the proposed fix work? |
| :--- | :--- |
| Cross-layer tracking | **Partly.** The speedup is real but modest at real depths (1.8× at 80 layers, not the order of magnitude implied by O(L³)→O(L)). And the proposed 2-layer window is too narrow: it captures 82% of what the full model captures, but the curve is still climbing at 24 layers. |
| Head cross-talk | ⛔ **Retired.** The failure it avoids is real — direct inversion produces entries of 5.8×10¹⁶ and is never usable. But **graphical lasso beats CLIME** in exactly the sample-starved regime CLIME was proposed for and is ~35× faster — and the real head graph is *irreproducible* below ~512 tokens anyway. [Evidence →](../../superseded/clime_head_crosstalk/) |
| Drift detection | **Yes, decisively.** The naive detector raises false alarms **31.7%** of the time when it should raise them 5%. GEE brings that to ~6%. This is the biggest measured win of the four. |
| Dual-expert routing | **Only in its calibrated form.** The raw tail-dependence score is so biased it ranks a merely-correlated pair *above* the genuinely tail-dependent one. Calibrated against a matched null it ranks correctly, costs 5.2 µs/token, and yields a live 8-pair routing table on the real adapter fleet. |

---

## 1. §12.2.1 — Vecchia Horizon Windowing

**The claim.** Conditioning each layer on its `m` predecessors turns the O(L³) precision
estimate into a banded GMRF at O(L·m²), and `m = 2` suffices because residual
skip-connections make cross-layer dependence local.

**How it was tested.** 41,535 real token positions captured from a `Qwen3.5-4B` forward
pass over 288 passages spanning all six domain corpora, using the true per-layer residual
delta `‖h_{ℓ+1} − h_ℓ‖`. Held-out negative log-likelihood decides `m` — in-sample
likelihood falls monotonically with `m` by construction and would "prove" any answer.

### Is `m = 2` enough? (real activations, 24,921 train / 16,614 held out)

```
┌─────┬───────────────┬──────────────┬─────────────┬────────────┬─────────┬─────────┐
│  m  │ held-out NLL  │ gain vs m=0  │ KL to dense │ band mass  │ params  │ fit ms  │
├─────┼───────────────┼──────────────┼─────────────┼────────────┼─────────┼─────────┤
│  0  │     44.5440   │        0.0%  │    13.2169  │      0.0%  │     32  │  0.711  │
│  1  │     35.3876   │       77.7%  │     3.1398  │     35.7%  │     63  │  2.730  │
│  2  │     34.9232   │       81.6%  │     2.6056  │     43.0%  │     93  │  4.097  │
│  4  │     34.2792   │       87.1%  │     1.8680  │     58.3%  │    150  │  5.267  │
│  8  │     33.3905   │       94.6%  │     0.6972  │     74.0%  │    252  │  6.001  │
│ 16  │     32.9700   │       98.2%  │     0.2498  │     90.3%  │    408  │  9.711  │
│ 24  │     32.8191   │       99.5%  │     0.0764  │     98.0%  │    500  │ 11.739  │
│dense│     32.7586   │      100.0%  │     0.0000  │    100.0%  │    528  │         │
└─────┴───────────────┴──────────────┴─────────────┴────────────┴─────────┴─────────┘
```

Mean |partial correlation| by layer separation:
`lag1=0.324, lag2=0.093, lag3=0.060, lag4=0.137, lag5=0.060, lag6=0.057`

**Verdict on `m = 2`: not sufficient.** It captures 81.6% of the held-out likelihood gain
but only **43.0% of the precision mass** — the majority of conditional dependence lies
*outside* the band. The curve does not flatten (within 0.05 nats) until **m = 24**. The
dependence does decay sharply after lag 1, so the banding premise is directionally right;
it just does not decay to nothing. **`m = 8` is the defensible operating point**: 94.6% of
the gain, KL 0.70, 252 stored parameters against the dense 528.

The **lag-4 bump (0.137)** breaks an otherwise monotone decay and is unexplained — worth a
look, since 4 is also the attention-adapter stride in the v7 adapters, though this arm
measures the base model with no adapters loaded.

### Is the speedup real? (banded fit vs dense covariance + inverse, m = 2)

```
┌──────┬───────────────┬─────────────────┬───────────┬────────────────┬──────────────────┐
│  L   │ dense fit ms  │ Vecchia fit ms  │  speedup  │ dense quad µs  │ Vecchia quad µs  │
├──────┼───────────────┼─────────────────┼───────────┼────────────────┼──────────────────┤
│    8 │       0.0100  │         0.0396  │    0.25x  │         1.12   │          5.37 ⚠️ │
│   32 │       0.0221  │         0.0440  │    0.50x  │         1.18   │          5.43 ⚠️ │
│   64 │       0.0610  │         0.0489  │    1.25x  │         1.37   │          5.45 ⚠️ │
│   80 │       0.0954  │         0.0533  │    1.79x  │         1.54   │          5.49 ⚠️ │
│  128 │       0.2476  │         0.0615  │    4.03x  │         1.98   │          5.53 ⚠️ │
│  256 │       0.9065  │         0.0851  │   10.65x  │         4.87   │          5.73 ⚠️ │
│  512 │       5.4681  │         0.1345  │   40.66x  │        20.02   │          6.54 🔥 │
└──────┴───────────────┴─────────────────┴───────────┴────────────────┴──────────────────┘
```

Measured complexity exponents at an update window of n = 64 tokens: **dense L^1.54,
Vecchia L^0.28**. Not L³ and not L.

**Verdict on the speedup: real, but 1.8× at the depths that exist**, not the order of
magnitude the O(L³) framing implies. Two reasons, both measured: forming the covariance
costs O(n·L²) and dominates the O(L³) inverse until L ≳ 256; and below L = 64 a banded
numpy fit loses to one LAPACK call on a small dense matrix. **The apply side is a loss at
every real depth** — a dense Θ·y matvec beats the banded quadratic form until L = 512.

> The proposed "<0.4 ms in a single Triton fused pass" is met on **CPU** — 0.05 ms at
> L = 80 — but so is the dense baseline, at 0.10 ms. No Triton/GPU figure is claimed here:
> these are 32×80-element problems where a kernel launch costs more than the arithmetic.

**Null control.** A surrogate stream built from real v7 adapter factors driven by Gaussian
inputs shows flat partial correlations (~0.03 at every lag) and a dense fit that does
*not* beat the diagonal model on held-out data. The estimator finds no structure when
there is none, which is what makes the real arm's numbers meaningful.

Raw telemetry: [`results/benchmarks/vecchia_layer_horizon.json`](../../../results/benchmarks/vecchia_layer_horizon.json)

---

## 2. §4.4.3 — CLIME Sparse Precision Inversion → ⛔ **RETIRED**

**Moved to [`benchmarks/superseded/clime_head_crosstalk/`](../../superseded/clime_head_crosstalk/)
on 2026-08-24, on its own evidence.** `docs/DECISIONS.md` §66.

Not retired for being wrong — its entrywise guarantee `‖SΘ−I‖∞ ≤ λ` held **exactly** on
every feasible column at every λ tested. Retired because:

1. **Graphical lasso beats it at n/d ≤ 1** — the B = 1 decode regime it was proposed for —
   while being **~35× faster** (1.3 ms vs 48 ms at d = 32), with λ chosen honestly by
   extended BIC on both arms.
2. **The real head graph is not reproducible at decode window sizes.** Two disjoint
   32-token windows of real activations each find ~80 head edges and share **zero**
   (Jaccard 0.000). ~512 tokens are needed for even half-agreement.

Finding 2 is a property of the **activations**, not of CLIME, so it closes the whole
application: *do not route or prune attention heads from a per-token head graph, with any
method.* The winning estimator, `graphical_lasso_admm`, stays live in
[`src/runtime/clime_precision.py`](../../../src/runtime/clime_precision.py).

**Read [its README](../../superseded/clime_head_crosstalk/) before proposing any per-token
head graph.**

---

## 3. §6.3 — GEE Trajectory Drift Detection

**The claim.** Per-turn signals inside a conversation are autocorrelated, so an
independence-assuming drift detector has the wrong variance and the wrong false-alarm
rate. A working correlation plus a sandwich variance fixes the calibration.

**How it was tested.** Monte Carlo with ground truth: trajectories generated with and
without a real slope, AR(1) errors at known ρ. A detector that is honest at a nominal 5%
fires on 5% of the no-drift draws. That is the entire benchmark — nothing else about a
drift detector matters if this number is wrong.

### False alarms under NO drift (K = 30 conversations, T = 20 turns, 3000 replicates each)

An honest detector fires **5.0%** of the time. Monte Carlo standard error ≈ 0.4%.

```
┌──────────┬─────────────────────┬───────────────────────┬────────────────┬───────────────────┐
│ AR(1) ρ  │ naive independence  │ independence+sandwich │ AR(1)+sandwich │ AR(1)+sandwich+df │
├──────────┼─────────────────────┼───────────────────────┼────────────────┼───────────────────┤
│    0.0   │        6.1%         │         7.5%          │      7.4%      │       7.1%        │
│    0.3   │       13.0%  ❌     │         6.2%          │      6.2%      │       5.8%        │
│    0.6   │       24.8%  ❌     │         6.4%          │      6.2%      │       6.0%        │
│    0.9   │       31.7%  ❌     │         6.0%          │      6.0%      │       5.6%        │
└──────────┴─────────────────────┴───────────────────────┴────────────────┴───────────────────┘
```

**This is the headline of the whole suite.** At ρ = 0.9 the independence detector fires on
**31.7%** of conversations that are not drifting at all — one false expert swap in every
three conversations. The sandwich arms hold ~6% across every ρ. The working correlation is
recovered accurately (α̂ = −0.002 / 0.295 / 0.594 / 0.896 against true 0.0 / 0.3 / 0.6 /
0.9), which is why the AR(1) arm is also the tightest.

**The honest caveat:** every arm, including the sandwich ones, sits at 6–7.5% rather than
5.0% even at ρ = 0. That is small-sample sandwich bias at K = 30, not autocorrelation —
and it is precisely what the cluster sweep below quantifies.

### Power against real drift (ρ = 0.6, K = 30, T = 20)

| slope | naive independence | independence+sandwich | AR(1)+sandwich | AR(1)+sandwich+df |
| ---: | ---: | ---: | ---: | ---: |
| 0.005 | 27.1% ❌ | 8.3% | 7.6% | 7.1% |
| 0.01 | 36.1% ❌ | 12.1% | 13.1% | 12.6% |
| 0.02 | 58.7% ❌ | 27.9% | 30.3% | 28.9% |
| 0.04 | 94.5% ❌ | 77.0% | **81.9%** | 81.1% |

The naive column looks like the best detector and is not a detector at all — an arm that
rejects 24.8% of null draws will also "detect" a lot of drift. Among the arms that hold
their size, **AR(1)+sandwich has the highest power at every slope**, so modelling the
correlation buys detection as well as calibration.

### The small-K boundary (ρ = 0.6, no drift)

| K conversations | naive | independence+sandwich | AR(1)+sandwich | AR(1)+sandwich+df |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 28.8% ❌ | 26.7% ❌ | 26.3% ❌ | 19.4% ❌ |
| 5 | 25.6% ❌ | 15.7% ❌ | 15.7% ❌ | 12.5% ❌ |
| 10 | 25.7% ❌ | 9.5% ❌ | 8.7% ❌ | 7.5% ❌ |
| 20 | 26.2% ❌ | 7.3% ❌ | 7.4% ❌ | **6.5%** |
| 40 | 25.9% ❌ | 6.5% | 6.2% | **5.6%** |
| 80 | 25.5% ❌ | 5.4% | 5.3% | **5.1%** |

**The sandwich needs ~20 concurrent conversations to be trustworthy and ~40 to be tight.**
Below that it over-rejects for its own reason — small-cluster bias, not autocorrelation —
and at K = 3 it is no better than the naive detector. The `df` correction helps (19.4% vs
26.3%) but does not rescue it. **Do not run this detector on a handful of conversations
and believe the p-value.**

### Refit latency

| K | T | observations | median ms |
| ---: | ---: | ---: | ---: |
| 3 | 20 | 60 | 0.18 |
| 10 | 20 | 200 | 0.57 |
| 30 | 20 | 600 | 1.25 |
| 30 | 50 | 1500 | 1.50 |

A full refit costs **0.2–1.5 ms**, comfortably inside a per-turn budget.

### Real multi-turn artifact

`results/benchmarks/multi_turn_execution_results_v4.json` — 3 arms × 15 turns of composite
score:

```
slope = +0.00131   α̂ = −0.397   robust p = 0.4311   naive p = 0.8028   fit 0.27 ms
```

No drift detected, and **K = 3 is far below the K ≈ 20 boundary measured above**, so this
is a demonstration of the plumbing, not a calibrated test. One detail worth keeping: α̂ is
*negative* here, and with negative autocorrelation the naive detector is **conservative**
(p = 0.80 vs 0.43), not liberal. The direction of the naive detector's error follows the
sign of the correlation — it is unreliable, not uniformly optimistic.

Raw telemetry: [`results/benchmarks/gee_trajectory_drift.json`](../../../results/benchmarks/gee_trajectory_drift.json)

---

## 4. §3.6 — Nonparametric Copula Tail Routing

**The claim.** Dual-expert folding should trigger on upper tail dependence λ_U between
expert activations, not on linear similarity, because the prompts needing two experts are
exactly the extreme co-activations a correlation cannot see.

### Can λ_U be estimated at the sample sizes a router has? (n = 8192)

```
┌─────────┬──────────────────┬──────────────────┬──────────────────┬────────────┐
│   k/n   │ gaussian (TRUE 0)│  t ν=4 (0.391)   │ gumbel θ=2 (0.586)│ separation │
├─────────┼──────────────────┼──────────────────┼──────────────────┼────────────┤
│  0.005  │      0.227       │      0.413       │      0.583       │    2.85 d  │
│  0.010  │      0.270       │      0.421       │      0.584       │    3.22 d  │
│  0.020  │      0.314       │      0.441       │      0.593       │    4.06 d  │
│  0.040  │      0.372       │      0.467       │      0.599       │    4.22 d  │
│  0.160  │      0.530       │      0.559       │      0.634       │    2.48 d  │
└─────────┴──────────────────┴──────────────────┴──────────────────┴────────────┘
```

**The Gaussian column is the finding.** A Gaussian copula has λ_U = 0 **exactly**, for
every ρ < 1. The estimator reports 0.23–0.53. Every one of those digits is finite-sample
bias, and it grows with ρ — so a merely-correlated pair scores *higher* than a genuinely
tail-dependent one with weaker correlation. **An absolute cutoff such as "λ_U > 0" is not
implementable**: it fires on 96–100% of tail-independent pairs.

Gumbel, by contrast, is recovered accurately (0.583–0.599 against a true 0.586) at small
k/n. The estimator is usable — as a **score to be calibrated**, never as a number to be
compared to a fixed constant.

### Calibrated cutoff (95th percentile of a matched tail-independent null)

| n | k | null cutoff | false positives at cutoff 0 | detect t(ν=4) | detect Gumbel | detect Gaussian |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 10 | 0.500 | 96% | 25% | 58% | 1% |
| 2048 | 41 | 0.415 | 100% | 52% | 100% | 4% |
| 8192 | 164 | 0.366 | 100% | **99%** | **100%** | 4% |

### Routing against ground truth (a wasted fold costs 0.94 ms, measured in this repo)

A stream where `db + systems` are **tail dependent with near-zero correlation**, and
`python + finance` are **correlated at ρ = 0.85 with λ_U = 0 exactly**. Only the first
pair genuinely needs dual folding; every activation of the second is wasted work.

```
┌──────────────────┬───────────┬─────────────┬────────────────┬─────────────┐
│      router      │ true pair │ distractor  │ ranks true 1st │ pAUC (≤5%)  │
├──────────────────┼───────────┼─────────────┼────────────────┼─────────────┤
│  pearson         │   0.227   │    0.839    │     False ❌   │    0.009    │
│  tail (raw)      │   0.297   │    0.438    │     False ❌   │    0.009    │
│  tail_calibrated │   0.219   │   −0.107    │     True  ✅   │    0.178    │
└──────────────────┴───────────┴─────────────┴─────────────────┴─────────────┘
```

Recall on cross-domain tokens at matched wasted-fold budgets:

| router | ≤0.5% waste | ≤1% | ≤2% | ≤5% |
| :--- | ---: | ---: | ---: | ---: |
| pearson | 0.0% | 0.0% | 0.0% | 28.5% |
| tail (raw) | 0.0% | 0.0% | 0.0% | 28.5% |
| **tail_calibrated** | **28.5%** | **28.5%** | **28.5%** | 28.5% |

**Verdict: the method works, but only in its calibrated form.** Raw λ_U is fooled by the
correlated distractor exactly as Pearson is — it ranks the wrong pair first and buys
nothing (pAUC 0.009 for both). Subtracting a matched Gaussian-copula null flips the
ranking and reaches the same recall at **a tenth of the wasted-fold budget**.

**Cost:** 5.27 µs per token routing decision (0.0053 ms) — the proposed "<0.1 ms" budget is
met with ~19× headroom. Calibration is a one-time 138 ms fit cost, not a per-token cost.

### Real v7 experts, on real tokens

Each expert scored by the **exact** response its adapter delta produces on the real
activations of **34,465 real token positions** (240 passages, all six domains, every one of
the 128 targeted modules hooked at its true input).

**The routing signal is real: per-token top-1 accuracy 43.3% against a 16.7% chance
baseline, domain argmax correct 5/6** — but only *after* the rank transform. On raw
magnitudes per-expert scale dominates and the argmax collapses onto whichever expert is
loudest overall. Any router scoring by raw activation magnitude is reading expert loudness,
not token content.

**8 of 15 expert pairs are tail dependent** above their matched null. And calibration
reorders the table on real adapters exactly as it did on the constructed stream:
`duckdb+postgresql` has the **second-highest Pearson correlation (0.819)** and the
third-highest raw λ_U (0.459), yet lands at **−0.013 against its own null — tail
independent.** The two SQL experts move together on average and separate in the extreme:
**substitutes, not complements.** A correlation router would co-fold them constantly, for
nothing.

Full table and the routing-signal validation: [`copula_routing/README.md`](copula_routing/#5-real-v7-experts-on-real-tokens).

Raw telemetry: [`results/benchmarks/copula_tail_routing.json`](../../../results/benchmarks/copula_tail_routing.json)

---

## 📊 Priority ranking — proposed vs measured

The original proposal ranked these P1 Vecchia → P2 CLIME → P3 Copula → P4 GEE. The
measurements invert most of it:

| Method | Proposed | **Measured** | Why the change |
| :--- | :---: | :---: | :--- |
| **§6.3 GEE** | P4 | **P1** | Largest measured effect of the four: false alarms **31.7% → ~6%** at a 0.2–1.5 ms per-turn cost. Nothing else here changes an operational number by 5×. |
| **§3.6 Copula** | P3 | **P2** | Works in its calibrated form, at 5.3 µs/token — same recall as Pearson for **1/10th** the wasted folds. Cheapest thing on the list. |
| **§12.2.1 Vecchia** | P1 | **P3** | Speedup is real but **1.8× at L = 80**, not order-of-magnitude, and the apply side is a regression below L = 256. Ship it with **m = 8**, not m = 2. |
| **§4.4.3 CLIME** | P2 | ⛔ **retired** | Guarantee holds exactly, but **graphical lasso beats it at n/d ≤ 1** — the regime it was proposed for — while being ~35× faster. And the real head graph is unstable below ~512 tokens, so nothing downstream should key on it per-step. [Moved to superseded →](../../superseded/clime_head_crosstalk/) |

---

## 🛠️ How to reproduce

```bash
# All three active benchmarks, concurrently, GPU used only for the activation capture
uv run python benchmarks/runtime/statistical/run_all.py --gpu-capture

# Serially -- use this when the numbers you care about are TIMINGS.
# Four concurrent processes contend for cache and memory bandwidth; every latency
# figure in this README came from a --serial run for that reason.
uv run python benchmarks/runtime/statistical/run_all.py --gpu-capture --serial

# The retired CLIME benchmark, for provenance only -- not part of the active suite
uv run python benchmarks/superseded/clime_head_crosstalk/benchmark_clime_inversion.py --repeats 5

# Fully CPU-bound, safe to run while the GPU is training
uv run python benchmarks/runtime/statistical/run_all.py

# Individually
uv run python benchmarks/runtime/statistical/vecchia_layer_horizon/benchmark_vecchia_horizon.py
uv run python benchmarks/runtime/statistical/gee_trajectory/benchmark_gee_trajectory.py --replicates 3000
uv run python benchmarks/runtime/statistical/copula_routing/benchmark_copula_tail_routing.py --replicates 300

# Unit tests for the estimators themselves (41 tests, <1 s)
uv run pytest tests/test_statistical_estimators.py -v
```

### GPU policy

The estimators **never** touch the GPU, in any mode. They operate on 32×32 matrices and
linear programs, where a kernel launch costs more than the entire computation — and the
`--repeats` loops are serial Python. `--gpu-capture` moves exactly one thing onto the
device: the base-model forward pass that produces real activations (39 s for 41,535 token
positions on an RX 7900 XTX, against ~10 min on CPU).

Without `--gpu-capture`, `enforce_cpu_only()` clears `CUDA_VISIBLE_DEVICES`,
`HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES` before torch initialises, and
`assert_gpu_untouched()` fails the run if any HIP context was created. Every artifact
records which mode produced it.

---

## 📂 Files

| Path | Role |
| :--- | :--- |
| [`src/runtime/vecchia_precision.py`](../../../src/runtime/vecchia_precision.py) | Banded modified-Cholesky factor, batched fit, banded solve, horizon selection |
| [`src/runtime/clime_precision.py`](../../../src/runtime/clime_precision.py) | Graphical lasso ADMM (**live** — won the comparison), support/guarantee diagnostics. CLIME entry points kept as the documented losing arm. |
| [`src/runtime/gee_trajectory.py`](../../../src/runtime/gee_trajectory.py) | GEE with AR(1)/exchangeable working correlation, sandwich variance, `DriftMonitor` |
| [`src/runtime/copula_routing.py`](../../../src/runtime/copula_routing.py) | Empirical copula, λ_U estimators, reference copulas, `CopulaTailRouter` |
| [`src/runtime/activation_features.py`](../../../src/runtime/activation_features.py) | Real activation capture (CPU or GPU) + adapter-factor surrogate control |
| [`src/runtime/cpu_bench.py`](../../../src/runtime/cpu_bench.py) | Device guard, thread cap, timing statistics, telemetry stamping |
| [`tests/test_statistical_estimators.py`](../../../tests/test_statistical_estimators.py) | 41 correctness tests, including asserted-known biases |
