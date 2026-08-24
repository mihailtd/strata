# Nonparametric Copula Tail Routing (Ch.3 §3.6)

**Status: ✅ VALIDATED — but ONLY in its calibrated form.**
Ranked P3 in the original proposal; measured P2. The raw textbook estimator is **rejected**;
see §"What does not work" below, because it fails in a way that is easy to miss.

Decides when a prompt needs **two** experts folded at once, from the extreme co-activation
structure between expert responses rather than from a linear similarity score.

---

## 💡 The problem in plain English

A cosine or dot-product router asks: *"do the Python expert and the Finance expert trend
together on average?"* They do — both fire on technical language. That is not the question.

The question is: *"do they spike together on the rare, weird prompt?"* — the portfolio-risk
calculation that needs custom Python vector math, the Postgres edge case that needs kernel
`epoll` knowledge. Those are **different questions with different answers**. Two variables
can be strongly correlated and never co-occur in their extremes; the Gaussian copula is
the textbook case, with **λ_U = 0 exactly** for every ρ < 1.

The **upper tail dependence coefficient** answers the second question:

$$\lambda_U = \lim_{u \to 1^-} P(U_1 > u \mid U_2 > u)$$

It depends only on the copula, so it is invariant to any monotone transform of the
marginals — log magnitudes, raw norms, softmax scores all give the same answer. The
Gaussian estimators in this family need their inputs Gaussianised first; this one does not.

---

## 🔬 What was measured

λ_U is a **limit**, estimated from the `k` largest observations only. Small `k` → low bias,
high variance; large `k` → the reverse. So it was swept against three copulas whose true
λ_U is known in closed form: **Gaussian (0 exactly)**, **Student-t**, and **Gumbel**.

---

## 📊 Results

### 1. Can λ_U be estimated at all? (n = 8192, 300 replicates)

```
┌─────────┬──────────────────┬──────────────────┬───────────────────┬────────────┐
│   k/n   │ gaussian (TRUE 0)│  t ν=4 (0.391)   │ gumbel θ=2 (0.586)│ separation │
├─────────┼──────────────────┼──────────────────┼───────────────────┼────────────┤
│  0.005  │      0.227       │      0.413       │      0.583        │    2.85 d  │
│  0.010  │      0.270       │      0.421       │      0.584        │    3.22 d  │
│  0.020  │      0.314       │      0.441       │      0.593        │    4.06 d  │
│  0.040  │      0.372       │      0.467       │      0.599        │    4.22 d  │
│  0.160  │      0.530       │      0.559       │      0.634        │    2.48 d  │
└─────────┴──────────────────┴──────────────────┴───────────────────┴────────────┘
```

**The gaussian column is the finding.** Its true λ_U is **0**. The estimator reports
0.23–0.53. Every one of those digits is finite-sample bias — and **the bias grows with ρ**,
so a merely-correlated pair scores *higher* than a genuinely tail-dependent pair with
weaker correlation.

Gumbel, by contrast, is recovered accurately (0.583–0.599 against a true 0.586) at small
k/n. Separation between tail-dependent and tail-independent peaks around **k/n ≈ 0.02–0.04**.

### 2. A cutoff that means something

Calibrated as the 95th percentile of λ̂_U under a **Gaussian copula matched to that pair's
own Pearson correlation**, at the same n and k:

| n | k | null cutoff | false positives at a naive cutoff of 0 | detect t(ν=4) | detect Gumbel | detect Gaussian |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 10 | 0.500 | 96% | 25% | 58% | 1% |
| 2048 | 41 | 0.415 | 100% | 52% | 100% | 4% |
| 8192 | 164 | 0.366 | 100% | **99%** | **100%** | 4% |

**An absolute cutoff such as "λ_U > 0" fires on 96–100% of tail-independent pairs.** The
calibrated cutoff holds its 5% false-positive rate and detects both alternatives at
n ≥ 2048.

### 3. Routing against ground truth

A stream built so the two dependence measures **must** disagree:

* `db + systems` — **tail dependent, near-zero linear correlation.** The pair that genuinely
  needs dual folding, and invisible to Pearson.
* `python + finance` — **correlated at ρ = 0.85, λ_U = 0 exactly.** The pair a correlation
  router co-activates for nothing. Every such fold is wasted work.

Marginals are pushed through `exp()` so magnitudes are heavily skewed — a rank-based
measure should not care, a moment-based one should.

```
┌──────────────────┬───────────┬─────────────┬────────────────┬─────────────┐
│      router      │ true pair │ distractor  │ ranks true 1st │ pAUC (≤5%)  │
├──────────────────┼───────────┼─────────────┼────────────────┼─────────────┤
│  pearson         │   0.227   │    0.839    │     False ❌   │    0.009    │
│  tail (raw)      │   0.297   │    0.438    │     False ❌   │    0.009    │
│  tail_calibrated │   0.219   │   −0.107    │     True  ✅   │    0.178    │
└──────────────────┴───────────┴─────────────┴────────────────┴─────────────┘
```

Recall on the cross-domain tokens, at matched wasted-fold budgets (a wasted fold costs
**0.94 ms**, the median expert fold latency measured in this repo's own multi-turn artifact):

| router | ≤0.5% waste | ≤1% | ≤2% | ≤5% |
| :--- | ---: | ---: | ---: | ---: |
| pearson | 0.0% | 0.0% | 0.0% | 28.5% |
| tail (raw) | 0.0% | 0.0% | 0.0% | 28.5% |
| **tail_calibrated** | **28.5%** | **28.5%** | **28.5%** | 28.5% |

**Read this row carefully: the calibrated router does not recall *more*. It recalls the
same 28.5% for a tenth of the wasted work.** Peak recall is capped by the `tail_quantile =
0.95` gate, not by the dependence measure — only 28.5% of cross-domain tokens push both
experts past their own 95th percentile in the first place. Raising recall means lowering
that gate, which is a separate (and untested) trade.

### 4. Cost

| | |
| ---: | :--- |
| routing decision | **5.27 µs/token** (0.0053 ms) — the "<0.1 ms" budget met with ~19× headroom |
| null calibration | ~137 ms, **one-time at fit**, not per token |

Identical latency across all three arms (5.22–5.27 µs): the dependence measure is chosen at
fit time, so it costs nothing at decode.

### 5. Real v7 experts, on REAL tokens

**34,465 real token positions**, 240 passages across all six domain corpora. Each expert is
scored by the **exact** response its adapter delta would produce on the real activation the
base model computed for that token — every one of the 128 targeted modules hooked at its
true input, not a Gaussian stand-in.

#### 5a. First: is the routing signal discriminative at all?

If experts cannot tell domains apart, their co-activation structure is moot. Mean
pseudo-observation per expert, by the domain the token came from (diagonal should dominate):

```
    token domain │   astral   duckdb  financial  postgres  py_modern  py_web │ argmax
  ───────────────┼───────────────────────────────────────────────────────────┼──────────
         astral  │   0.681    0.486     0.533     0.467      0.562    0.548  │ astral ✅
         duckdb  │   0.418    0.582     0.453     0.488      0.463    0.436  │ duckdb ✅
      financial  │   0.432    0.454     0.612     0.431      0.431    0.395  │ financial ✅
     postgresql  │   0.478    0.593     0.484     0.620      0.515    0.504  │ postgresql ✅
  python_modern  │   0.518    0.450     0.500     0.505      0.562    0.579  │ python_web ❌
     python_web  │   0.518    0.394     0.452     0.461      0.479    0.550  │ python_web ✅
```

**Domain argmax correct 5/6. Per-token top-1 accuracy 43.3% against a 16.7% chance
baseline — 2.6× chance.** The one miss is `python_modern` losing to `python_web`, which are
the pair deliberately split out of a shared corpus (DECISIONS §44), so confusing them is the
expected failure.

> ⚠️ **This only holds AFTER the rank transform.** On raw magnitudes the signal is *not*
> discriminative — per-expert scale dominates (mean magnitudes run 335–452 across experts),
> and the argmax collapses onto whichever expert happens to respond loudest overall
> (`python_web` wins 4 of 6 domains, `duckdb` the other 2). The rank transform removes that
> scale by construction. **Any router scoring experts by raw activation magnitude is
> reading expert loudness, not token content.**

#### 5b. The routing table

```
┌────────────────────────────────┬───────────┬──────────┬───────────┬──────────┬───────┐
│              pair              │ lambda_U  │ pearson  │  null 95% │  excess  │ above │
├────────────────────────────────┼───────────┼──────────┼───────────┼──────────┼───────┤
│       python_modern+python_web │   0.740   │  0.918   │   0.643   │  +0.097  │  YES  │
│        financial+python_modern │   0.435   │  0.706   │   0.342   │  +0.094  │  YES  │
│           financial+python_web │   0.403   │  0.679   │   0.320   │  +0.083  │  YES  │
│       postgresql+python_modern │   0.527   │  0.798   │   0.446   │  +0.080  │  YES  │
│              astral+python_web │   0.335   │  0.609   │   0.267   │  +0.069  │  YES  │
│           astral+python_modern │   0.318   │  0.595   │   0.250   │  +0.068  │  YES  │
│           financial+postgresql │   0.402   │  0.708   │   0.350   │  +0.052  │  YES  │
│          postgresql+python_web │   0.435   │  0.788   │   0.434   │  +0.001  │  YES  │
│              astral+postgresql │   0.226   │  0.572   │   0.237   │  −0.010  │   –   │
│              duckdb+postgresql │   0.459   │  0.819   │   0.472   │  −0.013  │   –   │
│               astral+financial │   0.254   │  0.639   │   0.290   │  −0.036  │   –   │
│               duckdb+financial │   0.277   │  0.668   │   0.313   │  −0.036  │   –   │
│                  astral+duckdb │   0.115   │  0.508   │   0.194   │  −0.079  │   –   │
│              duckdb+python_web │   0.340   │  0.774   │   0.422   │  −0.082  │   –   │
│           duckdb+python_modern │   0.372   │  0.821   │   0.478   │  −0.107  │   –   │
└────────────────────────────────┴───────────┴──────────┴───────────┴──────────┴───────┘
```

**8 of 15 pairs exceed their matched tail-independent null.**

#### 5c. Calibration reorders the table on real data

This is the section 3 result reproduced on production adapters rather than a constructed
stream. Ranked by **raw λ_U** the top three are `python_modern+python_web`,
`postgresql+python_modern`, `duckdb+postgresql`. Ranked by **calibrated excess** two of
those are displaced by the two `financial+python_*` pairs, whose raw λ_U is barely half as
large.

**`duckdb+postgresql` is the case worth remembering.** It has the second-highest Pearson
correlation of any pair (0.819) and the third-highest raw λ_U (0.459) — a correlation
router, or a raw-λ_U router, would co-fold it constantly. Against its own
correlation-matched null it lands at **−0.013: tail independent.** The two SQL experts move
together on average and *separate in the extreme* — they are **substitutes, not
complements**. You want one or the other, never both.

That is precisely the failure mode this method was proposed to fix, appearing in the real
adapter fleet.

#### 5d. What the surrogate said instead

For contrast, the Gaussian-input surrogate arm (`--no-gpu-capture` path) found **1 of 15**
pairs above null, all λ̂ ≈ 0.024–0.073. **It is not a usable substitute for the real capture**
— it understates both the magnitudes and the structure. It is kept only as a control that
the estimator reports nothing when there is nothing.

---

## ❌ What does NOT work

**Raw λ̂_U with a fixed threshold.** It is fooled by the correlated distractor exactly as
Pearson is — ranks the wrong pair first, and buys nothing (pAUC 0.009, identical to
Pearson to three decimals). If you take one thing from this benchmark: *the textbook
formula is not the shippable estimator.* The null subtraction is the method.

---

## ✅ Verdict and deployment rules

| | |
| :--- | :--- |
| **Ship it** | `src/runtime/copula_routing.py` → `CopulaTailRouter(pair_metric="tail_calibrated")` |
| **Never ship** | `pair_metric="tail"` (raw). It is a strictly worse Pearson. |
| **Cost** | 5.27 µs/token + 137 ms one-time calibration |
| **Requires** | n ≥ 2048 history samples, k/n ≈ 0.02, and a **rank transform** — raw magnitudes are not discriminative |
| **Buys** | same recall as Pearson at **1/10th** the wasted folds |
| **Live routing table** | 8 of 15 real expert pairs are tail dependent (§5b). **Never co-fold `duckdb+postgresql`** — highest correlation, tail *independent*: substitutes. |

---

## 🔭 Open work

**Closed 2026-08-24:** the real-token capture is done — see §5. It replaced "1 of 15 pairs,
weakly" with an 8-pair routing table on 34,465 real token positions.

Still open:

* **Recall is capped at 28.5% by the `tail_quantile = 0.95` gate**, not by the dependence
  measure. Lowering the gate trades recall against wasted folds and has not been swept.
* **The 8-pair table has not been A/B'd end-to-end.** It says which pairs co-spike; it does
  not yet show that co-folding those pairs improves answers. That needs a generation-quality
  run, not a statistics run.
* **`python_modern` vs `python_web` are not separable** by this signal (5/6 domain argmax,
  and that is the miss). If routing between those two matters, this is not the mechanism.

---

## 🛠️ Reproduce

```bash
# With the real-token expert scoring (GPU forward pass; ~65 s capture, then cached)
uv run python benchmarks/runtime/statistical/copula_routing/benchmark_copula_tail_routing.py \
    --replicates 300 --gpu-capture

# Surrogate-only, fully CPU-bound, safe while the GPU is training
uv run python benchmarks/runtime/statistical/copula_routing/benchmark_copula_tail_routing.py --replicates 300

uv run pytest tests/test_statistical_estimators.py -k "copula or tail or router or response" -v
```

Runtime ≈ 45 s CPU-only, ≈ 110 s on the first `--gpu-capture` run (cached afterwards).
Sections 1–4 are pure CPU in both modes; `--gpu-capture` moves only the base-model forward
pass in §5.

Raw telemetry: [`results/benchmarks/copula_tail_routing.json`](../../../../results/benchmarks/copula_tail_routing.json)
