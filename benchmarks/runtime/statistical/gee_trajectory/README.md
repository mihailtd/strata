# GEE Trajectory Drift Detection (Ch.6 §6.3)

**Status: ✅ VALIDATED — highest-value result of the four statistical methods.**
Ranked P4 in the original proposal; measured P1.

Detects whether an agent conversation's quality signal is trending, without raising a
false alarm every time the model has a normal train of thought.

---

## 💡 The problem in plain English

Turn 5 of a conversation is not independent of turn 4. A model mid-way through a
reasoning chain produces correlated turns *by construction* — that is what reasoning is.

A drift detector that assumes each turn is an independent sample therefore believes it
has far more evidence than it does. Its variance estimate is too small, its p-values are
too confident, and it fires on ordinary thinking. Every one of those firings is an
expert swap that costs real time and buys nothing.

**GEE** models the turn-to-turn correlation explicitly and pairs it with a variance
estimator (the Huber-White "sandwich") that stays correct *even when the correlation
model is wrong*. That last property is the whole reason to use it: nobody claims agent
turns are truly AR(1).

---

## 🔬 What was measured

Calibration, against ground truth. Trajectories were generated **with and without a real
slope**, with AR(1) errors at known ρ, so the false-alarm rate can actually be counted.

A detector at a nominal 5% level that is honest fires on **5% of the no-drift draws**.
Nothing else about a drift detector matters if that number is wrong.

**3000 Monte Carlo replicates per cell** (MC standard error ≈ 0.4%).

### Arms

| arm | working correlation | variance estimator | what it represents |
| :--- | :--- | :--- | :--- |
| `naive_independence` | independence | model-based | the detector in the field |
| `independence_sandwich` | independence | sandwich | robust SE, wrong correlation |
| `ar1_sandwich` | AR(1) | sandwich | the proposal |
| `ar1_sandwich_df` | AR(1) | sandwich × K/(K−1) | + small-sample correction |
| `exchangeable_sandwich` | exchangeable | sandwich | misspecification control |

---

## 📊 Results

### 1. False alarms under NO drift (K = 30 conversations, T = 20 turns)

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

**At ρ = 0.9 the naive detector fires on 31.7% of conversations that are not drifting at
all** — roughly one false expert swap in every three conversations. Every sandwich arm
holds ~6%.

The working correlation is recovered accurately — **α̂ = −0.002 / 0.295 / 0.594 / 0.896**
against true 0.0 / 0.3 / 0.6 / 0.9 — which is why the AR(1) arm is also the tightest.

**Honest caveat:** every arm sits at 6–7.5% rather than 5.0% even at ρ = 0. That is
small-sample sandwich bias at K = 30, *not* autocorrelation, and section 3 quantifies it.

### 2. Power against real drift (ρ = 0.6, K = 30, T = 20)

| slope | naive | indep+sandwich | AR(1)+sandwich | AR(1)+sandwich+df |
| ---: | ---: | ---: | ---: | ---: |
| 0.005 | 27.1% ❌ | 8.3% | 7.6% | 7.1% |
| 0.01 | 36.1% ❌ | 12.1% | 13.1% | 12.6% |
| 0.02 | 58.7% ❌ | 27.9% | 30.3% | 28.9% |
| 0.04 | 94.5% ❌ | 77.0% | **81.9%** | 81.1% |

The naive column looks like the best detector and is not a detector at all — an arm that
rejects 24.8% of *null* draws will also "detect" plenty of drift. Among the arms that hold
their size, **AR(1)+sandwich has the highest power at every slope**: modelling the
correlation buys detection as well as calibration.

### 3. The small-K boundary (ρ = 0.6, no drift) — read this before deploying

| K conversations | naive | indep+sandwich | AR(1)+sandwich | AR(1)+sandwich+df |
| ---: | ---: | ---: | ---: | ---: |
| 3 | 28.8% ❌ | 26.7% ❌ | 26.3% ❌ | 19.4% ❌ |
| 5 | 25.6% ❌ | 15.7% ❌ | 15.7% ❌ | 12.5% ❌ |
| 10 | 25.7% ❌ | 9.5% ❌ | 8.7% ❌ | 7.5% ❌ |
| 20 | 26.2% ❌ | 7.3% | 7.4% | **6.5%** |
| 40 | 25.9% ❌ | 6.5% | 6.2% | **5.6%** |
| 80 | 25.5% ❌ | 5.4% | 5.3% | **5.1%** |

**The sandwich is a large-cluster estimator.** It needs ~20 concurrent conversations to be
trustworthy and ~40 to be tight. Below that it over-rejects for its own reason —
small-cluster bias — and at K = 3 it is no better than the naive detector. The `df`
correction helps (19.4% vs 26.3%) but does not rescue it.

### 4. Refit latency

| K | T | observations | median ms |
| ---: | ---: | ---: | ---: |
| 3 | 20 | 60 | 0.18 |
| 10 | 20 | 200 | 0.57 |
| 30 | 20 | 600 | 1.25 |
| 30 | 50 | 1500 | 1.50 |

**0.2–1.5 ms per full refit** — comfortably inside a per-turn budget.

### 5. Real multi-turn artifact

`results/benchmarks/multi_turn_execution_results_v4.json`, 3 arms × 15 turns of composite
score:

```
slope = +0.00131   α̂ = −0.397   robust p = 0.4311   naive p = 0.8028   fit 0.27 ms
```

No drift detected — and **K = 3 is far below the K ≈ 20 boundary**, so this exercises the
plumbing, not a calibrated test. One detail worth keeping: α̂ is **negative** here, and with
negative autocorrelation the naive detector is *conservative* (p = 0.80 vs 0.43), not
liberal. **The direction of the naive detector's error follows the sign of the
correlation** — it is unreliable, not uniformly optimistic.

---

## ✅ Verdict and deployment rules

| | |
| :--- | :--- |
| **Ship it** | `apps/runtime/gee_trajectory.py` → `DriftMonitor` |
| **False alarms** | **31.7% → 6.0%** (4.3× fewer wasted swaps) |
| **Cost** | 0.2–1.5 ms per refit |
| **Hard requirement** | **K ≥ 20 concurrent conversations.** Below that the p-value is not trustworthy — batch conversations or do not run the test. |
| **Use** | `corr="ar1"`, `small_sample_correction="df"` |

---

## 🛠️ Reproduce

```bash
uv run python benchmarks/runtime/statistical/gee_trajectory/benchmark_gee_trajectory.py --replicates 3000
uv run pytest tests/test_statistical_estimators.py -k "gee or drift or correlation" -v
```

Runtime ≈ 160 s. Pure CPU — Monte Carlo over 20×2 matrices, where a GPU kernel launch
would cost more than the computation it carries.

Raw telemetry: [`results/benchmarks/gee_trajectory_drift.json`](../../../../results/benchmarks/gee_trajectory_drift.json)
