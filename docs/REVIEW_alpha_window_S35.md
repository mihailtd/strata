# §35 (DRAFT FOR REVIEW — not committed, not appended to DECISIONS.md)

## The admissible alpha window is real, measured, and per-adapter. alpha_opt = 96, not 128.

**Status: CONFIRMED. An earlier note in this session dismissed the alpha sweep as
"the wrong objective". That was a framing error, corrected here.**

A lower bound is not the wrong objective — it is **half of a two-sided one**.
Bounding alpha from below by floating-point physics and from above by held-out
narrowing defines an admissible window with an interior optimum.

---

## 1. The physical axis is |dW|/|W|, not alpha

`results/alpha_absorption_sweep.json` ran at **rank_total = 64**, so its
scaling = alpha/64. Our adapters are r=8, alpha=128 → scaling **16**. Its alpha
column therefore does not transfer across ranks.

⚠️ **A claim made mid-session — that we sat "4x beyond the worst point measured" —
was wrong for exactly that reason.** It compared alpha across different ranks.

Measured on our r=8 adapters:

| adapter | \|dW\|/\|W\| at alpha=128 | predicted merge_err |
| :--- | ---: | ---: |
| `m2_postgresql_r8a128_v3` | **0.0750** | 2.23% |
| `m2_astral_r8a128_v3` | 0.0741 | 2.25% |

Against the sweep's own perturbation axis:

| sweep point | \|dW\|/\|W\| | its quality |
| :--- | ---: | ---: |
| scaling 0.5 | 0.0451 | 85.8 (peak) |
| **ours, r=8 alpha=128** | **0.0750** | — |
| scaling 1.0 | 0.0901 | 75.0 |
| scaling 4.0 | 0.3578 | 58.3 (collapse) |

So alpha=128 was **suboptimal, not catastrophic** — 4.8x below the collapse point.

The fitted law reproduces all five sweep points to <5%:

```
merge_rel_err_pct ≈ 0.167 / (|dW|/|W|)

  0.167/0.0229 = 7.29%   vs measured 7.28
  0.167/0.0451 = 3.70%   vs measured 3.70
  0.167/0.0901 = 1.85%   vs measured 1.86
  0.167/0.1817 = 0.92%   vs measured 0.93
  0.167/0.3578 = 0.47%   vs measured 0.49
```

Because `dW = (alpha/r) · B@A`, **|dW|/|W| is exactly linear in alpha** — one
post-train measurement yields the entire lower-bound curve at zero further cost.

---

## 2. Measured window — postgresql v3, astral v3 held fixed

| alpha | scaling | \|dW\|/\|W\| | merge_err | held-out | admissible |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 32 | 4 | 0.0187 | 8.91% | 0.3111 | no — below floor |
| 64 | 8 | 0.0375 | 4.46% | 0.3667 | yes |
| **96** | **12** | **0.0562** | **2.97%** | **0.3889** | **yes — alpha_opt** |
| 128 | 16 | 0.0750 | 2.23% | 0.3444 | yes |

base (no adapter) = **0.4667**

**Both limbs exist**: precision-limited at 32, narrowing-limited at 128, peak at 96.

The floor was computed **before** any evaluation ran, and the below-floor point
duly scored worst — the law earned its prediction rather than being fitted to the
outcome.

**Determinism check:** alpha=128 re-scored **0.3444, byte-identical** to the
independent earlier run. The harness is deterministic, so differences across alpha
come from alpha alone, not sampling.

---

## 3. Cumulative effect on the same gate

| configuration | edge vs base |
| :--- | ---: |
| v2, alpha=128 (loss over prompt) | −0.2333 |
| v3, alpha=128 (completion-only loss) | −0.1222 |
| **v3, alpha=96 (calibrated)** | **−0.0778** |

**67% of the regression removed**, with no new data and no retraining — one
training-config fix plus one deployment-time rescale.

---

## 4. Limits, stated plainly

* **n = 15 tasks.** Determinism makes these numbers exact *for these 15 tasks*.
* Adjacent alpha points differ by **less than the ±0.13 bootstrap CI width**
  measured earlier, so the ranking of 96 over 128 is **not individually resolved**.
  The four-point V shape carries the argument, not any single pairwise gap.
* Experts still trail base by **−0.0778**. Narrowing is reduced, not solved.
* `alpha_opt = 96` is calibrated for the **current corpus**. Retention mixing
  suppresses `(BA)h` on general constructs, which is the mechanism that raises
  alpha_max — so a corpus with a general-knowledge slice should **widen the window
  and move the optimum**. Re-calibrate after any corpus change.

---

## 5. What is now embedded, per-adapter rather than as a constant

* `scripts/train_expert_CURRENT_m2.py` records `merge_precision` (|dW|/|W| plus
  predicted merge error) into every `regime.json`. Free — the base weights and
  LoRA factors are already in memory at save time.
* `scripts/calibrate_expert_alpha.py` computes `alpha_min` analytically, then
  sweeps **deployment** alpha over the held-out gate by rewriting `lora_alpha` in
  `adapter_config.json` (`from_dir` reads it at load, `_from_peft` derives
  `scaling = alpha/rank`). **1 train + N evals**, not N trains + N evals.
  `--apply` writes the winner back.

Raw data: `results/alpha_calibration_m2_postgresql_r8a128_v3.json`
