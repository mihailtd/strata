# Surgical notch sweep: is the shipped default surgery, or a rounding error?

Stage 1 experiment (real trained weights, no forward pass, no live server) that is nonetheless the **real anchor** for a question a different, fabricated cluster got wrong. See [`docs/EXPERIMENT_REAUDIT_2026-09.md`](../../../../docs/EXPERIMENT_REAUDIT_2026-09.md) Critical #2 and [`benchmarks/superseded/latent_variable_glasso_fabricated/`](../../../../benchmarks/superseded/latent_variable_glasso_fabricated/).

## Why this exists

`WeightFoldingEngine.activate_many(scale_mode="surgical")` is the runtime default (`docs/DECISIONS.md` §52), and its hyperparameters (`top_k=15`, `max_conflict_modules=2`) were justified by the Latent Variable Graphical Lasso (LV-GLasso) cluster's claim of "one true conflict edge" out of 512 modules. That claim turned out to be fabricated — driven by a synthetic input activation matrix multiplied by real adapter weights, not real activations.

This probe asks the practical question directly, using only real data: **does notching remove cross-adapter interference faster than it removes the in-domain signal it's supposed to protect?** If crosstalk and signal fall at the same rate, notching isn't surgery — it's just deleting neurons.

## Method

Everything is analytic in the real trained LoRA `(U, V)` factors — no dense `dW` is ever built, no forward pass, no synthetic data of any kind:

```
||dW_i||_F^2      = s_i^2 tr( (U_i^T U_i)(V_i V_i^T) )
<dW_i, dW_j>      = s_i s_j tr( (U_i^T U_j)(V_j V_i^T) )
row-wise crosstalk_r = s_i s_j * <row_r(U_i) V_i, row_r(U_j) V_j>
```

Masking multiplies rows of `U`, so every quantity above stays exact under a mask. The script directly calls the **production** `compute_surgical_notch_masks` (`apps/runtime/novel_peft.py` / `apps/runtime-ipwf/novel_peft.py`) — not a reimplementation — against the real, current `CANON.ADAPTER_VERSION` (v7) adapters, and sweeps `top_k` and `max_conflict_modules` around the shipped defaults.

## Result (real v7 adapters: astral, postgresql, duckdb, python_modern)

**All 96 of 96 MLP weight matrices exceed the conflict-sharpness gate (`>3.0`) in every tested pair** — not "1 of 512 modules" as the retired LV-GLasso cluster claimed:

| Pair | Modules passing gate | Sharpness (min / median / max) |
| :--- | :---: | :---: |
| astral + python_modern | 96 / 96 | 5.26 / 8.51 / 14.00 |
| postgresql + duckdb | 96 / 96 | 5.60 / 8.31 / 15.85 |
| astral + postgresql | 96 / 96 | 5.48 / 8.09 / 16.32 |

This directly contradicts LV-GLasso's "99.998% sparsity, one isolated collision" framing: real cross-adapter MLP crosstalk is widespread and modest, not sparse and localized. The shipped `max_conflict_modules=2` is a truncation to the top 2 of many roughly-equally-plausible candidates, not "the one conflict science found."

**Selectivity at the shipped default (`top_k=15, max_conflict_modules=2`) is real and positive:**

| Pair | Crosstalk removed | Signal removed | Selectivity |
| :--- | ---: | ---: | ---: |
| astral + python_modern | 0.042% | 0.015% | **2.83×** |
| postgresql + duckdb | 0.091% | 0.021% | **4.27×** |
| astral + postgresql | 0.045% | 0.018% | **2.52×** |

So: the mechanism is not harmful — it removes 2.5×–4.3× more crosstalk than signal, meaning it is doing *something* selective, not just deleting neurons at random. But the magnitudes are tiny either way (30 neurons zeroed out of 9,216, removing well under 0.1% of the stacked perturbation) — this is a small, real, positive effect, not the dramatic "one true conflict eliminated" story the fabricated cluster told.

The full sweep (`top_k ∈ {15, 50, 200, 1000}`, `max_conflict_modules ∈ {2, 8, 32, 128}`) shows selectivity degrades as either knob grows — the shipped defaults sit near the best end of the swept range, which is a point in their favor even though the original justification for picking exactly `15`/`2` was fabricated.

Full data: [`results/benchmarks/surgical_notch_sweep.json`](../../../../results/benchmarks/surgical_notch_sweep.json).

## What this does and doesn't answer

**Answers**: is the shipped notching mechanism selective (yes, modestly) and is "one isolated conflict module" a real description of the weight-space conflict structure (no — it's widespread and modest across all 96 MLP matrices).

**Doesn't answer**: what real per-token forward-pass *activation* covariance looks like across these adapters — this probe, like the fabricated cluster it replaces, never runs the model. That remains open if anyone wants to redo the LV-GLasso activation-covariance analysis properly (real forward hooks capturing real hidden states, the way `experiments/factory/geometry/activation_inertia_probe/probe_stacking_merit.py` does it, feeding the same reusable `solve_lv_glasso_admm` ADMM solver preserved in `benchmarks/superseded/latent_variable_glasso_fabricated/probe_lv_glasso_interference.py`).

## Run

```bash
CUDA_VISIBLE_DEVICES="" uv run python experiments/factory/geometry/surgical_notch_sweep/probe_notch_sweep.py
```

CPU-only, seconds to run. Saves to `results/benchmarks/surgical_notch_sweep.json`.
