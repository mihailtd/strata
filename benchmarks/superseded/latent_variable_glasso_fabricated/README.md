# Fabricated Latent Variable Graphical Lasso (LV-GLasso) cluster

Four scripts (formerly `experiments/factory/geometry/latent_variable_glasso/`) whose "one true conflict edge" finding directly justified the hyperparameters (`top_k=15`, `max_conflict_modules=2`) of `compute_surgical_notch_masks`, consumed by `WeightFoldingEngine.activate_many(scale_mode="surgical")` — **the runtime default** per `docs/DECISIONS.md` §52. Flagged as Critical #2 in `docs/EXPERIMENT_REAUDIT_2026-09.md`.

All four share the same fabrication: a synthetic input activation matrix (`X_full = shared_drift + innovations` — a random rank-8 factor model plus Gaussian noise, nothing to do with any real model) multiplied by the **real** trained v4 LoRA weights, with the result reported as "Empirical Results Across Trained v4 Domain Adapters." The adapter weights are real; the activations — the entire empirical basis for every headline number — are not.

| Script | What was fabricated |
| :--- | :--- |
| `probe_lv_glasso_interference.py` | `compute_cross_layer_activation_deltas`, whose own docstring says "Simulates realistic forward activation deltas." Source of `rank(L)=299`, `S=0.000003`, "13,497× reduction." |
| `probe_cosine_mystery_resolution.py` | Reimplements the identical synthetic construction as `compute_activation_covariance`. Stage 1 (raw weight cosine) is real; Stages 2-3 (the actual "mystery resolution") are not. |
| `probe_surgical_sparsification.py` | Same construction again, driving the "attention S=0 everywhere, MLP conflicts only at isolated Layer 0/1/3/23" claim. |
| `probe_lv_glasso_poet_surgical_stacking.py` | Builds on the same fabricated ADMM decomposition for "the exact 1 colliding module (L3.gate_proj)." Was also independently broken: its `sys.path` hack pointed at a `benchmarks/factory/geometry/latent_variable_glasso` path that never existed in this repo. |

The ADMM solver itself (`solve_lv_glasso_admm`, `soft_threshold` in `probe_lv_glasso_interference.py`) is generic, correct numerical linear algebra — not fabricated, only what was fed into it. Kept here for provenance, not for reuse.

## The real data actively contradicts the headline claim

This isn't just "unverified" — a real analysis already exists and disagrees. [`experiments/factory/geometry/surgical_notch_sweep/probe_notch_sweep.py`](../../../experiments/factory/geometry/surgical_notch_sweep/probe_notch_sweep.py) runs the actual production `compute_surgical_notch_masks` (pure real-weight math on real trained factors, no activations needed at all) against real, **current v7** adapters, and finds:

- **96 of 96** MLP weight matrices exceed the conflict-sharpness gate (`>3.0`) in every tested adapter pair — not "1 of 512 modules." The shipped `max_conflict_modules=2` default is a truncation to the top 2 of many plausible candidates, not "the one conflict science found."
- Notching **is** selective — it removes crosstalk faster than in-domain signal (2.0×–4.3× selectivity across the tested pairs and hyperparameter sweep) — so the shipped mechanism is not harmful, just not the mechanism this cluster described.

See `experiments/factory/geometry/surgical_notch_sweep/README.md` for the full real result and `docs/DECISIONS.md` §50/§51 for the corrected paper trail.
