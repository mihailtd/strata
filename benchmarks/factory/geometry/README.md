# Parameter Geometry & Subspace Structure

This module contains the mathematical probes, geometric analysis tools, and scaling calibrations that characterize the low-rank subspace geometry of fine-tuned adapters that are already integrated into the factory/runtime (Stage 3 — see `docs/METHODOLOGY.md`).

**Note (2026-09-12):** this directory only actually contains `alpha_sweep/` and `dynamic_alpha_calibration/` — the table below used to also list five clusters (`preflight_svd_probe`, `times_above_chance`, `poet_decomposition`, `poet_activation_crosstalk`, `riemannian_metric`) as if they lived here too. They don't and never moved here; they live under [`experiments/factory/geometry/`](../../../experiments/factory/geometry/) (Stage 1, pre-integration) — this was stale documentation left over from an earlier reorg. See that tree's own READMEs for those. `latent_variable_glasso/` also used to be listed here; it was retracted entirely (fabricated) — see below.

---

## Subsystem Classifications & Innovation Legend

| Subsystem / Module | Tier | Genuine Discovery & Mathematical Contribution |
| :--- | :---: | :--- |
| **[`alpha_sweep/`](alpha_sweep/)** | **🚀 Genuine Discovery** | **`bfloat16` Mantissa ULP Inverse Scaling Law**: Derived the mathematical formula for floating-point merge truncation ($merge\_rel\_err \approx \frac{0.167}{\|dW\|/\|W\|}$) and discovered the physical perturbation Goldilocks threshold. |
| **[`dynamic_alpha_calibration/`](dynamic_alpha_calibration/)** | **🔥 Applied Practice** | **Per-Adapter Dynamic $\alpha$-Calibration & ⭐ Stock LoRA ($r=8$, Dynamic $\alpha$)**: Decoupling rank from deployment scaling using LoRA linearity ($\Delta W \propto \alpha$) to calibrate $\alpha$ post-train between the analytic precision floor and the empirical narrowing ceiling without retraining. |
| ~~`latent_variable_glasso/`~~ | ~~🚀 Genuine Discovery~~ | **RETRACTED (2026-09-12):** was fabricated (synthetic activations × real weights), not a real discovery — see `docs/EXPERIMENT_REAUDIT_2026-09.md` Critical #2. Retired to [`benchmarks/superseded/latent_variable_glasso_fabricated/`](../../superseded/latent_variable_glasso_fabricated/). The real, non-fabricated answer to "does surgical notching work" is [`experiments/factory/geometry/surgical_notch_sweep/`](../../../experiments/factory/geometry/surgical_notch_sweep/) (96/96 modules conflict, not 1/512; real selectivity 2.5×–4.3×). |

---

## Directory Organization
- **[`alpha_sweep/`](alpha_sweep/)**: The Mantissa ULP Inverse Scaling Law and floating-point noise floor characterization.
- **[`dynamic_alpha_calibration/`](dynamic_alpha_calibration/)**: Automated post-train $\alpha$ calibration bounded by precision floor and held-out capability retention.
- ~~`latent_variable_glasso/`~~: retracted, see the table above — retired to `benchmarks/superseded/latent_variable_glasso_fabricated/`.

## Related pre-integration probes (Stage 1, live in `experiments/factory/geometry/` instead)

Not part of this directory — listed here only so a reader following an old link lands somewhere useful:

- [`preflight_svd_probe/`](../../../experiments/factory/geometry/preflight_svd_probe/): Pre-flight SVD subspace overlap probe — sub-second canonical angle computation across low-rank adapter pairs to predict cross-task additive composability prior to deployment.
- [`times_above_chance/`](../../../experiments/factory/geometry/times_above_chance/): Normalized Grassmannian floor metric — cross-domain adapters sit at ~1.10–1.28× chance overlap (statistically orthogonal).
- [`poet_decomposition/`](../../../experiments/factory/geometry/poet_decomposition/): POET Kronecker + sparse coordinate decomposition — why low-rank parameterization beats coordinate thresholding.
- [`poet_activation_crosstalk/`](../../../experiments/factory/geometry/poet_activation_crosstalk/): Multi-adapter activation cross-talk covariance decomposition and channel notch filtering.
- [`riemannian_metric/`](../../../experiments/factory/geometry/riemannian_metric/): **❌ Refuted** — AIRM geodesic distance between adapters. Correct, invariance-gated implementation, and the answer is negative: adapter subspaces are fully disjoint in all 128 weight matrices, and *the same domain trained twice is farther apart than two different domains*. Refutes `DECISIONS.md` §55; see §59. Read its README before reusing $d_R$ for anything — it does not carry domain information.
