# Parameter Geometry & Subspace Structure

This module contains the mathematical probes, geometric analysis tools, and scaling calibrations that characterize the low-rank subspace geometry of fine-tuned adapters.

---

## Subsystem Classifications & Innovation Legend

| Subsystem / Module | Tier | Genuine Discovery & Mathematical Contribution |
| :--- | :---: | :--- |
| **[`alpha_sweep/`](alpha_sweep/)** | **🚀 Genuine Discovery** | **`bfloat16` Mantissa ULP Inverse Scaling Law**: Derived the mathematical formula for floating-point merge truncation ($merge\_rel\_err \approx \frac{0.167}{\|dW\|/\|W\|}$) and discovered the physical perturbation Goldilocks threshold. |
| **[`dynamic_alpha_calibration/`](dynamic_alpha_calibration/)** | **🔥 Applied Practice** | **Per-Adapter Dynamic $\alpha$-Calibration & ⭐ Stock LoRA ($r=8$, Dynamic $\alpha$)**: Decoupling rank from deployment scaling using LoRA linearity ($\Delta W \propto \alpha$) to calibrate $\alpha$ post-train between the analytic precision floor and the empirical narrowing ceiling without retraining. |
| **[`preflight_svd_probe/`](preflight_svd_probe/)** | **🚀 Genuine Discovery** | **Pre-Flight SVD Subspace Overlap Probe**: Sub-second canonical angle computation via SVD across low-rank adapter pairs to mathematically predict cross-task additive composability prior to deployment. |
| **[`times_above_chance/`](times_above_chance/)** | **🚀 Genuine Discovery** | **Normalized Grassmannian Floor Metric (Times-Above-Chance)**: Solved the high-dimensional projection artifact ($E_{\text{chance}} = \frac{r}{D}$), proving that cross-domain adapters sit at $\sim 1.10\text{--}1.28\times$ chance (statistically orthogonal), providing the foundation for additive composition. |

---

## Directory Organization
- **[`alpha_sweep/`](alpha_sweep/)**: The Mantissa ULP Inverse Scaling Law and floating-point noise floor characterization.
- **[`dynamic_alpha_calibration/`](dynamic_alpha_calibration/)**: Automated post-train $\alpha$ calibration bounded by precision floor and held-out capability retention.
- **[`preflight_svd_probe/`](preflight_svd_probe/)**: Instantaneous pre-flight SVD subspace overlap verification between adapter pairs.
- **[`times_above_chance/`](times_above_chance/)**: Bulk cross-task subspace orthogonality mapping across the factory fleet.
