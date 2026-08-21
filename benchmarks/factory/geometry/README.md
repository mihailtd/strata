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
| **[`poet_decomposition/`](poet_decomposition/)** | **🚀 Genuine Discovery** | **POET Kronecker + Sparse Coordinate Decomposition Autopsy**: Proved the Ambient Coordinate Sparsity Curse ($\mathcal{O}(\rho \cdot d^2)$ parameter explosion vs $\mathcal{O}(r \cdot d)$ low-rank factorization), demonstrating why low-rank parameterization beats coordinate thresholding. |
| **[`poet_activation_crosstalk/`](poet_activation_crosstalk/)** | **🚀 Genuine Discovery** | **POET Multi-Adapter Activation Cross-Talk Telemetry**: Decomposes pairwise activation covariance $\Sigma_{\text{cross}} = L_{\text{pervasive}} + S_{\text{sparse}}$, proving $10.5\%$ energy is shared foundation representation and isolating $<0.1\%$ sparse conflict channels to cut interference by $1.4\text{--}5.4\times$. |
| **[`latent_variable_glasso/`](latent_variable_glasso/)** | **🚀 Genuine Discovery** | **Latent Variable Graphical Lasso (LV-GLasso: Chapter 9)**: Decomposes full precision matrix $\widetilde{\Theta} = S - L$, resolving the §38 cosine overlap mystery. Proves attention heads are $100.0\%$ conditionally orthogonal ($S=0$), refutes global $\sqrt{K}$ attenuation (which destroys $74.6\%$ clean capability), and establishes Surgical Stacking. |
| **[`riemannian_metric/`](riemannian_metric/)** | **❌ Refuted** | **AIRM Geodesic Distance in a Shared Subspace (Ch 3 §3.5, Ch 8 §8.1.4)**: Correct, invariance-gated implementation — and the answer is negative. Adapter subspaces are fully disjoint ($k = 2r$) in all 128 weight matrices, the 6×6 matrix spans **0.8% of its mean**, and *the same domain trained twice is farther apart than two different domains*. Refutes §55, and shows §56's routing term is inert (0/4000 decisions changed). See `DECISIONS.md` §59. |


---

## Directory Organization
- **[`alpha_sweep/`](alpha_sweep/)**: The Mantissa ULP Inverse Scaling Law and floating-point noise floor characterization.
- **[`dynamic_alpha_calibration/`](dynamic_alpha_calibration/)**: Automated post-train $\alpha$ calibration bounded by precision floor and held-out capability retention.
- **[`preflight_svd_probe/`](preflight_svd_probe/)**: Instantaneous pre-flight SVD subspace overlap verification between adapter pairs.
- **[`times_above_chance/`](times_above_chance/)**: Bulk cross-task subspace orthogonality mapping across the factory fleet.
- **[`poet_decomposition/`](poet_decomposition/)**: Low-Rank + Sparse POET and Van Loan-Pitsianis Kronecker decomposition probes.
- **[`poet_activation_crosstalk/`](poet_activation_crosstalk/)**: Dynamic activation cross-talk covariance and channel notch filtering probes.
- **[`latent_variable_glasso/`](latent_variable_glasso/)**: Latent Variable Graphical Lasso precision graph conflict isolation probe.
- **[`riemannian_metric/`](riemannian_metric/)**: Riemannian/log-Euclidean geodesics between adapters, with the rank-basis invariance gate. Read the README before reusing $d_R$ for anything — it does not carry domain information.
