# The Factory (Fine-Tuning & Adapter Architecture)

This directory serves as the centralized engine for all training, geometric calibration, and fine-tuning operations in the repository.

---

## Core Pillars & Innovations

| Area / Subsystem | Tier | Core Contribution |
| :--- | :---: | :--- |
| **[`geometry/alpha_sweep/`](geometry/alpha_sweep/)** | **🚀 Genuine Discovery** | **`bfloat16` Mantissa ULP Inverse Scaling Law**: Discovered the mathematical formula for floating-point merge truncation ($merge\_rel\_err \sim \frac{0.167}{|dW|/|W|}$) and calibrated the optimal $\alpha=128$ absorption threshold for lossless in-place folding. |
| **[`geometry/preflight_svd_probe/`](geometry/preflight_svd_probe/)** | **🚀 Genuine Discovery** | **Pre-Flight SVD Subspace Probe**: Sub-second canonical angle verification predicting additive multi-expert composability prior to deployment. |
| **[`geometry/times_above_chance/`](geometry/times_above_chance/)** | **🚀 Genuine Discovery** | **Times-Above-Chance Metric**: Solved the high-dimensional Grassmannian projection illusion ($E_{\text{chance}} = \frac{r}{D}$), proving cross-domain adapter orthogonality ($\sim 1.10\text{--}1.28\times$ chance). |
| **[`architecture_comparison/`](architecture_comparison/)** | **⭐ Standard** / **🔥 Applied** | **Stock LoRA vs. `id_kron` Controlled Head-to-Head**: Rigorous matched-scaling evaluation proving Stock LoRA remains stable across a 128x scaling range while `id_kron` diverges above scaling 2.0. |
| **[`m1_vs_m2_regime/`](m1_vs_m2_regime/)** | **⭐ Standard** / **🔥 Applied** | **M1 (4-bit NF4) vs. M2 (Native `bfloat16` + Liger) Audit**: Controlled head-to-head evaluation proving that training directly in native `bfloat16` eliminates the cross-precision quantization seam (+3.36 pp win). |
| **Training Pipelines** | **⭐ Industry Standard** | **Response-Only Completion Loss & Fused LoRA Backprop**: Unified SFT training script (`CURRENT_m2`) enforcing prompt masking (`label=-100`) and native `bfloat16` training on AMD ROCm. |

---

## Directory Organization
- **[`geometry/`](geometry/)**: Mathematical probes, subspace orthogonality heatmaps, and hyperparameter scale calibration.
- **[`architecture_comparison/`](architecture_comparison/)**: Controlled architecture benchmarks comparing fine-tuning micro-architectures across matched scaling regimes.
- **[`m1_vs_m2_regime/`](m1_vs_m2_regime/)**: Controlled A/B methodology benchmark comparing 4-bit NF4 training vs native `bfloat16` + fused backprop kernels.
