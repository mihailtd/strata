# 🏭 Pillar 1: The Factory (Fine-Tuning, Geometry & Adaptation)

```
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║                          THE FACTORY ARCHITECTURE & CALIBRATION                              ║
║                                                                                              ║
║  1. Geometric Stopping      : Automatically stops training when loss gradient flatlines      ║
║  2. Ledoit-Wolf Shrinkage   : Optimal covariance conditioning for Riemannian routing        ║
║  3. Subspace Orthogonality  : Proves LoRA adapters don't overwrite base model knowledge     ║
║  4. Outlier-Protected SSI   : Closed-form stress-strength clipping for low-bit quantization  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
```

## 🎯 High-Level Overview (For Humans / Layman's Terms)
When teaching a large AI new specialist skills (like advanced PostgreSQL query optimization or DuckDB analytics), standard fine-tuning is like giving the model brain surgery with a sledgehammer: it learns the new skill, but suffers "catastrophic forgetting" and loses general coding ability.

**The Factory is our precision surgical training facility:**
1. **Micro-Specialist Adapters (LoRA):** We freeze 100% of the base model's brain cells and train tiny, specialized "backpacks" (adapters) containing only the new domain syntax.
2. **Subspace Orthogonality:** We mathematically prove that each adapter operates in a completely different mathematical dimension, meaning they never collide or interfere with each other.
3. **Geometric Stopping:** Instead of guessing how many epochs to train, we monitor the geometric curvature of the loss manifold, stopping training at the exact microsecond the adapter reaches peak mastery.

---

## 🔬 Sub-Benchmark Suite

| Directory | Sub-Benchmark | Key Metric | Layman's Analogy |
| :--- | :--- | :--- | :--- |
| [`geometry/riemannian_metric/`](./geometry/riemannian_metric/README.md) | **Riemannian AIRM Metric & Ledoit-Wolf** | 6x6 Geodesic Distance Matrix | Measuring the true curved flight distance between cities on a globe rather than a flat map. |
| [`geometry/dynamic_alpha_calibration/`](./geometry/dynamic_alpha_calibration/) | **Dynamic Alpha Scaling Law** | Lossless $\alpha=128$ Absorption | Finding the exact volume setting where a microphone captures clear sound without static noise. |
| [`geometry/preflight_svd_probe/`](./geometry/preflight_svd_probe/PISSA_ASSESSMENT.md) | **Pre-Flight SVD Subspace Probe** | Sub-Second Angle Verification | Running a pre-flight checklist on an airplane before takeoff to guarantee stability. |
| [`geometry/times_above_chance/`](./geometry/times_above_chance/) | **Times-Above-Chance Orthogonality** | $1.10\text{--}1.28\times$ Chance Ratio | Verifying that different specialists speak in completely distinct radio frequencies. |
| [`quantization/`](./quantization/README.md) | **Leverage Outlier & SSI Quantization** | Closed-Form Optimal Clipping | Tailoring custom-fit suits for numbers so none of them get squeezed or distorted. |
| [`agentic/poet_tool_causal_graph/`](./agentic/poet_tool_causal_graph/README.md) | **POET Causal DAG Transitions** | NOTEARS Continuous Acyclicity | Drawing a clean, non-circular roadmap of which specialist tool to call next. |

---

## 📐 Mathematical Formalism

### 1. Ledoit-Wolf Covariance Shrinkage
Sample covariance $S = \frac{1}{N} X^T X$ is regularized toward the spherical identity target $F = \frac{\text{Tr}(S)}{p} I_p$:
$$\Sigma^* = (1 - \lambda^*) S + \lambda^* F$$
Where optimal shrinkage intensity $\lambda^* \in [0, 1]$ is computed in closed form:
$$\lambda^* = \frac{\sum_{i \neq j} \text{Var}(s_{ij})}{\sum_{i \neq j} s_{ij}^2 + (s_{ii} - \bar{s})^2}$$

### 2. Riemannian Affine-Invariant Distance (AIRM)
For two symmetric positive-definite covariance matrices $\Sigma_1, \Sigma_2 \in \mathcal{S}_+^p$:
$$\delta_{\text{AIRM}}(\Sigma_1, \Sigma_2) = \|\log(\Sigma_1^{-1/2} \Sigma_2 \Sigma_1^{-1/2})\|_F = \sqrt{\sum_{i=1}^p \ln^2(\lambda_i)}$$
Where $\lambda_i$ are the generalized eigenvalues of $(\Sigma_1, \Sigma_2)$.
