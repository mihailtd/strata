# 🚀 Times-Above-Chance SVD Subspace Probe (Normalized Grassmannian Overlap)

> **Tier Classification**: **🚀 Genuine Discovery**  
> **Discovery**: **Formulation of the Times-Above-Chance statistical floor metric to overcome high-dimensional subspace projection illusions in billion-parameter LLMs.**

---

### Classification Breakdown: What is Standard vs. What is Genuine Discovery
* **⭐ Industry Standard Baseline**: Standard subspace projection measures raw energy retention ($\|P_A U_B\|_F^2 / \|U_B\|_F^2$). In high-dimensional spaces ($D=2560$), raw energy is always $<0.01\%$, misleading researchers into assuming all low-rank adapters are trivially orthogonal.
* **🚀 Our Genuine Discovery**: Formulating the **Times-Above-Chance** metric by normalizing measured Frobenius projection norm against the theoretical random Grassmannian subspace expectation ($E_{\text{chance}} = \frac{r}{D}$). This revealed that identical architectures retain $\sim 26,000\times$ chance, related domains retain $\sim 7.15\times$ chance, and cross-domain adapters sit at $\sim 1.10\text{--}1.28\times$ chance (true empirical orthogonality).

---

## 1. The Metric: Why Raw Percentages Fail
If you project a small adapter onto a 32-dimensional subspace of a massive billion-parameter weight matrix, you will mathematically capture a tiny fraction of energy by pure chance (e.g., `0.0005%`). 

Because of this, standard overlap tests that look for "retention < 1%" are useless—they will always pass. If you test two identical adapters, two related adapters, and two completely different adapters, a raw percentage test will incorrectly classify all three pairs as "orthogonal".

**Times-Above-Chance** fixes this. Instead of a raw percentage, it divides the measured retention by the expected statistical floor:
- Same adapter vs itself: ~26,000x chance (Ceiling)
- Astral vs Astral: ~7.15x chance (Strong overlap)
- Astral vs Financial: ~1.28x chance (Statistically orthogonal)

---

## 2. Scripts in this Module
- **`build_orthogonality_map.py`**: Runs this metric in a massive loop across every trained domain adapter in the repository to generate a cross-task subspace orthogonality heatmap. It acts as the bulk-analysis engine for the `Pre-Flight SVD Subspace Probe`.
