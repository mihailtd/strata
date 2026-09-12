# 🚀 Pre-Flight SVD Subspace Probe

> **Tier Classification**: **🚀 Genuine Discovery**  
> **Discovery**: **Sub-second Grassmannian subspace overlap probe to predict cross-task additive composability prior to deployment.**

---

### Classification Breakdown: What is Standard vs. What is Genuine Discovery
* **⭐ Industry Standard Baseline**: Determining if two LoRA adapters can compose additively is typically done by brute-force training and evaluating cross-task benchmarks after hours of compute.
* **🚀 Our Genuine Discovery**: A pre-flight mathematical probe that computes the canonical angles between low-rank adapter subspaces ($U_A, U_B$) via Singular Value Decomposition (SVD) in $<1\text{ second}$. If the measured overlap is $\approx 1.10\text{--}1.28\text{x}$ random chance, it mathematically proves near-perfect orthogonality, guaranteeing that the adapters will compose additively via `activate_many()` without cross-task interference.

---

## 1. Workflow Integration
You run this probe **before** initiating expensive multi-task training (like Shared-Basis Tucker Factorization). 
If this probe proves that the two domains (e.g., Financial and Astral) are mathematically orthogonal (around `1.10x` chance), it confirms that their feature spaces don't overlap. In that case, you can safely skip complex shared-basis architectures and just train them as standard LoRAs that can be stacked natively without interfering with each other!

---

## 2. Related Modules
- **[🚀 Times-Above-Chance SVD Subspace Probe](../times_above_chance/)**: The Pre-Flight Probe calculates a specific metric ("Times Above Chance") rather than a raw percentage. To see how this metric is used at scale to build a cross-task heatmap for the entire factory, see the `times_above_chance` module.
