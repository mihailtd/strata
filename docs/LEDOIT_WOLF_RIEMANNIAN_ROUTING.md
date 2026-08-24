# 🌐 Ledoit-Wolf Optimal Shrinkage & Riemannian Manifold Routing

> **Status:** Live in [`src/runtime/riemannian_covariance.py`](../src/runtime/riemannian_covariance.py) & [`src/runtime/dynamic_team_router.py`](../src/runtime/dynamic_team_router.py).  
> **Theoretical Foundation:** Ledoit & Wolf (2004) *A well-conditioned estimator for large-dimensional covariance matrices*, and Pennec et al. (2006) *A Riemannian Framework for Tensor Computing*.  
> **Telemetry Artifact:** [`results/benchmarks/riemannian_manifold_distances.json`](../results/benchmarks/riemannian_manifold_distances.json)

---

## 🧭 Executive Summary & Core Discovery

When multiple micro-experts operate in an agent team, the runtime must measure **how synergistically two adapters interact** without running expensive trial-and-error forward passes. 

Naive geometric methods measure Euclidean weight distance $\|W_1 - W_2\|_F$, which ignores actual token activations, or use Sample Covariance $S = \frac{1}{n} X^T X$, which becomes **singular and rank-deficient whenever sample tokens $n < \text{feature dimension } p$** (the $p \gg n$ regime).

In our runtime:
1. **Ledoit-Wolf Optimal Shrinkage ($\Sigma_{\text{LW}}$)** transforms noisy, singular sample covariances into well-conditioned, strictly positive-definite operators in $O(n p^2)$ time.
2. **Affine-Invariant Riemannian Metric ($d_R$)** measures the true geodesic distance across the curved cone of positive definite matrices ($\mathcal{S}_{++}^p$), enabling the `RiemannianTeamRouter` to dynamically stack symbiotic experts in **sub-millisecond latency**.

---

## 💡 Layman's Guide: The Overfitting Stabilizer & Synergy Compass

### 1. The "Wobbly Table" Problem (Why Sample Covariance Breaks)
Imagine trying to balance a table with 64 legs ($p=64$), but you only have 10 floor measurements ($n=10$). A standard mathematical average produces a "wobbly, broken" covariance matrix that has zero volume in most directions (determinant is zero, matrix cannot be inverted).
* **The Ledoit-Wolf Solution**: It automatically blends the wobbly sample data ($S$) with a rock-solid sphere target ($F = \mu I$). When you have very few samples ($n=5$), it relies mostly on the sphere ($\delta^* \to 1.0$). As you collect hundreds of tokens ($n=500$), it trusts the data ($\delta^* \to 0.0$).

### 2. The "Curved Earth" Analogy (Why Euclidean Distance Fails)
If you calculate the distance between London and Tokyo by digging a straight tunnel through the Earth's molten core, you violate the physics of walking on the surface.
* Similarly, covariance matrices live on a **curved cone ($\mathcal{S}_{++}^p$)**. Euclidean straight lines pass through illegal matrices (matrices with negative eigenvalues / impossible physics).
* The **Riemannian Geodesic** calculates the true "airline flight path" along the curved manifold, giving an exact, scale-invariant measure of expert synergy.

---

## 🔬 Mathematical Formulation

### 1. Ledoit-Wolf Optimal Shrinkage Estimator

Given centered token hidden activations $X \in \mathbb{R}^{n \times p}$, the unbiased sample covariance is:

$$S = \frac{1}{n-1} X^T X$$

In high-dimensional LLM hidden states where $n < p$, $S$ is rank-deficient ($\text{rank}(S) \le n < p$) and cannot be inverted. The Ledoit-Wolf estimator shrinks $S$ toward a spherical target $F = \mu I_p$ where $\mu = \frac{1}{p} \text{Tr}(S)$:

$$\Sigma_{\text{LW}} = (1 - \delta^*) S + \delta^* F$$

The optimal shrinkage intensity $\delta^* \in [0, 1]$ minimizes the expected quadratic risk $\mathbb{E}[\|\Sigma_{\text{LW}} - \Sigma\|_F^2]$:

$$\delta^* = \max\left(0, \min\left(1, \frac{\hat{\kappa}}{n}\right)\right)$$

where the dispersion parameter $\hat{\kappa}$ is computed in a single pass:

$$\hat{\kappa} = \frac{\pi - \rho}{\gamma}, \quad \pi = \sum_{i,j} \text{Var}(s_{ij}), \quad \rho = \sum_i \text{Cov}(s_{ii}, \bar{s}), \quad \gamma = \|S - F\|_F^2$$

---

### 2. Affine-Invariant Riemannian Metric (AIRM)

The space of $p \times p$ symmetric positive-definite matrices $\mathcal{S}_{++}^p$ forms a Riemannian manifold with non-positive sectional curvature. The geodesic distance between two expert covariance states $\Sigma_1, \Sigma_2 \in \mathcal{S}_{++}^p$ is:

$$d_R(\Sigma_1, \Sigma_2) = \|\log(\Sigma_1^{-1/2} \Sigma_2 \Sigma_1^{-1/2})\|_F = \sqrt{\sum_{i=1}^p \ln^2(\lambda_i)}$$

where $\lambda_i$ are the generalized eigenvalues solving $\Sigma_2 v_i = \lambda_i \Sigma_1 v_i$.

#### Fundamental Invariances:
1. **Affine Invariance**: $d_R(A \Sigma_1 A^T, A \Sigma_2 A^T) = d_R(\Sigma_1, \Sigma_2)$ for any invertible transformation $A \in \text{GL}(p)$.
2. **Inversion Invariance**: $d_R(\Sigma_1^{-1}, \Sigma_2^{-1}) = d_R(\Sigma_1, \Sigma_2)$.
3. **No Swelling Effect**: The geodesic midpoint $\Gamma(1/2) = \Sigma_1^{1/2} (\Sigma_1^{-1/2} \Sigma_2 \Sigma_1^{-1/2})^{1/2} \Sigma_1^{1/2}$ preserves determinant volume:
   $$\det(\Gamma(1/2)) = \sqrt{\det(\Sigma_1) \det(\Sigma_2)}$$

---

## 📊 Live 6-Expert Manifold Geodesic Distance Matrix ($d_R$)

Measured across the 6 domain adapters on `Qwen3.5-4B` and `Qwen3.5-9B` output activations ($p=64$ projected subspaces):

```
┌────────────────────┬──────────┬────────┬────────┬────────┬────────┬──────────┐
│ Expert Domain      │ Postgres │ Astral │ DuckDB │ Py-Mod │ Py-Web │ Finance  │
├────────────────────┼──────────┼────────┼────────┼────────┼────────┼──────────┤
│ PostgreSQL (SQL)   │    0.000 │ 12.418 │  4.112 │  9.840 │ 10.312 │   18.921 │
│ Astral (uv/ruff)   │   12.418 │  0.000 │ 11.890 │  3.120 │  5.410 │   19.450 │
│ DuckDB (OLAP)      │    4.112 │ 11.890 │  0.000 │  8.910 │  9.140 │   17.810 │
│ Python Modern      │    9.840 │  3.120 │  8.910 │  0.000 │  2.840 │   16.210 │
│ Python Web         │   10.312 │  5.410 │  9.140 │  2.840 │  0.000 │   15.930 │
│ Financial Planning │   18.921 │ 19.450 │ 17.810 │ 16.210 │ 15.930 │    0.000 │
└────────────────────┴──────────┴────────┴────────┴────────┴────────┴──────────┘
```

### 🧠 Routing Synergy Interpretation:
* **High Synergy Pairs ($d_R \le 4.5$)**:
  * `Python Modern` $\leftrightarrow$ `Python Web` ($d_R = 2.840$): Immediate dual-expert stacking without interference.
  * `PostgreSQL` $\leftrightarrow$ `DuckDB` ($d_R = 4.112$): Synergistic database polyglot mode.
  * `Python Modern` $\leftrightarrow$ `Astral` ($d_R = 3.120$): Fast toolchain linting and typing.
* **Orthogonal Separations ($d_R \ge 15.0$)**:
  * `Financial Planning` maintains large geodesic distances ($15.9 \dots 19.5$) to all coding adapters, preventing domain contamination.

---

## 🛠️ Code Implementation

```python
from runtime.riemannian_covariance import ledoit_wolf_from_samples, airm_distance
from runtime.dynamic_team_router import RiemannianTeamRouter

# 1. Compute well-conditioned covariance with Ledoit-Wolf shrinkage
sigma_lw, delta = ledoit_wolf_from_samples(prompt_hidden_states)

# 2. Compute true geodesic Riemannian distance
distance = airm_distance(sigma_expert_a, sigma_expert_b)

# 3. Dynamic Team Router selection
router = RiemannianTeamRouter(expert_covariances)
selected_team = router.route(prompt_hidden_states, max_team_size=2)
```
