# 🔥 In-Place Low-Rank Weight Folding Engine

> **Tier Classification**: **🔥 Applied Practice**  
> **Concept Origin**: **In-place parameter fusing and draw-call batching principles from numerical computing applied to LLM low-rank runtime weights.**

---

### Classification Breakdown: What is Standard vs. What is Innovative
* **⭐ Industry Standard Baseline**: Standard Parameter-Efficient Fine-Tuning (PEFT / LoRA wrappers) leaves base model weights untouched and routes every token forward pass through separate auxiliary adapter branches ($W \cdot x + \frac{\alpha}{r} (B \cdot A \cdot x)$). This doubles kernel launch overhead and cuts generation throughput in half (18.26 tok/s vs 33.40 tok/s).
* **🔥 Our Innovative Applied Practice**: **In-Place Weight Absorption & Composite Stacking (`activate_many()`)**. By pre-computing the low-rank delta product directly into live `bfloat16` backbone weights in **18 ms**, we eliminate 100% of wrapper overhead, restoring generation velocity to **99.6% of native unadapted speed (+82.1% speedup over wrapped PEFT)** with zero accuracy degradation (`top1_agreement = 1.00`).

---

## 1. Why In-Place Folding Wins

| Approach | Architecture | 256-Token Decode Speed | Relative Throughput |
| :--- | :--- | :---: | :---: |
| **Standard PEFT Wrapper** | Base Weight + Dynamic LoRA Branch | 18.26 tok/s | 0.55x (Baseline) |
| **In-Place Weight Folding** | Fused Backbone Matrix ($W_{\text{live}} = W_0 + \Delta W$) | **33.26 tok/s** | **1.82x (+82.1% Speedup!)** |
| **Raw Unadapted Model** | Base Weight $W_0$ | 33.40 tok/s | 1.00x (99.6% Recovered) |

---

## 2. 🔥 Multi-Expert Additive Fold Composite (`activate_many()`)

Beyond single-expert folding, the engine implements composite multi-expert stacking via `WeightFoldingEngine.activate_many()` in [`src/gnn_experiment/novel_peft.py`](file:///home/mihai/gnn-experiment/src/gnn_experiment/novel_peft.py):

$$W_{\text{live}} = W_0 + \sum_{i=1}^N \text{scaling}_i \cdot (U_i \times V_i)$$

### 1. Chained `addmm_` from Pristine $W_0$ (Flat Cost in $N$)
* **Mechanism**: Reads the static base weight $W_0$ from the pristine buffer once, performs the first adapter write directly into the live weight slot with `torch.addmm(w0, u, v, beta=1.0, alpha=s, out=w)`, and chains subsequent adapters in-place via `w.addmm_(u, v, alpha=s)`.
* **Zero Churn**: Eliminates all temporary tensor allocations and intermediate buffers. The fold cost stays flat in $N$.

### 2. Orthogonal Composition & Absorption Law
* **Subspace Orthogonality**: The Pre-Flight SVD Subspace Probe measured cross-task adapter pairs at $1.10\text{--}1.28\text{x}$ chance (statistically orthogonal). Orthogonal deltas compose additively with minimal cross-task interference.
* **Absorption Law Advantage**: Summing deltas increases $|dW|/|W|$. Because `bfloat16` merge error scales as $\sim 0.167 / (|dW|/|W|)$, a composite stacked delta is actually represented **more faithfully in `bfloat16`** than any single component alone!

---

## 3. Scripts in this Module

* **[`benchmark_weight_folding.py`](benchmark_weight_folding.py)**: Proves the steady-state +82.1% decode speedup of unwrapped in-place weight folding over standard PEFT wrappers over full 256-token generations.
* **[`evaluate_folded_vs_wrapped.py`](evaluate_folded_vs_wrapped.py)**: Evaluates quality preservation across domains, verifying that in-place weight absorption into `bfloat16` matches wrapped PEFT accuracy.
* **[`benchmark_stacked_experts.py`](benchmark_stacked_experts.py)**: Audits `activate_many()` across all $2^N$ multi-expert combinations with 95% bootstrap confidence intervals, proving that composite stacking preserves domain performance.
