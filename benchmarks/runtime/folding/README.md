# 🔥 In-Place Low-Rank Weight Folding Engine

> **Tier Classification**: **🔥 Applied Practice**  
> **Concept Origin**: **In-place parameter fusing and draw-call batching principles from numerical computing applied to LLM low-rank runtime weights.**

---

### Classification Breakdown: What is Standard vs. What is Innovative
* **⭐ Industry Standard Baseline**: Standard Parameter-Efficient Fine-Tuning (PEFT / LoRA wrappers) leaves base model weights untouched and routes every token forward pass through separate auxiliary adapter branches ($W \cdot x + \frac{\alpha}{r} (B \cdot A \cdot x)$). This doubles kernel launch overhead and cuts generation throughput in half (18.26 tok/s vs 33.40 tok/s).
* **🔥 Our Innovative Applied Practice**: **In-Place Weight Absorption & Composite Stacking (`activate_many()`) in the Goldilocks Operating Window**. By pre-computing the low-rank delta product directly into live `bfloat16` backbone weights in **17.6 ms**, we eliminate 100% of wrapper overhead, restoring generation velocity to **99.6% of native unadapted speed (+82.1% speedup over wrapped PEFT)** with zero accuracy degradation (`top1_agreement = 1.00`) and zero numerical drift ($L_\infty = 0.00$).

---

## 1. Why In-Place Folding Wins

| Approach | Architecture | 256-Token Decode Speed | Relative Throughput |
| :--- | :--- | :---: | :---: |
| **Standard PEFT Wrapper** | Base Weight + Dynamic LoRA Branch | 18.26 tok/s | 0.55x (Baseline) |
| **In-Place Weight Folding** | Fused Backbone Matrix ($W_{\text{live}} = W_0 + \Delta W$) | **33.26 tok/s** | **1.82x (+82.1% Speedup!)** |
| **Raw Unadapted Model** | Base Weight $W_0$ | 33.40 tok/s | 1.00x (99.6% Recovered) |

---

## 2. Submodules & Capabilities

* **[`goldilocks_in_place_addmm/`](goldilocks_in_place_addmm/)**: Guaranteed lossless in-place `addmm()` execution within the calibrated Goldilocks operating window ($\alpha_{\min} \le \alpha \le \alpha_{\max}$), proving pointer stability and 0.00e+00 drift.
* **[`midstream_swap/`](midstream_swap/)**: Dynamic mid-generation expert hot-swapping without KV cache clearing or prefix re-prefill.

---

## 3. 🔥 Multi-Expert Additive Fold Composite (`activate_many()`)

Beyond single-expert folding, the engine implements composite multi-expert stacking via `WeightFoldingEngine.activate_many()` in [`src/gnn_experiment/novel_peft.py`](file:///home/mihai/gnn-experiment/src/gnn_experiment/novel_peft.py):

$$W_{\text{live}} = W_0 + \sum_{i=1}^N \text{scaling}_i \cdot (U_i \times V_i)$$

### 1. Chained `addmm_` from Pristine $W_0$ (Flat Cost in $N$)
* **Mechanism**: Reads the static base weight $W_0$ from the pristine buffer once, performs the first adapter write directly into the live weight slot with `torch.addmm(w0, u, v, beta=1.0, alpha=s, out=w)`, and chains subsequent adapters in-place via `w.addmm_(u, v, alpha=s)`.
* **Zero Churn**: Eliminates all temporary tensor allocations and intermediate buffers. The fold cost stays flat in $N$.

### 2. Orthogonal Composition & Absorption Law
* **Subspace Orthogonality**: The Pre-Flight SVD Subspace Probe measured cross-task adapter pairs at $1.10\text{--}1.28\text{x}$ chance (statistically orthogonal). Orthogonal deltas compose additively with minimal cross-task interference.
* **Absorption Law Advantage**: Summing deltas increases $|dW|/|W|$. Because `bfloat16` merge error scales as $\approx 0.167 / (|dW|/|W|)$, a composite stacked delta is actually represented **more faithfully in `bfloat16`** than any single component alone!

---

## 4. 🛡️ Two-Stage Surgical Stacking Protocol (`scale_mode="surgical"`)

As established in Decision **§50–§52**, `WeightFoldingEngine.activate_many()` uses **Surgical Stacking** as its canonical default mode:

```
                            activate_many(experts, scale_mode="surgical")
                                                 │
                     ┌───────────────────────────┴───────────────────────────┐
                     │                                                       │
           Clean Modules (All Attention + 99.7% MLP)               Targeted Collision Modules (e.g. L29/L30)
                     │                                                       │
                     ▼                                                       ▼
         In-Place AddMM (Full Alpha)                             Apply POET Channel Notch to U
       W.addmm_(u, v, alpha=128.0)                           u_notched = u * mask[:, None]
       [0% attenuation, 100% capacity]                       W.addmm_(u_notched, v, alpha=128.0)
                     │                                                       │
                     └───────────────────────────┬───────────────────────────┘
                                                 │
                                                 ▼
                                   Fused W_live in 17.6 ms (GPU) / ~1.2s (CPU)
                           Zero inference latency overhead during autoregressive generation!
```

### Key Empirical Properties:
1. **Refutation of Global $\sqrt{K}$ Attenuation**: Classical merging scales down all weights by $1/\sqrt{K}$, discarding $74.6\%$ of clean expert capabilities. Surgical Stacking leaves attention and clean MLP layers at **100% full capacity ($\alpha=128$, $0\%$ dampening)**.
2. **Micro POET Channel Notch**: Confines attenuation strictly to isolated output channels (15 neurons) on detected collision modules.
3. **Pristine Buffer Guarantee**: Exact $L_\infty = 0.00\text{e}+00$ bit-level restoration on `restore()`.

---

## 5. Scripts & Benchmarks in this Module

* **[`benchmark_6way_v6_surgical_stack.py`](benchmark_6way_v6_surgical_stack.py)**: Evaluates 6-way concurrent multi-expert stacking across all 6 v6 adapters (`astral`, `postgresql`, `duckdb`, `financial`, `python_modern`, `python_web`) with warm fold timing and drift audits.
* **[`benchmark_surgical_stacking_evaluation.py`](benchmark_surgical_stacking_evaluation.py)**: Targeted 4-expert real-world benchmark comparing Naive (`none`), Classical ($\sqrt{K}$), and Surgical (`surgical`) on real prompt batches.
* **[`goldilocks_in_place_addmm/benchmark_goldilocks_folding.py`](goldilocks_in_place_addmm/benchmark_goldilocks_folding.py)**: End-to-end verification of pointer invariance, zero numerical drift across 100 swap cycles, and Goldilocks merge precision bounds.
* **[`benchmark_weight_folding.py`](benchmark_weight_folding.py)**: Proves the steady-state +82.1% decode speedup of unwrapped in-place weight folding over standard PEFT wrappers.
* **[`evaluate_folded_vs_wrapped.py`](evaluate_folded_vs_wrapped.py)**: Evaluates quality preservation across domains, verifying that in-place weight absorption into `bfloat16` matches wrapped PEFT accuracy.
* **[`benchmark_stacked_experts.py`](benchmark_stacked_experts.py)**: Audits `activate_many()` across all $2^N$ multi-expert combinations with 95% bootstrap confidence intervals.

