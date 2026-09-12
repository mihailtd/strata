# 🧭 Research Roadmap & Architectural Ideas

This document serves as the long-term conceptual reference and research roadmap for Parameter-Efficient Fine-Tuning (PEFT), Mixture-of-Experts (MoE) adapter routing, dynamic post-hoc steering, and graphics-inspired runtime control.

For active findings, live adapters, and verified benchmarks, see [CURRENT.md](CURRENT.md) and [NOVELTY.md](NOVELTY.md).

---

## Table of Contents

1. [PEFT & Adapter Foundations](#1-peft--adapter-foundations)
2. [Model Control Limits & Post-Training Steerability](#2-model-control-limits--post-training-steerability)
3. [Mixture-of-Experts & Capacity Dynamics](#3-mixture-of-experts--capacity-dynamics)
4. [Zero-Cost Inference-Time Steering](#4-zero-cost-inference-time-steering)
5. [Novel Paradigms & Unexplored Frontiers](#5-novel-paradigms--unexplored-frontiers)
6. [Graphics-Inspired Runtime Concepts: Status & Lessons](#6-graphics-inspired-runtime-concepts-status--lessons)

---

## 1. PEFT & Adapter Foundations

Parameter-Efficient Fine-Tuning freezes the base model weights $W_0 \in \mathbb{R}^{d \times k}$ and injects small trainable parameter sets to represent $\Delta W$.

### 1.1 Taxonomy of Adaptation Techniques

| Method | Mathematical Structure | Trainable Footprint | Core Trade-offs & Properties |
| :--- | :--- | :---: | :--- |
| **LoRA** | $W = W_0 + \frac{\alpha}{r} B A$ ($A \in \mathbb{R}^{r \times k}, B \in \mathbb{R}^{d \times r}$) | Medium (~10–20 MB) | Standard industry baseline; merges linearly into base weights with zero inference latency. |
| **QLoRA** | $W = \text{dequant}(W_{\text{NF4}}) + \frac{\alpha}{r} B A$ | Medium (~10–20 MB) | 4-bit base model cuts VRAM during training; note: folding QLoRA adapters into bf16 base models creates a cross-precision seam (see [CURRENT.md](CURRENT.md)). |
| **DoRA** | $W = m \frac{W_0 + B A}{\|W_0 + B A\|_c}$ | Medium (~11–22 MB) | Decouples magnitude ($m$) and direction. High training overhead (77% slower) for marginal downstream gains on small models. |
| **PiSSA** | $W = (W_0 - U_r S_r V_r^T) + (U_r \sqrt{S_r} + B) (\sqrt{S_r} V_r^T + A)$ | Medium (~10–20 MB) | Initializes adapter directly on principal singular components of $W_0$. Requires exact SVD (randomized SVD is biased on flat singular spectra). |
| **VeRA** | $W = W_0 + \Lambda_b (B A) \Lambda_d$ ($A, B$ frozen random) | Very Small (~1–5 KB) | Freezes shared random projection matrices and trains only per-layer scaling vectors. |
| **$\text{IA}^3$** | $y = (W_0 x) \odot l$ | Ultra-light (~0.1 MB) | Multiplies activations by learned scaling vectors. Low parameter count, lower capacity for complex domain shifts. |
| **Prefix / Prompt Tuning** | $K \leftarrow [P_K; K], V \leftarrow [P_V; V]$ | Small (~0.5 MB) | Prepends virtual tokens to key/value states. Consumes context window and adds sequence overhead. |
| **Kronecker (`id_kron` / LoKr)** | $W = W_0 + (I_{r1} \otimes W_a) W_b$ | Small–Medium (~6–12 MB) | Block-diagonal Kronecker factorization. Yields parameter savings on disk, but expands to dense matrices when folded into base weights. |

### 1.2 Subspace Orthogonality & Compression Limits

Direct geometric measurements on fine-tuned adapters demonstrate a fundamental property:
- **Fine-tuning updates on dense transformer layers populate mutually near-orthogonal subspaces across tasks.**
- Projecting an adapter onto an independent task's SVD basis retains **$0.00\%$** variance (measured cosine similarity of $3.25 \times 10^{-4}$ vs $2.06 \times 10^{-4}$ expected by chance).
- **Implication**: Cross-task shared-basis compression (e.g. multi-task Tucker or shared basis banks) cannot succeed without task-specific subspace rotation.

---

## 2. Model Control Limits & Post-Training Steerability

Fine-tuning via adapters adjusts output token distributions without rewriting the model's fundamental reasoning circuits.

```
┌─────────────────────────────────────────────────────────────┐
│ HIGH CONTROL ("Soft Strings")                               │
│ • Syntax & schema lock-in (uv vs pip, JSON schema, Pydantic)│
│ • Tone, persona, and stylistic formatting                   │
│ • Tool invocation protocols (<tool_call>, CLI conventions)  │
├─────────────────────────────────────────────────────────────┤
│ MEDIUM CONTROL                                              │
│ • Domain vocabulary, technical taxonomy, acronym expansion  │
├─────────────────────────────────────────────────────────────┤
│ HARD LIMITS ("Architectural Walls")                         │
│ • Massive factual database injection (RAG is required)      │
│ • Fundamental reasoning capacity (sub-1B logic limits)      │
│ • Complete unlearning of base pretraining concepts          │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Mixture-of-Experts & Capacity Dynamics

### 3.1 The Expert Capacity Floor
- An individual sub-network requires sufficient parameter capacity to maintain multi-step reasoning.
- Splitting small base models (e.g. sub-1B) into shallow sub-experts degrades routing stability and reasoning quality.
- For small models (sub-10B), maintaining **full-rank base weights with folded domain adapters** outperforms sparse fine-grained MoE sub-networks.

### 3.2 Dense Base + Dynamic Weight Folding vs Sparse MoE
Instead of routing tokens through sparse FFN branches inside the network:
1. Keep the base backbone unified and dense.
2. Maintain resident domain adapters in low-rank factor form ($U, V$).
3. Mutate live weights in-place via $W_{\text{live}} \leftarrow W_0 + \frac{\alpha}{r} U V$ during domain transitions.
4. Preserves dense memory access patterns while supporting hundreds of resident domain specializations.

---

## 4. Zero-Cost Inference-Time Steering

Router logits in MoE gates or adapter mixture modules can be programmatically manipulated at inference time without gradient updates:

```python
import torch


def steered_router_hook(module, input, output):
    """Intercept and steer routing logits before top-k selection."""
    router_logits = output  # Shape: [batch_size, num_experts]

    # 1. Hard Suppression: Disable legacy or unwanted expert
    router_logits[:, 2] = -float("inf")

    # 2. Soft Biasing: Boost target domain expert
    router_logits[:, 5] += 3.0

    return router_logits
```

### Steering Modalities
1. **Soft Steering (Logit Biasing)**: Adds a continuous bias vector $b$ to router logits prior to softmax/top-k.
2. **Hard Override (Expert Masking)**: Sets non-target router logits to $-\infty$, guaranteeing deterministic expert dispatch.
3. **Dynamic Steering Vectors**: Scales routing logits based on real-time activation similarity against concept vectors extracted via Sparse Autoencoders (SAEs).
4. **Hessian-Aware Router Calibration (HARC)**: Closed-form adjustment of routing projection matrices to eliminate drift after model merging.

---

## 5. Novel Paradigms & Unexplored Frontiers

### 5.1 Rule-Gated / Grammar-Guided Adapter Gating
- **Concept**: Combine deterministic Abstract Syntax Tree (AST) / grammar parsers with neural routing.
- **Mechanism**: When an input prompt matches deterministic toolchain triggers (e.g. active workspace containing `pyproject.toml` with `uv` dependencies), an external deterministic mask forces the `astral` domain adapter to fold into execution, bypassing probabilistic gating ambiguity.

### 5.2 Latent Activation Steering (Zero-Training Router)
- **Concept**: Eliminate router training phases entirely.
- **Mechanism**: Monitor intermediate residual stream activations $h_l$. Project $h_l$ onto pre-computed task direction vectors. When cosine similarity exceeds threshold $\theta$, trigger adapter activation dynamically.

### 5.3 Token-Level Adaptive Rank Allocation
- **Concept**: Scale computation dynamically by token complexity.
- **Mechanism**: Simple boilerplate tokens (punctuation, indentation) execute through rank $r=0$ (bare base model), while high-entropy tokens activate higher rank paths ($r=16$).

---

## 6. Graphics-Inspired Runtime Concepts: Status & Lessons

| Concept | Game-Engine Analogy | Implementation Status & Lessons |
| :--- | :--- | :--- |
| **CUDA Graph Decode** | Draw-Call Batching | **⭐ Live & Verified**: Single-capture graph replay eliminates CPU kernel launch overhead. Pointers must remain stable across adapter mutations. |
| **Pristine State Buffer** | Master Assets / Textures | **🔥 Live & Verified**: Keeping an unmutated $W_0$ reference allows bit-exact ($L_\infty = 0.00$) in-place expert folding and restoration. |
| **Recurrent Checkpoint** | Save-State Rollback | **🔥 Live & Verified**: 52.5 MB fixed-size snapshot/restore unblocks speculative decoding on hybrid linear-attention models (`Qwen3_5GatedDeltaNet`). |
| **Centroid Billboard Impostors** | Billboard LOD Proxies | **❓ Theoretical Prototype**: Storing 1D mean centroid bias vectors in VRAM as fast routing stand-ins while prefetching full adapters over PCIe. |
| **Closed-Loop Latency Servo** | PID Frame Governor | **🟩 Tested**: Thompson Sampling Multi-Armed Bandit adjusts execution knobs ($K$) under latency bounds. |
| **Foveated LoRA / Velocity Gate** | Foveated Rendering | **❌ Evaluated & Refuted**: Gating layers during backward/forward based on hidden-state velocity ($\Delta h_l$) added overhead without improving quality or compute efficiency. |
| **Cross-Layer Tucker Adapters** | 3D Mesh Compression | **❌ Evaluated & Refuted**: Shared Tucker core forces layers into identical subspaces; downstream task adherence dropped from 34.5% to 15.66%. |
| **APSP Shortest-Path Router** | NavMesh Pathfinding | **❌ Evaluated & Refuted**: Transition costs are destination-only ($W_0 + dW$); direct edge is provably optimal for any expert count. Replaced by SLA-bounded cluster scheduling. |
| **Radix Cache 54% Prefill Claim** | Asset Streaming | **❌ Refuted Timing Artifact**: The 54% prefill figure was un-warmed Triton JIT compilation. Warmed prefill is only 5.9% of wall time at 8k context. |
