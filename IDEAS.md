# MoE & PEFT Architecture Ideas, Technical Concepts & Research Directions

This reference manual documents architectural concepts, PEFT methods, post-training steerability limits, Mixture-of-Experts (MoE) trade-offs, zero-cost inference steering techniques, and novel research paradigms.

---

## Table of Contents

1. [PEFT & Adapter Foundations](#1-peft--adapter-foundations)
   - [1.1 Overview: X-LoRA & MoE-LoRA Concept](#11-overview-x-lora--moe-lora-concept)
   - [1.2 PEFT Alternatives & Trade-offs](#12-peft-alternatives--trade-offs)
   - [1.3 Memory & Efficiency Innovations](#13-memory--efficiency-innovations)
   - [1.4 Convergence & Optimization Innovations](#14-convergence--optimization-innovations)
2. [Model Control Limits & Post-Training Steerability ("What Strings Can Be Pulled")](#2-model-control-limits--post-training-steerability-what-strings-can-be-pulled)
   - [2.1 Overview & Mechanisms](#21-overview--mechanisms)
   - [2.2 High & Medium Control Capabilities ("Soft Strings")](#22-high--medium-control-capabilities-soft-strings)
   - [2.3 Hard Limits & Architectural Walls](#23-hard-limits--architectural-walls)
3. [Mixture-of-Experts (MoE) Architecture Dynamics & Constraints](#3-mixture-of-experts-moe-architecture-dynamics--constraints)
   - [3.1 The Expert Parameter Capacity Floor](#31-the-expert-parameter-capacity-floor)
   - [3.2 Training Complexity & Routing Overhead](#32-training-complexity--routing-overhead)
   - [3.3 Architecture Trade-offs: Small Dense vs. Small MoE](#33-architecture-trade-offs-small-dense-vs-small-moe)
4. [Zero-Cost Inference-Time Expert Steering Without Retraining](#4-zero-cost-inference-time-expert-steering-without-retraining)
   - [4.1 Logit Steering Mechanism](#41-logit-steering-mechanism)
   - [4.2 Four Steering Techniques](#42-four-steering-techniques)
   - [4.3 PyTorch Forward Hook Implementation](#43-pytorch-forward-hook-implementation)
   - [4.4 Steering Benefits & Failure Modes ("The Good vs. The Catch")](#44-steering-benefits--failure-modes-the-good-vs-the-catch)
5. [State of the Art & Novel Research Paradigms](#5-state-of-the-art--novel-research-paradigms)
   - [5.1 Existing Literature Baseline](#51-existing-literature-baseline)
   - [5.2 The Unexplored Frontier: Three Novel Paradigms](#52-the-unexplored-frontier-three-novel-paradigms)
   - [5.3 Infrastructure & Serving Bottlenecks](#53-infrastructure--serving-bottlenecks)
6. [Cross-References & Navigation](#6-cross-references--navigation)

---

## 1. PEFT & Adapter Foundations

### 1.1 Overview: X-LoRA & MoE-LoRA Concept

**X-LoRA / MoE-LoRA** treats multiple pre-trained or fine-tuned LoRA adapters as "Experts". A light gating router predicts which adapter(s) to activate for a given input token, enabling dynamic multi-domain specialization without cross-task interference.

### 1.2 PEFT Alternatives & Trade-offs

| Alternative                                                                | How It Works                                                                                                                                 | Strengths vs. LoRA                                                                   | Weaknesses vs. LoRA                                                                                            |
| :------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- |
| **$\text{IA}^3$** (Infused Adapter by Inhibiting Intermediate Activations) | Multiplies key, value, and feed-forward activations by learned scaling vectors.                                                              | Extremely lightweight (~10x fewer parameters than LoRA) and extremely fast to train. | Lower ceiling for complex syntax or massive domain shifts.                                                     |
| **Houlsby Adapters** (Sequential Adapters)                                 | Inserts bottleneck layers directly between Transformer sub-layers (Down-projection $\rightarrow$ Non-linearity $\rightarrow$ Up-projection). | Strong baseline (predates LoRA).                                                     | Adds sequential layers, introducing minor latency during inference (unlike LoRA, which merges mathematically). |
| **Prefix Tuning / Prompt Tuning**                                          | Prepends trainable "virtual tokens" (continuous embeddings) directly to key/value states or input sequences.                                 | Base model remains 100% frozen; easy to swap per prompt.                             | Consumes part of context window; harder to train stably.                                                       |

### 1.3 Memory & Efficiency Innovations

- **QLoRA (Quantized LoRA)**: Quantizes the frozen base model to 4-bit NormalFloat (NF4) while keeping LoRA matrices in FP16/BF16. This enables fine-tuning of 70B parameter models on consumer GPUs (24GB–48GB VRAM).
- **VeRA (Vector-based Random Aggregation)**: Keeps low-rank projection matrices $A$ and $B$ frozen and randomly initialized. Trains only tiny scaling vectors per layer, reducing adapter parameter count by ~100x compared to standard LoRA while retaining model capacity.

### 1.4 Convergence & Optimization Innovations

- **DoRA (Weight-Decomposed Low-Rank Adaptation)**: Decouples the base model’s weight updates into two components: magnitude (how strong the weight vector is) and direction (where it points). DoRA trains magnitude explicitly while using LoRA for direction, yielding closer alignment to full fine-tuning performance.
- **PiSSA (Principal Singular values and Singular vectors Adaptation)**: Instead of initializing matrix $B$ to zero and $A$ randomly, PiSSA initializes $A$ and $B$ using the Singular Value Decomposition (SVD) of the base weights. This targets the most critical parameter dimensions immediately, leading to faster training convergence.
- **AdaLoRA**: Dynamically allocates the rank budget ($r$) across layers. Critical layers (like attention modules) receive higher rank/capacity, while less important layers are pruned during training.

---

## 2. Model Control Limits & Post-Training Steerability ("What Strings Can Be Pulled")

### 2.1 Overview & Mechanisms

Fine-tuning via adapters (or full SFT) does not rewrite how a neural network processes basic logic; instead, it re-weights token distributions to alter specific behaviors.

### 2.2 High & Medium Control Capabilities ("Soft Strings")

#### Syntax, Formatting, and Schema Lock-In (High Control)

- **Example**: Forcing a model to output `uv add` instead of `pip install`, or enforcing strict JSON schemas, SQL dialects, or Pydantic v2 code structures.
- **Mechanism**: The adapter heavily depresses the logits (probabilities) of legacy patterns (`pip`, `virtualenv`) and boosts preferred token probabilities (`uv`, `ruff`).

#### Tone, Style, and Persona (High Control)

- **Example**: Making a model concise, technical, opinionated, or adopting a specific domain voice.
- **Mechanism**: Fine-tuning acts as a behavioral filter over the base model's vast vocabulary.

#### Routing and Tool-Use Conventions (Medium-High Control)

- **Example**: Teaching a model when and how to construct specific API function calls or invoke local terminal commands.

#### Domain Vocabulary & Terminology (Medium Control)

- **Example**: Teaching internal company acronyms, proprietary medical codes, or custom API endpoints.

### 2.3 Hard Limits & Architectural Walls

#### Injecting Massive New Factual Knowledge

- **The Problem**: Adapters are poor "databases." Trying to teach a model thousands of raw new facts via LoRA often leads to hallucination or memorization without comprehension.
- **Better Solution**: Use RAG or structured context retrieval for external facts; use LoRA to teach the model how to reason over and format those facts.

#### Overcoming Weak Base Capabilities (Math & Logic)

- **The Problem**: If a 3B base model fundamentally struggles with complex mathematical reasoning or 5-step logic puzzles, fine-tuning an adapter will mostly teach it to mimic the appearance of reasoning without making the underlying model smarter.

#### Complete Erasure of Pre-trained Concepts ("Unlearning")

- **The Problem**: LoRA suppresses token probabilities, but the underlying pre-trained weights still contain the original data. Under adversarial prompting or jailbreaks, the base model's old habits can resurface.

---

## 3. Mixture-of-Experts (MoE) Architecture Dynamics & Constraints

### 3.1 The Expert Parameter Capacity Floor

For an expert network inside an MoE to be useful, it needs enough parameter capacity to learn meaningful representations.

- If you split a small model into many tiny experts (e.g., 8 experts of 250M parameters each), a 250M parameter sub-network lacks the capacity to hold complex, multi-step logic or deep domain knowledge.
- The experts become too shallow, leading to routing instability during training and poor reasoning performance.
- **Scaling Rule**: MoEs scale exceptionally well when individual experts are at least 1B to 3B parameters in size. This naturally pushes the total parameter count of a solid MoE into the 14B to 40B+ total parameter range.

### 3.2 Training Complexity & Routing Overhead

Training an MoE model introduces architectural complexities that are not present in dense models:

- **Auxiliary Loss & Load Balancing**: You must train a router network with auxiliary load-balancing loss to balance tokens across experts so a few experts don't do 90% of the work while others stay idle.
- **Fine-Tuning Friction**: Small dense models are trivial to fine-tune with QLoRA on a single consumer GPU. Fine-tuning MoE models requires balancing the router weights and experts, making community fine-tuning and adapter adaptation much harder.

### 3.3 Architecture Trade-offs: Small Dense vs. Small MoE

| Metric                      | Small Dense Model (e.g., 7B)       | Small MoE Model (e.g., 8x1B)                    |
| :-------------------------- | :--------------------------------- | :---------------------------------------------- |
| **Compute per Token**       | High (~7B ops)                     | Low (~2B ops)                                   |
| **VRAM Required**           | ~7GB–8GB                           | ~8GB–9GB                                        |
| **Intelligence Efficiency** | Higher intelligence per GB of VRAM | Lower intelligence per GB of VRAM               |
| **Memory Access Pattern**   | Sequential (Fast on consumer GPUs) | Sparse / Non-contiguous (Higher memory latency) |

---

## 4. Zero-Cost Inference-Time Expert Steering Without Retraining

### 4.1 Logit Steering Mechanism

You can completely alter, steer, or override how an MoE model routes tokens to its experts at inference time **without retraining a single weight**.

Because MoE gating mechanisms (like the Top-$K$ router) calculate numerical probabilities right before selecting which experts to fire, you can manipulate those routing logits programmatically on the fly.

### 4.2 Four Steering Techniques

#### 1. Logit Bias & Temperature Tweaking (Soft Steering)

The MoE router takes the input token vector $x$ and multiplies it by a linear gating layer $W_g$ to produce router logits. Normally, the model applies a Softmax and selects the top $k$ values.

At inference time, you can add a manual bias vector ($b$) directly to the router logits before the `topk()` operation:
$$\text{Logits}_{\text{steered}} = (x \cdot W_g) + b$$

_How to use it_: If you notice Expert #4 handles modern Python/`uv` syntax and Expert #7 handles legacy `pip`, you can statically add a $+2.0$ boost to Expert #4's router logit. The model routes to Expert #4 far more aggressively without altering its internal parameters.

#### 2. Expert Masking / Force Deactivation (Hard Override)

You can hard-code rule sets or safety filters directly into the inference loop to suppress or force specific experts:

- **Forced Deactivation**: Set the router logit of forbidden experts to $-\infty$. If a token would have routed to a legacy code expert, setting its probability to zero forces the router to pick the next highest $k$-th expert.
- **Forced Activation**: Manually pin specific experts to always execute on specific layers regardless of the router output (e.g., forcing Expert #12 to run on all coding queries).

#### 3. Dynamic "Expert Steering Vectors"

Recent techniques (like SteerMoE) identify which experts correlate with specific behaviors by tracking routing differences:

1. Pass prompt pairs (e.g., "Modern Python with uv" vs. "Legacy Python with pip") through the model and log which experts activate.
2. Identify the "Delta Vectors"—which experts activate specifically for your preferred style.
3. Multiply those expert router weights dynamically based on user intent.

#### 4. Router Calibration (Hessian-Aware Calibration)

When loading fine-tuned MoE weights or merging models, the router often suffers from "routing breakdown" (where it misinterprets inputs and dispatches tokens to wrong experts). You can apply closed-form mathematical calibrations (like HARC) directly to the router's projection matrix in milliseconds without performing backpropagation or running GPU training steps.

### 4.3 PyTorch Forward Hook Implementation

If you are running an MoE model (like Mixtral, Qwen3-MoE, or Phi-3.5-MoE) via PyTorch or a custom local runtime, you can intercept the routing layer in ~5 lines of code:

```python
import torch


# Hook into the MoE Router layer at inference time
def steered_router_hook(module, input, output):
    # output contains raw router logits of shape [batch_size, num_experts]
    router_logits = output

    # 1. HARD SUPPRESSION: Completely disable Expert #2 (e.g., legacy/bad syntax expert)
    router_logits[:, 2] = -float("inf")

    # 2. SOFT STEERING: Boost Expert #5 (e.g., modern uv/FastAPI expert) by +3.0
    router_logits[:, 5] += 3.0

    return router_logits


# Apply the hook to a specific MoE layer (e.g., Layer 12's gate)
model.model.layers[12].block_sparse_moe.gate.register_forward_hook(steered_router_hook)
```

### 4.4 Steering Benefits & Failure Modes ("The Good vs. The Catch")

- **The Good**: Instantaneous, zero-cost behavioral control. Force an MoE model to adopt specific domain habits, enforce strict safety, or suppress outdated coding syntax in a few milliseconds without spending hours on QLoRA training.
- **The Catch**: If you force a token to route to an expert that has zero understanding of the current context, the output can degrade into nonsense. The selected expert must still possess some baseline capacity to process the incoming token embeddings.

---

## 5. State of the Art & Novel Research Paradigms

### 5.1 Existing Literature Baseline

- **X-LoRA & MixLoRA**: Instead of fine-tuning an entire model, train multiple independent task/style LoRA adapters (e.g., one for FastAPI, one for Pydantic v2, one for SQL). A light, learned router network sits on top of the base model, taking hidden state activations per token and calculating softmax weights to dynamically blend the LoRAs.
- **Decoupled Two-Stage Training (e.g., TT-LoRA MoE)**: Experts are trained completely independently in Stage 1 and frozen. In Stage 2, only a sparse gating router is trained over those experts. This avoids inter-task interference and catastrophic forgetting.
- **Layer-Heterogeneous MoE-LoRA (MoLA / MoE-Sieve)**: Allocating equal numbers of LoRA experts to every layer is suboptimal. Middle layers benefit from higher expert density, while lower/upper layers require fewer, simpler adapters.

### 5.2 The Unexplored Frontier: Three Novel Paradigms

Standard MoE-LoRA papers rely on a trained linear router (a matrix that looks at hidden states and outputs expert logits). To create a novel paradigm or push the state-of-the-art forward, innovation lies in how the routing decision is computed and steered **without retraining the router**.

#### Paradigm 1: Rule-Gated / Grammar-Guided MoE-LoRA

- **The Idea**: Instead of relying purely on probabilistic neural routers (which can drift or hallucinate), combine deterministic grammar/syntax rules with MoE-LoRA routing.
- **How It Works**: If the user query or context contains specific AST patterns, API constraints, or domain flags (e.g., `toolchain=uv`), an external logit mask hard-overrides or heavily biases the MoE-LoRA router before the forward pass executes.
- **Why It's New**: Bridges classical deterministic systems programming with probabilistic deep learning—giving a 100% guarantee that the `uv` adapter fires when `uv` conditions are met, eliminating router misclassifications.

#### Paradigm 2: Activation-Steered MoE-LoRA (Zero-Training Router)

- **The Idea**: Completely eliminate Stage 2 router training.
- **How It Works**: Use Sparse Autoencoders (SAEs) or steering vectors (from Representation Engineering) to detect latent concepts in the base model's activation space in real-time. If the activation vector aligns with the direction of "modern Python 3.12 syntax," directly inject an activation bias into the corresponding LoRA expert's gating channel.
- **Why It's New**: Plug in new LoRA experts on the fly without running a global router fine-tuning pass.

#### Paradigm 3: Context-Adaptive Dynamic Rank Allocator

- **The Idea**: Dynamic rank assignment per token ($r=0$ to $r=64$).
- **How It Works**: Rather than just picking which LoRA expert to use (Top-1 or Top-2), the router dynamically scales the rank ($r$) of the LoRA adapter based on the query's complexity. Simple boilerplate code uses $r=2$ (saving compute/VRAM), while complex logical tasks activate the full $r=32$ path of the expert.

### 5.3 Infrastructure & Serving Bottlenecks

The primary reason these MoE-LoRA approaches are not ubiquitous yet is an **inference engine bottleneck** rather than a theoretical failure:

- Standard CUDA/HIP kernels in engines like vLLM or llama.cpp are designed to apply one LoRA adapter per request or apply LoRA to a native MoE model's base experts.
- Serving a dynamic, token-level mixture of multiple independent LoRA adapters (where Token 1 uses 30% Adapter A + 70% Adapter B, and Token 2 uses 100% Adapter C) requires custom Triton/HIP kernels like Punica or dedicated multi-adapter fused routing paths.
- High-performance engines are actively working on implementing dynamic multi-adapter router paths natively.

---

## 6. Cross-References & Navigation

- **High-Level Strategic Roadmap**: See [GOALS.md](file:///home/mihai/gnn-experiment/GOALS.md)
- **Empirical Optimization Benchmarks & Engine Implementation**: See [EXPERIMENTS.md](file:///home/mihai/gnn-experiment/EXPERIMENTS.md)

The short answer is **yes, several of these ideas have direct academic or industrial precedents in AI research, but almost none of them were originally conceived through the lens of 3D game engines.**

Mapping game-engine optimization physics (like frustum culling, LODs, motion vectors, and PID frame governors) to sparse neural network execution is a compelling conceptual framework.

Here is a breakdown of what has been tried, what is actively being researched, and where this unified framework pushes beyond standard LLM literature:

---

## 1. Frustum Culling $\rightarrow$ Threshold Expert Culling (T-09 & T-05)

- **Has it been tried?** **Yes.**
- **The AI Precedent:** Standard MoE uses static $Top\text{-}K$ (e.g., $K=2$ in Mixtral or $K=8$ in DeepSeek). However, papers like **"Dynamic Top-p Routing"** and **"Threshold-based MoE"** (such as _DTop-p MoE_) replace static $K$ with a dynamic logit cut-off threshold $\epsilon$.
- **How it compares:** In $DTop\text{-}p$, if a single expert has 95% confidence, only 1 expert fires. If the context is ambiguous, 3 or 4 experts fire.
- **The Game Engine Delta:** Framing this as **Frustum Culling with Prefetching (T-05)** is more advanced than standard $Top\text{-}p$. Standard $Top\text{-}p$ only drops compute _after_ looking at the router. Pairing threshold culling with **PCIe prefetching** (predicting which off-VRAM experts to load over the bus before the token hits the layer) mirrors game-engine texture streaming pipelines.

---

## 2. Level of Detail (LOD) $\rightarrow$ Context-Adaptive Dynamic Rank & Top-$K$ Allocator (T-10 & T-02)

- **Has it been tried?** **Yes, partially.**
- **The AI Precedent:**
- **Dynamic LoRA Rank ($r$):** Techniques like **AdaLoRA** and **Budgeted LoRA** dynamically adjust rank allocation during training or inference.
- **Dynamic $Top\text{-}K$:** Systems like **Ada-K** dial the number of active experts based on computational complexity.

- **The Game Engine Delta:** In AI research, dynamic rank and dynamic $Top\text{-}K$ are usually treated as separate research problems. Your LOD mapping unifies them under a single **Latency SLA Governor**. If frame-time (token generation budget) spikes, the engine downsamples LoRA rank $r$ and MoE $K$ simultaneously to hit real-time token delivery deadlines.

---

## 3. Temporal Motion Vectors $\rightarrow$ Router Logit Extrapolation (T-14)

- **Has it been tried?** **Yes, very recently.**
- **The AI Precedent:** Recent systems research on memory-constrained MoE inference (e.g., **ReMoE**) observed that in continuous domain generation (like Python code or legal prose), expert selection exhibits extreme temporal locality.
- **How it compares:** ReMoE biases the router toward recently selected experts to avoid re-fetching weights off disk/VRAM.
- **The Game Engine Delta:** ReMoE still runs the gating projection pass ($x \cdot W_g$) on every token. The TAA Motion Vector concept is more aggressive: it uses **Hidden State Velocity ($\Delta h_l$)**. If the hidden state's direction isn't changing rapidly, **bypass $W_g$ completely** and reuse the previous token's routing matrix—saving both memory bandwidth and compute cycles.

---

## 4. Closed-Loop PID Servo $\rightarrow$ MoE Load-Balancing & Latency Governors (T-15)

- **Has it been tried?** **Yes, but mostly during training, not live inference.**
- **The AI Precedent:** During pre-training, systems like DeepSeek-V3 and Switch Transformers use static or adaptive **auxiliary load-balancing losses** to prevent "expert hotspotting." Recently, control-theory research (_DTop-p MoE_ or _Adaptive-Expert-Weighting_) introduced **PI/PID controllers** to adjust gating thresholds and loss weights.
- **The Game Engine Delta:** Game engines use PID controllers for **Dynamic Resolution Scaling (DRS)** live during rendering. Applying a PID servo _at inference time_ to dynamically tweak router logit biases ($b$) based on real-time GPU queue depth or VRAM bus congestion turns MoE load-balancing into an active, hardware-aware runtime governor rather than a static model property.

---

## 5. Billboard Impostors $\rightarrow$ Low-Rank Expert Centroid Representations (T-16)

- **Has it been tried?** **No, this is largely unexplored in this specific form.**
- **The AI Precedent:** The closest analogue in AI is **Sparse-Dense Hybrids** or **Compressed Gating Proxies**, where a small draft model or linear layer approximates a full network's decision.
- **Why the Impostor Mapping is Novel:** Replacing off-chip / un-loaded LoRA adapters with a 1-rank "Centroid Billboard Vector" (a static, heavily compressed mathematical proxy of what that expert usually outputs) allows borderline tokens to be processed through the cheap proxy without triggering a multi-megabyte PCIe or VRAM weight fetch. This is an innovative system design.

---

## 6. Self-Speculative Decoding $\rightarrow$ Auxiliary Adapter Speculation (T-08 & T-17)

- **Has it been tried?** **Yes, extensively.**
- **The AI Precedent:** **Eagle**, **Medusa**, and **Speculative Decoding** use lightweight auxiliary heads attached to intermediate transformer layers to draft tokens, which are then verified in parallel by the target model.
- **How it compares:** Eagle-style drafting (your 52.4% acceptance result at Layer 28) is a proven technique in state-of-the-art inference engines (like `vLLM` and `SGLang`).

---

## 7. Heterogeneous Precision Allocation $\rightarrow$ Asymmetric Bitwidth Expert Allocation (T-11)

- **Has it been tried?** **Yes.**
- **The AI Precedent:** Mixed-precision quantization (e.g., **HQQ**, **SpQR**, and **AQLM**) frequently keeps router gates, attention layers, and input/output embeddings at higher precision (FP16 or 4-bit) while quantizing large Feed-Forward Networks (FFNs) down to 2-bit or 3-bit.
- **How it compares:** This is standard practice in ultra-low bitwidth deployments, as routing gates are notoriously sensitive to quantization noise.

---

## Summary Matrix

| Game Engine Technique               | AI Literature Precedent                   | Novelty of Your EXPERIMENTS.md Mapping                                                            |
| ----------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **1. Frustum Culling**              | **High** ($Top\text{-}p$ / Threshold MoE) | **Medium:** Adds PCIe predictive prefetching over the bus.                                        |
| **2. Level of Detail (LOD)**        | **High** (AdaLoRA / Ada-K)                | **High:** Combines rank $r$ and $Top\text{-}K$ into a unified dynamic SLA governor.               |
| **3. TAA Motion Vectors**           | **Medium** (ReMoE / Cache Locality)       | **High:** Uses hidden state velocity ($\Delta h_l$) to completely bypass gate projection ($W_g$). |
| **4. Closed-Loop PID Servo**        | **Medium** (PI-controlled training)       | **High:** Applies live PID feedback to inference queue depth and GPU bus latency.                 |
| **5. Billboard Impostors**          | **Low** (Unexplored in MoE/LoRA)          | **Very High:** Uses low-rank centroid vectors as cheap VRAM proxies for inactive adapters.        |
| **6. Speculative Fast-Forward**     | **High** (Eagle / Medusa / Draft Heads)   | **Validated:** Standard industry practice for speculative execution.                              |
| **7. Texture Asset LOD (Bitwidth)** | **High** (Mixed-Precision Quantization)   | **Standard:** Industry-standard quantization strategy.                                            |

### The Value of This Framework

While individual pieces (like speculative heads or mixed precision) exist in isolated AI papers, **treating an inference engine like a 3D graphics pipeline**—where tokens are frames, experts are mesh assets, and VRAM bandwidth is the draw-call budget—creates a unified runtime architecture.

It shifts the focus from static model tuning to **real-time, dynamic resource orchestration**.

Looking at the research landscape and open-source ecosystems: **Yes, almost all of these individual building blocks have been explored in academic papers or systems frameworks**, but **combining them into a single, unified game-engine engine loop is where the novel territory lies.**

Here is the exact breakdown of what has been tried, what failed in practice, and what remains an open frontier.

---

## 1. What Has Already Been Tried in Literature?

### A. Depth-Wise / Tensor-Factorized LoRA (Cross-Layer Adapters)

- **Status in Research:** **Widely Explored.**
- **Papers:** _FacT (Factorized Adaptation)_, _TensLoRA_, _LoTR_, and _LoRTA_.
- **What they found:** Stacking $A$ and $B$ adapter matrices into a single 3D/4D tensor (using Tensor-Train or Tucker Decomposition) reduces adapter parameter counts by **70%–90%**.
- **Why it hasn't killed standard LoRA:** Contraction operations across 4D tensors in PyTorch are slower than simple matrix multiplications ($A \cdot B$) unless written in custom CUDA/Triton kernels.
- **❌ EXPERIMENT RESULT — FAILED (do not pursue):** Tucker factorization was evaluated in this project across two configurations (Tucker v1: r=8, 487K params; Tucker v2: r=32 + per-layer diagonal scale, 2.05M params). Training loss recovered (1.399 vs 1.366 baseline), but downstream task adherence did not improve (15.66% vs 34.5% for standard QLoRA). Root cause: all layers in a shape-group are forced through the same shared basis directions (U_in/U_out). Per-layer diagonal scaling lets each layer scale that shared subspace but cannot rotate into a different subspace. The pip→uv / black→ruff swap behavior we measure requires each layer to push in a genuinely different gradient direction. Tucker is not a viable adapter strategy for this task. Code is preserved in `novel_peft.py` for reference.

### B. Activation Sparsity & SwiGLU Masking (Your T-05 / Foveated Rendering)

- **Status in Research:** **Widely Explored.**
- **Papers/Tools:** _PowerInfer_, _DejaVu_, _CATs (Context-Aware Thresholding)_.
- **What they found:** In SwiGLU architectures, 60%–80% of intermediate MLP neurons output $0$ after the activation function.
- **The Catch (Matches your T-05 note!):** In dense PyTorch, checking _which_ neurons are non-zero takes more compute than just running the dense matmul. Without a predictor or a custom block-sparse CUDA/Triton kernel (your T-06 block!), activation sparsity gives **zero wall-clock speedup** on GPUs.

### C. CUDA Graphs for Single-Batch Decode (Your T-12 / Draw-Call Batching)

- **Status in Research:** **Standard Industry Practice.**
- **Tools:** `vLLM`, `TensorRT-LLM`, `llama.cpp`.
- **What they found:** In batch-1 decode, PyTorch spends ~50% of its time launching C++/HIP kernel calls on the CPU. CUDA Graph capture eliminates this launch overhead, giving the exact **+50% to +70% speedup** you observed in T-12.

### D. Prefix Radix Caching & KV Mipmapping (Your T-04 / T-13)

- **Status in Research:** **Standard Industry Practice.**
- **Tools:** `SGLang` (RadixAttention), `vLLM` (PagedAttention), `KIVI` (2-bit KV quant).
- **What they found:** Keeping distant KV tokens in 2-bit/4-bit while maintaining recent tokens in 16-bit saves 60%–70% VRAM with negligible loss, exactly matching your T-04 results.

---

## 2. The Uncharted Frontier: "Foveated LoRA" (Skipping Adapter Passes)

Your idea of **Foveated LoRA**—using hidden-state velocity ($\Delta h_l$, T-14) or SwiGLU sparsity (T-05) to **dynamically skip the LoRA adapter pass ($A \cdot B$) on quiet layers during generation**—is **largely UNTRIED in mainstream open-source.**

### Why hasn't this been done?

1. **The "Always-On" Assumption:** Standard frameworks (`peft`, `vLLM`, `Unsloth`) treat LoRA as a static graph addition: $Y = W_0 X + (B \cdot A) X$. The adapter executes on every layer, every token, unconditionally.
2. **Dynamic Routing Overhead:** Checking whether $\Delta h_l < \text{threshold}$ inside a Python loop adds CPU overhead that cancels out the matrix multiplication savings—**unless the check is baked inside a CUDA Graph or Triton kernel.**

---

## 3. How to Make "Foveated LoRA" a Legitimate Systems Contribution

If you build **T-18 (Foveated LoRA)** on your ROCm/AMD setup, here is how to structure it so it yields a genuine performance win:

```dg
┌─────────────────────────────────────────────────────────────────┐
│ DEEP HIDDEN-STATE VELOCITY CHECK (T-14)                        │
│ Calculate: Δh_l = || h_l^{(t)} - h_l^{(t-1)} ||                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
               ┌────────────────┴────────────────┐
               │                                 │
     Δh_l > Threshold                   Δh_l ≤ Threshold
               │                                 │
               ▼                                 ▼
┌──────────────────────────────┐  ┌──────────────────────────────┐
│ FULL FORWARD PASS            │  │ FOVEATED FAST PASS           │
│ Base Weight W_0 + LoRA (A·B) │  │ Base Weight W_0 ONLY         │
│ (Active Feature Adaptation)  │  │ (SKIP LoRA Adapter Computation│
└──────────────────────────────┘  └──────────────────────────────┘

```

### Why This Works for Local Agents

When an LLM generates structured output (like writing an MCP JSON schema or `pyproject.toml` config):

- **High-Velocity Tokens (Keywords, Syntax, Logic):** The model hidden state shifts rapidly $\to$ **LoRA Fires** to enforce your custom rules (`uv`, `override-dependencies`).
- **Low-Velocity Tokens (Boilerplate, Punctuation, Whitespace):** The model hidden state barely moves $\to$ **LoRA Skips**, saving VRAM reads and execution FLOPs.

### The Implementation Requirement

To prevent PyTorch `if/else` checks from slowing down your engine, the velocity mask must be stored as a **static 1D tensor buffer inside your T-12 CUDA Graph**.

If you achieve this on your RX 7900 XTX, you will have created a local inference system that runs small models faster and more efficiently than standard stock frameworks.

When evaluating your summary matrix against the current state of AI research, the individual components map cleanly to existing concepts: **Frustum Culling** maps to _Threshold Routing / Speculative Prefetching_, **LOD** maps to _AdaLoRA_, **TAA** maps to _Cache Locality / ReMoE_, **PID Servo** maps to _Closed-Loop Controller Systems_, and **Billboard Impostors** maps to _Centroid/Prototype Compression_.

What remains **truly unexplored is the combination of these techniques into an integrated runtime system**.

---

## 1. What Is Truly Unexplored?

Isolated research papers analyze these concepts in laboratory silos. Almost nobody in open-source AI builds the **control-loop interactions** that arise when these systems execute concurrently on a single local GPU.

### The 3 Uncharted Frontiers

```dg
                              THE UNEXPLORED TRIAD

                 ┌──────────────────────────────────────────┐
                 │ 1. CROSS-TECHNIQUE CONTROL LOOPS         │
                 │ PID feedback driving TAA + Impostors     │
                 └────────────────────┬─────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────┐ ┌─────────────────────────────────────────┐
│ 2. DYNAMIC HARDWARE-ALIGNED SPARSITY    │ │ 3. DYNAMIC MULTI-LORA BILLBOARD SLOTS   │
│ Velocity masking inside CUDA Graphs     │ │ Prototype centroids mapped to GPU SRAM  │
└─────────────────────────────────────────┘ └─────────────────────────────────────────┘

```

---

### Frontier 1: Cross-Technique Feedback (The Closed-Loop Orchestrator)

In academic literature:

- Paper A tests TAA-style route re-use (ReMoE).
- Paper B tests PID-controlled batching.
- Paper C tests dynamic LoRA rank scaling (AdaLoRA).

**The Unexplored Territory:** **How do these controllers interact when running simultaneously?**

When system latency spikes (e.g., your local agent initiates a complex workspace search):

1. The **PID Servo** fires an output signal $u(t)$.
2. Instead of merely dropping batch size, $u(t)$ dynamically adjusts:

- The **TAA Velocity Threshold** ($\epsilon_{\text{TAA}}$), forcing the model to reuse previous hidden-state projections for quiet layers.
- The **Frustum Culling Radius** ($\epsilon_{\text{Cull}}$), dropping low-probability MoE experts or LoRA adapter channels.
- The **Billboard Impostor Depth**, substituting distant KV tokens or adapter matrices with pre-computed centroid vectors.

Testing this closed-loop cascading fallback—where the PID controller continuously dials back multi-technique parameters to guarantee a strict 25ms SLA—remains unbuilt in open-source frameworks.

---

### Frontier 2: Velocity-Driven Sparsity Captured Inside CUDA Graphs

In existing frameworks (like `vLLM` or `SGLang`), CUDA Graphs require **100% static memory shapes and static execution graphs**.

**The Unexplored Territory:** Creating a **Dynamic Foveated Fast-Pass** inside a static CUDA Graph using pre-allocated zero-copy mask buffers.

- **The Mechanism:** Instead of using Python `if/else` checks (which break CUDA Graph capture and reintroduce CPU overhead), a custom HIP kernel computes the hidden-state velocity ($\Delta h_l$) on token $t$, writes a binary flag (`0` or `1`) to a static GPU memory address, and uses masked GEMMs inside the captured CUDA Graph to skip LoRA matrix multiplications ($A \cdot B$) on quiet layers.
- **Why it's novel:** This achieves true hardware execution acceleration for hidden-state velocity tracking without incurring PyTorch/CPU launch penalties.

---

### Frontier 3: Multi-Adapter "Billboard Impostors" in Memory-Constrained VRAM

Existing multi-LoRA systems (like `S-LoRA` or `Punica`) stream full adapter weights into VRAM on demand, causing bus contention on single-GPU setups.

**The Unexplored Territory:** Using **Centroid "Billboard" Vectors** as low-rank proxies for un-fetched adapters.

- **The Mechanism:** Keep 10–20 domain adapters on system RAM. Compute a 1D "Billboard Centroid" vector ($c_i \in \mathbb{R}^d$) for each adapter that represents its mean directional bias. Keep all 20 Centroid Impostors resident in GPU VRAM (occupying under 2 MB total).
- **Execution:** When the router evaluates an input prompt:

1. It passes intermediate states through the cheap 2 MB **Impostor Centroids** to determine which full adapter is needed.
2. If the full adapter isn't in VRAM yet, the model executes the forward pass using the **Billboard Impostor** as an approximate stand-in while asynchronously prefetching the full LoRA over the PCIe bus (Frustum Culling).

---

## 2. High-Yield Combinations to Experiment With

To maximize speed, low VRAM usage, and opinionated agentic control on your workstation setup (RX 7900 XTX / ROCm / 24 GB VRAM), test these **three high-yield combinations**:

### Combination A: The "Zero-Jitter" Real-Time Engine

- **Formula:** `T-12 (CUDA Graphs)` + `T-15 (PID Servo)` + `T-14 (TAA Velocity Delta)` + `T-11 (GPTQ 4-bit)`
- **How it works:** Run a lossless 4-bit base model inside a CUDA Graph decoder loop. Use the PID Servo to measure real-time HIP kernel latency. If local system background processes cause VRAM/PCIe bus contention, the PID controller dynamically increases the TAA Velocity Threshold ($\epsilon$), skipping standard layer evaluations and maintaining a rock-steady token delivery rate (e.g., 80 tok/s with <4ms standard deviation).
- **Target Outcome:** Eliminates token stutter and latency spikes during long multi-turn agentic coding sessions.

---

### Combination B: The "Infinite-Context Workspace" Engine

- **Formula:** `T-13 (Radix Prefix Cache)` + `T-04 (Hierarchical KV Mipmapping)` + `T-16 (Billboard Impostor Tokens)`
- **How it works:** When your local agent reads a massive 50,000-token codebase or monorepo:

1. The system matches shared system prompts and workspace configs via **T-13 Radix Caching** (11x TTFT speedup).
2. Intermediate context tokens (1,000–8,000) are quantized to 4-bit via **T-04 Mipmapping**.
3. Distant background files (>8,000 tokens) are compressed into **T-16 Billboard Impostor Tokens** (270x reduction).

- **Target Outcome:** Reduces KV cache VRAM footprint from **12+ GB down to <500 MB**, allowing long-context monorepo analysis to run alongside multiple active LoRA adapters on a single 24 GB card.

---

### Combination C: The "Foveated Opinionated Adapter" Engine

- **Formula:** `T-18 (Foveated LoRA)` + `Depth-Wise Tensor Factorization` + `OpenCode Agent Harness`
- **How it works:** Fine-tune a 3D Tensor-Factorized LoRA adapter on modern `uv`, `FastAPI`, and `Pydantic v2` workspace troubleshooting. At runtime, use the hidden-state velocity ($\Delta h_l$) to trigger the adapter pass _only_ when the token stream requires strict domain authority (`pyproject.toml` editing, tool payload generation), skipping the adapter on generic prose or whitespace tokens.
- **Target Outcome:** Reduces adapter computational overhead by ~50% while retaining sharp behavioral enforcement on your specific development stack.

---

## Summary Protocol

```dg
                        SYSTEM ARCHITECTURE INTEGRATION

  ┌────────────────────────────────────────────────────────────────────────┐
  │ PID LATENCY GOVERNOR (T-15)                                            │
  │ Monitored Metric: Frame Budget / Token Latency (target: 25ms)          │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │ Output u(t) Adjusts Thresholds
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ DYNAMIC CONTROLLERS                                                    │
  │ • TAA Velocity Threshold (T-14)   ──> Skips quiet layer projections     │
  │ • Frustum Culler Radius (T-09)    ──> Drops weak MoE/LoRA experts       │
  │ • Billboard Impostor Depth (T-16) ──> Replaces distant KV/adapters     │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │ Static Mask Buffers
                                      ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ CUDA GRAPH CAPTURED DECODE ENGINE (T-12 + T-11)                        │
  │ Lossless GPTQ 4-bit Base Model + Zero-Copy HIP Execution               │
  └────────────────────────────────────────────────────────────────────────┘

```

By focusing on the **feedback loops between these techniques**—rather than treating each paper in isolation—you can build a specialized local inference runtime that operates with game-engine levels of responsiveness on consumer hardware.
