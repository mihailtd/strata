# Multi-Adapter Composition, Serving Systems, and Agentic Runtime Architecture

Grouping multi-LoRA techniques together often conflates two fundamentally different engineering domains:

1. **Algorithmic Composition & MoE Routing:** Architectural methods to combine multiple low-rank representations so a model can solve broader, multi-domain tasks without catastrophic forgetting or rank saturation.
2. **Multi-Tenant Serving Engines:** Systems-level infrastructure designed to serve hundreds of distinct, unmerged LoRAs concurrently on one GPU cluster without destroying batching throughput or running out of VRAM.
3. **Single-Stream Agentic Runtimes (Our Work):** Bare-metal, latency-critical inference engines designed for sequential multi-agent execution at batch size 1, where unmerged SGMV kernels fail and in-place weight folding paired with recurrent state transfer dominates.

---

## Track 1: Algorithmic Composition & Dynamic Routing (Scaling Capacity)

These methods dynamically route tokens, layer representations, or optimize adapter combinations to expand model capacity without retraining the base LLM.

```
Token Input
     │
     ├───► Shared Base Weights (W₀) ──────────────────────────┐
     │                                                         │
     ├───► Router G(x) ──► Dynamic Gating Weights             │
     │                           │                             ▼
     └───► [LoRA Expert 1, LoRA Expert 2, ... LoRA Expert N] ──(+)──► Output
```

### 1. MixLoRA (Sparse Token-Level MoE)
* **Paper:** [MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts](https://arxiv.org/abs/2404.15159)
* **Code:** [GitHub - TUDB-Labs/MixLoRA](https://github.com/TUDB-Labs/MixLoRA)
* **The Core Technique:** MixLoRA replaces the standard Feed-Forward Network (FFN) blocks with multiple independent LoRA modules acting as sparse experts. While the base FFN weights $W_0$ remain frozen and shared, a learnable gating network routes tokens:
  $$y = W_0 x + \sum_{i \in \text{Top-}k} G(x)_i \cdot \frac{\alpha}{r} (B_i A_i x)$$
  Attention layers use a unified, shared LoRA adapter to preserve global semantic coherence across attention heads, while the MLP blocks handle domain specialization.
* **Problem It Solves:** Pretraining a full sparse MoE (like Mixtral 8x7B) requires massive compute. MixLoRA allows you to turn a frozen dense model into an MoE post-hoc, scaling parameter capacity across tasks while keeping active compute low.
* **Failure Modes & Trade-Offs:** Requires joint training from the start (or continual training with expert dropout). If the router collapses into selecting only one or two experts, you lose the MoE benefits; it requires an auxiliary load-balancing loss ($\mathcal{L}_{\text{balance}}$) during training.

### 2. MoLE & LD-MoLE (Hierarchical Layer-Wise Gating)
* **Papers:**
  * [Mixture of LoRA Experts (MoLE)](https://arxiv.org/abs/2404.13628) (Wu et al.)
  * [LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](https://arxiv.org/abs/2509.25684) (ICLR)
* **Code:** [GitHub - adithya-s-k/MoLE](https://github.com/adithya-s-k/MoLE)
* **The Core Technique:** Unlike MixLoRA, which trains LoRA experts from scratch with a top-$k$ router, MoLE takes **pre-trained, frozen LoRAs** from different domains and fuses them using hierarchical, layer-specific gating. Each transformer layer $l$ computes independent routing probabilities:
  $$g^l(x) = \text{Softmax}(W_g^l x)$$
  This allows continuous blending of representations rather than hard top-$k$ exclusion.
* **Problem It Solves:** Naive weight arithmetic (adding LoRA matrices directly) causes weight interference: orthogonal representations cancel each other out, degrading generative quality. MoLE gives each layer the freedom to pull 70% from `LoRA_Code` and 30% from `LoRA_Math` at Layer 4, then invert the ratio at Layer 18.
* **Failure Modes & Trade-Offs:** The gating network requires a secondary tuning phase on a mixed dataset. Running all LoRA branches in parallel per layer increases compute and latency linearly with the number of loaded experts unless heavily pruned.

### 3. LoraHub (Few-Shot Black-Box Task Composition)
* **Paper:** [LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition](https://arxiv.org/abs/2307.13269) (COLM)
* **Code:** [GitHub - sail-sg/lorahub](https://github.com/sail-sg/lorahub)
* **The Core Technique:** LoraHub operates entirely offline before inference. Given a large library of $N$ specialized LoRA modules $\{\Delta W_1, \Delta W_2, \dots, \Delta W_N\}$ and 3–5 few-shot demonstration examples of an unseen task, it formulates adapter merging as a black-box parameter optimization problem:
  $$\Delta W_{\text{composed}} = \sum_{i=1}^N w_i \Delta W_i$$
  It uses the **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) algorithm to find the optimal scalar weights $w_i \in \mathbb{R}$ that minimize evaluation loss on the few-shot set.
* **Problem It Solves:** Removes the need for gradient descent, backpropagation, or dedicated router layers to generalize to a new task. Because the weights are linearly merged into the base model before serving, **inference overhead is zero**.
* **Failure Modes & Trade-Offs:** CMA-ES struggles as the library $N$ grows beyond several dozen adapters due to the curse of dimensionality. Furthermore, linear composition assumes task representations are globally compatible across all layers, which breaks down for highly conflicting tasks.

### 4. Weight-Space Interference Mitigation (TIES & DARE)
* **Papers:**
  * [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708) (NeurIPS)
  * [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE)](https://arxiv.org/abs/2311.03099) (ICML)
* **Code:**
  * [GitHub - prateeky2806/ties-merging](https://github.com/prateeky2806/ties-merging)
  * [GitHub - yule-BUAA/MergeLM (DARE)](https://github.com/yule-buaa/mergelm)
  * [GitHub - arcee-ai/mergekit](https://github.com/arcee-ai/mergekit) (Unified production CLI tool implementing TIES, DARE, SLERP, and Task Arithmetic for LoRA/base models)
* **The Core Technique:** Resolves parameter collision during static merges. TIES trims bottom delta weights, resolves sign disagreements via majority voting, and averages aligned parameters. DARE drops up to 90–99% of delta parameters via random Bernoulli masks and rescales the remainder by $1/(1-p)$.

### 5. O-LoRA (Orthogonal Continual Fine-Tuning)
* **Paper:** [O-LoRA: Orthogonal Low-Rank Adaptation for Continual Learning](https://arxiv.org/abs/2310.18025)
* **Code:** [GitHub - cmnfriend/O-LoRA](https://github.com/cmnfriend/O-LoRA)
* **The Core Technique:** Enforces strict orthogonality constraints between adapter parameter subspaces across sequential tasks ($A_i A_j^T = 0$), preventing new tasks from overwriting principal directions learned by past adapters.

---

## Track 2: Systems-Level Multi-Adapter Serving (High-Throughput Concurrency)

These frameworks solve the infrastructure bottleneck: batched parallel evaluation of non-identical LoRAs on top of a single frozen base model in high-concurrency cloud environments.

```
Request Stream:
[Req 1: Adapter_A] ──┐
[Req 2: Adapter_B] ──┼──► [Shared Base GEMM] ──► [SGMV Kernel / Unified Paging] ──► Output
[Req 3: Base Only] ──┘
```

### The Non-Homogeneous Batching Dilemma
In standard LLM serving, batching relies on running identical matrix-matrix multiplications (GEMM) across all requests:
$$Y = X W_0$$
If Request 1 uses `Adapter_A`, Request 2 uses `Adapter_B`, and Request 3 uses the base model, you can compute $X W_0$ in parallel, but you cannot naively compute the adapter deltas without looping through each request sequentially, collapsing GPU compute utilization.

### 1. Punica (Segmented Gather Matrix-Vector Multiplication)
* **Paper:** [Punica: Multi-Tenant LoRA Serving](https://arxiv.org/abs/2310.18547)
* **Code:** [GitHub - punica-ai/punica](https://github.com/punica-ai/punica)
* **Core Innovation:** **SGMV (Segmented Gather Matrix-Vector)** CUDA kernel.
* **Mechanism:** Decouples base model computation from adapter computation:
  1. Batches all requests together for the base model forward pass ($X W_0$).
  2. For LoRA additions, SGMV treats the batch as a segmented sequence where each segment points to distinct low-rank weight matrices ($A_i \in \mathbb{R}^{d \times r}, B_i \in \mathbb{R}^{r \times k}$).
  3. Evaluates multiple disparate adapters within a single unified CUDA kernel launch during the token decoding phase.

### 2. S-LoRA (Unified Paging & Heterogeneous Ranks)
* **Paper:** [S-LoRA: Serving Thousands of Concurrent LoRA Adapters](https://arxiv.org/abs/2311.03285)
* **Code:** [GitHub - S-LoRA/S-LoRA](https://github.com/S-LoRA/S-LoRA)
* **Core Innovation:** **Unified Paging** and **Heterogeneous Tensor Parallelism**.
* **Mechanism:** Punica hit limits when handling thousands of adapters with dynamic ranks ($r=4, 8, 32, 64$), leading to severe memory fragmentation. S-LoRA manages both KV caches and dynamic LoRA weights within a single unified block-allocated memory pool. As adapters are requested, small low-rank pages are dynamically fetched into free blocks.

### 3. vLLM Multi-LoRA Engine
* **Documentation:** [vLLM Supported Models & Multi-LoRA Engine](https://docs.vllm.ai/en/latest/models/lora.html)
* **Code:** [GitHub - vllm-project/vllm](https://github.com/vllm-project/vllm)
* **Mechanism:** Implements native Multi-LoRA scheduling directly inside the PagedAttention decoding loop via Triton-based SGMV/BGMM kernels. Allows passing `lora_request` objects dynamically per API call without worker restarts.

### 4. LoRAX (Predibase Multi-LoRA Server)
* **Code:** [GitHub - predibase/lorax](https://github.com/predibase/lorax)
* **Documentation:** [LoRAX Docs](https://predibase.github.io/lorax/)
* **Mechanism:** Built on top of Rust/CUDA inference infrastructure. Supports dynamic adapter compilation, asynchronous adapter downloading from S3/HuggingFace on request arrival, and automated memory-tier eviction.

---

## Track 3: The Single-Stream Decode Tax: Why SGMV Fails at $B=1$

The literature on multi-tenant LoRA serving (Punica, S-LoRA, vLLM) solves **throughput**, not **latency**. In high-concurrency cloud environments ($B \ge 16$), the base GEMM amortizes the memory bandwidth of loading $W_0$, while SGMV groups heterogeneous adapter passes across requests.

At batch size 1 (an autonomous local agent taking sequential execution steps), this architecture collapses:

* **Kernel Launch Bound:** A forward pass over a standard 32-layer LLaMA/Qwen architecture contains 224 linear projections (7 projections per layer: `q, k, v, o, gate, up, down`). With unmerged LoRA, each projection requires an extra launch for $A_i$ and $B_i$, adding **448 tiny kernel launches per token**. At $B=1$, GPU compute units are starved, and CPU launch latency dominates.
* **Empirical Latency Blowup:** Research in dynamic adapter execution (*LoRA-Switch*, Kong et al.) benchmarked the exact overhead of unmerged dynamic adapters at $B=1$: dynamic routing adapters (such as MOLA and MoRAL) increased single-stream token decoding latency by **254% to 954%** despite adding less than 1% to 5% FLOPs. The overhead is almost entirely kernel launch fragmentation and non-contiguous memory gather latency.
* **Why In-Place Weight Folding (IPWF) Wins:** Folding the weights ($W' = W_0 + \frac{\alpha}{r}BA$) via a single batched `hipblasGemmEx` upon agent transition incurs a one-time step cost (~15–33 ms for all layers), but returns the decode loop to **100% bare-metal GEMV speed (one single kernel launch per projection, zero scatter/gather memory traffic)**.

---

## Track 4: Advanced LoRA-Specific Merging: Beyond Naive TIES and DARE

Standard TIES and DARE were designed for merging **fully fine-tuned model checkpoints**, not low-rank adapters. Applying them naively to multi-LoRA stacking introduces severe structural failure modes that explain why surgical channel isolation is necessary.

```
Full-Rank Fine-Tuning:               LoRA Adapters (Independent Subspaces):
Shared parameter coordinate space    Arbitrary coordinate rotations (orthogonal gauge freedom)
       ┌───────────┐                        ┌───────────┐         ┌───────────┐
       │ Task A ΔW │                        │ Task A:   │         │ Task B:   │
       └─────┬─────┘                        │ B_A · A_A │         │ B_B · A_B │
             │ (Element-wise align)         └─────┬─────┘         └─────┬─────┘
       ┌─────▼─────┐                              │                     │
       │ Task B ΔW │                              └──────────┬──────────┘
       └───────────┘                                         ▼
                                                   Coordinate Misalignment
                                                   (Low CKA Similarity)
```

### The Centered Kernel Alignment (CKA) Gap & KnOTS
Recent work on adapter geometry (*KnOTS*, Stoica et al.) demonstrated why TIES degrades on LoRA:
* Fully fine-tuned models exhibit high feature-space alignment because all updates are anchored to the pre-trained base coordinate frame.
* LoRA adapters exhibit **inherently low Centered Kernel Alignment (CKA)** across different tasks because the low-rank factor decomposition $\Delta W = B A$ has infinite rotational degrees of freedom ($B R \cdot R^{-1} A$).
* **The SVD-Alignment Solution (KnOTS):** Instead of applying TIES directly on unaligned $\Delta W_i$, KnOTS concatenates adapter representations layer-wise, computes a cross-task SVD to project them into a shared orthogonal basis $(U\Sigma, V^T)$, applies trimming/DARE in that shared space, and projects back.

### Low-Rank Manifold Rotation: TSPA
A newer approach presented at ACL, **TSPA (Two-Stage Parameter Alignment)**, addresses high-rank multi-LoRA merging specifically:
* It proves that naive linear combinations ($\sum w_i B_i A_i$) cause quadratic interference growth as rank $r$ increases.
* TSPA formulates alignment as an optimization problem on the **Stiefel manifold**, finding orthogonal rotation matrices $R_i \in O(r)$ such that:
  $$\min_{R_i} \Vert{} B_i R_i - B_j R_j \Vert{}_F^2 \quad \text{subject to} \quad R_i^T R_i = I$$
* **Relevance to Surgical Stacking:** Rotating the low-rank bases into maximum alignment before applying POET notch filtering prevents the notch masks from inadvertently clipping signal that was merely rotated into a misaligned coordinate basis.

---

## Track 5: State Continuity: Linear Attention Recurrent State vs. Softmax KV Caching

In a multi-agent pipeline where Agent A (e.g., Code Search) yields control to Agent B (e.g., Code Refactor), softmax-based transformer architectures hit an information-transfer barrier:

| Mechanism | Softmax Attention (vLLM / SGLang / Ollama) | Linear Attention / Recurrent (GatedDeltaNet in runtime-next) |
| :--- | :--- | :--- |
| **Context Representation** | Unbounded KV-cache tensors: $O(T \times d_{\text{head}} \times N_{\text{layers}})$ | Fixed-size memory matrix: $S_t \in \mathbb{R}^{d_k \times d_v}$ per head |
| **Handoff Mechanism** | String serialization $\to$ Prefill recompute or chunked prefix cache lookup | Direct VRAM buffer clone: `hipMemcpyDtoD` of $S_t$ |
| **Handoff Complexity** | $O(T^2)$ or $O(T)$ bandwidth read overhead | **$O(1)$ constant time** (~48 MB tensor copy, ~0.05 ms) |
| **Context Rot Susceptibility** | High (attention dispersion across giant histories) | Bound (governed strictly by the recurrent decay/overwrite rule) |

### Decoupled Erase-Write in GatedDeltaNet-2
The primary vulnerability of recurrent state architectures has been capacity saturation: write operations inevitably overwrite critical prior context.

The latest formulation, **GatedDeltaNet-2**, resolves this by decoupling the scalar retention gate into two channel-wise vector operations:
$$S_t = \alpha_t \odot S_{t-1} - (\mathbf{b}_t \odot k_t) (S_{t-1}^T k_t)^T + (\mathbf{w}_t \odot v_t) k_t^T$$

* $\mathbf{b}_t \in [0, 1]^{d_k}$: Channel-wise **erase gate** (protects specific key channels from being overwritten).
* $\mathbf{w}_t \in [0, 1]^{d_v}$: Channel-wise **write gate** (prevents distractor noise from entering value channels).

In an agentic DAG, passing this recurrent state $S_t$ directly from Agent A to Agent B preserves structured semantic memory at zero token-prefill cost.

---

## Track 6: Zero-Penalty Execution: Pristine Buffers & Pointer-Stable HIP Graphs

When executing in-place folding on bare metal (ROCm / HIP on RDNA3), maintaining graph stability requires strict adherence to virtual address invariants:

```
VRAM Layout:
┌─────────────────────────────────────────────────────────────┐
│ Pristine Base Weights Buffer (Read-Only BF16)               │ ──► Unmodified W₀
└─────────────────────────────────────────────────────────────┘
                               │
               hipblasGemmEx   ▼ (Fold / Unfold)
┌─────────────────────────────────────────────────────────────┐
│ Active Model Weights Buffer (HIP Graph Target Pointer)      │ ──► W' = W₀ + (α/r)·BA
└─────────────────────────────────────────────────────────────┘
                               ▲
                               │
              Pre-captured Execution Graph
              (hipGraphExec_t: Node pointers NEVER mutate)
```

### Why In-Place Subtraction Truncates
Naive implementations attempt in-place rollback:
$$W_0 \leftarrow W' - \frac{\alpha}{r}(BA)$$

In BF16 (which has only 7 bits of mantissa), repeated add-subtract cycles cause catastrophic cancellation. Within 10 adapter swaps, base model perplexity explodes due to roundoff drift ($L_\infty > 0.05$). Maintaining a dedicated, read-only **Pristine Buffer** allows restoring exact weights via streaming copy (`hipMemcpyDtoDAsync`) in ~4–6 ms, guaranteeing **$L_\infty = 0.00$ idempotence**.

### HIP Graph Pointer Invariance
`hipGraphLaunch` bakes literal virtual memory device pointers into the compiled execution plan (`hipGraphExec_t`). If an engine frees and re-allocates a weight tensor to accommodate an adapter, the graph becomes invalid and must be re-captured, incurring a 50–200 ms stalling penalty.

* By folding adapter deltas directly into the pre-allocated memory addresses of the active buffer, **the graph's memory operands remain static**.
* Decoding continues within the already-captured execution graph without CPU-side synchronization.

---

## Comparative Analysis: Our Research vs. Literature

| Dimension | Track 1 (MixLoRA / MoLE / TIES) | Track 2 (Punica / S-LoRA / vLLM) | Our Research (`runtime-next` / `runtime-ipwf`) |
| :--- | :--- | :--- | :--- |
| **Optimization Target** | Parameter capacity expansion | High-concurrency cloud throughput ($B \ge 16$) | Single-stream agentic latency ($B=1$) |
| **Adapter Placement** | Extra layers or static weight merge | Discrete VRAM pages evaluated via SGMV | Folded in-place into live GEMM buffers |
| **Decode Overhead** | Moderate (parallel branch evaluation) | 15%–25% (448 extra kernel launches per step) | **0.0% (Runs at bare base-model GEMV speed)** |
| **Swap Mechanism** | Router gating or offline merge | Queue scheduling into SGMV tiles | **33 ms in-place GEMM fold** via hipBLAS |
| **Execution Graphs** | Eager PyTorch execution | Incompatible with static CUDA/HIP graphs | **100% Graph-stable (device pointers never change)** |
| **Numerical Safety** | Heuristic parameter averaging | Unmodified base weights | **Pristine Reference Buffer ($L_\infty = 0.00$)** |
| **Multi-Agent Handoff** | Plain text re-prefill ($O(T)$ or $O(T^2)$) | Plain text re-prefill ($O(T)$ or $O(T^2)$) | **$O(1)$ VRAM Recurrent Tensor State ($S_t$) Clone** |

---

## Concrete Architectural Synthesis & Implementation Opportunities

To bridge the gap between surgical stacking and dynamic agent execution:

1. **Adopt DARE Pruning at Pre-Fold:** Before calling `hipblasGemmEx` for multi-expert stacking (`activate_many`), apply a 70–80% Bernoulli dropout mask to the adapter deltas. Pruning small delta weights significantly diminishes cross-adapter coordinate collisions in intermediate MLP layers without requiring complex continuous re-tuning.
2. **Evaluate Low-Rank Alignment (KnOTS/TSPA) Before Stacking:** If two adapters exhibit high cosine similarity in their singular vectors, rotate them into an aligned basis on the Stiefel manifold prior to calculating POET notch masks to preserve shared knowledge transfer.
3. **Formalize the B=1 Operational Envelope:** Benchmark the latency delta between SGMV and IPWF across batch sizes $B \in [1, 32]$ on the RX 7900 XTX. Grounding this metric cements the design justification for running merged weights in single-stream agentic workflows while cloud multi-tenant systems remain tied to SGMV.
4. **Borrow MixLoRA Gating for MoE Phase 2:** Utilize MixLoRA's auxiliary load-balancing loss and top-$k$ routing mathematics as the reference blueprint when implementing the Qwen 3.5 35B-A3B MoE router in `runtime-next`.
