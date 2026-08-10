# Goal 1: Integrated Closed-Loop Inference Runtime & Multi-Adapter Engine

## Executive Summary & Objective

**Primary Objective:** Design, build, and benchmark an integrated local AI inference runtime system that combines graphics-inspired optimization techniques (**Frustum Culling**, **Level of Detail (LOD)**, **Temporal Anti-Aliasing (TAA)**, **PID Servos**, and **Billboard Impostors**) into a dynamic, closed-loop control framework.

While individual components map to known AI research concepts, **the core goal of this project is to pioneer the unexplored combination of these techniques into an integrated runtime system** executing concurrently on local consumer hardware (AMD Radeon RX 7900 XTX / ROCm / 24 GB VRAM).

### Graphics Concept to AI Architecture Mapping Matrix
* **Frustum Culling** $\rightarrow$ *Threshold Routing / Speculative Prefetching*
* **Level of Detail (LOD)** $\rightarrow$ *AdaLoRA / Rank-Adaptive Adapters*
* **Temporal Anti-Aliasing (TAA)** $\rightarrow$ *Cache Locality / ReMoE Hidden-State Projection Reuse*
* **PID Servo** $\rightarrow$ *Closed-Loop Latency Controller Systems*
* **Billboard Impostors** $\rightarrow$ *Centroid / Prototype Adapter Compression*

---

## 1. Unexplored Frontiers & Core Research Goals

```
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

### Goal 1.1: Cross-Technique Feedback (The Closed-Loop Orchestrator)
* **Context & Gap:** Existing literature tests TAA-style route reuse (ReMoE), PID-controlled batching, and dynamic LoRA scaling (AdaLoRA) in isolation. Their interaction under dynamic local GPU load remains unbuilt in open-source AI.
* **Implementation Plan:** Build a closed-loop controller where system latency spikes (e.g., local agent initiating workspace search) trigger output signal $u(t)$ from a **PID Servo**. This signal dynamically adjusts multiple parameters in real time to enforce a strict **25ms SLA**:
  1. **TAA Velocity Threshold** ($\epsilon_{\text{TAA}}$): Forces model to reuse previous hidden-state projections for quiet layers.
  2. **Frustum Culling Radius** ($\epsilon_{\text{Cull}}$): Drops low-probability MoE experts or LoRA channels.
  3. **Billboard Impostor Depth**: Substitutes distant KV tokens or adapter matrices with pre-computed centroid vectors.

### Goal 1.2: Velocity-Driven Sparsity Captured Inside Static CUDA / HIP Graphs
* **Context & Gap:** Serving frameworks (`vLLM`, `SGLang`) enforce static memory shapes and static execution graphs for CUDA Graphs, prohibiting Python runtime control flow.
* **Implementation Plan:** Create a **Dynamic Foveated Fast-Pass** inside a captured CUDA/HIP Graph using pre-allocated zero-copy mask buffers.
  * Custom HIP kernels compute hidden-state velocity ($\Delta h_l$) at token $t$.
  * Write binary flags (`0` or `1`) directly to static GPU memory addresses.
  * Use masked GEMMs inside captured graphs to bypass LoRA matrix multiplications ($A \cdot B$) on quiet layers without incurring CPU launch overhead.

### Goal 1.3: Multi-Adapter "Billboard Impostors" in Memory-Constrained VRAM
* **Context & Gap:** Multi-LoRA frameworks (`S-LoRA`, `Punica`) stream full adapter weights into VRAM on demand, creating severe PCIe bus contention.
* **Implementation Plan:** Keep 10–20 domain adapters on system RAM while storing 1D "Billboard Centroid" vectors ($c_i \in \mathbb{R}^d$) in GPU VRAM (occupying $< 2\text{ MB}$ total).
  * Router passes intermediate states through cheap 2 MB Centroid Impostors.
  * If the full adapter is not in VRAM, execute the forward pass using the Billboard Impostor as an approximate stand-in while asynchronously prefetching full weights over the PCIe bus (Frustum Culling).

---

## 2. Multi-Level Execution Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ LEVEL 1: TRAINING / FINE-TUNING                             │
│ • Cross-Layer Tensor Factorization (Tucker/Tensor-Train)    │
│ • Compresses adapter weights & AdamW optimizer state        │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ LEVEL 2: MULTI-ADAPTER SWAPPING / ROUTING                   │
│ • "Billboard Impostors" (Centroids resident in VRAM)        │
│ • Asynchronous PCIe prefetching over the bus                │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ LEVEL 3: TOKEN-BY-TOKEN INFERENCE RUNTIME                   │
│ • "Foveated LoRA" (Velocity-driven layer skipping)          │
│ • Captures execution inside a static CUDA/HIP Graph         │
└─────────────────────────────────────────────────────────────┘
```

### Level 1: Training (Parameter & Optimizer Level)
Treat the entire stack of adapters across all layers as a single 3D Tensor ($L \times d_{\text{in}} \times d_{\text{out}}$) using Tucker or Tensor-Train decomposition. Projection factors are shared globally across network depth, while tiny core tensors handle layer-specific variations.

### Level 2: Adapter Swapping & Routing (Memory / PCIe Level)
10–20 domain adapters reside in system RAM. Each adapter maintains a 1D "Billboard Centroid" vector ($c_i \in \mathbb{R}^d$) representing mean bias resident in GPU VRAM ($< 2\text{ MB}$). Incoming tokens evaluate against centroid proxies; full weights are asynchronously prefetched over PCIe.

### Level 3: Token Inference (Forward Pass / Kernel Execution Level)
During autoregressive generation, a custom kernel measures hidden-state velocity ($\Delta h_l$) between tokens. Low state velocity (punctuation, whitespace) dynamically bypasses LoRA passes ($A \cdot B$); adapters fire exclusively when state velocity spikes (syntax shifts, tool payloads, critical logic).

---

## 3. Targeted Experimental Engines (High-Yield Combinations)

### Combination A: The "Zero-Jitter" Real-Time Engine
* **Formula:** `T-12 (CUDA Graphs)` + `T-15 (PID Servo)` + `T-14 (TAA Velocity Delta)` + `T-11 (GPTQ 4-bit)`
* **Mechanism:** Run lossless 4-bit base model inside a CUDA Graph decoder loop. The PID Servo measures real-time HIP kernel latency. If local background processes cause VRAM/PCIe contention, PID increases the TAA Velocity Threshold ($\epsilon$), skipping standard layer evaluations to maintain a steady token delivery rate.
* **Target Metric:** **80+ tok/s** throughput with **$< 4\text{ ms}$** standard deviation (eliminating token stutter during multi-turn agent sessions).

### Combination B: The "Infinite-Context Workspace" Engine
* **Formula:** `T-13 (Radix Prefix Cache)` + `T-04 (Hierarchical KV Mipmapping)` + `T-16 (Billboard Impostor Tokens)`
* **Mechanism:**
  1. Match shared system prompts and workspace configs via **T-13 Radix Caching** (11x TTFT speedup).
  2. Quantize intermediate context tokens (1,000–8,000) to 4-bit via **T-04 Mipmapping**.
  3. Compress distant background files ($> 8,000$ tokens) into **T-16 Billboard Impostor Tokens** (270x reduction).
* **Target Metric:** Reduce KV cache VRAM footprint from **$12+\text{ GB}$ down to $< 500\text{ MB}$**, enabling long-context monorepo analysis alongside active adapters on a single 24 GB GPU.

### Combination C: The "Foveated Opinionated Adapter" Engine
* **Formula:** `T-18 (Foveated LoRA)` + `Depth-Wise Tensor Factorization` + `OpenCode Agent Harness`
* **Mechanism:** Fine-tune 3D Tensor-Factorized LoRA adapters for development stack tools (`uv`, `FastAPI`, `Pydantic v2`). Use hidden-state velocity ($\Delta h_l$) to trigger adapter passes *only* when token streams require strict domain authority (e.g., editing `pyproject.toml`, tool payload generation), skipping generic prose.
* **Target Metric:** Reduce adapter FLOPs by **~50%** while maintaining sharp domain rule enforcement.

---

## 4. System Architecture Protocol & Success Criteria

```
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

### Quantitative Performance Targets & Bottleneck Solutions

| Execution Phase | Baseline Bottleneck | Architectural Solution | Quantitative Success Target |
| :--- | :--- | :--- | :--- |
| **Level 1: Training** | **Optimizer State VRAM Bloat:** AdamW states ($m, v$) take 2–4$\times$ more memory than weights, choking 24 GB GPUs. | **Cross-Layer Tensor Factorization:** Shares low-rank factors across depth, shrinking trainable params by 80–90%. | **$< 10\text{ MB}$ Optimizer Memory:** Train 30B+ base models locally on single 24 GB GPU without CPU offloading. |
| **Level 2: Swapping** | **PCIe Bus Stalls:** Swapping 50MB–200MB adapters between RAM and VRAM causes severe latency spikes during agent handoffs. | **Billboard Impostor Vectors:** Uses 2 MB centroid proxies resident in VRAM for immediate routing while prefetching full weights. | **Zero-Stall Handshakes:** Multi-adapter routing latency drops from $> 150\text{ ms}$ to $< 5\text{ ms}$. |
| **Level 3: Inference** | **CPU Launch Overhead & FLOP Waste:** Running LoRA passes on every layer/token saturates bandwidth and causes stutter. | **Foveated LoRA + CUDA/HIP Graphs:** Uses velocity ($\Delta h_l$) to skip quiet layers inside captured CUDA/HIP Graphs. | **$> 85\text{ tok/s}$ Throughput:** Cuts adapter FLOPs by 40–50% with $< 4\text{ ms}$ latency standard deviation. |

---

## 5. State-of-the-Art Benchmarking & Technology Replacement

### Performance Comparison Matrix

| Metric / Capability | Unsloth / PEFT (Current SOTA) | GaLore / AdaLomo (Research SOTA) | Proposed Integrated Architecture |
| :--- | :--- | :--- | :--- |
| **Optimizer State Memory** | ~100 MB–800 MB (Standard AdamW) | ~50 MB–200 MB (Low-rank gradient) | **$< 10\text{ MB}$** (Cross-Layer Tensor Factorization) |
| **Max Trainable Model (24 GB VRAM)** | 14B Models (QLoRA) | 14B–20B Models (Quantized) | **30B+ Models** (Native, zero offloading) |
| **Train Time (Local Adaptation)** | ~10–25 Minutes (14B Model) | ~20–45 Minutes | **$< 3\text{ Minutes}$** |
| **Inference Token Delivery** | 35–55 tok/s (Batch 1 Decode) | 20–40 tok/s | **$> 85\text{ tok/s}$** (Captured CUDA/HIP Graphs) |

### Replaced Stack Architecture

```
┌──────────────────────────────────────────────────────────┐
│ OLD / TRADITIONAL STACK                                  │
│ • Training:  Hugging Face `peft` + `bitsandbytes`        │
│ • Serving:   Standard PyTorch forward loops              │
│ • Swapping:  `S-LoRA` / `Punica` heavy VRAM managers     │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│ REPLACED BY THIS UNIFIED ENGINE                          │
│ • Training:  Cross-Layer Tensor LoRA (Tucker/TT)         │
│ • Serving:   ROCm / HIP CUDA Graph Decoder Engine        │
│ • Swapping:  Centroid Billboard Impostor Prefetching     │
│ • Control:   Closed-Loop PID Latency Governor            │
└────────────────────────────┬─────────────────────────────┘
```

* **Replaces `peft` + `bitsandbytes`:** Eliminates dynamic quantization overhead during training with tensor-factorized updates and native 4-bit frozen base models.
* **Replaces PyTorch Sequential Loops:** Replaces Python-level iteration with pre-captured **CUDA/HIP Graphs**, removing CPU-to-GPU launch latency.
* **Replaces Multi-LoRA Managers (`S-LoRA`, `Punica`):** Replaces dynamic VRAM allocation schemes with lightweight **Billboard Impostor Centroids** and asynchronous PCIe prefetching.

---

## 6. User Experience Transformation

* **Before (Standard 24 GB Setup):** Token stutter on multi-turn prompts; small sub-10B models exhibit sycophancy/weak adherence without huge system prompts; training larger adapters causes VRAM Out-Of-Memory (OOM) crashes during backward passes.
* **After (Unified Engine Enabled):**
  1. **Instant Adherence:** Local 4B models instantly enforce modern tooling (`uv`, `FastAPI`, `Pydantic v2`) and reject bad code patterns without bloated prompts.
  2. **Deterministic MCP Tool Calls:** Clean, parseable tool payloads and configuration edits without formatting failures.
  3. **80+ tok/s Fluid Generation:** Zero CPU launch stalls or latency jitter during long monorepo context processing.
  4. **Rapid Local Fine-Tuning:** Fine-tune specialized adapters for local codebases in $< 3\text{ minutes}$ with $< 10\text{ MB}$ optimizer VRAM.

---

## 7. Technical Feasibility & System Novelty Verification

### Technical Reality Check: Training 30B+ Models on 24 GB VRAM
* **Base Model Footprint:** 4-bit quantized 30B models take **15–18 GB VRAM**, leaving **6–9 GB VRAM** for activations, context, and gradients.
* **The True Barrier (Activation Memory):** While AdamW optimizer states for 100M adapter parameters take ~800 MB, backward pass activation memory for 30B models at $2048$ sequence length requires **8–14 GB VRAM**, triggering OOM crashes in standard frameworks (`Unsloth`, `peft`, `Axolotl`).

### Required Low-Level Kernel Innovations
Achieving these targets requires solving three C++/HIP kernel challenges simultaneously:
1. **Cross-Layer Tensor Contracting Backward Kernels:** Writing custom Triton/HIP backward kernels for Tucker-decomposed cross-layer adapters (avoiding PyTorch `einsum` overhead).
2. **Fused Activation Checkpointing inside CUDA/HIP Graphs:** Recomputing activation memory on-the-fly inside fused kernels to bypass the 14 GB activation memory peak.
3. **Hardware-Native AMD ROCm (`gfx1100`) Acceleration:** Implementing custom tensor contraction backward loops natively for AMD ROCm on RDNA3 architecture.

> **System Novelty Statement:** No existing production framework (`vLLM`, `llama.cpp`, `Unsloth`, `Axolotl`, `S-LoRA`) achieves these metrics or combines cross-technique closed-loop feedback, velocity-driven CUDA graph skipping, and billboard impostor prefetching into a single engine.

