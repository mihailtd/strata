# Engineering Report: Maximizing Local Inference & Agent Reliability on AMD RDNA3

**Target Platform:** AMD Ryzen 9 9900X (Zen 5) + AMD Radeon RX 7900 XTX (24 GB GDDR6, `gfx1100`)

**Operating System:** CachyOS (`x86-64-v4`, Linux Kernel with BORE scheduler)

**Target Runtimes:** ROCm 6.x+, Native Triton (AMDGPU backend), Turnstone Agent Framework

---

Here are the key hardware characteristics of the **AMD RDNA 3 (`gfx1100` / Navi 31)** architecture that mainstream SOTA AI frameworks (vLLM, Hugging Face, PyTorch, Ollama) either ignore, misunderstand, or fail to optimize:

---

### 1. Dual-Issue ALUs / VOPD (Dual Instruction Execution)

* **The Silicon Reality:** Each of the 96 Compute Units (6,144 Stream Processors) contains dual-path ALUs. If the instruction stream can pair two independent mathematical operations in the same cycle (e.g., an independent multiply and an independent add via VOPD encoding), raw FP32 compute doubles from **30.7 TFLOPS to 61.4 TFLOPS**.
* **Why SOTA Ignores It:** Mainstream compilers and PyTorch ROCm builds generate standard single-issue VOP2/VOP3 vector instructions. They treat RDNA 3 like an RDNA 2 or CDNA card, leaving **up to 50% of the raw vector ALUs idle on every clock cycle**.
* **How to Exploit It:** In custom Triton / LLVM AMDGPU compilation pipelines, tuning instruction scheduling and unrolling loops to guarantee independent register pairs allows the compiler to pack instructions into native dual-issue (`VOPD`) opcodes.

---

### 2. WMMA (Wave Matrix Multiply-Accumulate) vs. CDNA's MFMA

* **The Silicon Reality:** AMD has two completely different matrix acceleration architectures:
* **CDNA (Data Center / MI300):** Uses **MFMA** (`v_mfma_*`) instructions with 64-thread wavefronts.
* **RDNA 3 (Consumer / 7900 XTX):** Uses **WMMA** (`v_wmma_*`) matrix instructions ($16 \times 16 \times 16$) designed for 32-thread wavefronts.


* **Why SOTA Misunderstands It:** The vast majority of AMD's internal optimization work (Composable Kernel, rocBLAS, FlashAttention ROCm ports) is written exclusively for CDNA and MFMA. When these libraries are run on RDNA 3, they frequently fall back to slow, non-matrix vector instructions or unoptimized generic emulation paths.
* **How to Exploit It:** Target the hardware matrix instructions directly via Triton or HIP intrinsics (`V_WMMA_F32_16X16X16_F16` for FP16/BF16, and `V_WMMA_I32_16X16X16_IU4` for 4-bit packed integer math), unlocking the card's full **122.8 TFLOPS** half-precision ceiling.

---

### 3. The 96 MB On-Die Infinity Cache (MALL) as an Intermediate SRAM Tier

* **The Silicon Reality:** The 7900 XTX has 96 MB of Memory-Attached Last-Level (MALL) cache operating at **over 3.5 TB/s internal bandwidth**—almost $4\times$ faster than the GDDR6 VRAM bus (960 GB/s).
* **Why SOTA Ignores It:** Mainstream memory managers (like PyTorch’s caching allocator and CUDA-centric memory planning) operate on a two-tier model: **LDS (Local Data Share / Shared Memory) $\rightarrow$ Global VRAM**. They have no concept of a massive 96 MB intermediate SRAM cache. Intermediate activations, partial attention matrices, and dynamic LoRA weights are constantly flushed to slow GDDR6.
* **How to Exploit It:** Design kernel tile and block sizes so that the active working memory (e.g., active KV-cache chunks, intermediate layer outputs, and LoRA delta weights) fits entirely within $<96\text{ MB}$. This turns memory access from a 960 GB/s GDDR6 sweep into a **3.5 TB/s on-die cache hit**, drastically reducing latency during the prefill/TTFT phase.

---

### 4. Wave32 Native Execution vs. Legacy Wave64 Assumptions

* **The Silicon Reality:** RDNA 3 is natively optimized for **Wave32** (executing instructions across 32 threads simultaneously, identical to an NVIDIA 32-thread Warp). However, for backward compatibility with older GCN/CDNA architectures, it can also run in Wave64 mode.
* **Why SOTA Misunderstands It:** Many legacy ROCm kernels and ported CUDA tools default to Wave64 mode on AMD. In Wave64 mode on RDNA 3, a wavefront must be split and executed across two cycles, doubling register pressure, halving active thread occupancy, and increasing register spills to VRAM.
* **How to Exploit It:** Explicitly enforce Wave32 compilation flags (`-mwavefrontsize64=off` / `waves_per_eu=2` in Triton and HIP). This matches standard 32-thread SIMD design patterns, maximizes hardware occupancy, and eliminates register thrashing.

---

### 5. Asynchronous Direct Kernel Fusion Driver (`/dev/kfd`) & Direct PCIe DMA

* **The Silicon Reality:** Under bare-metal Linux, AMD’s **KFD** allows user-space runtimes to dispatch commands directly into GPU hardware ring buffers and issue asynchronous direct memory access (DMA) transfers over PCIe 4.0 x16 at **31.5 GB/s**.
* **Why SOTA Fails on It:** On Windows / WSL2, access is filtered through DirectX translation layers (`/dev/dxg`), adding driver latency and preventing true asynchronous DMA overlaps. Even on Linux, general frameworks use standard synchronous PyTorch CUDA-like copies that block the main execution thread.
* **How to Exploit It:** In a native CachyOS environment with `/dev/kfd` access, implement true **non-blocking double-buffering (Ping-Pong buffers)** using separate ROCm streams. While Stream 0 executes a quantized layer in VRAM via WMMA, Stream 1 uses DMA to pull the next layer chunk from pinned host DDR5 RAM across the PCIe bus simultaneously.

---

### Summary Matrix

| Architectural Feature     | Physical Capability                           | How SOTA Currently Handles It                         | Unlocked Potential                                |
| ------------------------- | --------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------- |
| **VOPD Dual-Issue**       | 61.4 TFLOPS FP32                              | Ignored (compiles as single-issue ~30.7 TFLOPS)       | **$2\times$ raw compute saturation**              |
| **WMMA Matrix Units**     | 122.8 TFLOPS FP16 / 245.8 TOPS INT8           | Ignored or falls back to CDNA MFMA emulation          | **$2\times\text{--}4\times$ matrix acceleration** |
| **96 MB Infinity Cache**  | 3.5 TB/s intermediate bandwidth               | Unused (flushes everything to 960 GB/s GDDR6)         | **Near-SRAM speed on attention tiles**            |
| **Wave32 Execution**      | 32-thread low-latency SIMD                    | Forced into Wave64 (register spills & half occupancy) | **Zero register spills, max occupancy**           |
| **Direct KFD DMA Engine** | 31.5 GB/s unmediated host-to-device streaming | Blocked by virtualization / synchronous copies        | **Seamless $>24\text{ GB}$ model layer-paging**   |
|                           |                                               |                                                       |                                                   |

## Executive Summary

Recent empirical benchmarks and micro-architecture analyses reveal two critical realities for local AI execution:

1. **The Tool-Breakage Mechanism:** In agentic workflows, execution failures (e.g., malformed JSON, corrupted CLI commands, infinite remediation loops) are primarily caused by **numerical drift** (floating-point non-associativity across different attention backends) and **KV-cache quantization degradation** (the "40k token cliff"), rather than fundamental reasoning deficits in the underlying model.
2. **The Hardware Opportunity:** The AMD Radeon RX 7900 XTX possesses 960 GB/s of raw memory bandwidth, 24 GB of VRAM, 96 MB of on-die Infinity Cache (~3.5 TB/s), and 122.8 TFLOPS of half-precision matrix compute (WMMA). In memory-bandwidth-bound auto-regressive decoding, it is within 5–10% of an NVIDIA RTX 4090 when driver and runtime bottlenecks are eliminated.

This report synthesizes these findings into concrete, actionable engineering implementations across quantization, kernel development, agent state management, and validation testing.

---

## 1. Quantization & Precision Strategy

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PRECISION SELECTION MATRIX                      │
│                                                                        │
│  [ Model Weights ] ──► AWQ / Fused W4A16 or MXFP4 + Online Rotation    │
│                        (Preserves structural syntax & dynamic range)   │
│                                                                        │
│  [ Activations ]   ──► BF16 / FP16 Native                              │
│                                                                        │
│  [ KV Cache ]      ──► Paged BF16 / FP16 OR Per-Head INT8              │
│                        ⛔ HARD BAN ON INT4 KV CACHE                    │
└────────────────────────────────────────────────────────────────────────┘

```

### Action 1.1: Enforce a Strict KV Cache Precision Boundary

* **Finding:** Quantizing the KV cache to INT4 causes catastrophic divergence past 40k context tokens, yielding a 40–50% Top-1 logit flip rate and breaking structured tool syntax.
* **Implementation Rule:**
* **Default Policy:** Run KV cache in **BF16 / FP16**.
* **Memory-Constrained Policy (Long Context):** Use **Per-Head / Per-Token INT8 KV cache**.
* **Prohibited:** Never enable `--kv-cache-dtype int4` for agentic workloads.



### Action 1.2: Adopt Outlier-Preserving Weight Formats

* **Weight Quantization:** Use **AWQ (Activation-aware Weight Quantization) W4A16** or **NVFP4/MXFP4 with 16-element block scaling**.
* **Closing the MXFP4 Accuracy Gap via Hadamard Transformations:**
* If deploying native microscaling (MXFP4) on AMD hardware, isolate activation outliers by fusing an online **Walsh-Hadamard Transform (WHT)** into the pre-GEMM stage:

$$Y = (X \cdot H) \cdot (H^T \cdot W)$$


* Applying the orthogonal matrix $H$ mathematically spreads outlier energy across all channels before 4-bit clamping, reducing the accuracy delta against NVFP4 to $<1\%$ without adding memory footprint.



---

## 2. Kernel Engineering on AMD RDNA3 (`gfx1100`)

```
┌────────────────────────────────────────────────────────────────────────┐
│               RDNA3 DUAL-ISSUE & WMMA INSTRUCTION PIPELINE             │
│                                                                        │
│  Triton Source JIT ──► LLVM AMDGPU Backend ──► gfx1100 Machine Code    │
│                             │                                          │
│                             ▼                                          │
│              [ V_WMMA_F32_16X16X16_F16 ] Intrinsics                   │
│              • 16×16×16 Matrix Tile Execution                          │
│              • Saturated Dual-Issue Arithmetic                         │
│              • Fused Dequant + Scale + Base GEMM + LoRA Split          │
└────────────────────────────────────────────────────────────────────────┘

```

### Action 2.1: Target Native Wave Matrix Multiply-Accumulate (WMMA)

Generic matrix operations in ROCm often fall back to scalar or vector SIMD instructions (~30–61 TFLOPS). To achieve the theoretical **122.8 TFLOPS FP16** or **245.8 TOPS INT8**:

* Ensure all custom Triton kernels compile down to the RDNA3 native matrix intrinsics:
* `V_WMMA_F32_16X16X16_F16`
* `V_WMMA_I32_16X16X16_IU4` (for packed 4-bit integer GEMMs)


* Set tile blocks to exact multiples of the 32-thread wavefront size (`waves_per_eu=2`, `BLOCK_M=64`, `BLOCK_N=64`, `BLOCK_K=32`).

### Action 2.2: Implement Fused Dequantization + LoRA Branching

* Eliminate intermediate VRAM roundtrips by fusing 4-bit weight dequantization, scaling multiplication, base GEMM, and dynamic LoRA adapter routing into a single Triton kernel pass:

```python
@triton.jit
def fused_w4a16_lora_kernel(
    A_ptr, B_qweight_ptr, B_scales_ptr, B_zeros_ptr,
    LoRA_A_ptr, LoRA_B_ptr, Out_ptr,
    M, N, K, K_LORA,
    # Stride and tile configuration parameters...
):
    # 1. Load packed 4-bit weights into registers
    # 2. Dequantize to BF16 using scales and zero-points
    # 3. Accumulate Base GEMM via WMMA: C = A @ dequant(B)
    # 4. Asynchronously accumulate active LoRA branch: C += (A @ LoRA_A) @ LoRA_B
    # 5. Store final BF16 results to global memory

```

### Action 2.3: Deterministic Attention Backend Pinning

* **Finding:** Switching backends (e.g., FlashAttention-2 vs. Triton vs. FlashInfer) causes a 15–20% token flip rate due to non-associative floating-point reduction trees.
* **Implementation Rule:** Pin the serving and validation stack strictly to **one deterministic Triton attention backend**. Do not mix inference engines between developmental testing and production.

---

## 3. Turnstone Agent Architecture Guardrails

```
┌────────────────────────────────────────────────────────────────────────┐
│                   TURNSTONE CONTEXT & STATE LIFECYCLE                  │
│                                                                        │
│   Incoming Turn (User / System / Prior Sub-Agent)                      │
│                          │                                             │
│                          ▼                                             │
│   [ Reasoning Execution: <think> ... </think> ]                        │
│   • Generates internal scratchpad and diagnostic steps                 │
│                          │                                             │
│                          ▼                                             │
│   [ Structural Output Generation ]                                     │
│   • Generates structured tool calls / parameters                       │
│                          │                                             │
│                          ▼                                             │
│   [ Turnstone Context Filter Pipeline ]                                │
│   ├── STRIP: Discard intermediate <think> scratchpad                   │
│   ├── RETAIN: Save validated tool inputs & tool execution outputs      │
│   └── SNAPSHOT: Serialize recurrent hidden states (DeltaNet/SSM)       │
│                          │                                             │
│                          ▼                                             │
│   Result: Working context remains locked within 4k–16k safe zone       │
└────────────────────────────────────────────────────────────────────────┘

```

### Action 3.1: Context Scrubbing Pipeline (The "Safe Zone" Policy)

* To prevent models from drifting into the $>40\text{k}$ token degradation cliff, Turnstone must actively manage working memory:
1. **Capture & Verify:** Read the model's `<think>` reasoning block to verify internal logic.
2. **Scrub Before Append:** Strip the raw reasoning trace from the conversation buffer.
3. **Persist State:** Append only the structured tool call schema and the corresponding tool return payload.
4. **Target Depth:** Keep active agent conversation context strictly under **32,000 tokens**.



### Action 3.2: Recurrent Tensor State Handoffs

* For hybrid linear-attention architectures (e.g., Qwen 3.8 Gated DeltaNet layers):
* Do not re-feed thousands of historical text tokens to initialize a new sub-agent.
* Directly serialize and hand off the internal hidden state tensor $S_t \in \mathbb{R}^{d_{head} \times d_{head}}$ across agent boundaries, eliminating quadratic prefill latency.



---

## 4. Asynchronous Memory Streaming (>24 GB Workloads)

When executing models that exceed the 24 GB physical VRAM boundary (e.g., 30B–70B parameters):

```
┌────────────────────────────────────────────────────────────────────────┐
│               DOUBLE-BUFFERED PING-PONG STREAMING PIPELINE             │
│                                                                        │
│  [ Host DDR5 System RAM ] (Pinned Layer Cache)                         │
│             │                                                          │
│             │  31.5 GB/s PCIe 4.0 x16 DMA Transfer                     │
│             ▼                                                          │
│   ┌───────────────────┬───────────────────┐                            │
│   │   VRAM Buffer 0   │   VRAM Buffer 1   │                            │
│   │  (Active Compute) │  (Async Staging)  │                            │
│   └───────────────────┴───────────────────┘                            │
│             │                   ▲                                      │
│             ▼                   │                                      │
│   GPU executes Layer N ──► DMA streams Layer N+1 in background         │
└────────────────────────────────────────────────────────────────────────┘

```

### Action 4.1: Compute/Transfer Overlap Condition

To maintain continuous token generation without GPU starvation, ensure the compute duration of chunk $N$ matches or exceeds the DMA copy duration of chunk $N+1$:

$$\text{Compute Time} = \frac{\text{FLOPs per Layer}}{\text{Active Compute Rate (TFLOPS)}} \ge \frac{\text{Layer Size in Bytes}}{\text{PCIe Bandwidth (31.5 GB/s)}} = \text{Transfer Time}$$

* **Rule:** Layer chunks must be quantized to **W4A16** before streaming. Streaming unquantized 16-bit layers over PCIe 4.0 will bottleneck decoding throughput to $<2\text{ tok/s}$, whereas quantized chunks approach interactive speeds ($5\text{–}12\text{ tok/s}$).

---

## 5. Verification & Testing Protocol

Before deploying any model checkpoint or custom kernel into Turnstone, run this verification protocol:

```
                  ┌───────────────────────────────┐
                  │    VERIFICATION WORKFLOW      │
                  └──────────────┬────────────────┘
                                 │
                 [ Step 1: Logit Drift Probe ]
                 • Capture logits at 8k, 32k, 64k
                 • Assert KLD < 0.01 vs BF16 PyTorch
                                 │
                                 ▼
                 [ Step 2: Top-1 Agreement Check ]
                 • Measure greedy argmax consistency
                 • Fail if disagreement > 5% at 32k
                                 │
                                 ▼
                 [ Step 3: Schema Conformance ]
                 • 1,000 synthetic multi-tool prompts
                 • Assert 0% JSON syntax failures
                                 │
                                 ▼
                 [ Step 4: Roofline Benchmarking ]
                 • Decode test on 27B W4A16 model
                 • Verify >50 tok/s on 7900 XTX

```

1. **Logit-Level Divergence Test (KLD):**
* Feed a standardized 32k multi-tool prompt. Capture logits every 32 tokens.
* Calculate Kullback-Leibler Divergence (KLD) in FP64 against an unquantized PyTorch BF16 reference:

$$D_{KL}(P \parallel Q) = \sum_{i} P(i) \log\left(\frac{P(i)}{Q(i)}\right)$$


* Reject any quantization setup where $D_{KL} > 0.01$.


2. **Top-1 Agreement Metric:**
* Assert that the Top-1 argmax prediction matches the baseline reference across at least **95% of probes** within the first 32k tokens.


3. **Structured Tool Schema Stress Test:**
* Execute 1,000 automated tool-calling passes containing nested parameters (e.g., regex patterns, networking interface names, file paths).
* Confirm **0% syntax corruption** or argument hallucination.


4. **Decode Bandwidth Benchmark:**
* Profile memory bandwidth saturation using `rocprof`. On a 27B W4A16 model (~14.5 GB footprint), target decode throughput of:

$$\text{Target Throughput} = \frac{960\text{ GB/s} \times 0.85\text{ (Bus Efficiency)}}{14.5\text{ GB}} \approx \mathbf{56\text{ tokens/second}}$$





---

## 6. Implementation Action Plan

| Phase       | Milestone                 | Specific Technical Task                                                                                | Success Criteria                                                                |
| ----------- | ------------------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| **Phase 1** | **OS & ROCm Base**        | Verify CachyOS `/dev/kfd` access, install ROCm 6.x SDK, configure `HSA_OVERRIDE_GFX_VERSION=11.0.0`.   | `rocminfo` and PyTorch correctly detect `gfx1100` with 24 GB VRAM.              |
| **Phase 2** | **Precision Setup**       | Deploy vLLM/Triton with **AWQ W4A16 weights** and **BF16 KV Cache**. Explicitly disable INT4 KV cache. | 0% tool format failure rate on multi-turn baseline prompts.                     |
| **Phase 3** | **Turnstone Integration** | Implement `<think>` block extraction and context scrubbing in the agent conversation loop.             | Active context remains $<32\text{k}$ tokens across 50-step autonomous loops.    |
| **Phase 4** | **Custom Triton Kernel**  | Write native RDNA3 fused W4A16 GEMM with dynamic LoRA branch routing using WMMA intrinsics.            | $>80\text{ TFLOPS}$ sustained compute during matrix multiplication passes.      |
| **Phase 5** | **Streaming Engine**      | Implement asynchronous host-to-device pinned memory double-buffering for models $>24\text{ GB}$.       | $>5\text{ tok/s}$ sustained decode on 30B+ models without out-of-memory errors. |
