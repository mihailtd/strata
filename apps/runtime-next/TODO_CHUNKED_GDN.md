# Scoping Document: Chunked-GDN Parallel Prefill for `runtime-next`

## Executive Summary & Problem Definition

Empirical telemetry captured via `rocprofv3 --kernel-trace --stats` on a genuine 401-token prefill on AMD Radeon RX 7900 XTX established the definitive breakdown of prefill latency in `runtime-next`:

| Bucket | Operation / Kernels | Total Time | Share |
|---|---|---|---|
| **Bucket C — GDN Loop (24 layers)** | `gdn_recurrent_decode` (50.5%) + `causal_conv1d_update` (3.8%) + `gdn_gate_beta` (3.4%) | **246.2 ms** | **57.7%** |
| **Bucket B — Attention Loop (8 layers)** | `attention_decode` (19.1%) + `rope` (2.6%) + `kv_cache_append` (1.0%) | **97.0 ms** | **22.7%** |
| **Bucket A — Batched GEMMs (hipBLAS)** | 5 Tensile projection kernels + `gemv_bf16` | **76.7 ms** | **18.0%** |
| **Bucket D — Elementwise Ops** | RMSNorm, SwiGLU, add, extract_range | **7.1 ms** | **1.7%** |

### The Core Finding
- **`gdn_recurrent_decode_bf16_kernel` alone accounts for 50.47% (215.1 ms) of all kernel execution time.**
- **Why**: $401 \text{ tokens} \times 24 \text{ layers} = \mathbf{9,624 \text{ sequential kernel launches}}$, each taking ~22.4 $\mu$s.
- While the single-token kernel is highly tuned (§93), executing it sequentially across 401 positions creates an immutable **~246 ms latency floor**.
- Rewriting Attention (Bucket B) into a batched GEMM saves at most ~85–90 ms (moving TTFT from 613 ms to ~525 ms). 
- **Chunking the GDN recurrent state update is the ONLY architectural lever capable of moving TTFT from 613 ms down into `llama.cpp`'s territory (111 ms).**

---

## 1. Mathematical Formulation: Sequential vs. Chunked Delta Rule

### A. The Current Sequential Formulation (One Step per Token)
For each head $h \in [0, 31]$ at position $t$, with $d_k = 128, d_v = 128$:
$$q_t = \frac{\text{L2}(Q_t)}{\sqrt{d_k}}, \quad k_t = \text{L2}(K_t), \quad \alpha_t = \exp(g_t)$$
$$S_t = \alpha_t S_{t-1} + k_t \otimes \left(\beta_t (v_t - S_{t-1}^T k_t)\right)$$
$$o_t = S_t q_t$$

Because $S_t$ depends on $S_{t-1}$, token $t$'s computation cannot launch until token $t-1$ completes.

### B. The Chunked Parallel Formulation (Flash-Linear-Attention / Chunked DeltaNet)
Instead of 401 individual steps, divide the prompt tokens $T$ into chunks of size $C$ (e.g., $C = 64$ or $C = 128$):
$$N_{\text{chunks}} = \left\lceil \frac{T}{C} \right\rceil = \left\lceil \frac{401}{64} \right\rceil = 7 \text{ chunks}$$

Within each chunk $c \in [0, N_{\text{chunks}}-1]$:

1. **Cumulative Decay Vectorization**:
   Compute the intra-chunk decay matrix $\Lambda \in \mathbb{R}^{C \times C}$ in parallel via prefix sum:
   $$\Lambda_{i,j} = \exp\left(\sum_{m=j+1}^i g_m\right) \quad (i \ge j)$$

2. **Intra-Chunk Gram Matrix (Masked GEMM)**:
   Compute the pairwise key interactions within the chunk:
   $$M_{i,j} = \beta_i (k_i^T k_j) \odot \Lambda_{i,j} \quad (i > j)$$
   This is a strictly lower-triangular $C \times C$ matrix computed by a single batched GEMM ($K_{\text{chunk}} \cdot K_{\text{chunk}}^T$).

3. **Intra-Chunk Delta Inversion**:
   The intra-chunk state updates satisfy:
   $$U_{\text{chunk}} = (I + M)^{-1} \left(V_{\text{chunk}} \odot \text{diag}(\beta)\right)$$
   Because $(I + M)$ is strictly lower-triangular with unit diagonal, $(I + M)^{-1}$ can be evaluated via block forward-substitution or truncated Neumann series directly in on-chip LDS (Local Data Share).

4. **Inter-Chunk Recurrent State Update (7 Steps instead of 401)**:
   The boundary state $S_c$ between chunks is updated only once per chunk:
   $$S_c = \Lambda_{C, 0} \cdot S_{c-1} + \sum_{i=1}^C \Lambda_{C, i} \cdot (k_i \otimes u_i)$$
   This reduces the global recurrent recurrence from **401 sequential steps to 7 sequential steps**.

5. **Parallel Output Computation**:
   Output tokens within the chunk are produced simultaneously:
   $$O_{\text{chunk}} = \underbrace{(Q_{\text{chunk}} \cdot S_{c-1}) \odot \Lambda_{\text{inter}}}_{\text{Inter-chunk contribution (GEMM)}} + \underbrace{\text{IntraChunkAttention}(Q, K, U)}_{\text{Intra-chunk contribution (Tiled GEMM)}}$$

---

## 2. Hardware Mapping & RDNA3 (Navi 31) Constraints

### A. Dimensions & State Geometry in Qwen3.5-4B
- **Number of GDN Layers**: 24 layers.
- **Number of Key Heads ($H_k$)**: 16.
- **Number of Value Heads ($H_v$)**: 32 (GQA ratio: 2 value heads per key head).
- **Head Dimension ($d$)**: 128 ($d_k = 128, d_v = 128$).
- **Recurrent State Size per Head**: $128 \times 128 \times 4 \text{ bytes (f32)} = \mathbf{64 \text{ KB}}$.

### B. On-Chip SRAM (LDS) Budget on RX 7900 XTX
- Each RDNA3 Dual Compute Unit (WGP) has **128 KB of LDS** (Local Data Share).
- A chunk size of $C = 64$:
  - $Q_{\text{chunk}}, K_{\text{chunk}}, V_{\text{chunk}}$ buffers: $64 \times 128 \times 2 \text{ bytes (bf16)} = 16 \text{ KB}$ each.
  - Intra-chunk $64 \times 64$ Gram matrix: $64 \times 64 \times 4 \text{ bytes} = 16 \text{ KB}$.
  - Total scratchpad needed per block: $\sim 48 \text{ KB}$, comfortably fitting within the 64 KB/128 KB LDS ceiling while allowing 2 concurrent warps per CU.

### C. Kernel Architecture Decomposition
To implement chunked GDN in `runtime-next`, three distinct stages are required:
1. `gdn_chunk_precompute.hip`: Parallel L2-norm of $Q, K$ + cumulative decay prefix-sum $\Lambda$ (Memory-bandwidth bound).
2. `gdn_intra_chunk_gemm`: Batched GEMMs computing $M = K K^T$ and intra-chunk output (Matrix-Core WMMA bound via `blas.rs`).
3. `gdn_inter_chunk_scan.hip`: Inter-chunk state propagation ($S_c = \text{decay} \cdot S_{c-1} + \Delta S$) across the 7 chunk boundaries (Latency bound, but only 7 steps).

---

## 3. Engineering Complexity, Risks & Payoff

### A. Architectural Complexity & Implementation Surface

| Task | Scope & Lines of Code | Technical Risk | Numerical Sensitivity |
|---|---|---|---|
| **Attention Batched GEMM Rewrite** | ~80 lines Rust (`model.rs`) | **Very Low** (Reuses existing `blas.rs` hipBLAS GEMM) | Exact (standard causal softmax) |
| **In-Place LoRA Porting (IPWF)** | ~250 lines Rust (`model_loader.rs`, `blas.rs`) | **Low** (Deterministic in-place matrix addition) | Bit-identical to base model |
| **Chunked GDN Parallel Prefill** | ~500 lines HIP C++ + ~300 lines Rust | **High** (Custom RDNA3 triangular solver in LDS) | High (sensitive to fp32/bf16 rounding in LDS) |

### B. Primary Technical Risks
1. **Numerical Drift in Intra-Chunk Inversion**:
   Inverting $(I + M)$ on lower-triangular blocks using floating-point in LDS can accumulate rounding errors if $\beta_i$ or decay values produce ill-conditioned matrices. Must be validated against PyTorch's reference implementation to $\text{L2-error} < 1e-4$.
2. **Chunk Boundary State Equivalence**:
   The final recurrent state $S_{\text{final}}$ after chunked prefill must match the sequential decode state bit-identically (or within fp32 epsilon), or subsequent single-token decode steps will diverge on token 402.

### C. Projected Performance Payoff
- **Kernel Launch Reduction**: $9,624 \text{ launches} \longrightarrow 168 \text{ launches}$ (**57× fewer launches**).
- **Execution Time on 401 Tokens**:
  - Current Sequential GDN: **246.2 ms**
  - Projected Chunked GDN: **~30–45 ms** (saving ~200–215 ms).
- **Combined Impact on TTFT (with Attention Rewrite)**:
  - Baseline TTFT (§96): **613 ms**
  - Attention rewrite savings: **~85 ms**
  - Chunked GDN savings: **~205 ms**
  - **Projected Final TTFT**: **~150–180 ms** (bringing `runtime-next` within striking distance of `llama.cpp`'s 111 ms and beating `Ollama`'s 137 ms).

---

## 4. Phased Implementation Roadmap (If Pursued)

### Phase 1: Algorithmic Verification against Oracle Reference
- Implement an isolated CPU reference function (`chunked_gdn_reference(q, k, v, g, beta, chunk_size=64)`) in Python or Rust.
- Verify numerical outputs against `transformers`' sequential `torch_recurrent_gated_delta_rule` across random inputs and real Qwen3.5-4B prompt activations.
- Establish the tolerance threshold for fp32 vs. bf16 intermediate storage.

### Phase 2: Hand-Written RDNA3 Chunked Kernel Implementation
- Build `src/kernels/gdn_chunked.hip`.
- Implement Phase 1 (decay prefix sums) and Phase 2 (triangular forward-substitution in LDS for $C=64$).
- Wire into `src/kernels.rs` and write an isolated decisive unit test comparing single-chunk execution against `gdn_recurrent_decode_bf16_kernel`.

### Phase 3: Integration into `model.rs` & Multi-Chunk Validation
- Replace lines 1210–1275 in `src/model.rs` with `raw::gdn_chunked_forward`.
- Verify full test suite passes (63/63 tests).
- Re-run `rocprofv3 --kernel-trace --stats` on the 401-token prompt to measure the actual reduction in Bucket C.

---

## 5. Decision Framework & Strategic Trade-Offs

1. **Attention Batched GEMM Rewrite**:
   - **Complexity**: Low. Reuses existing `blas.rs` GEMM wrappers with zero new hand-written HIP kernels.
   - **Payoff**: Banks an honest, verified ~85 ms reduction in Bucket B, moving TTFT from 613 ms to ~525 ms.
2. **Chunked GDN Parallel Prefill**:
   - **Complexity**: High. Requires writing custom lower-triangular matrix inversion in LDS on RDNA3 and managing inter-chunk recurrence state handoff.
   - **Payoff**: The only structural path to reducing Bucket C (246.2 ms) to <45 ms, enabling sub-200 ms TTFT.
3. **In-Place LoRA Porting (IPWF)**:
   - **Complexity**: Low-to-Medium. Deterministic weight matrix addition with zero recurrent numerical sensitivity.
   - **Payoff**: Orthogonal to prefill latency—unlocks the multi-expert domain adapter fleet on `runtime-next` at 95 tok/s, creating a functional capability advantage that `llama.cpp` lacks.
4. **Architectural Status**:
   This document serves as the formal technical blueprint for Chunked GDN. It isolates the mathematical requirements and hardware bounds, ensuring that any implementation follows the strict empirical verification protocols established across §80–§97.
