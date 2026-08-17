# 🔬 Benchmarks, Geometric Probes & Hardware Verification Suites

> **System**: **Autonomous Runtime & Speculative Execution Engine**  
> **Target Hardware**: AMD ROCm (`gfx1100` / RX 7900 XTX 24GB VRAM)  
> **Core Model**: `Qwen/Qwen3.5-4B` Hybrid Linear-Attention (`bfloat16`)

This directory houses the formal empirical benchmarks, mathematical proofs, and hardware validation suites for the Autonomous Runtime Engine.

---

## 🧭 Innovation & Classification Legend

| Tier Badge | Definition | Core Examples in this Stack |
| :--- | :--- | :--- |
| **🚀 Genuine Discovery** | Fundamental zero-to-one mathematical or theoretical finding discovered during research. | `bfloat16` Mantissa ULP Inverse Scaling Law, Pre-Flight SVD Subspace Probe, Times-Above-Chance Metric. (The APSP VRAM State Router was listed here and has been **retired** — the cost model is destination-only, so the shortest-path solve was provably a no-op. See [`superseded/apsp_floyd_warshall/`](superseded/apsp_floyd_warshall/).) |
| **🔥 Applied Practice** | Adapting an established concept from outside LLMs to solve an active, unsolved LLM runtime problem. | **Transactional Memory State Checkpointing (from database write-ahead logging / game state snapshots) applied to the LLM runtime** (52.5 MB Recurrent Snapshot Buffer & Pristine State Buffer $W_0$). |
| **⭐ Industry Standard** | Standard engineering patterns used correctly and rigorously. | In-Place Low-Rank Factoring, CUDA Graph Replay, Continuous Batch Scaling ($B=1\dots 64$), A/B Controlled Evals. |

---

## 🏛️ Architectural Rationale: Native `bfloat16` Multi-Expert vs. Giant Quantized Generalist

A core architectural pillar of this engine is serving an unquantized 4B model in native `bfloat16` paired with **In-Place Low-Rank Weight Folding** rather than quantizing a larger 14B/32B model into 4-bit (NF4/AWQ):

1. **In-Place Low-Rank Weight Folding is Impossible in 4-Bit**: 
   * In native `bfloat16`, expert swapping is a blazing-fast BLAS matrix operation: $W_{\text{live}} \leftarrow W_0 + \frac{\alpha}{r} (U \times V)$ executing in **18.08 ms** with **0 bytes transient VRAM churn**.
   * In 4-bit packed integer formats, non-linear quantization scales prevent in-place linear tensor fusion, forcing runtimes to use PEFT wrappers (which destroy decode speed by ~45%).
2. **Decode Velocity & Speculative Decoding**:
   * Single-stream decode latency governs user experience in multi-tool agent loops. 14B 4-bit models decode at ~15–22 tok/s due to per-token dequantization overhead.
   * Our native `bfloat16` 4B engine with **Native MTP Speculation** decodes at **55.71 tok/s (2.20x speedup)** with sub-18ms token latency.
3. **Fleet Density (200+ Resident Experts on 1 GPU)**:
   * A 14B model consumes 10–18 GB just to load, fitting only 1 generalist model in 24GB VRAM.
   * Our engine requires only **8.52 GB for the base model** + **5.12 GB for the pristine buffer**, leaving ~8 GB of standby VRAM to hold **over 200 resident specialized domain experts** simultaneously (at 42.5 MB per expert).

---

## 💡 Parameter Absorption vs. System Prompt Bloat (The Agentic Tooling Dilemma)

When multi-agent coding tools (e.g., OpenCode, Cursor, Cline) interact with LLMs, they often inject **massive 2,000–6,000 token system prompts** containing verbose tool descriptions, complex schemas, and rigid reasoning instructions. On small-to-medium models, this causes severe degradation:

1. **Attention Dispersion**: Transformers have finite attention capacity per token. When 90% of the prompt is boilerplate system instructions, attention to the user's actual question diminishes exponentially, causing hallucinations and tool confusion.
2. **Parameter Absorption (Weights > Prompts)**: Instead of explaining domain rules across 3,000 prompt tokens, our engine bakes domain capabilities directly into **42.5 MB resident LoRA adapters**. System prompts shrink from 3,000 tokens to 20 tokens, freeing 95% of the context window.
3. **Agent Protocol Experts (e.g., OpenCode Expert)**: Training a dedicated protocol expert on multi-turn tool interaction traces (`<tool_call>`, `<tool_response>`, diff application) allows the model to natively internalize agentic tool-calling syntax. Using `WeightFoldingEngine.activate_many()`, an **OpenCode Protocol Expert** can be additively stacked with a **PostgreSQL Domain Expert**, giving the model bit-exact tool syntax and deep domain expertise simultaneously!

---

## ⚠️ Methodological Pitfalls & Hard-Won Lessons (Negative Examples to Avoid)

To ensure scientific rigor and prevent regressions, all future experiments **MUST** strictly adhere to the following hard-won lessons:

### 1. ⚠️ The Micro-Burst Token Horizon Trap (Distorting Steady-State)
* **The Pitfall**: Running short 32-token or 48-token generation runs to measure inference velocity.
* **Why It Failed**: Short sequences are heavily dominated by initial kernel launch and PyTorch dispatcher startup latency. On a 32-token run, speculative decoding showed an anemic ~1.14x speedup, and PEFT wrappers hid their memory bottleneck.
* **The Rule**: **All inference benchmarks must run at least 256 tokens.** Full steady-state generation reveals true hardware velocity: **2.20x net speedup at $K=6$** and **+82.1% speedup** for in-place weight folding over PEFT wrappers.

### 2. ⚠️ The Un-warmed JIT Compilation Trap (The "54% Prefill" Illusion)
* **The Pitfall**: Measuring prompt prefill vs. decode latency without per-shape GPU kernel warmups before starting the timer.
* **Why It Failed**: First-time Triton JIT compilation took ~33 seconds inside the timed loop, creating the false illusion that prefill consumed 54% of generation time.
* **The Rule**: **Always run a 2-token per-shape warmup pass outside the timed region.** With warmed kernels, prefill takes only **5.9% of wall time at 8k context**, proving decode governs **94.1%** of user latency.

### 3. ⚠️ The Cross-Precision Quantization Seam (4-Bit Train $\to$ `bfloat16` Serve)
* **The Pitfall**: Training LoRA adapters on a 4-bit NF4 quantized base model (`load_in_4bit=True`) and then folding them into unquantized `bfloat16` base weights at serving time.
* **Why It Failed**: Adapters learned deltas that compensated for quantization truncation rather than true domain knowledge. Folding them into `bfloat16` degraded domain accuracy by **3.36 percentage points**.
* **The Rule**: **Always train adapters in the exact same precision as the serving engine** (Unified M2 `bfloat16` + Liger fused kernels).

### 4. ⚠️ Direct SFT on Speculative Draft Heads (Destroying Alignment)
* **The Pitfall**: Fine-tuning the native checkpoint MTP draft head on domain datasets with next-token prediction loss.
* **Why It Failed**: The draft head became "domain fluent" but diverged from the base model's internal hidden state representations, causing draft acceptance ($\tau$) to collapse from **2.45 down to 1.53**.
* **The Rule**: **Draft heads must match backbone representations via logit distillation, never independent domain SFT.**

### 5. ⚠️ The High-Dimensional Grassmannian Projection Illusion
* **The Pitfall**: Interpreting small raw subspace angles between two adapters as evidence of domain interference.
* **Why It Failed**: In high-dimensional spaces ($D=2560$), two completely random rank-8 Gaussian matrices have an expected canonical overlap of $E_{\text{chance}} = \frac{r}{D} \approx 0.0031$.
* **The Rule**: **Always normalize projection metrics by Times-Above-Chance ($\frac{\text{overlap}}{r/D}$).** Real adapters show near-perfect orthogonality ($1.10\text{--}1.28\times$ chance).

### 6. ⚠️ The Missing Early CUDA GPU Context Trap
* **The Pitfall**: Importing `transformers` before initializing PyTorch's CUDA device context.
* **Why It Failed**: `flash-linear-attention` (`fla`) caches its hardware device probe at module import. If no GPU context exists, it silently latches to a slow CPU fallback, causing verification latency to spike from **1.19x back to 2.84x**.
* **The Rule**: **Always touch CUDA (`torch.zeros(1, device="cuda"); torch.cuda.synchronize()`) before importing `transformers`.**

---

## 🏛️ Subsystem Benchmark Index

### 1. [`runtime/`](runtime/) — The Runtime Engine
* **[`runtime/router/vram_state_routing/`](runtime/router/vram_state_routing/)**: **⭐ VRAM State Router**: SLA-bounded cluster scheduling over a *measured* destination-only cost model. Cuts GPU swaps 25.5 → 16.0 end-to-end, but swap overhead is only 0.86% of wall clock so mean latency is unchanged (95% CI [−1481, +1417]); the real win is SLA violations (98% → 79.5% at ρ=1.5). **No graph solver** — see [`superseded/apsp_floyd_warshall/`](superseded/apsp_floyd_warshall/).
* **[`runtime/folding/`](runtime/folding/)**: In-place weight folding engine eliminating 100% of PEFT wrapper overhead (+82.1% speedup over wrapped PEFT).
* **[`runtime/memory/factor_residency/`](runtime/memory/factor_residency/)**: 42.5 MB low-rank factor standby representation (200.6x compression), enabling up to 216 concurrent domain experts on 24GB VRAM.
* **[`runtime/memory/pristine_state_buffer/`](runtime/memory/pristine_state_buffer/)**: Transactional Memory State Checkpointing preserving $L_\infty = 0.00$ drift across infinite expert hot-swaps.
* **[`runtime/memory/cuda_graph/`](runtime/memory/cuda_graph/)**: Pointer-stable weight slots enabling zero-recapture CUDA Graph replay.
* **[`runtime/memory/zero_recapture_swapping/`](runtime/memory/zero_recapture_swapping/)**: Multi-turn expert swapping under single-capture CUDA Graphs with 0 bytes transient churn.
* **[`runtime/speculative/mtp_speculative/`](runtime/speculative/mtp_speculative/)**: Native MTP speculative engine with 52.5 MB recurrent rollback, delivering **2.20x net speedup at $K=6$**.
* **[`runtime/speculative/speculation_matrix/`](runtime/speculative/speculation_matrix/)**: 180-run 3x3 domain matrix audit routing speculative execution.
* **[`runtime/performance/prefill_vs_decode/`](runtime/performance/prefill_vs_decode/)**: Amdahl's Law audit proving decode governs 94.1% of user latency.
* **[`runtime/performance/fla_triton_kernels/`](runtime/performance/fla_triton_kernels/)**: ROCm Triton kernel acceleration flattening verification latency to 1.19x.
* **[`runtime/performance/batch_scaling/`](runtime/performance/batch_scaling/)**: High-batch throughput scaling up to $B=64$ (611 tok/s).

### 2. [`factory/`](factory/) — The Factory & Geometric Probes
* **[`factory/geometry/alpha_sweep/`](factory/geometry/alpha_sweep/)**: `bfloat16` Mantissa ULP Inverse Scaling Law and $\alpha=128$ absorption threshold.
* **[`factory/geometry/preflight_svd_probe/`](factory/geometry/preflight_svd_probe/)**: Sub-second pre-flight SVD subspace overlap verification.
* **[`factory/geometry/times_above_chance/`](factory/geometry/times_above_chance/)**: Grassmannian projection normalization proving cross-domain orthogonality.
* **[`factory/architecture_comparison/`](factory/architecture_comparison/)**: Stock LoRA vs `id_kron` controlled head-to-head evaluation.
* **[`factory/m1_vs_m2_regime/`](factory/m1_vs_m2_regime/)**: Controlled A/B audit establishing the unified M2 `bfloat16` + Liger pipeline (+3.36 pp win).

### 3. [`superseded/`](superseded/) — Historical Provenance
* Retired benchmarks and legacy prototypes preserved with full documentation of why they were superseded.
