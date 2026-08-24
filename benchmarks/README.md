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

### 3. ⚠️ The Scale-Dependent Quantization Boundary (Why QLoRA Chokes on 4B but Excels on 9B+)
* **The Pitfall**: Assuming 4-bit QLoRA behaves identically across all model parameter scales.
* **The Information-Theoretic Law**:
  * **On Small Models ($D=2,560$, 4B and below)**: Latent space dimensionality is narrow with low parameter redundancy. When base weights are quantized to 4-bit NF4, quantization noise perturbations $\epsilon = \text{dequant}(W_{\text{NF4}}) - W_{\text{exact}}$ propagate through activations $\widetilde{X}_\ell$, corrupting the LoRA adapter gradient signal ($\nabla_{A, B} \mathcal{L}$) and degrading downstream domain accuracy by **3.36 percentage points**. Full `bfloat16` base training is mandatory for sub-4B models.
  * **On Large Models ($D \ge 4,096$, 9B / 27B / 70B)**: High-dimensional manifolds possess massive architectural redundancy. 4-bit NF4 captures $>99\%$ of representational capacity; activation noise is absorbed by the wider latent space, allowing adapter gradients to converge cleanly to **94.0%–97.3% eval accuracy**.
* **The Physical VRAM Reality on 24GB GPUs**:
  * **4B + Standard LoRA (BF16)**: Base 8.0 GB $\to$ Peak **14.5 GB** (Fits with 9.5 GB free headroom; no need for 4-bit compromise).
  * **9B + Standard LoRA (BF16)**: Base 17.91 GB $\to$ Peak **23.76 GB (99.1%)** (Triggers OOM, compositor freezing, and VRAM paging).
  * **9B + QLoRA (NF4 Base + BF16 LoRA)**: Base 7.65 GB $\to$ Peak **17.35 GB** (Optimal: 6.65 GB free headroom, 2.9s/step throughput).
* **The Scaling Rule**:
  * **Sub-4B Models**: Always train with full **`bfloat16`** base weights whenever VRAM permits.
  * **$\ge$ 9B Models**: Train with **QLoRA (4-bit NF4 base + `bfloat16` LoRA gradients)**. Export adapters in full `bfloat16` for lossless in-place folding.

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
* **[`runtime/cost_model/`](runtime/cost_model/)**: Measured VRAM transition-cost calibration — the destination-only model `C(u,v) = f(v)` that retired both the APSP router and (with the single-tenant finding) the SLA scheduler. Run it before proposing any routing idea.
* **[`runtime/folding/`](runtime/folding/)**: In-place weight folding engine eliminating 100% of PEFT wrapper overhead (+82.1% speedup over wrapped PEFT).
* **[`runtime/folding/goldilocks_in_place_addmm/`](runtime/folding/goldilocks_in_place_addmm/)**: Guaranteed lossless in-place `addmm()` execution within the calibrated Goldilocks operating window ($\alpha_{\min} \le \alpha \le \alpha_{\max}$), proving pointer stability and 0.00e+00 drift across 100+ swaps.
* **[`runtime/memory/factor_residency/`](runtime/memory/factor_residency/)**: 42.5 MB low-rank factor standby representation (200.6x compression), enabling up to 216 concurrent domain experts on 24GB VRAM.
* **[`runtime/memory/pristine_state_buffer/`](runtime/memory/pristine_state_buffer/)**: Transactional Memory State Checkpointing preserving $L_\infty = 0.00$ drift across infinite expert hot-swaps.
* **[`runtime/memory/cuda_graph/`](runtime/memory/cuda_graph/)**: Pointer-stable weight slots enabling zero-recapture CUDA Graph replay.
* **[`runtime/memory/zero_recapture_swapping/`](runtime/memory/zero_recapture_swapping/)**: Multi-turn expert swapping under single-capture CUDA Graphs with 0 bytes transient churn.
* **[`runtime/speculative/mtp_speculative/`](runtime/speculative/mtp_speculative/)**: Native MTP speculative engine with 52.5 MB recurrent rollback, delivering **2.20x net speedup at $K=6$**.
* **[`runtime/speculative/range_statistic_gating/`](runtime/speculative/range_statistic_gating/)**: Single-Pass Range Statistic Early-Exit Gating (Chapter 8) pruning **36.4% of wasted drafts** for **+18.8% decode speedup (68.49 tok/s)**.
* **[`runtime/speculative/speculation_matrix/`](runtime/speculative/speculation_matrix/)**: 180-run 3x3 domain matrix audit routing speculative execution.
* **[`runtime/performance/`](runtime/performance/)**: Native RDNA3 Triton WMMA acceleration and Fused W4A16 + Dynamic LoRA branch kernel (3.88x VRAM compression, 1.17x–1.49x memory-bound decode speedup).
* **[`runtime/performance/prefill_vs_decode/`](runtime/performance/prefill_vs_decode/)**: Amdahl's Law audit proving decode governs 94.1% of user latency.
* **[`runtime/performance/fla_triton_kernels/`](runtime/performance/fla_triton_kernels/)**: ROCm Triton kernel acceleration flattening verification latency to 1.19x.
* **[`runtime/performance/batch_scaling/`](runtime/performance/batch_scaling/)**: High-batch throughput scaling up to $B=64$ (611 tok/s).
* **[`runtime/multi_agent/`](runtime/multi_agent/)**: Tensor-Level Recurrent State Handoff ($S_t$), Hybrid Dual Protocol, and 27B A/B benchmark vs. Ollama text re-prefill (**10.48x prefill acceleration**, **73.0%–98.8% context window preservation**).

### 2. [`factory/`](factory/) — The Factory & Geometric Probes
* **[`factory/EVALUATION_LADDER.md`](factory/EVALUATION_LADDER.md)**: The 4-stage fast-to-slow evaluation ladder (SVD checks → micro-loss probes → targeted logprob shift → downstream benchmarks).
* **[`factory/geometry/alpha_sweep/`](factory/geometry/alpha_sweep/)**: `bfloat16` Mantissa ULP Inverse Scaling Law ($merge\_rel\_err \approx \frac{0.167}{\|dW\|/\|W\|}$) and physical perturbation characterization.
* **[`factory/geometry/dynamic_alpha_calibration/`](factory/geometry/dynamic_alpha_calibration/)**: Per-Adapter Dynamic $\alpha$-Calibration & ⭐ Stock LoRA ($r=8$, Dynamic $\alpha$) decoupling rank from runtime scaling without retraining.
* **[`factory/geometry/preflight_svd_probe/`](factory/geometry/preflight_svd_probe/)**: Sub-second pre-flight SVD subspace overlap verification.
* **[`factory/geometry/times_above_chance/`](factory/geometry/times_above_chance/)**: Grassmannian projection normalization proving cross-domain orthogonality.
* **[`factory/m1_vs_m2_regime/`](factory/m1_vs_m2_regime/)**: Controlled A/B audit establishing the unified M2 `bfloat16` + Liger pipeline (+3.36 pp win).
* **[`factory/robust_distillation/`](factory/robust_distillation/)**: Breakdown-Bounded Loss (LAD-LASSO & Huber) insulation under $\le 50\%$ noisy synthetic tool trace contamination ($101.60\% \to 2.92\%$ error reduction over $L_2$).
* **[`factory/speculative_mtp_distillation/`](factory/speculative_mtp_distillation/)**: Domain-Specialized MTP Draft Head continuous distillation preventing representation collapse ($0.003 \to 0.731$ cosine alignment).

### 3. [`multi_turn_execution_benchmark.py`](multi_turn_execution_benchmark.py) — Chained Multi-Turn Execution Gate
* **15-Step Multi-Pipeline Evaluation**: Tests sequential task handoff across PostgreSQL schemas, FastMCP servers, and test suites with real Ruff linter and DuckDB sandbox execution.
* **Autonomous Intent Routing (Arm C = Arm B)**: Autonomous Engine matches Oracle Swarm accuracy with statistically significant positive edge over Base 4B (+0.045, 95% CI [+0.004, +0.084]).
* **Massive Token & Latency Efficiency**: Consumes **5,474 tokens vs 11,969 tokens** on Base 4B (**+54.3% token savings**), with average adapter swap time of **0.94 ms**.

### 4. [`superseded/`](superseded/) — Historical Provenance
* Retired benchmarks and legacy prototypes preserved with full documentation of why they were superseded (including [`ftq_0.8b_experiments.md`](superseded/ftq_0.8b_experiments.md) and [`AUDIT_HISTORY_2026-08-11.md`](superseded/AUDIT_HISTORY_2026-08-11.md)).
