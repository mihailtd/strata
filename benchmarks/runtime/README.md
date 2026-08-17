# The Runtime Engine

This directory contains the production inference machinery, speculative decoding mechanisms, and hardware-level memory optimizations that power the runtime serving stack on AMD ROCm hardware.

---

## Subsystem Classifications & Innovation Legend

| Subsystem / Module | Tier | Concept Origin & Application |
| :--- | :---: | :--- |
| **[`cost_model/`](cost_model/)** | **⭐ Industry Standard** | **VRAM transition-cost calibration.** Measures the transition matrix and decides whether cost is destination-only. It is the pre-flight that retired the APSP router (0 pairs improved) and grounded the scheduler retirement. |
| **[`folding/`](folding/)** | **🔥 Applied Practice** | **In-Place Weight Absorption & Composite Stacking (`activate_many()`)**: In-place parameter fusing principles from numerical linear algebra adapted to LLM runtime weight matrices, eliminating 100% of PEFT wrapper overhead (+82.1% speedup). |
| **[`memory/factor_residency/`](memory/factor_residency/)** | **⭐ Standard** / **🔥 Applied** | **Factor-Based VRAM Residency**: Keeps low-rank factor matrices ($U, V$) resident in standby memory (42.5 MB/expert), achieving a **200.6x memory reduction** and enabling **up to 216 concurrent domain experts** on a single 24GB GPU. |
| **[`memory/pristine_state_buffer/`](memory/pristine_state_buffer/)** | **🔥 Applied Practice** | **Zero-Drift Pristine Buffer ($W_0$)**: Transactional Memory State Checkpointing (from database write-ahead logging / game state snapshots) applied to LLM expert swapping, achieving bit-exact $L_\infty = 0.00$ drift across millions of hot-swaps. |
| **[`memory/cuda_graph/`](memory/cuda_graph/)** | **⭐ Industry Standard** / **🔥 Applied Practice** | **Pointer-Stable CUDA Graph Compatibility**: Static execution graph recording combined with in-place low-rank mutation on frozen VRAM pointers (`data_ptr()`), enabling graph replay across adapter swaps. |
| **[`memory/zero_recapture_swapping/`](memory/zero_recapture_swapping/)** | **🔥 Applied Practice** | **Zero-Recapture Swapping Synergy**: Proves single-capture CUDA Graph execution (`capture_count == 1`, 0.0 ms recapture overhead) across multi-turn expert swaps with 0 bytes transient VRAM churn. |
| **[`speculative/mtp_speculative/`](speculative/mtp_speculative/)** | **🔥 Applied Practice** | **Native MTP Speculative Engine (K-Sweep Frontier)**: Transactional Memory State Checkpointing applied to the LLM runtime via a 52.5 MB fixed-size recurrent snapshot buffer, unlocking speculative decoding on hybrid GatedDeltaNet architectures (2.20x speedup at $K=6$). |
| **[`speculative/speculation_matrix/`](speculative/speculation_matrix/)** | **⭐ Industry Standard** / **🔥 Applied Practice** | **3x3 Speculation Matrix & Dynamic Router**: 2160-generation audit (40 prompts/domain x 3 repeats x 2 arms) evaluating $\tau$ and measured speedup per domain. Router `{"astral": true, "postgresql": true, "financial_planning": false}` — financial is gated off as **unresolved** (median 1.011x, repeats 0.996–1.012 straddle 1.0), not as slow. Speedups carry a correctness caveat: the decoder diverges from its own verifier on 5–22.5% of generations. |
| **[`speculative/mtp_head_folding/`](speculative/mtp_head_folding/)** | **Closed Boundary** | **Draft Head Domain Adaptation Boundary**: Proved that domain fine-tuning on next-token loss destroys draft acceptance ($\tau$), establishing that draft heads must be trained via logit distillation. |
| **[`performance/prefill_vs_decode/`](performance/prefill_vs_decode/)** | **⭐ Standard** / **🔥 Applied** | **Prefill vs. Decode Latency Share**: Proves prompt prefill is only **5.9% of wall time at 8k context** while decode constitutes **94.1%**, proving that decode acceleration (speculation, folding) controls >94% of user latency. |
| **[`performance/fla_triton_kernels/`](performance/fla_triton_kernels/)** | **⭐ Industry Standard** / **🔥 Applied Practice** | **Fused Triton Linear Attention on AMD ROCm (`gfx1100`)**: Binding `flash-linear-attention` Triton kernels on consumer RDNA3 hardware, flattening $K=4$ verification latency to 1.19x. |
| **[`performance/batch_scaling/`](performance/batch_scaling/)** | **⭐ Industry Standard** | **High-Batch Scaling Frontier ($B=1 \dots 64$)**: Batch throughput scales 17.52x to 611 tok/s — 27.4% of the perfect 64x — while per-request throughput falls 34.87 → 9.55 tok/s. Speculation's break-even $\tau$ *rises* 1.15 → 1.45, so its margin erodes with batch (measured via a K=1 vs K=4 forward-cost proxy; no speculation was run at $B>1$). |

---

## Directory Organization
- **[`cost_model/`](cost_model/)**: transition-cost calibration. There is no router directory any more — routing and scheduling both retired; see [`superseded/`](../superseded/) and `docs/DECISIONS.md`.
- **[`folding/`](folding/)**: In-place weight folding engine, `activate_many()` multi-expert additive composition, and PEFT wrapper comparisons.
- **[`memory/`](memory/)**: Factor-based VRAM residency, pointer-stable CUDA Graph replay, pristine state buffer management, and zero-recapture swapping synergy.
- **[`speculative/`](speculative/)**: MTP draft head integration, recurrent state rollback engine, 3x3 domain matrix audits, and $K$-sweep scaling benchmarks.
- **[`performance/`](performance/)**: Amdahl's Law prefill vs decode profiling, high-batch throughput scaling frontiers, and fused `fla` Triton kernel profilers on AMD ROCm.
