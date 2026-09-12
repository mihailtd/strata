# 🔬 Benchmarks: Real End-to-End Performance of Already-Integrated Features

> **System**: **Autonomous Runtime, Multi-Expert Factory & RDNA3 Kernel Engine**  
> **Target Hardware**: AMD ROCm (`gfx1100` / RX 7900 XTX 24GB VRAM, 960 GB/s GDDR6)  
> **Target Model**: Qwen 3.8-27B Hybrid Recurrence (48 DeltaNet SSM + 48 Conv1D + 16 Attention Layers)

**Scope, since the [`experiments/`](../experiments/)/`benchmarks/`/[`evals/`](../evals/) split**: everything here measures a feature that's already live in `apps/runtime*` or `apps/factory` — throughput, latency, memory, hardware utilization. Pre-integration exploratory work (synthetic microbenchmarks, "does this idea work at all" probes) lives in [`experiments/`](../experiments/) instead; task/rubric-scored quality evaluation lives in [`evals/`](../evals/). See the repo's [`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md) for the full lifecycle these three trees implement. A number of entries in the matrix and deep-dive list below now point into those two sibling trees — the work they document didn't change, only where its writeup landed.

---

## 🧭 Master Matrix: What We Tried, What Failed & Production Verdict

This matrix summarizes the empirical findings, hardware trade-offs, and final production decisions across all evaluated techniques on AMD RDNA3 hardware:

| Technique & Benchmark | What We Tried | What Failed / Bottlenecks Encountered | Live Hardware Metric | Production Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Multi-Expert LoRA Stacking**<br>([`README_MULTI_EXPERT_STACKING.md`](../evals/execution_gate/README_MULTI_EXPERT_STACKING.md)) | Dynamic weight-space block concatenation ($B_{\text{stacked}} = [B_1, B_2]$, $A_{\text{stacked}} = [A_1; A_2]$) with exact zero cross-talk identity. | Inverted PEFT shape conventions initially caused feature concatenation instead of rank expansion; resolved by dynamic rank dimension orientation detection. | **86–124 ms swap latency**, 19.4–20.1 tok/s, 100% rubric score on dual/triple SWE tasks. 0 MB VRAM churn. | 🚀 **100% WORTH IT**<br>Simultaneous Multi-Domain Specialist Fusion with zero quality loss. |
| **Deterministic AST & Syntax Fast-Forwarding**<br>([`syntax_fast_forward/`](syntax_fast_forward/README.md)) | Zero-VRAM prefix suffix Token Trie (<0.4 µs lookup) proposing 3-token linear syntax chains into `forward_verify`. | In unconstrained code, static macros have only **16.0% acceptance** vs **89.6% for Neural MTP**. Rejected drafts cause chunk rollbacks (507 steps vs 501 steps, 21.2 tok/s vs 21.5 tok/s). | **21.2 tok/s (0.99x vs Linear MTP 21.5 tok/s, 1.84x vs Greedy 11.5 tok/s)**. 100% bit-exact parity. | ⚠️ **NOT RECOMMENDED FOR UNCONSTRAINED DECODE**<br>Keep Linear MTP + N-Gram as default (21.5 tok/s). Reserve Syntax Drafter for constrained GBNF/JSON only. |
| **Speculative Tree Decoding**<br>([`tree_speculation_e2e/`](tree_speculation_e2e/README.md)) | $2 \times 2$ static balanced tree, asymmetric tree, and dynamic entropy-adaptive candidate trees. | Hybrid N-Gram + Neural drafter already achieves ~75% top-1 accuracy. Salvage opportunity is only ~25%. GPU synchronization and multi-path rollback tax ate all gains. | **16.9–17.0 tok/s** (Tree) vs **20.0 tok/s** (Linear). Linear is +18% faster! | 🛑 **NOT WORTH IT**<br>Keep Linear Speculation ($K=2..3$). Do not deploy Tree. |
| **Recurrent State Handoff ($S_t$)**<br>([`state_handoff_e2e/`](state_handoff_e2e/README.md)) | True $O(1)$ state transfer across multi-agent turns without history re-prefill. | Early estimates assumed 3.1 MB; reality is **74.81 MB across 112 GPU tensors**. Raw references caused in-place decode mutation to corrupt retries. Solved by `clone_on_handoff = True` (1.34 ms). | **18.88x prefill speedup on Turn 6** (768 ms vs 14,516 ms). Total pipeline 2.10x faster. | 🚀 **100% WORTH IT**<br>Core Production Pillar for Multi-Agent Systems. |
| **Static Pre-Allocated LoRA Buffers**<br>([`README_STATIC_LORA_BUFFERS.md`](../evals/domain_rubric/README_STATIC_LORA_BUFFERS.md)) | Pre-allocated working buffers ($R=16$) mutated in-place via `.copy_()` with scalar folding. | Dynamic pointer re-assignment (`mod.lora_a = new`) invalidated ROCm HIP Graphs, triggering a **760 ms re-capture penalty** per adapter swap (810 ms total swap time). | Swap latency dropped from **810 ms $\to$ 37–60 ms** (up to 21.8x faster). 0 ms graph invalidation. | 🚀 **100% WORTH IT**<br>Essential Low-Level Infrastructure. |
| **Live Autonomous SWE-Bench**<br>([`swe_bench/README.md`](../evals/swe_bench/README.md)) | Live 27B model inference with real `pytest` execution sandboxes across 6 real-world domains. | Historical mock delay (`time.sleep`) faked 100% pass rate in 14.8s. Real execution exposed 0% Pass@1 due to conversational text wrapping and mock fixture mismatches. | Identified true AST patch application requirements on AMD hardware. | 🚀 **100% WORTH IT**<br>Zero-Simulation Mandate. Move to tool-calling diff format. |
| **ROCm HIP Graph Capture**<br>([`experiments/hip_graph_capture/`](../experiments/hip_graph_capture/README.md)) | Single-command GPU replay of 64 chained Triton layers to bypass CPU launch overhead. | Dynamic sequence lengths require static maximum arenas. Dynamic memory pointers crash graphs (resolved via static LoRA buffers). | Eliminates ~100 ms host dispatch latency per token. 1.000000 bit-exact identity. | 🚀 **100% WORTH IT**<br>Standard Decode Pipeline. |
| **Batched Chunked Prefill**<br>([`experiments/batched_prefill/`](../experiments/batched_prefill/README.md)) | Chunked sequence processing for 2K–32K long context prefill. | Monolithic prefill caused VRAM peak spikes and GPU execution timeouts on large contexts. | Smooth VRAM allocation curve, stable TTFT scaling up to 32K context. | 🚀 **100% WORTH IT**<br>Long-Context Stability. |
| **128-Bit Coalesced GEMV**<br>([`README_RDNA3_GEMV.md`](kernel/w4a16_gemv_m1/README_RDNA3_GEMV.md)) | Scribing global memory loads into `int32x4` vector bundles in Wave32 SIMD. | Uncoalesced scalar memory reads achieved $<30\%$ bus utilization on RDNA3 hardware. | **620.4 GB/s throughput** (64.6% physical GDDR6 bus saturation on 7900 XTX). | 🚀 **100% WORTH IT**<br>Foundational Kernel. |

---

## 🏛️ The Three Pillars of Engineering

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       AUTONOMOUS AI SYSTEM PILLARS                                              │
├────────────────────────────────┬────────────────────────────────┬──────────────────────────────────────────────┤
│ 🏭 Pillar 1: The Factory       │ ⚡ Pillar 2: The Runtime       │ 🚀 Pillar 3: The Kernel                      │
├────────────────────────────────┼────────────────────────────────┼──────────────────────────────────────────────┤
│ • Geometric Stopping           │ • Recurrent State Handoff (St) │ • 128-Bit Coalesced GEMV (620 GB/s)          │
│ • Ledoit-Wolf Shrinkage        │ • Static LoRA Buffers (37ms)   │ • Fused SwiGLU In-Register SiLU (733 GB/s)   │
│ • Subspace Orthogonality       │ • Linear Speculation (20 tok/s)│ • Linear MTP Verification Kernel             │
│ • Outlier-Protected SSI Quant  │ • Zero-Simulation Verification │ • Outlier-Protected W4A16 (Zero-Loss)        │
├────────────────────────────────┼────────────────────────────────┼──────────────────────────────────────────────┤
│ 📁 benchmarks/factory/         │ 📁 benchmarks/runtime/         │ 📁 benchmarks/kernel/                        │
└────────────────────────────────┴────────────────────────────────┴──────────────────────────────────────────────┘
```

Each pillar's still-exploratory work (not yet integrated into `apps/factory`/`apps/runtime*`) lives in the matching `experiments/factory/`, `experiments/runtime/`, `experiments/kernel/` — these three `benchmarks/` subdirectories hold only the subset that graduated.

---

## 🏆 Deep-Dive Technique Reports

For complete mathematical formulations, benchmark scorecards, failure analysis, and reproduction instructions, consult the dedicated reports:

1. [Multi-Expert LoRA Stacking & Dynamic Adapter Fusion (`README_MULTI_EXPERT_STACKING.md`)](../evals/execution_gate/README_MULTI_EXPERT_STACKING.md)
   - Zero cross-talk block concatenation ($B_{\text{stacked}} A_{\text{stacked}} \equiv \sum \gamma_k B_k A_k$) verified to $10^{-13}$ precision.
   - Dual-Expert and Triple-Expert SWE benchmark on live 27B model on AMD RX 7900 XTX (86–124 ms hot-swap, 0 MB memory churn).
2. [Deterministic AST & Syntax Fast-Forwarding](syntax_fast_forward/README.md)
   - Zero-VRAM prefix suffix Token Trie (<0.4 µs lookup) proposing 3–6 token linear syntax chains into `forward_verify`.
   - Eliminates N-gram cold-start slump on Python, FastAPI, PostgreSQL, DuckDB syntax.
3. [Speculative Tree Decoding vs. Linear Speculation](tree_speculation_e2e/README.md)
   - Detailed empirical comparison of 5 decoding architectures across 512 and 1,024 tokens.
   - Comprehensive analysis of why Linear Speculation beats Tree Speculation by +18% throughput.
3. [Recurrent State Handoff ($S_t$) Architectural Report](state_handoff_e2e/README.md)
   - 74.81 MB state bundle geometry (48 SSM + 48 Conv + 16 KV tensors).
   - In-place mutation pitfall and the `clone_on_handoff = True` immutability guarantee (1.34 ms).
   - 18.88x Turn 6 prefill acceleration and 82.3% context reduction.
3. [Static LoRA Buffers & Zero-Recapture Hot-Swapping (`README_STATIC_LORA_BUFFERS.md`)](../evals/domain_rubric/README_STATIC_LORA_BUFFERS.md)
   - Elimination of the 760 ms ROCm HIP Graph re-capture penalty.
   - 304 pre-allocated static working arenas with in-place scalar folding ($\alpha \times B$).
4. [Autonomous SWE-Bench Live Evaluation (`swe_bench/README.md`)](../evals/swe_bench/README.md)
   - Live execution against 6 authentic problem domains in real `pytest` sandboxes.
   - Breakdown of raw prompt patching bottlenecks and transition to structured tool calling.
5. [ROCm HIP Graph Capture for Custom Triton Layers](../experiments/hip_graph_capture/README.md)
   - Elimination of CPU host launch overhead for 64-layer inference.
6. [Batched Prefill for Long Context](../experiments/batched_prefill/README.md)
   - Chunked prefill scaling across 2K to 32K context windows.
7. [RDNA3 128-Bit Memory Coalescing (`README_RDNA3_GEMV.md`)](kernel/w4a16_gemv_m1/README_RDNA3_GEMV.md)
   - Achieving 620 GB/s memory bandwidth on AMD Radeon RX 7900 XTX.
