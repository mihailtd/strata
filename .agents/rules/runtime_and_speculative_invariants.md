# Rule: Runtime Engine, Speculative Decoding & Benchmark Invariants

## Core Principles

This rule defines non-negotiable invariants established through rigorous empirical testing on AMD Radeon RX 7900 XTX hardware (ROCm / gfx1100). All runtime enhancements, benchmark harnesses, and multi-agent systems must strictly adhere to these invariants.

---

## 1. Zero-Simulation Invariant for Autonomous Benchmarks

* **Forbidden Practice**: Never commit or rely on simulation stubs, mock sleep calls (`time.sleep`), or canned string responses in evaluation benchmarks. Mocking hides real failure modes, latency bottlenecks, and memory churn.
* **Mandatory Standard**:
  1. Every benchmark harness must execute genuine inference passes through the live engine (`Native27BEngine` or server endpoints).
  2. Generated patches and code solutions must be written to temporary isolated environments and tested against live operating system utilities (`uv run pytest`, `uv run ruff check`).
  3. Every scorecard must report live wall-clock execution time, real token throughput, and genuine subprocess exit codes.

---

## 2. Speculative Decoding Standard: Linear Speculation Dominates Tree Speculation

* **Empirical Verification (AMD RX 7900 XTX, 27B Model)**:
  - **Cascaded N-Gram + Linear Neural MTP ($K=2..3$)**: Achieves **20.0 tokens/sec (1.67x speedup vs greedy 12.0 tok/s)** with a **74.6% cycle acceptance rate**.
  - **Tree Speculation ($2 \times 2$ Static, Asymmetric, and Dynamic Adaptive)**: Achieves only **16.9–17.0 tokens/sec (1.41–1.42x speedup)** with an acceptance rate of **59.5%**.
* **Root Cause & Mechanism**:
  - When the draft proposal mechanism (Cascaded N-gram + Neural MTP) already achieves ~75% top-1 accuracy on structured code, the primary candidate is valid 3 out of 4 times.
  - The potential upside of salvaging the remaining 25% of rejections via a secondary branch is completely negated by:
    1. Multi-branch tensor indexing and GPU synchronization overhead on RDNA3 architecture.
    2. The state-rollback and KV cache management penalty of branch switching across multiple paths.
* **Production Invariant**:
  - The production decode engine MUST retain **Cascaded N-Gram + Linear Neural MTP ($K=2..3$)**.
  - Do NOT deploy multi-branch tree speculation ($2 \times 2$) to the production decoding pipeline.

---

## 3. Static LoRA Buffers Memory Invariant (Zero-Recapture Hot-Swapping)

* **Forbidden Practice**: Never re-assign PyTorch tensor attributes (e.g. `mod.lora_a = new_tensor` or `mod.weight = ...`) on HIP Graph captured models. Re-assignment changes virtual memory addresses, invalidating the ROCm HIP Graph and incurring a **760 ms graph re-capture penalty** per adapter swap (stalling the pipeline for >810 ms).
* **Mandatory Standard**:
  1. All 304 adapter-eligible linear layers must pre-allocate static VRAM buffers (`static_lora_a` with shape `[R, in_features]`, `static_lora_b` with shape `[out_features, R]`, $R=16$) during server initialization.
  2. Dynamic adapter hot-swapping must write in-place via PyTorch `.copy_()`:
     ```python
     mod.static_lora_a.copy_(adapter_weights["lora_a"])
     mod.static_lora_b.copy_(adapter_weights["lora_b"] * alpha_scale)
     ```
  3. Pre-fold the scaling factor $\alpha$ directly into the $B$ buffer to eliminate runtime scalar multiplications.
  4. With static buffers, adapter hot-swap latency is strictly bounded between **37 ms and 60 ms** with zero graph re-captures and zero memory allocation churn.

---

## 4. Recurrent State Handoff ($S_t$) & Immutability Guarantee

* **State Geometry & Footprint**:
  - For Qwen 3.8-27B hybrid recurrence, the complete recurrent state bundle consists of **74.81 MB across 112 GPU tensors**:
    - 48 SSM states: `(48, 16, 128, 128)` float32 = 48.00 MB.
    - 48 Conv states: `(48, 1, 5120, 4)` bfloat16 = 1.97 MB.
    - 16 Attention KV caches: `(16, 2, 1, 8, max_seq, 128)` bfloat16 = 24.84 MB.
* **Safety Invariant**:
  - Always enforce `clone_on_handoff = True` when handing off state between multi-agent turns.
  - Passing raw tensor references allows subsequent autoregressive decoding steps to mutate the parent turn's state in-place, corrupting backtrack, retry, or multi-branch pipelines.
  - Internal GPU VRAM tensor cloning takes only **1.34–1.67 ms** for the entire 74.81 MB bundle, providing 100% mathematical immutability at negligible computational cost.
* **Performance Benefit**:
  - Eliminates prompt re-prefill across multi-turn sessions, delivering up to **18.88x prefill speedup on Turn 6** (768 ms vs 14,516 ms) and an **82.3% reduction in processed context tokens**.

---

## 5. Elimination of Zero-Return Complexity: Prohibition of Static AST Macros in Speculative Decode

* **Empirical Verification (AMD RX 7900 XTX, 1,024-Token Best-Case Benchmark)**:
  - **Linear MTP + N-Gram (Arm 2)**: **21.50 tok/s (47.64s, 501 steps)** with **89.6% Neural MTP acceptance** and **57.6% N-Gram acceptance**.
  - **Syntax Fast-Forwarding + Trie (Arm 3)**: **21.21 tok/s (48.27s, 507 steps)** with only **16.0% Syntax Trie acceptance** (4 accepted out of 25 proposed).
* **The "Best-Case Negative = Dead on Arrival" Invariant**:
  - If a speculative drafter fails to outperform baseline in its hand-picked best-case scenario (e.g. repetitive microservice boilerplate), it is dead on arrival for general production.
  - Static Trie macros achieve only 16% acceptance on unconstrained code and cause verification chunk rollbacks (507 steps vs 501 steps), creating a net negative drag (0.99x).
  - The block-64 Neural MTP head is vastly superior (89.6% acceptance) because it conditions on the full 27B hidden state $h_{\text{curr}}$ and understands semantic context (variable naming, parameter order, status codes).
* **Production Invariant**:
  - Never place static Trie or AST macro drafters in the speculative drafting pipeline for general code decoding.
  - The production speculative decode pipeline is strictly: **In-Context N-Gram + Neural MTP Head (blk.64)**.
  - Do not initialize or compile static syntax tries during server boot.
  - Do not add architectural layers or maintenance complexity that yield 0 returns in benchmark testing.

