# Active Engineering Backlog

This document tracks active, unresolved engineering items for the Autonomous Runtime & Speculative Execution Engine.

For live verified findings, see [CURRENT.md](file:///home/mihai/gnn-experiment/CURRENT.md).  
For the complete 2026-08-11 audit history and architectural init bug analysis, see [benchmarks/superseded/AUDIT_HISTORY_2026-08-11.md](file:///home/mihai/gnn-experiment/benchmarks/superseded/AUDIT_HISTORY_2026-08-11.md).

---

## 🎯 Active Tasks

### 1. Verify CUDA Graph Replay Parity Across Expert Swaps
- **Goal:** Prove token-for-token exactness between graph replay and eager greedy decode across multi-turn domain swaps.
- **Context:** `FoldedCudaGraphDecoder.verify_against_eager()` is implemented in [src/runtime/cuda_graph.py](file:///home/mihai/gnn-experiment/src/runtime/cuda_graph.py#L229). Need to wire this validation directly into the continuous test suite.
- **Success Criteria:** 100% token match on 256-token generations across consecutive swaps between `m2_astral_r8a128_v4`, `m2_postgresql_r8a128_v4`, `m2_duckdb_r8a128_v4`, and `m2_financial_r8a128_v4`.

### 2. Empirical Liger Kernel Speedup A/B Benchmark
- **Goal:** Quantify the exact wall-clock training speedup of Liger fused kernels (`fused_linear_cross_entropy`, `rms_norm`, `swiglu`) during M2 expert training.
- **Context:** M2 training saves ~130s per 150-step run, but lacks a controlled side-by-side `--no-liger` baseline run recorded under identical hardware load.
- **Execution:** Run `scripts/train/train_expert.py` with and without `--no-liger` on `astral` domain and log exact timings.

### 3. Native MTP Draft Head Logit Distillation
- **Goal:** Train the checkpoint MTP draft head to increase speculative acceptance ($\tau$) without domain drift.
- **Context:** Prior runs showed next-token SFT degrades draft acceptance ($\tau$ 2.456 $\to$ 1.531) because the head drifted from the backbone's representations. Distillation against backbone logits is required.
- **Execution:** Implement distillation loss $\mathcal{L}_{\text{distill}} = \text{KL}(P_{\text{backbone}} \parallel P_{\text{MTP}})$ in draft training harness.

### 4. End-to-End OpenCode Evaluation on M2 Expert Set
- **Goal:** Benchmark multi-turn agentic problem solving on local monorepo dependency conflicts using [tests/opencode_evals/run_eval.py](file:///home/mihai/gnn-experiment/tests/opencode_evals/run_eval.py).
- **Execution:** Compare base `Qwen3.5-4B` vs folded `m2_astral` on real PubGrub conflict resolution speed and token efficiency.
