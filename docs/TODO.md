# Active Engineering Backlog

This document tracks active, unresolved engineering items for the Autonomous Runtime & Speculative Execution Engine.

For live verified findings, see [CURRENT.md](CURRENT.md).
For the complete 2026-08-11 audit history and architectural init bug analysis, see [`benchmarks/superseded/AUDIT_HISTORY_2026-08-11.md`](../benchmarks/superseded/AUDIT_HISTORY_2026-08-11.md).

---

## 🎯 Active Tasks

### 1. Verify CUDA Graph Replay Parity Across Expert Swaps
- **Goal:** Prove token-for-token exactness between graph replay and eager greedy decode across multi-turn domain swaps.
- **Context:** `FoldedCudaGraphDecoder.verify_against_eager()` is implemented in [`apps/runtime/cuda_graph.py`](../apps/runtime/cuda_graph.py). Need to wire this validation directly into the continuous test suite.
- **Success Criteria:** 100% token match on 256-token generations across consecutive swaps between the canonical fleet's adapters (`m2_<domain>_r8a128_v7` — this item predates the v4→v7 canon change; update the target adapters before running it, don't test against v4).

### 2. Empirical Liger Kernel Speedup A/B Benchmark
- **Goal:** Quantify the exact wall-clock training speedup of Liger fused kernels (`fused_linear_cross_entropy`, `rms_norm`, `swiglu`) during M2 expert training.
- **Context:** M2 training saves ~130s per 150-step run, but lacks a controlled side-by-side `--no-liger` baseline run recorded under identical hardware load.
- **Execution:** Run `apps/factory/train_expert.py` with and without `--no-liger` on `astral` domain and log exact timings.

### 3. Native MTP Draft Head Logit Distillation — ⚠️ likely already answered, verify before starting
- **Goal:** Train the checkpoint MTP draft head to increase speculative acceptance ($\tau$) without domain drift.
- **Context:** Prior runs showed next-token SFT degrades draft acceptance ($\tau$ 2.456 $\to$ 1.531) because the head drifted from the backbone's representations. Distillation against backbone logits is required.
- **Execution:** Implement distillation loss $\mathcal{L}_{\text{distill}} = \text{KL}(P_{\text{backbone}} \parallel P_{\text{MTP}})$ in draft training harness.
- **Read `docs/DECISIONS.md` §63 before starting this.** A real experiment on the live EAGLE draft head (not a toy simulation) already tested domain-adapting the MTP head — with a different loss (feature-alignment, not logit-KL distillation) — and found the stock, non-domain-tuned head wins in every one of the 6 canonical domains regardless, because the 1-layer head's limited capacity means any domain fine-tuning causes representation collapse. This item may already be closed; see also `docs/EXPERIMENT_REAUDIT_2026-09.md`'s `speculative_mtp_distillation` entry for a related toy-simulation result that got the same premise wrong.

### 4. End-to-End DSH Agent Evaluation on M2 Expert Set
- **Goal:** Benchmark multi-turn agentic problem solving on local monorepo dependency conflicts using [`evals/dsh_agent/run_eval.py`](../evals/dsh_agent/run_eval.py) (ported from the retired opencode-based version).
- **Execution:** Compare base `Qwen3.5-4B` vs folded `m2_astral` on real PubGrub conflict resolution speed and token efficiency.
