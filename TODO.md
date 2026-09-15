# Active Engineering Backlog

This document tracks active, unresolved engineering items for the Autonomous Runtime & Speculative Execution Engine.

For live verified findings, see [docs/CURRENT.md](docs/CURRENT.md).
For the full experiment-by-experiment audit (what's fabricated, what's real, what's integrated where, what's still open), see [docs/EXPERIMENT_REAUDIT_2026-09.md](docs/EXPERIMENT_REAUDIT_2026-09.md) — this file lists the actionable distillation of its still-open items, not a duplicate of it.
For the complete 2026-08-11 audit history and architectural init bug analysis, see [`benchmarks/superseded/AUDIT_HISTORY_2026-08-11.md`](benchmarks/superseded/AUDIT_HISTORY_2026-08-11.md).
For the forward-looking Rust-port backlog (`apps/runtime-next`, currently a stub), see [`apps/runtime-next/TODO.md`](apps/runtime-next/TODO.md) and [`apps/runtime-next/ECOSYSTEM_NOTES.md`](apps/runtime-next/ECOSYSTEM_NOTES.md) — a separate, actively-maintained backlog for the future native runtime, not tracked here.

---

## 🎯 Active Tasks

### 1. Verify CUDA Graph Replay Parity Across Expert Swaps
- **Goal:** Prove token-for-token exactness between graph replay and eager greedy decode across multi-turn domain swaps.
- **Context:** `FoldedCudaGraphDecoder.verify_against_eager()` is implemented in [`apps/runtime/cuda_graph.py`](apps/runtime/cuda_graph.py). Need to wire this validation directly into the continuous test suite.
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
- **Goal:** Benchmark multi-turn agentic problem solving on local monorepo dependency conflicts using [`evals/dsh_agent/run_eval.py`](evals/dsh_agent/run_eval.py) (ported from the retired opencode-based version).
- **Execution:** Compare base `Qwen3.5-4B` vs folded `m2_astral` on real PubGrub conflict resolution speed and token efficiency.

### 5. `macd_speculation_circuit_breaker.py` — zero real-model validation, ever
- **Goal:** Find out whether the circuit breaker that gates live speculative decode on both the 4B and 9B serving paths actually does anything useful against real model behavior.
- **Context:** Flagged in `docs/EXPERIMENT_REAUDIT_2026-09.md`'s Category 3 table as "the sharpest gap in the audit" — this module is unconditionally live inside `apps/runtime-ipwf/bucketed_speculative.py` on two model sizes, and its only existing validation is a synthetic acceptance-pattern simulation. Nobody has ever run it against a real model and checked whether its trip conditions correlate with real speculative-decode failure modes.
- **Execution:** Same real-treatment methodology already used for `weibull_hazard_gating` (§67) and `notears_causal_scheduler` (§71) — real Qwen3.5-4B, real `runtime-ipwf` server, arms toggled live, real prompts. `range_statistic_gate.py`'s own base (non-Weibull/Bollinger) gate shares this exact gap and could plausibly be measured in the same pass.

### 6. `fused_norm.py`'s unexplained 9B disable
- **Goal:** Find out why RMSNorm folding is explicitly skipped for 9B (`fold_norms_enabled = (...) and not is_9b` in both `apps/runtime/server.py` and `apps/runtime-ipwf/server.py`) and whether it's actually safe to enable.
- **Context:** Investigated this session — there is no comment, docstring, or recoverable git history explaining the guard (the file was reintroduced whole during the monorepo reorg, so its real prior history is gone). Qwen3.5-9B is the same architecture family as 4B (not a structural mismatch `fold_rmsnorm_into_linear` couldn't handle), so this isn't an obvious "can't work" case — it may be a defensive guard nobody ever revisited, or it may be hiding a real numerical issue nobody wrote down.
- **Execution:** Load Qwen3.5-9B, force `FLASH_NORM_FOLD=1` past the `is_9b` guard, and directly compare folded vs. unfolded logits on real prompts. If they match, the guard can be lifted (or the reason for it finally gets documented); if they diverge, that's a real, previously-undocumented bug worth its own root-cause pass.

### 7. `cut_set_router.py` — proven standalone, never wired in
- **Goal:** Wire the k-out-of-n reliability/hedging router (validated live on real 9B in `benchmark_9b_cut_set_live_pipeline.py`) into `runtime-ipwf`'s actual serving path.
- **Context:** `docs/EXPERIMENT_REAUDIT_2026-09.md` Category 3/4: this is an inverted gap — the standalone proof already exists, the integration doesn't. Wiring is the next step, not more benchmarking. Once wired, a real DSH-driven multi-tool-call session (`evals/dsh_agent/`) is the natural way to confirm the wiring actually improves real agentic throughput rather than just replaying the standalone harness's synthetic pipeline.
