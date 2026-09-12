# `agent_sandboxes/` — harness/benchmark target codebases

Every directory here is a generated coding-task scenario written to disk by an agent under test (`apps/harness/coordinator/*.py`), then graded by running `ruff`/`pytest` against what it produced. None of these are meant to be hand-edited as application code — they're targets, not source.

They split into two groups with different lifecycles:

## Ephemeral (gitignored, not tracked)

`bench3_arm_a_handoff/`, `bench3_arm_b_reprefill/`, `bench3_arm_c_llamacpp/` — the three arms of `benchmarks/state_handoff_e2e/benchmark_3way_harness.py`'s comparison. This script `shutil.rmtree()`s and regenerates all three from scratch on every run; whatever's on disk right now is just leftover output from the last run, not something anyone audits. Gitignored in root `.gitignore`.

## Audited fixtures (tracked)

`state_handoff_e2e_service/`, `text_prefill_e2e_service/` — fixed task scenarios repeatedly targeted by `benchmarks/state_handoff_e2e/benchmark_state_handoff_e2e_harness.py`. Nothing `rmtree`s these before writing — they're meant to persist and be reviewed across runs, which is why they stay tracked even though an LLM wrote every line of them. Treat a change here the way you'd treat any other reviewed diff: read it before committing.

## Removed: `real_coordinated_service/`, `real_cross_domain_service/`, `real_portfolio_service/`

These used to live here, produced by `apps/harness/run_real_agent_task.py`, `run_stacked_multi_expert_task.py`, and the pre-fix `coordinator/subagent.py`. All three were removed because their content was **fabricated, not agent-generated**: the generating scripts either ignored the model's real output entirely and wrote hardcoded template files instead, or (in `subagent.py`'s case) silently substituted a "curated" reference implementation whenever the model's real output didn't match a hardcoded set of expected substrings — while still reporting "100% REAL" / "100% GENUINE" success unconditionally, including in at least one case printing "3/3 tests passed" without ever checking the actual test exit code. See `apps/harness/coordinator/subagent.py`'s docstring and `evals/execution_gate/eval_harness_coordinator_live.py` for the fix (a real DSH-driven agent, no template fallback, honest success reporting).

`run_real_agent_task.py` and `run_stacked_multi_expert_task.py` were deleted outright (not worth salvaging line-by-line). `real_coordinated_service/` will regenerate honestly the next time `eval_harness_coordinator_live.py` is run against a live model server — it isn't tracked here yet because that hasn't happened in this environment (no server was running when the fix was written; see that eval's "NOT YET LIVE-VERIFIED" note).

`.adapters/` subdirectories (LoRA checkpoints these scenarios load) stay gitignored regardless — that's model weight data, not source.
