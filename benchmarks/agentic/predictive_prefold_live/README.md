# NOTEARS predictive pre-folding: real DSH-agent-driven measurement

Stage 3: real end-to-end measurement of an already-integrated feature (`NotearsCausalScheduler`, wired live into `apps/runtime-ipwf/server.py`'s request path as part of building this benchmark — it was previously instantiated at load time but never actually called). See [`docs/EXPERIMENT_REAUDIT_2026-09.md`](../../../docs/EXPERIMENT_REAUDIT_2026-09.md) Category 4 and [`docs/DECISIONS.md`](../../../docs/DECISIONS.md) §64/§71.

## Why this exists

`experiments/agentic/benchmark_predictive_prefold.py` validated the scheduler's prediction logic against a hand-scripted 15-turn conversation with simulated tool-execution time (`time.sleep()`) and a mocked VRAM fold (`MockVRAMFoldingEngine`, hardcoded 1.9ms). It was honestly disclosed as a mock in its own code, but `docs/DECISIONS.md` §64 cited its output as unqualified "Empirical Benchmark Results." This benchmark replaces both mocked halves with real ones.

## What's real, what isn't

**Real:**
- A live [DeepSeek Harness (DSH)](../../../apps/harness/) agent session per step, driving the actual `apps/runtime-ipwf` server (real Qwen3.5-4B, real generation, ~9s/turn).
- The production `NotearsCausalScheduler.predict_next_expert` / `async_prefold` / `record_turn_outcome` code, now actually wired into the request-completion path (previously dead code — instantiated, never called).
- `WeightFoldingEngine.activate()` — a real GPU weight fold (`torch.addmm`), server-measured with `time.perf_counter()`, not a `sleep()`.

**Two real bugs found and fixed while wiring this up** (not benchmark artifacts — actual server bugs):
1. `WeightFoldingEngine.activate()` used to unconditionally re-fold even when the requested expert was already active — this would have silently defeated the entire point of pre-folding (a "hit" would still pay the full fold cost). Fixed in both `apps/runtime/novel_peft.py` and `apps/runtime-ipwf/novel_peft.py` to skip when already active (deterministic operation, so this is a pure optimization, not a behavior change).
2. `compute_surgical_notch_masks`'s companion MTP draft-head loader picks a candidate matched adapter with no hint of which base model size is being served, and was loading the **9B**-shaped `mtp_{domain}_r64_a64_v7_9b` adapter for the **4B** model — an actual shape-mismatch crash (`RuntimeError` in `torch.addmm`) on the very first named-expert activation. Fixed defensively: a shape check before the fold, falling back to the pristine (unmatched) draft head with a loud warning instead of crashing. The root cause (ambiguous candidate resolution) is still open — see the warning text in `apps/runtime-ipwf/novel_peft.py`.

**Not real** — confirmed live: `apps/runtime-ipwf/server.py` has no OpenAI-style function/tool-calling support at all (`ChatCompletionRequest` uses `extra="ignore"`, silently dropping any `tools=` DSH sends). DSH's bash tool never actually fires; the model only describes commands in prose/code blocks, it doesn't execute them. Implementing real function-calling is a separate, substantial feature, not attempted here. `detect_tools()` (real, unchanged production code) pattern-matches the agent's real generated *text* for tool markers — this works identically whether a tool was actually executed or just mentioned, since the scheduler was always driven by text detection, not structured tool-call events. **Do not cite this as "the agent completed a real coding task."**

## Method

5 steps mirroring the canonical pipeline the scheduler was seeded with (astral → postgresql → duckdb → python_web → python_modern). DSH does not support resuming one `session_id` across separate harness instances (confirmed live: `session "X" already exists`), so each step is a fresh harness + session, `model=` set to that step's domain, against the same live server and the same real workspace directory. Run twice: prefolding disabled (Arm A) and enabled (Arm B), toggled live via `POST /api/engine/set_predictive_prefold`. Real per-step swap cost is a diffed cumulative counter (`cumulative_swap_ms`/`cumulative_swap_count` in `/api/engine/status`) rather than a "last value" — a single DSH `.run()` can trigger more than one real completion per step, and "last" would silently drop everything but the final one.

## Result (2026-09-13, Qwen3.5-4B, 5 real DSH-driven steps)

| Arm | Total real swap ms | Real folds | Hits | Misses |
| :--- | ---: | ---: | ---: | ---: |
| A — reactive only (no prediction) | 5.04 | 5 | — | — |
| B — NOTEARS prefold enabled | 4.02 | 4 | 2 | 2 |

**Arm B's real swap overhead is 0.80× of Arm A** (saved 1.02ms across 5 real steps) — a real, positive effect: step 2 (duckdb) landed as an actual hit, with `real_swap_ms=0.00` because the background pre-fold had already completed by the time the request arrived, so `WeightFoldingEngine.activate()` found the correct expert already resident and skipped the real fold entirely (this is the bug-fix above paying off — before it, this would have re-folded anyway).

**The honest caveat: the absolute numbers are tiny.** A real fold on this model/hardware costs ~1ms — an order of magnitude below the ~1.9ms assumed by the old mock, and completely dwarfed by real per-turn generation latency (~8.9 **seconds** per turn in this run). Predictive pre-folding saves roughly 1ms out of every ~9000ms turn here — a real effect, but not one that matters for end-to-end latency at this scale. It would matter more in a regime with much higher turn-switching frequency and much shorter per-turn generation (e.g. very short completions), which this benchmark does not test.

Full data: [`results/benchmarks/predictive_prefold_live.json`](../../../results/benchmarks/predictive_prefold_live.json).

## Honest limitations (what would strengthen or falsify this further)

- Only 5 steps, one pass per arm — no repeats, no variance estimate. The 2/4 hit rate observed here is a single sample.
- No real tool execution (see above) — the domain-relevant text the scheduler detects tools from is agent prose, not verified file changes.
- Real fold cost (~1ms) is so small relative to generation time that this feature's practical value is better judged by whether it ever introduces *harm* (race conditions, wrong folds) than by latency saved — the fold-lock and shape-guard fixes above were more valuable outcomes of this work than the headline percentage.
- Never run on 9B, and never run with a longer, higher-switching-frequency session.

## Run

Start the server first (loads real weights, real GPU):

```bash
cd apps/runtime-ipwf && AUTO_LOAD_MODEL=1 ./run_server.sh
```

Then, from repo root:

```bash
uv run python benchmarks/agentic/predictive_prefold_live/benchmark_predictive_prefold_live.py
```

Saves to `results/benchmarks/predictive_prefold_live.json`.
