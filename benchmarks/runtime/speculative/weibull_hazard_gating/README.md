# Weibull Hazard + Bollinger Band speculative gate: real, live-server measurement

Stage 3: real end-to-end measurement of an already-integrated, default-on production feature — `RangeStatisticGate(weibull_hazard_enabled=True, bollinger_bands_enabled=True)` (`apps/runtime-ipwf/range_statistic_gate.py`), consumed live in the speculative draft loop (`apps/runtime-ipwf/bucketed_speculative.py` / `mtp_draft.py`) on both the 4B and 9B serving paths.

## Why this exists

This gate had **zero real measurement** behind it before this benchmark — flagged as Critical #1 in [`docs/EXPERIMENT_REAUDIT_2026-09.md`](../../../../docs/EXPERIMENT_REAUDIT_2026-09.md). Its only prior justification, `experiments/runtime/speculative/weibull_hazard_gating/`, never loaded a model: both scripts hand-built a synthetic acceptance-decay curve shaped to contain exactly the signal the gate is designed to detect, and computed "tok/s" from a hardcoded linear formula. See [`benchmarks/superseded/weibull_hazard_gating_fabricated/`](../../../superseded/weibull_hazard_gating_fabricated/) for the retired originals.

## What this measures

The real, live `apps/runtime-ipwf` FastAPI server — not an in-process reimplementation. Arms are toggled through the actual `POST /api/engine/set_speculative_range_gate` endpoint, and every tok/s figure is the server's own `time.perf_counter()`-measured wall-clock decode time for a real speculative decode through the real CUDA graph and real MTP draft head, read back from each response's `usage` block.

**Fixing two real server bugs was a prerequisite**, not part of the benchmark itself: `apps/runtime-ipwf/server.py`'s request handlers called decoder methods that didn't exist (`spec_decoder.generate(temperature=..., stop_token_ids=...)` / `graph_decoder.generate(...)` / `graph_decoder.stream_generate(...)`) instead of the real `generate_with_graph()` / `generate_tokens_stream()` (which also yields `(text, hidden_state)` tuples, not bare strings) — so the server had never successfully completed a single real request before this. The range gate was also constructed at startup but never actually passed into the speculative draft call. Both are now fixed (see the `apps/runtime-ipwf` commit history) and covered by the existing smoke tests.

## Arms

All four go through the same live speculative decoder — only the gate config differs:

| Arm | Config |
| :--- | :--- |
| A `gate_disabled` | `enabled=false` — pure speculative decode, no early-exit gating at all |
| B `range_only` | `enabled=true, weibull_hazard_enabled=false, bollinger_bands_enabled=false` — base range-statistic gate only |
| C `weibull` | `enabled=true, weibull_hazard_enabled=true, bollinger_bands_enabled=false` |
| D `weibull_bollinger_SHIP` | `enabled=true, weibull_hazard_enabled=true, bollinger_bands_enabled=true` — the production default |

## Result: initial run (2026-09-12, Qwen3.5-4B, `spec_k=2`, 6 real prompts, 48 tokens, 3 repeats)

| Arm | tok/s (median) | vs. gate disabled |
| :--- | ---: | ---: |
| A gate disabled | 25.91 | 1.000x |
| B range only | 27.12 | 1.047x |
| C + weibull | 27.05 | 1.044x |
| D + weibull + bollinger (SHIP) | 27.11 | 1.047x |

**The shipped default measured +4.7% over gate-disabled on this real workload** — a real, positive, but modest effect, an order of magnitude smaller than the fabricated +49.3% claim it replaces. **Weibull and Bollinger contribute approximately nothing beyond the base range-statistic gate**: B, C, and D are within noise of each other (27.05–27.12 tok/s).

Full per-request data: [`results/benchmarks/weibull_hazard_gating_live.json`](../../../../results/benchmarks/weibull_hazard_gating_live.json).

## Result: full K-sweep + 9B run (2026-09-13, 4 real prompts, 48 tokens, 2 repeats per arm)

The initial run above only covered the server's default `spec_k=2` on 4B — flagged as an open question. Finished it: swept `spec_k ∈ {2, 4, 8}` on 4B (each requiring a full server restart to recapture the CUDA graph buckets at the new K) plus one run on the 9B model at its default K=2.

| Config | A disabled | B range-only | C +weibull | D +weibull+bollinger (SHIP) | Ship vs. A |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 4B, K=2 | 24.94 | 24.93 | 24.89 | 26.31 | **1.055×** |
| 4B, K=4 | 25.59 | 24.92 | 24.44 | 24.76 | **0.968×** |
| 4B, K=8 | 24.48 | 25.62 | 25.75 | 25.52 | **1.043×** |
| 9B, K=2 | 29.52 | 31.59 | 31.52 | 31.77 | **1.076×** |

**The real effect is not uniform, and the honest headline is that it flips sign.** K=2 and K=8 on 4B, and the 9B run, all show a real +4–8% gain from the shipped default. K=4 on 4B is a real **regression** (-3.2%) — and notably, at K=4 even the base range-only gate (B) and weibull-only (C) already underperform gate-disabled, so this isn't specifically a Bollinger problem at K=4, something about that draft depth interacts badly with the gate mechanism generally. This directly contradicts a story like "the gate reliably helps" — it depends on K, and not monotonically (helps at 2 and 8, hurts at 4).

**Caveat that matters as much as the headline**: per-arm standard deviation at this sample size (2 repeats × 4 prompts = 8 samples/arm) is 1.5–5.2 tok/s — comparable to or larger than the differences between arms (0.1–2.3 tok/s). The K=4 regression and the K=2/K=8/9B gains are each individually plausible given the sample size, but this run cannot rule out that the K=4 result specifically is noise rather than a real interaction effect. What's robust across all four configs: the shipped default is never dramatically better or worse than gate-disabled (every ratio is within ±8%), which alone already falsifies the original fabricated +49.3% claim regardless of which individual per-K sign is trusted.

Full per-request data: [`results/benchmarks/weibull_hazard_gating_live_k2.json`](../../../../results/benchmarks/weibull_hazard_gating_live_k2.json), [`..._k4.json`](../../../../results/benchmarks/weibull_hazard_gating_live_k4.json), [`..._k8.json`](../../../../results/benchmarks/weibull_hazard_gating_live_k8.json), [`..._9b.json`](../../../../results/benchmarks/weibull_hazard_gating_live_9b.json).

## Honest limitations (what would strengthen or falsify this further)

- **Sample size.** 2 repeats × 4 prompts per config is enough to establish "this is not a 49% win" but not enough to certify the K=4 regression as a real, reproducible interaction rather than noise — a proper paired bootstrap CI at each K, with more repeats, would settle it.
- **One domain (astral) throughout.** Never run against postgresql/duckdb/financial/etc. prompts, where acceptance-rate dynamics (and therefore how often the gate actually fires) could differ.
- **9B only run once, at K=2.** No 9B K-sweep — plausible next step given 4B's K-dependence turned out to matter.
- This is a real measurement at a point in time, not a permanent verdict — if the model, adapters, or prompts change, re-run before trusting these numbers again.

## Run

Start the server first (loads real weights, real GPU):

```bash
cd apps/runtime-ipwf && AUTO_LOAD_MODEL=1 ./run_server.sh
```

Then, from repo root:

```bash
uv run python benchmarks/runtime/speculative/weibull_hazard_gating/benchmark_weibull_hazard_gating_live.py
```

Saves to `results/benchmarks/weibull_hazard_gating_live.json`.
