# Fabricated Weibull/Bollinger speculative-gate "benchmarks"

Two scripts (formerly `experiments/runtime/speculative/weibull_hazard_gating/`) that were the *only* performance justification for a gate — `RangeStatisticGate(weibull_hazard_enabled=True, bollinger_bands_enabled=True)` — that ships default-on in production, live in the speculative draft loop on both the 4B and 9B serving paths. Flagged as Critical #1 in `docs/EXPERIMENT_REAUDIT_2026-09.md`.

Neither script ever loads a model. Both hand-construct "realistic synthetic token streams" from a hardcoded acceptance-decay curve (`[0.85, 0.72, 0.58, ...]`) specifically shaped to contain the signal the gate is designed to detect, and every reported "tok/s" comes from a hardcoded linear latency formula (`t_base_single * (1.0 + 0.12 * k)`), never a timed decode. The old README (kept here as `README_fabricated_original.md`) reported up to **+49.3% speedup at K=8** — an artifact of the input construction, not a finding.

See `benchmarks/superseded/README.md` for this repo's general retirement policy. Full detail is in each script's own retirement docstring.

| Script | What was fabricated |
| :--- | :--- |
| `benchmark_weibull_speculative_gating.py` | Synthetic acceptance-decay logit stream + hardcoded linear latency formula for "tok/s"; no model, no timed decode. |
| `benchmark_bollinger_weibull_speculative.py` | Same pattern, plus tensors placed on a real GPU device — running fabricated data on real hardware doesn't make the result real. |

## Replaced by

[`benchmarks/runtime/speculative/weibull_hazard_gating/`](../../runtime/speculative/weibull_hazard_gating/) — drives the real, live `runtime-ipwf` server (real model, real prompts, real CUDA-graph speculative decode), toggling arms through the actual `/api/engine/set_speculative_range_gate` endpoint. The real measurement found a **+4.7%** median speedup for the shipped default over gate-disabled, and found the Weibull/Bollinger additions contribute approximately nothing beyond the base range-statistic gate on their own (all three gated arms landed within noise of each other). See that folder's README for the full result and its own honest limitations.

Building the real benchmark also surfaced and fixed two unrelated pre-existing bugs in `apps/runtime-ipwf/server.py`: its request handlers called decoder methods that didn't exist (`generate()`/`stream_generate()` vs. the real `generate_with_graph()`/`generate_tokens_stream()`), and the range gate was constructed but never actually passed into the speculative draft call — meaning the server had never served a single real request successfully before this, gated or not.
