# Frontier ideas: live validation

Unlike the four "Frontier Inventions" scripts that used to live in `experiments/frontier/` — pure simulation, no model loaded, no torch import, every number a hand-authored literal; retired to [`benchmarks/superseded/frontier_fabricated/`](../../benchmarks/superseded/frontier_fabricated/) as Critical #5 in [`docs/EXPERIMENT_REAUDIT_2026-09.md`](../../docs/EXPERIMENT_REAUDIT_2026-09.md) — this script hits a **live, running** engine server over real HTTP and manages real server processes, to check whether "frontier" ideas actually hold up in a real request/response loop before anyone considers building them for real. Measures real GPU tok/s, TTFT, and Pass@1 accuracy across 3 real engineering challenge prompts, head-to-head against raw Ollama.

## Run

Requires the target runtime server already running.

```bash
uv run python experiments/frontier_e2e_validation/run_real_empirical_frontier_benchmark.py
```
