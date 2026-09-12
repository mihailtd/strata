# Plain end-to-end sanity benchmarks

Minimal, direct HTTP-only head-to-head benchmarks with no harness layer in between — real requests against real running servers, used as a sanity check before trusting a more elaborate benchmark's numbers.

| File | What it measures |
| :--- | :--- |
| `benchmark_engine_vs_ollama.py` | Simple head-to-head: tuned native ROCm engine vs raw Ollama, exact streaming tok/s and TTFT across 3 tasks, both over identical `/v1/chat/completions` endpoints. |
| `run_real_e2e_benchmark.py` | Real-world, unsimulated end-to-end benchmark for all models and adapters: real HTTP requests, real streamed tokens, real wall-clock tok/s. |
| `run_clean_sanity_benchmark.py` | Zero-fallback sanity check hitting both Ollama (`:11434`) and the native C++ engine (`:8001`, `apps/runtime-llama`) directly — no synthetic formulas, no multipliers, no regex heuristics. |

## Run

```bash
uv run python benchmarks/e2e_sanity/benchmark_engine_vs_ollama.py
uv run python benchmarks/e2e_sanity/run_real_e2e_benchmark.py
uv run python benchmarks/e2e_sanity/run_clean_sanity_benchmark.py
```

Requires the relevant server(s) already running on their expected ports.
