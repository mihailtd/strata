# Ornith-1.5 35B MoE fleet benchmark

Master empirical benchmark suite for the Ornith-1.5 35B MoE model — a different fleet/model from the Qwen 3.x 27B suites elsewhere in `benchmarks/`, kept separate for that reason.

Three suites in one script:
1. **Suite A** — head-to-head raw streaming speed: Ollama vs a direct in-process harness plugin.
2. **Suite B** — multi-turn domain adapter stacking across all 6 domains.
3. **Suite C** — autonomous DeepSeek Harness agentic task execution.

Manages real server processes (`subprocess`) for each arm.

## Run

```bash
uv run python benchmarks/ornith_35b_fleet/benchmark_ornith_35b_fleet.py
```
