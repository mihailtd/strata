# DeepSeek Harness (DSH) SDK benchmarks

Real end-to-end throughput/TTFT/task-completion benchmarks that drive the live runtime server through the official DeepSeek Harness Python SDK (`from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig`) rather than raw HTTP — see [`evals/dsh_agent/README.md`](../../evals/dsh_agent/README.md) for the SDK background and why it replaced the earlier opencode-based approach. These measure performance through the harness; `evals/dsh_agent/` measures whether an agent driven through it can actually solve a task.

| File | What it measures |
| :--- | :--- |
| `run_deepseek_harness_sdk_benchmark.py` | Multi-turn software-engineering and domain-specialization tasks via the SDK against the local runtime server: task completion, tool-calling capability (bash, editor, state handoff), real end-to-end tok/s and TTFT. |
| `run_direct_harness_plugin_benchmark.py` | 3-way architecture comparison: raw Ollama vs Python HTTP proxy vs a direct native harness plugin — exact streaming speed, TTFT, latency. Manages real server processes (`subprocess`) for each arm. |
| `run_harness_sdk_head_to_head.py` | Baseline 27B vs Supercharged 27B specialist LoRAs across multi-turn SWE and domain-engineering challenges, both driven through the SDK. |

## Run

```bash
uv run python benchmarks/harness_sdk/run_deepseek_harness_sdk_benchmark.py
uv run python benchmarks/harness_sdk/run_direct_harness_plugin_benchmark.py
uv run python benchmarks/harness_sdk/run_harness_sdk_head_to_head.py
```

Requires the target runtime server (and, for the 3-way comparison, Ollama) already running.
