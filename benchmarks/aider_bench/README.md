# 🚀 Aider Python "Hello World" Benchmark Suite

A lightweight, ultra-fast automated benchmark runner based on a tailored subset of **Aider's Polyglot Benchmark** (from Exercism Python practice exercises). Built to rapidly test, benchmark, and iterate on custom local runtimes (such as `runtime-next` on port 8003) and the DeepSeek Harness (`dsh`) agent with custom plugins (`tool-code-verify`, `infinite-agent`).

---

## 🎯 Why This Benchmark?

1. **Zero External Dependencies**: All exercises require standard Python 3. No external pip modules, databases, or complex system headers are required.
2. **Binary Truth Verification**: Every evaluation runs standard `pytest`. Exit code 0 means passed; exit code 1 means failed.
3. **Ultra-Fast Cycle Times**: A local 4B/8B model can read the prompt, write the fix, and trigger the test runner in **under 3 to 5 seconds**.
4. **Dual-Mode Sandboxing**:
   - **Local Subprocess (<100ms)**: Executes in an isolated `tempfile.TemporaryDirectory` with host pytest.
   - **Container Sandbox**: Mounts the temporary workspace into a clean Docker/Podman container (`python:3.11-slim`) with `--container`.
5. **Configurable Code Editing**: Supports both whole-file markdown blocks (`--edit-format whole`) and canonical Aider SEARCH/REPLACE diff blocks (`--edit-format diff`).
6. **Multi-Turn Repair Loop**: Configurable `--max-turns` (default 5). If tests fail, the harness feeds the pytest error traceback back to the model for iterative repair.

---

## 📦 Bundled Tasks

Located in `benchmarks/aider_bench/tasks/`:

| Task | Domain / Concepts Tested | Target Module |
| :--- | :--- | :--- |
| **`hello_world`** | Literal starting task; string return | `hello_world.py` |
| **`leap`** | Gregorian calendar conditional boolean logic | `leap.py` |
| **`reverse_string`** | String indexing / manipulation | `reverse_string.py` |
| **`two_fer`** | String formatting & default argument handling | `two_fer.py` |
| **`grains`** | Exponential arithmetic & exception handling (`ValueError`) | `grains.py` |

---

## 🛠️ Usage

### 1. Direct Runtime Benchmark (`--mode direct`)
Evaluates raw `/v1/chat/completions` performance against `runtime-next` (port 8003) or any OpenAI-compatible server.

```bash
# Run all 5 tasks against runtime-next (port 8003)
uv run python -m benchmarks.aider_bench.runner --base-url http://127.0.0.1:8003/v1 --model qwen3.5:4b-rust

# Run specific tasks with Aider SEARCH/REPLACE diff format and 3 repair turns
uv run python -m benchmarks.aider_bench.runner -t hello_world,leap --edit-format diff --max-turns 3

# Test with a specific LoRA domain adapter (e.g. astral or python_modern)
uv run python -m benchmarks.aider_bench.runner --adapter astral

# Execute inside Podman/Docker container sandbox
uv run python -m benchmarks.aider_bench.runner --container

# Export scorecard to JSON
uv run python -m benchmarks.aider_bench.runner -o results/benchmarks/aider_hello_world.json
```

### 2. Agentic DSH Benchmark (`--mode agentic`)
Drives the DeepSeek Harness Python SDK (`DeepSeekHarness`) mounted with custom plugins (`tool-code-verify` and `infinite-agent`), testing whether the agent can autonomously diagnose, edit, and pass tests.

```bash
uv run python -m benchmarks.aider_bench.runner --mode agentic --base-url http://127.0.0.1:8003/v1 --model qwen3.5:4b-rust
```

---

## 🧪 Offline Unit Tests (Zero GPU Required)

Run the strict offline test suite testing task loading, whole-file extraction, diff parsing, pytest isolation, and container command generation:

```bash
uv run pytest benchmarks/aider_bench/tests/test_harness.py -v
```
Output:
```text
benchmarks/aider_bench/tests/test_harness.py::test_tasks_load_all_five PASSED
benchmarks/aider_bench/tests/test_harness.py::test_task_single_load PASSED
benchmarks/aider_bench/tests/test_harness.py::test_extract_whole_file_markdown PASSED
benchmarks/aider_bench/tests/test_harness.py::test_apply_search_replace_diff_single_block PASSED
benchmarks/aider_bench/tests/test_harness.py::test_apply_search_replace_diff_multi_block PASSED
benchmarks/aider_bench/tests/test_harness.py::test_apply_search_replace_diff_not_found PASSED
benchmarks/aider_bench/tests/test_harness.py::test_apply_edit_dispatcher PASSED
benchmarks/aider_bench/tests/test_harness.py::test_executor_failing_stub PASSED
benchmarks/aider_bench/tests/test_harness.py::test_executor_passing_solution PASSED
benchmarks/aider_bench/tests/test_harness.py::test_executor_leap_passing_solution PASSED
benchmarks/aider_bench/tests/test_harness.py::test_executor_two_fer_passing_solution PASSED
benchmarks/aider_bench/tests/test_harness.py::test_build_container_command PASSED
benchmarks/aider_bench/tests/test_harness.py::test_direct_harness_prompt_builder PASSED
============================== 13 passed in 0.56s ==============================
```

---

## 📊 Telemetry & Metrics Tracked

- **Pass@1**: Proportion of tasks solved correctly on the initial attempt (turn 1).
- **Pass@k**: Proportion of tasks solved within $k$ iterative repair turns.
- **TTFT (ms)**: Time to First Token on streaming output.
- **Decode Speed (tok/s)**: Raw generation throughput.
- **Token Usage**: Total tokens consumed across prompt, generation, and multi-turn feedback.
- **Wall Clock Duration**: End-to-end latency per task and overall suite.
