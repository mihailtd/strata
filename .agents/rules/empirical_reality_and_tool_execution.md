# Rule: Empirical Reality & The 4-Stage Research Pipeline

## Core Principle
Never simulate, mock, or fake tool execution, training passes, or test results when tasked with building, evaluating, or verifying code. Always execute real OS tools with live subprocess telemetry.

## Mandatory 4-Stage Research & Experimentation Pipeline

Whenever designing, optimizing, or validating a new feature, kernel, or architectural component (e.g., KV cache, LoRA blending, GEMV kernels, speculation), follow this sequential 4-stage lifecycle:

### Stage 1: Synthetic Benchmark with Dedicated README
* Create an isolated benchmark script under `benchmarks/` (e.g. `benchmark_<feature>.py`).
* Accompany the benchmark with a dedicated `benchmarks/README_<FEATURE>.md` documenting:
  1. **Overview & Motivation**: What the feature is, why it improves the system, and what problem it solves.
  2. **Mathematical / Algorithmic Formulations**: Formulas, memory footprints, and data structures.
  3. **Hypothesis & Risk Analysis**: What are we testing, failure modes, precision limits, and potential overheads.
  4. **Empirical Synthetic Metrics**: Structured comparison across workloads/sequence lengths (e.g. 2K → 32K context).
* Save raw metrics to `results/benchmarks/<feature>_benchmark.json`.

### Stage 2: Real-World Empirical Test on Real Weights & Hardware
* Create a verification script (e.g. `benchmarks/eval_<feature>.py`) using real trained weights extracted from the GGUF blob (e.g. `models/qwen3.8-27b-triton/layer_3.pt`) on the active GPU (`cuda:0` / `gfx1100`).
* Execute multi-step autoregressive decode passes to measure:
  - Output numerical fidelity: Cosine similarity ($\ge 0.999$), max absolute error, zero NaNs/Infs.
  - Memory footprint and per-step latency.
* **Strict Gate**: Do NOT integrate code into the production runtime until this empirical test produces a verifiable `>>> PASSED <<<` result.

### Stage 3: Runtime Integration & Automated Unit Tests
* Integrate the validated component into the actual runtime (`src/runtime/`).
* Maintain strict backward compatibility (duck-typing, non-breaking function signatures).
* Add automated unit tests to `tests/test_<module>.py` covering functionality, edge cases, and memory reset.
* Run `uv run pytest` and verify all tests pass with Exit Code 0.

### Stage 4: Real End-to-End Evaluation & Dual A/B Comparison
* Execute an end-to-end evaluation script (e.g. `benchmarks/eval_end_to_end_real_usecase.py`) using real domain specialist prompts (Astral, PostgreSQL, DuckDB, FastAPI) through the live runtime API (`http://127.0.0.1:8000`).
* Conduct a mandatory **Dual A/B Comparison**:
  1. **A/B vs. Previous Version (Before vs Now)**: Quantify the exact wall-clock speedup, memory savings per layer, and allocator churn elimination on our custom engine.
  2. **A/B vs. State-of-the-Art Baseline (`llama.cpp`)**: Benchmark our custom runtime against the C++ HIP baseline (`qwen3.8:27b` on port 8001), measuring TTFT, throughput (tok/s), and pinpointing exact hardware/dispatch bottlenecks.
* Save structured results to `results/benchmarks/` and summarize in `walkthrough.md`.

---

## General Invariants & Guardrails
1. **Genuine GPU Backpropagation**:
   - Training runs must execute real PyTorch backward passes on the active GPU device (`hip:0` / `cuda:0`) and save verifiable `.safetensors` to disk.
2. **Live OS Execution for Code Projects**:
   - When verifying a project or microservice, run actual system binaries (`uv run pytest`, `uv run ruff check`) on the host filesystem.
   - Capture live return codes, stdout, and stderr. Autonomously fix real syntax or import errors until all tests pass with Exit Code 0.
