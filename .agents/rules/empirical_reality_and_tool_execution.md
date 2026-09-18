# Rule: Absolute Engineering Integrity, Complete Implementations & Zero-Mock Policy

## Core Principles
1. **Absolute Engineering Truth**: Never simulate, mock, fake, or hardcode data, weights, training passes, benchmarks, or tool execution. Every line of code, model weight, and metric must be genuine and verifiable.
2. **Complete & Pure Implementations**: Build complete, authentic implementations or do not build them at all. Never create mock workarounds, stub files, or synthetic placeholders that pretend to implement functionality.
3. **No Deceptive Naming or Facades**: Never wrap third-party software (e.g., Ollama, HuggingFace, stock llama.cpp) and label it as a custom proprietary engine, fleet, or novelty. Third-party baselines must be called exactly what they are.
4. **Verifiable Hardware Telemetry**: All throughput, latency, VRAM, and accuracy numbers must originate from live subprocess telemetry on physical hardware. Never fabricate or hardcode performance numbers.
5. **Labor-Neutral Scoping vs. Hardware Time**: Never evaluate or describe engineering work in terms of calendar days, developer hours of effort, or engineer headcount. However, **GPU execution time is critical**: benchmark duration, GPU busy time, and device lock duration must always be anticipated, measured, and reported. Scope and trade-offs are strictly evaluated by Technical Complexity, Risk, Hardware Execution Cost, Return, and Probability of Success.

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
* Integrate the validated component into the actual runtime (`apps/runtime/`).
* Maintain strict backward compatibility (duck-typing, non-breaking function signatures).
* Add automated unit tests to `tests/test_<module>.py` covering functionality, edge cases, and memory reset.
* Run `uv run pytest` and verify all tests pass with Exit Code 0.

### Stage 4: Real End-to-End Evaluation & Sequential Dual A/B Comparison
* Execute an end-to-end evaluation script (e.g. `benchmarks/eval_end_to_end_real_usecase.py`) using real domain specialist prompts (Astral, PostgreSQL, DuckDB, FastAPI) through the live runtime API (`http://127.0.0.1:8000`).
* Conduct a mandatory **Sequential Dual A/B Comparison**:
  - **Strict Sequential Protocol**: Never run engines concurrently. Benchmark Runtime A -> Fully Stop Runtime A (verify 0 VRAM) -> Benchmark Runtime B -> Fully Stop Runtime B -> Compare Scorecards.
  1. **A/B vs. Previous Version (Before vs Now)**: Quantify the exact wall-clock speedup, memory savings per layer, and allocator churn elimination on our custom engine.
  2. **A/B vs. State-of-the-Art Baselines (`llama.cpp` / Ollama)**: Benchmark our custom runtime against the standalone baseline on its isolated port, measuring TTFT, throughput (tok/s), and pinpointing exact hardware/dispatch bottlenecks.
* Save structured results to `results/benchmarks/` and summarize in `walkthrough.md`.

---

## General Invariants & Guardrails
1. **Genuine GPU Backpropagation**:
   - Training runs must execute real PyTorch backward passes on the active GPU device (`hip:0` / `cuda:0`) and save verifiable `.safetensors` to disk.
2. **Live OS Execution for Code Projects**:
   - When verifying a project or microservice, run actual system binaries (`uv run pytest`, `uv run ruff check`) on the host filesystem.
   - Capture live return codes, stdout, and stderr. Autonomously fix real syntax or import errors until all tests pass with Exit Code 0.

---

## 5. Prohibited Practices & Violations (Zero Tolerance)

| Prohibited Practice | Why It Is Forbidden | Required Authentic Alternative |
|---------------------|----------------------|--------------------------------|
| **Dummy Weights / Random Tensors** | Generating `torch.randn` or zero tensors and saving them as "trained adapters" or "models" poisons the repository with fake artifacts. | Run genuine PyTorch backpropagation on real training datasets, or do not create the adapter file. |
| **Architectural Facades & Disguised Proxies** | Calling an external HTTP service (e.g., Ollama on 11434) and wrapping it as "Custom MoA Fleet" or "Triton Engine" misleads the user and hides technical reality. | Clearly name external connections (e.g., `OLLAMA_BASELINE_PROXY`). Never pretend an external binary is our custom Triton engine. |
| **Hardcoded Performance Metrics** | Returning synthetic tok/s or latency numbers without running a live hardware timer produces fictitious claims. | Measure elapsed wall-clock time (`time.perf_counter()`) and token counts from live inference runs on the GPU. |
| **Mock Inference / Stubs** | Using `time.sleep()` or canned strings to simulate model output or kernel execution hides memory leaks, crashes, and real bottlenecks. | Always invoke the real engine (`Native27BEngine`, compiled binary, or live server) with real tensors. |
| **"Temporary" Workarounds Disguised as Solutions** | Leaving partial stubs with comments claiming future implementation while reporting the task as "done". | Either implement the complete, pure solution or explicitly document the limitation as unbuilt work. |
| **Human Labor & Calendar Estimates** | Framing technical tasks by calendar days, weeks, developer hours, or team size ("will take 3 days", "needs 2 engineers"). These abstractions are subjective, noisy, and technically irrelevant. *(Note: Physical GPU benchmark runtime and device occupancy duration are hardware metrics, not labor estimates, and remain strictly required).* | Frame work strictly by: (1) Technical Complexity, (2) Technical & Numerical Risk, (3) Hardware/GPU Execution Budget, (4) Measurable Return / Payoff, and (5) Probability of Success. |

---

## 6. Technical Scoping & Decision Framework (Complexity & Risk vs. Return)

When scoping features, writing architectural proposals, or comparing alternatives, evaluate options strictly across the following five technical dimensions:

1. **Technical Complexity**:
   - Lines of code and architectural surface area.
   - Algorithmic structure (e.g., standard GEMM reuse vs. custom triangular solvers).
   - Memory hierarchy requirements (LDS allocation, VGPR/register pressure, cache-line alignment).
2. **Technical & Numerical Risk**:
   - Sensitivity to floating-point precision (fp32 vs. bf16 rounding accumulation).
   - Potential for state corruption (read-modify-write recurrent state vs. immutable buffers).
   - Risk of regression against existing verified test suites.
3. **Hardware & GPU Execution Budget**:
   - Anticipated benchmark execution duration (e.g., 30 seconds vs. 15 minutes).
   - GPU occupancy and lock duration on the single 24GB VRAM device.
4. **Expected Return**:
   - Quantified performance ceiling (e.g., eliminating a 246 ms sequential bottleneck vs. an 85 ms attention loop).
   - Strategic functional capabilities unlocked (e.g., serving specialized domain LoRA adapters vs. vanilla base model only).
5. **Probability of Success**:
   - Mathematical soundness of the closed-form formulation.
   - Availability of ground-truth oracle references (e.g., PyTorch reference implementations) for bit-identical validation.

