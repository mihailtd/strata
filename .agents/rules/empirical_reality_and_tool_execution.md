# Rule: Absolute Empirical Reality & Live OS Tool Execution

## Core Principle
Never simulate, mock, or fake tool execution, training passes, or test results when tasked with building, evaluating, or verifying code. Always execute real OS tools with live subprocess telemetry.

## Invariants & Guardrails
1. **Genuine GPU Backpropagation**:
   - Training runs must execute real PyTorch backward passes on the active GPU device (`hip:0` / `cuda:0`) and save verifiable `.safetensors` to disk.
2. **Live OS Execution for Code Projects**:
   - When verifying a project or microservice, run actual system binaries (`uv run pytest`, `uv run ruff check`) on the host filesystem.
   - Capture live return codes, stdout, and stderr. Autonomously fix real syntax or import errors until all tests pass with Exit Code 0.
3. **Transparent Methodology Distinction**:
   - Clearly distinguish between:
     - **Synthetic Stress Benchmarks** (measuring token throughput and KV cache math using structured mock traces).
     - **Live Agent Workflows** (executing real terminal commands and file I/O on the local machine).
