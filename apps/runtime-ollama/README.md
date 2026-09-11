# Standalone Ollama Baseline Runtime (`runtime-ollama`)

A cleanly isolated, standalone baseline harness running upstream **Ollama** (`/usr/bin/ollama`) with AMD ROCm hardware acceleration (`gfx1100`, AMD Radeon RX 7900 XTX).

---

## Architecture & Responsibilities

- **Scope**: External SOTA / Commodity Baseline for comparative A/B evaluation, benchmark comparisons, and dataset generation.
- **Isolation**: Completely separated from `src/runtime` (Python/Triton), `src/runtime-llama` (C++ llama.cpp), and `src/runtime-next` (Rust).
- **Default Port**: `11434` (Ollama native API and OpenAI-compatible `/v1/chat/completions`).
- **Single-Engine Rule**: When benchmarking or running tasks, **never run Ollama concurrently with other runtimes**. Run one runtime, benchmark it, stop it, and then benchmark the other.

---

## Quick Start

### 1. Verify Setup & Available Baseline Models
```bash
cd src/runtime-ollama
./setup.sh
```

### 2. Start Standalone Ollama Server
```bash
cd src/runtime-ollama
./run_server.sh
```

This sets optimal AMD ROCm environment flags:
- `HSA_OVERRIDE_GFX_VERSION=11.0.0`
- `OLLAMA_FLASH_ATTENTION=1`
- `OLLAMA_KV_CACHE_TYPE=q8_0`
- `OLLAMA_HOST=127.0.0.1:11434`

### 3. Stop Ollama Server (Release All GPU VRAM)
```bash
cd src/runtime-ollama
./run_server.sh stop
```

---

## Sequential A/B Benchmarking Workflow

To benchmark our custom Triton runtime against the Ollama baseline:

1. **Step 1 (Triton Benchmark)**:
   ```bash
   # Start custom Triton runtime (port 8000)
   PYTHONUNBUFFERED=1 PYTHONPATH=. uv run python -m src.runtime.server &
   # Connect DSH harness / run benchmark
   uv run python benchmarks/eval_end_to_end_real_usecase.py --endpoint http://127.0.0.1:8000/v1 --out results/benchmarks/scorecard_triton.json
   # Stop Triton runtime
   pkill -f "src.runtime.server"
   ```

2. **Step 2 (Ollama Benchmark)**:
   ```bash
   # Start Ollama baseline (port 11434)
   ./src/runtime-ollama/run_server.sh
   # Connect DSH harness / run identical benchmark
   uv run python benchmarks/eval_end_to_end_real_usecase.py --endpoint http://127.0.0.1:11434/v1 --model ornith-1.5:35b --out results/benchmarks/scorecard_ollama.json
   # Stop Ollama baseline
   ./src/runtime-ollama/run_server.sh stop
   ```

3. **Step 3 (Compare)**:
   - Compare `scorecard_triton.json` vs `scorecard_ollama.json` directly.
