# runtime-triton — Native Triton 27B W4A16 Inference Engine

> **Single-purpose engine.** Serves one model: **Qwen3.8:27B**, fully quantized to W4A16, running entirely inside custom AMD ROCm Triton GEMV kernels. For unquantized 3B/9B models, see [`runtime-ipwf`](../runtime-ipwf/).

---

## What It Is

A focused, single-tenant OpenAI-compatible REST server that:

1. Unpacks the 27B GGUF blob from disk and loads all 64 transformer layers as W4A16 weight tensors directly into GDDR6 at static addresses.
2. Runs inference via custom Triton GEMV kernels with 128-bit memory coalescing (~620 GB/s GDDR6 saturation).
3. Swaps LoRA domain adapters **in-memory** at inference time (zero model reload, sub-millisecond swap).
4. Executes requests strictly in **arrival order** — one at a time, no concurrency.

**Architecture:**
- 48 Gated DeltaNet SSM blocks + 16 Full Attention blocks (hybrid Qwen3.5 architecture)
- In-Register Mixture-of-Adapters (MoA) — multiple LoRA factors accumulated in registers
- Constant O(1) recurrent state handoff (no KV cache growth across turns)
- Hardware: AMD RX 7900 XTX (Navi 31), 24 GB GDDR6, ROCm

---

## Port

**8000** (default). Configure via `PORT` env var.

---

## How to Start

```bash
# From repo root — standby mode (engine loads on first request)
./apps/runtime-triton/run_server.sh

# Pre-load the 27B model on startup
AUTO_LOAD_MODEL=1 ./apps/runtime-triton/run_server.sh

# Different port
PORT=9000 ./apps/runtime-triton/run_server.sh
```

Or directly via Python (this is a self-sufficient, independently-installable
uv project — no repo-wide PYTHONPATH needed):

```bash
cd apps/runtime-triton
uv sync
uv run python server.py
```

### Engine Load / Unload

After startup in standby mode, trigger load explicitly:

```bash
# Load the 27B engine
curl -X POST http://localhost:8000/api/engine/load \
  -H "Content-Type: application/json" \
  -d '{"num_layers": 64}'

# Check status
curl http://localhost:8000/api/engine/status

# Unload (free VRAM)
curl -X POST http://localhost:8000/api/engine/unload
```

---

## Supported Models

All aliases resolve to the same loaded W4A16 weights. The LoRA suffix determines which domain adapter is swapped in:

| Model alias | LoRA adapter |
|---|---|
| `qwen3.8:27b` | Auto-routed (from prompt content) |
| `qwen3.8:27b-triton` | Auto-routed |
| `qwen3.8-27b-auto` | Auto-routed |
| `qwen3.8-27b-astral` | Astral (uv, ruff, pyproject) |
| `qwen3.8-27b-postgresql` | PostgreSQL 17 + pgvector |
| `qwen3.8-27b-duckdb` | DuckDB / Parquet / OLAP |
| `qwen3.8-27b-fastapi` | FastAPI + async web |
| `qwen3.8-27b-financial` | Financial modeling (VaR/CVaR) |
| `qwen3.8-27b-python-modern` | PEP 695 / modern Python |

---

## API Endpoints

Standard OpenAI-compatible API:

```
GET  /health                       Liveness check
GET  /v1/models                    List available model aliases
POST /v1/chat/completions          Chat completion (streaming or JSON)
POST /v1/completions               Legacy completion (wraps into user message)
GET  /api/engine/status            Live engine state + telemetry
POST /api/engine/load              Load the 27B engine into VRAM
POST /api/engine/unload            Unload and free VRAM
GET  /api/telemetry                Cumulative token/request counters
```

### Example — Non-streaming

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27b-astral",
    "messages": [{"role": "user", "content": "How do I add a dependency with uv?"}],
    "stream": false
  }'
```

### Example — Streaming (SSE)

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8:27b",
    "messages": [{"role": "user", "content": "Explain DuckDB QUALIFY clause"}],
    "stream": true
  }'
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | HTTP bind port |
| `HOST` | `0.0.0.0` | HTTP bind address |
| `AUTO_LOAD_MODEL` | `0` | Set to `1` to pre-load on startup |
| `STREAM_ORDER_TIMEOUT_S` | `300` | Seconds to wait for streaming client before moving on |

---

## A/B Testing Protocol

This engine uses **all 24 GB VRAM**. Never run it alongside `runtime-ipwf`, `runtime-llama`, or `runtime-ollama`.

Sequential A/B procedure:

```bash
# Step 1 — Start this engine
AUTO_LOAD_MODEL=1 ./apps/runtime-triton/run_server.sh

# Step 2 — Run DSH harness evaluation
cd apps/harness && uv run python eval.py --endpoint http://localhost:8000 \
  --output results/benchmarks/scorecard_triton.json

# Step 3 — Fully stop this engine (Ctrl+C or kill), verify VRAM free:
curl http://localhost:8000/api/engine/unload  # or just kill the process

# Step 4 — Start the other engine (e.g. runtime-llama on port 8001)
./apps/runtime-llama/server --port 8001 ...

# Step 5 — Run the same harness evaluation against port 8001
# Step 6 — Compare results/benchmarks/scorecard_triton.json vs scorecard_llama.json
```

---

## Module Dependencies

This is a **self-sufficient, independently-installable uv project** — its own
`pyproject.toml`, `uv.lock`, and `.venv`. It vendors its own copy of the
serving engine rather than importing the shared `apps/runtime` package:

- `native_27b_engine.py` — `Native27BEngine` class (GGUF unpacking + Triton kernels), vendored locally
- `gguf_unpacker.py` — Streaming GGUF blob reader, vendored locally
- `w4a16_loader.py` — W4A16 weight dequantization, vendored locally
- `triton_w4a16.py` — Custom Triton GEMV kernel (W4A16), vendored locally

The *only* thing pulled in from outside this directory is
[`runtime-common`](../runtime-common/) (a path dependency in `pyproject.toml`), for:

- `runtime_common.gpu_preflight` — VRAM exclusivity checks + sysfs telemetry
- `runtime_common.canon` — Canonical constants, KV-cache validation, attention config

The vendored engine files are a deliberate copy of `apps/runtime`'s versions,
not a re-export — see the docstring at the top of `native_27b_engine.py` for
why. Fix bugs in both copies, or accept the drift as the cost of independence.
