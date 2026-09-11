# runtime-ipwf — In-Place Weight Folding Inference Engine (3B / 9B)

> **Single-purpose engine.** Serves unquantized **Qwen3.5-4B** and **Qwen3.5-9B** models with in-place weight mutation, CUDA Graph decoding, and Riemannian domain-expert LoRA teams. For the quantized 27B model, see [`runtime-triton`](../runtime-triton/).

---

## What It Is

A focused, single-tenant OpenAI-compatible REST server that:

1. Loads a full-precision (bfloat16) base model into VRAM.
2. Applies **FlashNorm** — RMSNorm scale weights are folded into downstream Linear layers at load time, eliminating norm overhead per token.
3. Holds six domain expert LoRA adapters in VRAM simultaneously (at static addresses).
4. Mutates weights **in place** (`W_live = W0 + s * U@V`) when switching experts — no memory copy, no reload.
5. Dispatches through a **pre-captured CUDA/HIP Graph** for zero Python overhead per decode step.
6. Optionally runs **Bucketed Speculative Decoding** with a Multi-Token Prediction (MTP) draft head.
7. Uses a **Riemannian Team Router** to co-activate multiple expert adapters simultaneously and a **NOTEARS Causal Scheduler** to pre-fold the predicted next expert.
8. Executes requests strictly in **arrival order** — one at a time, no concurrency.

**Hardware:** AMD RX 7900 XTX (Navi 31), 24 GB GDDR6, ROCm.

---

## Port

**8002** (default). Configure via `PORT` env var.

> Port 8002 was chosen to avoid conflict with `runtime-triton` (8000) and `runtime-llama` (8001).

---

## How to Start

```bash
# From repo root — standby mode (engine loads when you POST /api/engine/load)
./src/runtime-ipwf/run_server.sh

# Pre-load the 4B model on startup
AUTO_LOAD_MODEL=1 ./src/runtime-ipwf/run_server.sh

# Load the 9B model instead
DEFAULT_MODEL_ID=Qwen/Qwen3.5-9B AUTO_LOAD_MODEL=1 ./src/runtime-ipwf/run_server.sh

# Disable speculative decoding (e.g. for pure baseline measurement)
SPECULATIVE_DECODE=0 ./src/runtime-ipwf/run_server.sh
```

Or directly via Python:

```bash
cd /path/to/gnn-experiment
PYTHONPATH=src uv run python -m src.runtime_ipwf.server
```

### Engine Load / Unload

In standby mode, load explicitly:

```bash
# Load 4B (default)
curl -X POST http://localhost:8002/api/engine/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "Qwen/Qwen3.5-4B"}'

# Load 9B
curl -X POST http://localhost:8002/api/engine/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "Qwen/Qwen3.5-9B"}'

# Check status
curl http://localhost:8002/api/engine/status

# Unload (free VRAM)
curl -X POST http://localhost:8002/api/engine/unload
```

---

## Supported Models

| Model alias | Base | Expert |
|---|---|---|
| `Qwen/Qwen3.5-4B` | 4B bfloat16 | None (pristine) |
| `qwen3.5-4b-dynamic` | 4B | Auto-routed from prompt |
| `qwen3.5-4b-astral` | 4B | Astral (uv, ruff, pyproject) |
| `qwen3.5-4b-postgresql` | 4B | PostgreSQL 17 + pgvector |
| `qwen3.5-4b-duckdb` | 4B | DuckDB / Parquet / OLAP |
| `qwen3.5-4b-fastapi` | 4B | FastAPI + async web |
| `qwen3.5-4b-financial` | 4B | Financial modeling (VaR/CVaR) |
| `qwen3.5-4b-python-modern` | 4B | PEP 695 / modern Python |
| `Qwen/Qwen3.5-9B` | 9B bfloat16 | None (pristine) |
| `qwen3.5-9b-<domain>` | 9B | Same domains as 4B |
| `astral`, `postgresql`, etc. | Active base | Domain alias (short form) |
| `dynamic` | Active base | Auto-routed from prompt |

---

## API Endpoints

```
GET  /health                       Liveness check
GET  /v1/models                    List available model aliases
POST /v1/chat/completions          Chat completion (streaming or JSON)
POST /v1/completions               Legacy completion
GET  /api/engine/status            Full engine state (model, spec, team, VRAM)
POST /api/engine/load              Load the base model + experts + CUDA graphs
POST /api/engine/unload            Unload and free VRAM
POST /api/engine/reset             Force reset VRAM state
GET  /api/telemetry                Cumulative request/token counters
```

### Example — Non-streaming

```bash
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5-4b-duckdb",
    "messages": [{"role": "user", "content": "Write a DuckDB query with QUALIFY"}],
    "stream": false
  }'
```

### Example — Streaming

```bash
curl -N http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dynamic",
    "messages": [{"role": "user", "content": "What is a Cholesky decomposition?"}],
    "stream": true
  }'
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8002` | HTTP bind port |
| `HOST` | `0.0.0.0` | HTTP bind address |
| `DEFAULT_MODEL_ID` | `Qwen/Qwen3.5-4B` | Base model to load |
| `AUTO_LOAD_MODEL` | `0` | Set to `1` to pre-load on startup |
| `SPECULATIVE_DECODE` | `1` | Enable bucketed speculative decoding |
| `SPECULATIVE_K` | `2` | Speculative draft depth |
| `SPECULATIVE_MAX_SEQ_LEN` | `4096` | Sequence length for speculative graphs |
| `FLASH_NORM_FOLD` | `1` | Fold RMSNorm into Linear (4B only; skip for 9B) |
| `MAX_SEQ_LEN` | `16384` (4B) / `4096` (9B) | Max context length |
| `RING_BUFFER_MODE` | `selective_hybrid` | KV ring strategy: `dense\|poet\|selective_hybrid\|pointer` |
| `KV_CACHE_DTYPE` | `bfloat16` | KV cache precision (INT4 is banned) |
| `SPECULATIVE_RANGE_GATE` | `1` | Enable Range Statistic speculative gate |
| `SPECULATIVE_RANGE_THRESHOLD` | `5.0` | Logit range spread threshold |
| `STREAM_ORDER_TIMEOUT_S` | `300` | Streaming client timeout (seconds) |

---

## A/B Testing Protocol

This engine uses **all 24 GB VRAM**. Never run it alongside `runtime-triton`, `runtime-llama`, or `runtime-ollama`.

Sequential A/B procedure:

```bash
# Step 1 — Start IPWF engine
DEFAULT_MODEL_ID=Qwen/Qwen3.5-9B AUTO_LOAD_MODEL=1 ./src/runtime-ipwf/run_server.sh

# Step 2 — Run DSH harness evaluation
cd src/harness && uv run python eval.py --endpoint http://localhost:8002 \
  --output results/benchmarks/scorecard_ipwf_9b.json

# Step 3 — Stop the engine
curl -X POST http://localhost:8002/api/engine/unload
# Wait for VRAM to clear, then Ctrl+C the server process

# Step 4 — Start a different engine (e.g. runtime-triton)
AUTO_LOAD_MODEL=1 ./src/runtime-triton/run_server.sh

# Step 5 — Run same harness evaluation against port 8000
# Step 6 — Compare the two scorecard JSON files
```

---

## Module Dependencies

All engine modules live in `src/runtime/` (the shared package):

- `runtime.novel_peft` — `FoldableExpert`, `WeightFoldingEngine`, `set_hard_vram_cap`
- `runtime.cuda_graph` — `FoldedCudaGraphDecoder` (pre-captured HIP graph)
- `runtime.fused_norm` — FlashNorm: `inject_exact_rmsnorm`, `fold_rmsnorm_into_linear`, `scale_expert_factors_for_folded_norms`
- `runtime.dynamic_team_router` — `RiemannianTeamRouter` (geodesic multi-expert activation)
- `runtime.notears_causal_scheduler` — `NotearsCausalScheduler` (NOTEARS-DAG predictive pre-folding)
- `runtime.bucketed_speculative` — `BucketedSpeculativeDecoder` (optional)
- `runtime.mtp_draft` — `Qwen35MTPDraftHead` (MTP speculative draft head)
- `runtime.range_statistic_gate` — `RangeStatisticGate` (speculative early-exit)
- `runtime.adapter_stacker` — Multi-adapter stacking utilities
- `runtime.gpu_preflight` — VRAM exclusivity + sysfs telemetry
- `runtime.canon` — Canonical paths, KV-cache validation, attention config
