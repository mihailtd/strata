# ⚡ In-Place Folding OpenAI-Compatible Engine & REST Server

An OpenAI-compatible local REST server built on **in-place weight folding**: micro-expert adapters are merged into the base weights in VRAM so inference runs with no adapter wrappers in the execution path.

Exposes local micro-experts (`postgresql`, `astral`, `financial_planning`) through standard
`/v1/chat/completions` and `/v1/models` endpoints. A swap is an in-place mutation
$W_{\text{live}} \leftarrow W_0 + s \cdot U V$ measured at **18.8-19.3 ms** with **0 bytes**
transient VRAM churn (peak-above-baseline).

**Measured, on an RX 7900 XTX (gfx1100) / ROCm 7.2:**

| | |
| --- | --- |
| decode, adapter wrapped | 25.09 tok/s |
| decode, adapter folded | **30.38 tok/s** (+21.1%, = unadapted base speed) |
| expert swap (factors resident) | 18.8-19.3 ms; pays for itself after ~3 tokens |
| putting the swap behind an API boundary | +0.012 ms (a ~20 byte command) |
| CUDA graph replay on top of folding | **~1.01x** -- decode here is kernel-execution bound, not launch bound |
| AMD AITER | **not used**: its rmsnorm silently returns zeros at hidden=2560 on gfx1100, and the one correct kernel is 2-6x slower than ATen |


---

## 🚀 Key Performance Specs

All figures below are gated on a correctness check first: the folded model must
reproduce the wrapped adapter's tokens, and graph replay must reproduce eager
greedy decode token-for-token. Speed numbers taken without that gate passing are
not reported, because a graph that decodes against a stale mask is *faster*
precisely because it is doing the wrong thing.

| Optimization | Decode | Notes |
| :--- | :---: | :--- |
| adapter wrapped (`NovelLoraLinear`) | $25.09\text{ tok/s}$ | 256 extra kernel launches/token at batch 1 |
| **adapter folded into base weights** | **$30.38\text{ tok/s}$ ($+21.1\%$)** | reaches unadapted base speed; does not exceed it |
| folded + CUDA graph replay | $\approx 1.01\times$ over folded alone | decode here is kernel-execution bound, not launch bound |
| in-place expert swap | $18.8-19.3\text{ ms}$, $0$ bytes transient churn | break-even after ~3 generated tokens |

> **Retracted.** An earlier version of this table claimed $32.89\text{ tok/s}$
> ($1.21\times$) for "AITER + Folded CUDA Graph". That arm was confounded four
> ways: only it had an adapter folded, it used a different generation loop, its
> timing window excluded prefill, and the graph was decoding incorrectly so it
> never hit EOS and ran to the token cap. AITER was also never active -- see
> `src/gnn_experiment/fused_norm.py` for the measurements.

---

## 💻 Quick Start: Launching the REST Server

To launch the production OpenAI-compatible REST server daemon on `http://127.0.0.1:8000`:

```bash
uv run --env-file .env python3 scripts/run_openai_api_server.py --host 127.0.0.1 --port 8000
```

The server initializes `Qwen/Qwen3.5-4B`, swaps in exact PyTorch RMSNorm stand-ins (CUDA-graph friendly; not a speedup), pre-loads the factor micro-experts, captures the CUDA/HIP graph once, and listens for HTTP requests.

---

## 🔌 Connecting Tools & Clients

### 1. Using OpenCode CLI

With `opencode.json` configured, you can select micro-experts directly using `opencode --model imb/<expert>`:

#### Bash / Linux / macOS
```bash
# Launch interactive session with Astral expert
opencode --model imb/astral

# One-off command execution with PostgreSQL expert
opencode run --model imb/postgresql "Design a PostgreSQL 18 schema for embeddings using pgvector"
```

#### Windows PowerShell
```powershell
# Launch interactive session with Astral expert
opencode --model imb/astral

# One-off command execution with PostgreSQL expert
opencode run --model imb/postgresql "Design a PostgreSQL 18 schema for embeddings using pgvector"
```

---

### 2. Using `curl` (HTTP REST Calls)

#### List Available Models (`GET /v1/models`)
```bash
curl http://127.0.0.1:8000/v1/models
```

#### Non-Streaming Chat Completion (`POST /v1/chat/completions`)
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "postgresql",
    "messages": [{"role": "user", "content": "Design a PostgreSQL schema for storing vector embeddings."}],
    "max_tokens": 64
  }'
```

#### SSE Token Streaming (`stream: true`)
```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "astral",
    "messages": [{"role": "user", "content": "Write a FastAPI similarity search endpoint."}],
    "max_tokens": 64,
    "stream": true
  }'
```

---

### 3. Using Official OpenAI Python SDK

```python
from openai import OpenAI

# Point client to local IMB REST server
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy")

# Instant zero-copy in-place expert swap to postgresql expert
response = client.chat.completions.create(
    model="postgresql",
    messages=[{"role": "user", "content": "Explain pgvector HNSW index creation."}],
    max_tokens=64,
)

print(response.choices[0].message.content)
```

---

## 🎯 Pre-Loaded Micro-Experts

| Expert ID | Primary Domain & Target Specialization | In-Place Activation Latency |
| :--- | :--- | :---: |
| `postgresql` (or `postgres`) | Database schema design, `pgvector`, HNSW indexes, SQL queries | **$19.29\text{ ms}$** |
| `astral` | Python FastAPI, standard libraries, high-performance async APIs | **$19.29\text{ ms}$** |
| `financial_planning` (or `fin`) | Financial planning strategy, sequence-of-returns risk, wealth modeling | **$19.29\text{ ms}$** |
| `base` (or `qwen3.5`) | Standard Qwen 3.5 4B base model without expert weights | **$0.00\text{ ms}$** |

---

## 🧪 Testing & Verification Suite

Run the full GPU integration test suite:

```bash
# Full REST API server integration test suite
uv run --env-file .env python3 scripts/test_openai_api_server.py


# Zero-recapture expert swapping synergy benchmark
uv run --env-file .env python3 scripts/benchmark_zero_recapture_swapping.py
```

---

## 🛠️ Project Structure

```text
gnn-experiment/
├── src/gnn_experiment/
│   ├── server.py               # Production FastAPI OpenAI REST server & IMB gatekeeper
│   ├── fused_norm.py           # Exact PyTorch RMSNorm stand-ins for Qwen3.5
│   ├── cuda_graph.py           # FoldedCudaGraphDecoder locked graph replay engine
│   └── novel_peft.py           # WeightFoldingEngine & FoldableExpert mechanics
├── scripts/
│   ├── run_openai_api_server.py # Server CLI launcher daemon
│   ├── test_openai_api_server.py# Complete REST server integration test suite
│   └── benchmark_zero_recapture_swapping.py # Synergy benchmark
└── README.md                   # System documentation
```
