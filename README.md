# ⚡ Autonomous Runtime & Speculative Execution Engine

An ultra-high-throughput, zero-drift inference runtime built on **In-Place Low-Rank Weight Folding**, **Transactional Recurrent-State Checkpointing (52.5 MB)**, and **Pointer-Stable CUDA Graph Replay** on AMD ROCm hardware (`gfx1100`).

Exposes local resident domain experts (`postgresql`, `astral`, `financial_planning`) through standard OpenAI `/v1/chat/completions` and `/v1/models` endpoints. A live expert swap is an in-place mutation $W_{\text{live}} \leftarrow W_0 + s \cdot U V$ executing in **18.08 ms** with **0 bytes transient VRAM churn**.

This README is a map and a quickstart, not a research log — deep empirical results live in [`docs/MEASURED_FINDINGS.md`](docs/MEASURED_FINDINGS.md) and [`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## 📂 Repository map

This is a Moon-orchestrated monorepo: `apps/*` are independently-managed projects (their own `pyproject.toml`/`.venv` where their dependencies genuinely conflict with the root's), everything else is repo-wide shared source, tooling, or data. **[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) explains the lifecycle these top-level folders implement** — read it before assuming a folder's name tells you its status.

### The engines (`apps/`)

**[`apps/RUNTIME.md`](apps/RUNTIME.md) is the map of every runtime** — what each one is, its port, whether it has its own venv, and which are ours vs. upstream baselines. Summary:

| Project | Role | Port | README |
| :--- | :--- | :--- | :--- |
| [`apps/runtime`](apps/runtime/) | **LEGACY** — original dual-purpose engine, now a shared module library only | 8000 | [README](apps/runtime/README.md) |
| [`apps/runtime-common`](apps/runtime-common/) | Shared canon + GPU-exclusivity guard, no server | n/a | [README](apps/runtime-common/README.md) |
| [`apps/runtime-triton`](apps/runtime-triton/) | Self-sufficient 27B W4A16 engine (quantized) | 8000 | [README](apps/runtime-triton/README.md) |
| [`apps/runtime-ipwf`](apps/runtime-ipwf/) | Self-sufficient in-place weight-folding engine, 3B/9B models (unquantized) | 8002 | [README](apps/runtime-ipwf/README.md) |
| [`apps/runtime-llama`](apps/runtime-llama/) | Upstream `llama.cpp` baseline (HIP build) | 8001 | [README](apps/runtime-llama/README.md) |
| [`apps/runtime-ollama`](apps/runtime-ollama/) | Upstream `ollama` baseline | 11434 | [README](apps/runtime-ollama/README.md) |
| [`apps/runtime-vllm`](apps/runtime-vllm/) | Upstream `vllm` baseline (ROCm) | 8004 | [README](apps/runtime-vllm/README.md) |
| [`apps/runtime-next`](apps/runtime-next/) | **Strata** — High-throughput native Rust/HIP serving engine for Qwen 3.5 | 8003 | [README](apps/runtime-next/README.md) |

Never run two of these against the GPU at once — see [apps/RUNTIME.md](apps/RUNTIME.md)'s "Single-GPU discipline."

### The other apps

- **[`apps/factory/`](apps/factory/)** — the training pipeline. Builds every LoRA adapter every runtime above loads. Its own [README](apps/factory/README.md) is comprehensive: how to train/retrain one or all adapters, how to add a domain, the full version changelog, and why its dependencies (`unsloth`, an older torch pin) can't share the root venv.
- **[`apps/harness/`](apps/harness/)** — integration with DSH (DeepSeek Harness), the coding-agent tool this repo standardizes on: web UI setup, model aliasing to specific LoRA experts, and the multi-subagent coordinator pipeline (`coordinator/`). See its [README](apps/harness/README.md).
- **[`apps/dashboard/`](apps/dashboard/)** — Next.js 16 web app for live telemetry, the adapter-training goldilocks curves, and a chat interface. See its [README](apps/dashboard/README.md).

### The measurement lifecycle

- **[`experiments/`](experiments/)** — Stage 1: synthetic microbenchmarks and quick empirical viability checks, pre-integration. [README](experiments/README.md).
- **[`benchmarks/`](benchmarks/)** — Stage 3: real end-to-end performance measurement of features already integrated into `apps/runtime*`/`apps/factory`. [README](benchmarks/README.md). Includes [`benchmarks/superseded/`](benchmarks/superseded/), one unified graveyard for retired ideas from any of the three trees below.
- **[`evals/`](evals/)** — task/rubric/ground-truth quality evaluation: Agent-as-Judge conversations, domain rubrics, SWE-bench, trained-adapter held-out evaluation, real sandbox execution scoring, and DSH agentic tool-use. [README](evals/README.md).
- **[`tests/`](tests/)** — not a flat suite; a signpost. Every real test lives inside the `apps/<project>/tests/` it tests, or colocated with the experiment/benchmark it pins. [README](tests/README.md) explains where to actually look.

**⚠️ Before citing anything from `experiments/` or a shipped default that traces back to one**, check [`docs/EXPERIMENT_REAUDIT_2026-09.md`](docs/EXPERIMENT_REAUDIT_2026-09.md) — a systematic re-audit found several results, including at least one default-on production feature, resting on synthetic data presented as a real measurement.

### Repo-wide tooling and shared resources

- **[`audit/`](audit/)** — the only 4 tools in the repo that are genuinely repo-wide (none owned by one app): `check_canon.py` (fails the build on a hardcoded decode budget or stale adapter version), `audit_adapters.py` (adapter inventory/drift detection), `audit_eval_rubrics.py` (catches eval rubrics that can never match), `check_gpu.py` (ROCm/GPU preflight). [README](audit/README.md).
- **[`data/`](data/)** — shared eval corpora (`evaluation_data*.jsonl` per domain), consumed across every tree above — not just `evals/`. [README](data/README.md) explains why it isn't nested under `evals/`. Training corpora live in `apps/factory/data/` instead.
- **[`results/`](results/)** — trained adapters (`results/adapters/`, gitignored, 3.8GB) and tracked benchmark/eval scorecards (small JSON/MD, kept in git as "the evidence behind every claim in this README").
- **[`agent_sandboxes/`](agent_sandboxes/)** — generated coding-task scenarios written to disk by a real coding agent under test, then graded with real `ruff`/`pytest`. Not application code — targets, not source. [README](agent_sandboxes/README.md).
- **`models/`** — GGUF/unpacked model weights (gitignored).
- **[`docs/`](docs/)** — see the Documentation Map below.

### Root-level files

- **`AGENTS.md`** — instructions auto-loaded by coding-agent tooling (route-to-expert-adapter protocol, the Zero-Mock integrity invariant, single-GPU execution rules). Stays at root because that's where agent tooling looks for it, unlike every other doc.
- **`pyproject.toml` / `uv.lock`** — the root project (`name = "runtime"`, `module-root = "apps"` — this **is** `apps/runtime`'s package manifest, it has no separate one of its own yet).
- **`.moon/`, `.moonignore`, `.prototools`** — Moon workspace/task config and the `proto` toolchain pins (Node/Rust versions); see `.moon/toolchain.yml`'s comments for the moon-vs-proto version-of-record split.
- **`ADAPTER_MANIFEST.json`** — the tracked-in-git output of `audit/audit_adapters.py --write`; diffed against on every run to catch silent adapter drift.

---

## ⚡ Setup & Quickstart

### 1. Root Python environment (covers `apps/runtime`, `apps/harness`, `benchmarks/`, `experiments/`, `evals/`, `audit/`)

```bash
uv sync   # from repo root
```

To actually serve a model, use one of the two independent runtimes below
(`runtime-triton` for 27B W4A16, `runtime-ipwf` for 3B/9B) — `apps/runtime`
itself is legacy and its `server.py` should not be started for new work; it's
kept only as a shared module library some benchmarks/evals/tests still import
from directly. See [`apps/RUNTIME.md`](apps/RUNTIME.md).

### 2. The dashboard

```bash
cd apps/dashboard
pnpm install
pnpm dev --port 3000
```

- **Dynamic Agent Matrix**: `http://localhost:3000/`
- **Morphing Studio & Chat**: `http://localhost:3000/chat`
- **NOTEARS Causal DAG**: `http://localhost:3000/dag`
- **Training Factory & Audit**: `http://localhost:3000/training`
- **Research Docs & Benchmarks**: `http://localhost:3000/docs`

### 3. Independent projects (each has its own venv — see each README for why)

```bash
cd apps/factory && uv sync          # training pipeline (unsloth, older torch pin)
cd apps/runtime-triton && uv sync   # self-sufficient 27B engine (quantized)
cd apps/runtime-ipwf && uv sync     # self-sufficient 3B/9B engine (unquantized)
cd apps/runtime-vllm && uv sync     # vLLM baseline (different ROCm wheel set)
cd apps/runtime-common && uv sync   # shared canon lib (no torch at all)
```

`apps/runtime-llama` and `apps/runtime-ollama` are thin shell wrappers around externally-installed binaries (`llama.cpp` built by `apps/runtime-llama/setup.sh`, `ollama` from your package manager) — no Python venv of their own.

### 4. Moon task runner

Every `apps/*` project has `format`/`lint`/`test` moon tasks (and `sync`/`update-deps` for the independent ones). Run one project's task with `moon run <project>:<task>` (e.g. `moon run factory:test`), or a task across every project at once with `moon run :<task>` (e.g. `moon run :format`).

### 5. Verification

```bash
uv run pytest apps/runtime/tests -v                                  # unit tests
uv run --env-file .env python apps/runtime/integration_check.py      # GPU integration check (manual, not part of pytest)
uv run --env-file .env python benchmarks/runtime/memory/zero_recapture_swapping/benchmark_zero_recapture_swapping.py
```

---

## 🔌 Connecting Tools & Clients

### 1. Using DSH (DeepSeek Harness)

This repo's chosen coding-agent harness — see [`apps/harness/README.md`](apps/harness/README.md) for the full web UI setup and expert-adapter model aliasing, and [`evals/dsh_agent/`](evals/dsh_agent/) for driving it programmatically via the DSH Python SDK against a fixed task. (An earlier pass used the third-party `opencode` CLI instead; retired as too heavyweight for local-model use — see `evals/dsh_agent/README.md` for that history.)

```bash
bash apps/harness/start_dsh.sh   # web UI on http://localhost:4000
```

### 2. Using `curl` (HTTP REST Calls)

```bash
curl http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "postgresql",
    "messages": [{"role": "user", "content": "Design a PostgreSQL schema for storing vector embeddings."}],
    "max_tokens": 64
  }'

# SSE streaming: add "stream": true to the payload above
```

### 3. Using the Official OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="dummy")

# Instant zero-copy in-place expert swap to the postgresql expert
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

## 🚀 Headline performance numbers

Full detail, methodology, and every retraction in [`docs/MEASURED_FINDINGS.md`](docs/MEASURED_FINDINGS.md).

| Metric | Measured Result |
| :--- | :---: |
| In-Place Weight Folding vs Wrapped PEFT | **33.26 tok/s (+82.1%)** |
| Native MTP Speculative Decoding ($K=6$) | **55.71 tok/s (2.20x net)** |
| Expert Hot-Swap Latency | **18.08 ms (566 GB/s)** |
| Factor Standby Residency | **42.47 MB/expert (200.6x compression)** |
| Batch Scaling Throughput | **611.04 tok/s at $B=64$** |

---

## 📚 Documentation map

| Document | What it's for |
| :--- | :--- |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | The experiment → integration → benchmark/eval/test lifecycle — read this first |
| [`docs/MEASURED_FINDINGS.md`](docs/MEASURED_FINDINGS.md) | The deep empirical results behind this README's headline numbers |
| [`docs/EXPERIMENT_REAUDIT_2026-09.md`](docs/EXPERIMENT_REAUDIT_2026-09.md) | Which `experiments/` findings are real vs. fabricated-as-real, and which shipped defaults need re-verification — **read before trusting an experiment result** |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | What was retired, and the measurement that retired it (companion to the audit above) |
| [`docs/CURRENT.md`](docs/CURRENT.md) | Live ground truth: which trainer, which adapters, which findings currently hold |
| [`docs/NOVELTY.md`](docs/NOVELTY.md) | Honest tiering of what this repo actually contributes vs. known techniques |
| [`docs/GLOSSARY.md`](docs/GLOSSARY.md) | Which of this repo's names are invented, and which already had established names |
| [`docs/CHANGELOG.md`](docs/CHANGELOG.md) | What changed between adapter versions, and what the measurement said afterward |
| [`docs/SYSTEM.md`](docs/SYSTEM.md) | Host/ROCm/toolchain environment snapshot |
| [`docs/AMD_SPECIFFIC.md`](docs/AMD_SPECIFFIC.md) | RDNA3 hardware characteristics mainstream frameworks miss |
| [`TODO.md`](TODO.md) | Active engineering backlog |
| [`docs/HARNESS_COORDINATOR_AND_LORA_SUBAGENTS.md`](docs/HARNESS_COORDINATOR_AND_LORA_SUBAGENTS.md) | The multi-subagent coordinator design (`apps/harness/coordinator/`) |
| [`docs/WHITE_PAPER_THE_CHICKEN_AND_EGG_RUNTIME_PARADOX.md`](docs/WHITE_PAPER_THE_CHICKEN_AND_EGG_RUNTIME_PARADOX.md) | Whitepaper |
| [`docs/WHITE_PAPER_WEIGHT_ADAPTATION_VS_PROMPT_ENGINEERING.md`](docs/WHITE_PAPER_WEIGHT_ADAPTATION_VS_PROMPT_ENGINEERING.md) | Whitepaper (includes a retraction — see its §3) |
| [`docs/W4A16_RDNA3_BEATING_OLLAMA.md`](docs/W4A16_RDNA3_BEATING_OLLAMA.md) | The W4A16 kernel work vs. Ollama baseline |
| the rest of `docs/*.md` | Deep-dive reports on specific subsystems (POET decomposition, Ledoit-Wolf routing, MACD circuit breaker, corpus design, research roadmap, etc.) — self-descriptive filenames |

Every `apps/*`, `experiments/`, `benchmarks/`, `evals/` subdirectory has its own README where one is warranted — this map links the top-level ones; follow the trail from there.
