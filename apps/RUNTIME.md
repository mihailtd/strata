# Runtimes in `apps/`

This repo runs the same family of local LLMs through several different serving
stacks, on one AMD RX 7900 XTX (24GB, ROCm/HIP, `gfx1100`). Only **one runtime
process may hold the GPU at a time** — `apps/runtime-common`'s `gpu_preflight`
guard enforces this. Pick the one your task needs, start it, stop it, then
switch.

| Project | Role | Port | Own venv? |
| :--- | :--- | :--- | :--- |
| [`runtime`](runtime/) | **LEGACY** — original dual-purpose 27B + 3B/9B server, superseded as a server; still a shared module library some benchmarks/evals import | 8000 | No — root `pyproject.toml` |
| [`runtime-common`](runtime-common/) | Shared canon + GPU-exclusivity library, no server of its own | n/a | Yes (no torch dep) |
| [`runtime-triton`](runtime-triton/) | Self-sufficient 27B W4A16 engine (quantized) — successor to `runtime`'s 27B path | 8000 | Yes |
| [`runtime-ipwf`](runtime-ipwf/) | Self-sufficient in-place weight-folding engine for 3B/9B models (unquantized) — successor to `runtime`'s 4B/9B path | 8002 | Yes |
| [`runtime-llama`](runtime-llama/) | Upstream `llama.cpp` baseline (HIP build) | 8001 | No — shell wrapper only |
| [`runtime-ollama`](runtime-ollama/) | Upstream `ollama` baseline | 11434 | No — shell wrapper only |
| [`runtime-vllm`](runtime-vllm/) | Upstream `vllm` baseline (ROCm) | 8004 | Yes |
| [`runtime-next`](runtime-next/) | **Strata** — High-throughput native Rust/HIP serving engine for Qwen 3.5 | 8003 | n/a (Cargo, not uv) |

Non-runtime projects that also live in `apps/`: [`factory`](factory/) (the
training pipeline — builds the LoRA adapters every engine above loads),
[`harness`](harness/) (the DSH agent harness that drives requests into
whichever runtime is running), and [`dashboard`](dashboard/) (the Next.js
telemetry/chat UI).

---

## The engines we built (this repo's own code)

### `runtime` — LEGACY: the original dual-purpose server, now a shared library only
This is the root Python project (`name = "runtime"` in the root
`pyproject.toml`, `module-root = "apps"`) — it predates the Moon
restructuring. Both engines it used to be the only implementation of now have
self-sufficient successors: `runtime-triton` (27B W4A16, quantized) and
`runtime-ipwf` (3B/9B, unquantized). **Do not start `apps/runtime/server.py`
for new work** — start one of those two instead. `apps/runtime/server.py`
carries a deprecation banner to this effect.

It still plays one real role:

**A shared module library** (`novel_peft.py`, `mtp_draft.py`,
`w4a16_loader.py`, `training_db.py`, `micro_probe/`, `eval/eval_suite.py`,
`utils/logger.py`, `datagen/`, …) that `apps/factory` still imports directly
(via a `sys.path` bootstrap, since it can't take a package dependency without
dragging in the root project's `torch>=2.13.0`/`vllm` graph), and that ~20
not-yet-migrated benchmarks/evals/tests still import symbols from directly
(request/response models, `ADAPTER_MAP_27B`, `get_27b_tokenizer`,
`get_native_triton_27b_engine`, etc.) — migrating those call sites onto
`runtime-triton`/`runtime-ipwf` is tracked as separate follow-up work, not
done as part of the runtime split. `runtime-ipwf` no longer imports from this
library at all: it vendors its own copy of everything it needs (see below).

### `runtime-triton` — the 27B engine's self-sufficient replacement
Vendors its own copy of `native_27b_engine.py` and every module it needs
(`w4a16_loader`, `triton_w4a16`, `gguf_unpacker`, plus `syntax_drafter` and
`adapter_stacker`) into its own `pyproject.toml`/`.venv`. Depends on
`runtime-common` only (`canon` + `gpu_preflight`) — explicitly does **not**
import from `apps/runtime`. Same port (8000) as `runtime`'s 27B path because
they're two implementations of the same job, never run together.

### `runtime-ipwf` — 3B/9B in-place weight folding
Serves unquantized Qwen3.5-4B/9B with the same in-place-mutation +
HIP-Graph approach as `runtime`'s FlashNorm path, plus a Riemannian
multi-expert team router and NOTEARS causal scheduler for co-activating
adapters. Self-sufficient as of the runtime split: its own
`pyproject.toml`/`.venv`, vendoring its own copy of the weight-folding/
routing/speculation library (`novel_peft`, `cuda_graph`, `fused_norm`,
`range_statistic_gate`, `dynamic_team_router`, `notears_causal_scheduler`,
`riemannian_covariance`, `bucketed_speculative`, `mtp_draft`,
`state_ring_buffer`, `macd_speculation_circuit_breaker`, `tool_trace`).
Depends on `runtime-common` only, same as `runtime-triton` — does **not**
import from `apps/runtime`.

### `runtime-next` (`strata`) — Native Rust/HIP Serving Engine
A native Rust engine with hand-written HIP device kernels for Qwen 3.5's hybrid
Gated DeltaNet linear attention + full attention architecture on AMD hardware
(`gfx1100`, `gfx90a`, `gfx942`). Zero heap allocation per token, byte-identical
Jinja chat templating, tool calling, and dynamic In-Place Weight Folding (IPWF).
See [`apps/runtime-next/README.md`](runtime-next/README.md) for architecture and build details.

### `runtime-common`
Not a server. Just `canon.py` (the single source of truth for
`ADAPTER_VERSION` and other canonical constants) and `gpu_preflight.py` (the
GPU-exclusivity guard every engine calls before allocating). Deliberately has
no torch dependency so every engine above can depend on it without pulling
its own torch/ROCm pin into conflict.

---

## The baselines we compare against (upstream, not ours)

Every engine above gets benchmarked against these — same hardware, same
models where possible, to get an honest comparison number. Each is isolated
in its own project specifically because its dependency pins would otherwise
conflict with everything else in this repo (three-plus incompatible
torch/ROCm pins already coexist here on purpose).

- **`runtime-llama`** — upstream `llama.cpp`, built from source with
  `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100`. Not tracked in git at all —
  `setup.sh` clones and builds it fresh on demand; it's third-party code, not
  ours to vendor.
- **`runtime-ollama`** — upstream `ollama` binary (`/usr/bin/ollama`), a pure
  shell wrapper around it. The simplest baseline: no build step, no repo of
  its own to manage.
- **`runtime-vllm`** — upstream `vllm`, in its own venv pinned to
  `torch==2.12.0+rocm7.14.0` + the `vllm-rdna` wheel index. Moved here from
  the old root-level `serving/` folder (pre-Moon layout), unchanged except
  for name and location.

---

## Single-GPU discipline

Never start two of these at once. The standard loop for any A/B comparison:

```bash
# 1. start engine A, benchmark it, stop it
cd apps/runtime-triton && ./run_server.sh &
uv run python benchmarks/<your_benchmark>.py --endpoint http://127.0.0.1:8000/v1 ...
pkill -f runtime-triton  # or the engine's own stop path

# 2. start engine B, benchmark it, stop it
cd apps/runtime-ollama && ./run_server.sh
uv run python benchmarks/<your_benchmark>.py --endpoint http://127.0.0.1:11434/v1 ...
./run_server.sh stop

# 3. compare the two scorecards
```

`gpu_preflight.py` (in `runtime-common`) is a safety net for this rule, not a
substitute for following it — it scans VRAM and running processes and halts
loudly rather than let a collision silently corrupt a benchmark or crash the
driver.
