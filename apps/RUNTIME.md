# Runtimes in `apps/`

This repo runs the same family of local LLMs through several different serving
stacks, on one AMD RX 7900 XTX (24GB, ROCm/HIP, `gfx1100`). Only **one runtime
process may hold the GPU at a time** — `apps/runtime-common`'s `gpu_preflight`
guard enforces this. Pick the one your task needs, start it, stop it, then
switch.

| Project | Role | Port | Own venv? |
| :--- | :--- | :--- | :--- |
| [`runtime`](runtime/) | Original/current 27B + 3B/9B engine and shared module library | 8000 | No — root `pyproject.toml` |
| [`runtime-common`](runtime-common/) | Shared canon + GPU-exclusivity library, no server of its own | n/a | Yes (no torch dep) |
| [`runtime-triton`](runtime-triton/) | Self-sufficient rewrite of the 27B W4A16 engine | 8000 | Yes |
| [`runtime-ipwf`](runtime-ipwf/) | In-place weight-folding engine for 3B/9B models | 8002 | No — shares root venv (Stage 3 pending) |
| [`runtime-llama`](runtime-llama/) | Upstream `llama.cpp` baseline (HIP build) | 8001 | No — shell wrapper only |
| [`runtime-ollama`](runtime-ollama/) | Upstream `ollama` baseline | 11434 | No — shell wrapper only |
| [`runtime-vllm`](runtime-vllm/) | Upstream `vllm` baseline (ROCm) | 8004 | Yes |
| [`runtime-next`](runtime-next/) | Experimental from-scratch Rust/HIP engine (skeleton only) | 8003 | n/a (Cargo, not uv) |

Non-runtime projects that also live in `apps/`: [`factory`](factory/) (the
training pipeline — builds the LoRA adapters every engine above loads),
[`harness`](harness/) (the DSH agent harness that drives requests into
whichever runtime is running), and [`dashboard`](dashboard/) (the Next.js
telemetry/chat UI).

---

## The engines we built (this repo's own code)

### `runtime` — the original engine and shared library
This is the root Python project (`name = "runtime"` in the root
`pyproject.toml`, `module-root = "apps"`) — it predates the Moon
restructuring and hasn't been split into its own independent project yet
(tracked as "Stage 3/4" work). It plays two roles at once:

1. **A FastAPI server** (`apps/runtime/server.py`, port 8000) implementing
   in-place weight folding (`W_live = W0 + s·U@V`), CUDA/HIP Graph decode, and
   the domain-expert LoRA/MoA machinery — originally the only engine in the
   repo.
2. **A shared module library** (`novel_peft.py`, `mtp_draft.py`,
   `w4a16_loader.py`, `training_db.py`, `micro_probe/`, `eval/eval_suite.py`,
   `utils/logger.py`, `datagen/`, …) that `runtime-ipwf` and `apps/factory`
   still import directly (via `sys.path` bootstrap for `factory`, since it
   can't take a package dependency without dragging in the root project's
   `torch>=2.13.0`/`vllm` graph).

As the other engines below absorb its serving responsibilities into
self-sufficient projects, `apps/runtime`'s job should shrink toward just (2).

### `runtime-triton` — the 27B engine's self-sufficient replacement
Vendors its own copy of `native_27b_engine.py` and every module it needs
(`w4a16_loader`, `triton_w4a16`, `gguf_unpacker`) into its own
`pyproject.toml`/`.venv`. Depends on `runtime-common` only (`canon` +
`gpu_preflight`) — explicitly does **not** import from `apps/runtime`. Same
port (8000) as `runtime`'s 27B path because they're two implementations of
the same job, never run together.

### `runtime-ipwf` — 3B/9B in-place weight folding
Serves unquantized Qwen3.5-4B/9B with the same in-place-mutation +
HIP-Graph approach as `runtime`'s FlashNorm path, plus a Riemannian
multi-expert team router and NOTEARS causal scheduler for co-activating
adapters. Still on the shared root venv (`dependsOn: [runtime, runtime-common]`)
— hasn't been given its own manifest yet.

### `runtime-next` — experimental Rust rewrite
A Cargo project, currently just a startup print statement and a handful of
smoke tests (including one asserting its reserved port, 8003, to avoid
collisions with everything else in this table). Exists to explore a
zero-allocation, monolithic-HIP-Graph pipeline in Rust instead of Python —
not yet a working engine.

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
