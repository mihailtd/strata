# `apps/runtime` — the original engine and shared module library

See [`apps/RUNTIME.md`](../RUNTIME.md) for how this fits alongside the other six runtimes. This README covers this project specifically.

This is the root Python project (`name = "runtime"` in the root `pyproject.toml`, `module-root = "apps"`, no independent `pyproject.toml`/venv of its own — it rides the shared root venv). It predates the Moon monorepo restructuring and plays two roles at once:

1. **A FastAPI server** (`server.py`, port 8000) — in-place weight folding (`W_live = W0 + s·U@V`), CUDA/HIP Graph decode, domain-expert LoRA/MoA routing, MTP speculative decoding, and the 27B native engine path (`native_27b_engine.py`).
2. **A shared module library** — `novel_peft.py`, `mtp_draft.py`, `bucketed_speculative.py`, `w4a16_loader.py`, `training_db.py`, `micro_probe/`, `eval/eval_suite.py`, `utils/logger.py`, `dynamic_team_router.py`, `cut_set_router.py`, and more — imported directly (in-process, not as a package) by `apps/runtime-ipwf` and, via a `sys.path` bootstrap, by `apps/factory`.

**Important nuance** (see `docs/EXPERIMENT_REAUDIT_2026-09.md`): despite the "shared module library" framing, the 27B serving path never actually reaches most of this library — `server.py`'s `load_inference_engine()` returns immediately on the `is_27b` branch, before the code that sets up `novel_peft`, `cuda_graph`, `range_statistic_gate`, `dynamic_team_router`, etc. Those modules are really a **4B/9B-only** library; 27B has its own largely-independent implementation of the same concepts.

## Quick start

```bash
# From repo root (shared root venv, uv sync already done for the whole repo)
uv run --env-file .env python apps/runtime/run_server.py --host 127.0.0.1 --port 8000
```

- **Health check**: `http://127.0.0.1:8000/health`
- **OpenAI chat**: `http://127.0.0.1:8000/v1/chat/completions`
- **Swagger docs**: `http://127.0.0.1:8000/docs`

See the root [`README.md`](../../README.md) for the full quickstart (curl examples, OpenAI SDK usage, pre-loaded experts) and [`docs/MEASURED_FINDINGS.md`](../../docs/MEASURED_FINDINGS.md) for the performance numbers this engine has actually measured.

## Tests

```bash
uv run pytest apps/runtime/tests -v          # unit tests (31+ files)
uv run --env-file .env python apps/runtime/integration_check.py   # GPU integration check, not part of the pytest suite
```

## Operational scripts

`run_server.py` (server launcher), `ab.py` (A/B compare base/system-prompt/every expert on one prompt), `integration_check.py` (manual GPU integration test — see its own docstring) all live alongside this README, moved here from the old root `scripts/serve/` in an earlier reorg because they're specific to this runtime, not repo-wide.
