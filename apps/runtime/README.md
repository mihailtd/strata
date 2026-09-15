# `apps/runtime` — LEGACY: the original dual-purpose engine, now a shared library

> **LEGACY.** Do not start `server.py` for new work. Its two serving paths
> have self-sufficient successors: [`runtime-triton`](../runtime-triton/)
> (27B W4A16, quantized) and [`runtime-ipwf`](../runtime-ipwf/) (3B/9B,
> unquantized). This project is kept alive only because `apps/factory` and
> ~20 not-yet-migrated benchmarks/evals/tests still import symbols from it
> directly as a library. See [`apps/RUNTIME.md`](../RUNTIME.md) for how this
> fits alongside the other runtimes.

This is the root Python project (`name = "runtime"` in the root `pyproject.toml`, `module-root = "apps"`, no independent `pyproject.toml`/venv of its own — it rides the shared root venv). It predates the Moon monorepo restructuring and originally played two roles at once:

1. ~~A FastAPI server (`server.py`, port 8000)~~ — **superseded**: the 27B native engine path now lives independently in `runtime-triton`, and the 4B/9B weight-folding path now lives independently in `runtime-ipwf`. `server.py` itself is untouched and still runs, but it should not be started for new work — its serving role is fully covered by the two successors.
2. **A shared module library** — `novel_peft.py`, `mtp_draft.py`, `bucketed_speculative.py`, `w4a16_loader.py`, `training_db.py`, `micro_probe/`, `eval/eval_suite.py`, `utils/logger.py`, `dynamic_team_router.py`, `cut_set_router.py`, and more — imported directly (in-process, not as a package), via a `sys.path` bootstrap, by `apps/factory`, and by symbol from `server.py`/these modules by benchmarks, evals, and tests that predate the runtime split and haven't been migrated onto `runtime-triton`/`runtime-ipwf` yet. `runtime-ipwf` itself no longer imports anything from here — it vendors its own copy of everything it needs.

**Important nuance** (see `docs/EXPERIMENT_REAUDIT_2026-09.md`): despite the "shared module library" framing, the 27B serving path never actually reached most of this library even when `server.py` was the only 27B implementation — `load_inference_engine()`'s `is_27b` branch returned immediately, before the code that sets up `novel_peft`, `cuda_graph`, `range_statistic_gate`, `dynamic_team_router`, etc. Those modules were really a **4B/9B-only** library; 27B always had its own largely-independent implementation of the same concepts, which is exactly what `runtime-triton` now vendors on its own.

## Quick start

**For new work, don't start this.** Start
[`apps/runtime-triton/run_server.sh`](../runtime-triton/run_server.sh) (27B,
quantized) or [`apps/runtime-ipwf/run_server.sh`](../runtime-ipwf/run_server.sh)
(3B/9B, unquantized) instead.

The legacy server still runs, for anything not yet migrated off it:

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
