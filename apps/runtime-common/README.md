# `apps/runtime-common` — shared canon and GPU-exclusivity guard

The one dependency every runtime engine can safely share, because it deliberately has **no torch dependency** (see its `pyproject.toml`'s comment for why — every other runtime pins a different, sometimes incompatible, torch/ROCm build).

Two modules:
- **`canon.py`** — the single source of truth for `CANON.ADAPTER_VERSION`, `CANON.MAX_NEW_TOKENS`, `REPO_ROOT` (resolved by walking up to `.git`, never by `__file__` arithmetic — that broke 31 scripts once during a reorg), and `adapter_path()`. Import these rather than hardcoding a version or a decode budget; `audit/check_canon.py` fails the build if you don't.
- **`gpu_preflight.py`** — the GPU-exclusivity guard every engine calls before allocating: scans VRAM and running processes, halts loudly on a conflict rather than let two runtimes silently fight over one GPU.

## Quick start

```bash
cd apps/runtime-common
uv sync
uv run pytest tests -v
```

Consumed as a path dependency (`{ path = "../runtime-common", editable = true }`) by the root project and by `apps/runtime-triton`, `apps/runtime-ipwf`, `apps/runtime-ollama`, `apps/runtime-llama`, and `apps/runtime-vllm` — see [`apps/RUNTIME.md`](../RUNTIME.md).
