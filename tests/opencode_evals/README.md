# opencode autonomous-agent evals

Tests whether a model, driven autonomously by [opencode](https://opencode.ai),
can diagnose and actually fix a real dependency-resolution conflict — not
whether it can *describe* the fix, but whether `uv lock` exits 0 afterward.

## Isolation guarantees

- This is **not** a uv workspace member of the parent `gnn-experiment` or
  `serving` projects, and never will be — `fixtures/monorepo_conflict.py`
  writes its `pyproject.toml` files directly with Python file I/O, never via
  `uv init`, which is what caused `uv` to auto-register a nested project as a
  workspace member the one time we tried it here (see project memory).
- Each eval run generates a fresh, throwaway broken uv workspace under
  `runs/<timestamp>/repo/` (gitignored, never committed). It is unrelated to
  and does not affect this repo's own `pyproject.toml`/`uv.lock` in any way.
- `opencode.json` (the provider config pointing at the local model server) is
  written **inside each generated fixture directory**, not globally — running
  an eval never touches your global opencode config.

## opencode: native Linux CLI execution

`opencode` runs directly as a native Linux binary on CachyOS with direct POSIX shell and filesystem execution. Verify with:

```bash
opencode run "run 'pwd' and tell me the exact output" --dir /tmp --format json
```

## Setup

1. Start the local model server (see `../../serving/`):

   ```bash
   cd ../../serving
   ./llama.cpp/build/bin/llama-server -m models/Qwen3.5-4B-Q8_0.gguf \
       --host 127.0.0.1 --port 8080 -ngl 999
   ```

2. Run an eval:

   ```bash
   uv run --no-project tests/opencode_evals/run_eval.py \
       --model local-llamacpp/models/Qwen3.5-4B-Q8_0.gguf
   ```

   The model id must exactly match what llama-server reports at
   `/v1/models` (check with `curl http://127.0.0.1:8080/v1/models`).
   `--no-project` is used because this script only needs the stdlib — no
   need to involve gnn-experiment's own dependency set at all.

## Isolation / reset without Docker

Every eval run generates a **brand-new fixture in a fresh timestamped
directory**, so there's never stale state to reset between eval runs. For
poking at a fixture by hand between runs, `fixtures/monorepo_conflict.py`
also `git init`s each fixture and commits the broken state as the initial
commit:

- `reset(base_dir)` — `git checkout -- . && git clean -fdx`, snaps back to
  the broken state instantly, no Docker, no regenerating from scratch.
- `diff(base_dir)` — `git diff HEAD`, shows exactly what an agent changed.
  `run_eval.py` calls this automatically and includes it in the transcript.

## What it checks

- **Ground truth**: after opencode finishes (or times out), the script runs
  `uv lock` in the fixture directory itself and checks the exit code — this
  is the actual pass/fail signal, independent of anything the model claims.
- **Tokens consumed** and **wall-clock time**, parsed from opencode's
  `--format json` event stream.
- **Fix method**: whether the agent fixed the conflict via a real `uv`
  command (`uv add`/`remove`/`lock`/`sync` — the elegant path, since uv then
  owns rewriting `pyproject.toml` and relocking itself) vs. a raw
  `edit`/`write` to `pyproject.toml` text, vs. neither. This conflict is
  solvable in one command, verified directly:
  `uv add "pydantic>=2.0.0" --package legacy-connector` from the workspace
  root — no manual file or lockfile editing needed at all.
- Writes `runs/<timestamp>/transcript.md` (full conversation + summary +
  `git diff` of what actually changed) and `runs/<timestamp>/report.json`
  (machine-readable summary).

## Files

- `fixtures/monorepo_conflict.py` — generates the broken workspace
  (`apps/web-api` wants `pydantic>=2.0`, `libs/legacy-connector` wants
  `pydantic<2.0`)
- `run_eval.py` — orchestrates a single eval run end to end
