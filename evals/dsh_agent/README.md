# DSH agentic-capability eval

Can an autonomous coding agent, using this repo's own local model server as its backend, actually diagnose and fix a real dependency conflict — not just describe it?

## Why this exists, and why it changed harnesses

This is this repo's evaluation of **agentic tool-use capability** — a different axis from `evals/judge/` (multi-turn conversation quality), `evals/domain_rubric/` (single-shot domain rubric scoring), or `evals/swe_bench/` (fixed coding tasks graded by pytest). Here the model drives an actual coding agent loop: read files, run shell commands, edit code, and the eval only counts a real, independently-checked fix (`uv lock` exits 0 afterward), not the model's own claim that it fixed something.

**This used to run on [`opencode`](https://opencode.ai)**, a third-party autonomous coding CLI. That pass concluded opencode was too heavyweight for local-model use in this repo's workflow, and the harness was switched to **DSH** (DeepSeek Harness, `apps/harness/` in this repo) — the harness this repo has otherwise standardized on. This eval is the port of that same fixture and grading logic onto DSH instead, with every opencode trace removed from the repo (root `opencode.json`, the "Using OpenCode CLI" section of the root `README.md`, and this directory's former location at `tests/opencode_evals/`).

## Mechanically, what changed

DSH ships an official **Python SDK** (`deepseek-harness-sdk`, package `deepseek_harness`), which is a cleaner fit than shelling out to a CLI the way the opencode version did:

| | opencode (before) | DSH SDK (now) |
| :--- | :--- | :--- |
| Invocation | `subprocess.run(["opencode","run",prompt,"--dir",...,"--auto","--format","json"])` | `DeepSeekHarness(cwd=..., dsh_home=..., profile="sdk-minimal").run(prompt, session_id=...)` |
| Target directory | `--dir` flag | `cwd` constructor param |
| Auto-approval | `--auto` flag | `sdk-minimal` profile pins `danger-full-access` by default |
| Transcript | newline-delimited JSON on stdout | uncompressed JSONL under `<dsh_home>/sessions/` |
| Isolation | project-local `opencode.json` | fresh `dsh_home` per run — its own sessions/plugins/config, no `~/.dsh` pollution |
| Tool surface | editor + bash by default | bash only by default; `editor.patch.yml` in this directory opts into `str_replace_editor` for parity |

The fixture (`fixtures/monorepo_conflict.py`: a `uv` workspace with a real PubGrub conflict — one member wants `pydantic>=2.0`, another wants `pydantic<2.0`) and the grading logic (run `uv lock` in the fixture afterward, exit 0 or it didn't actually fix anything) are unchanged — those were never opencode-specific.

## Status

**Not yet live-verified end-to-end** — this port was written without a running model server in the environment it was authored in. Two things need confirming against a real run before trusting results from it:

1. The exact session JSONL layout under `<dsh_home>/sessions/` (`load_session_events()` in `run_eval.py` picks the most recently written `*.jsonl` file rather than a hardcoded path, since the public SDK docs don't pin the exact nesting).
2. The exact field names in a `tool/call` event (`classify_fix_method()` assumes `data.name`/`data.arguments`, carried over from the CLI/web session format — not yet cross-checked against the SDK's own output).

**Before trusting a real eval run**, do a throwaway smoke check first:
```bash
python python/sdk/examples/minimal.py \
  --workspace /tmp/dsh-smoke-workspace \
  --dsh-home /tmp/dsh-smoke-home \
  --session-id smoke-001 \
  "list the files here"
```
(from a `deepseek-harness` checkout with the SDK installed, `DEEPSEEK_BASE_URL` pointed at this repo's own runtime server) — confirm it completes and produces a session file, then inspect that file's actual JSON shape and adjust `load_session_events()`/`classify_fix_method()` if it differs from what's assumed above.

## Running it

```bash
uv run evals/dsh_agent/run_eval.py --model qwen3.5-4b
uv run evals/dsh_agent/run_eval.py --model qwen3.5-4b --prompt "what is the problem with the dependencies?"
```

Requires the repo's own runtime server running (default `http://127.0.0.1:8000/v1`, override with `--base-url`) and `deepseek-harness-sdk` installed (root `pyproject.toml` dependency). Output goes to `runs/<UTC-timestamp>/` (gitignored): `report.json`, `transcript.md`, the generated fixture under `repo/`, and the isolated `dsh_home/`.
