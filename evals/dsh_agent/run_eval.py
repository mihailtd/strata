"""Run an autonomous DSH agent against a fresh, isolated broken-monorepo
fixture, and report whether it actually fixed the dependency conflict.

Ported from the former tests/opencode_evals/run_eval.py, which drove the
third-party `opencode` CLI. That was this repo's first pass at agentic-
capability evaluation; opencode was retired as too bloated for local-model
use, and DSH (apps/harness/) is now the chosen harness. See README.md in
this directory for the full rationale and the mechanical differences from
the opencode version.

Fully self-contained: generates the fixture on disk under
evals/dsh_agent/runs/<timestamp>/repo/ (gitignored), gives each run its own
isolated DSH home under evals/dsh_agent/runs/<timestamp>/dsh_home/ (so
sessions/plugins/config never bleed between runs or pollute ~/.dsh), drives
the DeepSeek Harness Python SDK against a local OpenAI-compatible model
server, and grades success by actually running `uv lock` in the fixture
afterward -- not by trusting the model's self-report.

    uv run evals/dsh_agent/run_eval.py --model qwen3.5-4b
    uv run evals/dsh_agent/run_eval.py --model qwen3.5-4b \\
        --prompt "what is the problem with the dependencies?"

NOT YET LIVE-VERIFIED end-to-end in this environment (no model server was
running during this port) -- run the smoke check in this directory's
README before trusting a real eval run.
"""

import argparse
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fixtures.monorepo_conflict import create as create_monorepo_fixture
from fixtures.monorepo_conflict import diff as diff_monorepo_fixture

UV_COMMAND_RE = re.compile(r"\buv\s+(add|remove|lock|sync)\b")

HERE = Path(__file__).parent
RUNS_DIR = HERE / "runs"

DEFAULT_PROMPT = "explain what is the problem with the dependencies, and fix it."
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"

# str_replace_editor is opt-in for sdk-minimal (bash-only by default) -- add
# it so the agent has the same tool-choice dichotomy the opencode eval
# measured (a uv command via bash vs. a raw text edit via the editor tool).
EDITOR_PATCH = HERE / "editor.patch.yml"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="model id, e.g. qwen3.5-4b")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="local OpenAI-compatible server")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="seconds before the eval is marked FAIL (timeout)",
    )
    return p.parse_args()


def run_dsh_agent(repo_dir: Path, dsh_home: Path, model: str, base_url: str, prompt: str, session_id: str, timeout: int):
    """Drive the DSH Python SDK against the fixture. Returns (final_response, elapsed, timed_out, error)."""
    import os

    from deepseek_harness import DeepSeekHarness

    os.environ.setdefault("DEEPSEEK_API_KEY", "sk-local-dev-key")  # local server does not check it
    os.environ["DEEPSEEK_BASE_URL"] = base_url

    start = time.perf_counter()
    final_response = ""
    error = None
    timed_out = False
    try:
        with DeepSeekHarness(
            provider="deepseek-official",
            model=model,
            cwd=str(repo_dir),
            dsh_home=str(dsh_home),
            profile="sdk-minimal",
            patches=(str(EDITOR_PATCH),) if EDITOR_PATCH.exists() else (),
        ) as harness:
            result = harness.run(prompt, session_id=session_id)
            final_response = getattr(result, "final_response", "") or str(result)
    except Exception as exc:  # noqa: BLE001 -- eval harness: any failure is a FAIL, not a crash
        error = str(exc)
    elapsed = time.perf_counter() - start
    return final_response, elapsed, timed_out, error


def load_session_events(dsh_home: Path) -> list[dict]:
    """Read this run's session JSONL (uncompressed, unlike the web/CLI path's
    zstd-compressed ~/.dsh/sessions/ format). Session file layout under a
    fresh SDK dsh_home is not pinned by the public docs beyond "uncompressed
    JSONL under sessions/" -- pick the most recently written *.jsonl file
    rather than hardcoding an exact nested path, and confirm this against a
    real run (see README's smoke-check step) before trusting it blindly."""
    sessions_dir = dsh_home / "sessions"
    if not sessions_dir.exists():
        return []
    jsonl_files = sorted(sessions_dir.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not jsonl_files:
        return []
    events = []
    for line in jsonl_files[-1].read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def classify_fix_method(events: list[dict]) -> str:
    """How did the agent change the dependency constraints: a real `uv`
    command (add/remove/lock/sync) via bash, or a raw file edit to
    pyproject.toml via str_replace_editor? Both, or neither, are possible.

    Ported from opencode's part.tool/state.input schema to DSH's
    tool/call event schema (data.name, data.arguments) -- confirm the exact
    field names against a real session file during the smoke check, this
    was derived from the CLI/web session format, not yet cross-checked
    against the SDK's own uncompressed JSONL shape.
    """
    used_uv_command = False
    used_raw_edit = False
    for event in events:
        if event.get("type") != "tool/call":
            continue
        data = event.get("data", {})
        name = data.get("name")
        try:
            arguments = json.loads(data.get("arguments", "{}"))
        except json.JSONDecodeError:
            arguments = {}

        if name == "bash" and UV_COMMAND_RE.search(str(arguments.get("command", ""))):
            used_uv_command = True
        elif name == "str_replace_editor" and "pyproject.toml" in str(arguments.get("path", "")):
            used_raw_edit = True

    if used_uv_command and used_raw_edit:
        return "mixed (both a uv command and a raw pyproject.toml edit)"
    if used_uv_command:
        return "uv command (add/remove/lock/sync)"
    if used_raw_edit:
        return "raw pyproject.toml edit"
    return "neither (no dependency-file change detected)"


def check_ground_truth(repo_dir: Path) -> tuple[bool, str]:
    import subprocess

    result = subprocess.run(["uv", "lock"], cwd=repo_dir, capture_output=True, text=True, timeout=60)
    return result.returncode == 0, result.stderr


def render_markdown(meta: dict, events: list[dict], ground_truth_ok: bool, ground_truth_err: str) -> str:
    lines = [
        f"# DSH agent eval: {meta['model']}",
        "",
        f"- **Status**: {'✅ SUCCESS' if meta['success'] else '❌ FAIL'}",
        f"- **Prompt**: {meta['prompt']!r}",
        f"- **Model**: {meta['model']}",
        f"- **Time taken**: {meta['elapsed_s']:.2f}s",
        f"- **Ground truth (`uv lock` in fixture after run)**: "
        f"{'exit 0 (fixed)' if ground_truth_ok else 'still fails'}",
        f"- **Fix method**: {meta['fix_method']}",
        f"- **Error**: {meta['error']}",
        "",
        "---",
        "",
        "## Final response",
        "",
        meta["final_response"],
        "",
        "---",
        "",
        "## Tool calls (from session log)",
        "",
    ]
    for event in events:
        if event.get("type") == "tool/call":
            data = event.get("data", {})
            lines.append(f"**[tool call: `{data.get('name')}`]**")
            lines.append(f"```json\n{data.get('arguments', '{}')}\n```")
            lines.append("")

    if not ground_truth_ok and ground_truth_err:
        lines += ["---", "", "## Ground truth `uv lock` error (after run)", "", "```", ground_truth_err[:2000], "```"]

    if meta.get("diff"):
        lines += ["---", "", "## What actually changed (`git diff`)", "", "```diff", meta["diff"][:3000], "```"]

    return "\n".join(lines)


def main():
    args = parse_args()

    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_id
    repo_dir = run_dir / "repo"
    dsh_home = run_dir / "dsh_home"

    print(f"Creating fixture at {repo_dir}")
    create_monorepo_fixture(repo_dir)

    print(f"Running DSH agent (model={args.model}, timeout={args.timeout}s)...")
    final_response, elapsed, timed_out, error = run_dsh_agent(
        repo_dir, dsh_home, args.model, args.base_url, args.prompt, session_id=f"eval-{run_id}", timeout=args.timeout
    )

    events = [] if error else load_session_events(dsh_home)
    ground_truth_ok, ground_truth_err = (False, error or "error before ground-truth check") if error else check_ground_truth(repo_dir)
    fix_method = "n/a (error)" if error else classify_fix_method(events)
    fixture_diff = "" if error else diff_monorepo_fixture(repo_dir)

    meta = {
        "model": args.model,
        "prompt": args.prompt,
        "elapsed_s": elapsed,
        "timed_out": timed_out,
        "error": error,
        "success": ground_truth_ok and error is None,
        "fix_method": fix_method,
        "final_response": final_response,
        "diff": fixture_diff,
        "run_id": run_id,
    }

    (run_dir / "report.json").write_text(json.dumps(meta, indent=2))

    transcript_md = render_markdown(meta, events, ground_truth_ok, ground_truth_err)
    transcript_path = run_dir / "transcript.md"
    transcript_path.write_text(transcript_md)

    print(f"\n=== RESULT: {'SUCCESS' if meta['success'] else 'FAIL'} ===")
    print(f"Time: {elapsed:.2f}s")
    print(f"Fix method: {fix_method}")
    print(f"Transcript: {transcript_path}")
    print(f"Report: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
