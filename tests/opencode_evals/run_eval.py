"""Run an autonomous opencode agent against a fresh, isolated broken-monorepo
fixture, and report whether it actually fixed the dependency conflict.

Fully self-contained: generates the fixture on disk under
tests/opencode_evals/runs/<timestamp>/ (gitignored), writes a project-local
opencode.json there (never touches this repo's own pyproject.toml/uv config),
invokes `opencode run` non-interactively, and grades success by actually
running `uv lock` in the fixture afterward — not by trusting the model's
self-report.

    uv run tests/opencode_evals/run_eval.py --model local-llamacpp/qwen3.5-4b
    uv run tests/opencode_evals/run_eval.py --model local-llamacpp/qwen3.5-4b \\
        --prompt "what is the problem with the dependencies?"
"""

import argparse
import json
import re
import subprocess
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

OPENCODE_CONFIG_TEMPLATE = {
    "$schema": "https://opencode.ai/config.json",
    "provider": {
        "local-llamacpp": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Local llama.cpp",
            "options": {"baseURL": "http://127.0.0.1:8080/v1"},
            "models": {},
        }
    },
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        required=True,
        help="opencode model string, e.g. local-llamacpp/qwen3.5-4b",
    )
    p.add_argument("--model-display-name", default=None)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="seconds before the eval is marked FAIL (timeout)",
    )
    return p.parse_args()


def write_opencode_config(repo_dir: Path, model_id: str, display_name: str):
    config = json.loads(json.dumps(OPENCODE_CONFIG_TEMPLATE))  # deep copy
    config["provider"]["local-llamacpp"]["models"][model_id] = {"name": display_name}
    (repo_dir / "opencode.json").write_text(json.dumps(config, indent=2))


def run_opencode(repo_dir: Path, model: str, prompt: str, timeout: int):
    cmd = [
        "opencode",
        "run",
        prompt,
        "--dir",
        str(repo_dir),
        "--model",
        model,
        "--auto",
        "--format",
        "json",
    ]
    start = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired as e:
        result = e
        timed_out = True
    elapsed = time.perf_counter() - start
    return result, elapsed, timed_out


def parse_events(stdout: str):
    """opencode --format json emits newline-delimited JSON events. Extract
    text turns and cumulative token usage."""
    turns = []
    total_tokens = {"input": 0, "output": 0, "reasoning": 0, "total": 0}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        part = event.get("part", {})
        ptype = part.get("type")

        if ptype == "text" and part.get("text"):
            turns.append({"type": "text", "text": part["text"]})
        elif ptype == "tool":
            turns.append(
                {
                    "type": "tool",
                    "tool": part.get("tool"),
                    "state": part.get("state", {}),
                }
            )
        elif ptype == "step-finish":
            tok = part.get("tokens", {})
            for k in ("input", "output", "reasoning"):
                total_tokens[k] += tok.get(k, 0)

    # Each step's "total" is the full context size at that point (including
    # cached-read tokens), not an incremental cost — summing it across steps
    # wildly over-counts. The actual tokens processed across the run is the
    # sum of each step's new input + output.
    total_tokens["total"] = total_tokens["input"] + total_tokens["output"]

    return turns, total_tokens


def classify_fix_method(turns: list) -> str:
    """How did the agent change the dependency constraints: a real `uv`
    command (add/remove/lock/sync) that lets uv own pyproject.toml + the
    lockfile, or a raw file edit to pyproject.toml? Both, or neither, are
    possible outcomes too."""
    used_uv_command = False
    used_raw_edit = False
    for turn in turns:
        if turn["type"] != "tool":
            continue
        state = turn.get("state", {})
        input_ = state.get("input", {})
        if turn["tool"] == "bash" and UV_COMMAND_RE.search(
            str(input_.get("command", ""))
        ):
            used_uv_command = True
        elif turn["tool"] in ("edit", "write", "patch") and "pyproject.toml" in str(
            input_.get("filePath", "")
        ):
            used_raw_edit = True

    if used_uv_command and used_raw_edit:
        return "mixed (both a uv command and a raw pyproject.toml edit)"
    if used_uv_command:
        return "uv command (add/remove/lock/sync)"
    if used_raw_edit:
        return "raw pyproject.toml edit"
    return "neither (no dependency-file change detected)"


def check_ground_truth(repo_dir: Path) -> tuple[bool, str]:
    result = subprocess.run(
        ["uv", "lock"], cwd=repo_dir, capture_output=True, text=True, timeout=60
    )
    return result.returncode == 0, result.stderr


def render_markdown(
    meta: dict, turns: list, ground_truth_ok: bool, ground_truth_err: str
) -> str:
    lines = [
        f"# opencode eval: {meta['model']}",
        "",
        f"- **Status**: {'✅ SUCCESS' if meta['success'] else '❌ FAIL'}",
        f"- **Prompt**: {meta['prompt']!r}",
        f"- **Model**: {meta['model']}",
        f"- **Time taken**: {meta['elapsed_s']:.2f}s",
        f"- **Tokens consumed**: {meta['tokens']['total']} "
        f"(input: {meta['tokens']['input']}, output: {meta['tokens']['output']}, "
        f"reasoning: {meta['tokens']['reasoning']})",
        f"- **Ground truth (`uv lock` in fixture after run)**: "
        f"{'exit 0 (fixed)' if ground_truth_ok else 'still fails'}",
        f"- **Fix method**: {meta['fix_method']}",
        f"- **Timed out**: {meta['timed_out']}",
        "",
        "---",
        "",
        "## Conversation",
        "",
    ]
    for turn in turns:
        if turn["type"] == "text":
            lines.append(turn["text"])
            lines.append("")
        elif turn["type"] == "tool":
            state = turn.get("state", {})
            status = state.get("status", "?")
            input_ = state.get("input", {})
            lines.append(f"**[tool call: `{turn['tool']}`, status: {status}]**")
            if input_:
                lines.append(f"```json\n{json.dumps(input_, indent=2)}\n```")
            output = state.get("output")
            if output:
                snippet = str(output)[:2000]
                lines.append(f"```\n{snippet}\n```")
            lines.append("")

    if not ground_truth_ok and ground_truth_err:
        lines += [
            "---",
            "",
            "## Ground truth `uv lock` error (after run)",
            "",
            "```",
            ground_truth_err[:2000],
            "```",
        ]

    if meta.get("diff"):
        lines += [
            "---",
            "",
            "## What actually changed (`git diff`)",
            "",
            "```diff",
            meta["diff"][:3000],
            "```",
        ]

    return "\n".join(lines)


def main():
    args = parse_args()
    display_name = args.model_display_name or args.model.split("/")[-1]
    model_id = args.model.split("/", 1)[1] if "/" in args.model else args.model

    run_id = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / run_id
    repo_dir = run_dir / "repo"

    print(f"Creating fixture at {repo_dir}")
    create_monorepo_fixture(repo_dir)
    write_opencode_config(repo_dir, model_id, display_name)

    print(f"Running opencode (model={args.model}, timeout={args.timeout}s)...")
    result, elapsed, timed_out = run_opencode(
        repo_dir, args.model, args.prompt, args.timeout
    )

    stdout = "" if timed_out else result.stdout
    stderr = "" if timed_out else result.stderr
    turns, tokens = (
        parse_events(stdout)
        if not timed_out
        else ([], {"input": 0, "output": 0, "reasoning": 0, "total": 0})
    )

    ground_truth_ok, ground_truth_err = (
        (False, "timed out") if timed_out else check_ground_truth(repo_dir)
    )
    fix_method = "n/a (timed out)" if timed_out else classify_fix_method(turns)
    fixture_diff = "" if timed_out else diff_monorepo_fixture(repo_dir)

    meta = {
        "model": args.model,
        "prompt": args.prompt,
        "elapsed_s": elapsed,
        "tokens": tokens,
        "timed_out": timed_out,
        "success": ground_truth_ok and not timed_out,
        "fix_method": fix_method,
        "diff": fixture_diff,
        "run_id": run_id,
    }

    (run_dir / "report.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "raw_stdout.jsonl").write_text(stdout)
    if stderr:
        (run_dir / "raw_stderr.log").write_text(stderr)

    transcript_md = render_markdown(meta, turns, ground_truth_ok, ground_truth_err)
    transcript_path = run_dir / "transcript.md"
    transcript_path.write_text(transcript_md)

    print(f"\n=== RESULT: {'SUCCESS' if meta['success'] else 'FAIL'} ===")
    print(f"Time: {elapsed:.2f}s")
    print(
        f"Tokens: {tokens['total']} (input={tokens['input']}, output={tokens['output']})"
    )
    print(f"Fix method: {fix_method}")
    print(f"Transcript: {transcript_path}")
    print(f"Report: {run_dir / 'report.json'}")


if __name__ == "__main__":
    main()
