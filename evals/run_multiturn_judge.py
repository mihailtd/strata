#!/usr/bin/env python3
"""Multi-Turn Conversational E2E Judge Runner.

Executes sequential multi-turn prompts in a shared context window against the
inference server, displaying full un-truncated outputs and quality metrics
so an engineer or AI agent can judge the generation quality directly.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import sys

from evals.judge_harness import E2EJudgeHarness, TurnResult
from evals.prompts import DASHBOARD_MULTITURN_SEQUENCE, EvalPrompt


def run_multiturn_eval(
    base_url: str = "http://127.0.0.1:8000",
    prompts: list[EvalPrompt] | None = None,
    expert: str = "dynamic",
    max_tokens: int = 400,
    thinking_effort: str = "off",
    output_dir: Path | None = None,
) -> tuple[list[TurnResult], str]:
    harness = E2EJudgeHarness(base_url=base_url)

    if not harness.check_server_health():
        print(f"❌ Error: Inference server is not reachable at {base_url}.")
        print("Please start the server first (e.g. via 'uv run src/runtime/server.py')")
        sys.exit(1)

    eval_prompts = prompts or DASHBOARD_MULTITURN_SEQUENCE
    print("=" * 80)
    print(f"🚀 STARTING MULTI-TURN E2E JUDGE EVALUATION ({len(eval_prompts)} turns)")
    print(f"Target: {base_url} | Expert Mode: {expert} | Thinking: {thinking_effort}")
    print("=" * 80)

    conversation_history: list[dict[str, str]] = []
    results: list[TurnResult] = []

    for turn_idx, p in enumerate(eval_prompts, start=1):
        print(f"\n▶️  Executing Turn {turn_idx}/{len(eval_prompts)}: [{p.domain.upper()}] - '{p.prompt[:60]}...'")
        
        # Add user prompt to shared context
        conversation_history.append({"role": "user", "content": p.prompt})

        result = harness.execute_turn_live(
            conversation_history=conversation_history,
            turn_idx=turn_idx,
            prompt_id=p.id,
            domain=p.domain,
            expert=expert,
            max_tokens=max_tokens,
            thinking_effort=thinking_effort,
        )
        results.append(result)

        # Print output preview to terminal immediately
        print(f"   ⚡ Speed: {result.tokens_per_second} tok/s | TTFT: {result.ttft_ms}ms | Tokens: {result.total_tokens}")
        print(f"   🏷️  Active Team: {result.active_team or ['base']}")
        if result.python_syntax_valid is not None:
            syntax_str = "✅ Valid Python AST" if result.python_syntax_valid else f"❌ {result.python_syntax_error}"
            print(f"   🔍 Syntax: {syntax_str}")
        print(f"   💬 Output:\n{result.assistant_response.strip()}\n")

        # Append assistant response to conversational context for next turn
        conversation_history.append({"role": "assistant", "content": result.assistant_response})

    # Build Markdown Report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_lines = [
        f"# Multi-Turn E2E Judge Evaluation Report ({timestamp})",
        "",
        f"- **Base URL**: `{base_url}`",
        f"- **Expert Strategy**: `{expert}`",
        f"- **Thinking Effort**: `{thinking_effort}`",
        f"- **Total Turns**: `{len(results)}`",
        "",
        "## 📊 Summary Matrix",
        "",
        harness.format_summary_table(results),
        "",
        "## 📝 Turn-by-Turn Outputs (For Human/Agent Judge Inspection)",
        "",
    ]

    for r in results:
        report_lines.append(harness.format_turn_markdown(r))

    full_report = "\n".join(report_lines)

    # Save artifact
    out_dir = output_dir or (Path(__file__).parent.parent / "results" / "evals")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"multiturn_judge_{timestamp}.md"
    report_path.write_text(full_report)
    print("=" * 80)
    print(f"✅ Evaluation Complete! Report saved to: {report_path}")
    print("=" * 80)

    return results, full_report


def main():
    parser = argparse.ArgumentParser(description="Run Multi-Turn Conversational E2E Judge Evaluation")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--expert", type=str, default="dynamic", help="Expert mode ('dynamic', 'astral', 'postgresql', etc.)")
    parser.add_argument("--max-tokens", type=int, default=400, help="Max completion tokens per turn")
    parser.add_argument("--thinking", type=str, default="off", choices=["off", "low", "medium", "high"], help="Thinking effort")
    args = parser.parse_args()

    run_multiturn_eval(
        base_url=args.url,
        expert=args.expert,
        max_tokens=args.max_tokens,
        thinking_effort=args.thinking,
    )


if __name__ == "__main__":
    main()
