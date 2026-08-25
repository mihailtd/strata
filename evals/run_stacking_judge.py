#!/usr/bin/env python3
"""Hybrid Multi-Expert Stacking & Dynamic Team Judge Runner.

Tests complex multi-domain queries designed to trigger multi-expert LoRA blends
(e.g., FastAPI + DuckDB, Postgres + Modern Python) and verifies:
  - Team selection logic (whether both experts clear the threshold)
  - Output coherence without catastrophic weight interference
  - Execution speed (tok/s) under multi-expert surgical scaling
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.judge_harness import E2EJudgeHarness, TurnResult
from evals.prompts import HYBRID_STACKING_PROMPTS


def run_stacking_eval(
    base_url: str = "http://127.0.0.1:8000",
    max_tokens: int = 450,
    thinking_effort: str = "off",
    output_dir: Path | None = None,
) -> tuple[list[TurnResult], str]:
    harness = E2EJudgeHarness(base_url=base_url)

    if not harness.check_server_health():
        print(f"❌ Error: Inference server is not reachable at {base_url}.")
        print("Please start the server first (e.g. via 'uv run src/runtime/server.py')")
        sys.exit(1)

    print("=" * 80)
    print(f"🧩 STARTING HYBRID EXPERT STACKING JUDGE EVALUATION ({len(HYBRID_STACKING_PROMPTS)} prompts)")
    print(f"Target: {base_url} | Mode: Dynamic Team Stacking")
    print("=" * 80)

    results: list[TurnResult] = []

    for idx, p in enumerate(HYBRID_STACKING_PROMPTS, start=1):
        print(f"\n▶️  Evaluating Hybrid Probe {idx}/{len(HYBRID_STACKING_PROMPTS)}: `{p.id}`")
        print(f"   Target Domains: {p.expected_experts}")
        print(f"   Prompt: '{p.prompt}'")

        result = harness.execute_turn_live(
            conversation_history=[{"role": "user", "content": p.prompt}],
            turn_idx=idx,
            prompt_id=p.id,
            domain=p.domain,
            expert="dynamic",
            max_tokens=max_tokens,
            thinking_effort=thinking_effort,
        )
        results.append(result)

        team_str = " + ".join(result.active_team) if result.active_team else "base"
        print(f"   🏷️  Selected Team: `{team_str}` (Expected: {p.expected_experts})")
        print(f"   ⚡ Speed: {result.tokens_per_second} tok/s | Tokens: {result.total_tokens}")
        print(f"   💬 Output:\n{result.assistant_response.strip()}\n")

    # Build Markdown Report
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_lines = [
        f"# Hybrid Expert Stacking Judge Report ({timestamp})",
        "",
        f"- **Base URL**: `{base_url}`",
        f"- **Evaluated Probes**: `{len(results)}`",
        "",
        "## 📊 Summary Matrix",
        "",
        harness.format_summary_table(results),
        "",
        "## 📝 Full Outputs for Quality Inspection",
        "",
    ]

    for r in results:
        report_lines.append(harness.format_turn_markdown(r))

    full_report = "\n".join(report_lines)

    out_dir = output_dir or (Path(__file__).parent.parent / "results" / "evals")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"stacking_judge_{timestamp}.md"
    report_path.write_text(full_report)
    print("=" * 80)
    print(f"✅ Evaluation Complete! Report saved to: {report_path}")
    print("=" * 80)

    return results, full_report


def main():
    parser = argparse.ArgumentParser(description="Run Hybrid Expert Stacking Judge Evaluation")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000", help="Server base URL")
    parser.add_argument("--max-tokens", type=int, default=450, help="Max completion tokens")
    parser.add_argument("--thinking", type=str, default="off", choices=["off", "low", "medium", "high"])
    args = parser.parse_args()

    run_stacking_eval(base_url=args.url, max_tokens=args.max_tokens, thinking_effort=args.thinking)


if __name__ == "__main__":
    main()
