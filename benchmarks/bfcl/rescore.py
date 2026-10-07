"""Re-grade saved BFCL tool-call scorecards with the current evaluator.

Scorecards keep each case's raw `tool_calls`, `ground_truth`, `completion` and
`finish_reason`, so an evaluator fix can be applied to every size uniformly
without re-running the GPU. The pre-fix summary is preserved in place under
`summary_pre_rescore`, so the size of the correction stays visible.

Usage:
    uv run python -m benchmarks.bfcl.rescore results/benchmarks/scorecard_bfcl_toolcalls_*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .evaluator import evaluate_tool_calls
from .runner import classify_failure

EVALUATOR_VERSION = "nested-structure-match-v1"


def rescore(path: Path) -> dict:
    card = json.loads(path.read_text(encoding="utf-8"))
    if card.get("evaluator_version") != EVALUATOR_VERSION:
        card.setdefault("summary_pre_rescore", card["summary"])

    outcomes: dict[str, int] = {}
    for r in card["results"]:
        if r.get("outcome") == "request_error":
            outcomes["request_error"] = outcomes.get("request_error", 0) + 1
            continue
        res = evaluate_tool_calls(r["tool_calls"], r["ground_truth"])
        r["passed"] = res.passed
        r["error"] = res.error
        resp = {"content": r["completion"], "finish_reason": r["finish_reason"], "tool_calls": r["tool_calls"]}
        r["outcome"] = "pass" if res.passed else classify_failure(resp, res.error)
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1

    passed = sum(1 for r in card["results"] if r["passed"])
    card["summary"] = {
        **card["summary"],
        "passed": passed,
        "accuracy_pct": round(100 * passed / max(1, len(card["results"])), 2),
        "outcomes": dict(sorted(outcomes.items())),
    }
    card["evaluator_version"] = EVALUATOR_VERSION
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card


def main(paths: list[str]) -> int:
    for p in sorted(paths):
        card = rescore(Path(p))
        before = card["summary_pre_rescore"]["accuracy_pct"]
        after = card["summary"]["accuracy_pct"]
        print(f"{Path(p).name:60s} {before:6.2f}% -> {after:6.2f}%  {card['summary']['outcomes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
