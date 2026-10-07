"""CLI runner for Berkeley Function Calling Leaderboard (BFCL).

CRITICAL INVARIANT: Java is completely ignored and excluded.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .data_loader import ALLOWED_CATEGORIES, load_bfcl_data
from .evaluator import BFCLEvalResult, evaluate_tool_calls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Berkeley Function Calling Leaderboard (BFCL) Runner")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8003/v1",
        help="OpenAI-compatible base URL (default: http://127.0.0.1:8003/v1)",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5:4b-rust",
        help="Model ID to evaluate (default: qwen3.5:4b-rust)",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional LoRA adapter name to activate (e.g. 'agentic')",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="simple",
        help=f"Comma-separated categories to run. Allowed: {ALLOWED_CATEGORIES} (Java excluded).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of test cases to evaluate",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Generation budget per case (default 4096; thinking models need room to finish <think>)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional output path to save JSON scorecard",
    )
    return parser.parse_args()


def query_tool_completion(
    base_url: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
    adapter: str | None = None,
    max_tokens: int = 4096,
    timeout: float = 600.0,
) -> dict:
    """Queries an OpenAI-compatible endpoint with a tools schema.

    Returns the raw response facts, not just the extracted calls: `content`,
    `finish_reason`, the server's own `tool_calls`, and any fallback parse
    errors. The previous version returned only the calls and silently dropped
    everything it could not `json.loads`, which is how a harness/format
    mismatch (Qwen3.5's tool-call format is XML, not JSON) reported 0/10 as if
    it were a model failure.
    """
    payload: dict = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if adapter:
        payload["adapter"] = adapter

    req = Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except URLError as exc:
        raise ConnectionError(f"Failed to connect to endpoint {base_url}: {exc}") from exc
    duration = time.perf_counter() - t0

    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content") or ""
    server_calls = message.get("tool_calls") or []
    tool_calls = list(server_calls)
    parse_errors: list[str] = []

    # Fallback for servers that return calls only inside `content` as JSON.
    # Kept for other runtimes, but failures are now RECORDED, never swallowed.
    if not server_calls:
        import re

        for m in re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
            try:
                parsed = json.loads(m.strip())
                args = parsed.get("arguments", {})
                tool_calls.append({
                    "function": {
                        "name": parsed.get("name"),
                        "arguments": json.dumps(args) if isinstance(args, dict) else str(args),
                    }
                })
            except Exception as exc:  # noqa: BLE001 - recorded below, not hidden
                parse_errors.append(f"{type(exc).__name__}: {exc}")

    usage = data.get("usage", {})
    comp_tokens = usage.get("completion_tokens", 0)
    return {
        "tool_calls": tool_calls,
        "server_emitted_tool_calls": bool(server_calls),
        "content": content,
        "finish_reason": choice.get("finish_reason"),
        "parse_errors": parse_errors,
        "latency_ms": duration * 1000,
        "tok_s": (comp_tokens / duration) if duration > 0 and comp_tokens > 0 else 0.0,
        "completion_tokens": comp_tokens,
    }


def classify_failure(resp: dict, eval_error: str) -> str:
    """Buckets a failure by WHERE it happened, which decides what would fix it.

    - truncated:     hit max_tokens (usually still inside <think>) -- a budget
                     problem; grammar-constrained decoding cannot help.
    - no_call:       finished but never emitted <tool_call> -- the model chose
                     to answer in prose; grammar cannot help.
    - unparseable:   emitted <tool_call> but nothing parsed -- the ONLY bucket
                     grammar-constrained decoding would fix.
    - wrong_count / wrong_function / wrong_args: well-formed but wrong --
                     a capability problem.
    """
    body = resp["content"].rsplit("</think>", 1)[-1] if "</think>" in resp["content"] else resp["content"]
    if not resp["tool_calls"]:
        if resp["finish_reason"] == "length":
            return "truncated"
        if "<tool_call>" in body:
            return "unparseable"
        return "no_call"
    if eval_error.startswith("Count mismatch"):
        return "wrong_count"
    if "Name mismatch" in eval_error:
        return "wrong_function"
    return "wrong_args"


def run_benchmark():
    args = parse_args()
    cats = [c.strip().lower() for c in args.categories.split(",")]
    # Enforce strict invariant: remove any accidental Java requests
    clean_cats = [c for c in cats if "java" not in c or "javascript" in c]
    if not clean_cats:
        print("Error: No valid categories selected (Java is excluded).")
        return

    test_cases = load_bfcl_data(clean_cats)
    if args.limit:
        test_cases = test_cases[: args.limit]

    print("=" * 80)
    print("🚀 BERKELEY FUNCTION CALLING LEADERBOARD (BFCL) RUNNER")
    print(f"   Target URL   : {args.base_url}")
    print(f"   Model        : {args.model}")
    print(f"   Adapter      : {args.adapter or 'none (base)'}")
    print(f"   Categories   : {clean_cats} (Java strictly excluded)")
    print(f"   Total Cases  : {len(test_cases)}")
    print("=" * 80)

    results: list[dict] = []
    t_start = time.perf_counter()

    for idx, tc in enumerate(test_cases, 1):
        print(f"\n▶ [{idx}/{len(test_cases)}] ({tc.category}) {tc.id}", flush=True)
        try:
            resp = query_tool_completion(
                base_url=args.base_url,
                model=args.model,
                messages=tc.messages,
                tools=tc.tools,
                adapter=args.adapter,
                max_tokens=args.max_tokens,
            )
            eval_res: BFCLEvalResult = evaluate_tool_calls(resp["tool_calls"], tc.ground_truth)
            outcome = "pass" if eval_res.passed else classify_failure(resp, eval_res.error)
            status_str = "✅ PASS" if eval_res.passed else f"❌ FAIL [{outcome}]"
            print(
                f"   {status_str} | {resp['latency_ms']:7.1f}ms | {resp['completion_tokens']} tok | "
                f"finish={resp['finish_reason']} | calls {eval_res.matched_calls}/{eval_res.expected_calls} "
                f"(generated {len(resp['tool_calls'])})",
                flush=True,
            )
            if not eval_res.passed and eval_res.error:
                print(f"   ⚠️ Reason: {eval_res.error[:150]}")

            results.append({
                "id": tc.id,
                "category": tc.category,
                "passed": eval_res.passed,
                "outcome": outcome,
                "latency_ms": round(resp["latency_ms"], 2),
                "tok_s": round(resp["tok_s"], 2),
                "completion_tokens": resp["completion_tokens"],
                "finish_reason": resp["finish_reason"],
                "server_emitted_tool_calls": resp["server_emitted_tool_calls"],
                "tool_calls": resp["tool_calls"],
                "parse_errors": resp["parse_errors"],
                "error": eval_res.error,
                "ground_truth": tc.ground_truth,
                # The raw generation, always. Without it a 0/10 cannot be told
                # apart from a harness bug -- which is exactly what happened.
                "completion": resp["content"],
            })
        except Exception as exc:
            print(f"   ⚠️ ERROR: {exc}", flush=True)
            results.append({
                "id": tc.id,
                "category": tc.category,
                "passed": False,
                "outcome": "request_error",
                "error": str(exc),
            })

    total_wall_s = time.perf_counter() - t_start
    passed_count = sum(1 for r in results if r["passed"])
    accuracy = (passed_count / max(1, len(results))) * 100

    print("\n" + "=" * 80)
    print("📊 BFCL BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Evaluated : {len(results)}")
    print(f"Passed          : {passed_count}")
    print(f"Accuracy Rate   : {accuracy:.2f}%")
    print(f"Total Wall Clock: {total_wall_s:.2f}s")
    outcomes: dict[str, int] = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    print(f"Outcomes        : {dict(sorted(outcomes.items()))}")
    print("=" * 80)

    scorecard = {
        "benchmark": "BFCL",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "adapter": args.adapter,
            "categories": clean_cats,
            "limit": args.limit,
            "max_tokens": args.max_tokens,
        },
        "summary": {
            "total_cases": len(results),
            "passed": passed_count,
            "accuracy_pct": round(accuracy, 2),
            "total_duration_s": round(total_wall_s, 2),
            "outcomes": dict(sorted(outcomes.items())),
        },
        "results": results,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
        print(f"💾 Saved scorecard to {out_path}")


if __name__ == "__main__":
    run_benchmark()
