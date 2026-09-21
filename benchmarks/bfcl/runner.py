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

from .data_loader import ALLOWED_CATEGORIES, BFCLTestCase, load_bfcl_data
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
    timeout: float = 30.0,
) -> tuple[list[dict], float, float, int]:
    """Queries OpenAI-compatible endpoint with tools schema.
    Returns (tool_calls, ttft_ms, decode_tok_s, tokens).
    """
    payload: dict = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.0,
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
    tool_calls = message.get("tool_calls") or []

    if not tool_calls:
        content = message.get("content", "")
        import re
        for m in re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
            try:
                parsed = json.loads(m.strip())
                tool_calls.append({
                    "function": {
                        "name": parsed.get("name"),
                        "arguments": json.dumps(parsed.get("arguments", {})) if isinstance(parsed.get("arguments"), dict) else str(parsed.get("arguments")),
                    }
                })
            except Exception:
                pass

    usage = data.get("usage", {})
    comp_tokens = usage.get("completion_tokens", 0)
    tok_s = (comp_tokens / duration) if duration > 0 and comp_tokens > 0 else 0.0

    return tool_calls, duration * 1000, tok_s, comp_tokens


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
            tool_calls, latency_ms, tok_s, tok_count = query_tool_completion(
                base_url=args.base_url,
                model=args.model,
                messages=tc.messages,
                tools=tc.tools,
                adapter=args.adapter,
            )
            eval_res: BFCLEvalResult = evaluate_tool_calls(tool_calls, tc.ground_truth)
            status_str = "✅ PASS" if eval_res.passed else "❌ FAIL"
            print(
                f"   {status_str} | Latency: {latency_ms:5.1f}ms | Calls: {eval_res.matched_calls}/{eval_res.expected_calls} | "
                f"Generated: {len(tool_calls)}",
                flush=True,
            )
            if not eval_res.passed and eval_res.error:
                print(f"   ⚠️ Reason: {eval_res.error[:150]}")

            results.append({
                "id": tc.id,
                "category": tc.category,
                "passed": eval_res.passed,
                "latency_ms": round(latency_ms, 2),
                "tok_s": round(tok_s, 2),
                "tool_calls": tool_calls,
                "error": eval_res.error,
            })
        except Exception as exc:
            print(f"   ⚠️ ERROR: {exc}", flush=True)
            results.append({
                "id": tc.id,
                "category": tc.category,
                "passed": False,
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
        },
        "summary": {
            "total_cases": len(results),
            "passed": passed_count,
            "accuracy_pct": round(accuracy, 2),
            "total_duration_s": round(total_wall_s, 2),
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
