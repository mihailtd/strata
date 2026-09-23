"""CLI runner for HumanEval benchmark against local LLM runtimes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .dataset import HumanEvalProblem, load_humaneval_problems
from .executor import HumanEvalExecutor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HumanEval 164 Problems Benchmark Runner")
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
        help="Optional LoRA adapter name to activate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of problems to evaluate (e.g. 10 or 50)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start problem index (default: 0)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Subprocess execution timeout in seconds per problem (default: 5.0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional output path to save JSON scorecard",
    )
    return parser.parse_args()


def query_model(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    adapter: str | None = None,
    timeout: float = 60.0,
) -> tuple[str, float, float, int]:
    """Queries OpenAI-compatible streaming /chat/completions endpoint.
    Returns (completion_text, ttft_ms, decode_tok_s, token_count).
    """
    system_prompt = (
        "You are an expert Python coding assistant. "
        "Complete the requested function cleanly. "
        "Output valid Python code without Markdown explanations if possible, or inside a ```python block."
    )
    user_prompt = f"Please complete the following Python function:\n\n```python\n{prompt}\n```"

    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "stream": True,
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
    first_token_time: float | None = None
    chunks: list[str] = []

    try:
        with urlopen(req, timeout=timeout) as resp:
            for line in resp:
                line_str = line.decode("utf-8").strip()
                if not line_str or not line_str.startswith("data:"):
                    continue
                data_part = line_str.removeprefix("data:").strip()
                if data_part == "[DONE]":
                    break
                try:
                    chunk_json = json.loads(data_part)
                    choices = chunk_json.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        chunks.append(content)
                except json.JSONDecodeError:
                    continue
    except URLError as exc:
        raise ConnectionError(f"Failed to connect to endpoint {base_url}: {exc}") from exc

    t_end = time.perf_counter()
    full_text = "".join(chunks)
    ttft_ms = ((first_token_time - t0) * 1000) if first_token_time else 0.0
    decode_duration = (t_end - first_token_time) if first_token_time else (t_end - t0)
    tok_count = len(chunks)
    tok_s = (tok_count / decode_duration) if decode_duration > 0 and tok_count > 0 else 0.0

    return full_text, ttft_ms, tok_s, tok_count


def run_benchmark():
    args = parse_args()
    all_problems = load_humaneval_problems()

    selected = all_problems[args.start_idx :]
    if args.limit:
        selected = selected[: args.limit]

    print("=" * 80)
    print("🚀 HUMANEVAL 164 BENCHMARK RUNNER")
    print(f"   Target URL   : {args.base_url}")
    print(f"   Model        : {args.model}")
    print(f"   Adapter      : {args.adapter or 'none (base)'}")
    print(f"   Problems     : {len(selected)} (range [{args.start_idx}:{args.start_idx + len(selected)}])")
    print("=" * 80)

    executor = HumanEvalExecutor(timeout_seconds=args.timeout)
    results: list[dict] = []
    t_start = time.perf_counter()

    for idx, prob in enumerate(selected, 1):
        print(f"\n▶ [{idx}/{len(selected)}] {prob.task_id} ({prob.entry_point})", flush=True)
        try:
            completion, ttft_ms, tok_s, tok_count = query_model(
                base_url=args.base_url,
                model=args.model,
                prompt=prob.prompt,
                temperature=args.temperature,
                adapter=args.adapter,
            )
            exec_res = executor.execute(prob, completion)
            status_str = "✅ PASS" if exec_res.passed else "❌ FAIL"
            print(
                f"   {status_str} | TTFT: {ttft_ms:5.1f}ms | Speed: {tok_s:5.1f} tok/s | "
                f"Tokens: {tok_count} | Exec: {exec_res.duration_s * 1000:4.1f}ms",
                flush=True,
            )
            results.append({
                "task_id": prob.task_id,
                "entry_point": prob.entry_point,
                "passed": exec_res.passed,
                "ttft_ms": round(ttft_ms, 2),
                "tok_s": round(tok_s, 2),
                "tokens": tok_count,
                "exec_duration_s": round(exec_res.duration_s, 4),
                "error": exec_res.error[:200] if exec_res.error else "",
                # Kept so a scoring/extraction change can be re-evaluated
                # offline instead of re-running every generation on the GPU.
                "completion": completion,
            })
        except Exception as exc:
            print(f"   ⚠️ ERROR: {exc}", flush=True)
            results.append({
                "task_id": prob.task_id,
                "entry_point": prob.entry_point,
                "passed": False,
                "error": str(exc),
            })

    total_wall_s = time.perf_counter() - t_start
    passed_count = sum(1 for r in results if r["passed"])
    pass_at_1 = (passed_count / max(1, len(results))) * 100

    print("\n" + "=" * 80)
    print("📊 HUMANEVAL BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Total Evaluated : {len(results)}")
    print(f"Passed          : {passed_count}")
    print(f"Pass@1 Rate     : {pass_at_1:.2f}%")
    print(f"Total Time      : {total_wall_s:.2f}s")
    print("=" * 80)

    scorecard = {
        "benchmark": "HumanEval",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "adapter": args.adapter,
            "limit": args.limit,
            "start_idx": args.start_idx,
            "temperature": args.temperature,
            "timeout": args.timeout,
        },
        "summary": {
            "total_problems": len(results),
            "passed": passed_count,
            "pass_rate_pct": round(pass_at_1, 2),
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
