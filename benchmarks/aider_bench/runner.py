"""Main CLI runner for the Aider Python Hello World benchmark suite."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from benchmarks.aider_bench.executor import PytestExecutor
from benchmarks.aider_bench.harness_direct import DirectHarness, DirectTaskResult
from benchmarks.aider_bench.tasks import list_available_tasks, load_task, load_tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aider Python Hello World Benchmark Runner for local runtimes and DSH harness."
    )
    parser.add_argument(
        "--tasks",
        "-t",
        type=str,
        default=None,
        help="Comma-separated task names to evaluate (e.g. 'hello_world,leap'). Default: all.",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "agentic"],
        default="direct",
        help="Evaluation mode: 'direct' (raw /v1/chat/completions) or 'agentic' (DSH SDK with tools).",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8003/v1",
        help="Base URL for the OpenAI-compatible runtime (default: http://127.0.0.1:8003/v1).",
    )
    parser.add_argument(
        "--model",
        default="qwen3.5:4b-rust",
        help="Model ID to query (default: qwen3.5:4b-rust).",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional LoRA adapter name to activate (e.g. 'astral', 'python_modern').",
    )
    parser.add_argument(
        "--edit-format",
        choices=["whole", "diff"],
        default="whole",
        help="Code edit format: 'whole' (full markdown file) or 'diff' (SEARCH/REPLACE blocks).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Maximum iterative repair turns per task (default: 5).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--container",
        action="store_true",
        help="Execute pytest inside a Docker/Podman container instead of host temp directory.",
    )
    parser.add_argument(
        "--container-image",
        default="python:3.11-slim",
        help="Container image to use when --container is enabled (default: python:3.11-slim).",
    )
    parser.add_argument(
        "--suite",
        choices=["core", "full", "all", "polyglot", "rust", "go", "javascript", "cpp", "java"],
        default="core",
        help="Task suite to run: 'core' (10 Python), 'full'/'all' (140 Python), 'polyglot' (all 225), or language track.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of tasks to execute.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to write JSON results scorecard.",
    )
    return parser.parse_args()


def run_benchmark():
    args = parse_args()

    # Load requested tasks
    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",")]
        available = list_available_tasks(suite="all")
        for t in task_names:
            if t not in available:
                print(f"Error: Unknown task '{t}'.", file=sys.stderr)
                sys.exit(1)
    else:
        task_names = list_available_tasks(suite=args.suite)

    if args.limit:
        task_names = task_names[: args.limit]

    tasks = [load_task(t) for t in task_names]

    print("=" * 90)
    print("🚀 AIDER PYTHON BENCHMARK RUNNER")
    print(f"   Mode        : {args.mode.upper()}")
    print(f"   Target URL  : {args.base_url}")
    print(f"   Model       : {args.model}")
    print(f"   Adapter     : {args.adapter or 'none (base)'}")
    print(f"   Format      : {args.edit_format}")
    print(f"   Max Turns   : {args.max_turns}")
    print(f"   Sandboxing  : {'Container (' + args.container_image + ')' if args.container else 'Local Subprocess (<100ms)'}")
    print(f"   Tasks ({len(tasks)}) : {', '.join(t.name for t in tasks)}")
    print("=" * 90)

    executor = PytestExecutor(
        use_container=args.container,
        container_image=args.container_image,
        timeout_seconds=15,
    )

    t_suite_start = time.perf_counter()
    results: list[dict] = []

    if args.mode == "direct":
        harness = DirectHarness(
            base_url=args.base_url,
            model=args.model,
            adapter=args.adapter,
            edit_format=args.edit_format,
            max_turns=args.max_turns,
            temperature=args.temperature,
            executor=executor,
        )

        for idx, task in enumerate(tasks, 1):
            print(f"\n▶ [{idx}/{len(tasks)}] Task: {task.name}")

            def on_turn(turn_idx, t_data):
                status = "✅ PASS" if (t_data.pytest_result and t_data.pytest_result.passed) else "❌ FAIL"
                if not t_data.edit_applied:
                    status = f"⚠️ EDIT_ERR ({t_data.edit_error})"
                adapter_name = t_data.adapter.split("/")[-1].split("@")[0] if t_data.adapter else "base"
                print(
                    f"   Turn {turn_idx} [{adapter_name}]: {status} | TTFT: {t_data.ttft_ms:5.1f}ms | "
                    f"Speed: {t_data.decode_tok_s:5.1f} tok/s | Tokens: {t_data.generated_tokens:3d} | "
                    f"Wall: {t_data.wall_clock_s:4.2f}s",
                    flush=True,
                )

            res: DirectTaskResult = harness.run_task(task, on_turn_complete=on_turn)
            results.append({
                "task": res.task_name,
                "passed": res.passed,
                "pass_turn": res.pass_turn,
                "total_turns": res.total_turns,
                "total_tokens": res.total_tokens,
                "avg_ttft_ms": round(res.average_ttft_ms, 2),
                "avg_tok_s": round(res.average_tok_s, 2),
                "duration_s": round(res.total_duration_s, 2),
                "error": res.error_message,
            })
    else:
        from benchmarks.aider_bench.harness_agentic import AgenticHarness

        agentic_harness = AgenticHarness(
            base_url=args.base_url,
            model=args.model,
            executor=executor,
        )

        for idx, task in enumerate(tasks, 1):
            print(f"\n▶ [{idx}/{len(tasks)}] Agentic Task: {task.name}", flush=True)
            res = agentic_harness.run_task(task)
            status = "✅ PASSED" if res.passed else "❌ FAILED"
            print(f"   Result: {status} in {res.total_duration_s:4.2f}s | Tool calls: {res.tool_calls_count}")
            results.append({
                "task": res.task_name,
                "passed": res.passed,
                "duration_s": round(res.total_duration_s, 2),
                "tool_calls": res.tool_calls_count,
                "error": res.error_message,
            })

    total_suite_duration = time.perf_counter() - t_suite_start

    # Summary table
    passed_count = sum(1 for r in results if r["passed"])
    pass_at_1_count = sum(1 for r in results if r.get("pass_turn") == 1)
    pass_rate = (passed_count / max(1, len(results))) * 100
    pass_at_1_rate = (pass_at_1_count / max(1, len(results))) * 100

    print("\n" + "=" * 90)
    print("📊 BENCHMARK SCORECARD SUMMARY")
    print("=" * 90)
    print(f"{'Task':<22} | {'Status':<10} | {'Pass Turn':<10} | {'Tokens':<8} | {'Avg TTFT':<10} | {'Speed (tok/s)':<14}")
    print("-" * 90)
    for r in results:
        status_str = "✅ PASS" if r["passed"] else "❌ FAIL"
        pass_turn_str = f"Turn {r.get('pass_turn')}" if r.get("pass_turn", 0) > 0 else "N/A"
        tokens_str = str(r.get("total_tokens", "-"))
        ttft_str = f"{r.get('avg_ttft_ms', 0):.1f} ms" if "avg_ttft_ms" in r else "-"
        tok_s_str = f"{r.get('avg_tok_s', 0):.1f}" if "avg_tok_s" in r else "-"
        print(f"{r['task']:<22} | {status_str:<10} | {pass_turn_str:<10} | {tokens_str:<8} | {ttft_str:<10} | {tok_s_str:<14}")
    print("-" * 90)
    print(f"Overall Pass Rate : {passed_count}/{len(results)} ({pass_rate:.1f}%)")
    if args.mode == "direct":
        print(f"Pass@1 Rate       : {pass_at_1_count}/{len(results)} ({pass_at_1_rate:.1f}%)")
    print(f"Total Wall Clock  : {total_suite_duration:.2f}s")
    print("=" * 90)

    scorecard = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "mode": args.mode,
            "base_url": args.base_url,
            "model": args.model,
            "adapter": args.adapter,
            "edit_format": args.edit_format,
            "max_turns": args.max_turns,
            "container": args.container,
        },
        "summary": {
            "total_tasks": len(results),
            "passed": passed_count,
            "pass_rate_pct": round(pass_rate, 2),
            "pass_at_1": pass_at_1_count,
            "pass_at_1_pct": round(pass_at_1_rate, 2),
            "total_duration_s": round(total_suite_duration, 2),
        },
        "tasks": results,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
        print(f"💾 Saved scorecard to {out_path}")


if __name__ == "__main__":
    run_benchmark()
