"""CLI runner for Toolery deterministic tool-use benchmark (143 scenarios across 4 difficulty tiers)."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

# Ensure vendor toolery package is loaded
import benchmarks.toolery
from toolery.adapters.openai_raw import OpenAIRawAdapter
from toolery.core.models import Scenario, ScenarioResult, TraceResult
from toolery.core.scorer import evaluate

from benchmarks.toolery import TIERS, load_all_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Toolery Deterministic Tool-Use Benchmark Runner (143 scenarios)"
    )
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
        "--tier",
        choices=["all", "easy", "medium", "hard", "very_hard"],
        default="all",
        help="Scenario tier to run (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of scenarios to evaluate",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=40,
        help="Timeout in seconds per scenario (default: 40)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional output path to save JSON scorecard",
    )
    return parser.parse_args()


class LoRAOpenAIRawAdapter(OpenAIRawAdapter):
    """Subclass of OpenAIRawAdapter that injects adapter name into the request payload."""

    def __init__(self, base_url: str, adapter_name: str | None = None, **kwargs):
        super().__init__(base_url=base_url, **kwargs)
        self.adapter_name = adapter_name

    async def run_scenario(self, scenario: Scenario, model: str, timeout: int) -> TraceResult:
        model_name = f"{model}:{self.adapter_name}" if self.adapter_name else model
        return await super().run_scenario(scenario, model=model_name, timeout=timeout)


async def execute_all_scenarios(
    scenarios: list[Scenario],
    adapter: LoRAOpenAIRawAdapter,
    model: str,
    timeout: int,
) -> list[ScenarioResult]:
    results: list[ScenarioResult] = []

    for idx, sc in enumerate(scenarios, 1):
        print(f"\n▶ [{idx}/{len(scenarios)}] [{sc.tier.upper()}] {sc.id}: {sc.title}", flush=True)
        try:
            trace = await adapter.run_scenario(sc, model=model, timeout=timeout)
            res = evaluate(sc, trace)

            status_icon = "✅ PASS" if res.status == "pass" else ("⚠️ PARTIAL" if res.status == "partial" else "❌ FAIL")
            print(
                f"   {status_icon} (Score: {res.score:.2f}) | Calls: {res.call_count}/{res.budget_max} | "
                f"Latency: {res.latency_ms:5.1f}ms",
                flush=True,
            )
            if res.status != "pass":
                failed_checks = [c for c in res.checks if c.result == "fail"]
                if failed_checks:
                    print(f"   Failed checks: {', '.join(f'{c.check}: {c.detail}' for c in failed_checks[:2])}")

            results.append(res)
        except Exception as exc:
            print(f"   ⚠️ ERROR: {exc}", flush=True)

    await adapter.aclose()
    return results


def run_benchmark():
    args = parse_args()
    tier_filter = None if args.tier == "all" else args.tier
    scenarios = load_all_scenarios(tier_filter)

    if args.limit:
        scenarios = scenarios[: args.limit]

    print("=" * 80)
    print("🚀 TOOLERY DETERMINISTIC TOOL-USE BENCHMARK RUNNER")
    print(f"   Target URL   : {args.base_url}")
    print(f"   Model        : {args.model}")
    print(f"   Adapter      : {args.adapter or 'none (base)'}")
    print(f"   Tier         : {args.tier.upper()}")
    print(f"   Scenarios    : {len(scenarios)}")
    print("=" * 80)

    t_start = time.perf_counter()
    adapter = LoRAOpenAIRawAdapter(
        base_url=args.base_url,
        adapter_name=args.adapter,
    )

    results = asyncio.run(
        execute_all_scenarios(
            scenarios=scenarios,
            adapter=adapter,
            model=args.model,
            timeout=args.timeout,
        )
    )
    total_wall_s = time.perf_counter() - t_start

    # Aggregate scores overall and by tier
    tier_stats: dict[str, dict] = {}
    for t in TIERS:
        t_results = [r for r, sc in zip(results, scenarios) if sc.tier == t]
        if not t_results:
            continue
        t_passed = sum(1 for r in t_results if r.status == "pass")
        t_avg_score = sum(r.score for r in t_results) / len(t_results)
        tier_stats[t] = {
            "total": len(t_results),
            "passed": t_passed,
            "pass_rate_pct": round((t_passed / len(t_results)) * 100, 2),
            "avg_score": round(t_avg_score, 4),
        }

    total_passed = sum(1 for r in results if r.status == "pass")
    overall_pass_rate = (total_passed / max(1, len(results))) * 100
    overall_avg_score = (sum(r.score for r in results) / max(1, len(results))) if results else 0.0

    print("\n" + "=" * 80)
    print("📊 TOOLERY BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Tier':<12} | {'Total':<8} | {'Passed':<8} | {'Pass Rate':<12} | {'Avg Score':<10}")
    print("-" * 80)
    for t, s in tier_stats.items():
        print(f"{t.capitalize():<12} | {s['total']:<8} | {s['passed']:<8} | {s['pass_rate_pct']:.1f}%{'':<6} | {s['avg_score']:.4f}")
    print("-" * 80)
    print(f"OVERALL      | {len(results):<8} | {total_passed:<8} | {overall_pass_rate:.1f}%{'':<6} | {overall_avg_score:.4f}")
    print(f"Total Wall Clock: {total_wall_s:.2f}s")
    print("=" * 80)

    scorecard = {
        "benchmark": "Toolery",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "adapter": args.adapter,
            "tier": args.tier,
            "limit": args.limit,
        },
        "summary": {
            "total_scenarios": len(results),
            "passed": total_passed,
            "pass_rate_pct": round(overall_pass_rate, 2),
            "avg_score": round(overall_avg_score, 4),
            "tier_breakdown": tier_stats,
            "total_duration_s": round(total_wall_s, 2),
        },
        "results": [r.model_dump() for r in results],
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
        print(f"💾 Saved scorecard to {out_path}")


if __name__ == "__main__":
    run_benchmark()
