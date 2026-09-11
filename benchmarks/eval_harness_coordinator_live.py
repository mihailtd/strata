"""Empirical Live Benchmark for Harness Coordinator & LoRA Subagent Orchestration.

Executes an end-to-end multi-subagent pipeline on AMD Radeon RX 7900 XTX:
1. Astral Subagent: Generates pyproject.toml and linter configuration.
2. Python Modern Subagent: Generates Pydantic v2 ConfigDict and PEP 695 generics.
3. PostgreSQL Subagent: Generates pgvector HNSW (<=>) database connection layer.
4. DuckDB Subagent: Generates native SQL QUALIFY window analytics.
5. FastAPI Subagent: Generates @asynccontextmanager lifespan web application.
6. Stacked Integration Subagent: Generates comprehensive Pytest test suite.

Measures:
- Subagent In-Place Swap Latency (ms)
- Generation Throughput (tok/s) across all subagents
- Real OS Pytest and Ruff Exit Codes
- Telemetry logging to `results/benchmarks/harness_coordinator_live_benchmark.json`
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.coordinator.orchestrator import HarnessCoordinator


def main():
    print("=" * 105)
    print("🔬 EMPIRICAL BENCHMARK: HARNESS COORDINATOR & LORA SUBAGENT FLEET")
    print("   Target Model: ornith-1.5:35b | Target GPU: AMD Radeon RX 7900 XTX (24 GB VRAM)")
    print("=" * 105, flush=True)

    project_dir = REPO_ROOT / "projects" / "real_coordinated_service"
    goal = "Build a high-performance vector search and analytics microservice with FastAPI, PostgreSQL 17 pgvector, and DuckDB"

    coordinator = HarnessCoordinator(model_name="ornith-1.5:35b")
    manifest = coordinator.execute_goal(goal, project_dir)

    out_file = REPO_ROOT / "results" / "benchmarks" / "harness_coordinator_live_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "goal": manifest.goal,
        "total_tokens": manifest.total_tokens,
        "total_duration_s": manifest.total_duration_s,
        "all_tests_passed": manifest.all_tests_passed,
        "subagents": [
            {
                "task_id": r.task_id,
                "specialist": r.specialist.value,
                "tok_s": r.tok_s,
                "ttft_ms": r.ttft_ms,
                "swap_latency_ms": r.swap_latency_ms,
                "tokens": r.tokens_generated,
                "artifacts": r.artifacts_created,
            }
            for r in manifest.results
        ],
    }

    with open(out_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n💾 Saved full live benchmark telemetry to: {out_file}")


if __name__ == "__main__":
    main()
