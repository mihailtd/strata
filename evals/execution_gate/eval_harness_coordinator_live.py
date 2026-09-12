"""Live execution-gate eval for the Harness Coordinator & LoRA Subagent pipeline.

Drives a real end-to-end multi-subagent pipeline on AMD Radeon RX 7900 XTX:
1. Astral Subagent: writes pyproject.toml and linter configuration.
2. Python Modern Subagent: writes Pydantic v2 ConfigDict and PEP 695 generics.
3. PostgreSQL Subagent: writes pgvector HNSW (<=>) database connection layer.
4. DuckDB Subagent: writes native SQL QUALIFY window analytics.
5. FastAPI Subagent: writes @asynccontextmanager lifespan web application.
6. Stacked Integration Subagent: writes a comprehensive Pytest test suite.

Each subagent is a real DSH-driven agent (apps/harness/coordinator/subagent.py)
with no template fallback -- success/failure and the artifacts graded by
`orchestrator.py`'s real ruff+pytest verification are exactly what the agent
produced. This is an eval (does the pipeline actually build a working
service), not a performance benchmark -- see docs/METHODOLOGY.md.

Measures:
- Subagent In-Place Swap Latency (ms) -- real adapter-fusion math
- Wall-clock duration and tool-call count per subagent
- Real OS Pytest and Ruff Exit Codes
- Telemetry logging to `results/benchmarks/harness_coordinator_live_benchmark.json`

NOT YET LIVE-VERIFIED (see apps/harness/coordinator/subagent.py's docstring
and evals/dsh_agent/README.md's smoke-check step).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.coordinator.orchestrator import HarnessCoordinator


def main():
    print("=" * 105)
    print("🔬 EXECUTION-GATE EVAL: HARNESS COORDINATOR & LORA SUBAGENT FLEET")
    print("   Target Model: ornith-1.5:35b | Target GPU: AMD Radeon RX 7900 XTX (24 GB VRAM)")
    print("=" * 105, flush=True)

    project_dir = REPO_ROOT / "agent_sandboxes" / "real_coordinated_service"
    goal = "Build a high-performance vector search and analytics microservice with FastAPI, PostgreSQL 17 pgvector, and DuckDB"

    coordinator = HarnessCoordinator(model_name="ornith-1.5:35b")
    manifest = coordinator.execute_goal(goal, project_dir)

    out_file = REPO_ROOT / "results" / "benchmarks" / "harness_coordinator_live_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    summary_data = {
        "goal": manifest.goal,
        "total_duration_s": manifest.total_duration_s,
        "all_tests_passed": manifest.all_tests_passed,
        "subagents_succeeded": sum(1 for r in manifest.results if r.success),
        "subagents_total": len(manifest.results),
        "subagents": [
            {
                "task_id": r.task_id,
                "specialist": r.specialist.value,
                "success": r.success,
                "duration_s": r.duration_s,
                "tool_calls": r.tool_calls,
                "swap_latency_ms": r.swap_latency_ms,
                "artifacts": r.artifacts_created,
                "error_message": r.error_message,
            }
            for r in manifest.results
        ],
    }

    with open(out_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n💾 Saved full live benchmark telemetry to: {out_file}")


if __name__ == "__main__":
    main()
