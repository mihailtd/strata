"""CLI & Programmatic Entrypoint for the Harness LoRA Coordinator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.harness.coordinator.orchestrator import HarnessCoordinator


def main():
    parser = argparse.ArgumentParser(description="Harness Coordinator: LoRA-Specialized Subagent Orchestrator")
    parser.add_argument(
        "--goal",
        type=str,
        default="Build an async FastAPI vector analytics service with PostgreSQL 17 pgvector and DuckDB window analytics",
        help="High-level engineering goal for subagent decomposition",
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=str(REPO_ROOT / "projects" / "real_coordinated_service"),
        help="Target project directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ornith-1.5:35b",
        help="Base MoE model engine",
    )

    args = parser.parse_args()

    coordinator = HarnessCoordinator(model_name=args.model)
    manifest = coordinator.execute_goal(args.goal, Path(args.project_dir))

    if not manifest.all_tests_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
