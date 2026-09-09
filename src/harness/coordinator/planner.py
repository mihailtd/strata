"""Autonomous Task Decomposition & Subagent Planning Engine."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from src.harness.coordinator.types import DomainSpecialist, SubagentTask


class HarnessPlanner:
    """Decomposes a complex software engineering goal into LoRA-specialized subagent tasks."""

    def plan_project(self, goal: str, project_dir: Path) -> List[SubagentTask]:
        """Analyzes the goal and generates an ordered list of specialized subagent tasks."""
        tasks: List[SubagentTask] = []
        goal_lower = goal.lower()

        # Step 1: Project Toolchain & Environment (Astral Specialist)
        tasks.append(
            SubagentTask(
                task_id="step_1_toolchain",
                title="Project Initialization & Astral uv/ruff Configuration",
                specialist=DomainSpecialist.ASTRAL,
                adapter_weights={"astral": 1.0},
                prompt=(
                    f"Configure the modern Python project structure for: '{goal}'. "
                    "Generate a valid `pyproject.toml` with strict Astral `[tool.ruff.lint]` configuration tables "
                    "and all necessary dependencies."
                ),
                expected_artifacts=["pyproject.toml"],
                dependencies=[],
            )
        )

        # Step 2: Modern Domain Models & Generics (Python Modern Specialist)
        tasks.append(
            SubagentTask(
                task_id="step_2_models",
                title="Domain Models & Python 3.12 Generics",
                specialist=DomainSpecialist.PYTHON_MODERN,
                adapter_weights={"python_modern": 1.0},
                prompt=(
                    f"Design the core data models for: '{goal}'. "
                    "Use Pydantic v2 `ConfigDict` and modern Python 3.12 PEP 695 type parameter syntax "
                    "(e.g. `class ResponseEnvelope[T]:` or `class Entity[T]:`) without legacy TypeVar."
                ),
                expected_artifacts=["models.py"],
                dependencies=["step_1_toolchain"],
            )
        )

        # Step 3: Database & Vector Schema (PostgreSQL Specialist)
        if any(w in goal_lower for w in ["database", "postgres", "vector", "embedding", "search", "store", "sql", "hnsw"]):
            tasks.append(
                SubagentTask(
                    task_id="step_3_database",
                    title="PostgreSQL 17 & pgvector HNSW Schema & Engine",
                    specialist=DomainSpecialist.POSTGRESQL,
                    adapter_weights={"postgresql": 1.0},
                    prompt=(
                        "Write a clean async database module using asyncpg. "
                        "Define the schema creation SQL with an HNSW index on vector embeddings using Cosine distance (<=>), "
                        "and implement parameterized ($1, $2) vector similarity queries."
                    ),
                    expected_artifacts=["db.py"],
                    dependencies=["step_2_models"],
                )
            )

        # Step 4: Analytics Engine (DuckDB Specialist)
        if any(w in goal_lower for w in ["analytics", "duckdb", "parquet", "olap", "ranking", "percentile", "metrics", "window"]):
            tasks.append(
                SubagentTask(
                    task_id="step_4_analytics",
                    title="DuckDB Vectorized SQL Analytics & Window Processing",
                    specialist=DomainSpecialist.DUCKDB,
                    adapter_weights={"duckdb": 1.0},
                    prompt=(
                        "Implement high-performance in-memory analytics using DuckDB. "
                        "Use native SQL window clauses with the `QUALIFY` keyword for percentile filtering and ranking."
                    ),
                    expected_artifacts=["analytics.py"],
                    dependencies=["step_2_models"],
                )
            )

        # Step 5: Web API & Lifespan Architecture (FastAPI Specialist)
        if any(w in goal_lower for w in ["api", "fastapi", "web", "service", "endpoint", "server", "microservice", "http"]):
            tasks.append(
                SubagentTask(
                    task_id="step_5_web_api",
                    title="FastAPI Async Web Service & Lifespan Context",
                    specialist=DomainSpecialist.PYTHON_WEB,
                    adapter_weights={"python_web": 1.0},
                    prompt=(
                        "Implement the main FastAPI web application. "
                        "Use the modern `@asynccontextmanager async def lifespan(app: FastAPI):` architecture "
                        "for clean startup/shutdown, dependency injection, and expose REST endpoints."
                    ),
                    expected_artifacts=["main.py"],
                    dependencies=["step_2_models"],
                )
            )

        # Step 6: End-to-End Integration & Multi-Expert Test Suite (Stacked Specialist)
        tasks.append(
            SubagentTask(
                task_id="step_6_integration_tests",
                title="Cross-Domain Integration & Pytest Verification Suite",
                specialist=DomainSpecialist.STACKED_CROSS_DOMAIN,
                adapter_weights={"postgresql": 0.35, "python_web": 0.35, "duckdb": 0.30},
                prompt=(
                    "Write a comprehensive Pytest test suite in `tests/test_service.py` verifying all components end-to-end: "
                    "1. Pydantic v2 & PEP 695 models. "
                    "2. Database vector querying and connection lifespan. "
                    "3. DuckDB analytics execution. "
                    "Ensure all tests are clean, isolated, and executable with `uv run pytest`."
                ),
                expected_artifacts=["tests/test_service.py"],
                dependencies=[t.task_id for t in tasks],
            )
        )

        return tasks
