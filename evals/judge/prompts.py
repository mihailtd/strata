"""Curated Prompts and Conversational Scenarios for E2E Judge Evaluation.

Contains standardized multi-turn conversations, single-domain specialist prompts,
and hybrid multi-expert challenges designed to test runtime capabilities:
  - LoRA expert specialization & tone
  - Dynamic Riemannian multi-expert stacking
  - Multi-turn KV cache continuity & context retention
  - Renko brick smoothing & latent drift detection
  - Speculative decoding fidelity vs baseline
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EvalPrompt:
    id: str
    domain: str
    expected_experts: list[str]
    prompt: str
    description: str
    category: Literal["single_domain", "multi_turn", "hybrid_stacking", "stress_test"] = "single_domain"


# -----------------------------------------------------------------------------
# 1. Standard Dashboard 4-Turn Sequential Chat (Shared Context Window)
# -----------------------------------------------------------------------------
DASHBOARD_MULTITURN_SEQUENCE: list[EvalPrompt] = [
    EvalPrompt(
        id="dash_turn_1_astral",
        domain="astral",
        expected_experts=["astral"],
        prompt="Add ruff and ty as dev dependencies, then format and lint the whole codebase.",
        description="Tests Astral tooling specialist (ruff & ty CLI commands, pyproject.toml dev group).",
        category="multi_turn",
    ),
    EvalPrompt(
        id="dash_turn_2_postgres",
        domain="postgresql",
        expected_experts=["postgresql"],
        prompt="We store product descriptions in Postgres and want 'find me similar products' without standing up new infrastructure.",
        description="Tests PostgreSQL / pgvector specialist (hnsw index, cosine ops, vector embedding column).",
        category="multi_turn",
    ),
    EvalPrompt(
        id="dash_turn_3_fastapi",
        domain="python_web",
        expected_experts=["python_web", "python_modern"],
        prompt="Write an async FastAPI endpoint with Pydantic request and response models and dependency injection.",
        description="Tests FastAPI + Pydantic multi-expert blend (async route, dependency injection, type annotations).",
        category="multi_turn",
    ),
    EvalPrompt(
        id="dash_turn_4_duckdb",
        domain="duckdb",
        expected_experts=["duckdb"],
        prompt="Aggregate a directory of parquet files and return the top 3 rows per group.",
        description="Tests DuckDB / Parquet analytics specialist (glob querying, window functions / ROW_NUMBER).",
        category="multi_turn",
    ),
]


# -----------------------------------------------------------------------------
# 2. Hybrid Multi-Domain Challenges (Testing Multi-Expert Dynamic Stacking)
# -----------------------------------------------------------------------------
HYBRID_STACKING_PROMPTS: list[EvalPrompt] = [
    EvalPrompt(
        id="hybrid_fastapi_duckdb",
        domain="hybrid",
        expected_experts=["python_web", "duckdb"],
        prompt="Build an async FastAPI streaming endpoint that queries a local DuckDB parquet lakehouse using arrow streaming and returns ndjson.",
        description="Demands simultaneous mastery of modern Python async web routing + DuckDB in-process OLAP.",
        category="hybrid_stacking",
    ),
    EvalPrompt(
        id="hybrid_postgres_python",
        domain="hybrid",
        expected_experts=["postgresql", "python_modern"],
        prompt="Write a Python asyncpg connection pool helper with robust retry backoff and a pgvector cosine similarity search function.",
        description="Demands modern Python async/typing + specialized PostgreSQL pgvector queries.",
        category="hybrid_stacking",
    ),
    EvalPrompt(
        id="hybrid_astral_fullstack",
        domain="hybrid",
        expected_experts=["astral", "python_modern"],
        prompt="Create a uv workspace configuration with multiple Python packages, custom ruff lint rules for modern Python 3.12+ syntax, and ty typechecking.",
        description="Demands modern Astral toolchain configuration + Python 3.12+ type system idioms.",
        category="hybrid_stacking",
    ),
]


# -----------------------------------------------------------------------------
# 3. Domain Specialist Probes (Single-Expert Quality Benchmarks)
# -----------------------------------------------------------------------------
DOMAIN_SPECIALIST_PROBES: dict[str, list[EvalPrompt]] = {
    "astral": [
        EvalPrompt(
            id="astral_uv_pip_compile",
            domain="astral",
            expected_experts=["astral"],
            prompt="How do I use uv to compile locked dependencies for multiple platforms into standard requirements files?",
            description="Tests advanced uv CLI flags and cross-platform compilation.",
        ),
    ],
    "postgresql": [
        EvalPrompt(
            id="postgres_partitioning_jsonb",
            domain="postgresql",
            expected_experts=["postgresql"],
            prompt="Design a declarative time-partitioned table in PostgreSQL 16 with a GIN index on a JSONB metadata payload and an automated cleanup retention policy.",
            description="Tests deep PostgreSQL storage engines, indexing strategies, and retention triggers.",
        ),
    ],
    "duckdb": [
        EvalPrompt(
            id="duckdb_window_qualify",
            domain="duckdb",
            expected_experts=["duckdb"],
            prompt="Write a DuckDB SQL query using QUALIFY, ASOF JOIN, and window functions to compute rolling 7-day user churn over partitioned iceberg files.",
            description="Tests DuckDB-specific SQL dialect extensions (QUALIFY, ASOF JOIN).",
        ),
    ],
    "python_web": [
        EvalPrompt(
            id="python_web_lifespan_middleware",
            domain="python_web",
            expected_experts=["python_web", "python_modern"],
            prompt="Implement an ASGI lifespan context manager in FastAPI with custom rate-limiting middleware that uses token bucket algorithm with redis.",
            description="Tests modern FastAPI lifespan protocol + ASGI middleware architecture.",
        ),
    ],
}
