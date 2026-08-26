"""SWE-Bench Task Definitions for Domain-Specialist LLM Evaluation.

Each task represents a realistic software engineering bug or missing feature across:
  1. Astral Toolchain & Modern Python (uv, ruff, pyproject)
  2. PostgreSQL 17 & Vector Search (pgvector HNSW, asyncpg)
  3. DuckDB OLAP (Parquet streaming, window functions, QUALIFY)
  4. FastAPI & Async Web (Async DI, Pydantic v2 validation)
  5. Financial Planning & Wealth Modeling (Monte Carlo, Cholesky correlation)
  6. Multi-Domain Cross Enterprise (FastAPI + pgvector + DuckDB + Astral)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class SWEBenchTask:
    task_id: str
    domain: str
    title: str
    difficulty: str  # "Medium", "Hard", "Expert"
    problem_statement: str
    initial_code: str
    test_code: str
    canonical_solution: str
    evaluation_criteria: List[str]


SWE_BENCH_TASKS: List[SWEBenchTask] = [
    # -------------------------------------------------------------------------
    # Task 1: Astral Toolchain (uv / pyproject / dependency resolution)
    # -------------------------------------------------------------------------
    SWEBenchTask(
        task_id="swe_01_astral",
        domain="astral",
        title="Modern UV Workspace Cyclic Dependency Resolver",
        difficulty="Medium",
        problem_statement=(
            "We are migrating a monorepo to Astral `uv` workspaces. A custom dependency resolver in "
            "`workspace_resolver.py` fails when cyclic or optional extras dependencies are declared. "
            "Implement `resolve_workspace_dependencies(manifests: dict[str, dict]) -> list[str]` to return a valid "
            "topological build order using Kahn's algorithm, raising `CyclicDependencyError` with the cycle cycle trace "
            "if a hard cycle exists, while cleanly resolving self-referential workspace members."
        ),
        initial_code='''
class CyclicDependencyError(Exception):
    pass

def resolve_workspace_dependencies(manifests: dict[str, dict]) -> list[str]:
    # Buggy naive implementation: fails on cycles and extras
    order = []
    for name in manifests:
        order.append(name)
    return order
''',
        test_code='''
import pytest

def test_uv_workspace_resolution():
    manifests = {
        "core": {"dependencies": []},
        "db": {"dependencies": ["core"]},
        "api": {"dependencies": ["core", "db"]},
    }
    order = resolve_workspace_dependencies(manifests)
    assert order.index("core") < order.index("db")
    assert order.index("db") < order.index("api")

def test_uv_cyclic_detection():
    manifests = {
        "pkg_a": {"dependencies": ["pkg_b"]},
        "pkg_b": {"dependencies": ["pkg_a"]},
    }
    with pytest.raises(CyclicDependencyError):
        resolve_workspace_dependencies(manifests)
''',
        canonical_solution='''
class CyclicDependencyError(Exception):
    pass

def resolve_workspace_dependencies(manifests: dict[str, dict]) -> list[str]:
    from collections import defaultdict, deque
    
    in_degree = {name: 0 for name in manifests}
    adj = defaultdict(list)
    
    for name, data in manifests.items():
        for dep in data.get("dependencies", []):
            if dep in manifests:
                adj[dep].append(name)
                in_degree[name] += 1
                
    queue = deque([name for name, deg in in_degree.items() if deg == 0])
    order = []
    
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                
    if len(order) != len(manifests):
        raise CyclicDependencyError("Cyclic dependency detected in workspace members.")
        
    return order
''',
        evaluation_criteria=[
            "Uses Kahn's algorithm or DFS topological sort with in-degree tracking",
            "Raises CyclicDependencyError on invalid graph cycles",
            "Handles workspace members correctly with zero external library bloat",
        ],
    ),

    # -------------------------------------------------------------------------
    # Task 2: PostgreSQL 17 & Vector Search (pgvector HNSW + asyncpg)
    # -------------------------------------------------------------------------
    SWEBenchTask(
        task_id="swe_02_postgresql",
        domain="postgresql",
        title="High-Concurrency pgvector HNSW Query Builder & Pooler",
        difficulty="Hard",
        problem_statement=(
            "Write a robust asynchronous vector similarity search function `async def query_hybrid_documents(pool, query_embedding, filter_metadata, ef_search=64, limit=10)` "
            "that executes an HNSW vector cosine distance query (`<=>`) with local index tuning (`SET LOCAL hnsw.ef_search = ...`) "
            "inside a read-only transaction, ensuring parameter binding to prevent SQL injection and handling asyncpg connection leases safely."
        ),
        initial_code='''
async def query_hybrid_documents(pool, query_embedding, filter_metadata, ef_search=64, limit=10):
    # Buggy: unsafe string formatting and missing transaction isolation
    query = f"SELECT id, title, embedding <=> '{query_embedding}' as dist FROM documents LIMIT {limit}"
    return []
''',
        test_code='''
import pytest
import json

class MockConnection:
    def __init__(self):
        self.executed_stmts = []
        
    async def execute(self, stmt):
        self.executed_stmts.append(stmt)
        
    async def fetch(self, stmt, *args):
        self.executed_stmts.append((stmt, args))
        return [{"id": 1, "title": "Doc 1", "score": 0.95}]
        
    def transaction(self, readonly=False):
        class Tx:
            async def __aenter__(self): pass
            async def __aexit__(self, *a): pass
        return Tx()

class MockPool:
    def acquire(self):
        class Context:
            async def __aenter__(self_ctx): return MockConnection()
            async def __aexit__(self_ctx, *a): pass
        return Context()

@pytest.mark.asyncio
async def test_pgvector_query_structure():
    pool = MockPool()
    results = await query_hybrid_documents(pool, [0.1]*1536, {"category": "ai"}, ef_search=128, limit=5)
    assert len(results) == 1
    assert results[0]["id"] == 1
''',
        canonical_solution='''
import json

async def query_hybrid_documents(pool, query_embedding, filter_metadata, ef_search=64, limit=10):
    vec_str = f"[{','.join(str(x) for x in query_embedding)}]"
    meta_json = json.dumps(filter_metadata) if filter_metadata else None
    
    async with pool.acquire() as conn:
        async with conn.transaction(readonly=True):
            await conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
            if meta_json:
                stmt = """
                    SELECT id, title, (1 - (embedding <=> $1::vector)) AS score
                    FROM documents
                    WHERE metadata @> $2::jsonb
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3;
                """
                rows = await conn.fetch(stmt, vec_str, meta_json, limit)
            else:
                stmt = """
                    SELECT id, title, (1 - (embedding <=> $1::vector)) AS score
                    FROM documents
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2;
                """
                rows = await conn.fetch(stmt, vec_str, limit)
                
            return [dict(r) for r in rows]
''',
        evaluation_criteria=[
            "Sets local HNSW probe depth via SET LOCAL hnsw.ef_search",
            "Uses parameterized $1, $2 query placeholders instead of raw string interpolation",
            "Employs cosine distance (<=>) with 1 - distance score conversion",
        ],
    ),

    # -------------------------------------------------------------------------
    # Task 3: DuckDB Vectorized Analytics (Parquet streaming & QUALIFY)
    # -------------------------------------------------------------------------
    SWEBenchTask(
        task_id="swe_03_duckdb",
        domain="duckdb",
        title="Zero-Copy Out-of-Core Parquet Window Aggregator",
        difficulty="Hard",
        problem_statement=(
            "Implement `execute_rolling_analytics(con, parquet_glob_pattern: str, window_days: int = 7) -> list[dict]` in DuckDB. "
            "The query must read directly from multi-file Parquet globs with zero intermediate Pandas materialization, "
            "compute rolling 7-day revenue and cumulative user volume using window functions (`OVER (PARTITION BY ... ORDER BY ...)`), "
            "and filter for top anomalies using DuckDB native `QUALIFY` clause."
        ),
        initial_code='''
def execute_rolling_analytics(con, parquet_glob_pattern: str, window_days: int = 7) -> list[dict]:
    # Buggy: Missing window function and QUALIFY clause
    return []
''',
        test_code='''
import pytest

class MockDuckDBCon:
    def __init__(self):
        self.last_query = ""
        
    def execute(self, query, params=None):
        self.last_query = query
        class Cursor:
            def fetchall(self): return [(101, "2026-08-01", 1540.0, 3)]
            def description(self): return [("customer_id",), ("date",), ("rolling_rev",), ("window_count",)]
        return Cursor()

def test_duckdb_parquet_streaming():
    con = MockDuckDBCon()
    res = execute_rolling_analytics(con, "data/orders_*.parquet", window_days=7)
    assert len(res) == 1
    assert "read_parquet" in con.last_query or "parquet_scan" in con.last_query or "data/orders_" in con.last_query
    assert "QUALIFY" in con.last_query.upper()
    assert "OVER" in con.last_query.upper()
''',
        canonical_solution='''
def execute_rolling_analytics(con, parquet_glob_pattern: str, window_days: int = 7) -> list[dict]:
    query = f"""
        SELECT 
            customer_id,
            order_date,
            SUM(amount) OVER (
                PARTITION BY customer_id 
                ORDER BY order_date 
                RANGE BETWEEN INTERVAL '{int(window_days)}' DAYS PRECEDING AND CURRENT ROW
            ) AS rolling_rev,
            COUNT(*) OVER (
                PARTITION BY customer_id 
                ORDER BY order_date 
                RANGE BETWEEN INTERVAL '{int(window_days)}' DAYS PRECEDING AND CURRENT ROW
            ) AS window_count
        FROM read_parquet('{parquet_glob_pattern}')
        QUALIFY rolling_rev > 1000.0
        ORDER BY rolling_rev DESC;
    """
    cursor = con.execute(query)
    cols = [d[0] for d in cursor.description()]
    rows = cursor.fetchall()
    return [dict(zip(cols, r)) for r in rows]
''',
        evaluation_criteria=[
            "Uses read_parquet() direct streaming scan",
            "Implements partition window range with INTERVAL days",
            "Applies QUALIFY clause for analytical post-filtering without nested subqueries",
        ],
    ),

    # -------------------------------------------------------------------------
    # Task 4: FastAPI & Async Web Architecture (DI & Pydantic v2)
    # -------------------------------------------------------------------------
    SWEBenchTask(
        task_id="swe_04_fastapi",
        domain="fastapi",
        title="Async Dependency Injection Lifetime & SSE Streamer",
        difficulty="Medium",
        problem_statement=(
            "Create a robust FastAPI async route handler `build_sse_event_streamer(client_id: str, session_manager)` "
            "that streams Server-Sent Events (SSE) formatted as `data: {...}\\n\\n` using an `AsyncGenerator`, "
            "guaranteeing graceful shutdown and connection cleanup when the client disconnects."
        ),
        initial_code='''
async def build_sse_event_streamer(client_id: str, session_manager):
    # Incomplete generator
    yield "test"
''',
        test_code='''
import pytest
import asyncio
import json

class MockSessionManager:
    def __init__(self):
        self.unregistered = False
        
    async def poll_events(self, client_id):
        yield {"type": "heartbeat", "seq": 1}
        yield {"type": "payload", "data": "ready"}
        
    async def unregister(self, client_id):
        self.unregistered = True

@pytest.mark.asyncio
async def test_sse_event_generator():
    mgr = MockSessionManager()
    gen = build_sse_event_streamer("client_123", mgr)
    chunks = []
    async for chunk in gen:
        chunks.append(chunk)
    assert len(chunks) == 2
    assert chunks[0].startswith("data: ")
    assert chunks[0].endswith("\\n\\n")
    assert mgr.unregistered == True
''',
        canonical_solution='''
import json
from typing import AsyncGenerator

async def build_sse_event_streamer(client_id: str, session_manager) -> AsyncGenerator[str, None]:
    try:
        async for event in session_manager.poll_events(client_id):
            payload = json.dumps(event)
            yield f"data: {payload}\\n\\n"
    finally:
        await session_manager.unregister(client_id)
''',
        evaluation_criteria=[
            "Formats SSE protocol strictly as data: {json}\\n\\n",
            "Uses try/finally block to guarantee session_manager.unregister on disconnect",
            "Employs clean typing AsyncGenerator[str, None]",
        ],
    ),

    # -------------------------------------------------------------------------
    # Task 5: Financial Planning & Wealth Modeling (Vectorized Monte Carlo)
    # -------------------------------------------------------------------------
    SWEBenchTask(
        task_id="swe_05_financial",
        domain="financial",
        title="Vectorized Monte Carlo Portfolio Survival Engine",
        difficulty="Hard",
        problem_statement=(
            "Implement `simulate_portfolio_survival(initial_wealth: float, annual_withdrawal: float, returns_mean: list[float], returns_cov: list[list[float]], num_years: int = 30, num_simulations: int = 10000) -> dict` "
            "using NumPy vectorized Cholesky decomposition ($L L^T = \\Sigma$) to generate correlated multi-asset Gaussian returns, "
            "computing empirical ruin probability (wealth <= 0) and 10th/50th/90th percentile ending wealth."
        ),
        initial_code='''
def simulate_portfolio_survival(initial_wealth: float, annual_withdrawal: float, returns_mean: list[float], returns_cov: list[list[float]], num_years: int = 30, num_simulations: int = 10000) -> dict:
    # Buggy scalar loop
    return {"survival_probability": 0.0}
''',
        test_code='''
import pytest
import numpy as np

def test_monte_carlo_cholesky_simulation():
    means = [0.08, 0.04]  # Stocks, Bonds
    cov = [[0.04, 0.005], [0.005, 0.01]]
    res = simulate_portfolio_survival(
        initial_wealth=1_000_000.0,
        annual_withdrawal=40_000.0,
        returns_mean=means,
        returns_cov=cov,
        num_years=10,
        num_simulations=5000,
    )
    assert "survival_probability" in res
    assert 0.85 <= res["survival_probability"] <= 1.0
    assert res["p50_wealth"] > 500_000.0
''',
        canonical_solution='''
import numpy as np

def simulate_portfolio_survival(
    initial_wealth: float,
    annual_withdrawal: float,
    returns_mean: list[float],
    returns_cov: list[list[float]],
    num_years: int = 30,
    num_simulations: int = 10000,
) -> dict:
    mu = np.array(returns_mean)
    sigma = np.array(returns_cov)
    num_assets = len(mu)
    
    # Cholesky decomposition for correlated returns
    L = np.linalg.cholesky(sigma)
    
    # Generate uncorrelated standard normal shocks: (num_simulations, num_years, num_assets)
    z = np.random.standard_normal((num_simulations, num_years, num_assets))
    
    # Correlated asset returns: (num_simulations, num_years, num_assets)
    correlated_returns = mu + np.einsum('ij,syj->syi', L, z)
    
    # Equal weighted portfolio return across assets
    portfolio_annual_returns = np.mean(correlated_returns, axis=-1)
    
    wealth = np.full(num_simulations, initial_wealth)
    ruined = np.zeros(num_simulations, dtype=bool)
    
    for year in range(num_years):
        ret = portfolio_annual_returns[:, year]
        wealth = wealth * (1.0 + ret) - annual_withdrawal
        ruined = ruined | (wealth <= 0)
        wealth = np.maximum(wealth, 0.0)
        
    survival_prob = float(np.mean(~ruined))
    p10, p50, p90 = np.percentile(wealth, [10, 50, 90])
    
    return {
        "survival_probability": round(survival_prob, 4),
        "ruin_probability": round(1.0 - survival_prob, 4),
        "p10_wealth": float(p10),
        "p50_wealth": float(p50),
        "p90_wealth": float(p90),
    }
''',
        evaluation_criteria=[
            "Performs Cholesky decomposition via np.linalg.cholesky",
            "Vectorizes across all simulation paths simultaneously",
            "Calculates ruin probability and percentiles correctly",
        ],
    ),

    # -------------------------------------------------------------------------
    # Task 6: Cross-Domain Enterprise (FastAPI + pgvector + DuckDB + Astral)
    # -------------------------------------------------------------------------
    SWEBenchTask(
        task_id="swe_06_cross_domain",
        domain="cross_domain",
        title="Full-Stack Vector OLAP Analytics Streamer",
        difficulty="Expert",
        problem_statement=(
            "Build an integrated pipeline function `async def run_hybrid_vector_olap(vector_pool, duckdb_con, query_embedding, partition_date: str)` "
            "that first queries PostgreSQL 17 HNSW index for top-50 candidate documents, loads them into an in-memory DuckDB zero-copy view, "
            "and joins against the Parquet user log partition to compute conversion rates by document score tier."
        ),
        initial_code='''
async def run_hybrid_vector_olap(vector_pool, duckdb_con, query_embedding, partition_date: str):
    # Incomplete pipeline
    return {}
''',
        test_code='''
import pytest

class MockVectorPool:
    def acquire(self):
        class Conn:
            async def __aenter__(self):
                class C:
                    async def fetch(self, *a):
                        return [{"doc_id": 1, "score": 0.92}, {"doc_id": 2, "score": 0.85}]
                return C()
            async def __aexit__(self, *a): pass
        return Conn()

class MockDuckDB:
    def __init__(self):
        self.registered_data = None
        self.executed_query = ""
    def register(self, name, df):
        self.registered_data = df
    def execute(self, q):
        self.executed_query = q
        class Cursor:
            def fetchall(self): return [("tier_1", 0.45), ("tier_2", 0.28)]
        return Cursor()

@pytest.mark.asyncio
async def test_cross_domain_pipeline():
    vpool = MockVectorPool()
    dcon = MockDuckDB()
    res = await run_hybrid_vector_olap(vpool, dcon, [0.1]*1536, "2026-08-26")
    assert len(res) == 2
    assert "tier_1" in res
''',
        canonical_solution='''
async def run_hybrid_vector_olap(vector_pool, duckdb_con, query_embedding, partition_date: str) -> dict[str, float]:
    vec_str = f"[{','.join(str(x) for x in query_embedding)}]"
    
    # 1. PostgreSQL HNSW Vector Query
    async with vector_pool.acquire() as conn:
        stmt = """
            SELECT id AS doc_id, (1 - (embedding <=> $1::vector)) AS score
            FROM documents
            ORDER BY embedding <=> $1::vector
            LIMIT 50;
        """
        rows = await conn.fetch(stmt, vec_str)
        doc_candidates = [dict(r) for r in rows]
        
    # 2. DuckDB In-Memory View & Parquet Analytical Join
    import pyarrow as pa
    arrow_table = pa.Table.from_pylist(doc_candidates)
    duckdb_con.register("vector_candidates", arrow_table)
    
    olap_query = f"""
        SELECT 
            CASE WHEN score >= 0.90 THEN 'tier_1' ELSE 'tier_2' END AS score_tier,
            AVG(CASE WHEN event_type = 'conversion' THEN 1.0 ELSE 0.0 END) AS conversion_rate
        FROM vector_candidates vc
        LEFT JOIN read_parquet('data/events/date={partition_date}/*.parquet') ev
            ON vc.doc_id = ev.doc_id
        GROUP BY score_tier;
    """
    cursor = duckdb_con.execute(olap_query)
    results = {row[0]: float(row[1]) for row in cursor.fetchall()}
    return results
''',
        evaluation_criteria=[
            "Executes async PostgreSQL vector query with cosine distance operator",
            "Registers candidate set directly to DuckDB via PyArrow zero-copy table",
            "Performs analytical Parquet partition join with conditional aggregation",
        ],
    ),
]
