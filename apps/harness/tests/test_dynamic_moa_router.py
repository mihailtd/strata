"""DynamicMoARouter multi-domain intent classification.

Split out of the former tests/test_multi_expert_stacking.py -- the first
real behavioral test in apps/harness/tests/ (previously only structural
smoke checks, see test_smoke_harness.py).
"""

from __future__ import annotations

from harness.router.dynamic_moa_router import DynamicMoARouter


def test_moa_router_multi_domain_classification():
    """Verifies DynamicMoARouter classifies multi-domain prompts and computes weights."""
    router = DynamicMoARouter()
    prompt = (
        "Build a FastAPI web microservice with @asynccontextmanager lifespan and Pydantic v2 schemas, "
        "querying PostgreSQL 17 pgvector with cosine distance <=> "
        "and streaming into DuckDB with QUALIFY ROW_NUMBER() window clauses."
    )
    route_res = router.route_and_stack(prompt)
    assert route_res["is_multi_expert"] is True
    assert "postgresql" in route_res["experts"]
    assert "python_web" in route_res["experts"]
    assert "duckdb" in route_res["experts"]
