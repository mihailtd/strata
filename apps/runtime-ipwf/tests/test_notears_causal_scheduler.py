"""Unit tests for NOTEARS Causal Structure Learning & Predictive Pre-Folding Scheduler."""

from __future__ import annotations

import time

import numpy as np
import scipy.linalg as sla
from notears_causal_scheduler import (
    NotearsCausalScheduler,
    notears_linear,
)


def test_notears_acyclicity_constraint():
    """Verify NOTEARS output satisfies the smooth acyclicity condition h(W) <= 1e-6."""
    np.random.seed(42)
    # Generate 5-node linear SEM
    d = 5
    n = 100
    W_true = np.array(
        [
            [0.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.6, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.7, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.8],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    Z = np.random.randn(n, d)
    X = Z @ sla.inv(np.eye(d) - W_true)

    W_est = notears_linear(X - X.mean(0), lambda1=0.01, w_threshold=0.1)

    # Check acyclicity h(W) = Tr(exp(W o W)) - d
    E = sla.expm(W_est * W_est)
    h = np.trace(E) - d
    assert h < 1e-5, f"Estimated DAG violates acyclicity: h(W) = {h}"
    # Verify no self-loops
    assert np.all(np.diag(W_est) == 0.0)


def test_causal_scheduler_initialization():
    """Verify scheduler initializes with canonical structural priors and non-zero edges."""
    scheduler = NotearsCausalScheduler()
    assert scheduler.fitted is True
    assert scheduler.d > 0
    assert len(scheduler.nodes) == scheduler.d

    dag_info = scheduler.get_dag_structure()
    assert len(dag_info["edges"]) > 0
    assert "telemetry" in dag_info


def test_causal_scheduler_prediction():
    """Verify next-expert prediction aligns with tool surfaces and causal transitions."""
    scheduler = NotearsCausalScheduler()

    # Step 1: astral project setup -> predicts database setup (postgresql) or typing setup (python_modern)
    pred_pg, conf_pg = scheduler.predict_next_expert(tools=["uv_init", "uv_add"], current_expert="astral")
    assert pred_pg in ["postgresql", "python_modern"]
    assert conf_pg >= 0.30

    # Step 2: postgresql schema creation -> predicts duckdb analytics / parquet ingestion
    pred_duck, conf_duck = scheduler.predict_next_expert(tools=["sql_ddl", "pgvector"], current_expert="postgresql")
    assert pred_duck == "duckdb"
    assert conf_duck >= 0.50

    # Step 3: duckdb analytics -> predicts python_web API endpoint development
    pred_web, conf_web = scheduler.predict_next_expert(
        tools=["duckdb_query", "export_parquet"], current_expert="duckdb"
    )
    assert pred_web == "python_web"
    assert conf_web >= 0.50

    # Step 4: python_modern type check & lint -> predicts python_web API endpoint
    pred_web2, conf_web2 = scheduler.predict_next_expert(
        tools=["ty_check", "ruff_format"], current_expert="python_modern"
    )
    assert pred_web2 == "python_web"
    assert conf_web2 >= 0.50


def test_causal_scheduler_async_prefold():
    """Verify async pre-folding executes without blocking."""
    scheduler = NotearsCausalScheduler()

    class MockFoldingEngine:
        def __init__(self):
            self.active = None
            self.fold_count = 0

        def activate(self, expert_name: str):
            time.sleep(0.01)  # Simulate 10ms folding
            self.active = expert_name
            self.fold_count += 1

    engine = MockFoldingEngine()

    # Prefold with high confidence
    success = scheduler.async_prefold(engine, "postgresql", confidence=0.85, threshold=0.70)
    assert success is True

    # Give worker thread 50ms to finish
    time.sleep(0.05)
    assert engine.active == "postgresql"
    assert engine.fold_count == 1
    assert scheduler.telemetry["prefold_triggers"] == 1


def test_causal_scheduler_fit_traces():
    """Verify fitting on session traces updates structural adjacency matrix."""
    scheduler = NotearsCausalScheduler()

    mock_traces = [
        {"session": "s1", "ts": 1.0, "tools": ["uv_init"], "primary": "astral"},
        {"session": "s1", "ts": 2.0, "tools": ["sql_ddl"], "primary": "postgresql"},
        {"session": "s1", "ts": 3.0, "tools": ["duckdb_query"], "primary": "duckdb"},
        {"session": "s2", "ts": 1.0, "tools": ["uv_init"], "primary": "astral"},
        {"session": "s2", "ts": 2.0, "tools": ["sql_ddl"], "primary": "postgresql"},
        {"session": "s2", "ts": 3.0, "tools": ["duckdb_query"], "primary": "duckdb"},
        {"session": "s3", "ts": 1.0, "tools": ["uv_init"], "primary": "astral"},
        {"session": "s3", "ts": 2.0, "tools": ["sql_ddl"], "primary": "postgresql"},
        {"session": "s3", "ts": 3.0, "tools": ["duckdb_query"], "primary": "duckdb"},
        {"session": "s4", "ts": 1.0, "tools": ["uv_init"], "primary": "astral"},
        {"session": "s4", "ts": 2.0, "tools": ["sql_ddl"], "primary": "postgresql"},
    ]

    dag = scheduler.fit_from_traces(mock_traces)
    assert dag["fitted"] is True
    assert len(dag["edges"]) > 0
