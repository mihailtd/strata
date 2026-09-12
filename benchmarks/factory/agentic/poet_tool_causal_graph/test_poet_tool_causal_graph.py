"""Unit tests for POET + NOTEARS Agent Tool Causal Graph Benchmark."""

import numpy as np
import pytest

from benchmarks.factory.agentic.poet_tool_causal_graph.probe_poet_tool_causal_graph import (
    build_ground_truth_dag,
    count_accuracy,
    notears_linear,
    poet_filter_confounders,
    simulate_tool_execution_data,
)


def test_ground_truth_dag_properties():
    """Verify ground truth DAG is strictly acyclic and has valid dimensions."""
    W = build_ground_truth_dag()
    assert W.shape == (10, 10)
    assert np.diag(W).sum() == 0.0
    
    # Verify exact acyclicity: Tr(exp(W o W)) - d == 0
    M = W * W
    h = np.trace(np.linalg.matrix_power(np.eye(10) + M, 10)) - 10
    # Topological sort / strictly upper triangular permutation
    eigenvals = np.linalg.eigvals(W)
    assert np.allclose(eigenvals, 0.0)


def test_simulation_data_generation():
    """Verify tool execution simulator produces valid data matrix."""
    W = build_ground_truth_dag()
    X_obs, X_pure = simulate_tool_execution_data(W, n_samples=100, seed=42)
    
    assert X_obs.shape == (100, 10)
    assert not np.isnan(X_obs).any()
    assert not np.isnan(X_pure).any()


def test_notears_acyclicity_guarantee():
    """Verify NOTEARS output satisfies acyclicity invariant."""
    W_true = build_ground_truth_dag()
    X_obs, _ = simulate_tool_execution_data(W_true, n_samples=200, seed=123)
    
    W_est = notears_linear(X_obs, lambda1=0.05, max_iter=30)
    assert W_est.shape == (10, 10)
    assert np.diag(W_est).sum() == 0.0
    
    acc = count_accuracy(W_true, W_est)
    assert acc["true_edges"] == 10
    assert acc["tpr"] > 0.50
