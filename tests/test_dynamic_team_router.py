"""Unit tests for RiemannianTeamRouter and Dynamic Expert Morphing."""

import numpy as np
import pytest
import torch

from gnn_experiment.dynamic_team_router import RiemannianTeamRouter
from gnn_experiment.novel_peft import FoldableExpert


def create_dummy_expert(name: str, dim: int = 64, rank: int = 8) -> FoldableExpert:
    """Creates a dummy FoldableExpert for testing."""
    factors = {
        "model.layers.0.mlp.down_proj.weight": (torch.randn(dim, rank), torch.randn(rank, dim)),
        "model.layers.0.mlp.gate_proj.weight": (torch.randn(dim, rank), torch.randn(rank, dim)),
    }
    return FoldableExpert(factors, scaling=16.0, name=name)


def test_riemannian_team_router_initialization():
    """Verify router initializes and computes internal distance matrix."""
    domains = ["astral", "postgresql", "duckdb", "financial"]
    experts = {d: create_dummy_expert(d) for d in domains}

    router = RiemannianTeamRouter(experts=experts, domains=domains)
    assert router.dist_mat.shape == (4, 4)
    # Diagonal must be 0
    assert np.allclose(np.diag(router.dist_mat), 0.0, atol=1e-4)
    # Symmetric
    assert np.allclose(router.dist_mat, router.dist_mat.T, atol=1e-4)


def test_team_selection_prefers_harmonic_synergy():
    """Verify router prefers geometrically harmonious team over distant outliers."""
    domains = ["astral", "postgresql", "duckdb", "financial"]
    experts = {d: create_dummy_expert(d) for d in domains}

    # Synthetic distance matrix where (postgresql, duckdb) is very close (0.1)
    # but financial is very distant from both (0.9)
    dist_mat = np.array([
        [0.0, 0.3, 0.3, 0.8],
        [0.3, 0.0, 0.1, 0.9],
        [0.3, 0.1, 0.0, 0.9],
        [0.8, 0.9, 0.9, 0.0],
    ])

    router = RiemannianTeamRouter(experts=experts, distance_matrix=dist_mat, domains=domains)

    # Both postgresql, duckdb, and financial have high relevance
    candidate_scores = {
        "postgresql": 0.95,
        "duckdb": 0.90,
        "financial": 0.92,
    }

    team, meta = router.select_team(candidate_scores, max_team_size=2, harmony_penalty=1.0)

    # The router should select [postgresql, duckdb] because financial adds 0.9 penalty
    assert set(team) == {"postgresql", "duckdb"}
    assert meta["intra_team_distance"] == 0.1


def test_team_selection_max_size_constraint():
    """Verify router respects max_team_size constraint."""
    domains = ["astral", "postgresql", "duckdb", "financial", "python_modern", "python_web"]
    experts = {d: create_dummy_expert(d) for d in domains}
    router = RiemannianTeamRouter(experts=experts, domains=domains)

    candidate_scores = {d: 0.9 for d in domains}
    team, meta = router.select_team(candidate_scores, max_team_size=3)

    assert len(team) <= 3
