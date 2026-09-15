"""Unit tests for RiemannianTeamRouter and Dynamic Expert Morphing."""

from __future__ import annotations

import numpy as np
import torch
from dynamic_team_router import RiemannianTeamRouter
from novel_peft import FoldableExpert


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


def test_secondary_expert_must_earn_its_seat():
    """A second expert joins on its OWN relevance, not because the pair looks close."""
    domains = ["astral", "postgresql", "duckdb", "financial"]
    experts = {d: create_dummy_expert(d) for d in domains}
    router = RiemannianTeamRouter(experts=experts, domains=domains, distance_matrix=np.zeros((4, 4)))

    # A clear primary and a genuinely relevant second -> both fold.
    team, meta = router.select_team({"postgresql": 0.95, "duckdb": 0.90, "financial": 0.05}, max_team_size=2)
    assert set(team) == {"postgresql", "duckdb"}

    # Same primary, but nothing else is relevant -> solo, not a padded team.
    team, _ = router.select_team({"postgresql": 0.95, "duckdb": 0.05, "financial": 0.05}, max_team_size=2)
    assert team == ["postgresql"]


def test_low_relevance_generalist_is_not_welded_to_every_team():
    """Verify low relevance generalist is not added when not relevant."""
    domains = ["astral", "postgresql", "python_modern"]
    experts = {d: create_dummy_expert(d) for d in domains}
    router = RiemannianTeamRouter(experts=experts, domains=domains, distance_matrix=np.zeros((3, 3)))
    team, _ = router.select_team({"astral": 0.90, "python_modern": 0.20, "postgresql": 0.05}, max_team_size=2)
    assert team == ["astral"]


def test_primary_folds_even_when_everything_is_weak():
    domains = ["astral", "postgresql"]
    experts = {d: create_dummy_expert(d) for d in domains}
    router = RiemannianTeamRouter(experts=experts, domains=domains, distance_matrix=np.zeros((2, 2)))
    team, _ = router.select_team({"astral": 0.05, "postgresql": 0.05})
    assert len(team) == 1


def test_team_selection_max_size_constraint():
    """Verify router respects max_team_size constraint."""
    domains = ["astral", "postgresql", "duckdb", "financial", "python_modern", "python_web"]
    experts = {d: create_dummy_expert(d) for d in domains}
    router = RiemannianTeamRouter(experts=experts, domains=domains)

    candidate_scores = {d: 0.9 for d in domains}
    team, meta = router.select_team(candidate_scores, max_team_size=3)

    assert len(team) <= 3
