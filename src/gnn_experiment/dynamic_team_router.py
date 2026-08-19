"""Dynamic Expert Team Router using Riemannian Manifold Geodesics & Fréchet Blending.

References:
- Chapter 3: §3.5 (Log of a Covariance Matrix & Riemannian Geodesics)
- Chapter 8: §8.1.4 (Ledoit-Wolf Optimal Shrinkage)
- Bhatia (2007): Positive Definite Matrices
- Decision §55: Riemannian 6x6 Domain Distance Matrix

Provides:
1. RiemannianTeamRouter: Selects optimal cohesive K-expert teams by balancing prompt relevance
   against intra-team geodesic distance d_R on the manifold.
2. Fast Weight Morphing: Smoothly swaps / blends folded expert adapters in base model modules in ~1 ms.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gnn_experiment.novel_peft import FoldableExpert, WeightFoldingEngine
from gnn_experiment.riemannian_covariance import (
    ledoit_wolf_shrinkage,
    log_euclidean_distance,
    riemannian_affine_invariant_distance,
)



class RiemannianTeamRouter:
    """Dynamically selects synergistic teams of experts using precomputed Riemannian distances."""

    def __init__(
        self,
        experts: dict[str, FoldableExpert],
        distance_matrix: np.ndarray | None = None,
        domains: list[str] | None = None,
    ):
        self.experts = experts
        self.domains = domains or list(experts.keys())
        self.domain_to_idx = {d: i for i, d in enumerate(self.domains)}

        if distance_matrix is not None:
            self.dist_mat = distance_matrix
        else:
            self.dist_mat = self._compute_default_distance_matrix()

    def _compute_default_distance_matrix(self) -> np.ndarray:
        """Computes pairwise Riemannian AIRM distances on down_proj layer 0 Gramians."""
        N = len(self.domains)
        mat = np.zeros((N, N))

        # Find first shared module
        first_exp = self.experts[self.domains[0]]
        sample_key = next((k for k in first_exp.factors.keys() if "down_proj" in k), list(first_exp.factors.keys())[0])

        gramians = []
        for d in self.domains:
            u, v = self.experts[d].factors[sample_key]
            s = self.experts[d].scaling
            G = (s**2) * (v.float() @ v.float().T) + (u.float().T @ u.float())
            G_lw, _ = ledoit_wolf_shrinkage(G)
            G_lw = G_lw + 1e-5 * torch.eye(G.shape[0], dtype=G.dtype)
            gramians.append(G_lw)

        for i in range(N):
            for j in range(i + 1, N):
                d_r = riemannian_affine_invariant_distance(gramians[i], gramians[j])
                mat[i, j] = d_r
                mat[j, i] = d_r

        return mat

    def team_intra_distance(self, team: list[str]) -> float:
        """Computes mean pairwise Riemannian distance between all pairs in the team."""
        if len(team) <= 1:
            return 0.0
        pairs = list(itertools.combinations(team, 2))
        total_dist = sum(self.dist_mat[self.domain_to_idx[d1], self.domain_to_idx[d2]] for d1, d2 in pairs)
        return float(total_dist / len(pairs))

    def select_team(
        self,
        candidate_scores: dict[str, float],
        max_team_size: int = 2,
        harmony_penalty: float = 0.5,
    ) -> tuple[list[str], dict[str, Any]]:
        """Selects the optimal team maximizing prompt relevance while penalizing Riemannian manifold distance.

        Score(Team) = sum(Relevance(e)) - harmony_penalty * mean(d_R(e_i, e_j))

        Args:
            candidate_scores: Dict mapping domain name -> relevance score in [0, 1].
            max_team_size: Maximum number of active experts in the team (typically 2 or 3).
            harmony_penalty: Weight penalty for intra-team geometric distance.

        Returns:
            (best_team, routing_metadata)
        """
        candidates = [d for d in candidate_scores.keys() if d in self.domain_to_idx]
        if not candidates:
            return [], {"score": 0.0, "candidates": candidate_scores}

        best_team = []
        best_score = -float("inf")
        evaluated_teams = []

        # Evaluate candidate combinations of sizes 1 up to max_team_size
        for k in range(1, min(len(candidates), max_team_size) + 1):
            for combo in itertools.combinations(candidates, k):
                team = list(combo)
                rel_score = sum(candidate_scores[d] for d in team)
                dist_penalty = self.team_intra_distance(team) if len(team) > 1 else 0.0
                total_score = rel_score - harmony_penalty * dist_penalty

                evaluated_teams.append({
                    "team": team,
                    "relevance": rel_score,
                    "distance_penalty": dist_penalty,
                    "total_score": total_score,
                })

                if total_score > best_score:
                    best_score = total_score
                    best_team = team

        meta = {
            "selected_team": best_team,
            "team_score": best_score,
            "intra_team_distance": self.team_intra_distance(best_team),
            "candidates": candidate_scores,
            "all_evaluated_teams": evaluated_teams,
        }
        return best_team, meta

    def morph_stack(
        self,
        folding_engine: WeightFoldingEngine,
        current_team: list[str],
        target_team: list[str],
        scale_mode: str = "surgical",
    ) -> dict[str, Any]:
        """Smoothly transitions the active folded expert stack in the base model via WeightFoldingEngine.

        If current_team == target_team, does nothing (0 ms).
        Otherwise, folds new target team using surgical stacking.
        """
        t0 = time.time()
        if set(current_team) == set(target_team) and folding_engine.active is not None:
            return {"elapsed_ms": 0.0, "status": "unchanged", "active_team": target_team}

        # Activate target team
        target_experts = [self.experts[d] for d in target_team if d in self.experts]
        if target_experts:
            folding_engine.activate_many(target_experts, scale_mode=scale_mode)
        else:
            folding_engine.restore()

        elapsed_ms = (time.time() - t0) * 1000.0
        return {
            "elapsed_ms": elapsed_ms,
            "status": "morphed",
            "previous_team": current_team,
            "active_team": target_team,
            "folded_experts_count": len(target_experts),
        }

