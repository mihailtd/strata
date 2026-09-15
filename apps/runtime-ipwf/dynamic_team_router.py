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
from typing import Any

import numpy as np
import torch
from novel_peft import FoldableExpert, WeightFoldingEngine
from riemannian_covariance import (
    joint_subspace_operators,
    riemannian_affine_invariant_distance,
)


class RenkoBrickSmoother:
    """Filters high-frequency latent noise during continuous streaming generation.

    Tracks accumulated displacement in Riemannian latent space. Only emits a routing
    re-classification event when displacement breaks a discrete Renko Brick Boundary.
    """

    def __init__(self, epsilon_box: float = 5.0):
        self.epsilon_box = epsilon_box
        self.last_brick: torch.Tensor | None = None
        self.accumulated_displacement: float = 0.0

    def step(self, h_t: torch.Tensor) -> bool:
        """Returns True if a routing re-classification event should be emitted."""
        if self.last_brick is None:
            self.last_brick = h_t.detach().clone()
            return True  # Always trigger on the very first token

        displacement = torch.norm(h_t - self.last_brick, p=2).item()
        self.accumulated_displacement += displacement
        self.last_brick = h_t.detach().clone()

        if self.accumulated_displacement >= self.epsilon_box:
            self.accumulated_displacement = 0.0
            return True
        return False


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
        """Computes pairwise Riemannian AIRM distances on down_proj layer 0 operators."""
        N = len(self.domains)
        mat = np.zeros((N, N))

        # Find first shared module
        first_exp = self.experts[self.domains[0]]
        sample_key = next((k for k in first_exp.factors if "down_proj" in k), list(first_exp.factors.keys())[0])

        for i in range(N):
            da = self.domains[i]
            ua, va = self.experts[da].factors[sample_key]
            sa = self.experts[da].scaling
            for j in range(i + 1, N):
                db = self.domains[j]
                ub, vb = self.experts[db].factors[sample_key]
                sb = self.experts[db].scaling

                Sa, Sb, _ = joint_subspace_operators(
                    ua.float().cpu(),
                    va.float().cpu(),
                    sa,
                    ub.float().cpu(),
                    vb.float().cpu(),
                    sb,
                    delta=0.05,
                )
                d_ij = riemannian_affine_invariant_distance(Sa, Sb)
                mat[i, j] = d_ij
                mat[j, i] = d_ij

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
        min_secondary_relevance: float = 0.45,
    ) -> tuple[list[str], dict[str, Any]]:
        """Picks the highest-relevance expert, plus any others that clear a bar.

        WHY THE GEODESIC PENALTY IS GONE
        --------------------------------
        The rule used to be Score(Team) = sum(relevance) - penalty * mean(d_R).
        DECISIONS.md §59 measured that term as inert: swapping the distance matrix
        for a CONSTANT changed the selected team in 0 of 4000 random relevance
        draws, because d_R spans 0.8% of its mean across every pair. With
        len(team)==1 exempted from the penalty, all it ever did was impose a fixed
        cost on adding a second expert -- a size penalty wearing a geometry
        costume.

        It was also a live footgun. `harmony_penalty=0.5` was tuned against the
        OLD broken metric's ~0.26 scale; the corrected metric returns ~14.2, so
        0.5 * 14.2 = 7.1 against a maximum relevance gain of ~1.0 would have made
        every team a solo team, silently.

        So the size penalty is now stated as what it is: a second expert joins
        only if IT is relevant on its own (>= 0.45 and >= 50% of primary), not
        because the pair looks harmonious. This stops a generalist from being
        polluted into pure SQL or database queries.

        Args:
            candidate_scores: domain -> relevance in [0, 1].
            max_team_size: cap on simultaneously folded experts.
            min_secondary_relevance: bar a non-primary expert must clear to join.

        Returns:
            (best_team, routing_metadata)
        """
        candidates = [d for d in candidate_scores if d in self.domain_to_idx]
        if not candidates:
            return [], {"score": 0.0, "candidates": candidate_scores}

        ranked = sorted(candidates, key=lambda d: -candidate_scores[d])
        primary_score = candidate_scores[ranked[0]]
        best_team = [ranked[0]]  # the primary always folds
        for d in ranked[1:max_team_size]:
            score_d = candidate_scores[d]
            if score_d >= min_secondary_relevance and score_d >= (0.50 * primary_score):
                best_team.append(d)

        meta = {
            "selected_team": best_team,
            "team_score": sum(candidate_scores[d] for d in best_team),
            "intra_team_distance": self.team_intra_distance(best_team),
            "candidates": candidate_scores,
            "min_secondary_relevance": min_secondary_relevance,
            "runners_up": [(d, candidate_scores[d]) for d in ranked[1:4]],
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
