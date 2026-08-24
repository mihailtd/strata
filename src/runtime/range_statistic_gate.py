"""Single-Pass Range Statistic Speculative Early-Exit Gating.

Grounding: Chapter 8 (Outlier Detection Based on Range Statistic Empirical Evaluation
and Comparisons, Dania Dallah, Hana Sulieman, Ayman Alzaatreh).

Mathematical Formulation:
1. Extreme Range Spread Statistic:
   R_M = max_{j=1..M} z_(j) - min_{j=1..M} z_(j)
   computed over top-M candidate logits z_(1) >= z_(2) >= ... >= z_(M).
2. Studentized Range Statistic:
   q_M = R_M / s_IQR
   where s_IQR = (Q_3 - Q_1) / 1.349 provides scale-invariant outlier estimation
   without 2-pass variance computation.
3. Decision Boundary:
   If R_M < tau_range:
     Logit distribution over top candidates is flat/uniform (high uncertainty).
     -> Abort speculative draft chain immediately to avoid generating junk tokens.
   Else:
     Top candidate dominates with high margin.
     -> Continue multi-token speculative drafting.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RangeStatisticGate(nn.Module):
    """Zero-overhead single-pass range statistic early-exit gate for speculative drafting."""

    def __init__(
        self,
        top_m: int = 8,
        threshold: float = 3.5,
        mode: Literal["extreme_range", "studentized_range", "iqr_spread"] = "extreme_range",
    ) -> None:
        super().__init__()
        if top_m < 2:
            raise ValueError(f"top_m must be at least 2, got {top_m}")
        self.top_m = top_m
        self.threshold = threshold
        self.mode = mode

    @torch.no_grad()
    def compute_range(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute the range statistic on top-M candidate logits.

        Args:
            logits: Logit tensor of shape (..., V)

        Returns:
            Range statistic tensor of shape (...)
        """
        # Extract top-M logits in a single fast partial sort
        # Shape: (..., top_m)
        top_vals, _ = torch.topk(logits, k=self.top_m, dim=-1, largest=True, sorted=True)

        if self.mode == "extreme_range":
            # R = max(top_m) - min(top_m) = top_vals[..., 0] - top_vals[..., -1]
            return top_vals[..., 0] - top_vals[..., -1]

        elif self.mode == "iqr_spread":
            # Interquartile range spread over top candidates
            q75_idx = max(0, int(0.25 * self.top_m))
            q25_idx = min(self.top_m - 1, int(0.75 * self.top_m))
            iqr = top_vals[..., q75_idx] - top_vals[..., q25_idx]
            return iqr

        elif self.mode == "studentized_range":
            # Studentized range: extreme range normalized by robust IQR standard deviation
            r_ext = top_vals[..., 0] - top_vals[..., -1]
            q75_idx = max(0, int(0.25 * self.top_m))
            q25_idx = min(self.top_m - 1, int(0.75 * self.top_m))
            iqr = (top_vals[..., q75_idx] - top_vals[..., q25_idx]).clamp(min=1e-5)
            s_est = iqr / 1.349
            return r_ext / s_est

        raise ValueError(f"Unknown range statistic mode: {self.mode}")

    @torch.no_grad()
    def should_early_exit(self, logits: torch.Tensor, step_idx: int = 0) -> Tuple[bool, float]:
        """Decides whether to abort the speculative draft chain on this token.

        Args:
            logits: Logit tensor of shape (1, 1, V) or (1, V)
            step_idx: Step index within the speculative window (0..K-1)

        Returns:
            Tuple of (should_abort: bool, range_value: float)
        """
        flat_logits = logits.view(-1, logits.shape[-1])
        range_val = self.compute_range(flat_logits).squeeze().item()

        # If range spread is below critical threshold, distribution is flat -> abort
        should_abort = bool(range_val < self.threshold)
        return should_abort, float(range_val)

    @classmethod
    def calibrate_threshold(
        cls,
        logits_history: torch.Tensor,
        verification_matches: torch.Tensor,
        top_m: int = 8,
        target_precision: float = 0.90,
        min_support: int = 10,
    ) -> float:
        """Calibrates optimal range threshold against empirical verification acceptance.

        Finds the smallest range cutoff tau_R such that drafting is permitted only
        when empirical acceptance probability >= target_precision with sufficient support.
        """
        gate = cls(top_m=top_m, mode="extreme_range")
        ranges = gate.compute_range(logits_history).view(-1)
        matches = verification_matches.view(-1).float()

        # Sweep candidate percentiles from 10th to 90th percentile
        sorted_ranges, _ = torch.sort(ranges)
        n = len(sorted_ranges)

        best_tau = float(sorted_ranges[int(0.5 * n)].item())
        min_samples = max(min_support, int(0.05 * n))

        # Check candidate cutoffs from lower to higher
        step = max(1, n // 200)
        for i in range(0, n, step):
            tau = float(sorted_ranges[i].item())
            retained_mask = ranges >= tau
            count = int(retained_mask.sum().item())
            if count >= min_samples:
                prec = float(matches[retained_mask].mean().item())
                if prec >= target_precision:
                    best_tau = tau
                    break

        return best_tau
