"""Single-Pass Range Statistic & Weibull Hazard Speculative Early-Exit Gating.

Theoretical Grounding:
1. Chapter 8 (Outlier Detection Based on Range Statistic Empirical Evaluation
   and Comparisons, Dania Dallah, Hana Sulieman, Ayman Alzaatreh):
   - Fast extreme range spread: R_M = max_{j=1..M} z_(j) - min_{j=1..M} z_(j) in O(1) registers.
2. Chapter 3 (Lifetime Distributions – Weibull Distribution & Failure Rate,
   Jaejin Hwang, Reliability Analysis Using MINITAB and Python):
   - Speculative draft failure exhibits wear-out characteristic (beta > 1):
     h(k; beta, eta) = (beta / eta) * ((k + 1) / eta)^(beta - 1)
   - Spatio-temporal dynamic threshold:
     tau_eff(k) = tau_0 * [1 + gamma * h(k)]

As draft depth k increases (e.g. k=4..8), the confidence bar automatically tightens,
pruning doomed tail draft tokens and eliminating wasted verification passes.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RangeStatisticGate(nn.Module):
    """Zero-overhead single-pass range statistic & Weibull hazard early-exit gate for speculative drafting."""

    def __init__(
        self,
        top_m: int = 8,
        threshold: float = 3.5,
        mode: Literal["extreme_range", "studentized_range", "iqr_spread"] = "extreme_range",
        weibull_hazard_enabled: bool = True,
        weibull_beta: float = 2.2,
        weibull_eta: float = 4.0,
        weibull_gamma: float = 0.6,
    ) -> None:
        super().__init__()
        if top_m < 2:
            raise ValueError(f"top_m must be at least 2, got {top_m}")
        if weibull_beta <= 0:
            raise ValueError(f"weibull_beta must be > 0, got {weibull_beta}")
        if weibull_eta <= 0:
            raise ValueError(f"weibull_eta must be > 0, got {weibull_eta}")
        if weibull_gamma < 0:
            raise ValueError(f"weibull_gamma must be >= 0, got {weibull_gamma}")

        self.top_m = top_m
        self.threshold = threshold
        self.mode = mode
        self.weibull_hazard_enabled = weibull_hazard_enabled
        self.weibull_beta = weibull_beta
        self.weibull_eta = weibull_eta
        self.weibull_gamma = weibull_gamma

    def compute_hazard_rate(self, step_idx: int) -> float:
        """Computes instantaneous Weibull wear-out hazard rate for draft step k (0-indexed).

        Args:
            step_idx: Zero-indexed draft step (0 for first draft token, 1 for second, etc.)

        Returns:
            Instantaneous hazard rate h(k) >= 0.0
        """
        step = step_idx + 1  # 1-indexed lifetime elapsed
        hazard = (self.weibull_beta / self.weibull_eta) * (
            (step / self.weibull_eta) ** (self.weibull_beta - 1.0)
        )
        return float(hazard)

    def get_effective_threshold(self, step_idx: int = 0) -> float:
        """Computes dynamic confidence threshold taking elapsed draft horizon into account.

        Args:
            step_idx: Zero-indexed draft step

        Returns:
            Effective threshold tau_eff(k)
        """
        if not self.weibull_hazard_enabled:
            return float(self.threshold)

        h_k = self.compute_hazard_rate(step_idx)
        tau_eff = self.threshold * (1.0 + self.weibull_gamma * h_k)
        return float(tau_eff)

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
    def should_early_exit(self, logits: torch.Tensor, step_idx: int = 0) -> Tuple[bool, float, float]:
        """Decides whether to abort the speculative draft chain on this token.

        Args:
            logits: Logit tensor of shape (1, 1, V) or (1, V)
            step_idx: Step index within the speculative window (0..K-1)

        Returns:
            Tuple of (should_abort: bool, range_value: float, effective_threshold: float)
        """
        flat_logits = logits.view(-1, logits.shape[-1])
        range_val = float(self.compute_range(flat_logits).squeeze().item())
        effective_tau = self.get_effective_threshold(step_idx)

        # If range spread is below dynamic threshold, abort
        should_abort = bool(range_val < effective_tau)
        return should_abort, range_val, effective_tau

    @classmethod
    def calibrate_threshold(
        cls,
        logits_history: torch.Tensor,
        verification_matches: torch.Tensor,
        top_m: int = 8,
        target_precision: float = 0.90,
        min_support: int = 10,
        weibull_hazard_enabled: bool = True,
        weibull_beta: float = 2.2,
        weibull_eta: float = 4.0,
        weibull_gamma: float = 0.6,
    ) -> float:
        """Calibrates optimal baseline range threshold against empirical verification acceptance."""
        gate = cls(
            top_m=top_m,
            mode="extreme_range",
            weibull_hazard_enabled=weibull_hazard_enabled,
            weibull_beta=weibull_beta,
            weibull_eta=weibull_eta,
            weibull_gamma=weibull_gamma,
        )
        ranges = gate.compute_range(logits_history).view(-1)
        matches = verification_matches.view(-1).float()

        sorted_ranges, _ = torch.sort(ranges)
        n = len(sorted_ranges)

        best_tau = float(sorted_ranges[int(0.5 * n)].item())
        min_samples = max(min_support, int(0.05 * n))

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
