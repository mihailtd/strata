"""Single-Pass Range Statistic, Weibull Hazard & Bollinger Band Volatility Speculative Early-Exit Gating.

Theoretical Grounding:
1. Chapter 8 (Outlier Detection Based on Range Statistic Empirical Evaluation
   and Comparisons, Dania Dallah, Hana Sulieman, Ayman Alzaatreh):
   - Fast extreme range spread: R_M = max_{j=1..M} z_(j) - min_{j=1..M} z_(j) in O(1) registers.
2. Chapter 3 (Lifetime Distributions – Weibull Distribution & Failure Rate,
   Jaejin Hwang, Reliability Analysis Using MINITAB and Python):
   - Speculative draft failure exhibits wear-out characteristic (beta > 1):
     h(k; beta, eta) = (beta / eta) * ((k + 1) / eta)^(beta - 1)
   - Spatio-temporal dynamic threshold:
     tau_eff(k) = tau_0 * [1 + gamma_weibull * h(k)]
3. Chapter 5 (Volatility Indicators – Bollinger Bands & Average True Range,
   Algorithmic Trading & Technical Indicators):
   - Track rolling top-1/top-2 logit margin spread: Delta l_t = z_(1) - z_(2)
   - Rolling Exponential Moving Average and Variance:
     mu_t = (1 - alpha) * mu_{t-1} + alpha * Delta l_t
     sigma_t = sqrt((1 - alpha) * sigma_{t-1}^2 + alpha * (Delta l_t - mu_t)^2)
     Lower Band_t = mu_t - k_bb * sigma_t
   - True Range & ATR:
     TR_t = max(|Delta l_t - Delta l_{t-1}|, |Delta l_t - Lower Band_{t-1}|)
     ATR_t = (1 - alpha_atr) * ATR_{t-1} + alpha_atr * TR_t
   - Volatility Deficit Penalty:
     VolPenalty = max(0.0, (Lower Band_t - Delta l_t) / (ATR_t + 1e-6))
   - Tri-Modal Composite Dynamic Threshold:
     tau_eff(k, Delta l_t) = tau_0 * [1 + gamma_weibull * h(k) + gamma_bollinger * VolPenalty]

As draft depth k increases or local token confidence experiences a volatility breakdown,
the confidence bar automatically tightens, pruning doomed tail draft tokens and eliminating wasted verification passes.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import torch
import torch.nn as nn


class RangeOutput(tuple):
    """Container allowing both tuple unpacking (range_tensor, top_vals) and direct tensor operations."""

    def __new__(cls, range_val: torch.Tensor, top_vals: torch.Tensor):
        return super().__new__(cls, (range_val, top_vals))

    @property
    def range(self) -> torch.Tensor:
        return self[0]

    @property
    def top_vals(self) -> torch.Tensor:
        return self[1]

    def item(self):
        return self[0].item()

    def squeeze(self, *args, **kwargs):
        return self[0].squeeze(*args, **kwargs)


class RangeStatisticGate(nn.Module):
    """Zero-overhead single-pass range statistic, Weibull hazard & Bollinger Band early-exit gate for speculative drafting."""

    def __init__(
        self,
        top_m: int = 8,
        threshold: float = 3.5,
        mode: Literal["extreme_range", "studentized_range", "iqr_spread"] = "extreme_range",
        weibull_hazard_enabled: bool = True,
        weibull_beta: float = 2.2,
        weibull_eta: float = 4.0,
        weibull_gamma: float = 0.6,
        bollinger_bands_enabled: bool = True,
        bollinger_k: float = 2.0,
        bollinger_alpha: float = 0.25,
        bollinger_gamma: float = 0.50,
        atr_alpha: float = 0.20,
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
        if bollinger_k < 0:
            raise ValueError(f"bollinger_k must be >= 0, got {bollinger_k}")

        self.top_m = top_m
        self.threshold = threshold
        self.mode = mode
        self.weibull_hazard_enabled = weibull_hazard_enabled
        self.weibull_beta = weibull_beta
        self.weibull_eta = weibull_eta
        self.weibull_gamma = weibull_gamma

        # Bollinger Bands & ATR state
        self.bollinger_bands_enabled = bollinger_bands_enabled
        self.bollinger_k = bollinger_k
        self.bollinger_alpha = bollinger_alpha
        self.bollinger_gamma = bollinger_gamma
        self.atr_alpha = atr_alpha

        # Rolling state registers (O(1) in-memory)
        self.running_mean_spread: float = 3.0
        self.running_var_spread: float = 1.0
        self.running_atr: float = 1.0
        self.last_spread: float | None = None
        self.history_count: int = 0

    @classmethod
    def calibrate_threshold(
        cls,
        logits_history: torch.Tensor,
        verification_matches: torch.Tensor,
        top_m: int = 8,
        target_precision: float = 0.90,
        min_support: int = 10,
    ) -> float:
        """Calibrates optimal baseline range threshold against empirical verification acceptance."""
        gate = cls(top_m=top_m, mode="extreme_range", weibull_hazard_enabled=False, bollinger_bands_enabled=False)
        range_out = gate.compute_range(logits_history)
        ranges = range_out.range.view(-1)
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

    def reset_state(self, initial_spread: float = 3.0) -> None:
        """Resets the rolling volatility indicators for a new generation sequence."""
        self.running_mean_spread = initial_spread
        self.running_var_spread = 1.0
        self.running_atr = 1.0
        self.last_spread = None
        self.history_count = 0

    def compute_hazard_rate(self, step_idx: int) -> float:
        """Computes instantaneous Weibull wear-out hazard rate for draft step k (0-indexed)."""
        if not self.weibull_hazard_enabled:
            return 0.0
        step = step_idx + 1  # 1-indexed lifetime elapsed
        hazard = (self.weibull_beta / self.weibull_eta) * ((step / self.weibull_eta) ** (self.weibull_beta - 1.0))
        return float(hazard)

    def update_volatility(self, top_vals: torch.Tensor) -> tuple[float, float, float, float, float]:
        """Updates rolling Bollinger Bands and ATR based on top-1 vs top-2 logit spread.

        Evaluates the incoming logit spread against the prior rolling volatility regime (pre-update),
        then updates EMA registers for the next step.

        Args:
            top_vals: Tensor of sorted top-M logits of shape (..., top_m)

        Returns:
            Tuple of (spread: float, lower_band: float, upper_band: float, atr: float, vol_penalty: float)
        """
        # top-1 vs top-2 margin spread
        spread = float((top_vals[..., 0] - top_vals[..., 1]).squeeze().item())

        if self.history_count == 0:
            self.running_mean_spread = spread
            self.running_var_spread = 0.5
            self.running_atr = 0.5
            self.last_spread = spread
            self.history_count = 1
            std = math.sqrt(self.running_var_spread)
            lower_band = self.running_mean_spread - self.bollinger_k * std
            upper_band = self.running_mean_spread + self.bollinger_k * std
            return spread, lower_band, upper_band, self.running_atr, 0.0

        # 1. Compute bands from established prior regime
        prior_std = math.sqrt(max(1e-4, self.running_var_spread))
        lower_band = self.running_mean_spread - self.bollinger_k * prior_std
        upper_band = self.running_mean_spread + self.bollinger_k * prior_std

        # 2. Check volatility deficit against established lower band
        vol_penalty = 0.0
        if self.bollinger_bands_enabled and spread < lower_band:
            deficit = lower_band - spread
            vol_penalty = deficit / max(1e-4, self.running_atr)

        # 3. Update rolling EMA registers for next observation
        alpha = self.bollinger_alpha
        diff = spread - self.running_mean_spread
        self.running_mean_spread += alpha * diff
        self.running_var_spread = (1.0 - alpha) * self.running_var_spread + alpha * (diff**2)

        prev_spread = self.last_spread if self.last_spread is not None else spread
        tr = max(abs(spread - prev_spread), 1e-4)
        self.running_atr = (1.0 - self.atr_alpha) * self.running_atr + self.atr_alpha * tr
        self.last_spread = spread
        self.history_count += 1

        return spread, lower_band, upper_band, self.running_atr, vol_penalty

    def get_effective_threshold(self, step_idx: int = 0, vol_penalty: float = 0.0) -> float:
        """Computes dynamic confidence threshold taking elapsed draft horizon and volatility into account."""
        h_k = self.compute_hazard_rate(step_idx)
        weibull_mult = self.weibull_gamma * h_k if self.weibull_hazard_enabled else 0.0
        bollinger_mult = self.bollinger_gamma * vol_penalty if self.bollinger_bands_enabled else 0.0

        tau_eff = self.threshold * (1.0 + weibull_mult + bollinger_mult)
        return float(tau_eff)

    @torch.no_grad()
    def compute_range(self, logits: torch.Tensor) -> RangeOutput:
        """Compute the range statistic and return top-M values in a single partial sort."""
        top_vals, _ = torch.topk(logits, k=self.top_m, dim=-1, largest=True, sorted=True)

        if self.mode == "extreme_range":
            range_val = top_vals[..., 0] - top_vals[..., -1]
        elif self.mode == "iqr_spread":
            q75_idx = max(0, int(0.25 * self.top_m))
            q25_idx = min(self.top_m - 1, int(0.75 * self.top_m))
            range_val = top_vals[..., q75_idx] - top_vals[..., q25_idx]
        elif self.mode == "studentized_range":
            r_ext = top_vals[..., 0] - top_vals[..., -1]
            q75_idx = max(0, int(0.25 * self.top_m))
            q25_idx = min(self.top_m - 1, int(0.75 * self.top_m))
            iqr = (top_vals[..., q75_idx] - top_vals[..., q25_idx]).clamp(min=1e-5)
            s_est = iqr / 1.349
            range_val = r_ext / s_est
        else:
            raise ValueError(f"Unknown range statistic mode: {self.mode}")

        return RangeOutput(range_val, top_vals)

    @torch.no_grad()
    def should_early_exit(self, logits: torch.Tensor, step_idx: int = 0) -> tuple[bool, float, float]:
        """Decides whether to abort the speculative draft chain on this token."""
        flat_logits = logits.view(-1, logits.shape[-1])
        range_tensor, top_vals = self.compute_range(flat_logits)
        range_val = float(range_tensor.squeeze().item())

        spread, lower_band, upper_band, atr, vol_penalty = self.update_volatility(top_vals)
        effective_tau = self.get_effective_threshold(step_idx, vol_penalty)

        # 1. Hard Volatility Breakdown: Margin spread severely collapses below lower band
        hard_vol_breakout = False
        if self.bollinger_bands_enabled and (spread < lower_band - 1.5 * atr):
            hard_vol_breakout = True

        # 2. Dynamic Threshold Check
        threshold_breached = bool(range_val < effective_tau)

        should_abort = hard_vol_breakout or threshold_breached
        return should_abort, range_val, effective_tau

    @torch.no_grad()
    def inspect_decision(self, logits: torch.Tensor, step_idx: int = 0) -> dict[str, Any]:
        """Returns full diagnostic telemetry for research and dashboard telemetry."""
        flat_logits = logits.view(-1, logits.shape[-1])
        range_tensor, top_vals = self.compute_range(flat_logits)
        range_val = float(range_tensor.squeeze().item())

        spread, lower_band, upper_band, atr, vol_penalty = self.update_volatility(top_vals)
        h_k = self.compute_hazard_rate(step_idx)
        effective_tau = self.get_effective_threshold(step_idx, vol_penalty)

        hard_vol_breakout = bool(self.bollinger_bands_enabled and (spread < lower_band - 1.5 * atr))
        threshold_breached = bool(range_val < effective_tau)
        should_abort = hard_vol_breakout or threshold_breached

        reason = "PASSED"
        if hard_vol_breakout:
            reason = "BOLLINGER_HARD_BREAKDOWN"
        elif threshold_breached:
            reason = "RANGE_BELOW_DYNAMIC_THRESHOLD"

        return {
            "should_abort": should_abort,
            "reason": reason,
            "range_val": range_val,
            "top1_top2_spread": spread,
            "lower_band": lower_band,
            "upper_band": upper_band,
            "mean_spread": self.running_mean_spread,
            "atr": atr,
            "vol_penalty": vol_penalty,
            "hazard_rate": h_k,
            "effective_threshold": effective_tau,
            "step_idx": step_idx,
        }
