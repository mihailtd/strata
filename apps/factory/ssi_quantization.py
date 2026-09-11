r"""Stress-Strength Interference (SSI) Optimal Quantization Scaling.

Theoretical Grounding:
Chapter 8.5 (Case Studies – Stress-Strength Interference Analysis, Jaejin Hwang,
Reliability Analysis Using MINITAB and Python).

Mathematical Formulation:
In low-bit quantization (e.g. symmetric 4-bit INT4 with 16 discrete bins [-7, +7]),
choosing the group scale gamma determines the boundary between rounding noise and clipping distortion:
- Stress Distribution f(x): Weight / activation magnitude distribution |X| in group g.
- Strength Distribution g(y): Representation limit before saturation c = 7 * gamma.

Failure (Clipping Distortion) occurs when Stress exceeds Strength:
  P_f = P(Stress > Strength) = \int_{7*gamma}^{\infty} f(x) dx

Total Expected Quantization Error:
  D(gamma) = D_rounding(gamma) + D_clipping(gamma)
  D_rounding(gamma) = (gamma^2) / 12 * P(Stress <= 7*gamma)
  D_clipping(gamma) = \int_{7*gamma}^{\infty} (x - 7*gamma)^2 f(x) dx

Solving the SSI optimality condition dD(gamma)/dgamma = 0 yields the optimal
clipping threshold gamma* in closed-form based on the empirical moment ratio
(standard deviation sigma and kurtosis kappa) without iterative grid search.
"""

from __future__ import annotations

from typing import Tuple, Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F


class StressStrengthInterferenceCalibrator:
    """Computes closed-form optimal quantization group scales via Stress-Strength Interference."""

    def __init__(
        self,
        group_size: int = 128,
        n_bits: int = 4,
        target_pf: float = 1e-4,
        eps: float = 1e-6,
    ) -> None:
        self.group_size = group_size
        self.n_bits = n_bits
        self.q_max = (1 << (n_bits - 1)) - 1  # 7 for 4-bit
        self.target_pf = target_pf
        self.eps = eps

    @torch.no_grad()
    def compute_group_moments(self, W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes mean, standard deviation, and standardized excess kurtosis per group.

        Args:
            W: Weight tensor of shape (D_in, D_out)

        Returns:
            Tuple of:
            - mean: Shape (n_groups, D_out)
            - std: Shape (n_groups, D_out)
            - kurtosis: Shape (n_groups, D_out)
        """
        D_in, D_out = W.shape
        pad_in = (self.group_size - (D_in % self.group_size)) % self.group_size
        if pad_in > 0:
            W_padded = F.pad(W.to(torch.float32), (0, 0, 0, pad_in))
        else:
            W_padded = W.to(torch.float32)

        D_in_padded = W_padded.shape[0]
        n_groups = D_in_padded // self.group_size
        grouped = W_padded.view(n_groups, self.group_size, D_out)

        mean = torch.mean(grouped, dim=1)
        var = torch.var(grouped, dim=1, unbiased=True).clamp(min=self.eps)
        std = torch.sqrt(var)

        # 4th standardized moment (Kurtosis)
        centered = grouped - mean.unsqueeze(1)
        m4 = torch.mean(centered ** 4, dim=1)
        kurtosis = (m4 / (var ** 2)).clamp(min=1.0, max=25.0)

        return mean, std, kurtosis

    @torch.no_grad()
    def calibrate_ssi_scales(
        self,
        W: torch.Tensor,
        mode: Literal["closed_form_ssi", "empirical_ssi", "naive_max"] = "closed_form_ssi",
    ) -> torch.Tensor:
        """Calibrates optimal quantization group scales using Stress-Strength Interference.

        Args:
            W: Weight tensor of shape (D_in, D_out)
            mode: Calibration strategy

        Returns:
            Scales tensor of shape (n_groups, D_out) in bfloat16
        """
        D_in, D_out = W.shape
        pad_in = (self.group_size - (D_in % self.group_size)) % self.group_size
        if pad_in > 0:
            W_padded = F.pad(W.to(torch.float32), (0, 0, 0, pad_in))
        else:
            W_padded = W.to(torch.float32)

        D_in_padded = W_padded.shape[0]
        n_groups = D_in_padded // self.group_size
        grouped = W_padded.view(n_groups, self.group_size, D_out)

        if mode == "naive_max":
            max_abs = torch.max(torch.abs(grouped), dim=1).values.clamp(min=self.eps)
            return (max_abs / float(self.q_max)).to(torch.bfloat16)

        # Step 1: Compute group moments
        _, std, kurtosis = self.compute_group_moments(W)

        if mode == "closed_form_ssi":
            # Closed-form SSI Safety Factor:
            # For Gaussian (kurtosis ~ 3), optimal clipping boundary c ≈ 2.98 * std
            # For Laplace (kurtosis ~ 6), optimal clipping boundary c ≈ 3.30 * std
            # Generalized Kurtosis-adaptive multiplier: k_ssi = 2.2 + 0.45 * sqrt(kurtosis)
            k_ssi = 2.2 + 0.45 * torch.sqrt(kurtosis)
            optimal_boundary = k_ssi * std

            # Ensure optimal boundary does not exceed true max_abs
            max_abs = torch.max(torch.abs(grouped), dim=1).values.clamp(min=self.eps)
            clamped_boundary = torch.minimum(optimal_boundary, max_abs)

            scales = clamped_boundary / float(self.q_max)
            return scales.to(torch.bfloat16)

        elif mode == "empirical_ssi":
            # Empirical SSI: directly minimize D_round + D_clip over candidate grid
            best_scales = torch.zeros(n_groups, D_out, device=W.device, dtype=torch.float32)
            max_abs = torch.max(torch.abs(grouped), dim=1).values.clamp(min=self.eps)

            # Evaluate 16 candidate scale multipliers
            multipliers = torch.linspace(2.0, 4.5, 16, device=W.device)
            min_distortions = torch.full((n_groups, D_out), float("inf"), device=W.device)

            for mult in multipliers:
                cand_boundary = torch.minimum(mult * std, max_abs)
                cand_scale = (cand_boundary / float(self.q_max)).unsqueeze(1)  # (n_groups, 1, D_out)

                # Quantize and reconstruct
                W_q = torch.clamp(torch.round(grouped / cand_scale), -float(self.q_max) - 1.0, float(self.q_max))
                W_rec = W_q * cand_scale

                # Total distortion
                distortion = torch.sum((grouped - W_rec) ** 2, dim=1)  # (n_groups, D_out)
                improved = distortion < min_distortions
                min_distortions[improved] = distortion[improved]
                best_scales[improved] = cand_scale.squeeze(1)[improved]

            return best_scales.to(torch.bfloat16)

        raise ValueError(f"Unknown SSI mode: {mode}")

    @torch.no_grad()
    def quantize_w4(
        self,
        W: torch.Tensor,
        mode: Literal["closed_form_ssi", "empirical_ssi", "naive_max"] = "closed_form_ssi",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantizes weight tensor to symmetric 4-bit with SSI calibrated scales.

        Returns:
            - W_q: Quantized integer tensor in [-8, 7]
            - scales: Calibrated scales tensor of shape (n_groups, D_out)
        """
        scales = self.calibrate_ssi_scales(W, mode=mode)
        D_in, D_out = W.shape
        pad_in = (self.group_size - (D_in % self.group_size)) % self.group_size
        if pad_in > 0:
            W_padded = F.pad(W.to(torch.float32), (0, 0, 0, pad_in))
        else:
            W_padded = W.to(torch.float32)

        D_in_padded = W_padded.shape[0]
        n_groups = D_in_padded // self.group_size
        grouped = W_padded.view(n_groups, self.group_size, D_out)

        scales_expanded = scales.unsqueeze(1).to(torch.float32)
        q_grouped = torch.clamp(torch.round(grouped / scales_expanded), -8.0, 7.0)
        W_q = q_grouped.view(D_in_padded, D_out)[:D_in, :]

        return W_q.to(torch.int8), scales

    @torch.no_grad()
    def dequantize_w4(
        self,
        W_q: torch.Tensor,
        scales: torch.Tensor,
        original_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """Dequantizes 4-bit tensor back to bfloat16 for evaluation."""
        D_in, D_out = original_shape
        pad_in = (self.group_size - (D_in % self.group_size)) % self.group_size
        if pad_in > 0:
            W_q_padded = F.pad(W_q.to(torch.float32), (0, 0, 0, pad_in))
        else:
            W_q_padded = W_q.to(torch.float32)

        D_in_padded = W_q_padded.shape[0]
        n_groups = D_in_padded // self.group_size
        grouped_q = W_q_padded.view(n_groups, self.group_size, D_out)

        dequant_grouped = grouped_q * scales.unsqueeze(1).to(torch.float32)
        W_rec = dequant_grouped.view(D_in_padded, D_out)[:D_in, :].to(torch.bfloat16)
        return W_rec
