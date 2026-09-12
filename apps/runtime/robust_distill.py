"""Breakdown-Bounded Loss Functions for LLM Distillation and Robust Adaptation.

Grounding: Chapter 4 (The Conditional Breakdown Properties of LAD-LASSO Regression,
Boning Feng, Avi Giloni, Jeffrey S. Simonoff).

Mathematical Formulation:
1. Ordinary Least Squares (L2):
   L_2(e) = 1/2 * e^2  ==>  grad = e  (Unbounded: breakdown point epsilon* = 1/n -> 0)
2. Least Absolute Deviations (L1):
   L_1(e) = |e|        ==>  grad = sign(e)  (Bounded: breakdown point epsilon* = 0.50)
3. Huber Loss (Smooth L1/L2 Hybrid):
   L_delta(e) = { 0.5 * e^2           if |e| <= delta
                { delta * (|e| - 0.5*delta) if |e| > delta
   ==> grad = clip(e, -delta, +delta) (Strictly bounded: ||grad|| <= delta)
4. LAD-LASSO Objective:
   min_W  || Y - X W ||_1 + alpha * || W ||_1
   Provides simultaneous 50% breakdown robustness and sparse low-rank support.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberDistillationLoss(nn.Module):
    """Huber continuous representation loss for student-teacher distillation.

    Guarantees that the gradient with respect to student hidden representations
    is strictly saturated at ||grad|| <= delta, preventing anomalous activation
    spikes or hallucinated synthetic tokens from destabilizing low-rank weights.
    """

    def __init__(
        self,
        delta: float = 1.0,
        reduction: Literal["mean", "sum", "none"] = "mean",
        feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        if delta <= 0:
            raise ValueError(f"Huber delta must be positive, got {delta}")
        self.delta = delta
        self.reduction = reduction
        self.feature_dim = feature_dim

    def forward(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Huber distillation loss between student and teacher hidden tensors.

        Args:
            student_hidden: Tensor of shape (..., D)
            teacher_hidden: Tensor of shape (..., D)
            mask: Optional boolean or float mask of shape (...)

        Returns:
            Scalar loss (or unreduced tensor if reduction='none').
        """
        # Element-wise or feature-vector residual
        residual = student_hidden - teacher_hidden
        abs_err = torch.abs(residual)

        # Quadratic region: |e| <= delta
        # Linear region:    |e| > delta
        quadratic = torch.clamp(abs_err, max=self.delta)
        linear = abs_err - quadratic
        loss = 0.5 * (quadratic**2) + self.delta * linear

        if mask is not None:
            # Broadcast mask across hidden dimensions if necessary
            if mask.dim() < loss.dim():
                mask = mask.unsqueeze(-1)
            loss = loss * mask.to(loss.dtype)
            if self.reduction == "mean":
                denom = mask.sum().clamp(min=1.0) * (student_hidden.shape[-1] if mask.dim() < loss.dim() else 1.0)
                return loss.sum() / denom

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

    @torch.no_grad()
    def gradient_bound(self) -> float:
        """Returns the theoretical maximum element-wise gradient norm."""
        return float(self.delta)


class LADLassoLoss(nn.Module):
    """Least Absolute Deviations (L1) with LASSO sparsity penalty.

    Achieves a finite-sample 50% breakdown point (epsilon* = 0.5), meaning up to
    half the synthetic training dataset can be arbitrary outlier corruption without
    driving parameter estimates to infinity.
    """

    def __init__(
        self,
        alpha: float = 1e-4,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor,
        model_parameters: list[torch.Tensor] | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes LAD loss + optional L1 LASSO parameter penalty."""
        residual = student_hidden - teacher_hidden
        abs_err = torch.abs(residual)

        if mask is not None:
            if mask.dim() < abs_err.dim():
                mask = mask.unsqueeze(-1)
            abs_err = abs_err * mask.to(abs_err.dtype)
            if self.reduction == "mean":
                denom = mask.sum().clamp(min=1.0)
                lad_loss = abs_err.sum() / denom
            elif self.reduction == "sum":
                lad_loss = abs_err.sum()
            else:
                lad_loss = abs_err
        else:
            if self.reduction == "mean":
                lad_loss = abs_err.mean()
            elif self.reduction == "sum":
                lad_loss = abs_err.sum()
            else:
                lad_loss = abs_err

        # LASSO L1 parameter penalty: alpha * sum(|W|)
        if model_parameters is not None and self.alpha > 0:
            l1_pen = torch.tensor(0.0, device=student_hidden.device, dtype=student_hidden.dtype)
            for p in model_parameters:
                if p.requires_grad:
                    l1_pen = l1_pen + torch.sum(torch.abs(p))
            return lad_loss + self.alpha * l1_pen

        return lad_loss


class RobustTokenCrossEntropyLoss(nn.Module):
    """Outlier-clamped Cross-Entropy for training on noisy synthetic tool traces.

    Clamps individual token losses at `outlier_clamp` nats to prevent corrupted
    tokens from generating destabilizing gradients.
    """

    def __init__(
        self,
        outlier_clamp: float = 5.0,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.outlier_clamp = outlier_clamp
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Computes token cross-entropy with outlier loss clamping."""
        # Unreduced cross-entropy
        raw_nll = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        valid_mask = (targets.view(-1) != self.ignore_index).float()
        clamped_nll = torch.clamp(raw_nll, max=self.outlier_clamp) * valid_mask

        denom = valid_mask.sum().clamp(min=1.0)
        return clamped_nll.sum() / denom
