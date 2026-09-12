r"""SCAD and MCP: continuous thresholding for the velocity gate (Ch.2 §2.2).

WHY, CONCRETELY
---------------
`VelocityGate` (novel_peft.py) masks the bottom `quiet_percentile`% of layers by EMA
velocity with a BINARY mask. A layer sitting on the percentile boundary flips between
gain 1.0 and gain 0.0 between consecutive steps, so its parameters see a step-function
in the gradient they receive. That is a discontinuity in the optimisation path, not a
regulariser.

L1 / soft-thresholding removes the discontinuity but introduces a different defect: it
shrinks EVERY coefficient by lambda, including the large ones that should be left
alone. On a velocity gate that means the loudest, most task-relevant layers get
attenuated for no reason.

SCAD (Fan & Li, 2001) and MCP (Zhang, 2010) are the penalties designed to have both
properties at once:

    |z| <= lambda        -> exactly 0        sparsity, same as hard masking
    lambda < |z| < gamma*lambda -> smooth ramp   continuity, unlike hard masking
    |z| >= gamma*lambda  -> exactly z        UNBIASED, unlike soft thresholding

The middle region is the whole point. Hard thresholding has regions 1 and 3 with a
jump between them; soft thresholding has regions 1 and 2 and never reaches 3.

    MCP:   prox(z) = 0                                  |z| <= lambda
                     (z - lambda*sign(z)) / (1 - 1/gamma)   lambda < |z| <= gamma*lambda
                     z                                    |z| > gamma*lambda

    SCAD:  prox(z) = soft(z, lambda)                       |z| <= 2*lambda
                     soft(z, gamma*lambda/(gamma-1)) / (1 - 1/(gamma-1))
                                                           2*lambda < |z| <= gamma*lambda
                     z                                     |z| > gamma*lambda

⚠️ WHAT IS NOT ESTABLISHED
--------------------------
That this HELPS. The claim "hard freezing creates discontinuous gradient shocks" is
plausible and the operators above provably remove the discontinuity, but no measured
comparison exists on this model: there are no per-layer velocity traces on disk
(`results/loss_curves/*.json` records a single global `grad_norm` per step, not a
per-layer distribution). Testing it needs an instrumented training run with the
velocity arm, the SCAD arm and the existing `selection="random"` control -- the
control matters, because the gate's own docstring records that random masking of the
same size is the arm that tells you whether the velocity ranking carries any signal
at all.
"""

from __future__ import annotations

import torch

__all__ = ["soft_threshold", "hard_threshold", "mcp_prox", "scad_prox", "mcp_penalty", "scad_penalty", "velocity_gain"]


def soft_threshold(z: torch.Tensor, lam: float) -> torch.Tensor:
    """L1 prox. Sparse, but biases every surviving coefficient toward zero by lam."""
    return torch.sign(z) * torch.clamp(z.abs() - lam, min=0.0)


def hard_threshold(z: torch.Tensor, lam: float) -> torch.Tensor:
    """What VelocityGate does today (as a magnitude rule). Unbiased but discontinuous."""
    return torch.where(z.abs() > lam, z, torch.zeros_like(z))


def mcp_prox(z: torch.Tensor, lam: float, gamma: float = 3.0) -> torch.Tensor:
    """Minimax Concave Penalty proximal operator. Requires gamma > 1."""
    if gamma <= 1.0:
        raise ValueError(f"MCP needs gamma > 1 (got {gamma}); at gamma=1 it is hard thresholding")
    a = z.abs()
    mid = (torch.sign(z) * torch.clamp(a - lam, min=0.0)) / (1.0 - 1.0 / gamma)
    return torch.where(a > gamma * lam, z, torch.where(a <= lam, torch.zeros_like(z), mid))


def scad_prox(z: torch.Tensor, lam: float, gamma: float = 3.7) -> torch.Tensor:
    """SCAD proximal operator. Requires gamma > 2."""
    if gamma <= 2.0:
        raise ValueError(f"SCAD needs gamma > 2 (got {gamma})")
    a = z.abs()
    lo = soft_threshold(z, lam)
    mid = soft_threshold(z, gamma * lam / (gamma - 1.0)) / (1.0 - 1.0 / (gamma - 1.0))
    return torch.where(a > gamma * lam, z, torch.where(a <= 2.0 * lam, lo, mid))


def mcp_penalty(z: torch.Tensor, lam: float, gamma: float = 3.0) -> torch.Tensor:
    """p_lambda(|z|), the penalty value itself. Flat above gamma*lam -- zero gradient
    on large coefficients, which is what 'unbiased' means here."""
    a = z.abs()
    return torch.where(a <= gamma * lam, lam * a - a * a / (2.0 * gamma), torch.full_like(a, gamma * lam * lam / 2.0))


def scad_penalty(z: torch.Tensor, lam: float, gamma: float = 3.7) -> torch.Tensor:
    a = z.abs()
    p1 = lam * a
    p2 = (2.0 * gamma * lam * a - a * a - lam * lam) / (2.0 * (gamma - 1.0))
    p3 = torch.full_like(a, lam * lam * (gamma + 1.0) / 2.0)
    return torch.where(a <= lam, p1, torch.where(a <= gamma * lam, p2, p3))


def velocity_gain(
    velocity: torch.Tensor,
    quiet_percentile: float = 25.0,
    gamma: float = 3.0,
    rule: str = "mcp",
) -> torch.Tensor:
    """Per-layer gain in [0, 1] to replace VelocityGate's binary mask.

    lambda is set to the `quiet_percentile` of the CURRENT velocity distribution, for
    the same reason the existing gate uses a percentile rather than a fixed number: a
    fixed 0.45 threshold, tuned on one dataset, silently produced 0% quiet layers when
    the data composition changed. A percentile cannot go inert that way.

    Returned as a GAIN (prox(v)/v) rather than a shrunken velocity, because the
    consumer multiplies a layer's contribution, and gain is what stays bounded when a
    velocity is near zero.
    """
    v = velocity.abs().float()
    finite = v[torch.isfinite(v)]
    if finite.numel() == 0:
        return torch.ones_like(v)
    lam = float(torch.quantile(finite, quiet_percentile / 100.0))
    if lam <= 0.0:
        return torch.ones_like(v)
    prox = {"mcp": mcp_prox, "scad": scad_prox}[rule](v, lam, gamma)
    gain = torch.where(v > 0, prox / v.clamp(min=1e-12), torch.zeros_like(v))
    return torch.nan_to_num(gain, nan=0.0).clamp(0.0, 1.0)
