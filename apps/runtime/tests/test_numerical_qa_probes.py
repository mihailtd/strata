"""Numerical QA probes: KL-divergence and top-1 argmax agreement between two logit distributions.

Split out of the former tests/test_category1_hygiene.py (Section 5) -- a
generic fidelity check used to distinguish acceptable numerical noise
(bf16 rounding, quantization) from a real correctness regression, not tied
to one specific runtime module.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


def compute_kld_in_fp64(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """Computes Kullback-Leibler Divergence D_KL(P || Q) in double precision."""
    p_probs = F.softmax(p_logits.to(torch.float64), dim=-1)
    q_log_probs = F.log_softmax(q_logits.to(torch.float64), dim=-1)
    p_log_probs = F.log_softmax(p_logits.to(torch.float64), dim=-1)
    # D_KL(P || Q) = sum(P * (log P - log Q))
    kld = torch.sum(p_probs * (p_log_probs - q_log_probs), dim=-1).mean().item()
    return float(max(0.0, kld))


def compute_top1_agreement(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """Computes fraction of positions where greedy argmax predictions match."""
    p_top1 = torch.argmax(p_logits, dim=-1)
    q_top1 = torch.argmax(q_logits, dim=-1)
    matches = (p_top1 == q_top1).float().mean().item()
    return float(matches)


def test_kld_and_top1_numerical_probe_identical_distributions():
    """Identical distributions must have KLD = 0.0 and Top-1 agreement = 1.0."""
    torch.manual_seed(42)
    logits = torch.randn(10, 128, dtype=torch.bfloat16)
    kld = compute_kld_in_fp64(logits, logits)
    top1 = compute_top1_agreement(logits, logits)

    assert kld == pytest.approx(0.0, abs=1e-7)
    assert top1 == pytest.approx(1.0, abs=1e-7)


def test_kld_and_top1_numerical_probe_small_perturbation():
    """Small numerical noise (e.g. BF16 rounding) must stay well below KLD < 0.01 threshold."""
    torch.manual_seed(42)
    base_logits = torch.randn(32, 1024, dtype=torch.float32)
    noisy_logits = base_logits + torch.randn_like(base_logits) * 0.01

    kld = compute_kld_in_fp64(base_logits, noisy_logits)
    top1 = compute_top1_agreement(base_logits, noisy_logits)

    assert kld < 0.01, f"KLD exceeded safety threshold: {kld:.6f}"
    assert top1 >= 0.95, f"Top-1 agreement fell below 95%: {top1:.4f}"


def test_kld_and_top1_numerical_probe_detects_severe_divergence():
    """Severe quantization noise (simulating INT4 collapse) triggers KLD > 0.01 failure."""
    torch.manual_seed(42)
    base_logits = torch.randn(32, 1024, dtype=torch.float32)
    corrupted_logits = base_logits + torch.randn_like(base_logits) * 1.5

    kld = compute_kld_in_fp64(base_logits, corrupted_logits)
    top1 = compute_top1_agreement(base_logits, corrupted_logits)

    assert kld > 0.01, f"Expected KLD failure on corrupt logits, got: {kld:.6f}"
    assert top1 < 0.95, f"Expected Top-1 agreement failure, got: {top1:.4f}"
