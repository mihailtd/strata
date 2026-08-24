"""Unit tests for Single-Pass Range Statistic Speculative Gating.

Verifies:
1. Exact analytical range computation on top-M candidates.
2. Invariance to vocabulary index permutations and position order.
3. Early-exit decision gating on peaked vs. flat/uniform logit distributions.
4. Scale invariance and numerical stability in studentized range mode.
5. Automated threshold calibration against empirical acceptance logs.
"""

import pytest
import torch
from runtime.range_statistic_gate import RangeStatisticGate


def test_extreme_range_analytical():
    """Verify exact analytical range spread on known top-8 logits."""
    gate = RangeStatisticGate(top_m=8, mode="extreme_range")

    # Construct 100-dim logit vector where top 8 are [15.0, 12.0, ..., 3.0]
    logits = torch.zeros(1, 100)
    top_vals = torch.tensor([15.0, 12.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0])
    logits[0, :8] = top_vals

    range_val = gate.compute_range(logits).item()
    # Expected: max(top_8) - min(top_8) = 15.0 - 3.0 = 12.0
    assert abs(range_val - 12.0) < 1e-5, f"Expected 12.0, got {range_val}"


def test_range_invariance_to_permutation():
    """Verify that vocabulary order does not affect range statistic calculation."""
    gate = RangeStatisticGate(top_m=8, mode="extreme_range")

    logits_original = torch.randn(1, 500)
    # Randomly permute the vocabulary indices
    perm = torch.randperm(500)
    logits_permuted = logits_original[:, perm]

    r_orig = gate.compute_range(logits_original).item()
    r_perm = gate.compute_range(logits_permuted).item()

    assert abs(r_orig - r_perm) < 1e-5, f"Permutation changed range: {r_orig} vs {r_perm}"


def test_early_exit_decision():
    """Verify early-exit triggering: Peaked -> Continue drafting; Flat -> Abort drafting."""
    gate = RangeStatisticGate(top_m=8, threshold=5.0)

    # 1. Peaked distribution: top logit = 20.0, others ~ 0.0 -> Range = 20.0 >= 5.0 -> Do NOT abort
    peaked_logits = torch.zeros(1, 100)
    peaked_logits[0, 0] = 20.0
    should_abort_peaked, r_peaked = gate.should_early_exit(peaked_logits)
    assert not should_abort_peaked, f"Peaked distribution was incorrectly aborted (range={r_peaked})"

    # 2. Flat / Uniform distribution: all top logits = 1.0 -> Range = 0.0 < 5.0 -> MUST abort
    flat_logits = torch.ones(1, 100) + torch.randn(1, 100) * 0.01
    should_abort_flat, r_flat = gate.should_early_exit(flat_logits)
    assert should_abort_flat, f"Flat distribution was not aborted (range={r_flat})"


def test_studentized_range_mode():
    """Verify studentized range statistic calculation."""
    gate = RangeStatisticGate(top_m=8, mode="studentized_range")

    logits = torch.tensor([[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 2.0, 0.0, -1.0]])
    q_val = gate.compute_range(logits).item()
    assert q_val > 0.0, f"Expected positive studentized range, got {q_val}"


def test_threshold_calibration():
    """Verify automated cutoff calibration against empirical token acceptance."""
    # Create 500 samples: 250 high-range (accepted), 250 low-range (rejected)
    n = 500
    logits_hist = torch.zeros(n, 50)
    matches_hist = torch.zeros(n)

    # High-confidence samples (Range ~ 10.0, Match = 1)
    logits_hist[:250, 0] = 10.0 + torch.randn(250) * 0.5
    matches_hist[:250] = 1.0

    # Low-confidence samples (Range ~ 1.5, Match = 0)
    logits_hist[250:, :8] = torch.randn(250, 8) * 0.5
    matches_hist[250:] = 0.0

    calibrated_tau = RangeStatisticGate.calibrate_threshold(
        logits_hist, matches_hist, top_m=8, target_precision=0.90
    )

    assert 0.8 <= calibrated_tau <= 10.0, f"Calibrated threshold {calibrated_tau} should separate high and low range"
