"""Unit tests for Chapter 3 Weibull Hazard Spatio-Temporal Speculative Draft Truncation."""

import torch
from runtime.range_statistic_gate import RangeStatisticGate


def test_weibull_hazard_monotonic_increasing_for_beta_gt_1():
    """Verify that hazard rate strictly increases over draft steps when beta > 1 (wear-out model)."""
    gate = RangeStatisticGate(
        threshold=3.5,
        weibull_hazard_enabled=True,
        weibull_beta=2.2,
        weibull_eta=4.0,
        weibull_gamma=0.6,
    )

    hazards = [gate.compute_hazard_rate(k) for k in range(8)]

    # Must be strictly monotonically increasing
    for i in range(len(hazards) - 1):
        assert hazards[i + 1] > hazards[i], (
            f"Hazard at step {i + 1} ({hazards[i + 1]}) should exceed step {i} ({hazards[i]})"
        )


def test_weibull_hazard_constant_for_beta_eq_1():
    """Verify that hazard rate is constant when beta = 1 (memoryless exponential distribution)."""
    gate = RangeStatisticGate(
        threshold=3.5,
        weibull_hazard_enabled=True,
        weibull_beta=1.0,
        weibull_eta=5.0,
        weibull_gamma=0.6,
    )

    hazards = [gate.compute_hazard_rate(k) for k in range(6)]
    expected_hazard = 1.0 / 5.0  # beta / eta = 1 / 5 = 0.2

    for h in hazards:
        assert abs(h - expected_hazard) < 1e-6, f"Expected constant hazard {expected_hazard}, got {h}"


def test_dynamic_threshold_tightening():
    """Verify that the effective confidence threshold increases with draft depth."""
    gate = RangeStatisticGate(
        threshold=3.5,
        weibull_hazard_enabled=True,
        weibull_beta=2.2,
        weibull_eta=4.0,
        weibull_gamma=0.6,
    )

    thresholds = [gate.get_effective_threshold(k) for k in range(8)]

    assert thresholds[0] > 3.5, "Step 0 threshold should exceed baseline due to initial hazard"
    assert thresholds[7] > thresholds[0] + 2.0, "Step 7 threshold should be significantly tighter than step 0"

    for i in range(len(thresholds) - 1):
        assert thresholds[i + 1] > thresholds[i], "Dynamic threshold must increase monotonically"


def test_weibull_dynamic_abort_on_marginal_tail_tokens():
    """Verify that a marginal token (range = 4.5) is permitted at step 0 but aborted at step 4."""
    gate = RangeStatisticGate(
        threshold=3.5,
        weibull_hazard_enabled=True,
        weibull_beta=2.2,
        weibull_eta=4.0,
        weibull_gamma=0.6,
    )

    # Construct marginal logit distribution with range = 4.5
    # (top = 4.5, 8th = 0.0)
    marginal_logits = torch.zeros(1, 100)
    marginal_logits[0, 0] = 4.5

    # At Step 0: tau_eff ≈ 3.72 < 4.5 -> should NOT abort
    abort_step0, r0, tau0 = gate.should_early_exit(marginal_logits, step_idx=0)
    assert not abort_step0, f"Step 0 should permit marginal drafting (range={r0:.2f} >= tau={tau0:.2f})"

    # At Step 4: tau_eff ≈ 4.66 > 4.5 -> MUST abort
    abort_step4, r4, tau4 = gate.should_early_exit(marginal_logits, step_idx=4)
    assert abort_step4, f"Step 4 must abort marginal drafting (range={r4:.2f} < tau={tau4:.2f})"


def test_weibull_disabled_mode():
    """Verify that when weibull_hazard_enabled=False, effective threshold remains constant."""
    gate = RangeStatisticGate(
        threshold=4.2,
        weibull_hazard_enabled=False,
    )

    for k in range(8):
        tau = gate.get_effective_threshold(k)
        assert abs(tau - 4.2) < 1e-6, f"Expected static threshold 4.2 at step {k}, got {tau}"
