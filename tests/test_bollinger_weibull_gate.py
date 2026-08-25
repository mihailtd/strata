"""Unit tests for Chapter 5 Bollinger Bands & ATR Volatility Gating combined with Chapter 3 Weibull Hazard."""

import pytest
import torch
from runtime.range_statistic_gate import RangeStatisticGate


def test_bollinger_state_initialization_and_reset():
    gate = RangeStatisticGate(top_m=8, threshold=3.5, bollinger_bands_enabled=True)
    assert gate.running_mean_spread == 3.0
    assert gate.running_var_spread == 1.0
    assert gate.history_count == 0

    # Feed some tokens
    for _ in range(5):
        logits = torch.randn(1, 100)
        gate.should_early_exit(logits, step_idx=0)

    assert gate.history_count == 5
    gate.reset_state(initial_spread=4.0)
    assert gate.running_mean_spread == 4.0
    assert gate.history_count == 0


def test_bollinger_bands_tighten_and_expand_on_volatility():
    gate = RangeStatisticGate(top_m=8, threshold=3.0, bollinger_bands_enabled=True, bollinger_k=2.0)
    gate.reset_state(initial_spread=5.0)

    # Feed 10 high-confidence tokens with spread ~ 5.0
    for _ in range(10):
        logits = torch.zeros(1, 100)
        logits[0, 0] = 10.0
        logits[0, 1] = 5.0  # spread = 5.0
        gate.should_early_exit(logits, step_idx=0)

    assert gate.running_mean_spread > 4.5
    # Now feed a sudden low-confidence collapse: spread = 0.2
    collapsed_logits = torch.zeros(1, 100)
    collapsed_logits[0, 0] = 5.0
    collapsed_logits[0, 1] = 4.8  # spread = 0.2
    for j in range(2, 8):
        collapsed_logits[0, j] = 4.0

    diag = gate.inspect_decision(collapsed_logits, step_idx=0)
    assert diag["top1_top2_spread"] < diag["lower_band"]
    assert diag["vol_penalty"] > 0.0
    assert diag["effective_threshold"] > gate.threshold


def test_synergy_weibull_plus_bollinger_composite():
    """Verify that Weibull temporal wear-out and Bollinger volatility combine additively."""
    gate = RangeStatisticGate(
        top_m=8,
        threshold=3.0,
        weibull_hazard_enabled=True,
        weibull_beta=2.2,
        weibull_eta=4.0,
        weibull_gamma=0.6,
        bollinger_bands_enabled=True,
        bollinger_gamma=0.5,
    )
    gate.reset_state(initial_spread=5.0)

    # Step 0 with no volatility penalty
    tau_step0 = gate.get_effective_threshold(step_idx=0, vol_penalty=0.0)
    # Step 3 with Weibull wear-out (h(3) > h(0))
    tau_step3_clean = gate.get_effective_threshold(step_idx=3, vol_penalty=0.0)
    assert tau_step3_clean > tau_step0

    # Step 3 with combined Weibull wear-out AND Bollinger volatility breakout penalty
    tau_step3_vol = gate.get_effective_threshold(step_idx=3, vol_penalty=2.5)
    assert tau_step3_vol > tau_step3_clean + 2.0


def test_hard_breakdown_aborts_immediately_on_flat_distribution():
    """Flat distribution where top-1 is virtually indistinguishable from runners-up triggers hard abort."""
    gate = RangeStatisticGate(top_m=8, threshold=3.5, bollinger_bands_enabled=True)
    gate.reset_state(initial_spread=4.0)

    # Prime history with 5 normal tokens
    for _ in range(5):
        logits = torch.zeros(1, 100)
        logits[0, 0] = 8.0
        logits[0, 1] = 4.0
        gate.should_early_exit(logits, step_idx=0)

    # Uniform noisy distribution
    flat_logits = torch.ones(1, 100) * 2.0
    flat_logits[0, 0] = 2.05
    flat_logits[0, 1] = 2.04

    should_abort, range_val, tau = gate.should_early_exit(flat_logits, step_idx=2)
    assert should_abort is True


def test_disabled_modes():
    """Verify that turning off Weibull or Bollinger reverts to standard fixed threshold."""
    gate = RangeStatisticGate(
        top_m=8,
        threshold=3.5,
        weibull_hazard_enabled=False,
        bollinger_bands_enabled=False,
    )
    tau_0 = gate.get_effective_threshold(step_idx=0, vol_penalty=5.0)
    tau_5 = gate.get_effective_threshold(step_idx=5, vol_penalty=5.0)
    assert abs(tau_0 - 3.5) < 1e-5
    assert abs(tau_5 - 3.5) < 1e-5
