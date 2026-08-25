"""Unit Tests for Dual-EMA / MACD Speculation Circuit-Breaker (Chapters 5 & 8)."""

import pytest
from runtime.macd_speculation_circuit_breaker import MACDSpeculationCircuitBreaker


def test_initial_state():
    cb = MACDSpeculationCircuitBreaker(
        alpha_fast=0.25,
        alpha_slow=0.08,
        disengage_threshold=1.8,
        reengage_threshold=2.2,
        initial_tau=2.5,
        enabled=True,
    )
    assert cb.is_engaged is True
    assert cb.ema_fast == 2.5
    assert cb.ema_slow == 2.5
    assert cb.macd == 0.0
    run_draft, k = cb.should_draft()
    assert run_draft is True
    assert k == 0


def test_bearish_crossover_trips_circuit_breaker():
    cb = MACDSpeculationCircuitBreaker(
        alpha_fast=0.25,
        alpha_slow=0.08,
        disengage_threshold=1.8,
        reengage_threshold=2.2,
        initial_tau=2.5,
        enabled=True,
    )

    # Feed a series of rejected steps (tau = 1.0, meaning 0 draft tokens accepted)
    tripped = False
    for step in range(8):
        is_engaged = cb.update_acceptance(1.0)
        if not is_engaged:
            tripped = True
            break

    assert tripped is True, "Circuit breaker should trip on consecutive low acceptance yields"
    assert cb.is_engaged is False
    assert cb.macd < 0.0
    assert cb.ema_fast < 1.8
    assert cb.telemetry.trips_count == 1

    # In disengaged state, should_draft should return False (raw W=1 decode)
    run_draft, k = cb.should_draft()
    assert run_draft is False
    assert k == 1


def test_probing_and_bullish_reengagement():
    cb = MACDSpeculationCircuitBreaker(
        alpha_fast=0.25,
        alpha_slow=0.08,
        disengage_threshold=1.8,
        reengage_threshold=2.2,
        initial_tau=2.5,
        probe_interval=4,
        enabled=True,
    )

    # Force a trip
    for _ in range(8):
        cb.update_acceptance(1.0)
    assert cb.is_engaged is False

    # Simulate disengaged steps until probe interval
    cb.steps_since_disengage = 3
    run_draft, k = cb.should_draft()
    assert run_draft is False

    cb.steps_since_disengage = 4  # Probe step!
    run_draft, k = cb.should_draft()
    assert run_draft is True
    assert k == 1

    # Simulate high-yield acceptance resuming (e.g. boilerplate text where tau=3.5)
    for _ in range(6):
        cb.update_acceptance(3.5)

    assert cb.is_engaged is True, "Circuit breaker should re-engage on sustained high acceptance"
    assert cb.macd > 0.0
    assert cb.ema_fast > 2.2
    assert cb.telemetry.reengages_count >= 1


def test_disabled_circuit_breaker():
    cb = MACDSpeculationCircuitBreaker(enabled=False)
    assert cb.update_acceptance(0.0) is True
    run_draft, k = cb.should_draft()
    assert run_draft is True
    assert k == 0
