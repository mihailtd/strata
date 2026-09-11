"""Dual-EMA / MACD Speculation Circuit-Breaker (Chapters 5 & 8).

Theoretical Grounding:
1. Chapter 5 (Trend-Following Technical Indicators & MACD Oscillator):
   - Fast Exponential Moving Average:
     EMA_fast(tau_t) = alpha_fast * tau_t + (1 - alpha_fast) * EMA_fast_{t-1}
   - Slow Exponential Moving Average:
     EMA_slow(tau_t) = alpha_slow * tau_t + (1 - alpha_slow) * EMA_slow_{t-1}
   - MACD Divergence:
     MACD_t = EMA_fast(t) - EMA_slow(t)

2. Chapter 8 (Dynamic Verification Tax Elimination):
   - When draft acceptance rate tau drops below 1.8, the speculative verification
     tax drops throughput below raw W=1 CUDA graph decode.
   - Bearish Crossover (MACD < 0 and EMA_fast < 1.8):
     Instantly trips the circuit-breaker, disengaging speculative drafting and
     routing to raw W=1 CUDA graph decode (guaranteed baseline speed floor).
   - Bullish Crossover (MACD > 0 and EMA_fast > 2.2):
     Re-engages speculation when predictable boilerplate or high-confidence text resumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import torch


@dataclass
class CircuitBreakerTelemetry:
    total_steps: int = 0
    speculative_steps: int = 0
    disengaged_steps: int = 0
    trips_count: int = 0
    reengages_count: int = 0
    current_ema_fast: float = 2.5
    current_ema_slow: float = 2.5
    current_macd: float = 0.0
    is_engaged: bool = True
    tau_history: list[float] = field(default_factory=list)


class MACDSpeculationCircuitBreaker:
    """Intelligent algorithmic thermostat for speculative decoding.
    
    Dynamically engages/disengages speculative drafting based on real-time
    dual-EMA MACD momentum of draft acceptance rate tau.
    """

    def __init__(
        self,
        alpha_fast: float = 0.25,
        alpha_slow: float = 0.08,
        disengage_threshold: float = 1.8,
        reengage_threshold: float = 2.2,
        initial_tau: float = 2.5,
        probe_interval: int = 6,
        enabled: bool = True,
    ):
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.disengage_threshold = disengage_threshold
        self.reengage_threshold = reengage_threshold
        self.initial_tau = initial_tau
        self.probe_interval = probe_interval
        self.enabled = enabled

        # State variables
        self.ema_fast: float = initial_tau
        self.ema_slow: float = initial_tau
        self.macd: float = 0.0
        self.is_engaged: bool = True
        self.steps_since_disengage: int = 0

        # Telemetry
        self.telemetry = CircuitBreakerTelemetry()

    def reset(self) -> None:
        """Resets the circuit breaker state for a new generation turn."""
        self.ema_fast = self.initial_tau
        self.ema_slow = self.initial_tau
        self.macd = 0.0
        self.is_engaged = True
        self.steps_since_disengage = 0
        self.telemetry = CircuitBreakerTelemetry()

    def update_acceptance(self, tau_step: float) -> bool:
        """Updates EMA and MACD with the latest step's acceptance yield (tau_step).
        
        tau_step is typically (n_accepted + 1), the number of committed tokens this step.
        Returns True if speculation is currently ENGAGED, False if DISENGAGED.
        """
        if not self.enabled:
            return True

        self.telemetry.total_steps += 1
        self.telemetry.tau_history.append(tau_step)

        # 1. Update Dual-EMA
        self.ema_fast = (self.alpha_fast * tau_step) + ((1.0 - self.alpha_fast) * self.ema_fast)
        self.ema_slow = (self.alpha_slow * tau_step) + ((1.0 - self.alpha_slow) * self.ema_slow)
        self.macd = self.ema_fast - self.ema_slow

        self.telemetry.current_ema_fast = round(self.ema_fast, 3)
        self.telemetry.current_ema_slow = round(self.ema_slow, 3)
        self.telemetry.current_macd = round(self.macd, 3)

        # 2. State Machine Transition Checks
        if self.is_engaged:
            self.telemetry.speculative_steps += 1
            # Bearish Crossover Check: Momentum negative and fast acceptance below speed parity floor
            if self.macd < 0.0 and self.ema_fast < self.disengage_threshold:
                self.is_engaged = False
                self.steps_since_disengage = 0
                self.telemetry.trips_count += 1
                self.telemetry.is_engaged = False
        else:
            self.telemetry.disengaged_steps += 1
            self.steps_since_disengage += 1
            # Bullish Crossover Check: Momentum positive and fast acceptance clears re-engagement bar
            if self.macd > 0.0 and self.ema_fast > self.reengage_threshold:
                self.is_engaged = True
                self.telemetry.reengages_count += 1
                self.telemetry.is_engaged = True

        return self.is_engaged

    def should_draft(self) -> tuple[bool, int]:
        """Determines whether the upcoming step should run speculative drafting.
        
        Returns:
            (run_speculative, effective_k)
            - If ENGAGED: (True, standard_k)
            - If DISENGAGED & on probe step: (True, 1)  [Probe with K=1 to test waters]
            - If DISENGAGED: (False, 1)  [Raw W=1 CUDA Graph decode]
        """
        if not self.enabled:
            return True, 0  # 0 indicates use default decoder k

        if self.is_engaged:
            return True, 0

        # When disengaged, periodically probe with K=1 every probe_interval steps
        if self.steps_since_disengage > 0 and (self.steps_since_disengage % self.probe_interval == 0):
            return True, 1  # Probe step with minimal K=1 risk

        return False, 1  # Raw W=1 CUDA Graph decode

    def get_summary(self) -> dict[str, Any]:
        """Returns structured summary telemetry."""
        tot = max(1, self.telemetry.total_steps)
        disengaged_pct = (self.telemetry.disengaged_steps / tot) * 100.0
        return {
            "enabled": self.enabled,
            "is_currently_engaged": self.is_engaged,
            "ema_fast": round(self.ema_fast, 3),
            "ema_slow": round(self.ema_slow, 3),
            "macd": round(self.macd, 3),
            "trips_count": self.telemetry.trips_count,
            "reengages_count": self.telemetry.reengages_count,
            "speculative_steps": self.telemetry.speculative_steps,
            "disengaged_steps": self.telemetry.disengaged_steps,
            "disengaged_pct": round(disengaged_pct, 1),
            "avg_tau": round(sum(self.telemetry.tau_history) / tot, 2) if self.telemetry.tau_history else 0.0,
        }
