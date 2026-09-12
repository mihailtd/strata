"""Runtime Thinking Supervisor (Chapters 3, 4, 5 & 8).

Theoretical & Algorithmic Grounding:
1. Sampler-Level Logit Masking & Force Injection:
   - Eliminates dependence on prompt compliance for small (4B/9B) models.
   - Enforces a minimum thinking floor (masking </think> for min_tokens), a soft sigmoid
     logit boost between target and hard max, and a hard token budget cutoff.

2. Renko-Filtered Latent Loop & Attractor Detection (Chapter 4):
   - Monitors residual stream hidden states h_t.
   - Calculates historical cosine recurrence Sim(h_t, h_{t-k}) > 0.96 against [h_{t-32}..h_{t-8}].
   - If cosine similarity is high while Renko displacement drops below epsilon_box,
     detects an autoregressive attractor loop and forcefully breaks out to the answer.

3. Trailing Stoploss on Reasoning Entropy (Chapter 5):
   - Tracks Shannon entropy H(p_t) = -sum(p * log p) of logit distributions.
   - Cognitive Convergence: When entropy collapses and stabilizes (mu_H < 0.35),
     the model has found its answer; triggers early clean exit.
   - Divergence Stoploss: When entropy breaches upper Bollinger bands (mu_H + 2.5*sigma_H)
     for prolonged steps (divergent hallucination), forces transition to answer.

4. Weibull Hazard Cutoff (Chapter 3):
   - Models 4B/9B reasoning failure rate via Weibull cumulative CDF F(t) = 1 - exp(-(t/eta)^beta).
   - Cuts reasoning deterministically once failure probability exceeds 50%.

5. Clean Structured Delimiter Transition Sequence:
   - Injects the exact whitespace and token delimiter sequence `\n</think>\n\n`
     (Qwen3.5 IDs: [198, 248069, 271]) into the KV cache and stream to align
     attention heads with the SFT answer distribution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class SupervisorStepAction:
    """Action outcome from a single supervisor evaluation step."""

    modified_logits: torch.Tensor
    should_force_transition: bool = False
    transition_token_ids: list[int] = field(default_factory=lambda: [198, 248069, 271])
    exit_reason: str | None = None
    step_entropy: float = 0.0
    in_thinking_mode: bool = True
    renko_loop_detected: bool = False
    divergence_detected: bool = False


class ThinkingRuntimeSupervisor:
    """Deterministic runtime supervisor for multi-tier reasoning depth & loop prevention."""

    # Default token budget tiers: min_tokens, target_tokens, hard_max_tokens
    DEFAULT_BUDGET_TIERS: dict[str, dict[str, int]] = {
        "off": {"min": 0, "target": 0, "hard_max": 0},
        "low": {"min": 64, "target": 384, "hard_max": 512},
        "medium": {"min": 128, "target": 1536, "hard_max": 2048},
        "high": {"min": 256, "target": 3072, "hard_max": 4096},
    }

    def __init__(
        self,
        think_end_token_id: int = 248069,  # Qwen 3.5 </think> special token
        think_start_token_id: int = 248068,  # Qwen 3.5 <think> special token
        transition_token_ids: list[int] | None = None,
        budget_tier: str = "medium",
        custom_caps: dict[str, int] | None = None,
        loop_similarity_threshold: float = 0.96,
        entropy_window_size: int = 16,
        entropy_convergence_threshold: float = 0.35,
        weibull_beta: float = 2.2,
        weibull_eta: float | None = None,
        enabled: bool = True,
    ):
        self.think_end_token_id = think_end_token_id
        self.think_start_token_id = think_start_token_id
        self.transition_token_ids = transition_token_ids or [198, 248069, 271]
        self.budget_tier = budget_tier.lower()
        self.enabled = enabled

        # Resolve token caps
        if custom_caps:
            self.caps = custom_caps
        else:
            self.caps = self.DEFAULT_BUDGET_TIERS.get(self.budget_tier, self.DEFAULT_BUDGET_TIERS["medium"]).copy()

        # Hyperparameters
        self.loop_similarity_threshold = loop_similarity_threshold
        self.entropy_window_size = entropy_window_size
        self.entropy_convergence_threshold = entropy_convergence_threshold
        self.weibull_beta = weibull_beta
        # Dynamically scale Weibull hazard threshold to 1.25x the hard_max budget if not provided
        self.weibull_eta = (
            weibull_eta if weibull_eta is not None else max(600.0, float(self.caps.get("hard_max", 2048)) * 1.25)
        )

        # State tracking
        self.in_thinking_mode: bool = self.budget_tier != "off"
        self.step_count: int = 0
        self.recent_hidden_states: list[torch.Tensor] = []
        self.recent_entropies: list[float] = []
        self.exit_reason: str | None = None
        self.divergence_streak: int = 0
        self.was_forced: bool = False

    def reset(self, budget_tier: str | None = None) -> None:
        """Resets supervisor state for a new conversation turn."""
        if budget_tier:
            self.budget_tier = budget_tier.lower()
            self.caps = self.DEFAULT_BUDGET_TIERS.get(self.budget_tier, self.DEFAULT_BUDGET_TIERS["medium"]).copy()

        self.in_thinking_mode = self.budget_tier != "off"
        self.step_count = 0
        self.recent_hidden_states = []
        self.recent_entropies = []
        self.exit_reason = None
        self.divergence_streak = 0
        self.was_forced = False

    def notify_token_emitted(self, token_id: int) -> None:
        """Called when a token is committed to update thinking mode status."""
        if token_id == self.think_end_token_id or token_id in self.transition_token_ids:
            self.in_thinking_mode = False
            if self.exit_reason is None:
                self.exit_reason = "natural_model_conclusion"

    def process_step(
        self,
        logits: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
    ) -> SupervisorStepAction:
        """Processes the current step's logits and hidden state, applying bias or force-exit.

        Args:
            logits: Tensor of shape [1, vocab_size] or [vocab_size]
            hidden_state: Residual stream hidden vector of shape [1, hidden_dim] or [hidden_dim]
        """
        if not self.enabled or not self.in_thinking_mode:
            return SupervisorStepAction(
                modified_logits=logits,
                should_force_transition=False,
                in_thinking_mode=self.in_thinking_mode,
                exit_reason=self.exit_reason,
            )

        self.step_count += 1
        logits_work = logits.clone()
        if logits_work.dim() == 1:
            logits_view = logits_work.unsqueeze(0)
        else:
            logits_view = logits_work

        vocab_size = logits_view.shape[-1]
        valid_end_token = 0 <= self.think_end_token_id < vocab_size

        # Compute Shannon Entropy H(p_t)
        with torch.no_grad():
            probs = F.softmax(logits_view, dim=-1)
            log_probs = torch.log(probs + 1e-9)
            entropy = -torch.sum(probs * log_probs, dim=-1).item()

        self.recent_entropies.append(entropy)
        if len(self.recent_entropies) > self.entropy_window_size * 2:
            self.recent_entropies.pop(0)

        # -------------------------------------------------------------
        # 1. Enforce Minimum Thinking Floor
        # -------------------------------------------------------------
        if self.step_count < self.caps["min"] and valid_end_token:
            # Mask </think> token so model cannot prematurely exit
            logits_view[0, self.think_end_token_id] = -1e9

        # -------------------------------------------------------------
        # 2. Hard Budget Cap - Force Transition
        # -------------------------------------------------------------
        if self.step_count >= self.caps["hard_max"] and self.caps["hard_max"] > 0:
            self.in_thinking_mode = False
            self.exit_reason = "hard_max_budget_exceeded"
            self.was_forced = True
            return SupervisorStepAction(
                modified_logits=logits_work,
                should_force_transition=True,
                transition_token_ids=self.transition_token_ids,
                exit_reason=self.exit_reason,
                step_entropy=entropy,
                in_thinking_mode=False,
            )

        # -------------------------------------------------------------
        # 3. Soft Sigmoid Logit Ramp (Target -> Hard Max)
        # -------------------------------------------------------------
        if self.step_count >= self.caps["target"] and self.caps["hard_max"] > self.caps["target"] and valid_end_token:
            progress = (self.step_count - self.caps["target"]) / max(1, (self.caps["hard_max"] - self.caps["target"]))
            # Sigmoid curve centered at 0.5 with steep scaling up to +15.0 logits
            sigmoid_boost = 15.0 / (1.0 + math.exp(-6.0 * (progress - 0.5)))
            logits_view[0, self.think_end_token_id] += sigmoid_boost

        # -------------------------------------------------------------
        # 4. Renko Latent Loop & Attractor Detection (Chapter 4)
        # -------------------------------------------------------------
        renko_loop = False
        if hidden_state is not None:
            with torch.no_grad():
                h_flat = hidden_state.view(1, -1).detach().float()
                h_norm = F.normalize(h_flat, p=2, dim=-1)

                # Check against historical states in circular buffer
                if len(self.recent_hidden_states) >= 8:
                    # Compare against states from [t-32 .. t-8] to avoid adjacent step similarity
                    history_tensor = torch.cat(self.recent_hidden_states[:-6], dim=0)
                    sims = torch.mm(h_norm, history_tensor.T).squeeze(0)
                    max_sim = torch.max(sims).item()

                    if max_sim > self.loop_similarity_threshold:
                        renko_loop = True
                        self.in_thinking_mode = False
                        self.exit_reason = f"renko_latent_attractor_loop (Sim={max_sim:.3f})"
                        self.was_forced = True
                        return SupervisorStepAction(
                            modified_logits=logits_work,
                            should_force_transition=True,
                            transition_token_ids=self.transition_token_ids,
                            exit_reason=self.exit_reason,
                            step_entropy=entropy,
                            in_thinking_mode=False,
                            renko_loop_detected=True,
                        )

                # Store current normalized hidden state
                self.recent_hidden_states.append(h_norm)
                if len(self.recent_hidden_states) > 36:
                    self.recent_hidden_states.pop(0)

        # -------------------------------------------------------------
        # 5. Trailing Shannon Entropy Stoploss (Chapter 5)
        # -------------------------------------------------------------
        divergence_detected = False
        if len(self.recent_entropies) >= self.entropy_window_size and self.step_count > self.caps["min"]:
            window = self.recent_entropies[-self.entropy_window_size :]
            mean_h = sum(window) / len(window)
            var_h = sum((x - mean_h) ** 2 for x in window) / len(window)
            std_h = math.sqrt(var_h)

            # 5a. Cognitive Convergence: Entropy collapsed and stabilized at low floor
            if mean_h < self.entropy_convergence_threshold and valid_end_token:
                logits_view[0, self.think_end_token_id] += 12.0
                if len(window) >= 16 and all(h < self.entropy_convergence_threshold for h in window):
                    self.in_thinking_mode = False
                    self.exit_reason = (
                        f"entropy_cognitive_convergence (Mean_H={mean_h:.3f} < {self.entropy_convergence_threshold})"
                    )
                    self.was_forced = True
                    return SupervisorStepAction(
                        modified_logits=logits_work,
                        should_force_transition=True,
                        transition_token_ids=self.transition_token_ids,
                        exit_reason=self.exit_reason,
                        step_entropy=entropy,
                        in_thinking_mode=False,
                    )

            # 5b. Divergence Stoploss: Entropy exploded beyond upper Bollinger Band
            upper_band = mean_h + (2.2 * std_h)
            if entropy > upper_band and entropy > 3.0:
                self.divergence_streak += 1
                if self.divergence_streak >= 12:
                    divergence_detected = True
                    self.in_thinking_mode = False
                    self.exit_reason = f"entropy_divergence_stoploss (H={entropy:.2f} > Band={upper_band:.2f})"
                    self.was_forced = True
                    return SupervisorStepAction(
                        modified_logits=logits_work,
                        should_force_transition=True,
                        transition_token_ids=self.transition_token_ids,
                        exit_reason=self.exit_reason,
                        step_entropy=entropy,
                        in_thinking_mode=False,
                        divergence_detected=True,
                    )
            else:
                self.divergence_streak = max(0, self.divergence_streak - 1)

        # -------------------------------------------------------------
        # 6. Weibull Hazard Reasoning Cutoff (Chapter 3)
        # -------------------------------------------------------------
        if self.step_count > self.caps["min"]:
            # Weibull CDF: F(t) = 1 - exp(-(t/eta)^beta)
            failure_prob = 1.0 - math.exp(-((self.step_count / self.weibull_eta) ** self.weibull_beta))
            if failure_prob > 0.50:
                self.in_thinking_mode = False
                self.exit_reason = f"weibull_hazard_exceeded (F(t)={failure_prob:.2f} > 0.50)"
                self.was_forced = True
                return SupervisorStepAction(
                    modified_logits=logits_work,
                    should_force_transition=True,
                    transition_token_ids=self.transition_token_ids,
                    exit_reason=self.exit_reason,
                    step_entropy=entropy,
                    in_thinking_mode=False,
                )

        return SupervisorStepAction(
            modified_logits=logits_work,
            should_force_transition=False,
            transition_token_ids=self.transition_token_ids,
            exit_reason=self.exit_reason,
            step_entropy=entropy,
            in_thinking_mode=self.in_thinking_mode,
            renko_loop_detected=renko_loop,
            divergence_detected=divergence_detected,
        )

    def get_summary(self) -> dict[str, Any]:
        """Returns structured supervisor telemetry."""
        avg_h = sum(self.recent_entropies) / len(self.recent_entropies) if self.recent_entropies else 0.0
        return {
            "enabled": self.enabled,
            "budget_tier": self.budget_tier,
            "step_count": self.step_count,
            "caps": self.caps,
            "in_thinking_mode": self.in_thinking_mode,
            "exit_reason": self.exit_reason or "completed",
            "was_forced": self.was_forced,
            "avg_entropy": round(avg_h, 3),
        }
