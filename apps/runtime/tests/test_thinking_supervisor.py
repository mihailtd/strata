"""Unit Tests for ThinkingRuntimeSupervisor (Chapters 3, 4, 5 & 8)."""

import torch
import torch.nn.functional as F
from runtime.thinking_supervisor import ThinkingRuntimeSupervisor


def test_min_thinking_floor_masking():
    sup = ThinkingRuntimeSupervisor(
        think_end_token_id=248069,
        budget_tier="low",  # min=64, target=384, hard_max=512
    )
    assert sup.in_thinking_mode is True

    vocab_size = 250000
    dummy_logits = torch.zeros((1, vocab_size), dtype=torch.float32)

    # Step 1 < min (64): </think> must be masked out with -1e9
    action = sup.process_step(dummy_logits)
    assert action.modified_logits[0, 248069].item() < -1e8
    assert action.should_force_transition is False
    assert action.in_thinking_mode is True


def test_sigmoid_logit_boosting():
    sup = ThinkingRuntimeSupervisor(
        think_end_token_id=248069,
        budget_tier="low",  # min=64, target=384, hard_max=512
    )

    vocab_size = 250000
    dummy_logits = torch.zeros((1, vocab_size), dtype=torch.float32)

    # Fast forward to target step (384)
    sup.step_count = 383
    action = sup.process_step(dummy_logits)
    # At step 384 (target), sigmoid boost should begin
    assert action.modified_logits[0, 248069].item() > 0.5
    assert action.should_force_transition is False


def test_hard_max_budget_force_injection():
    sup = ThinkingRuntimeSupervisor(
        think_end_token_id=248069,
        budget_tier="low",  # min=64, target=384, hard_max=512
    )

    vocab_size = 250000
    dummy_logits = torch.zeros((1, vocab_size), dtype=torch.float32)

    # Fast forward to hard_max step (512)
    sup.step_count = 511
    action = sup.process_step(dummy_logits)
    assert action.should_force_transition is True
    assert action.transition_token_ids == [198, 248069, 271]
    assert action.in_thinking_mode is False
    assert "hard_max_budget" in action.exit_reason


def test_renko_latent_loop_attractor_detection():
    sup = ThinkingRuntimeSupervisor(
        think_end_token_id=248069,
        budget_tier="high",  # target=768, hard_max=896
        loop_similarity_threshold=0.95,
    )

    hidden_dim = 256
    vocab_size = 250000
    dummy_logits = torch.zeros((1, vocab_size), dtype=torch.float32)

    # Generate an attractor vector
    attractor_state = torch.randn(1, hidden_dim)

    # Feed distinct states first to build history
    for i in range(20):
        h = torch.randn(1, hidden_dim)
        action = sup.process_step(dummy_logits, hidden_state=h)
        assert action.should_force_transition is False

    # Inject the attractor state into history
    sup.recent_hidden_states[5] = F.normalize(attractor_state, p=2, dim=-1)

    # Now step with the identical attractor state
    action = sup.process_step(dummy_logits, hidden_state=attractor_state)
    assert action.should_force_transition is True
    assert action.renko_loop_detected is True
    assert "renko_latent_attractor_loop" in action.exit_reason


def test_entropy_cognitive_convergence_boost():
    sup = ThinkingRuntimeSupervisor(
        think_end_token_id=248069,
        budget_tier="medium",  # min=128
        entropy_convergence_threshold=0.35,
    )

    vocab_size = 250000
    # Create peaked (low entropy) logits
    peaked_logits = torch.zeros((1, vocab_size), dtype=torch.float32)
    peaked_logits[0, 100] = 50.0  # Sharp peak -> entropy ~ 0.0

    sup.step_count = 135  # Past min floor (128)
    for _ in range(16):
        action = sup.process_step(peaked_logits)

    # After 16 low-entropy steps, cognitive convergence boost should trigger (+12.0)
    assert action.modified_logits[0, 248069].item() >= 8.0


def test_weibull_hazard_cutoff():
    sup = ThinkingRuntimeSupervisor(
        think_end_token_id=248069,
        budget_tier="low",  # min=64
        weibull_beta=2.2,
        weibull_eta=100.0,  # At t=100, F(t) = 1 - e^-1 = 0.632 > 0.50
    )

    vocab_size = 250000
    dummy_logits = torch.zeros((1, vocab_size), dtype=torch.float32)

    sup.step_count = 99  # Past min floor (64)
    action = sup.process_step(dummy_logits)
    assert action.should_force_transition is True
    assert "weibull_hazard_exceeded" in action.exit_reason
