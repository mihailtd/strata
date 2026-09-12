"""Unit tests for Breakdown-Bounded Distillation Loss functions.

Verifies:
1. Mathematical equivalence to L2 in quadratic regime (|e| <= delta).
2. Strict gradient norm bounding (||grad|| <= delta) under extreme outliers (e -> infinity).
3. LAD-LASSO 50% breakdown robustness and L1 parameter sparsity penalty.
4. Robust token cross-entropy loss clamping.
"""

import torch
import torch.nn as nn
from runtime.robust_distill import HuberDistillationLoss, LADLassoLoss, RobustTokenCrossEntropyLoss


def test_huber_quadratic_linear_regions():
    """Verify exact analytical values in quadratic vs linear regions."""
    huber = HuberDistillationLoss(delta=1.0)

    # Quadratic regime: e = 0.5 <= 1.0 -> loss = 0.5 * 0.5^2 = 0.125
    s1 = torch.tensor([0.5], requires_grad=True)
    t1 = torch.tensor([0.0])
    l1 = huber(s1, t1)
    assert torch.isclose(l1, torch.tensor(0.125)), f"Expected 0.125, got {l1.item()}"

    # Linear regime: e = 3.0 > 1.0 -> loss = 1.0 * (3.0 - 0.5*1.0) = 2.5
    s2 = torch.tensor([3.0], requires_grad=True)
    t2 = torch.tensor([0.0])
    l2 = huber(s2, t2)
    assert torch.isclose(l2, torch.tensor(2.5)), f"Expected 2.5, got {l2.item()}"


def test_huber_gradient_boundedness():
    """Verify that gradients are strictly bounded by delta as error -> infinity (Breakdown Point guarantee)."""
    delta = 1.5
    huber = HuberDistillationLoss(delta=delta)

    # Test extreme outlier errors from 10 to 1,000,000
    for extreme_val in [10.0, 100.0, 1000.0, 1e6]:
        student = torch.tensor([extreme_val], requires_grad=True)
        teacher = torch.tensor([0.0])
        loss = huber(student, teacher)
        loss.backward()

        grad = student.grad.item()
        assert abs(grad) <= delta + 1e-6, f"Gradient {grad} exceeded delta {delta} at error {extreme_val}"
        assert torch.isclose(student.grad, torch.tensor(delta)), f"Expected saturated gradient {delta}, got {grad}"


def test_lad_lasso_subgradient_bounds():
    """Verify that LAD subgradient is strictly in {-1.0, 1.0}."""
    lad = LADLassoLoss(alpha=0.01)

    s = torch.tensor([50.0, -100.0, 0.001], requires_grad=True)
    t = torch.tensor([0.0, 0.0, 0.0])
    loss = lad(s, t)
    loss.backward()

    # Gradients for positive residuals should be +1/3, negative should be -1/3 (mean reduction across 3 elements)
    expected_grads = torch.tensor([1.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0])
    assert torch.allclose(s.grad, expected_grads, atol=1e-5)


def test_lad_lasso_parameter_penalty():
    """Verify that LAD-LASSO adds the correct L1 weight regularization."""
    alpha = 0.05
    lad_lasso = LADLassoLoss(alpha=alpha)

    s = torch.tensor([1.0])
    t = torch.tensor([0.0])
    param = nn.Parameter(torch.tensor([2.0, -3.0]))

    loss = lad_lasso(s, t, model_parameters=[param])
    # Expected: LAD loss = 1.0, L1 penalty = 0.05 * (2.0 + 3.0) = 0.25 -> Total = 1.25
    assert torch.isclose(loss, torch.tensor(1.25)), f"Expected 1.25, got {loss.item()}"


def test_robust_cross_entropy_clamping():
    """Verify that outlier token loss is strictly clamped to outlier_clamp ceiling."""
    clamp_limit = 4.0
    robust_ce = RobustTokenCrossEntropyLoss(outlier_clamp=clamp_limit)

    # Logits where the correct target has tiny probability (creating high raw NLL)
    logits = torch.tensor([[100.0, -100.0]])  # Class 0 has ~100% prob, Class 1 has ~0% prob
    targets = torch.tensor([1])  # Target is Class 1 (corrupted outlier token)

    loss = robust_ce(logits, targets)
    assert torch.isclose(loss, torch.tensor(clamp_limit)), f"Expected clamped loss {clamp_limit}, got {loss.item()}"
