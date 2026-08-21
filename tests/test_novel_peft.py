"""Unit tests for novel_peft weight folding algebra, FoldableExpert, and WeightFoldingEngine."""

import pytest
import torch
import torch.nn as nn
from runtime.novel_peft import FoldableExpert, WeightFoldingEngine


class MockTransformerBlock(nn.Module):
    """Synthetic multi-layer module on CPU to test exact weight folding and restoration."""

    def __init__(self, in_features: int = 32, out_features: int = 64):
        super().__init__()
        self.q_proj = nn.Linear(in_features, out_features, bias=False)
        self.k_proj = nn.Linear(in_features, out_features, bias=False)
        self.v_proj = nn.Linear(in_features, out_features, bias=False)
        # Initialize with known deterministic values
        nn.init.orthogonal_(self.q_proj.weight)
        nn.init.orthogonal_(self.k_proj.weight)
        nn.init.orthogonal_(self.v_proj.weight)


def create_synthetic_expert(name: str, rank: int = 8, in_features: int = 32, out_features: int = 64, alpha: float = 128.0) -> FoldableExpert:
    scaling = alpha / rank
    # FoldableExpert expects U: (out_features, rank), V: (rank, in_features)
    # such that U @ V is (out_features, in_features)
    factors = {
        "q_proj.weight": (torch.randn(out_features, rank), torch.randn(rank, in_features)),
        "k_proj.weight": (torch.randn(out_features, rank), torch.randn(rank, in_features)),
    }
    return FoldableExpert(factors, scaling=scaling, name=name)


def test_foldable_expert_properties():
    """Verify FoldableExpert scaling computation and byte tracking."""
    expert = create_synthetic_expert("test_exp", rank=8, in_features=32, out_features=64, alpha=128.0)
    assert expert.name == "test_exp"
    assert expert.scaling == 16.0  # 128 / 8
    assert expert.nbytes > 0
    assert "q_proj.weight" in expert.factors
    assert "k_proj.weight" in expert.factors

    # Test device/dtype migration
    expert.to(torch.device("cpu"), torch.float64)
    u, v = expert.factors["q_proj.weight"]
    assert u.dtype == torch.float64
    assert v.dtype == torch.float64


def test_weight_folding_exact_math_and_restoration():
    """Verify in-place weight mutation W = W0 + scaling * (U @ V) and bit-exact pristine restore."""
    model = MockTransformerBlock(in_features=32, out_features=64)
    q_orig = model.q_proj.weight.clone()
    k_orig = model.k_proj.weight.clone()
    v_orig = model.v_proj.weight.clone()

    expert = create_synthetic_expert("expert_1", rank=8, in_features=32, out_features=64, alpha=64.0)
    u_q, v_q = expert.factors["q_proj.weight"]
    expected_delta_q = expert.scaling * (u_q @ v_q)

    engine = WeightFoldingEngine(model, [expert], keep_pristine=True)

    # 1. Activate expert
    engine.activate(expert)
    assert engine.active == "expert_1"
    # q_proj and k_proj should be folded
    assert torch.allclose(model.q_proj.weight, q_orig + expected_delta_q, atol=1e-5)
    # v_proj was untouched by this expert and must remain exactly equal to v_orig
    assert torch.equal(model.v_proj.weight, v_orig)

    # 2. Restore to base
    engine.restore()
    assert engine.active is None
    # Must be BIT-EXACT (zero float drift)
    assert torch.equal(model.q_proj.weight, q_orig)
    assert torch.equal(model.k_proj.weight, k_orig)
    assert torch.equal(model.v_proj.weight, v_orig)


def test_multi_expert_additive_stacking():
    """Verify that activate_many([e1, e2]) additively composes deltas linearly."""
    model = MockTransformerBlock(in_features=32, out_features=64)
    q_orig = model.q_proj.weight.clone()

    e1 = create_synthetic_expert("e1", rank=8, in_features=32, out_features=64, alpha=32.0)
    e2 = create_synthetic_expert("e2", rank=8, in_features=32, out_features=64, alpha=64.0)

    u1, v1 = e1.factors["q_proj.weight"]
    u2, v2 = e2.factors["q_proj.weight"]
    delta_1 = e1.scaling * (u1 @ v1)
    delta_2 = e2.scaling * (u2 @ v2)

    engine = WeightFoldingEngine(model, [e1, e2], keep_pristine=True)

    # Stack both experts
    engine.activate_many([e1, e2])
    assert engine.active == "e1+e2"
    assert torch.allclose(model.q_proj.weight, q_orig + delta_1 + delta_2, atol=1e-5)

    # Restore bit-exact
    engine.restore()
    assert torch.equal(model.q_proj.weight, q_orig)


def test_weight_folding_unknown_parameter_error():
    """Assert KeyError when an expert references a weight key not present in the model."""
    model = MockTransformerBlock(in_features=32, out_features=64)
    bad_factors = {"nonexistent_layer.weight": (torch.randn(64, 8), torch.randn(8, 32))}
    bad_expert = FoldableExpert(bad_factors, scaling=1.0, name="bad_expert")

    with pytest.raises(KeyError, match="is not a parameter of the model"):
        WeightFoldingEngine(model, [bad_expert])


def test_weight_folding_dimension_mismatch_error():
    """Assert ValueError when factor matrix inner dimensions do not match target weight."""
    model = MockTransformerBlock(in_features=32, out_features=64)
    # Target weight is (64, 32); provide (128, 8) and (8, 32) producing delta (128, 32)
    mismatched_factors = {"q_proj.weight": (torch.randn(128, 8), torch.randn(8, 32))}
    bad_expert = FoldableExpert(mismatched_factors, scaling=1.0, name="mismatched_expert")

    with pytest.raises(ValueError, match="delta .* != weight"):
        WeightFoldingEngine(model, [bad_expert])


def test_activate_many_surgical_stacking():
    """Verify that activate_many(scale_mode='surgical') applies POET channel masks on conflict modules."""
    from runtime.novel_peft import compute_surgical_notch_masks

    # Create mock transformer with attention (q_proj) and MLP (down_proj)
    class MockBlockWithMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 64, bias=False)
            self.down_proj = nn.Linear(32, 64, bias=False)

    model = MockBlockWithMLP()
    q_orig = model.q_proj.weight.clone()
    down_orig = model.down_proj.weight.clone()

    # Create two experts with conflicting channel 7 in down_proj
    u1_q, v1_q = torch.randn(64, 8), torch.randn(8, 32)
    u2_q, v2_q = torch.randn(64, 8), torch.randn(8, 32)

    u1_mlp, v1_mlp = torch.randn(64, 8), torch.randn(8, 32)
    u2_mlp, v2_mlp = torch.randn(64, 8), torch.randn(8, 32)
    # Inject collision at channel 7
    u1_mlp[7, :] = u2_mlp[7, :] = 10.0

    e1 = FoldableExpert({"q_proj.weight": (u1_q, v1_q), "down_proj.weight": (u1_mlp, v1_mlp)}, scaling=16.0, name="e1")
    e2 = FoldableExpert({"q_proj.weight": (u2_q, v2_q), "down_proj.weight": (u2_mlp, v2_mlp)}, scaling=16.0, name="e2")

    engine = WeightFoldingEngine(model, [e1, e2], keep_pristine=True)

    # 1. Activate in surgical mode
    engine.activate_many([e1, e2], scale_mode="surgical")
    assert engine.active == "e1+e2"

    # Attention must remain 100% full alpha without any notching
    delta_q = 16.0 * (u1_q @ v1_q) + 16.0 * (u2_q @ v2_q)
    assert torch.allclose(model.q_proj.weight, q_orig + delta_q, atol=1e-5)

    # Conflict channel 7 in down_proj must have ZERO delta contribution (notched)
    delta_down_row7 = model.down_proj.weight[7, :] - down_orig[7, :]
    assert torch.allclose(delta_down_row7, torch.zeros_like(delta_down_row7), atol=1e-6), "Channel 7 must be zeroed by surgical notch mask"

    # Non-conflicting channel 0 must have active non-zero delta
    delta_down_row0 = model.down_proj.weight[0, :] - down_orig[0, :]
    assert torch.norm(delta_down_row0) > 1e-3, "Clean channel 0 must be active"

    # 2. Restore bit-exact
    engine.restore()
    assert torch.equal(model.q_proj.weight, q_orig)
    assert torch.equal(model.down_proj.weight, down_orig)
    assert engine.max_drift() == 0.0

