"""Unit tests for FlashNorm-style weight folding, unfolding, and adapter factor scaling."""

from __future__ import annotations

from typing import Any, cast

import torch
from fused_norm import (
    ExactRMSNorm,
    ScaleFreeRMSNorm,
    fold_rmsnorm_into_linear,
    scale_expert_factors_for_folded_norms,
    unfold_rmsnorm,
)
from novel_peft import FoldableExpert, WeightFoldingEngine
from torch import nn


class MockAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size // 2, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size // 2, bias=False)
        self.o_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        return self.o_proj(q)


class MockMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class MockDecoderLayer(nn.Module):
    input_layernorm: Any
    self_attn: MockAttention
    post_attention_layernorm: Any
    mlp: MockMLP

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.input_layernorm = ExactRMSNorm(hidden_size, unit_offset=True)
        self.self_attn = MockAttention(hidden_size)
        self.post_attention_layernorm = ExactRMSNorm(hidden_size, unit_offset=True)
        self.mlp = MockMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        h = x + self.self_attn(self.input_layernorm(x))
        # Pre-norm MLP
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class MockBackbone(nn.Module):
    layers: list[MockDecoderLayer]
    norm: ExactRMSNorm

    def __init__(self, num_layers: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.layers = cast(
            Any,
            nn.ModuleList([MockDecoderLayer(hidden_size, intermediate_size) for _ in range(num_layers)]),
        )
        self.norm = ExactRMSNorm(hidden_size, unit_offset=True)


class MockModel(nn.Module):
    model: MockBackbone
    lm_head: nn.Linear

    def __init__(self, num_layers: int = 2, hidden_size: int = 64, intermediate_size: int = 128):
        super().__init__()
        self.model = MockBackbone(num_layers, hidden_size, intermediate_size)
        self.lm_head = nn.Linear(hidden_size, 32, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.norm(x)
        return self.lm_head(x)


def test_scale_free_norm_basic():
    """ScaleFreeRMSNorm produces identical output to ExactRMSNorm when gamma=0 (scale=1)."""
    hidden_size = 128
    exact = ExactRMSNorm(hidden_size, unit_offset=True)
    scale_free = ScaleFreeRMSNorm(hidden_size, unit_offset=True)

    x = torch.randn(2, 4, hidden_size)
    y_exact = exact(x)
    y_sf = scale_free(x)

    assert torch.allclose(y_exact, y_sf, atol=1e-6)


def test_fold_rmsnorm_numerical_equality():
    """Folding RMSNorm scales into Linears produces mathematically identical output."""
    torch.manual_seed(42)
    hidden_size = 64
    intermediate_size = 128
    model = MockModel(num_layers=2, hidden_size=hidden_size, intermediate_size=intermediate_size)

    # Initialize non-trivial weights on norms
    for layer in model.model.layers:
        layer.input_layernorm.weight.data.uniform_(-0.2, 0.2)
        layer.post_attention_layernorm.weight.data.uniform_(-0.2, 0.2)
    model.model.norm.weight.data.uniform_(-0.2, 0.2)

    x = torch.randn(2, 8, hidden_size)

    with torch.no_grad():
        y_orig = model(x)

    # Apply FlashNorm fold
    count = fold_rmsnorm_into_linear(model)
    assert count == 5  # 2 layers * 2 + 1 final = 5

    # Check that norms are now ScaleFreeRMSNorm
    for layer in model.model.layers:
        assert isinstance(layer.input_layernorm, ScaleFreeRMSNorm)
        assert isinstance(layer.post_attention_layernorm, ScaleFreeRMSNorm)
    assert isinstance(model.model.norm, ScaleFreeRMSNorm)

    with torch.no_grad():
        y_folded = model(x)

    # Outputs must be mathematically identical (rel L2 < 1e-5)
    rel_l2 = torch.linalg.norm(y_orig - y_folded) / torch.linalg.norm(y_orig)
    assert rel_l2.item() < 1e-5
    assert torch.allclose(y_orig, y_folded, rtol=1e-4, atol=1e-4)


def test_unfold_rmsnorm_reversibility():
    """Unfolding fully restores the original model state and weights."""
    torch.manual_seed(42)
    hidden_size = 64
    intermediate_size = 128
    model = MockModel(num_layers=2, hidden_size=hidden_size, intermediate_size=intermediate_size)

    for layer in model.model.layers:
        layer.input_layernorm.weight.data.uniform_(-0.2, 0.2)
        layer.post_attention_layernorm.weight.data.uniform_(-0.2, 0.2)
    model.model.norm.weight.data.uniform_(-0.2, 0.2)

    x = torch.randn(2, 8, hidden_size)

    with torch.no_grad():
        y_orig = model(x)

    # Fold
    fold_rmsnorm_into_linear(model)

    # Unfold
    unfolded_count = unfold_rmsnorm(model)
    assert unfolded_count == 5

    # Verify restored modules
    for layer in model.model.layers:
        assert isinstance(layer.input_layernorm, ExactRMSNorm)
        assert isinstance(layer.post_attention_layernorm, ExactRMSNorm)
    assert isinstance(model.model.norm, ExactRMSNorm)

    with torch.no_grad():
        y_restored = model(x)

    rel_l2 = torch.linalg.norm(y_orig - y_restored) / torch.linalg.norm(y_orig)
    assert rel_l2.item() < 1e-5
    assert torch.allclose(y_orig, y_restored, rtol=1e-4, atol=1e-4)


def test_fold_weights_disabled_toggle():
    """fold_weights=False leaves model untouched."""
    model = MockModel()
    count = fold_rmsnorm_into_linear(model, fold_weights=False)
    assert count == 0
    assert isinstance(model.model.layers[0].input_layernorm, ExactRMSNorm)


def test_adapter_factor_scaling():
    """Adapter delta scaling by (1+γ) ensures folded engine output matches adapted model."""
    torch.manual_seed(42)
    hidden_size = 64
    intermediate_size = 128
    rank = 4
    scaling = 2.0

    model = MockModel(num_layers=1, hidden_size=hidden_size, intermediate_size=intermediate_size)
    layer = model.model.layers[0]
    layer.input_layernorm.weight.data.uniform_(-0.2, 0.2)

    # Create dummy expert factors for q_proj: (out, rank) and (rank, in)
    u_q = torch.randn(hidden_size * 2, rank)
    v_q = torch.randn(rank, hidden_size)

    factors = {"model.layers.0.self_attn.q_proj.weight": (u_q.clone(), v_q.clone())}
    expert = FoldableExpert(factors, scaling=scaling, name="test_expert")

    # Fold base model
    fold_rmsnorm_into_linear(model)

    # Scale expert factors
    scale_expert_factors_for_folded_norms(model, [expert])

    # WeightFoldingEngine activate
    engine = WeightFoldingEngine(model, [expert], keep_pristine=True)
    engine.activate(expert)

    # Verify that live weight matches the folded adapted weight
    gamma = 1.0 + layer.input_layernorm._original_weight.float()
    w_base_orig = engine.pristine["model.layers.0.self_attn.q_proj.weight"].float() / gamma.unsqueeze(0)
    w_adapted_expected = (w_base_orig + scaling * (u_q @ v_q)) * gamma.unsqueeze(0)

    w_live = layer.self_attn.q_proj.weight.data.float()
    rel_l2 = torch.linalg.norm(w_live - w_adapted_expected) / torch.linalg.norm(w_adapted_expected)
    assert rel_l2.item() < 1e-5


def test_flash_norm_9b_dimensions_and_scaling():
    """Verifies FlashNorm folding and LoRA factor scaling on true 9B architecture dimensions.

    Qwen3.5-9B geometry:
      - hidden_size = 4096
      - intermediate_size = 12288
      - rank = 8, alpha = 128 (scaling = 16.0)
    """
    torch.manual_seed(42)
    h_9b = 4096
    m_9b = 12288
    r = 8
    scaling = 16.0

    model = MockModel(num_layers=1, hidden_size=h_9b, intermediate_size=m_9b)
    layer = model.model.layers[0]
    layer.post_attention_layernorm.weight.data.uniform_(-0.15, 0.15)

    u_gate = torch.randn(m_9b, r)
    v_gate = torch.randn(r, h_9b)
    factors = {"model.layers.0.mlp.gate_proj.weight": (u_gate.clone(), v_gate.clone())}
    expert = FoldableExpert(factors, scaling=scaling, name="expert_9b")

    # 1. Fold base model
    folded_count = fold_rmsnorm_into_linear(model)
    assert folded_count == 3  # input_norm, post_attention_norm, final_norm
    assert isinstance(layer.post_attention_layernorm, ScaleFreeRMSNorm)

    # 2. Scale expert factors
    scaled_count = scale_expert_factors_for_folded_norms(model, [expert])
    assert scaled_count == 1

    # 3. Activate via WeightFoldingEngine
    engine = WeightFoldingEngine(model, [expert], keep_pristine=True)
    engine.activate(expert)

    # 4. Verify numerical equivalence on 9B dimensions
    orig_w = layer.post_attention_layernorm._original_weight
    assert isinstance(orig_w, torch.Tensor)
    gamma: torch.Tensor = orig_w.float() + 1.0
    w_base_orig = engine.pristine["model.layers.0.mlp.gate_proj.weight"].float() / gamma.unsqueeze(0)
    w_adapted_expected = (w_base_orig + scaling * (u_gate @ v_gate)) * gamma.unsqueeze(0)

    w_live = layer.mlp.gate_proj.weight.data.float()
    rel_l2 = torch.linalg.norm(w_live - w_adapted_expected) / torch.linalg.norm(w_adapted_expected)
    assert rel_l2.item() < 1e-5
