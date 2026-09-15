"""Multi-expert LoRA stacking (dynamic adapter fusion) -- dimensions, math, and server resolution.

Self-sufficient test suite for runtime-triton.
"""

from __future__ import annotations

import torch
from adapter_stacker import DynamicAdapterStacker
from server import ChatMessage, resolve_27b_adapter_id


def test_dual_expert_stacking_dimensions():
    """Verifies that stacking two rank-8 experts yields an exact Rank-16 adapter."""
    stacker = DynamicAdapterStacker()
    fused_dict, fused_cfg = stacker.stack_adapters(
        {"postgresql": 1.0, "python_web": 1.0},
        normalize_weights=False,
    )
    assert fused_cfg["r"] == 16
    assert "model.layers.0.mlp.down_proj.lora_A.weight" in fused_dict
    assert "model.layers.0.mlp.down_proj.lora_B.weight" in fused_dict

    wa = fused_dict["model.layers.0.mlp.down_proj.lora_A.weight"]
    wb = fused_dict["model.layers.0.mlp.down_proj.lora_B.weight"]

    # In native 27B format: A is (in_features, r), B is (r, out_features)
    assert wa.shape == (17408, 16)
    assert wb.shape == (16, 5120)


def test_triple_expert_stacking_dimensions():
    """Verifies that stacking three rank-8 experts yields an exact Rank-24 adapter."""
    stacker = DynamicAdapterStacker()
    fused_dict, fused_cfg = stacker.stack_adapters(
        {"postgresql": 1.0, "duckdb": 1.0, "python_web": 1.0},
        normalize_weights=False,
    )
    assert fused_cfg["r"] == 24
    wa = fused_dict["model.layers.0.mlp.down_proj.lora_A.weight"]
    wb = fused_dict["model.layers.0.mlp.down_proj.lora_B.weight"]

    assert wa.shape == (17408, 24)
    assert wb.shape == (24, 5120)


def test_stacking_mathematical_identity():
    """Mathematically verifies (x @ A_stacked) @ B_stacked == sum(x @ A_k @ B_k)."""
    in_feat, out_feat = 512, 1024
    r1, r2 = 8, 8

    A1 = torch.randn(in_feat, r1, dtype=torch.float64)
    B1 = torch.randn(r1, out_feat, dtype=torch.float64)
    A2 = torch.randn(in_feat, r2, dtype=torch.float64)
    B2 = torch.randn(r2, out_feat, dtype=torch.float64)

    x = torch.randn(4, in_feat, dtype=torch.float64)

    # Separate
    y_sep = (x @ A1) @ B1 + (x @ A2) @ B2

    # Stacked along native dimensions
    A_stacked = torch.cat([A1, A2], dim=1)  # (in_feat, 16)
    B_stacked = torch.cat([B1, B2], dim=0)  # (16, out_feat)
    y_stacked = (x @ A_stacked) @ B_stacked

    diff = torch.max(torch.abs(y_sep - y_stacked)).item()
    assert diff < 1e-12


def test_server_resolve_adapter_stacked():
    """Verifies resolve_27b_adapter_id handles explicit multi-expert strings and dynamic moa."""
    # 1. Explicit stacked spec
    res1, aid1 = resolve_27b_adapter_id("qwen3.8:27b:postgresql+python_web", [])
    assert res1 == "postgresql+python_web"
    assert aid1 == -1

    # 2. Dynamic MoA via prompt
    msg = [ChatMessage(role="user", content="FastAPI with PostgreSQL pgvector and DuckDB")]
    res2, aid2 = resolve_27b_adapter_id("qwen3.8:27b:moa", msg)
    assert res2 is not None
    assert "postgresql" in res2
    assert "python_web" in res2
    assert aid2 == -1
