"""Unit tests for Supercharged 27B Inference Engine."""

import time
import pytest
import torch
from runtime.supercharged_engine import (
    OutlierProtectedW4Linear,
    SuperchargedEngineConfig,
    SuperchargedTransformerBlock,
    Supercharged27BEngine,
)


def test_outlier_protected_linear():
    dev = "cuda:0"
    layer = OutlierProtectedW4Linear(5120, 17408, group_size=128, n_outliers=16, device=dev)
    x = torch.randn((1, 5120), dtype=torch.bfloat16, device=dev)
    out = torch.empty((1, 17408), dtype=torch.bfloat16, device=dev)

    res = layer(x, out=out)
    assert res.shape == (1, 17408)
    assert not torch.isnan(res).any()


def test_supercharged_transformer_block():
    dev = "cuda:0"
    config = SuperchargedEngineConfig(n_layers=2, device=dev)
    block = SuperchargedTransformerBlock(config, layer_idx=0)
    h = torch.randn((1, config.d_model), dtype=torch.bfloat16, device=dev)

    h_out = block(h)
    assert h_out.shape == (1, config.d_model)
    assert not torch.isnan(h_out).any()


def test_supercharged_engine_tree_step():
    dev = "cuda:0"
    # Test with 4 blocks for fast verification
    config = SuperchargedEngineConfig(n_layers=4, device=dev)
    engine = Supercharged27BEngine(config)
    h = torch.randn((1, config.d_model), dtype=torch.bfloat16, device=dev)

    h_next, n_accepted = engine.generate_tree_speculative_step(h)
    assert h_next.shape == (1, config.d_model)
    assert n_accepted >= 1
    print(f"✅ Supercharged Engine Verified: Step generated with {n_accepted} accepted tokens.")


if __name__ == "__main__":
    test_outlier_protected_linear()
    test_supercharged_transformer_block()
    test_supercharged_engine_tree_step()
    print("🎉 ALL SUPERCHARGED ENGINE TESTS PASSED!")
