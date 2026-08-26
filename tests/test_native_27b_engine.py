"""Unit and integration tests for Native 27B Triton Serving Engine."""

import pytest
import torch

from runtime.native_27b_engine import EngineConfig27B, Native27BEngine


def test_native_27b_engine_initialization():
    # Test with scaled down micro-dimensions for fast test execution
    cfg = EngineConfig27B(
        num_layers=4,
        hidden_dim=256,
        ffn_dim=512,
        num_heads_q=4,
        num_heads_kv=2,
        head_dim=64,
        vocab_size=1000,
        dtype=torch.float32,
        device="cpu",
    )
    engine = Native27BEngine(config=cfg)
    assert len(engine.layers) == 4
    assert engine.recurrent_state.shape == (4, 1, 4, 64, 64)


def test_native_27b_single_token_forward():
    cfg = EngineConfig27B(
        num_layers=2,
        hidden_dim=128,
        ffn_dim=256,
        num_heads_q=2,
        num_heads_kv=1,
        head_dim=64,
        vocab_size=500,
        dtype=torch.float32,
        device="cpu",
    )
    engine = Native27BEngine(config=cfg)

    token_id = 42
    logits, next_state = engine.forward_token(token_id)
    assert logits.shape == (1, 500)
    assert next_state.shape == engine.recurrent_state.shape
    assert not torch.isnan(logits).any()


def test_native_27b_speculative_generation():
    cfg = EngineConfig27B(
        num_layers=2,
        hidden_dim=128,
        ffn_dim=256,
        num_heads_q=2,
        num_heads_kv=1,
        head_dim=64,
        vocab_size=500,
        dtype=torch.float32,
        device="cpu",
    )
    engine = Native27BEngine(config=cfg)

    prompt = [10, 20, 30]
    out_tokens = engine.generate_streaming(prompt, max_new_tokens=16)
    assert len(out_tokens) == 16
    assert all(isinstance(t, int) for t in out_tokens)
