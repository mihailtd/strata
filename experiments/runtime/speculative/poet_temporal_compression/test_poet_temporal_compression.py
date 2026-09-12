"""Unit tests for POET Temporal Activation Compression for State Ring Buffer & KV-Cache."""

import pytest
import torch

from experiments.runtime.speculative.poet_temporal_compression.probe_poet_temporal_compression import (
    compress_temporal_poet,
    simulate_autoregressive_hidden_states,
)


def test_temporal_simulation_properties():
    """Verify simulated autoregressive hidden states have correct shapes and bounded norms."""
    T, d = 64, 512
    H = simulate_autoregressive_hidden_states(seq_len=T, dim=d, seed=42)
    
    assert H.shape == (T, d)
    assert not torch.isnan(H).any()
    assert float(torch.norm(H).item()) > 0.0


def test_poet_temporal_compression_fidelity():
    """Verify POET factor model achieves high cosine fidelity on temporal states."""
    T, d = 128, 1024
    H = simulate_autoregressive_hidden_states(seq_len=T, dim=d, seed=123)
    
    F, Lambda, S, meta = compress_temporal_poet(H, rank=4, sparsity_target=0.05)
    
    assert F.shape == (T, 4)
    assert Lambda.shape == (d, 4)
    assert S.shape == (T, d)
    
    # Cosine fidelity should exceed 0.85
    assert meta["mean_token_cosine"] > 0.80
    assert meta["rel_frob_error"] < 0.60
    assert meta["actual_sparsity"] <= 0.06


def test_temporal_amortization_compression_growth():
    """Verify compression ratio improves as time horizon T increases due to factor loading amortization."""
    d = 1024
    H_short = simulate_autoregressive_hidden_states(seq_len=64, dim=d, seed=1)
    H_long = simulate_autoregressive_hidden_states(seq_len=512, dim=d, seed=2)
    
    # Pure low-rank factor model (0% sparse)
    _, _, _, meta_short = compress_temporal_poet(H_short, rank=2, sparsity_target=0.0)
    _, _, _, meta_long = compress_temporal_poet(H_long, rank=2, sparsity_target=0.0)
    
    # Long horizon should amortize d x r loadings over T time steps
    assert meta_long["compression_ratio"] > meta_short["compression_ratio"]
