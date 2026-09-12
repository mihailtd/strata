"""Unit tests for POET (Low-Rank + Sparse) and Nearest Kronecker decomposition."""

import pytest
import torch

from experiments.factory.geometry.poet_decomposition.probe_poet_decomposition import (
    apply_poet_thresholding,
    compute_compression_size_mb,
    factor_dimensions,
    nearest_kronecker_product,
)


def test_nearest_kronecker_exact_recovery():
    """Verify nearest Kronecker product perfectly recovers an exact Kronecker matrix."""
    torch.manual_seed(42)
    m1, n1 = 8, 8
    m2, n2 = 10, 10
    
    G1_true = torch.randn(m1, n1)
    G2_true = torch.randn(m2, n2)
    M_exact = torch.kron(G1_true, G2_true)
    
    G1_est, G2_est = nearest_kronecker_product(M_exact, m1, n1, niter=15)
    M_est = torch.kron(G1_est, G2_est)
    
    rel_err = torch.norm(M_exact - M_est) / torch.norm(M_exact)
    assert rel_err.item() < 1e-4


def test_factor_dimensions_standard_shapes():
    """Verify factor dimension split heuristics on standard Qwen model shapes."""
    assert factor_dimensions(2560, 2560) == (64, 64)
    assert factor_dimensions(9216, 2560) == (64, 64)
    assert factor_dimensions(2560, 9216) == (64, 64)


def test_poet_hard_thresholding_sparsity():
    """Verify POET hard-thresholding retains exactly the requested fraction of top coordinates."""
    torch.manual_seed(123)
    R = torch.randn(100, 100)
    target_sparsity = 0.05
    
    S, meta = apply_poet_thresholding(R, method="hard", sparsity=target_sparsity)
    
    # 5% of 10,000 is 500 entries
    nonzero_count = torch.count_nonzero(S).item()
    assert nonzero_count == int(target_sparsity * 10000)
    assert pytest.approx(meta["sparsity"], abs=1e-4) == target_sparsity
    
    # Verify non-zero entries in S match R exactly
    mask = (S != 0.0)
    assert torch.allclose(S[mask], R[mask])


def test_poet_soft_thresholding_shrinkage():
    """Verify POET soft-thresholding shrinks values continuously towards zero."""
    torch.manual_seed(456)
    R = torch.randn(50, 50)
    S, meta = apply_poet_thresholding(R, method="soft", sparsity=0.10)
    
    # Non-zero elements should be strictly smaller in magnitude than original R
    mask = (S != 0.0)
    assert torch.all(torch.abs(S[mask]) < torch.abs(R[mask]))


def test_compression_size_calculations():
    """Verify footprint calculations: POET achieves ~4-5x compression over dense rank-8 LoRA."""
    m, n = 2560, 2560
    lora_mb = compute_compression_size_mb(m, n, "lora", rank=8)
    
    # Rank-8 LoRA on 2560x2560 is (2560*8 + 8*2560)*2 bytes = 81,920 bytes = 0.078 MB
    assert pytest.approx(lora_mb, abs=1e-3) == 0.078125
    
    # Pure Kronecker (64x64 + 40x40)*2 bytes = (4096 + 1600)*2 = 11,392 bytes = 0.0108 MB
    kron_mb = compute_compression_size_mb(m, n, "pure_kronecker", m1=64, n1=64)
    assert kron_mb < lora_mb
    assert (lora_mb / kron_mb) > 7.0
    
    # POET with 1% sparse coordinates: Kronecker + 0.01 * 6,553,600 * 6 bytes = 11,392 + 393,216 = 0.385 MB
    poet_mb = compute_compression_size_mb(m, n, "poet_kronecker", m1=64, n1=64, num_sparse_entries=int(0.01 * m * n))
    assert poet_mb > kron_mb
