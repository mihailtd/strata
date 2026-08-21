import pytest
import torch
import numpy as np

from runtime.riemannian_covariance import (
    ledoit_wolf_from_samples,
    riemannian_affine_invariant_distance,
    airm_components,
)
from benchmarks.factory.geometry.riemannian_metric.benchmark_activation_covariance_geodesics import (
    load_eval_prompts,
)


def test_load_eval_prompts():
    shared_suite, domain_prompts = load_eval_prompts(max_prompts_per_domain=3)
    assert len(shared_suite) > 0
    assert len(domain_prompts) == 6
    for d in ["astral", "postgresql", "duckdb", "financial", "python_modern", "python_web"]:
        assert d in domain_prompts
        assert len(domain_prompts[d]) > 0


def test_activation_covariance_ledoit_wolf_n_less_than_p():
    # n=100 tokens, p=256 channels (n < p regime)
    g = torch.Generator().manual_seed(42)
    X = torch.randn(100, 256, generator=g)
    
    Sigma, delta = ledoit_wolf_from_samples(X)
    assert Sigma.shape == (256, 256)
    assert 0.0 < delta <= 1.0
    
    # Positive definiteness check
    evals = torch.linalg.eigvalsh(Sigma)
    assert (evals > 0).all()


def test_activation_airm_scale_vs_shape_decomposition():
    g = torch.Generator().manual_seed(123)
    X1 = torch.randn(150, 128, generator=g)
    X2 = torch.randn(150, 128, generator=g) * 1.5 + 0.2
    
    S1, _ = ledoit_wolf_from_samples(X1)
    S2, _ = ledoit_wolf_from_samples(X2)
    
    comp = airm_components(S1, S2)
    assert comp["total"] > 0
    assert comp["scale"] >= 0
    assert comp["shape"] >= 0
    reconstructed = (comp["scale"] ** 2 + comp["shape"] ** 2) ** 0.5
    assert abs(comp["total"] - reconstructed) < 1e-5


def test_airm_rmsnorm_diagonal_congruence_invariance():
    # d_R(D S1 D, D S2 D) == d_R(S1, S2) for any diagonal matrix D
    g = torch.Generator().manual_seed(77)
    X1 = torch.randn(100, 64, generator=g)
    X2 = torch.randn(100, 64, generator=g)
    
    S1, _ = ledoit_wolf_from_samples(X1)
    S2, _ = ledoit_wolf_from_samples(X2)
    
    d_orig = riemannian_affine_invariant_distance(S1, S2)
    
    # Diagonal scaling (RMSNorm simulation)
    d_vec = torch.exp(torch.randn(64, generator=g) * 0.5)
    D = torch.diag(d_vec)
    
    S1_scaled = D @ S1 @ D
    S2_scaled = D @ S2 @ D
    
    d_scaled = riemannian_affine_invariant_distance(S1_scaled, S2_scaled)
    assert abs(d_orig - d_scaled) < 1e-4
