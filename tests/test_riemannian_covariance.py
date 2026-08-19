"""Unit tests for Riemannian Manifold Geometry and Ledoit-Wolf Shrinkage."""

import pytest
import torch
import torch.nn.functional as F

from gnn_experiment.riemannian_covariance import (
    compute_adapter_gramian,
    ledoit_wolf_shrinkage,
    log_euclidean_distance,
    matrix_inv_sqrt,
    matrix_log,
    matrix_sqrt,
    matrix_sym_eigh,
    riemannian_affine_invariant_distance,
)


def random_spd_matrix(dim: int = 8) -> torch.Tensor:
    """Generates a random strictly positive-definite matrix."""
    A = torch.randn(dim, dim)
    SPD = A @ A.T + 0.1 * torch.eye(dim)
    return SPD


def test_matrix_log_and_exp_invertibility():
    """Verify matrix logarithm correctly inverts matrix exponential on SPD matrices."""
    A = random_spd_matrix(dim=8)
    log_A = matrix_log(A)

    # Reconstruct via exp(log(A))
    evals, evecs = torch.linalg.eigh(log_A)
    exp_log_A = evecs @ torch.diag_embed(torch.exp(evals)) @ evecs.T

    diff = torch.norm(A - exp_log_A, p="fro").item()
    assert diff < 1e-4


def test_matrix_sqrt_and_inv_sqrt():
    """Verify matrix sqrt and inverse sqrt properties."""
    A = random_spd_matrix(dim=8)
    sqrt_A = matrix_sqrt(A)
    inv_sqrt_A = matrix_inv_sqrt(A)

    # sqrt(A) @ sqrt(A) == A
    recon_A = sqrt_A @ sqrt_A
    assert torch.allclose(A, recon_A, atol=1e-4)

    # inv_sqrt(A) @ A @ inv_sqrt(A) == I
    identity_check = inv_sqrt_A @ A @ inv_sqrt_A
    I = torch.eye(8, dtype=A.dtype, device=A.device)
    assert torch.allclose(identity_check, I, atol=1e-4)


def test_ledoit_wolf_shrinkage_regularizes_singular_matrix():
    """Verify Ledoit-Wolf shrinkage restores positive-definiteness on rank-deficient Gramians."""
    # Create rank-2 matrix in R^{16 x 16} (severely rank deficient)
    X = torch.randn(16, 2)
    singular_S = X @ X.T
    min_eig_orig = torch.min(torch.linalg.eigvalsh(singular_S)).item()
    assert min_eig_orig < 1e-5  # Singular

    shrunk_S, delta = ledoit_wolf_shrinkage(singular_S)
    min_eig_shrunk = torch.min(torch.linalg.eigvalsh(shrunk_S)).item()

    assert 0.0 <= delta <= 1.0
    assert min_eig_shrunk > 0.0  # Strictly positive definite!


def test_riemannian_airm_metric_invariants():
    """Verify Affine-Invariant Riemannian Metric satisfies distance axioms and Lie group invariances."""
    A = random_spd_matrix(dim=8).to(torch.float64)
    B = random_spd_matrix(dim=8).to(torch.float64)

    # 1. Identity of indiscernibles
    d_AA = riemannian_affine_invariant_distance(A, A)
    assert abs(d_AA) < 1e-5

    # 2. Symmetry
    d_AB = riemannian_affine_invariant_distance(A, B)
    d_BA = riemannian_affine_invariant_distance(B, A)
    assert abs(d_AB - d_BA) < 1e-5
    assert d_AB > 0.0

    # 3. Congruence Invariance: d_R(M A M^T, M B M^T) == d_R(A, B)
    M = torch.randn(8, 8, dtype=torch.float64)
    while torch.abs(torch.linalg.det(M)) < 0.1:
        M = torch.randn(8, 8, dtype=torch.float64)

    M_A = M @ A @ M.T
    M_B = M @ B @ M.T
    d_congruent = riemannian_affine_invariant_distance(M_A, M_B)
    assert abs(d_AB - d_congruent) < 1e-4

    # 4. Inversion Invariance: d_R(A^-1, B^-1) == d_R(A, B)
    inv_A = torch.linalg.inv(A)
    inv_B = torch.linalg.inv(B)
    d_inv = riemannian_affine_invariant_distance(inv_A, inv_B)
    assert abs(d_AB - d_inv) < 1e-4



def test_log_euclidean_distance_axioms():
    """Verify Log-Euclidean Riemannian metric satisfies metric properties."""
    A = random_spd_matrix(dim=8)
    B = random_spd_matrix(dim=8)

    d_AA = log_euclidean_distance(A, A)
    assert abs(d_AA) < 1e-5

    d_AB = log_euclidean_distance(A, B)
    d_BA = log_euclidean_distance(B, A)
    assert abs(d_AB - d_BA) < 1e-5
    assert d_AB > 0.0


def test_compute_adapter_gramian():
    """Verify compute_adapter_gramian produces well-conditioned SPD matrix."""
    U = torch.randn(2560, 8)
    V = torch.randn(2560, 8)

    G, delta = compute_adapter_gramian(U, V, scaling=16.0, apply_ledoit_wolf=True)
    assert G.shape == (8, 8)
    assert 0.0 <= delta <= 1.0

    # Verify eigenvalues are strictly positive
    evals = torch.linalg.eigvalsh(G)
    assert torch.all(evals > 0)
