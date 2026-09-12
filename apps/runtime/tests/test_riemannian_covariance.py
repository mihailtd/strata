"""Riemannian SPD geometry -- primitives, and the invariance that makes it mean anything.

The tests that matter here are the invariance ones. The first version of this
module passed a full suite of "does matrix_log invert matrix_exp" tests while
computing a distance that changed when you rotated an adapter's rank basis --
i.e. a distance that was not a function of the adapter. Testing the primitives
proves the linear algebra; only the invariance tests prove the MEASUREMENT.
"""

import numpy as np
import pytest
import torch
from runtime.riemannian_covariance import (
    airm_components,
    joint_subspace_operators,
    ledoit_wolf_from_samples,
    log_euclidean_distance,
    matrix_inv_sqrt,
    matrix_log,
    matrix_sqrt,
    riemannian_affine_invariant_distance,
    spherical_shrinkage,
)


def spd(dim: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, dim, generator=g, dtype=torch.float64)
    return A @ A.T + 0.5 * torch.eye(dim, dtype=torch.float64)


def lora(d_out: int, d_in: int, r: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    U = torch.randn(d_out, r, generator=g, dtype=torch.float64)
    V = torch.randn(r, d_in, generator=g, dtype=torch.float64)
    return U, V


# --------------------------------------------------------------------------- primitives


def test_matrix_log_inverts_exp():
    A = spd()
    L = matrix_log(A)
    ev, evec = torch.linalg.eigh(L)
    assert torch.norm(A - evec @ torch.diag(torch.exp(ev)) @ evec.T) < 1e-8


def test_matrix_sqrt_and_inv_sqrt():
    A = spd()
    S = matrix_sqrt(A)
    assert torch.norm(S @ S - A) < 1e-8
    assert torch.norm(matrix_inv_sqrt(A) @ S - torch.eye(8, dtype=A.dtype)) < 1e-8


def test_airm_is_a_metric():
    A, B, C = spd(seed=1), spd(seed=2), spd(seed=3)
    assert riemannian_affine_invariant_distance(A, A) < 1e-9
    d_ab = riemannian_affine_invariant_distance(A, B)
    assert abs(d_ab - riemannian_affine_invariant_distance(B, A)) < 1e-9
    assert d_ab > 0
    assert d_ab <= (riemannian_affine_invariant_distance(A, C) + riemannian_affine_invariant_distance(C, B) + 1e-9)


def test_airm_congruence_invariance():
    """d_R(MAM^T, MBM^T) = d_R(A, B). This is the property the shared-basis
    construction leans on: the arbitrary orientation of Q cancels."""
    A, B = spd(seed=1), spd(seed=2)
    g = torch.Generator().manual_seed(7)
    M = torch.randn(8, 8, generator=g, dtype=torch.float64) + 3 * torch.eye(8, dtype=torch.float64)
    d0 = riemannian_affine_invariant_distance(A, B)
    d1 = riemannian_affine_invariant_distance(M @ A @ M.T, M @ B @ M.T)
    assert abs(d0 - d1) < 1e-6 * max(1.0, d0)


def test_airm_inversion_invariance():
    A, B = spd(seed=1), spd(seed=2)
    d0 = riemannian_affine_invariant_distance(A, B)
    d1 = riemannian_affine_invariant_distance(torch.linalg.inv(A), torch.linalg.inv(B))
    assert abs(d0 - d1) < 1e-6 * d0


def test_airm_components_decompose_exactly():
    A, B = spd(seed=4), spd(seed=5)
    c = airm_components(A, B)
    assert abs(c["total"] - riemannian_affine_invariant_distance(A, B)) < 1e-9
    assert abs(c["total"] ** 2 - (c["scale"] ** 2 + c["shape"] ** 2)) < 1e-9


def test_scaling_a_matrix_is_pure_scale_no_shape():
    """d_R(A, cA) is entirely the scale component: same directions, more of them."""
    A = spd(seed=6)
    c = airm_components(A, 4.0 * A)
    assert c["shape"] < 1e-8
    assert abs(c["scale"] - abs(np.log(4.0)) * np.sqrt(8)) < 1e-8


# --------------------------------------------------------------------------- shrinkage


def test_spherical_shrinkage_makes_singular_matrices_pd():
    U, V = lora(64, 64, 8, seed=0)
    S = (U @ V) @ (U @ V).T
    assert torch.linalg.eigvalsh(S).min() < 1e-8  # rank 8 in 64 dims
    assert torch.linalg.eigvalsh(spherical_shrinkage(S, 0.05)).min() > 0


def test_spherical_shrinkage_preserves_trace():
    S = spd()
    for d in (0.0, 0.3, 1.0):
        assert abs(float(torch.trace(spherical_shrinkage(S, d)) - torch.trace(S))) < 1e-8


def test_spherical_shrinkage_commutes_with_orthogonal_conjugation():
    """Required for the shared-basis construction to be well defined."""
    S = spd()
    g = torch.Generator().manual_seed(11)
    O, _ = torch.linalg.qr(torch.randn(8, 8, generator=g, dtype=torch.float64))
    lhs = spherical_shrinkage(O.T @ S @ O, 0.2)
    rhs = O.T @ spherical_shrinkage(S, 0.2) @ O
    assert torch.norm(lhs - rhs) < 1e-9


def test_spherical_shrinkage_rejects_delta_out_of_range():
    with pytest.raises(ValueError):
        spherical_shrinkage(spd(), 1.5)


def test_ledoit_wolf_delta_shrinks_more_when_n_is_small():
    """The whole point of LW: less data -> trust the sample covariance less.

    The population covariance has to have STRUCTURE for this to be visible. On
    white noise the spherical target is already correct, LW returns delta = 1.0
    at every n, and the test would compare 1.0 to 1.0 -- which is the estimator
    behaving properly, not a signal.
    """
    g = torch.Generator().manual_seed(3)
    p, k = 40, 5
    load = torch.randn(k, p, generator=g, dtype=torch.float64)  # factor model

    def draw(n):
        f = torch.randn(n, k, generator=g, dtype=torch.float64)
        return f @ load + 0.3 * torch.randn(n, p, generator=g, dtype=torch.float64)

    _, d_big = ledoit_wolf_from_samples(draw(4000))
    _, d_small = ledoit_wolf_from_samples(draw(50))
    assert 0.0 <= d_big <= 1.0 and 0.0 <= d_small <= 1.0
    assert d_small > d_big


def test_ledoit_wolf_delta_is_one_on_white_noise():
    """Target already correct -> shrink all the way to it. Documents the case
    that made the previous test vacuous."""
    g = torch.Generator().manual_seed(5)
    _, delta = ledoit_wolf_from_samples(torch.randn(4000, 40, generator=g, dtype=torch.float64))
    assert delta > 0.99


def test_ledoit_wolf_output_is_pd_when_n_below_p():
    g = torch.Generator().manual_seed(4)
    X = torch.randn(10, 40, generator=g, dtype=torch.float64)  # n < p, S singular
    Sigma, delta = ledoit_wolf_from_samples(X)
    assert delta > 0
    assert torch.linalg.eigvalsh(Sigma.double()).min() > 0


# --------------------------------------------------------------------------- THE ONES THAT MATTER


def test_airm_invariant_to_rank_basis():
    """U -> U R, V -> R^T V leaves dW bit-identical, so d_R MUST NOT MOVE.

    This is the regression guard for the defect that made DECISIONS.md §55 wrong:
    the old r x r Gramian G = s^2 VV^T + U^T U transforms as R^T G R, so its d_R
    varied by 0.054 on a real adapter pair whose whole reported signal was 0.024.
    """
    Ua, Va = lora(128, 96, 8, seed=1)
    Ub, Vb = lora(128, 96, 8, seed=2)
    base = airm_components(*joint_subspace_operators(Ua, Va, 16.0, Ub, Vb, 16.0)[:2])["total"]

    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        R, _ = torch.linalg.qr(torch.randn(8, 8, generator=g, dtype=torch.float64))
        assert torch.norm((Ub @ R) @ (R.T @ Vb) - Ub @ Vb) < 1e-9  # dW unchanged
        Sa, Sb, _ = joint_subspace_operators(Ua, Va, 16.0, Ub @ R, R.T @ Vb, 16.0)
        assert abs(airm_components(Sa, Sb)["total"] - base) < 1e-6


def test_joint_subspace_self_distance_is_zero():
    Ua, Va = lora(128, 96, 8, seed=1)
    Sa, Sb, k = joint_subspace_operators(Ua, Va, 16.0, Ua, Va, 16.0)
    assert k == 8  # ranges coincide, not 2r
    assert riemannian_affine_invariant_distance(Sa, Sb) < 1e-9


def test_joint_subspace_k_reports_actual_overlap():
    """k = 2r when the two ranges are disjoint, r when they coincide, and
    strictly between when they partially overlap. k is the overlap read-out."""
    Ua, Va = lora(128, 96, 8, seed=1)
    Ub, Vb = lora(128, 96, 8, seed=2)
    assert joint_subspace_operators(Ua, Va, 16.0, Ub, Vb, 16.0)[2] == 16
    assert joint_subspace_operators(Ua, Va, 16.0, Ua, Va, 16.0)[2] == 8
    Uc = torch.cat([Ua[:, :4], Ub[:, :4]], dim=1)  # half shared with Ua
    assert joint_subspace_operators(Ua, Va, 16.0, Uc, Vb, 16.0)[2] == 12


def test_orthogonal_subspaces_are_farther_than_overlapping_ones():
    """Sanity on direction: sharing a subspace must read as closer."""
    Ua, Va = lora(128, 96, 8, seed=1)
    Ub, Vb = lora(128, 96, 8, seed=2)
    shared = airm_components(*joint_subspace_operators(Ua, Va, 16.0, Ua, Vb, 16.0)[:2])["total"]
    disjoint = airm_components(*joint_subspace_operators(Ua, Va, 16.0, Ub, Vb, 16.0)[:2])["total"]
    assert shared < disjoint


def test_lerm_agrees_with_airm_only_when_commuting():
    """d_LE == d_R to 4 decimals is a WARNING (shared regulariser dominating),
    not a confirmation -- it holds exactly when the two matrices commute."""
    g = torch.Generator().manual_seed(9)
    O, _ = torch.linalg.qr(torch.randn(8, 8, generator=g, dtype=torch.float64))
    A = O @ torch.diag(torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.9, 0.8, 0.7], dtype=torch.float64)) @ O.T
    B = O @ torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=torch.float64)) @ O.T
    assert abs(riemannian_affine_invariant_distance(A, B) - log_euclidean_distance(A, B)) < 1e-8
    C = spd(seed=12)
    assert abs(riemannian_affine_invariant_distance(A, C) - log_euclidean_distance(A, C)) > 1e-3


# Merged from the former tests/test_benchmark_activation_covariance.py -- same
# module, tested against raw random activations instead of synthetic SPD matrices.


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
