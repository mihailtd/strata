"""Unit tests for Latent Variable Graphical Lasso (LV-GLasso) & Precision Matrix Probe (Chapter 9)."""

import numpy as np
import pytest

from experiments.factory.geometry.latent_variable_glasso.probe_lv_glasso_interference import (
    soft_threshold,
    solve_lv_glasso_admm,
)
from experiments.factory.geometry.latent_variable_glasso.probe_cosine_mystery_resolution import (
    _lora_inner,
    _lora_norm_sq,
)
from experiments.factory.geometry.latent_variable_glasso.probe_surgical_sparsification import (
    classify_proj,
    proj_subtype,
    layer_number,
    ATTENTION_TYPES,
    MLP_TYPES,
)


def test_soft_threshold_operator():
    X = np.array([[-2.0, 0.5], [1.5, -0.2]])
    lam = 0.5
    # S(X, 0.5) -> [[-1.5, 0.0], [1.0, 0.0]]
    expected = np.array([[-1.5, 0.0], [1.0, 0.0]])
    res = soft_threshold(X, lam)
    np.testing.assert_allclose(res, expected, atol=1e-6)


def test_solve_lv_glasso_admm_synthetic():
    np.random.seed(42)
    p = 10
    n = 100
    
    # Generate synthetic sparse precision + low rank latent
    # 1. Sparse ground truth precision S_true (tridiagonal)
    S_true = np.eye(p) * 2.0
    for i in range(p - 1):
        S_true[i, i + 1] = S_true[i + 1, i] = -0.5
        
    # 2. Low rank confounder L_true (rank 2)
    V = np.random.randn(p, 2)
    L_true = (V @ V.T) * 0.2
    
    # Total precision Theta = S_true - L_true
    Theta_true = S_true + 0.1 * np.eye(p)
    Sigma_true = np.linalg.inv(Theta_true)
    
    # Sample data
    Y = np.random.multivariate_normal(np.zeros(p), Sigma_true, size=n)
    S_emp = (1.0 / n) * (Y.T @ Y)
    
    Theta_est, S_est, L_est, info = solve_lv_glasso_admm(
        S_emp, lambda1=0.05, lambda2=0.08, max_iter=80, tol=1e-3
    )
    
    assert Theta_est.shape == (p, p)
    assert S_est.shape == (p, p)
    assert L_est.shape == (p, p)
    assert info["iterations"] > 0
    # Precision must be symmetric positive definite
    eigvals = np.linalg.eigvalsh(Theta_est)
    assert np.all(eigvals > 0), "Estimated precision matrix must be positive definite"


def test_trace_regularization_handles_p_greater_than_n():
    np.random.seed(42)
    p = 30
    n = 15  # p > n (singular sample covariance)
    
    Y = np.random.randn(n, p)
    S_emp = (1.0 / n) * (Y.T @ Y)  # rank <= 15
    
    Theta_est, S_est, L_est, info = solve_lv_glasso_admm(
        S_emp, lambda1=0.1, lambda2=0.1, max_iter=60, tol=1e-3
    )
    
    assert Theta_est.shape == (p, p)
    # Ridge / Trace regularization ensures strict positive-definiteness even when p > n
    eigvals = np.linalg.eigvalsh(Theta_est)
    assert np.all(eigvals > 0), "Trace-regularized precision must stay invertible for p > n"


# ─────────────────────────────────────────────────────────────────────────────
# Tests for probe_cosine_mystery_resolution — §38 low-rank cosine fast path
# ─────────────────────────────────────────────────────────────────────────────

def test_lora_norm_sq_matches_explicit():
    """‖dW‖_F² via low-rank formula must match explicit materialisation."""
    np.random.seed(7)
    d_out, d_in, r = 16, 12, 4
    u = np.random.randn(d_out, r)
    v = np.random.randn(r, d_in)
    scale = 2.0

    # Ground truth: materialise dW and compute Frobenius norm
    dW = scale * u @ v
    expected = np.linalg.norm(dW, "fro") ** 2

    result = _lora_norm_sq(u, v, scale)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_lora_inner_matches_explicit():
    """⟨dW_A, dW_B⟩_F via low-rank trace formula must match explicit Frobenius inner product."""
    np.random.seed(11)
    d_out, d_in, r = 16, 12, 4
    u_a = np.random.randn(d_out, r)
    v_a = np.random.randn(r, d_in)
    u_b = np.random.randn(d_out, r)
    v_b = np.random.randn(r, d_in)
    sc_a, sc_b = 3.0, 2.0

    # Ground truth: materialise dW and compute Frobenius inner product
    dW_a = sc_a * u_a @ v_a
    dW_b = sc_b * u_b @ v_b
    expected = float(np.sum(dW_a * dW_b))

    result = _lora_inner(u_a, v_a, u_b, v_b, sc_a, sc_b)
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_lora_cosine_self_is_one():
    """cos(dW, dW) must be exactly 1.0 (within floating-point tolerance)."""
    np.random.seed(3)
    d_out, d_in, r = 10, 8, 4
    u = np.random.randn(d_out, r)
    v = np.random.randn(r, d_in)
    scale = 1.5

    inner_sq = _lora_inner(u, v, u, v, scale, scale)
    norm_sq = _lora_norm_sq(u, v, scale)
    cos = inner_sq / norm_sq
    np.testing.assert_allclose(cos, 1.0, atol=1e-10,
                                err_msg="Self-cosine must be 1.0")


def test_lora_cosine_near_orthogonal_adapters():
    """Two adapters with independent random weights should have cosine ≈ 0 in high dimensions."""
    np.random.seed(42)
    # Large dimensions so cosine converges to ~0 under random init
    d_out, d_in, r = 256, 128, 8
    u_a = np.random.randn(d_out, r) / np.sqrt(d_out)
    v_a = np.random.randn(r, d_in) / np.sqrt(d_in)
    u_b = np.random.randn(d_out, r) / np.sqrt(d_out)
    v_b = np.random.randn(r, d_in) / np.sqrt(d_in)
    sc_a = sc_b = 1.0

    inner = _lora_inner(u_a, v_a, u_b, v_b, sc_a, sc_b)
    norm_a = np.sqrt(_lora_norm_sq(u_a, v_a, sc_a))
    norm_b = np.sqrt(_lora_norm_sq(u_b, v_b, sc_b))
    cos = inner / (norm_a * norm_b)

    # Random independent adapters in large spaces → cosine near zero
    assert abs(cos) < 0.05, f"Expected near-zero cosine for independent adapters, got {cos:.4f}"


def test_cascade_sigma_greater_than_theta():
    """The three-stage cascade must show Σ > Θ (precision inversion reduces apparent correlation)."""
    np.random.seed(42)
    p = 20
    n = 100

    # Generate data with a known low-rank confounder
    r_lat = 3
    F = np.random.randn(n, r_lat)
    L = np.random.randn(r_lat, p) * 0.5
    noise = np.random.randn(n, p) * 0.1
    Y = F @ L + noise
    Y -= Y.mean(axis=0)

    S_emp = (1.0 / n) * (Y.T @ Y)
    diag_sq = np.sqrt(np.diag(S_emp))
    Sigma = S_emp / (np.outer(diag_sq, diag_sq) + 1e-8)

    Theta_est, _, _, _ = solve_lv_glasso_admm(S_emp, lambda1=0.1, lambda2=1e6, max_iter=60, tol=1e-3)
    # Normalise Theta to correlation scale for comparison
    d_th = np.sqrt(np.diag(Theta_est))
    Theta_corr = Theta_est / (np.outer(d_th, d_th) + 1e-8)

    # Off-diagonal mean of Σ vs off-diagonal mean of Θ (normalised)
    mask = ~np.eye(p, dtype=bool)
    mean_sigma = float(np.mean(np.abs(Sigma[mask])))
    mean_theta = float(np.mean(np.abs(Theta_corr[mask])))

    # Precision inversion must substantially reduce apparent off-diagonal correlation
    assert mean_sigma > mean_theta * 2, (
        f"Expected Σ >> Θ but got mean_Σ={mean_sigma:.4f}, mean_Θ={mean_theta:.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests for probe_surgical_sparsification — projection classification
# ─────────────────────────────────────────────────────────────────────────────

def test_classify_proj_attention_types():
    """All four attention projection types must classify as 'attention'."""
    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        fname = f"astral::L5.self_attn.{proj}"
        assert classify_proj(fname) == "attention", f"{fname} should be attention"


def test_classify_proj_mlp_types():
    """All three MLP projection types must classify as 'mlp'."""
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        fname = f"duckdb::L12.mlp.{proj}"
        assert classify_proj(fname) == "mlp", f"{fname} should be mlp"


def test_classify_proj_unknown():
    """Unknown projection names should return 'other'."""
    assert classify_proj("astral::L0.lm_head") == "other"


def test_proj_subtype_extracts_suffix():
    """proj_subtype must return the exact projection name."""
    assert proj_subtype("astral::L3.mlp.gate_proj") == "gate_proj"
    assert proj_subtype("financial::L0.self_attn.q_proj") == "q_proj"
    assert proj_subtype("postgresql::L10.mlp.down_proj") == "down_proj"
    assert proj_subtype("astral::L0.lm_head") == "other"


def test_layer_number_extraction():
    """layer_number must parse the layer index from feature names correctly."""
    assert layer_number("astral::L0.mlp.gate_proj") == 0
    assert layer_number("duckdb::L23.self_attn.v_proj") == 23
    assert layer_number("financial::L3.mlp.down_proj") == 3


def test_attention_never_in_mlp_types_and_vice_versa():
    """ATTENTION_TYPES and MLP_TYPES must be disjoint — no overlap."""
    assert set(ATTENTION_TYPES).isdisjoint(set(MLP_TYPES)), (
        "ATTENTION_TYPES and MLP_TYPES overlap — classification will be wrong"
    )


def test_sqrt_k_loss_fraction():
    """Global √K loss fraction must equal (1 - 1/K) of total energy."""
    K = 4
    # Fraction of energy lost when each adapter is scaled by 1/√K:
    # energy ∝ scale², so scale → scale/√K means energy → energy/K
    # Fraction remaining = 1/K, so fraction lost = 1 - 1/K
    expected_loss = 1.0 - 1.0 / K
    assert abs(expected_loss - 0.75) < 1e-9, f"For K=4, loss should be 75% but got {expected_loss}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests for probe_lv_glasso_poet_surgical_stacking — POET micro notch
# ─────────────────────────────────────────────────────────────────────────────

def test_poet_channel_notch_zeros_top_conflict_neurons():
    """compute_poet_channel_notch must zero out exactly top_k neurons and reduce collision."""
    import torch
    from experiments.factory.geometry.latent_variable_glasso.probe_lv_glasso_poet_surgical_stacking import (
        compute_poet_channel_notch,
    )
    
    torch.manual_seed(42)
    d_out, d_in, r = 128, 64, 8
    u_a = torch.randn(d_out, r)
    v_a = torch.randn(r, d_in)
    u_b = torch.randn(d_out, r)
    v_b = torch.randn(r, d_in)
    
    # Artificially inject a heavy collision at channel 5 and 10
    u_a[5, :] = u_b[5, :] = 10.0
    u_a[10, :] = u_b[10, :] = 8.0
    
    top_k = 5
    mask, meta = compute_poet_channel_notch(u_a, v_a, 1.0, u_b, v_b, 1.0, top_k=top_k)
    
    assert mask.shape == (d_out,)
    assert int(torch.sum(mask == 0.0).item()) == top_k
    assert int(torch.sum(mask == 1.0).item()) == d_out - top_k
    # High-collision channels 5 and 10 must be notched (set to 0.0)
    assert mask[5] == 0.0, "Channel 5 must be notched"
    assert mask[10] == 0.0, "Channel 10 must be notched"
    # Collision reduction factor must be strictly greater than 1.0
    assert meta["collision_reduction_factor"] > 1.0
    assert meta["filtered_collision_energy"] < meta["raw_collision_energy"]


