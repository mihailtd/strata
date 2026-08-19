"""Riemannian Manifold Geometry for Covariance Matrices & Ledoit-Wolf Shrinkage.

References:
- Chapter 3: §3.3 (Norms and Proximity of Matrices), §3.5 (Log of a Covariance Matrix)
- Chapter 8: §8.1.4 (Ledoit-Wolf Optimal Shrinkage Estimators)
- Bhatia (2007): Positive Definite Matrices (Affine-Invariant Riemannian Metric)
- Arsigny et al. (2006): Log-Euclidean Metrics for Symmetric Positive Definite Matrices

Formulas:
1. Matrix Logarithm:
   log(A) = P * diag(log(lambda_i)) * P^T  for spectral decomposition A = P * Lambda * P^T
2. Affine-Invariant Riemannian Distance (AIRM):
   d_R(Sigma_1, Sigma_2) = ||log(Sigma_1^{-1/2} Sigma_2 Sigma_1^{-1/2})||_F
3. Log-Euclidean Riemannian Distance:
   d_LE(Sigma_1, Sigma_2) = ||log(Sigma_1) - log(Sigma_2)||_F
4. Ledoit-Wolf Analytical Optimal Shrinkage:
   Sigma_LW = (1 - delta) * S + delta * F
   where F = (tr(S)/p) * I is the spherical shrinkage target.
"""

from __future__ import annotations

import torch


def matrix_sym_eigh(A: torch.Tensor, eps: float = 1e-7) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric eigendecomposition of positive-definite matrix A with eigenvalue floor."""
    # Ensure exact symmetry
    A_sym = 0.5 * (A + A.T)
    evals, evecs = torch.linalg.eigh(A_sym)
    evals = torch.clamp(evals, min=eps)
    return evals, evecs


def matrix_log(A: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Computes the matricial logarithm log(A) = P * diag(log(evals)) * P^T (Ch 3 §3.5)."""
    evals, evecs = matrix_sym_eigh(A, eps=eps)
    log_evals = torch.log(evals)
    return evecs @ torch.diag_embed(log_evals) @ evecs.T


def matrix_sqrt(A: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Computes matrix square root A^{1/2} = P * diag(sqrt(evals)) * P^T."""
    evals, evecs = matrix_sym_eigh(A, eps=eps)
    sqrt_evals = torch.sqrt(evals)
    return evecs @ torch.diag_embed(sqrt_evals) @ evecs.T


def matrix_inv_sqrt(A: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Computes matrix inverse square root A^{-1/2} = P * diag(1/sqrt(evals)) * P^T."""
    evals, evecs = matrix_sym_eigh(A, eps=eps)
    inv_sqrt_evals = 1.0 / torch.sqrt(evals)
    return evecs @ torch.diag_embed(inv_sqrt_evals) @ evecs.T


def ledoit_wolf_shrinkage(
    S: torch.Tensor,
    target: torch.Tensor | None = None,
    shrinkage_intensity: float | None = None,
) -> tuple[torch.Tensor, float]:
    """Computes Ledoit-Wolf optimal shrinkage estimator Sigma_LW = (1 - delta)*S + delta*F.

    Args:
        S: Sample covariance or Gramian matrix [p, p].
        target: Shrinkage target F. Defaults to spherical CS(0, mu): mu * I_p.
        shrinkage_intensity: Optional pre-fixed delta in [0, 1]. If None, computes analytical optimal delta.

    Returns:
        (Sigma_LW, delta): Shrunk positive-definite matrix and shrinkage intensity.
    """
    p = S.shape[0]
    S_sym = 0.5 * (S + S.T)

    # Spherical target: F = mu * I_p, where mu = tr(S) / p
    mu = float(torch.trace(S_sym).item() / max(1, p))
    if target is None:
        F = mu * torch.eye(p, dtype=S.dtype, device=S.device)
    else:
        F = target

    if shrinkage_intensity is not None:
        delta = float(max(0.0, min(1.0, shrinkage_intensity)))
    else:
        # Analytical optimal Ledoit-Wolf shrinkage intensity estimation
        # gamma_sq = ||S - F||_F^2
        gamma_sq = float(torch.norm(S_sym - F, p="fro").item() ** 2)
        if gamma_sq < 1e-12:
            delta = 0.0
        else:
            # Estimate asymptotic variance pi = sum Var(s_ij)
            # For Gramian / sample outer product matrices:
            pi_hat = float(torch.sum((S_sym - torch.diag(torch.diag(S_sym))) ** 2).item())
            delta = float(max(0.0, min(1.0, pi_hat / (gamma_sq * max(1, p)))))

    Sigma_LW = (1.0 - delta) * S_sym + delta * F
    return Sigma_LW, delta


def riemannian_affine_invariant_distance(
    Sigma_A: torch.Tensor,
    Sigma_B: torch.Tensor,
    eps: float = 1e-7,
) -> float:
    """Computes Affine-Invariant Riemannian Metric (AIRM) geodesic distance between SPD matrices:

    d_R(Sigma_A, Sigma_B) = || log( Sigma_A^{-1/2} * Sigma_B * Sigma_A^{-1/2} ) ||_F

    Invariants:
    1. Congruence invariance: d_R(M*A*M^T, M*B*M^T) = d_R(A, B) for any invertible M.
    2. Inversion invariance: d_R(A^-1, B^-1) = d_R(A, B).
    3. Scale invariance under coordinate change.
    """
    A_inv_sqrt = matrix_inv_sqrt(Sigma_A, eps=eps)
    M = A_inv_sqrt @ Sigma_B @ A_inv_sqrt
    log_M = matrix_log(M, eps=eps)
    return float(torch.norm(log_M, p="fro").item())


def log_euclidean_distance(
    Sigma_A: torch.Tensor,
    Sigma_B: torch.Tensor,
    eps: float = 1e-7,
) -> float:
    """Computes Log-Euclidean Riemannian Metric (LERM) geodesic distance:

    d_LE(Sigma_A, Sigma_B) = || log(Sigma_A) - log(Sigma_B) ||_F

    Provides a bi-invariant metric on the Lie group of positive definite matrices.
    Computationally faster than AIRM while preserving scale-invariance and non-Euclidean curvature.
    """
    log_A = matrix_log(Sigma_A, eps=eps)
    log_B = matrix_log(Sigma_B, eps=eps)
    return float(torch.norm(log_A - log_B, p="fro").item())


def compute_adapter_gramian(
    U: torch.Tensor,
    V: torch.Tensor,
    scaling: float = 16.0,
    eps: float = 1e-5,
    apply_ledoit_wolf: bool = True,
) -> tuple[torch.Tensor, float]:
    """Computes well-conditioned SPD Gramian representation for LoRA adapter delta W = scaling * U @ V^T.

    Since W in R^{d_out x d_in} is low rank (rank r=8), the direct inner Gramian G = V @ V^T or
    outer Gramian U @ U^T defines the subspace curvature.
    We compute the r x r subspace Gramian: G = scaling^2 * (V^T @ V) or outer (U^T @ U).

    Returns:
        (Sigma_LW, delta_intensity)
    """
    # Low-rank factor Gramian in R^{r x r}
    # captures the principal directional energies of the adapter
    G = (scaling**2) * (V.T @ V) + (U.T @ U)
    if apply_ledoit_wolf:
        Sigma_LW, delta = ledoit_wolf_shrinkage(G)
        # Add slight regularizer floor for numerical stability
        Sigma_LW = Sigma_LW + eps * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
        return Sigma_LW, delta
    else:
        G_reg = G + eps * torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
        return G_reg, 0.0
