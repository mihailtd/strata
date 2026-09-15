r"""Riemannian geometry on the SPD cone, for comparing low-rank adapter operators.

References
----------
- Ch.3 §3.3 Norms and Proximity of Matrices; §3.5 Log of a Covariance Matrix
- Ch.8 §8.1.4 Ledoit-Wolf Optimal Shrinkage Estimators
- Bhatia (2007), Positive Definite Matrices  -- affine-invariant metric
- Arsigny et al. (2006), Log-Euclidean Metrics for SPD Matrices

    d_R(A, B)  = ||log(A^{-1/2} B A^{-1/2})||_F        affine-invariant (AIRM)
    d_LE(A, B) = ||log A - log B||_F                    log-Euclidean (LERM)

╔══════════════════════════════════════════════════════════════════════════════╗
║ THE MISTAKE THIS MODULE WAS REWRITTEN TO REMOVE -- read before editing        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ A LoRA delta is dW = s * U @ V with U (d_out x r), V (r x d_in). The first     ║
║ version of this module reduced each adapter to an r x r matrix                ║
║                                                                               ║
║     G = s^2 (V V^T) + (U^T U)          <-- WRONG, do not reintroduce          ║
║                                                                               ║
║ and fed G_i, G_j straight into d_R. Two independent things are broken:        ║
║                                                                               ║
║  1. IT IS NOT A FUNCTION OF THE ADAPTER. Rotate the rank basis, U -> U R and  ║
║     V -> R^T V for orthogonal R: dW is bit-identical, the adapter IS the same  ║
║     object -- but G -> R^T G R, so d_R moves. Measured on the real v6 pair    ║
║     (astral, postgresql), down_proj layer 0: rotating postgres' basis alone   ║
║     swung d_R over 0.2768 .. 0.3305. The entire shipped 6x6 off-diagonal      ║
║     spread was 0.2600 .. 0.2837. The basis noise was 2.3x the signal.         ║
║                                                                               ║
║  2. G_i and G_j live in DIFFERENT coordinate frames -- each adapter's own     ║
║     rank-8 basis. Comparing them compares spectra, and the relative           ║
║     ORIENTATION of the two subspaces (the thing that decides whether two      ║
║     experts stack) never enters the arithmetic at all.                        ║
║                                                                               ║
║ The fix is `joint_subspace_operators`: project BOTH adapters into ONE shared  ║
║ orthonormal basis of span(range dW_i U range dW_j), then measure there. AIRM  ║
║ is congruence-invariant, so the arbitrary orientation of that shared basis    ║
║ cancels exactly -- which is why the answer is well defined. Proven in         ║
║ tests/test_riemannian_covariance.py::test_airm_invariant_to_rank_basis.       ║
║                                                                               ║
║ Also: the old `ledoit_wolf_shrinkage(S)` took a Gramian and computed          ║
║ delta = offdiag_energy / (p ||S - F||^2). That has no n in it. Ledoit-Wolf's  ║
║ delta is an estimate of sum Var(s_ij), which needs the SAMPLES. It is not     ║
║ estimable from a Gramian, so that function is now `spherical_shrinkage` and   ║
║ says what it is: a regularizer with a delta you choose and report. The real   ║
║ estimator is `ledoit_wolf_from_samples`, for activation covariances.          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import torch

__all__ = [
    "matrix_sym_eigh",
    "matrix_log",
    "matrix_sqrt",
    "matrix_inv_sqrt",
    "spherical_shrinkage",
    "ledoit_wolf_from_samples",
    "joint_subspace_operators",
    "airm_components",
    "riemannian_affine_invariant_distance",
    "log_euclidean_distance",
]


# ---------------------------------------------------------------------------
# SPD primitives
# ---------------------------------------------------------------------------


def matrix_sym_eigh(A: torch.Tensor, eps: float = 1e-7):
    """Symmetric eigendecomposition with an eigenvalue floor."""
    evals, evecs = torch.linalg.eigh(0.5 * (A + A.T))
    return torch.clamp(evals, min=eps), evecs


def matrix_log(A: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    evals, evecs = matrix_sym_eigh(A, eps)
    return evecs @ torch.diag(torch.log(evals)) @ evecs.T


def matrix_sqrt(A: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    evals, evecs = matrix_sym_eigh(A, eps)
    return evecs @ torch.diag(torch.sqrt(evals)) @ evecs.T


def matrix_inv_sqrt(A: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    evals, evecs = matrix_sym_eigh(A, eps)
    return evecs @ torch.diag(torch.rsqrt(evals)) @ evecs.T


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------


def spherical_shrinkage(S: torch.Tensor, delta: float) -> torch.Tensor:
    r"""Sigma = (1 - delta) S + delta * (tr(S)/p) I -- Ch.8 (8.2) with F = mu I.

    This is the *form* of a Ledoit-Wolf estimator, not the estimator: `delta` is
    supplied, not estimated, because a Gramian carries no sample count and the
    LW optimum (8.14) is a function of Var(s_ij). Any caller MUST report the
    delta it used and show that its conclusion survives a sweep over it -- see
    the delta-sensitivity table in the riemannian_metric benchmark.

    Two properties this relies on downstream:
      * the target mu*I is invariant under orthogonal conjugation, so applying
        this inside a shared subspace does not break AIRM's congruence
        invariance;
      * for a rank-q S in R^{p x p} with q < p it lifts every zero eigenvalue to
        delta*mu > 0, which is the only reason the geodesic exists at all.
    """
    if not 0.0 <= delta <= 1.0:
        raise ValueError(f"delta must be in [0, 1], got {delta}")
    p = S.shape[0]
    S = 0.5 * (S + S.T)
    mu = torch.trace(S) / p
    return (1.0 - delta) * S + delta * mu * torch.eye(p, dtype=S.dtype, device=S.device)


def ledoit_wolf_from_samples(X: torch.Tensor) -> tuple[torch.Tensor, float]:
    r"""The actual Ledoit-Wolf (2004) estimator, spherical target. X is (n, p).

    Needs the samples, because delta estimates sum Var(s_ij):

        S = X^T X / n,   F = (tr S / p) I
        d2   = ||S - F||_F^2 / p
        bbar2 = (1/n^2) sum_t ||x_t x_t^T - S||_F^2 / p,   capped at d2
        delta = bbar2 / d2

    This is the branch that belongs on ACTIVATION covariance Sigma_h = E[h h^T],
    where n tokens vs p=2560 channels is the genuine n < p regime LW exists for.
    It is not usable on a weight Gramian -- there is no n there.
    """
    n, p = X.shape
    X = X.to(torch.float64)
    S = (X.T @ X) / n
    mu = torch.trace(S) / p
    F = mu * torch.eye(p, dtype=S.dtype, device=S.device)
    d2 = (torch.norm(S - F, p="fro") ** 2) / p
    if float(d2) < 1e-30:
        return S.to(torch.float32), 0.0
    # sum_t ||x_t x_t^T - S||^2 = sum_t (||x_t||^4) - n ||S||^2
    sq = (X * X).sum(dim=1)
    bbar2 = ((sq * sq).sum() - n * torch.norm(S, p="fro") ** 2) / (n * n * p)
    bbar2 = torch.clamp(bbar2, min=0.0, max=float(d2))
    delta = float(bbar2 / d2)
    return ((1.0 - delta) * S + delta * F).to(torch.float32), delta


# ---------------------------------------------------------------------------
# The comparison that is actually well defined
# ---------------------------------------------------------------------------


def joint_subspace_operators(
    Ua: torch.Tensor,
    Va: torch.Tensor,
    sa: float,
    Ub: torch.Tensor,
    Vb: torch.Tensor,
    sb: float,
    delta: float = 0.05,
    tol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    r"""Project two LoRA deltas into ONE shared basis and return two SPD operators.

    For dW = s U V the output-side operator is

        Sigma = dW dW^T = s^2 U (V V^T) U^T          (d_out x d_out, rank r)

    which is a function of the adapter alone -- a rank-basis rotation U -> U R,
    V -> R^T V leaves it untouched. Directly it is a 2560 x 2560 matrix of rank 8,
    so every eigenvalue but 8 is zero and any regulariser you add to invert it
    dominates the answer. So restrict to Q, an orthonormal basis of
    span(range dW_a  U  range dW_b), dimension k <= 2r:

        Sigma~ = (Q^T U) [s^2 V V^T] (Q^T U)^T       (k x k)

    Both operators are now in the SAME frame, so their relative orientation is
    what the metric sees. A different valid Q is Q O for orthogonal O, which maps
    both to O^T Sigma~ O; AIRM is congruence-invariant and the spherical target
    is orthogonally invariant, so the distance does not depend on which Q you got.

    Returns (Sigma_a, Sigma_b, k), both shrunk to strictly positive definite.
    """
    Ua, Va, Ub, Vb = (t.to(torch.float64) for t in (Ua, Va, Ub, Vb))
    # Shared output-side subspace. SVD not QR: the ranges can overlap, and then
    # QR hands back near-null columns whose coordinates are numerical dust.
    stacked = torch.cat([Ua, Ub], dim=1)  # (d_out, 2r)
    Q, S, _ = torch.linalg.svd(stacked, full_matrices=False)
    k = int((tol * S[0] < S).sum())
    Q = Q[:, :k]  # (d_out, k)

    def project(U, V, s):
        P = Q.T @ U  # (k, r)
        M = (s * s) * (V @ V.T)  # (r, r), SPD
        return P @ M @ P.T  # (k, k), rank <= r

    Sa = spherical_shrinkage(project(Ua, Va, sa), delta)
    Sb = spherical_shrinkage(project(Ub, Vb, sb), delta)
    return Sa, Sb, k


def airm_components(
    A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12, A_inv_sqrt: torch.Tensor | None = None
) -> dict:
    r"""AIRM distance split into the part that is scale and the part that is shape.

    With L = log(A^{-1/2} B A^{-1/2}) and k = dim,

        tr L      = logdet B - logdet A          pure volume ratio
        scale     = |tr L| / sqrt(k)
        shape     = ||L - (tr L / k) I||_F
        total^2   = scale^2 + shape^2

    Worth having because "financial is far from everything" has two completely
    different readings -- its delta is BIGGER, or its delta points ELSEWHERE --
    and only the second one is a statement about the domain.
    """
    k = A.shape[0]
    # A_inv_sqrt is cacheable: comparing many B against ONE reference A (base vs every
    # expert, at every layer) recomputes the same eigendecomposition once per call.
    # The activation benchmark did 40 of them for 5 distinct matrices.
    Ai = matrix_inv_sqrt(A, eps) if A_inv_sqrt is None else A_inv_sqrt
    L = matrix_log(Ai @ B @ Ai, eps)
    tr = float(torch.trace(L))
    dev = L - (tr / k) * torch.eye(k, dtype=L.dtype, device=L.device)
    scale = abs(tr) / (k**0.5)
    shape = float(torch.norm(dev, p="fro"))
    return {"total": float((scale**2 + shape**2) ** 0.5), "scale": scale, "shape": shape, "k": k}


def riemannian_affine_invariant_distance(
    A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12, A_inv_sqrt: torch.Tensor | None = None
) -> float:
    """d_R(A, B) = ||log(A^{-1/2} B A^{-1/2})||_F. Both must be SPD.

    Pass `A_inv_sqrt` when A is a fixed reference compared against many B."""
    Ai = matrix_inv_sqrt(A, eps) if A_inv_sqrt is None else A_inv_sqrt
    return float(torch.norm(matrix_log(Ai @ B @ Ai, eps), p="fro"))


def log_euclidean_distance(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-12) -> float:
    """d_LE(A, B) = ||log A - log B||_F. Cheaper than AIRM, agrees with it when
    A and B nearly commute -- so d_LE == d_R to 4 decimals is a WARNING that both
    are dominated by a shared regulariser, not a confirmation."""
    return float(torch.norm(matrix_log(A, eps) - matrix_log(B, eps), p="fro"))
