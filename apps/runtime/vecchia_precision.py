r"""Vecchia-approximated cross-layer precision: a banded GMRF over the residual stream.

Theoretical reference
---------------------
- Regressions in Covariances, Dependencies and Graphs (Pourahmadi & Arabpour), Ch.12 §12.2.1
  (Vecchia's approximation) and Ch.10 (modified Cholesky decomposition of a covariance).
- Vecchia (1988), JRSS-B: approximate likelihoods for large Gaussian fields.
- Katzfuss & Guinness (2021): a general framework for Vecchia approximations.

THE IDEA, IN ONE PARAGRAPH
--------------------------
Tracking how layer activations co-vary across an L-layer stack means estimating an
L x L covariance and inverting it: O(L^3) per update, on L = 32..80. Vecchia factorises
the joint density into ordered conditionals with a truncated conditioning set,

    p(x_1..x_L) ~= prod_l p(x_l | x_{c(l)}),   c(l) = {l-m .. l-1},  |c(l)| <= m

which is exactly a modified Cholesky decomposition with a BANDED triangular factor:

    T Sigma T' = D,    T unit lower triangular, nonzero only on the m sub-diagonals
    Theta = Sigma^{-1} = T' D^{-1} T                       (banded GMRF, bandwidth m)

Fitting is L independent ridge regressions of layer l on its m predecessors -- O(L m^2 n)
-- and every downstream solve is a banded triangular pass, O(L m). No L x L inverse ever
forms.

WHAT THIS BUYS AND WHAT IT ASSUMES
----------------------------------
The speedup is arithmetic and not in question. The ASSUMPTION is that conditional
dependence between layer l and layers further back than m is negligible once the m
intermediate layers are conditioned on. That is an empirical claim about this model's
residual stream, NOT something the method provides, and `conditional_decay_profile`
plus the held-out NLL sweep in
benchmarks/runtime/statistical/vecchia_layer_horizon/ exist to test it rather than
assume it. If held-out NLL keeps improving past m = 2, then m = 2 is wrong for this
stack and the honest answer is the larger m -- which is still O(L m^2).

ORDERING
--------
Vecchia's accuracy depends on the ordering. Layer index is the natural (and causal)
order here: layer l is computed from layer l-1, so conditioning on predecessors matches
the generative direction of the forward pass. Any other ordering would need justifying.

CPU-only: pure numpy, no torch, no device allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "VecchiaFactor",
    "fit_vecchia",
    "fit_vecchia_batched",
    "vecchia_nll",
    "vecchia_quadratic",
    "to_banded_storage",
    "banded_solve_lapack",
    "to_dense_precision",
    "banded_precision_solve",
    "dense_precision",
    "dense_nll",
    "band_mass_fraction",
    "conditional_decay_profile",
    "gaussian_kl",
    "select_horizon",
]

_VAR_FLOOR = 1e-10


@dataclass(frozen=True)
class VecchiaFactor:
    """Banded modified-Cholesky factor: Theta = T' D^{-1} T with bandwidth m.

    coeffs[l, k] is the regression weight of layer (l-1-k) in the conditional mean of
    layer l, k = 0..m-1 (k = 0 is the immediately preceding layer). Entries whose
    predictor falls before layer 0 are exactly zero.
    """

    m: int
    mean: np.ndarray  # [L]
    coeffs: np.ndarray  # [L, m]
    resid_var: np.ndarray  # [L]
    fit_seconds: float = 0.0

    @property
    def n_layers(self) -> int:
        return int(self.mean.shape[0])

    @property
    def logdet_precision(self) -> float:
        """log det Theta = -sum log d_l, since det T = 1."""
        return float(-np.sum(np.log(self.resid_var)))

    def nonzeros(self) -> int:
        """Stored parameters: the band plus the diagonal."""
        L = self.n_layers
        return int(sum(min(self.m, ell) for ell in range(L)) + L)


def _conditioning_slice(layer: int, m: int) -> tuple[int, int]:
    start = max(0, layer - m)
    return start, layer


def fit_vecchia(X: np.ndarray, m: int, ridge: float = 1e-6) -> VecchiaFactor:
    """Fit the banded factor by L ridge regressions on the m preceding columns.

    X is [n_samples, L], columns in layer order. `ridge` is relative to the trace of
    each local Gram matrix, so it is scale-free: it stabilises the m x m solve without
    choosing a shrinkage level for the user.
    """
    import time

    if X.ndim != 2:
        raise ValueError(f"X must be [n_samples, n_layers], got shape {X.shape}")
    if m < 0:
        raise ValueError(f"conditioning-set size m must be >= 0, got {m}")

    t0 = time.perf_counter()
    n, L = X.shape
    mean = X.mean(axis=0)
    Xc = X - mean

    coeffs = np.zeros((L, max(m, 1)), dtype=np.float64)
    resid_var = np.empty(L, dtype=np.float64)

    for ell in range(L):
        start, stop = _conditioning_slice(ell, m)
        y = Xc[:, ell]
        k = stop - start
        if k == 0:
            resid_var[ell] = max(float(y @ y) / n, _VAR_FLOOR)
            continue
        Z = Xc[:, start:stop]  # [n, k], oldest .. newest
        G = (Z.T @ Z) / n
        g = (Z.T @ y) / n
        lam = ridge * max(float(np.trace(G)) / k, _VAR_FLOOR)
        b = np.linalg.solve(G + lam * np.eye(k), g)
        r = y - Z @ b
        resid_var[ell] = max(float(r @ r) / n, _VAR_FLOOR)
        # store newest-first so coeffs[ell, k] always multiplies layer ell-1-k
        coeffs[ell, :k] = b[::-1]

    return VecchiaFactor(
        m=m,
        mean=mean,
        coeffs=coeffs,
        resid_var=resid_var,
        fit_seconds=time.perf_counter() - t0,
    )


def fit_vecchia_batched(X: np.ndarray, m: int, ridge: float = 1e-6) -> VecchiaFactor:
    """Same estimator as `fit_vecchia`, without the per-layer Python loop.

    WHY THIS EXISTS
    ---------------
    The first version of this module fit the band with an explicit `for ell in range(L)`
    and the benchmark duly reported that Vecchia was 4-6x SLOWER than a dense inverse at
    L = 32..80. That measured the interpreter, not the algorithm: L small numpy calls at
    ~10 us of dispatch each will lose to one LAPACK call every time.

    The batched form computes only the BANDED entries of the covariance -- lag k
    cross-products for k = 0..m, which is O(n L m) rather than the O(n L^2) full Gram --
    and then solves all L systems at once with a single stacked `np.linalg.solve` over an
    [L, m, m] array. The first m layers have short conditioning sets and are finished in a
    loop of length m, not L.

    Numerically identical to `fit_vecchia` up to the solve; the test suite asserts it.
    """
    import time

    if X.ndim != 2:
        raise ValueError(f"X must be [n_samples, n_layers], got shape {X.shape}")
    if m < 0:
        raise ValueError(f"conditioning-set size m must be >= 0, got {m}")

    t0 = time.perf_counter()
    n, L = X.shape
    mean = X.mean(axis=0)
    Xc = X - mean

    coeffs = np.zeros((L, max(m, 1)), dtype=np.float64)
    resid_var = np.empty(L, dtype=np.float64)

    # banded covariance: cov_lag[k][ell] = Cov(x_ell, x_{ell-k}),  k = 0..m
    cov_lag = [np.einsum("ij,ij->j", Xc, Xc) / n]
    for k in range(1, m + 1):
        c = np.zeros(L)
        if k < L:
            c[k:] = np.einsum("ij,ij->j", Xc[:, k:], Xc[:, : L - k]) / n
        cov_lag.append(c)

    if m == 0 or L == 1:
        resid_var[:] = np.maximum(cov_lag[0], _VAR_FLOOR)
        return VecchiaFactor(m=m, mean=mean, coeffs=coeffs, resid_var=resid_var, fit_seconds=time.perf_counter() - t0)

    # short conditioning sets: layers 0..m-1 (a loop of length m, independent of L)
    for ell in range(min(m, L)):
        if ell == 0:
            resid_var[0] = max(cov_lag[0][0], _VAR_FLOOR)
            continue
        Z = Xc[:, :ell]
        G = (Z.T @ Z) / n
        g = (Z.T @ Xc[:, ell]) / n
        lam = ridge * max(float(np.trace(G)) / ell, _VAR_FLOOR)
        b = np.linalg.solve(G + lam * np.eye(ell), g)
        r = Xc[:, ell] - Z @ b
        resid_var[ell] = max(float(r @ r) / n, _VAR_FLOOR)
        coeffs[ell, :ell] = b[::-1]

    if m >= L:
        return VecchiaFactor(m=m, mean=mean, coeffs=coeffs, resid_var=resid_var, fit_seconds=time.perf_counter() - t0)

    # full conditioning sets: layers m..L-1, solved as one batch
    rows = np.arange(m, L)
    a_idx = np.arange(m)
    # predictor a of layer ell is column (ell - 1 - a); G[a, b] = Cov of those two columns
    pred_cols = rows[:, None] - 1 - a_idx[None, :]  # [R, m]
    lag_ab = np.abs(a_idx[:, None] - a_idx[None, :])  # [m, m]
    newer = np.maximum(pred_cols[:, :, None], pred_cols[:, None, :])  # [R, m, m]
    cov_stack = np.stack(cov_lag, axis=0)  # [m+1, L]
    G_batch = cov_stack[lag_ab[None, :, :], newer]  # [R, m, m]
    g_batch = cov_stack[a_idx + 1, rows[:, None]]  # [R, m]

    trace = np.einsum("rii->r", G_batch) / m
    lam = ridge * np.maximum(trace, _VAR_FLOOR)
    G_batch = G_batch + lam[:, None, None] * np.eye(m)[None, :, :]
    b_batch = np.linalg.solve(G_batch, g_batch[..., None])[..., 0]  # [R, m]

    quad = np.einsum("ra,rab,rb->r", b_batch, G_batch, b_batch)
    resid = cov_lag[0][rows] - 2.0 * np.einsum("ra,ra->r", b_batch, g_batch) + quad
    coeffs[rows] = b_batch
    resid_var[rows] = np.maximum(resid, _VAR_FLOOR)

    return VecchiaFactor(m=m, mean=mean, coeffs=coeffs, resid_var=resid_var, fit_seconds=time.perf_counter() - t0)


def _innovations(factor: VecchiaFactor, X: np.ndarray) -> np.ndarray:
    """r = T (x - mu): the one-step conditional residuals. O(n L m)."""
    Xc = X - factor.mean
    R = Xc.copy()
    for k in range(factor.m):
        lag = k + 1
        if lag >= factor.n_layers:
            break
        w = factor.coeffs[lag:, k]  # weight of layer ell-lag in layer ell
        R[:, lag:] -= Xc[:, : factor.n_layers - lag] * w
    return R


def vecchia_nll(factor: VecchiaFactor, X: np.ndarray) -> float:
    """Average per-sample negative log-likelihood under the banded model. O(n L m).

    This is the metric that decides m. It is computed on HELD-OUT samples in the
    benchmark: in-sample NLL falls monotonically with m by construction and would
    "prove" that a wider band is always better.
    """
    R = _innovations(factor, X)
    quad = float(np.mean(np.sum(R * R / factor.resid_var, axis=1)))
    L = factor.n_layers
    return 0.5 * (L * np.log(2.0 * np.pi) + float(np.sum(np.log(factor.resid_var))) + quad)


def vecchia_quadratic(factor: VecchiaFactor, X: np.ndarray) -> np.ndarray:
    """Per-sample (x-mu)' Theta (x-mu) in O(L m), vectorised over samples.

    This is the operation a live covariance tracker actually performs per token --
    a Mahalanobis score against the layer-covariance model -- so it is the apply-side
    cost that belongs in a latency comparison, not a linear solve.
    """
    R = _innovations(factor, np.atleast_2d(X))
    return np.sum(R * R / factor.resid_var, axis=1)


def to_banded_storage(factor: VecchiaFactor) -> np.ndarray:
    """Theta in scipy's symmetric-banded upper form [m+1, L], built in O(L m^2).

    Lets `scipy.linalg.solveh_banded` solve Theta x = y in compiled O(L m^2) instead of
    a Python substitution loop -- the comparison a dense LAPACK inverse deserves.
    """
    L = factor.n_layers
    m = factor.m
    inv_d = 1.0 / factor.resid_var
    ab = np.zeros((m + 1, L))

    # Theta = T' D^{-1} T with T unit lower triangular, band m.
    # Theta[i, j] = sum_r T[r, i] T[r, j] / d_r over rows r that touch both i and j.
    def t_entry(row: int, col: int) -> float:
        if row == col:
            return 1.0
        k = row - 1 - col
        if 0 <= k < m:
            return -factor.coeffs[row, k]
        return 0.0

    for i in range(L):
        for j in range(i, min(i + m + 1, L)):
            acc = 0.0
            for r in range(j, min(j + m + 1, L)):
                acc += t_entry(r, i) * t_entry(r, j) * inv_d[r]
            ab[m - (j - i), j] = acc
    return ab


def banded_solve_lapack(ab: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve Theta x = y from banded storage via LAPACK (compiled O(L m^2))."""
    from scipy.linalg import solveh_banded

    return solveh_banded(ab, y, lower=False)


def to_dense_precision(factor: VecchiaFactor) -> np.ndarray:
    """Materialise Theta = T' D^{-1} T. Diagnostics only -- the runtime never needs it."""
    L = factor.n_layers
    T = np.eye(L, dtype=np.float64)
    for ell in range(L):
        for k in range(min(factor.m, ell)):
            T[ell, ell - 1 - k] = -factor.coeffs[ell, k]
    return T.T @ (T / factor.resid_var[:, None])


def banded_precision_solve(factor: VecchiaFactor, y: np.ndarray) -> np.ndarray:
    """Solve Theta x = y in O(L m) via the banded triangular factors.

    Theta = T' D^{-1} T, so x = T^{-1} D T^{-T} y: one back-substitution on T', one
    diagonal scale, one forward-substitution on T. This is the operation a runtime
    covariance tracker actually performs (whitening a layer state), and the reason
    the banded form is worth having at all.
    """
    L = factor.n_layers
    z = np.array(y, dtype=np.float64).copy()

    # T' z = y  (T' is upper triangular, unit diagonal): descending order
    for ell in range(L - 1, -1, -1):
        for k in range(min(factor.m, ell)):
            z[ell - 1 - k] -= -factor.coeffs[ell, k] * z[ell]
    z *= factor.resid_var  # D
    # T x = z  (unit lower triangular): ascending order
    for ell in range(L):
        acc = 0.0
        for k in range(min(factor.m, ell)):
            acc += -factor.coeffs[ell, k] * z[ell - 1 - k]
        z[ell] -= acc
    return z


def dense_precision(X: np.ndarray, ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Baseline: full sample covariance and its explicit inverse. O(L^3).

    Returns (Theta, mean). The ridge is relative to tr(S)/L -- with n < L the sample
    covariance is singular and an unregularised inverse is not merely inaccurate, it
    does not exist.
    """
    n, L = X.shape
    mean = X.mean(axis=0)
    Xc = X - mean
    S = (Xc.T @ Xc) / n
    lam = ridge * max(float(np.trace(S)) / L, _VAR_FLOOR)
    return np.linalg.inv(S + lam * np.eye(L)), mean


def dense_nll(Theta: np.ndarray, mean: np.ndarray, X: np.ndarray) -> float:
    """Average per-sample NLL of a dense Gaussian, for like-for-like comparison."""
    sign, logdet = np.linalg.slogdet(Theta)
    if sign <= 0:
        return float("inf")
    Xc = X - mean
    quad = float(np.mean(np.einsum("ij,jk,ik->i", Xc, Theta, Xc)))
    L = Theta.shape[0]
    return 0.5 * (L * np.log(2.0 * np.pi) - float(logdet) + quad)


def band_mass_fraction(Theta: np.ndarray, m: int) -> float:
    """Fraction of off-diagonal |Theta| mass lying within m diagonals of the main one.

    This is the direct test of the banding premise: at m = 2, a value near 1.0 means
    the residual stream really is locally Markov and Vecchia loses nothing; a value
    near 0.3 means most conditional dependence is long-range and the band is throwing
    signal away.
    """
    L = Theta.shape[0]
    idx = np.arange(L)
    lag = np.abs(idx[:, None] - idx[None, :])
    off = lag > 0
    total = float(np.abs(Theta[off]).sum())
    if total <= 0:
        return 0.0
    within = float(np.abs(Theta[off & (lag <= m)]).sum())
    return within / total


def conditional_decay_profile(Theta: np.ndarray, max_lag: int | None = None) -> dict[int, float]:
    """Mean |partial correlation| by layer separation: -Theta_ij / sqrt(Theta_ii Theta_jj).

    Partial correlation, not correlation. Marginal correlation between layers is high
    everywhere simply because the residual stream carries the same signal forward; the
    conditional quantity is the one that decides whether a band is defensible.
    """
    L = Theta.shape[0]
    d = np.sqrt(np.maximum(np.diag(Theta), _VAR_FLOOR))
    P = -Theta / np.outer(d, d)
    top = max_lag if max_lag is not None else L - 1
    profile: dict[int, float] = {}
    for lag in range(1, top + 1):
        vals = np.abs(np.diagonal(P, offset=lag))
        if vals.size:
            profile[lag] = float(vals.mean())
    return profile


def gaussian_kl(Sigma_ref: np.ndarray, Theta_approx: np.ndarray) -> float:
    """KL( N(0, Sigma_ref) || N(0, Theta_approx^{-1}) ), in nats.

    KL = 0.5 [ tr(Theta_approx Sigma_ref) - L - logdet(Theta_approx Sigma_ref) ].
    """
    L = Sigma_ref.shape[0]
    M = Theta_approx @ Sigma_ref
    sign, logdet = np.linalg.slogdet(M)
    if sign <= 0:
        return float("inf")
    return 0.5 * (float(np.trace(M)) - L - float(logdet))


def select_horizon(
    X_train: np.ndarray,
    X_val: np.ndarray,
    m_grid: tuple[int, ...] = (0, 1, 2, 3, 4, 6, 8),
    tolerance_nats: float = 0.01,
    ridge: float = 1e-6,
) -> dict[str, Any]:
    """Smallest m whose held-out NLL is within `tolerance_nats` of the best on the grid.

    Deliberately not "the m with the lowest NLL": past the point where the curve is
    flat, extra bandwidth costs runtime and buys noise.
    """
    curve: list[dict[str, float]] = []
    for m in m_grid:
        f = fit_vecchia(X_train, m=m, ridge=ridge)
        curve.append({"m": m, "val_nll": vecchia_nll(f, X_val), "fit_seconds": f.fit_seconds})
    best = min(c["val_nll"] for c in curve)
    chosen = next(c["m"] for c in curve if c["val_nll"] <= best + tolerance_nats)
    return {
        "selected_m": int(chosen),
        "best_val_nll": float(best),
        "tolerance_nats": tolerance_nats,
        "curve": curve,
    }
