r"""GEE with a working correlation: drift detection on autocorrelated agent trajectories.

Theoretical reference
---------------------
- Regressions in Covariances, Dependencies and Graphs (Pourahmadi & Arabpour), Ch.6 §6.3.
- Liang & Zeger (1986), Biometrika: "Longitudinal data analysis using generalized
  linear models".
- Huber (1967) / White (1980): the sandwich (robust) variance estimator.

THE PROBLEM
-----------
Per-turn quality signals inside one agent conversation are not independent draws. A
model that is mid-way through a reasoning chain produces correlated turns by
construction. A drift detector that assumes independence uses an effective sample
size it does not have, so its variance estimate is too small and its p-values are too
optimistic: it fires on a normal thinking chain. The failure is not subtle -- with
AR(1) errors at rho = 0.6, a nominal 5% test can reject far more often than 5% of the
time under no drift at all, and every one of those rejections is a false drift alarm
that would swap an expert mid-conversation.

THE ESTIMATOR
-------------
GEE models only the marginal mean mu_it = E[Y_it] and plugs a *working* correlation
R(alpha) into the estimating equation:

    sum_i D_i' V_i^{-1} (Y_i - mu_i) = 0,     V_i = phi A_i^{1/2} R_i(alpha) A_i^{1/2}

Two properties are the reason to use it here:

  1. The working correlation only has to be a guess. Consistency of beta_hat does not
     depend on R being right.
  2. The SANDWICH variance is consistent even when R is misspecified. This is what the
     benchmark measures: not whether AR(1) is the true structure of an agent
     trajectory (it is not), but whether the resulting test holds its nominal size
     when the truth is something else.

WHAT IT DOES NOT FIX
--------------------
Small cluster counts. The sandwich is a large-K estimator; with a handful of
conversations it is biased downward and the test over-rejects again -- for a different
reason. `small_sample_correction="df"` applies the standard K/(K-1) inflation, and the
benchmark reports type-I error as a function of K so the boundary is visible rather
than assumed away.

CPU-only: numpy + scipy.stats for the normal tail. No torch, no device allocation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import norm

__all__ = [
    "GEEFit",
    "fit_gee",
    "working_correlation_inverse",
    "DriftMonitor",
]

_EPS = 1e-10


@dataclass
class GEEFit:
    """Coefficients plus BOTH variance estimates -- the comparison is the whole point."""

    beta: np.ndarray
    se_model: np.ndarray       # naive / model-based: assumes R(alpha) is correct
    se_sandwich: np.ndarray    # Huber-White: consistent under misspecified R
    alpha: float
    phi: float
    n_clusters: int
    n_obs: int
    iterations: int
    converged: bool
    corr_structure: str
    family: str
    fit_seconds: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def wald(self, index: int, robust: bool = True) -> dict[str, float]:
        """Two-sided Wald test on one coefficient. `robust=False` is the naive arm."""
        se = (self.se_sandwich if robust else self.se_model)[index]
        z = float(self.beta[index] / se) if se > _EPS else 0.0
        return {"coefficient": float(self.beta[index]), "se": float(se), "z": z,
                "p_value": float(2.0 * norm.sf(abs(z)))}

    def confidence_interval(self, index: int, level: float = 0.95, robust: bool = True) -> tuple[float, float]:
        se = (self.se_sandwich if robust else self.se_model)[index]
        z_crit = float(norm.ppf(0.5 + level / 2.0))
        return (float(self.beta[index] - z_crit * se), float(self.beta[index] + z_crit * se))


def working_correlation_inverse(structure: str, alpha: float, T: int) -> np.ndarray:
    """R(alpha)^{-1} for the supported structures.

    AR(1) uses the closed-form tridiagonal inverse (O(T) to build, exact) rather than
    inverting a dense T x T matrix -- the same reason the Vecchia factor is banded:
    a Markov correlation has a sparse inverse, and forming the dense one is wasted work.
    """
    if structure == "independence" or T == 1:
        return np.eye(T)

    if structure == "ar1":
        a = float(np.clip(alpha, -0.95, 0.95))
        denom = 1.0 - a * a
        Rinv = np.zeros((T, T))
        for t in range(T):
            Rinv[t, t] = (1.0 + a * a) / denom
        Rinv[0, 0] = 1.0 / denom
        Rinv[T - 1, T - 1] = 1.0 / denom
        for t in range(T - 1):
            Rinv[t, t + 1] = Rinv[t + 1, t] = -a / denom
        return Rinv

    if structure == "exchangeable":
        a = float(np.clip(alpha, -1.0 / (T - 1) + 1e-6, 0.95))
        R = np.full((T, T), a)
        np.fill_diagonal(R, 1.0)
        return np.linalg.inv(R)

    raise ValueError(f"unknown working correlation {structure!r}")


def _mean_and_derivative(X: np.ndarray, beta: np.ndarray, family: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mu, dmu/deta, variance function A)."""
    eta = X @ beta
    if family == "gaussian":
        mu = eta
        return mu, np.ones_like(mu), np.ones_like(mu)
    if family == "binomial":
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        v = np.clip(mu * (1.0 - mu), _EPS, None)
        return mu, v, v
    raise ValueError(f"unknown family {family!r}")


def fit_gee(
    y_list: list[np.ndarray],
    X_list: list[np.ndarray],
    corr: str = "ar1",
    family: str = "gaussian",
    max_iter: int = 50,
    tol: float = 1e-7,
    small_sample_correction: str | None = None,
) -> GEEFit:
    """Fit marginal coefficients by Fisher scoring, with moment updates for phi/alpha.

    y_list / X_list are per-cluster (per-conversation) arrays: y_i is [T_i], X_i is
    [T_i, p]. Clusters may have different lengths; a real conversation set does.
    """
    t0 = time.perf_counter()
    if len(y_list) != len(X_list):
        raise ValueError("y_list and X_list must have the same number of clusters")
    K = len(y_list)
    p = X_list[0].shape[1]
    N = int(sum(len(y) for y in y_list))

    beta = np.zeros(p)
    # Least-squares start: Fisher scoring from zero diverges on a logit fit.
    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)
    if family == "gaussian":
        beta = np.linalg.lstsq(X_all, y_all, rcond=None)[0]

    alpha = 0.0
    phi = 1.0
    converged = False
    iterations = 0

    for _ in range(max_iter):
        iterations += 1

        # --- moment estimates of phi and alpha from current Pearson residuals -----
        resid = []
        for y_i, X_i in zip(y_list, X_list, strict=True):
            mu_i, _, A_i = _mean_and_derivative(X_i, beta, family)
            resid.append((y_i - mu_i) / np.sqrt(A_i))
        phi = float(sum(float(r @ r) for r in resid) / max(N - p, 1))
        phi = max(phi, _EPS)

        if corr == "independence":
            alpha = 0.0
        elif corr == "ar1":
            num = sum(float(r[:-1] @ r[1:]) for r in resid if len(r) > 1)
            den = sum(len(r) - 1 for r in resid if len(r) > 1)
            alpha = num / (max(den - p, 1) * phi) if den > 0 else 0.0
            alpha = float(np.clip(alpha, -0.95, 0.95))
        elif corr == "exchangeable":
            num = 0.0
            den = 0
            for r in resid:
                T_i = len(r)
                if T_i > 1:
                    num += float((r.sum() ** 2 - r @ r) / 2.0)
                    den += T_i * (T_i - 1) // 2
            alpha = num / (max(den - p, 1) * phi) if den > 0 else 0.0
            alpha = float(np.clip(alpha, -0.45, 0.95))
        else:
            raise ValueError(f"unknown working correlation {corr!r}")

        # --- Fisher scoring step on beta ----------------------------------------
        B = np.zeros((p, p))
        score = np.zeros(p)
        rinv_cache: dict[int, np.ndarray] = {}
        for y_i, X_i in zip(y_list, X_list, strict=True):
            T_i = len(y_i)
            mu_i, dmu_i, A_i = _mean_and_derivative(X_i, beta, family)
            if T_i not in rinv_cache:
                rinv_cache[T_i] = working_correlation_inverse(corr, alpha, T_i)
            Rinv = rinv_cache[T_i]
            sa = np.sqrt(A_i)
            Vinv = (Rinv / np.outer(sa, sa)) / phi
            D_i = X_i * dmu_i[:, None]
            DtV = D_i.T @ Vinv
            B += DtV @ D_i
            score += DtV @ (y_i - mu_i)

        step = np.linalg.solve(B + _EPS * np.eye(p), score)
        beta = beta + step
        if float(np.max(np.abs(step))) < tol:
            converged = True
            break

    # --- final variance estimates ------------------------------------------------
    B = np.zeros((p, p))
    M = np.zeros((p, p))
    rinv_cache = {}
    for y_i, X_i in zip(y_list, X_list, strict=True):
        T_i = len(y_i)
        mu_i, dmu_i, A_i = _mean_and_derivative(X_i, beta, family)
        if T_i not in rinv_cache:
            rinv_cache[T_i] = working_correlation_inverse(corr, alpha, T_i)
        Rinv = rinv_cache[T_i]
        sa = np.sqrt(A_i)
        Vinv = (Rinv / np.outer(sa, sa)) / phi
        D_i = X_i * dmu_i[:, None]
        DtV = D_i.T @ Vinv
        e_i = y_i - mu_i
        B += DtV @ D_i
        u_i = DtV @ e_i
        M += np.outer(u_i, u_i)

    Binv = np.linalg.inv(B + _EPS * np.eye(p))
    cov_model = Binv
    cov_sandwich = Binv @ M @ Binv

    correction = 1.0
    if small_sample_correction == "df" and K > 1:
        correction = K / (K - 1.0)
    elif small_sample_correction == "df_p" and p < K:
        correction = K / (K - float(p))
    cov_sandwich = cov_sandwich * correction

    return GEEFit(
        beta=beta,
        se_model=np.sqrt(np.maximum(np.diag(cov_model), 0.0)),
        se_sandwich=np.sqrt(np.maximum(np.diag(cov_sandwich), 0.0)),
        alpha=float(alpha),
        phi=float(phi),
        n_clusters=K,
        n_obs=N,
        iterations=iterations,
        converged=converged,
        corr_structure=corr,
        family=family,
        fit_seconds=time.perf_counter() - t0,
        meta={"small_sample_correction": small_sample_correction, "correction_factor": correction},
    )


class DriftMonitor:
    """Runtime-side drift detector: one GEE refit per turn over the live conversations.

    Feature design is deliberately minimal -- intercept plus centred turn index -- so
    the tested coefficient IS the trajectory slope, and a rejection means "this
    conversation's quality signal is trending", not "some covariate is significant".
    """

    def __init__(
        self,
        corr: str = "ar1",
        family: str = "gaussian",
        alpha_level: float = 0.05,
        min_turns: int = 6,
        small_sample_correction: str | None = "df",
    ):
        self.corr = corr
        self.family = family
        self.alpha_level = alpha_level
        self.min_turns = min_turns
        self.small_sample_correction = small_sample_correction
        self._series: dict[str, list[float]] = {}

    def observe(self, conversation_id: str, value: float) -> None:
        self._series.setdefault(conversation_id, []).append(float(value))

    def reset(self, conversation_id: str | None = None) -> None:
        if conversation_id is None:
            self._series.clear()
        else:
            self._series.pop(conversation_id, None)

    def _design(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        y_list, X_list = [], []
        for values in self._series.values():
            if len(values) < self.min_turns:
                continue
            t = np.arange(len(values), dtype=np.float64)
            t = t - t.mean()
            y_list.append(np.asarray(values, dtype=np.float64))
            X_list.append(np.column_stack([np.ones_like(t), t]))
        return y_list, X_list

    def assess(self) -> dict[str, Any]:
        """Wald test on the slope. Returns a verdict, not a decision -- the caller acts."""
        y_list, X_list = self._design()
        if not y_list:
            return {"status": "insufficient_data", "n_clusters": 0, "drifting": False}

        fit = fit_gee(
            y_list, X_list,
            corr=self.corr, family=self.family,
            small_sample_correction=self.small_sample_correction,
        )
        robust = fit.wald(1, robust=True)
        naive = fit.wald(1, robust=False)
        return {
            "status": "ok",
            "drifting": bool(robust["p_value"] < self.alpha_level),
            "slope": robust["coefficient"],
            "z_robust": robust["z"],
            "p_robust": robust["p_value"],
            "z_naive": naive["z"],
            "p_naive": naive["p_value"],
            "alpha_hat": fit.alpha,
            "n_clusters": fit.n_clusters,
            "n_obs": fit.n_obs,
            "fit_ms": fit.fit_seconds * 1000.0,
        }
