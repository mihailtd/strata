r"""CLIME: constrained l1-minimisation for sparse precision estimation when n << d.

╔══════════════════════════════════════════════════════════════════════════════╗
║ STATUS: the CLIME path is RETIRED. `graphical_lasso_admm` below is the live   ║
║ estimator -- it WON the comparison that retired CLIME.                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Measured on ground truth (benchmarks/superseded/clime_head_crosstalk/):        ║
║   * glasso beats CLIME at n/d <= 1 -- the B=1 decode regime CLIME was          ║
║     proposed for -- while being ~35x faster (1.3 ms vs 48 ms at d=32).         ║
║   * The real head cross-talk graph is not reproducible at decode window        ║
║     sizes at all: two disjoint 32-token windows share ZERO edges. That is a    ║
║     property of the activations, so no estimator rescues the application.      ║
║                                                                               ║
║ This file stays live because `graphical_lasso_admm`, `support_metrics`,        ║
║ `max_constraint_violation` and `naive_inverse_precision` are the reusable      ║
║ pieces. `clime_column` / `clime_precision` remain as the documented losing     ║
║ arm -- do not reach for them for new work without new evidence.                ║
║ See docs/DECISIONS.md §66.                                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Theoretical reference
---------------------
- Regressions in Covariances, Dependencies and Graphs (Pourahmadi & Arabpour), Ch.4 §4.4.3.
- Cai, Liu & Luo (2011), JASA: "A Constrained l1 Minimization Approach to Sparse
  Precision Matrix Estimation".

THE PROBLEM
-----------
At B = 1 decode (or speculative M = 2..4) the number of activation samples available
inside one step is orders of magnitude smaller than the feature dimension. The sample
covariance Sigma_hat is then singular by construction -- rank <= n - 1 < d -- and
inverting it does not estimate anything: it amplifies the noise directions that happen
to have near-zero sample variance into huge spurious precision entries, i.e. dense
fake cross-talk between features that are actually unrelated.

THE ESTIMATOR
-------------
CLIME solves d independent linear programs, one per column of Theta:

    min ||theta_j||_1   s.t.  || Sigma_hat theta_j - e_j ||_inf <= lambda

then symmetrises by taking, for each off-diagonal pair, whichever of the two solved
entries has the smaller magnitude. Two properties matter here:

  1. COLUMN DECOUPLING. The columns never talk to each other, so the estimator is
     embarrassingly parallel -- unlike graphical lasso, whose coordinate descent
     sweeps the whole matrix and must be run to convergence as one object. This is
     the property that makes CLIME a candidate for a background compute stream.
  2. THE CONSTRAINT IS THE GUARANTEE. Feasibility gives the entrywise bound
     ||Sigma_hat Theta_hat - I||_inf <= lambda directly, with no assumption on the
     spectrum. `max_constraint_violation` checks it on the solved output rather than
     trusting it, because a solver that returns an infeasible or truncated solution
     would otherwise pass silently.

WHAT LAMBDA COSTS YOU
---------------------
lambda is not a tuning knob you can push to zero: below some value the LP is
infeasible (no theta reproduces e_j that closely under a rank-deficient Sigma_hat),
and near that value the solution is dense. The theory-driven scale is
lambda ~ C sqrt(log d / n) -- `clime_lambda_theoretical` -- and the benchmark sweeps
around it rather than reporting one hand-picked point.

CPU-only: numpy + scipy HiGHS. No torch, no device allocation.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linprog

__all__ = [
    "ClimeResult",
    "clime_column",
    "clime_precision",
    "clime_lambda_theoretical",
    "naive_inverse_precision",
    "graphical_lasso_admm",
    "max_constraint_violation",
    "support_metrics",
    "symmetry_gap",
    "sparsity",
]


@dataclass
class ClimeResult:
    """Symmetrised precision estimate plus the evidence that it solved what it claims."""

    Theta: np.ndarray
    Theta_raw: np.ndarray
    lam: float
    column_seconds: np.ndarray
    feasible: np.ndarray
    total_seconds: float
    n_jobs: int = 1
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_infeasible(self) -> int:
        return int(np.sum(~self.feasible))

    @property
    def feasible_fraction(self) -> float:
        return float(np.mean(self.feasible)) if self.feasible.size else 0.0

    @property
    def critical_path_seconds(self) -> float:
        """Slowest single column: the wall time of a perfectly parallel CLIME."""
        return float(self.column_seconds.max()) if self.column_seconds.size else 0.0

    @property
    def ideal_parallel_speedup(self) -> float:
        total = float(self.column_seconds.sum())
        return total / self.critical_path_seconds if self.critical_path_seconds > 0 else 1.0


def clime_lambda_theoretical(n: int, d: int, c: float = 2.0) -> float:
    """The rate-optimal scale lambda = C sqrt(log d / n) (Cai, Liu & Luo 2011, Thm 1)."""
    return float(c * np.sqrt(np.log(max(d, 2)) / max(n, 1)))


def clime_column(S: np.ndarray, j: int, lam: float) -> tuple[np.ndarray, bool]:
    """One CLIME column as a linear program. Returns (theta_j, feasible).

    Split theta = u - v with u, v >= 0 so the l1 objective is linear:

        min 1'(u + v)   s.t.   [ S  -S] [u;v] <=  lam*1 + e_j
                               [-S   S] [u;v] <=  lam*1 - e_j
    """
    d = S.shape[0]
    e_j = np.zeros(d)
    e_j[j] = 1.0

    A_ub = np.block([[S, -S], [-S, S]])
    b_ub = np.concatenate([lam + e_j, lam - e_j])
    c_obj = np.ones(2 * d)

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method="highs")
    if not res.success:
        return np.zeros(d), False
    return res.x[:d] - res.x[d:], True


def clime_precision(S: np.ndarray, lam: float, n_jobs: int = 1) -> ClimeResult:
    """Solve every column, then symmetrise by the smaller-magnitude rule.

    The symmetrisation is part of the estimator, not cosmetics: the column LPs have
    no joint constraint, so Theta_raw is not symmetric, and `symmetry_gap` on the raw
    matrix is a useful diagnostic -- a large gap means lambda is too small for the
    sample size and the columns are disagreeing about the same edge.
    """
    d = S.shape[0]
    column_seconds = np.zeros(d)
    feasible = np.zeros(d, dtype=bool)
    Theta_raw = np.zeros((d, d))

    def solve_one(j: int) -> tuple[int, np.ndarray, bool, float]:
        t0 = time.perf_counter()
        theta_j, ok = clime_column(S, j, lam)
        return j, theta_j, ok, time.perf_counter() - t0

    t_start = time.perf_counter()
    if n_jobs > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(solve_one, range(d)))
    else:
        results = [solve_one(j) for j in range(d)]
    total = time.perf_counter() - t_start

    for j, theta_j, ok, secs in results:
        Theta_raw[:, j] = theta_j
        column_seconds[j] = secs
        feasible[j] = ok

    keep_left = np.abs(Theta_raw) <= np.abs(Theta_raw.T)
    Theta = np.where(keep_left, Theta_raw, Theta_raw.T)

    return ClimeResult(
        Theta=Theta,
        Theta_raw=Theta_raw,
        lam=lam,
        column_seconds=column_seconds,
        feasible=feasible,
        total_seconds=total,
        n_jobs=n_jobs,
    )


def naive_inverse_precision(S: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """The thing CLIME replaces: a direct (optionally ridged) inverse of Sigma_hat.

    With ridge = 0 and n < d this is a pseudo-inverse of a singular matrix -- kept as
    an arm precisely so the failure is measured rather than asserted.
    """
    d = S.shape[0]
    if ridge > 0:
        return np.linalg.inv(S + ridge * np.eye(d))
    try:
        return np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(S)


def graphical_lasso_admm(
    S: np.ndarray,
    lam: float,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """l1-penalised Gaussian MLE by ADMM -- the estimator CLIME is usually compared to.

    min -logdet(Theta) + tr(S Theta) + lam ||Theta||_{1,off}

    Included as the honest baseline: it optimises a likelihood (CLIME does not), but
    every sweep couples all d^2 entries, so it cannot be split across independent
    workers the way the CLIME columns can.
    """
    d = S.shape[0]
    Theta = np.eye(d)
    Z = np.eye(d)
    U = np.zeros((d, d))
    history: list[float] = []
    converged = False

    iterations = 0
    for _ in range(max_iter):
        iterations += 1
        # Theta-update: eigen-decompose rho(Z - U) - S
        M = rho * (Z - U) - S
        M = 0.5 * (M + M.T)
        w, V = np.linalg.eigh(M)
        d_eig = (w + np.sqrt(w**2 + 4.0 * rho)) / (2.0 * rho)
        Theta = V @ np.diag(d_eig) @ V.T

        # Z-update: soft-threshold off-diagonal only
        Z_old = Z
        A = Theta + U
        Z = np.sign(A) * np.maximum(np.abs(A) - lam / rho, 0.0)
        np.fill_diagonal(Z, np.diag(A))

        U = U + Theta - Z

        r_norm = float(np.linalg.norm(Theta - Z))
        s_norm = float(rho * np.linalg.norm(Z - Z_old))
        history.append(r_norm)
        if r_norm < tol * max(d, 1) and s_norm < tol * max(d, 1):
            converged = True
            break

    return Z, {"iterations": iterations, "converged": converged, "final_primal_residual": history[-1]}


def max_constraint_violation(S: np.ndarray, Theta: np.ndarray, columns: np.ndarray | None = None) -> float:
    """||S Theta - I||_inf, optionally restricted to a subset of columns.

    On the RAW (un-symmetrised) solution over FEASIBLE columns this must not exceed
    lambda -- that is the CLIME guarantee, checked rather than assumed. Symmetrisation
    can push it above lambda, which is expected and is why both are reported.
    An infeasible column is returned as zeros, whose violation is exactly 1.0; scoring
    it alongside the solved columns would report a solver failure as an accuracy result.
    """
    d = S.shape[0]
    R = S @ Theta - np.eye(d)
    if columns is not None:
        R = R[:, columns]
    if R.size == 0:
        return float("nan")
    return float(np.max(np.abs(R)))


def support_metrics(Theta_hat: np.ndarray, Theta_true: np.ndarray, tol: float = 1e-4) -> dict[str, float]:
    """Off-diagonal edge recovery against a known precision matrix."""
    d = Theta_true.shape[0]
    off = ~np.eye(d, dtype=bool)
    pred = (np.abs(Theta_hat) > tol) & off
    true = (np.abs(Theta_true) > tol) & off

    tp = float(np.sum(pred & true))
    fp = float(np.sum(pred & ~true))
    fn = float(np.sum(~pred & true))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_edges": float(np.sum(true) / 2),
        "predicted_edges": float(np.sum(pred) / 2),
    }


def symmetry_gap(Theta_raw: np.ndarray) -> float:
    """max |Theta_ij - Theta_ji| on the un-symmetrised column solutions."""
    return float(np.max(np.abs(Theta_raw - Theta_raw.T)))


def sparsity(Theta: np.ndarray, tol: float = 1e-4) -> float:
    """Fraction of off-diagonal entries that are (numerically) zero."""
    d = Theta.shape[0]
    off = ~np.eye(d, dtype=bool)
    return float(np.mean(np.abs(Theta[off]) <= tol))
