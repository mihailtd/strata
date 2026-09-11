r"""Nonparametric copulas and upper tail dependence for multi-expert routing (Ch.3 §3.6).

Theoretical reference
---------------------
- Regressions in Covariances, Dependencies and Graphs (Pourahmadi & Arabpour), Ch.3 §3.6.
- Sklar (1959); Joe (1997); Nelsen (2006), An Introduction to Copulas.
- Schmidt & Stadtmuller (2006): nonparametric estimation of tail dependence.

THE PROBLEM
-----------
A router that scores experts by cosine or dot-product similarity is using a LINEAR
dependence measure. Linear correlation says nothing about whether two experts fire
together in the extreme -- and the case that matters for dual-expert folding is exactly
the extreme one: the rare prompt that needs the Postgres expert AND the systems expert
at once. Two variables can have correlation near zero and still co-occur in their upper
tails almost surely, and two variables can be strongly correlated with no tail
dependence at all. The Gaussian copula is the canonical example of the second: any
rho < 1 gives lambda_U = 0.

THE MEASURE
-----------
By Sklar's theorem the joint law splits into marginals and a copula, F(x) =
C(F_1(x_1), ..., F_d(x_d)). The upper tail dependence coefficient

    lambda_U = lim_{u->1-} P(U_1 > u | U_2 > u) = lim_{u->1-} (1 - 2u + C(u,u)) / (1 - u)

is a property of C alone, so it is invariant to any strictly increasing transform of the
marginals -- log magnitudes, raw norms, softmax scores, all give the same answer. That
invariance is the practical argument for it here: the Gaussian estimators in this family
need their inputs Gaussianised first, and this one does not.

WHAT IT COSTS
-------------
lambda_U is a LIMIT, and it is estimated from the k largest observations only. Small k
means low bias and high variance; large k means the opposite. There is no way around
this, so `tail_dependence_matrix` takes k explicitly and the benchmark sweeps it against
copulas whose true lambda_U is known in closed form. An estimator that cannot separate
lambda_U = 0 from lambda_U = 0.59 at the sample sizes a router actually has is not
usable for routing, whatever the theory says.

CPU-only: numpy + scipy.stats for the reference distributions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "pseudo_observations",
    "empirical_copula",
    "upper_tail_dependence",
    "upper_tail_dependence_log",
    "tail_dependence_matrix",
    "theoretical_lambda_u_t",
    "theoretical_lambda_u_gumbel",
    "sample_gaussian_copula",
    "sample_t_copula",
    "sample_gumbel_copula",
    "CopulaTailRouter",
    "RoutingDecision",
    "gaussian_null_cutoff",
]


def pseudo_observations(X: np.ndarray) -> np.ndarray:
    """Rank transform to (0,1): U_ij = rank(X_ij) / (n+1), column-wise.

    The n+1 denominator keeps the pseudo-observations strictly inside the unit cube,
    which matters because the log estimator takes log(u).
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n = X.shape[0]
    order = np.argsort(np.argsort(X, axis=0), axis=0) + 1
    return order / (n + 1.0)


def empirical_copula(U: np.ndarray, u: float, v: float, i: int = 0, j: int = 1) -> float:
    """C_n(u, v) = (1/n) #{ U_i <= u and U_j <= v }."""
    return float(np.mean((U[:, i] <= u) & (U[:, j] <= v)))


def upper_tail_dependence(U: np.ndarray, i: int, j: int, k: int) -> float:
    """Joint-exceedance estimator: lambda_U ~= #{both above the (1 - k/n) quantile} / k.

    The direct nonparametric estimator. Uses only the k most extreme observations, so
    its variance at small k is large -- which is the point of sweeping k rather than
    fixing one.
    """
    n = U.shape[0]
    if k <= 0 or k >= n:
        raise ValueError(f"k must satisfy 0 < k < n, got k={k}, n={n}")
    thresh = 1.0 - k / n
    return float(np.sum((U[:, i] > thresh) & (U[:, j] > thresh)) / k)


def upper_tail_dependence_log(U: np.ndarray, i: int, j: int, k: int) -> float:
    """Log estimator: lambda_U ~= 2 - log C_n(q, q) / log q at q = 1 - k/n.

    Lower variance than the joint-exceedance count because it uses every observation
    below the threshold as well, at the cost of an extra approximation.
    """
    n = U.shape[0]
    q = 1.0 - k / n
    if not 0.0 < q < 1.0:
        raise ValueError(f"k={k} gives an out-of-range threshold for n={n}")
    c = empirical_copula(U, q, q, i, j)
    if c <= 0.0:
        return float("nan")
    return float(2.0 - np.log(c) / np.log(q))


def tail_dependence_matrix(X: np.ndarray, k: int, estimator: str = "exceedance") -> np.ndarray:
    """Pairwise lambda_U over the columns of X. Diagonal set to 1 by definition."""
    U = pseudo_observations(X)
    d = U.shape[1]
    fn = upper_tail_dependence if estimator == "exceedance" else upper_tail_dependence_log
    M = np.eye(d)
    for a in range(d):
        for b in range(a + 1, d):
            M[a, b] = M[b, a] = fn(U, a, b, k)
    return M


# ---------------------------------------------------------------------------
# Reference copulas: known lambda_U, so the estimator can be scored rather than trusted
# ---------------------------------------------------------------------------
def theoretical_lambda_u_t(rho: float, nu: float) -> float:
    """Student-t copula: lambda_U = 2 * t_{nu+1}( -sqrt((nu+1)(1-rho)/(1+rho)) )."""
    from scipy.stats import t as student_t

    arg = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return float(2.0 * student_t.cdf(arg, df=nu + 1.0))


def theoretical_lambda_u_gumbel(theta: float) -> float:
    """Gumbel copula: lambda_U = 2 - 2^{1/theta}, for theta >= 1."""
    return float(2.0 - 2.0 ** (1.0 / theta))


def sample_gaussian_copula(n: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Tail-INDEPENDENT reference: lambda_U = 0 for every rho < 1."""
    from scipy.stats import norm

    cov = np.array([[1.0, rho], [rho, 1.0]])
    Z = rng.multivariate_normal(np.zeros(2), cov, size=n)
    return norm.cdf(Z)


def sample_t_copula(n: int, rho: float, nu: float, rng: np.random.Generator) -> np.ndarray:
    """Tail-DEPENDENT reference with symmetric tails."""
    from scipy.stats import t as student_t

    cov = np.array([[1.0, rho], [rho, 1.0]])
    Z = rng.multivariate_normal(np.zeros(2), cov, size=n)
    W = rng.chisquare(nu, size=n)
    T = Z / np.sqrt(W / nu)[:, None]
    return student_t.cdf(T, df=nu)


def sample_gumbel_copula(n: int, theta: float, rng: np.random.Generator) -> np.ndarray:
    """Upper-tail dependent, lower-tail independent -- the asymmetric reference.

    Marshall-Olkin: V ~ positive stable(1/theta) drawn by Chambers-Mallows-Stuck,
    E_i ~ Exp(1), U_i = exp(-(E_i / V)^{1/theta}).
    """
    if theta < 1.0:
        raise ValueError(f"Gumbel theta must be >= 1, got {theta}")
    alpha = 1.0 / theta
    th = rng.uniform(0.0, np.pi, size=n)
    w = rng.exponential(1.0, size=n)
    V = (np.sin(alpha * th) / np.sin(th) ** (1.0 / alpha)) * (np.sin((1.0 - alpha) * th) / w) ** (
        (1.0 - alpha) / alpha
    )
    E = rng.exponential(1.0, size=(n, 2))
    return np.exp(-((E / V[:, None]) ** alpha))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def gaussian_null_cutoff(
    n: int, k: int, rho: float, n_null: int, rng: np.random.Generator, quantile: float = 0.95
) -> float:
    """Upper `quantile` of lambda_U_hat under a Gaussian copula with this rho, at this (n, k).

    The Gaussian copula has lambda_U = 0 exactly for every rho < 1, so anything the
    estimator returns here is finite-sample bias -- and the bias grows with rho. This is
    the reference a lambda_U estimate has to beat before it means anything.
    """
    rho = float(np.clip(rho, -0.95, 0.95))
    vals = np.empty(n_null)
    for i in range(n_null):
        U = pseudo_observations(sample_gaussian_copula(n, rho, rng))
        vals[i] = upper_tail_dependence(U, 0, 1, k)
    return float(np.quantile(vals, quantile))


@dataclass
class RoutingDecision:
    primary: str
    co_activated: list[str]
    pseudo_observations: dict[str, float]
    in_upper_tail: bool
    decision_us: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def experts(self) -> list[str]:
        return [self.primary, *self.co_activated]


class CopulaTailRouter:
    """Route to one expert, and fold a second ONLY on a tail-dependent co-activation.

    Three ways to rank co-activation partners, differing in exactly one place so a
    benchmark that swaps the flag is measuring the dependence measure and nothing else:

      "pearson"         linear correlation on the history.
      "tail"            the raw lambda_U estimate.
      "tail_calibrated" lambda_U MINUS the 95th percentile of lambda_U under a Gaussian
                        copula fitted to that pair's own Pearson correlation, at the same
                        n and k. The raw estimator is badly biased upward on tail-
                        INDEPENDENT pairs, and the bias grows with correlation -- so a raw
                        cutoff systematically prefers the merely-correlated pair over the
                        genuinely tail-dependent one. Subtracting the matched null is what
                        makes the score comparable across pairs.

    A second expert is folded only when BOTH conditions hold: the pair has tail
    dependence above `pair_threshold`, and this token puts both experts above
    `tail_quantile`. Folding costs a real weight write, so a router that co-activates on
    ordinary traffic is not cheap-but-imprecise; it is a latency regression.
    """

    def __init__(
        self,
        pair_metric: str = "tail",
        pair_threshold: float = 0.15,
        tail_quantile: float = 0.95,
        k: int | None = None,
        n_null: int = 64,
        seed: int = 0,
    ):
        self.pair_metric = pair_metric
        self.pair_threshold = pair_threshold
        self.tail_quantile = tail_quantile
        self.k = k
        self.n_null = n_null
        self.seed = seed
        self.fit_seconds: float = 0.0
        self.names: list[str] = []
        self._sorted: np.ndarray | None = None
        self._pair: np.ndarray | None = None
        self._n: int = 0

    def fit(self, history: np.ndarray, names: list[str]) -> CopulaTailRouter:
        H = np.asarray(history, dtype=np.float64)
        n, d = H.shape
        if d != len(names):
            raise ValueError(f"history has {d} columns but {len(names)} names were given")
        self.names = list(names)
        self._n = n
        # Sorted columns: turns a per-token rank lookup into a binary search, O(d log n).
        self._sorted = np.sort(H, axis=0)
        k = self.k if self.k is not None else max(int(np.sqrt(n)), 2)
        self.k = k
        t0 = time.perf_counter()
        if self.pair_metric == "tail":
            self._pair = tail_dependence_matrix(H, k=k)
        elif self.pair_metric == "pearson":
            self._pair = np.corrcoef(H, rowvar=False)
        elif self.pair_metric == "tail_calibrated":
            lam = tail_dependence_matrix(H, k=k)
            corr = np.corrcoef(H, rowvar=False)
            rng = np.random.default_rng(self.seed)
            excess = np.zeros_like(lam)
            for a in range(d):
                for b in range(a + 1, d):
                    cutoff = gaussian_null_cutoff(n, k, float(corr[a, b]), self.n_null, rng)
                    excess[a, b] = excess[b, a] = lam[a, b] - cutoff
            self._pair = excess
            self._raw_tail = lam
        else:
            raise ValueError(f"unknown pair_metric {self.pair_metric!r}")
        np.fill_diagonal(self._pair, 1.0)
        self.fit_seconds = time.perf_counter() - t0
        return self

    @property
    def pair_matrix(self) -> np.ndarray:
        if self._pair is None:
            raise RuntimeError("router is not fitted")
        return self._pair

    def route(self, x: np.ndarray) -> RoutingDecision:
        if self._sorted is None or self._pair is None:
            raise RuntimeError("router is not fitted")
        t0 = time.perf_counter()
        x = np.asarray(x, dtype=np.float64)

        # pseudo-observations by binary search against the stored history
        ranks = np.empty(x.shape[0])
        for j in range(x.shape[0]):
            ranks[j] = np.searchsorted(self._sorted[:, j], x[j], side="right")
        u = ranks / (self._n + 1.0)

        primary = int(np.argmax(u))
        in_tail = bool(u[primary] >= self.tail_quantile)
        co: list[str] = []
        if in_tail:
            for j in range(len(self.names)):
                if j == primary:
                    continue
                if self._pair[primary, j] >= self.pair_threshold and u[j] >= self.tail_quantile:
                    co.append(self.names[j])

        return RoutingDecision(
            primary=self.names[primary],
            co_activated=co,
            pseudo_observations={n: float(v) for n, v in zip(self.names, u, strict=True)},
            in_upper_tail=in_tail,
            decision_us=(time.perf_counter() - t0) * 1e6,
            meta={"pair_metric": self.pair_metric, "k": self.k},
        )
