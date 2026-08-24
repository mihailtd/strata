"""Unit tests for the four CPU statistical estimators (Vecchia, CLIME, GEE, copula).

These are correctness tests, not benchmarks: every assertion here is a property the
estimator must satisfy for its benchmark numbers to mean anything. Where an estimator
is known to be biased (the tail dependence coefficient on tail-independent data), the
test asserts the BIAS EXISTS rather than pretending it does not -- a future change that
silently "fixes" it by changing the estimand should fail here.

No GPU, no model loading. The slowest test is a few hundred milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from runtime.clime_precision import (
    clime_column,
    clime_lambda_theoretical,
    clime_precision,
    graphical_lasso_admm,
    max_constraint_violation,
    naive_inverse_precision,
    sparsity,
)
from runtime.copula_routing import (
    CopulaTailRouter,
    gaussian_null_cutoff,
    pseudo_observations,
    sample_gaussian_copula,
    sample_gumbel_copula,
    sample_t_copula,
    tail_dependence_matrix,
    theoretical_lambda_u_gumbel,
    theoretical_lambda_u_t,
    upper_tail_dependence,
)
from runtime.cpu_bench import median_iqr, parse_bootstrap_flags
from runtime.gee_trajectory import DriftMonitor, fit_gee, working_correlation_inverse
from runtime.vecchia_precision import (
    band_mass_fraction,
    banded_precision_solve,
    banded_solve_lapack,
    conditional_decay_profile,
    dense_nll,
    dense_precision,
    fit_vecchia,
    fit_vecchia_batched,
    gaussian_kl,
    select_horizon,
    to_banded_storage,
    to_dense_precision,
    vecchia_nll,
    vecchia_quadratic,
)


def ar2_samples(n: int, L: int, seed: int = 0) -> np.ndarray:
    """Layer-indexed AR(2) field: conditional dependence extends exactly 2 layers back."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, L))
    X[:, 0] = rng.standard_normal(n)
    X[:, 1] = 0.7 * X[:, 0] + 0.5 * rng.standard_normal(n)
    for j in range(2, L):
        X[:, j] = 0.6 * X[:, j - 1] - 0.3 * X[:, j - 2] + 0.5 * rng.standard_normal(n)
    return X


# ---------------------------------------------------------------------------
# Vecchia
# ---------------------------------------------------------------------------
def test_full_band_vecchia_equals_dense_inverse():
    """At m = L-1 the approximation is exact: it IS the modified Cholesky of the sample cov."""
    X = ar2_samples(4000, 10)
    Theta_full = to_dense_precision(fit_vecchia(X, m=9))
    Theta_dense, _ = dense_precision(X, ridge=1e-12)
    assert np.abs(Theta_full - Theta_dense).max() / np.abs(Theta_dense).max() < 1e-4


@pytest.mark.parametrize(("L", "m"), [(32, 2), (32, 8), (12, 11), (6, 2), (80, 4)])
def test_batched_fit_matches_loop_fit(L, m):
    """The vectorised fit is an optimisation, not a different estimator."""
    X = ar2_samples(1024, L, seed=L)
    loop = fit_vecchia(X, m=m)
    batched = fit_vecchia_batched(X, m=m)
    assert np.abs(loop.coeffs - batched.coeffs).max() < 1e-10
    assert np.abs(loop.resid_var - batched.resid_var).max() < 1e-5


def test_banded_solve_matches_dense_solve():
    X = ar2_samples(2000, 20)
    f = fit_vecchia_batched(X, m=3)
    Theta = to_dense_precision(f)
    y = np.random.default_rng(1).standard_normal(20)

    assert np.allclose(banded_precision_solve(f, y), np.linalg.solve(Theta, y), atol=1e-10)
    assert np.allclose(banded_solve_lapack(to_banded_storage(f), y), np.linalg.solve(Theta, y), atol=1e-10)


def test_quadratic_form_and_nll_match_the_dense_gaussian():
    X = ar2_samples(2000, 16)
    f = fit_vecchia_batched(X, m=2)
    Theta = to_dense_precision(f)
    Xc = X[:20] - f.mean

    assert np.allclose(vecchia_quadratic(f, X[:20]), np.einsum("ij,jk,ik->i", Xc, Theta, Xc), atol=1e-8)
    assert vecchia_nll(f, X) == pytest.approx(dense_nll(Theta, f.mean, X), rel=1e-9)


def test_horizon_selection_recovers_the_generating_order():
    """AR(2) data must select m = 2: not 1 (underfits), not 8 (buys nothing)."""
    X = ar2_samples(6000, 16, seed=3)
    chosen = select_horizon(X[:3000], X[3000:], m_grid=(0, 1, 2, 3, 4, 6, 8))["selected_m"]
    assert chosen == 2


def test_band_mass_and_decay_profile_see_the_ar2_structure():
    X = ar2_samples(6000, 20, seed=4)
    Theta, _ = dense_precision(X, ridge=1e-8)

    assert band_mass_fraction(Theta, 2) > 0.85
    decay = conditional_decay_profile(Theta, max_lag=5)
    assert decay[1] > decay[3] and decay[2] > decay[4]


def test_gaussian_kl_is_zero_against_itself_and_positive_otherwise():
    X = ar2_samples(4000, 12, seed=5)
    Theta, _ = dense_precision(X, ridge=1e-10)
    Sigma = np.linalg.inv(Theta)

    assert gaussian_kl(Sigma, Theta) == pytest.approx(0.0, abs=1e-6)
    assert gaussian_kl(Sigma, to_dense_precision(fit_vecchia_batched(X, m=1))) > 0.0


def test_vecchia_rejects_bad_input():
    with pytest.raises(ValueError):
        fit_vecchia(np.zeros((5, 5, 5)), m=1)
    with pytest.raises(ValueError):
        fit_vecchia_batched(np.zeros((10, 4)), m=-1)


# ---------------------------------------------------------------------------
# CLIME
# ---------------------------------------------------------------------------
def banded_truth(d: int, off: float = 0.35) -> np.ndarray:
    Theta = np.eye(d) * 1.2
    for i in range(d - 1):
        Theta[i, i + 1] = Theta[i + 1, i] = off
    return Theta


def test_clime_constraint_bound_holds_on_feasible_columns():
    """The whole point of CLIME: ||S theta_j - e_j||_inf <= lambda, by construction."""
    rng = np.random.default_rng(2)
    d, n = 24, 16                                   # n < d on purpose
    Sigma = np.linalg.inv(banded_truth(d))
    X = rng.multivariate_normal(np.zeros(d), Sigma, size=n)
    S = np.cov(X, rowvar=False, bias=True)

    lam = 0.4
    res = clime_precision(S, lam)
    feasible = np.where(res.feasible)[0]
    assert feasible.size > 0
    assert max_constraint_violation(S, res.Theta_raw, feasible) <= lam + 1e-6


def test_clime_output_is_symmetric_and_sparse():
    rng = np.random.default_rng(3)
    d, n = 20, 40
    S = np.cov(rng.multivariate_normal(np.zeros(d), np.linalg.inv(banded_truth(d)), size=n),
               rowvar=False, bias=True)
    res = clime_precision(S, 0.3)

    assert np.allclose(res.Theta, res.Theta.T)
    assert sparsity(res.Theta) > 0.5


def test_clime_column_reports_infeasibility_instead_of_guessing():
    """Below the feasibility frontier the LP has no solution; that must be visible."""
    rng = np.random.default_rng(4)
    d, n = 20, 5
    S = np.cov(rng.standard_normal((n, d)), rowvar=False, bias=True)

    _, feasible = clime_column(S, 0, lam=1e-4)
    assert feasible is False


def test_threaded_clime_matches_serial_result():
    rng = np.random.default_rng(5)
    d, n = 16, 32
    S = np.cov(rng.multivariate_normal(np.zeros(d), np.linalg.inv(banded_truth(d)), size=n),
               rowvar=False, bias=True)

    assert np.allclose(clime_precision(S, 0.3, n_jobs=1).Theta, clime_precision(S, 0.3, n_jobs=4).Theta)


def test_naive_inverse_blows_up_when_n_is_below_d():
    """The failure CLIME exists to avoid, asserted rather than asserted-about."""
    rng = np.random.default_rng(6)
    d, n = 32, 16
    S = np.cov(rng.standard_normal((n, d)), rowvar=False, bias=True)

    assert np.abs(naive_inverse_precision(S, ridge=0.0)).max() > 1e6
    assert np.abs(naive_inverse_precision(S, ridge=0.5)).max() < 1e3


def test_graphical_lasso_converges_to_a_positive_definite_estimate():
    rng = np.random.default_rng(7)
    d, n = 16, 64
    S = np.cov(rng.multivariate_normal(np.zeros(d), np.linalg.inv(banded_truth(d)), size=n),
               rowvar=False, bias=True)

    Theta, info = graphical_lasso_admm(S, lam=0.1)
    assert info["converged"]
    assert np.all(np.linalg.eigvalsh(0.5 * (Theta + Theta.T)) > 0)


def test_theoretical_lambda_scales_with_log_d_over_n():
    assert clime_lambda_theoretical(100, 64) < clime_lambda_theoretical(25, 64)
    assert clime_lambda_theoretical(100, 1024) > clime_lambda_theoretical(100, 64)


# ---------------------------------------------------------------------------
# GEE
# ---------------------------------------------------------------------------
def ar1_clusters(rng: np.random.Generator, K: int, T: int, rho: float, slope: float = 0.0):
    y_list, X_list = [], []
    t = np.arange(T, dtype=float)
    t = t - t.mean()
    for _ in range(K):
        e = np.empty(T)
        e[0] = rng.normal(0.0, 1.0 / np.sqrt(max(1 - rho**2, 1e-6)))
        for i in range(1, T):
            e[i] = rho * e[i - 1] + rng.normal()
        y_list.append(0.5 + slope * t + e)
        X_list.append(np.column_stack([np.ones(T), t]))
    return y_list, X_list


def test_independence_gaussian_gee_is_exactly_ols():
    rng = np.random.default_rng(8)
    y_list, X_list = ar1_clusters(rng, K=6, T=12, rho=0.5)
    fit = fit_gee(y_list, X_list, corr="independence")
    ols = np.linalg.lstsq(np.vstack(X_list), np.concatenate(y_list), rcond=None)[0]

    assert np.allclose(fit.beta, ols, atol=1e-8)


@pytest.mark.parametrize("rho", [0.3, 0.6, 0.9])
def test_ar1_working_correlation_recovers_rho(rho):
    rng = np.random.default_rng(int(rho * 100))
    y_list, X_list = ar1_clusters(rng, K=40, T=25, rho=rho)
    fit = fit_gee(y_list, X_list, corr="ar1")

    assert fit.converged
    assert abs(fit.alpha - rho) < 0.08


def test_sandwich_tracks_the_empirical_spread_and_the_naive_se_does_not():
    """The claim the whole GEE benchmark rests on, at small scale."""
    rng = np.random.default_rng(9)
    slopes, naive_se, robust_se = [], [], []
    for _ in range(120):
        y_list, X_list = ar1_clusters(rng, K=25, T=20, rho=0.7)
        ar1 = fit_gee(y_list, X_list, corr="ar1")
        indep = fit_gee(y_list, X_list, corr="independence")
        slopes.append(ar1.beta[1])
        robust_se.append(ar1.se_sandwich[1])
        naive_se.append(indep.se_model[1])

    empirical = float(np.std(slopes))
    assert abs(np.mean(robust_se) - empirical) / empirical < 0.25
    assert np.mean(naive_se) < 0.75 * empirical      # the naive SE is too small, by a lot


def test_ar1_correlation_inverse_matches_direct_inversion():
    for alpha in (0.0, 0.4, 0.85):
        T = 7
        R = alpha ** np.abs(np.subtract.outer(np.arange(T), np.arange(T)))
        assert np.allclose(working_correlation_inverse("ar1", alpha, T), np.linalg.inv(R), atol=1e-9)


def test_working_correlation_rejects_unknown_structure():
    with pytest.raises(ValueError):
        working_correlation_inverse("toeplitz", 0.5, 5)
    with pytest.raises(ValueError):
        fit_gee([np.zeros(4)], [np.ones((4, 1))], corr="banana")


def test_drift_monitor_flags_drift_and_stays_quiet_without_it():
    """Asserted as a RATE over repeats, not on one draw.

    A detector with a correct 5% size fires on ~1 null draw in 20 by definition, so a
    single-draw assertion here would be a coin flip dressed as a test. The benchmark
    measures the exact size; this test only has to catch a detector that is broken.
    """
    rng = np.random.default_rng(11)
    repeats = 20

    false_alarms = 0
    detections = 0
    for _ in range(repeats):
        quiet = DriftMonitor(min_turns=5)
        drifting = DriftMonitor(min_turns=5)
        for cid in range(30):
            for v in ar1_clusters(rng, K=1, T=20, rho=0.6)[0][0]:
                quiet.observe(f"c{cid}", float(v))
            for v in ar1_clusters(rng, K=1, T=20, rho=0.6, slope=0.25)[0][0]:
                drifting.observe(f"c{cid}", float(v))
        false_alarms += int(quiet.assess()["drifting"])
        detections += int(drifting.assess()["drifting"])

    assert false_alarms <= repeats * 0.3
    assert detections >= repeats * 0.9


def test_drift_monitor_reports_insufficient_data():
    monitor = DriftMonitor(min_turns=10)
    monitor.observe("only-one", 0.5)
    assert monitor.assess()["status"] == "insufficient_data"


# ---------------------------------------------------------------------------
# Copula tail dependence
# ---------------------------------------------------------------------------
def test_pseudo_observations_are_strictly_inside_the_unit_square():
    U = pseudo_observations(np.random.default_rng(12).standard_normal((200, 3)))
    assert U.min() > 0.0 and U.max() < 1.0
    assert np.allclose(np.sort(U[:, 0]), np.arange(1, 201) / 201.0)


def test_pseudo_observations_are_invariant_to_monotone_transforms():
    """The property that lets the router skip Gaussianising its inputs."""
    X = np.abs(np.random.default_rng(13).standard_normal((300, 2))) + 0.1
    assert np.allclose(pseudo_observations(X), pseudo_observations(np.exp(3.0 * X)))


def test_gumbel_tail_dependence_is_recovered_within_tolerance():
    rng = np.random.default_rng(14)
    for theta in (1.5, 2.0):
        U = pseudo_observations(sample_gumbel_copula(20000, theta, rng))
        est = upper_tail_dependence(U, 0, 1, k=140)
        assert abs(est - theoretical_lambda_u_gumbel(theta)) < 0.06


def test_t_copula_has_more_tail_dependence_than_gaussian_at_equal_correlation():
    rng = np.random.default_rng(15)
    k = 80
    gauss = upper_tail_dependence(pseudo_observations(sample_gaussian_copula(8000, 0.7, rng)), 0, 1, k)
    t_cop = upper_tail_dependence(pseudo_observations(sample_t_copula(8000, 0.7, 4.0, rng)), 0, 1, k)
    assert t_cop > gauss


def test_estimator_is_biased_upward_on_tail_independent_data():
    """Documented failure mode: the Gaussian copula has lambda_U = 0 EXACTLY, and the
    finite-sample estimate does not. Absolute cutoffs are therefore unusable, which is
    why `gaussian_null_cutoff` and the calibrated router metric exist."""
    rng = np.random.default_rng(16)
    est = upper_tail_dependence(pseudo_observations(sample_gaussian_copula(4096, 0.7, rng)), 0, 1, k=82)
    assert est > 0.15
    assert gaussian_null_cutoff(2048, 41, 0.7, n_null=24, rng=rng) > 0.15


def test_theoretical_lambda_u_formulas():
    assert theoretical_lambda_u_gumbel(1.0) == pytest.approx(0.0)
    assert theoretical_lambda_u_gumbel(2.0) == pytest.approx(2 - np.sqrt(2))
    assert theoretical_lambda_u_t(0.7, 4.0) > 0.0
    assert theoretical_lambda_u_t(0.7, 100.0) < theoretical_lambda_u_t(0.7, 2.0)


def test_tail_dependence_matrix_is_symmetric_with_unit_diagonal():
    M = tail_dependence_matrix(np.random.default_rng(17).standard_normal((1000, 4)), k=40)
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 1.0)


def test_upper_tail_dependence_rejects_impossible_k():
    U = pseudo_observations(np.random.default_rng(18).standard_normal((100, 2)))
    with pytest.raises(ValueError):
        upper_tail_dependence(U, 0, 1, k=0)
    with pytest.raises(ValueError):
        upper_tail_dependence(U, 0, 1, k=100)


def test_calibrated_router_prefers_the_tail_dependent_pair_over_the_correlated_one():
    """Columns 0,1 are tail dependent; columns 2,3 are strongly correlated with
    lambda_U = 0. The raw metrics can be fooled; the calibrated one must not be."""
    from scipy.stats import norm

    rng = np.random.default_rng(19)
    n = 3000
    H = np.empty((n, 4))
    H[:, :2] = sample_gumbel_copula(n, 3.0, rng)
    Z = rng.multivariate_normal(np.zeros(2), np.array([[1.0, 0.9], [0.9, 1.0]]), size=n)
    H[:, 2:] = norm.cdf(Z)
    names = ["a", "b", "c", "d"]

    calibrated = CopulaTailRouter(pair_metric="tail_calibrated", seed=1).fit(H, names)
    assert calibrated.pair_matrix[0, 1] > calibrated.pair_matrix[2, 3]

    pearson = CopulaTailRouter(pair_metric="pearson").fit(H, names)
    assert pearson.pair_matrix[2, 3] > pearson.pair_matrix[0, 1]


def test_router_co_activates_only_in_the_joint_upper_tail():
    rng = np.random.default_rng(20)
    H = sample_gumbel_copula(2000, 4.0, rng)
    router = CopulaTailRouter(pair_metric="tail", pair_threshold=0.1, tail_quantile=0.95).fit(H, ["x", "y"])

    assert router.route(np.array([0.999, 0.999])).co_activated == ["y"]
    assert router.route(np.array([0.5, 0.5])).co_activated == []
    assert router.route(np.array([0.999, 0.2])).co_activated == []


def test_router_validates_its_inputs():
    router = CopulaTailRouter()
    with pytest.raises(RuntimeError):
        router.route(np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        router.fit(np.zeros((10, 3)), ["only", "two"])
    with pytest.raises(ValueError):
        CopulaTailRouter(pair_metric="spearman").fit(np.zeros((10, 2)), ["a", "b"])


# ---------------------------------------------------------------------------
# Activation features
# ---------------------------------------------------------------------------
def test_low_rank_response_norm_identity():
    """The identity `real_expert_response_magnitudes` relies on to stay cheap.

    For dW = s * U @ V, the per-token norm of the delta output is

        ||s * (x @ V') @ U'||  ==  s * sqrt( z' (U'U) z ),   z = x @ V'

    so the [T, d_out] output never has to be materialised. This is exact, not an
    approximation -- if it ever stops holding, every real-token expert magnitude is
    silently wrong, and nothing else in the pipeline would notice.
    """
    rng = np.random.default_rng(21)
    T, r, d_in, d_out = 17, 8, 64, 128
    scale = 16.0
    U = rng.standard_normal((d_out, r))
    V = rng.standard_normal((r, d_in))
    x = rng.standard_normal((T, d_in))

    direct = np.linalg.norm(scale * ((x @ V.T) @ U.T), axis=1)
    z = x @ V.T
    cheap = scale * np.sqrt(np.maximum(np.einsum("ta,ab,tb->t", z, U.T @ U, z), 0.0))

    assert np.allclose(direct, cheap, rtol=1e-10, atol=1e-10)


def test_corpus_dirs_cover_every_expert_domain():
    """financial -> financial_planning is a real legacy mismatch; keep it mapped."""
    from runtime.activation_features import CORPUS_DIRS
    from runtime.canon import DOMAINS

    for domain in DOMAINS:
        if domain in ("merged_sql", "merged_all"):
            continue
        assert domain in CORPUS_DIRS, f"{domain} has no corpus directory mapping"
    assert CORPUS_DIRS["financial"] == "financial_planning"


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------
def test_median_iqr_reports_interpolated_quantiles():
    stats = median_iqr([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["median"] == 3.0
    assert stats["q1"] == 2.0 and stats["q3"] == 4.0
    assert stats["iqr"] == 2.0 and stats["n"] == 5


def test_bootstrap_flags_are_read_before_argparse_would_run():
    assert parse_bootstrap_flags(["prog"]) == {"threads": 8, "allow_gpu": False}
    assert parse_bootstrap_flags(["prog", "--threads", "16"])["threads"] == 16
    assert parse_bootstrap_flags(["prog", "--gpu-capture"])["allow_gpu"] is True
    assert parse_bootstrap_flags(["prog", "--threads", "oops"])["threads"] == 8
