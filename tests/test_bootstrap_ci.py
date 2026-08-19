"""Unit tests for paired bootstrap confidence interval calculation and statistical edge cases."""

import numpy as np
import pytest
from benchmarks.multi_turn_execution_benchmark import bootstrap_ci


def test_bootstrap_ci_basic_distribution():
    """Verify bootstrap CI correctly computes empirical mean and 95% interval on known distribution."""
    np.random.seed(42)
    # Uniform values centered around +0.5
    data = [0.4, 0.5, 0.6, 0.5, 0.45, 0.55, 0.48, 0.52]
    mean, low, high = bootstrap_ci(data, n_resamples=2000)

    assert pytest.approx(mean, abs=1e-3) == 0.50
    assert low < mean
    assert high > mean
    assert 0.40 <= low <= 0.50
    assert 0.50 <= high <= 0.60


def test_bootstrap_ci_empty_and_single_element():
    """Verify clean handling of empty arrays and single-element inputs without throwing."""
    # Empty input
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)

    # Single-element input
    assert bootstrap_ci([0.75]) == (0.75, 0.75, 0.75)


def test_bootstrap_ci_zero_variance_identical_elements():
    """Verify zero-variance inputs collapse to identical mean and CI bounds."""
    data = [0.33, 0.33, 0.33, 0.33, 0.33]
    mean, low, high = bootstrap_ci(data, n_resamples=1000)
    assert pytest.approx(mean, abs=1e-5) == 0.33
    assert pytest.approx(low, abs=1e-5) == 0.33
    assert pytest.approx(high, abs=1e-5) == 0.33


def test_bootstrap_ci_all_zeros():
    """Verify all-zero deltas return exact zero interval."""
    data = [0.0, 0.0, 0.0, 0.0]
    mean, low, high = bootstrap_ci(data, n_resamples=500)
    assert mean == 0.0
    assert low == 0.0
    assert high == 0.0
