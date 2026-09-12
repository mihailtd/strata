"""Sanity check for benchmark_activation_covariance_geodesics.py's eval-prompt loader.

Split out of the former tests/test_benchmark_activation_covariance.py, which
mixed this with apps/runtime/riemannian_covariance.py tests (those moved to
apps/runtime/tests/test_riemannian_covariance.py).
"""

from experiments.factory.geometry.riemannian_metric.benchmark_activation_covariance_geodesics import (
    load_eval_prompts,
)


def test_load_eval_prompts():
    shared_suite, domain_prompts = load_eval_prompts(max_prompts_per_domain=3)
    assert len(shared_suite) > 0
    assert len(domain_prompts) == 6
    for d in ["astral", "postgresql", "duckdb", "financial", "python_modern", "python_web"]:
        assert d in domain_prompts
        assert len(domain_prompts[d]) > 0
