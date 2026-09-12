"""Tests for Dynamic Alpha Calibration endpoint and mathematical bounds."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from runtime.alpha_calibration import calibrate_adapter_alpha, measure_adapter_perturbation
from runtime.canon import REPO_ROOT
from runtime.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_measure_adapter_perturbation_postgresql():
    """Verify perturbation measurement on PostgreSQL adapter."""
    adapter_dir = REPO_ROOT / "results" / "adapters" / "m2_postgresql_r8a128_v7"
    if not adapter_dir.exists():
        adapter_dir = REPO_ROOT / "results" / "adapters" / "m2_postgresql_r8a128_v6"

    if not adapter_dir.exists():
        pytest.skip("No postgresql adapter on disk")

    m = measure_adapter_perturbation(adapter_dir)
    assert "r" in m
    assert "alpha" in m
    assert "dw_over_w" in m
    assert m["dw_over_w"] > 0.01
    assert m["pairs"] > 0


def test_calibrate_adapter_alpha_precision_curve():
    """Verify IEEE 754 precision floor and alpha_opt selection."""
    adapter_dir = REPO_ROOT / "results" / "adapters" / "m2_astral_r8a128_v7"
    if not adapter_dir.exists():
        adapter_dir = REPO_ROOT / "results" / "adapters" / "m2_astral_r8a128_v6"

    if not adapter_dir.exists():
        pytest.skip("No astral adapter on disk")

    res = calibrate_adapter_alpha(
        adapter_dir=adapter_dir,
        alphas=[16, 32, 48, 64, 80, 96, 112, 128],
        apply=False,
    )

    assert "alpha_min" in res
    assert "alpha_opt" in res
    assert "curve" in res
    assert len(res["curve"]) == 8

    # Check that merge error decreases as alpha increases
    curve = res["curve"]
    assert curve[0]["merge_err_pct"] > curve[-1]["merge_err_pct"]
    assert res["alpha_min"] >= 16


def test_api_factory_calibrate_alpha_endpoint(client):
    """Verify FastAPI endpoint /api/factory/calibrate_alpha."""
    resp = client.post(
        "/api/factory/calibrate_alpha",
        json={"domain": "postgresql", "apply": False},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "alpha_opt" in data
    assert "alpha_min" in data
    assert "curve" in data
    assert data["applied"] is False


def test_api_factory_calibrate_alpha_404(client):
    """Verify 404 response on non-existent domain."""
    resp = client.post(
        "/api/factory/calibrate_alpha",
        json={"domain": "non_existent_domain_xyz_123"},
    )
    assert resp.status_code == 404
