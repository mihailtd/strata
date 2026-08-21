"""SCAD / MCP thresholding -- the three properties that motivate using them at all.

The point of these penalties over what VelocityGate does today is a specific pair of
guarantees, and each one is a test here:

    sparsity      exactly 0 below lambda          (hard thresholding has it, L1 has it)
    continuity    no jump at the threshold        (hard thresholding does NOT)
    unbiasedness  exactly identity above gamma*lambda  (L1 does NOT)

A test suite that only checked "returns a tensor of the right shape" would pass on an
implementation with either defect.
"""

import pytest
import torch

from runtime.sparse_penalties import (
    hard_threshold, mcp_penalty, mcp_prox, scad_penalty, scad_prox,
    soft_threshold, velocity_gain,
)

LAM = 1.0
GRID = torch.linspace(-8.0, 8.0, 4001, dtype=torch.float64)


@pytest.mark.parametrize("prox,gamma", [(mcp_prox, 3.0), (scad_prox, 3.7)])
def test_exact_zero_below_lambda(prox, gamma):
    z = torch.tensor([-0.99, -0.5, 0.0, 0.5, 0.99], dtype=torch.float64)
    assert torch.all(prox(z, LAM, gamma) == 0.0)


@pytest.mark.parametrize("prox,gamma", [(mcp_prox, 3.0), (scad_prox, 3.7)])
def test_exact_identity_above_gamma_lambda(prox, gamma):
    """UNBIASEDNESS. This is the property L1 does not have and the reason to bother."""
    z = torch.tensor([gamma + 0.01, gamma + 2.0, 50.0, -gamma - 0.01, -50.0],
                     dtype=torch.float64)
    assert torch.allclose(prox(z, LAM, gamma), z, atol=1e-12)


def test_soft_threshold_is_biased_and_that_is_the_contrast():
    """Documents what we are avoiding: L1 shrinks a large coefficient by lambda."""
    z = torch.tensor([50.0], dtype=torch.float64)
    assert torch.allclose(soft_threshold(z, LAM), z - LAM)
    assert not torch.allclose(soft_threshold(z, LAM), z)


@pytest.mark.parametrize("prox,gamma", [(mcp_prox, 3.0), (scad_prox, 3.7)])
def test_continuous_everywhere(prox, gamma):
    """No jump at lambda, 2*lambda or gamma*lambda. Hard thresholding fails this."""
    y = prox(GRID, LAM, gamma)
    step = float((y[1:] - y[:-1]).abs().max())
    grid_step = float(GRID[1] - GRID[0])
    # Lipschitz-bounded: the steepest segment is the middle ramp, slope < ~2
    assert step < 3.0 * grid_step, f"jump of {step} on a grid of {grid_step}"


def test_hard_threshold_is_discontinuous():
    """The defect being fixed, asserted so the contrast cannot silently disappear."""
    y = hard_threshold(GRID, LAM)
    step = float((y[1:] - y[:-1]).abs().max())
    assert step > 0.9, "hard thresholding should jump by ~lambda at the boundary"


@pytest.mark.parametrize("prox,gamma", [(mcp_prox, 3.0), (scad_prox, 3.7)])
def test_monotone_and_odd(prox, gamma):
    y = prox(GRID, LAM, gamma)
    assert torch.all(y[1:] - y[:-1] >= -1e-12)
    assert torch.allclose(prox(-GRID, LAM, gamma), -y, atol=1e-12)


@pytest.mark.parametrize("prox,gamma", [(mcp_prox, 3.0), (scad_prox, 3.7)])
def test_shrinkage_never_overshoots(prox, gamma):
    y = prox(GRID, LAM, gamma)
    assert torch.all(y.abs() <= GRID.abs() + 1e-12)


def test_mcp_approaches_soft_threshold_as_gamma_grows():
    """gamma -> infinity removes the unbiased region and MCP degenerates to L1."""
    z = torch.tensor([2.0, 5.0], dtype=torch.float64)
    assert torch.allclose(mcp_prox(z, LAM, gamma=1e7), soft_threshold(z, LAM), atol=1e-5)


def test_gamma_guards():
    z = torch.tensor([1.0])
    with pytest.raises(ValueError):
        mcp_prox(z, LAM, gamma=1.0)
    with pytest.raises(ValueError):
        scad_prox(z, LAM, gamma=2.0)


@pytest.mark.parametrize("pen,gamma", [(mcp_penalty, 3.0), (scad_penalty, 3.7)])
def test_penalty_is_flat_above_gamma_lambda(pen, gamma):
    """Flat penalty == zero gradient on large coefficients == unbiased."""
    z = torch.tensor([gamma + 1.0, gamma + 9.0], dtype=torch.float64)
    v = pen(z, LAM, gamma)
    assert torch.allclose(v, v[0].expand_as(v), atol=1e-12)


@pytest.mark.parametrize("pen,gamma", [(mcp_penalty, 3.0), (scad_penalty, 3.7)])
def test_penalty_continuous_and_nondecreasing(pen, gamma):
    g = torch.linspace(0.0, 8.0, 4001, dtype=torch.float64)
    v = pen(g, LAM, gamma)
    assert torch.all(v[1:] - v[:-1] >= -1e-12)
    assert float((v[1:] - v[:-1]).abs().max()) < 0.01


# --------------------------------------------------------------------------- gate

def test_velocity_gain_quiets_the_requested_percentile():
    v = torch.linspace(0.1, 10.0, 32)
    g = velocity_gain(v, quiet_percentile=25.0)
    assert torch.all((g >= 0.0) & (g <= 1.0))
    assert float((g == 0.0).float().mean()) >= 0.20      # ~bottom quartile silenced
    assert g[-1] == pytest.approx(1.0)                    # loudest layer untouched


def test_velocity_gain_is_smooth_where_the_binary_mask_jumps():
    """The whole motivation: a layer at the percentile boundary should not flip 1->0."""
    v = torch.linspace(0.1, 10.0, 512)
    g = velocity_gain(v, quiet_percentile=25.0)
    partial = ((g > 0.0) & (g < 1.0)).sum()
    assert partial > 0, "no transition band -- this has degenerated to hard masking"


def test_velocity_gain_survives_nan_and_all_zero():
    """VelocityGate initialises its EMA buffer to NaN and holds it through warmup."""
    assert torch.all(velocity_gain(torch.full((8,), float("nan"))) == 1.0)
    assert torch.all(velocity_gain(torch.zeros(8)) == 1.0)


def test_velocity_gain_rejects_unknown_rule():
    with pytest.raises(KeyError):
        velocity_gain(torch.linspace(0.1, 1.0, 8), rule="lasso")
