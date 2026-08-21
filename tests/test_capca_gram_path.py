"""The Gram fast path must agree with the reference d x d implementation.

spectrum_via_gram() exists purely as an optimisation: the direct version built the
full d x d covariance and eigendecomposed it, which for down_proj (d=9216, n=1037)
spent ~40 minutes resolving 8000 structurally-zero eigenvalues. An optimisation that
changes the answer is not an optimisation, so it is pinned against the slow path here.
"""

import importlib.util
from pathlib import Path

import torch

from runtime.riemannian_covariance import ledoit_wolf_from_samples

_spec = importlib.util.spec_from_file_location(
    "capca", Path(__file__).resolve().parents[1]
    / "benchmarks/factory/geometry/activation_init_premise/probe_capca_premise.py")
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
spectrum_via_gram = _m.spectrum_via_gram
RANK = _m.RANK


def acts(n, d, spikes=5, seed=0, gap_at=None):
    """Spiked activations -- a low-rank factor plus noise, like a real residual stream.

    `gap_at` puts a magnitude cliff after that many spikes, so the leading subspace of
    that size is well separated and therefore identifiable. Without a cliff, equal
    spikes give adjacent eigenvalues within ~20% of each other and the k-dim subspace
    is only defined up to rotation inside the near-degenerate block -- which makes any
    projector comparison a test of luck.
    """
    g = torch.Generator().manual_seed(seed)
    load = torch.randn(spikes, d, generator=g, dtype=torch.float64) * 3.0
    if gap_at is not None:
        load[gap_at:] *= 0.08
    f = torch.randn(n, spikes, generator=g, dtype=torch.float64)
    return f @ load + 0.4 * torch.randn(n, d, generator=g, dtype=torch.float64)


def test_delta_matches_reference():
    for n, d in [(64, 200), (200, 64), (120, 120)]:
        H = acts(n, d)
        _, ref = ledoit_wolf_from_samples(H)
        got = spectrum_via_gram(H)["delta"]
        assert abs(got - ref) < 1e-8, f"n={n} d={d}: {got} vs {ref}"


def test_topk_energy_matches_reference():
    """Raw top-k energy fraction against eigvalsh of the (uncentred) sample covariance."""
    for n, d in [(64, 200), (200, 64)]:
        H = acts(n, d)
        S = (H.T @ H) / n
        ev = torch.linalg.eigvalsh(S).flip(0).clamp(min=0)
        ref = float(ev[:RANK].sum() / ev.sum())
        got = spectrum_via_gram(H)["raw_topk_energy"]
        assert abs(got - ref) < 1e-9, f"n={n} d={d}: {got} vs {ref}"


def test_eigvecs_span_the_same_subspace():
    """Retention only depends on the SUBSPACE, so compare projectors, not vectors --
    eigenvector signs and the ordering of degenerate pairs are both arbitrary."""
    H = acts(80, 160)
    S = (H.T @ H) / 80
    ref = torch.linalg.eigh(S)[1].flip(1)[:, :RANK]
    got = spectrum_via_gram(H)["eigvecs"]
    P_ref, P_got = ref @ ref.T, got @ got.T
    assert torch.norm(P_ref - P_got) < 1e-6


def test_eigvecs_orthonormal():
    got = spectrum_via_gram(acts(80, 160))["eigvecs"]
    assert torch.norm(got.T @ got - torch.eye(RANK, dtype=torch.float64)) < 1e-8


def test_rank_deficient_case_is_reported():
    """n < RANK leaves fewer than 8 usable directions; the padding must not be silent."""
    out = spectrum_via_gram(acts(5, 300))
    assert out["rank"] == 5
    assert out["eigvecs"].shape == (300, RANK)
    assert torch.allclose(out["eigvecs"][:, 5:], torch.zeros(300, RANK - 5, dtype=torch.float64))


def test_shrinkage_does_not_rotate_eigvecs():
    """The property the probe relies on: retention is identical on S and Sigma_LW.

    Needs MORE than RANK spikes. With 5 spikes the 6th-8th eigenvalues sit in the
    noise floor (1.07, 0.98, 0.95 -- separated by <10%), the top-8 subspace is not
    identifiable, and the comparison fails on arbitrary rotation inside a degenerate
    block rather than on anything the code did. That is a property of eigenvectors,
    not a defect, so the fixture has to avoid it.
    """
    H = acts(90, 220, spikes=12, gap_at=RANK)
    Sigma, delta = ledoit_wolf_from_samples(H)
    ev = torch.linalg.eigvalsh(Sigma.double()).flip(0)
    assert ev[RANK - 1] > 1.5 * ev[RANK], "top-8 block is not separated; fixture is bad"
    lw = torch.linalg.eigh(Sigma.double())[1].flip(1)[:, :RANK]
    raw = spectrum_via_gram(H)["eigvecs"]
    assert delta > 0
    assert torch.norm(lw @ lw.T - raw @ raw.T) < 1e-4


def test_reference_impl_downcasts_to_float32():
    """Documented wart: ledoit_wolf_from_samples computes in float64 and returns
    float32. Harmless for a distance, lossy for an eigendecomposition -- which is why
    spectrum_via_gram stays in float64 end to end rather than reusing it."""
    Sigma, _ = ledoit_wolf_from_samples(acts(40, 80))
    assert Sigma.dtype is torch.float32
