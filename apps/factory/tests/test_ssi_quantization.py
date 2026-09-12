"""Unit tests for Chapter 8.5 Stress-Strength Interference (SSI) Quantization Calibration."""

import torch
from ssi_quantization import StressStrengthInterferenceCalibrator


def test_ssi_moment_calculation():
    """Verify group mean, standard deviation, and kurtosis estimation."""
    torch.manual_seed(42)
    D_in, D_out = 512, 1024
    # Gaussian weights ~ N(0, 1) -> expected kurtosis ~ 3.0
    W_gauss = torch.randn(D_in, D_out)

    calibrator = StressStrengthInterferenceCalibrator(group_size=128)
    mean, std, kurt = calibrator.compute_group_moments(W_gauss)

    assert mean.shape == (4, 1024)
    assert std.shape == (4, 1024)
    assert kurt.shape == (4, 1024)

    # Standard deviation should be close to 1.0, mean close to 0.0, kurtosis close to 3.0
    assert torch.mean(torch.abs(mean)).item() < 0.08
    assert torch.mean(torch.abs(std - 1.0)).item() < 0.08
    assert torch.mean(torch.abs(kurt - 3.0)).item() < 0.50


def test_closed_form_ssi_scales_bounded_by_max():
    """Verify that SSI scales are positive, well-formed, and strictly bounded by naive max."""
    torch.manual_seed(123)
    D_in, D_out = 256, 512
    W = torch.randn(D_in, D_out, dtype=torch.bfloat16)

    calibrator = StressStrengthInterferenceCalibrator(group_size=128)
    scales_ssi = calibrator.calibrate_ssi_scales(W, mode="closed_form_ssi")
    scales_max = calibrator.calibrate_ssi_scales(W, mode="naive_max")

    assert torch.all(scales_ssi > 0)
    # SSI scale should never exceed naive max scale
    assert torch.all(scales_ssi <= scales_max + 1e-4)


def test_ssi_quantization_roundtrip_shape():
    """Verify that quantize_w4 and dequantize_w4 preserve exact tensor dimensions."""
    torch.manual_seed(777)
    D_in, D_out = 384, 768  # 384 = 3 * 128 groups
    W = torch.randn(D_in, D_out, dtype=torch.bfloat16) * 0.05

    calibrator = StressStrengthInterferenceCalibrator(group_size=128)
    W_q, scales = calibrator.quantize_w4(W, mode="closed_form_ssi")
    W_rec = calibrator.dequantize_w4(W_q, scales, (D_in, D_out))

    assert W_q.shape == (384, 768)
    assert scales.shape == (3, 768)
    assert W_rec.shape == (384, 768)
    assert W_q.dtype == torch.int8
    assert scales.dtype == torch.bfloat16
    assert W_rec.dtype == torch.bfloat16


def test_ssi_reduces_total_quantization_distortion():
    """Verify that SSI calibration outperforms naive max scaling in SNR (dB)."""
    torch.manual_seed(999)
    D_in, D_out = 1024, 2048
    # Heavy-tailed Laplace distribution to simulate real LLM weights
    dist = torch.distributions.Laplace(loc=0.0, scale=0.02)
    W = dist.sample((D_in, D_out)).to(torch.bfloat16)

    calibrator = StressStrengthInterferenceCalibrator(group_size=128)

    # 1. Naive Max Quantization
    W_q_max, scales_max = calibrator.quantize_w4(W, mode="naive_max")
    W_rec_max = calibrator.dequantize_w4(W_q_max, scales_max, (D_in, D_out))

    # 2. SSI Calibrated Quantization
    W_q_ssi, scales_ssi = calibrator.quantize_w4(W, mode="closed_form_ssi")
    W_rec_ssi = calibrator.dequantize_w4(W_q_ssi, scales_ssi, (D_in, D_out))

    # Compute Signal-to-Noise Ratio
    w_norm_sq = torch.sum(W.to(torch.float32) ** 2).item()
    err_max = torch.sum((W.to(torch.float32) - W_rec_max.to(torch.float32)) ** 2).item()
    err_ssi = torch.sum((W.to(torch.float32) - W_rec_ssi.to(torch.float32)) ** 2).item()

    snr_max = 10.0 * torch.log10(torch.tensor(w_norm_sq / err_max)).item()
    snr_ssi = 10.0 * torch.log10(torch.tensor(w_norm_sq / err_ssi)).item()

    # SSI must achieve lower MSE / higher SNR
    assert snr_ssi > snr_max, f"SSI SNR ({snr_ssi:.2f} dB) should exceed naive max ({snr_max:.2f} dB)"
    assert snr_ssi >= snr_max + 0.5, (
        f"SSI should provide measurable SNR gain (+0.5 dB min), got +{snr_ssi - snr_max:.2f} dB"
    )
