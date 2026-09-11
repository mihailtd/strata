"""Unit tests for Chapter 21 High-Dimensional Leverage Outlier Detection & Mixed-Precision W4A16 Packing."""

import pytest
import torch
from leverage_quantization import (
    HighDimensionalLeverageScorer,
    MixedPrecisionW4A16Packer,
)


def test_leverage_detects_synthetic_outlier_spikes():
    """Verifies that HighDimensionalLeverageScorer recovers 100% of injected outlier channels."""
    torch.manual_seed(42)
    N, D = 256, 2048
    X = torch.randn(N, D)

    # Inject 8 extreme outlier channels (magnitude ~ 85.0)
    true_outliers = torch.tensor([42, 189, 512, 777, 1024, 1337, 1600, 2000])
    X[:, true_outliers] *= 85.0

    scorer = HighDimensionalLeverageScorer(k_components=8)
    detected = scorer.identify_outlier_channels(X, top_k=8)

    assert detected.shape[0] == 8
    assert torch.equal(detected.cpu(), true_outliers)


def test_leverage_invariance_to_global_scaling():
    """Verifies that multiplying all activations by a constant scale does not change detected channels."""
    torch.manual_seed(123)
    N, D = 128, 1024
    X = torch.randn(N, D)
    true_outliers = torch.tensor([10, 50, 100, 500])
    X[:, true_outliers] *= 50.0

    scorer = HighDimensionalLeverageScorer(k_components=4)
    detected_1 = scorer.identify_outlier_channels(X, top_k=4)
    detected_2 = scorer.identify_outlier_channels(X * 3.7, top_k=4)

    assert torch.equal(detected_1, detected_2)


def test_mixed_precision_splitting():
    """Verifies correct tensor shapes and lossless retention of protected BF16 channels."""
    torch.manual_seed(777)
    D_in, D_out = 2048, 4096
    W = torch.randn(D_in, D_out, dtype=torch.bfloat16)

    outliers = torch.tensor([12, 34, 56, 78])
    packer = MixedPrecisionW4A16Packer(group_size=128)

    W_normal, W_protected, normal_indices = packer.split_weights(W, outliers)

    assert W_normal.shape == (2048 - 4, 4096)
    assert W_protected.shape == (4, 4096)
    assert normal_indices.shape == (2048 - 4,)
    # Protected slice must match exact original values
    assert torch.equal(W_protected, W[outliers, :])


def test_reconstruction_snr_gain_over_uniform_w4():
    """Verifies that Mixed-Precision W4A16 with Leverage Outlier protection achieves high output SNR."""
    torch.manual_seed(999)
    N, D_in, D_out = 128, 1024, 2048
    X = torch.randn(N, D_in)
    outliers = torch.tensor([15, 88, 240, 512, 800])
    X[:, outliers] *= 120.0  # Massive outlier spikes

    # Synthetic weight matrix
    W = torch.randn(D_in, D_out, dtype=torch.bfloat16) * 0.02
    Y_true = X.to(torch.float32) @ W.to(torch.float32)

    # Leverage Outlier Detection
    scorer = HighDimensionalLeverageScorer(k_components=4)
    detected = scorer.identify_outlier_channels(X, top_k=5)

    packer = MixedPrecisionW4A16Packer(group_size=128)
    W_normal, W_protected, normal_indices = packer.split_weights(W, detected)
    W_q, scales = packer.quantize_normal_channels(W_normal)
    W_rec_leverage = packer.reconstruct_mixed_precision(
        W_q, scales, W_protected, detected, normal_indices, (D_in, D_out)
    )

    # Naive Uniform W4 Quantization (no outlier protection)
    W_q_naive, scales_naive = packer.quantize_normal_channels(W)
    D_norm = W_q_naive.shape[0]
    n_groups = scales_naive.shape[0]
    grouped_naive = W_q_naive.view(n_groups, 128, D_out).to(torch.float32)
    W_rec_naive = (grouped_naive * scales_naive.unsqueeze(1).to(torch.float32)).view(D_in, D_out).to(torch.bfloat16)

    Y_naive = X.to(torch.float32) @ W_rec_naive.to(torch.float32)
    Y_leverage = X.to(torch.float32) @ W_rec_leverage.to(torch.float32)

    # Compute Output Signal-to-Noise Ratio
    y_norm_sq = torch.sum(Y_true ** 2).item()
    err_leverage = torch.sum((Y_true - Y_leverage) ** 2).item()
    err_naive = torch.sum((Y_true - Y_naive) ** 2).item()

    snr_leverage = 10.0 * torch.log10(torch.tensor(y_norm_sq / err_leverage)).item()
    snr_naive = 10.0 * torch.log10(torch.tensor(y_norm_sq / err_naive)).item()

    # Leverage mixed-precision must achieve significantly higher output SNR (+10 dB gain)
    assert snr_leverage > snr_naive + 10.0, f"Leverage SNR ({snr_leverage:.2f} dB) should outperform naive ({snr_naive:.2f} dB)"
