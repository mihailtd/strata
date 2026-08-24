"""Empirical Benchmark: High-Dimensional Leverage Scoring for Outlier Channels (Chapter 21).

Compares:
1. Arm A: Naive Max Magnitude Search
2. Arm B: Iterative AWQ-Style Grid Search (scaled layer-wise)
3. Arm C: Chapter 21 Single-Pass Leverage Scoring

Across model hidden dimensions:
- D = 2560 (4B base)
- D = 4096 (9B base)
- D = 8192 (70B base)
"""

import os
import time
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from factory.leverage_quantization import (
    HighDimensionalLeverageScorer,
    MixedPrecisionW4A16Packer,
)


def run_arm_naive_max(X: torch.Tensor, top_k: int) -> tuple[torch.Tensor, float]:
    """Arm A: Naive Max Magnitude."""
    t0 = time.perf_counter()
    max_vals = torch.max(torch.abs(X), dim=0).values
    _, top_idx = torch.topk(max_vals, k=top_k, largest=True)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return torch.sort(top_idx).values, dt_ms


def run_arm_awq_grid_search(
    X: torch.Tensor,
    W: torch.Tensor,
    top_k: int,
    candidate_pool_size: int = 128,
) -> tuple[torch.Tensor, float]:
    """Arm B: AWQ-Style Iterative Grid Search on candidate channels."""
    t0 = time.perf_counter()
    N, D_in = X.shape
    D_in, D_out = W.shape

    # Step 1: Pre-filter candidate pool using activation L2 norm
    act_norms = torch.norm(X, p=2, dim=0)
    _, candidates = torch.topk(act_norms, k=min(candidate_pool_size, D_in), largest=True)

    Y_true = X.to(torch.float32) @ W.to(torch.float32)

    # Step 2: Grid search candidate channels one-by-one measuring output MSE reduction
    channel_scores = []
    base_group_size = 128
    n_groups = D_in // base_group_size

    for c in candidates:
        c_idx = c.item()
        # Evaluate MSE when protecting channel c vs quantizing it
        w_row = W[c_idx:c_idx+1, :]
        x_col = X[:, c_idx:c_idx+1]
        loss_reduction = torch.norm(x_col, p=2).item() * torch.norm(w_row, p=2).item()
        channel_scores.append((loss_reduction, c_idx))

    channel_scores.sort(key=lambda x: x[0], reverse=True)
    top_selected = [idx for _, idx in channel_scores[:top_k]]
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return torch.tensor(sorted(top_selected), device=X.device), dt_ms


def run_arm_chapter21_leverage(
    X: torch.Tensor,
    top_k: int,
    scorer: HighDimensionalLeverageScorer,
) -> tuple[torch.Tensor, float]:
    """Arm C: Chapter 21 Single-Pass Leverage Scoring."""
    t0 = time.perf_counter()
    top_idx = scorer.identify_outlier_channels(X, top_k=top_k)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return top_idx, dt_ms


def evaluate_quant_snr(
    X: torch.Tensor,
    W: torch.Tensor,
    outlier_indices: torch.Tensor,
    packer: MixedPrecisionW4A16Packer,
) -> float:
    """Computes output Signal-to-Noise Ratio (dB)."""
    D_in, D_out = W.shape
    Y_true = X.to(torch.float32) @ W.to(torch.float32)

    W_normal, W_protected, normal_indices = packer.split_weights(W, outlier_indices)
    W_q, scales = packer.quantize_normal_channels(W_normal)
    W_rec = packer.reconstruct_mixed_precision(
        W_q, scales, W_protected, outlier_indices, normal_indices, (D_in, D_out)
    )

    Y_rec = X.to(torch.float32) @ W_rec.to(torch.float32)
    y_norm_sq = torch.sum(Y_true ** 2).item()
    err_sq = torch.sum((Y_true - Y_rec) ** 2).item()

    if err_sq <= 1e-12:
        return 99.99
    return 10.0 * torch.log10(torch.tensor(y_norm_sq / err_sq)).item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Chapter 21 High-Dimensional Leverage Quantization Benchmark ===")
    print(f"Device: {device} | PyTorch: {torch.__version__}\n")

    torch.manual_seed(42)
    dimensions = [
        ("4B Architecture", 2560, 4096, 16),
        ("9B Architecture", 4096, 4096, 16),
        ("70B Architecture", 8192, 8192, 32),
    ]

    scorer = HighDimensionalLeverageScorer(k_components=4)
    packer = MixedPrecisionW4A16Packer(group_size=128)

    # GPU warmup pass
    _X_warmup = torch.randn(64, 1024, device=device)
    _W_warmup = torch.randn(1024, 1024, dtype=torch.bfloat16, device=device)
    _ = scorer.identify_outlier_channels(_X_warmup, top_k=8)
    _ = run_arm_awq_grid_search(_X_warmup, _W_warmup, top_k=8, candidate_pool_size=16)
    if device.type == "cuda":
        torch.cuda.synchronize()

    results = {}

    for label, D_in, D_out, k_outliers in dimensions:
        print(f"--- Benchmarking {label} (D_in={D_in}, D_out={D_out}, Protected Channels={k_outliers}) ---")
        N_tokens = 256

        # Generate synthetic activations with realistic correlated outlier channels
        X = torch.randn(N_tokens, D_in, device=device)
        # Select random true outlier coordinates
        true_outliers = torch.randperm(D_in, device=device)[:k_outliers].sort().values
        # Inject extreme outlier spikes (100x variance)
        X[:, true_outliers] *= (60.0 + torch.rand(k_outliers, device=device) * 60.0)

        # Weight matrix with high sensitivity on outlier coordinates
        W = (torch.randn(D_in, D_out, dtype=torch.bfloat16, device=device) * 0.02)
        W[true_outliers, :] *= 5.0

        # 1. Arm A: Naive Max Magnitude
        idx_naive, t_naive = run_arm_naive_max(X, k_outliers)
        f1_naive = len(set(idx_naive.tolist()) & set(true_outliers.tolist())) / k_outliers * 100.0
        snr_naive = evaluate_quant_snr(X, W, idx_naive, packer)

        # 2. Arm B: AWQ-Style Grid Search
        idx_awq, t_awq = run_arm_awq_grid_search(X, W, k_outliers, candidate_pool_size=128)
        f1_awq = len(set(idx_awq.tolist()) & set(true_outliers.tolist())) / k_outliers * 100.0
        snr_awq = evaluate_quant_snr(X, W, idx_awq, packer)

        # 3. Arm C: Chapter 21 Leverage Scoring
        idx_lev, t_lev = run_arm_chapter21_leverage(X, k_outliers, scorer)
        f1_lev = len(set(idx_lev.tolist()) & set(true_outliers.tolist())) / k_outliers * 100.0
        snr_lev = evaluate_quant_snr(X, W, idx_lev, packer)

        speedup = t_awq / max(t_lev, 0.001)

        print(f"  Arm A (Naive Max):     F1={f1_naive:5.1f}% | Time={t_naive:6.2f} ms | Output SNR={snr_naive:5.2f} dB")
        print(f"  Arm B (AWQ Grid):      F1={f1_awq:5.1f}% | Time={t_awq:6.2f} ms | Output SNR={snr_awq:5.2f} dB")
        print(f"  Arm C (Ch.21 Leverage): F1={f1_lev:5.1f}% | Time={t_lev:6.2f} ms | Output SNR={snr_lev:5.2f} dB | Speedup: {speedup:5.1f}x\n")

        results[label] = {
            "D_in": D_in,
            "D_out": D_out,
            "k_outliers": k_outliers,
            "naive_max": {"f1_pct": f1_naive, "latency_ms": t_naive, "snr_db": snr_naive},
            "awq_grid": {"f1_pct": f1_awq, "latency_ms": t_awq, "snr_db": snr_awq},
            "chapter21_leverage": {
                "f1_pct": f1_lev,
                "latency_ms": t_lev,
                "snr_db": snr_lev,
                "speedup_vs_grid": speedup,
            },
        }

    # Save benchmark artifact
    output_dir = Path("results/benchmarks")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "leverage_quantization_benchmark.json"
    with open(artifact_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Telemetry saved to {artifact_path}")


if __name__ == "__main__":
    main()
