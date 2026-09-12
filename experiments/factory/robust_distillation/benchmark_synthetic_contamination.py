"""Empirical Benchmark: Breakdown Points in Noisy Synthetic SFT & Low-Rank Adaptation.

Grounding: Chapter 4 (The Conditional Breakdown Properties of LAD-LASSO Regression,
Boning Feng, Avi Giloni, Jeffrey S. Simonoff).

Experimental Design:
Simulates student adapter distillation on synthetic multi-turn tool traces with
controlled outlier contamination eta in [0%, 5%, 15%, 30%, 50%].
Compares:
- Arm A: Standard Ordinary Least Squares (L2 MSE, epsilon* = 0%)
- Arm B: Huber Bounded Distillation Loss (delta=1.0, epsilon* = 50%)
- Arm C: LAD-LASSO (L1 residual + L1 regularization, epsilon* = 50%)

Measures parameter error ||W_hat - W*|| / ||W*||, clean recovery loss, peak gradient norm,
and empirical breakdown point.
"""

import json
import os
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from runtime.robust_distill import HuberDistillationLoss, LADLassoLoss

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class LowRankStudent(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int = 8, alpha: float = 128.0):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(in_dim, rank) * (1.0 / (in_dim ** 0.5)))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaling * (x @ self.lora_A @ self.lora_B)

    def get_effective_delta_w(self) -> torch.Tensor:
        return self.scaling * (self.lora_A @ self.lora_B)


def run_synthetic_contamination_benchmark():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running Synthetic Contamination Benchmark on: {device}")

    d_in = 512
    d_out = 512
    rank = 8
    alpha = 64.0
    n_samples = 1024
    n_test = 256
    steps = 150
    lr = 0.01

    # Ground truth low-rank weights W*
    A_true = torch.randn(d_in, rank, device=device) * 0.1
    B_true = torch.randn(rank, d_out, device=device) * 0.1
    W_star = (alpha / rank) * (A_true @ B_true)
    norm_W_star = float(torch.norm(W_star, p="fro").item())

    # Clean input training and test data
    X_train = torch.randn(n_samples, d_in, device=device)
    Y_train_clean = X_train @ W_star
    X_test = torch.randn(n_test, d_in, device=device)
    Y_test_clean = X_test @ W_star

    contamination_levels = [0.0, 0.05, 0.15, 0.30, 0.50]
    outlier_magnitude = 50.0  # 50x standard deviation spike

    methods = [
        ("L2_MSE", "Standard L2 MSE (epsilon* = 0%)"),
        ("Huber_Delta1.0", "Huber Bounded Loss (delta=1.0, epsilon* = 50%)"),
        ("LAD_LASSO", "LAD-LASSO L1 Loss (alpha=1e-4, epsilon* = 50%)"),
    ]

    benchmark_results = {
        "benchmark": "Breakdown-Bounded Loss in Noisy Synthetic Adaptation",
        "reference": "Chapter 4 (Feng, Giloni, Simonoff)",
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dimensions": {"d_in": d_in, "d_out": d_out, "rank": rank, "n_samples": n_samples},
        "experiments": [],
    }

    huber_loss_fn = HuberDistillationLoss(delta=1.0)
    lad_lasso_fn = LADLassoLoss(alpha=1e-4)
    l2_loss_fn = nn.MSELoss()

    for eta in contamination_levels:
        print(f"\n--- Contamination Level: {eta*100:.0f}% Outliers ---")
        # Create contaminated dataset
        n_outliers = int(n_samples * eta)
        Y_train_noisy = Y_train_clean.clone()

        if n_outliers > 0:
            outlier_indices = torch.randperm(n_samples)[:n_outliers]
            # Inject extreme outlier spikes simulating broken tool payloads / JSON syntax corruptions
            outlier_noise = (torch.randn(n_outliers, d_out, device=device) * outlier_magnitude)
            Y_train_noisy[outlier_indices] += outlier_noise

        eta_results = {
            "contamination_fraction": eta,
            "outlier_count": n_outliers,
            "arms": {},
        }

        for method_id, method_desc in methods:
            # Initialize fresh student
            student = LowRankStudent(d_in, d_out, rank=rank, alpha=alpha).to(device)
            # Re-init B with small noise so gradients flow smoothly
            with torch.no_grad():
                student.lora_B.copy_(torch.randn_like(student.lora_B) * 0.01)

            optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)

            peak_grad_norm = 0.0
            t_start = time.perf_counter()

            for step in range(steps):
                optimizer.zero_grad()
                pred = student(X_train)

                if method_id == "L2_MSE":
                    loss = l2_loss_fn(pred, Y_train_noisy)
                elif method_id == "Huber_Delta1.0":
                    loss = huber_loss_fn(pred, Y_train_noisy)
                elif method_id == "LAD_LASSO":
                    loss = lad_lasso_fn(pred, Y_train_noisy, model_parameters=[student.lora_A, student.lora_B])

                loss.backward()

                # Measure unclipped gradient norm to assess breakdown vulnerability
                total_grad_norm = 0.0
                for p in student.parameters():
                    if p.grad is not None:
                        total_grad_norm += p.grad.data.norm(2).item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                if total_grad_norm > peak_grad_norm:
                    peak_grad_norm = total_grad_norm

                optimizer.step()

            fit_ms = (time.perf_counter() - t_start) * 1000

            # Evaluate recovered weights against pristine ground truth W*
            W_recovered = student.get_effective_delta_w()
            rel_param_err = float(torch.norm(W_recovered - W_star, p="fro").item() / norm_W_star)

            # Evaluate clean test set generalization MSE
            with torch.no_grad():
                test_pred = student(X_test)
                clean_test_mse = float(l2_loss_fn(test_pred, Y_test_clean).item())

            # Breakdown defined as relative error > 100% (estimator exploded)
            breakdown_triggered = rel_param_err > 1.0

            eta_results["arms"][method_id] = {
                "name": method_desc,
                "relative_parameter_error": round(rel_param_err, 4),
                "clean_test_mse": round(clean_test_mse, 6),
                "peak_gradient_norm": round(peak_grad_norm, 2),
                "breakdown_triggered": breakdown_triggered,
                "fit_ms": round(fit_ms, 2),
            }

            status = "❌ BREAKDOWN" if breakdown_triggered else "✅ STABLE"
            print(f"  [{method_id:15s}] Rel Error: {rel_param_err*100:6.2f}% | Test MSE: {clean_test_mse:8.4f} | Peak Grad: {peak_grad_norm:8.1f} | {status}")

        benchmark_results["experiments"].append(eta_results)

    output_path = RESULTS_DIR / "synthetic_contamination_breakdown.json"
    with open(output_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\n[+] Telemetry successfully exported to: {output_path}")
    return benchmark_results


if __name__ == "__main__":
    run_synthetic_contamination_benchmark()
