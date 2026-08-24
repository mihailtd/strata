"""Empirical Benchmark: Stress-Strength Interference (SSI) Quantization Calibration (Chapter 8.5).

Theoretical Grounding:
Chapter 8.5 (Case Studies – Stress-Strength Interference Analysis, Jaejin Hwang, Reliability Analysis).

Evaluates:
- Arm A: Naive Min-Max Scaling (scale = max(|W|) / 7)
- Arm B: Heuristic 99.9% Percentile Clipping (scale = perc_99.9(|W|) / 7)
- Arm C: Closed-Form Stress-Strength Interference (SSI) Calibration

Across layer dimensions:
- D = 2560 (4B Architecture)
- D = 4096 (9B Architecture)
- D = 8192 (70B Architecture)
"""

import os
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F

from factory.ssi_quantization import StressStrengthInterferenceCalibrator

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_arm_naive_max(
    calibrator: StressStrengthInterferenceCalibrator,
    W: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    t0 = time.perf_counter()
    W_q, scales = calibrator.quantize_w4(W, mode="naive_max")
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return W_q, scales, dt_ms


def run_arm_heuristic_clipping(
    calibrator: StressStrengthInterferenceCalibrator,
    W: torch.Tensor,
    percentile: float = 0.999,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    t0 = time.perf_counter()
    D_in, D_out = W.shape
    pad_in = (calibrator.group_size - (D_in % calibrator.group_size)) % calibrator.group_size
    if pad_in > 0:
        W_padded = F.pad(W.to(torch.float32), (0, 0, 0, pad_in))
    else:
        W_padded = W.to(torch.float32)

    n_groups = W_padded.shape[0] // calibrator.group_size
    grouped = W_padded.view(n_groups, calibrator.group_size, D_out)

    # Compute empirical percentile per group
    q_val = torch.quantile(torch.abs(grouped), q=percentile, dim=1).clamp(min=1e-6)
    scales = (q_val / 7.0).to(torch.bfloat16)

    scales_exp = scales.unsqueeze(1).to(torch.float32)
    q_grouped = torch.clamp(torch.round(grouped / scales_exp), -8.0, 7.0)
    W_q = q_grouped.view(-1, D_out)[:D_in, :].to(torch.int8)

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return W_q, scales, dt_ms


def run_arm_ssi_closed_form(
    calibrator: StressStrengthInterferenceCalibrator,
    W: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    t0 = time.perf_counter()
    W_q, scales = calibrator.quantize_w4(W, mode="closed_form_ssi")
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return W_q, scales, dt_ms


def compute_snr_db(true_tensor: torch.Tensor, rec_tensor: torch.Tensor) -> float:
    t_f = true_tensor.to(torch.float32)
    r_f = rec_tensor.to(torch.float32)
    signal = torch.sum(t_f ** 2).item()
    noise = torch.sum((t_f - r_f) ** 2).item()
    if noise <= 1e-12:
        return 99.99
    return 10.0 * torch.log10(torch.tensor(signal / noise)).item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Chapter 8.5 Stress-Strength Interference (SSI) Calibration Benchmark ===")
    print(f"Device: {device} | PyTorch: {torch.__version__}\n")

    torch.manual_seed(42)
    dimensions = [
        ("4B Layer", 2560, 4096),
        ("9B Layer", 4096, 4096),
        ("70B Layer", 8192, 8192),
    ]

    calibrator = StressStrengthInterferenceCalibrator(group_size=128)

    # Warmup GPU
    _W = torch.randn(512, 512, dtype=torch.bfloat16, device=device)
    _ = calibrator.quantize_w4(_W, mode="closed_form_ssi")
    if device.type == "cuda":
        torch.cuda.synchronize()

    results = {}

    for label, D_in, D_out in dimensions:
        print(f"--- Benchmarking {label} (D_in={D_in}, D_out={D_out}) ---")
        N_tokens = 256

        # Simulate realistic weights with Laplace / Student-t tails
        dist = torch.distributions.Laplace(loc=0.0, scale=0.025)
        W = dist.sample((D_in, D_out)).to(dtype=torch.bfloat16, device=device)
        X = torch.randn(N_tokens, D_in, device=device)
        Y_true = X.to(torch.float32) @ W.to(torch.float32)

        # 1. Arm A: Naive Max Scaling
        W_q_a, scales_a, t_a = run_arm_naive_max(calibrator, W)
        W_rec_a = calibrator.dequantize_w4(W_q_a, scales_a, (D_in, D_out))
        Y_rec_a = X.to(torch.float32) @ W_rec_a.to(torch.float32)
        weight_snr_a = compute_snr_db(W, W_rec_a)
        output_snr_a = compute_snr_db(Y_true, Y_rec_a)

        # 2. Arm B: Heuristic 99.9% Percentile
        W_q_b, scales_b, t_b = run_arm_heuristic_clipping(calibrator, W, percentile=0.999)
        W_rec_b = calibrator.dequantize_w4(W_q_b, scales_b, (D_in, D_out))
        Y_rec_b = X.to(torch.float32) @ W_rec_b.to(torch.float32)
        weight_snr_b = compute_snr_db(W, W_rec_b)
        output_snr_b = compute_snr_db(Y_true, Y_rec_b)

        # 3. Arm C: Closed-Form SSI
        W_q_c, scales_c, t_c = run_arm_ssi_closed_form(calibrator, W)
        W_rec_c = calibrator.dequantize_w4(W_q_c, scales_c, (D_in, D_out))
        Y_rec_c = X.to(torch.float32) @ W_rec_c.to(torch.float32)
        weight_snr_c = compute_snr_db(W, W_rec_c)
        output_snr_c = compute_snr_db(Y_true, Y_rec_c)

        gain_snr = output_snr_c - output_snr_a

        print(f"  Arm A (Naive Max):     Weight SNR={weight_snr_a:5.2f} dB | Output SNR={output_snr_a:5.2f} dB | Time={t_a:6.2f} ms")
        print(f"  Arm B (99.9% Perc):    Weight SNR={weight_snr_b:5.2f} dB | Output SNR={output_snr_b:5.2f} dB | Time={t_b:6.2f} ms")
        print(f"  Arm C (SSI Calibration): Weight SNR={weight_snr_c:5.2f} dB | Output SNR={output_snr_c:5.2f} dB | Time={t_c:6.2f} ms | Gain: +{gain_snr:4.2f} dB\n")

        results[label] = {
            "D_in": D_in,
            "D_out": D_out,
            "naive_max": {"weight_snr_db": weight_snr_a, "output_snr_db": output_snr_a, "time_ms": t_a},
            "heuristic_percentile": {"weight_snr_db": weight_snr_b, "output_snr_db": output_snr_b, "time_ms": t_b},
            "ssi_closed_form": {
                "weight_snr_db": weight_snr_c,
                "output_snr_db": output_snr_c,
                "time_ms": t_c,
                "snr_gain_db": gain_snr,
            },
        }

    artifact_path = RESULTS_DIR / "ssi_quantization_benchmark.json"
    with open(artifact_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Telemetry saved to {artifact_path}")


if __name__ == "__main__":
    main()
