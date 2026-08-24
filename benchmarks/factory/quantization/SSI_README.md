# 📐 Stress–Strength Interference (SSI) Optimal Quantization Scaling (Chapter 8.5)

> **Implementation**: [`src/factory/ssi_quantization.py`](../../../src/factory/ssi_quantization.py) & [`src/factory/leverage_quantization.py`](../../../src/factory/leverage_quantization.py)  
> **Theoretical Grounding**: Chapter 8.5 (*Case Studies – Stress-Strength Interference Analysis*, Jaejin Hwang, *Reliability Analysis Using MINITAB and Python*)  
> **Telemetry Artifact**: [`results/benchmarks/ssi_quantization_benchmark.json`](../../../results/benchmarks/ssi_quantization_benchmark.json)

---

## 🎯 Executive Summary & Innovation

In low-bit weight quantization (e.g. symmetric 4-bit INT4 with 16 discrete levels $[-7, +7]$), setting the group scaling factor $\gamma$ creates an inevitable tradeoff:
* **Too Large $\gamma$**: No clipping occurs, but coarse bin spacing inflates rounding noise: $D_{\text{round}} \approx \frac{\gamma^2}{12}$.
* **Too Small $\gamma$**: Fine bin spacing reduces rounding error, but tail weights exceed $c = 7\gamma$, causing severe clipping distortion: $D_{\text{clip}} = \int_{7\gamma}^{\infty} (x - 7\gamma)^2 f(x) dx$.

Industry standard tools (e.g., naive min-max scaling) set $\gamma = \frac{\max(|W|)}{7}$, letting a single extreme tail value degrade the precision of all 128 weights in the group. Heuristic percentile clipping (e.g. 99.9%) requires expensive sorting and lacks statistical grounding.

**Chapter 8.5 Stress-Strength Interference (SSI)** models weight distributions as Stress $f(x)$ and hardware representation limits as Strength $g(y)$, deriving the **closed-form optimal clipping boundary**:
$$\gamma^* = \frac{\min\left( (2.2 + 0.45 \sqrt{\kappa}) \cdot \sigma_W, \max(|W|) \right)}{7}$$
where $\sigma_W$ is group standard deviation and $\kappa$ is standardized kurtosis.

This achieves **$+0.79\text{ dB}$ higher Signal-to-Noise Ratio** in **$0.18\text{--}0.25\text{ ms}$ per layer** on GPU with zero iterative search.

---

## 📊 Measured Benchmark Telemetry (AMD Radeon RX 7900 XTX)

Group Size: $G=128$ | Precision: INT4 Symmetric $[-7, +7]$ | Calibration Window: $N=256$ tokens

```
┌───────────────────┬──────────────┬──────────────────┬──────────────────────┬────────────────────┬───────────┐
│ Layer Shape       │ Dimension D  │ Arm A: Naive Max │ Arm B: 99.9% Percent │ Arm C: SSI Optimal │ SNR Gain  │
├───────────────────┼──────────────┼──────────────────┼──────────────────────┼────────────────────┼───────────┤
│ Qwen 4B Base      │ D = 2,560    │ 15.90 dB (0.37ms)│ 16.09 dB (97.72ms)   │ 16.67 dB (0.25ms)  │  +0.78 dB │
│ Qwen 9B Base      │ D = 4,096    │ 15.89 dB (0.27ms)│ 16.09 dB (0.27ms)    │ 16.67 dB (0.18ms)  │  +0.79 dB │
│ LLaMA / Qwen 70B  │ D = 8,192    │ 15.90 dB (0.25ms)│ 16.10 dB (0.33ms)    │ 16.69 dB (0.20ms)  │  +0.79 dB │
└───────────────────┴──────────────┴──────────────────┴──────────────────────┴────────────────────┴───────────┘
```

### 🔬 Key Findings:
1. **Direct Signal-to-Noise Ratio Gain**: SSI closed-form scaling consistently provides **$+0.78\text{--}0.79\text{ dB}$ higher SNR** over naive max scaling.
2. **Sub-Millisecond Execution**: SSI computes moments (variance, kurtosis) in a single vectorized GPU pass, executing in **$0.18\text{--}0.25\text{ ms}$ per layer** (faster than quantile sort in Arm B).
3. **Synergy with Chapter 21 Leverage Scoring**: When paired with protected $\text{bfloat16}$ outlier channels, overall output SNR reaches **$48.0\text{--}50.4\text{ dB}$** across full transformer layers.

---

## 🔬 Mathematical Formulation

Total Quantization Distortion:
$$D(\gamma) = D_{\text{rounding}}(\gamma) + D_{\text{clipping}}(\gamma) = \frac{\gamma^2}{12} \cdot P(|X| \le 7\gamma) + \int_{7\gamma}^{\infty} (x - 7\gamma)^2 f(x) \, dx$$

Solving $\frac{\partial D(\gamma)}{\partial \gamma} = 0$:
$$c^* = 7\gamma^* = k_{\text{SSI}}(\kappa) \cdot \sigma_X$$
$$k_{\text{SSI}}(\kappa) = 2.2 + 0.45 \sqrt{\kappa}$$

```
For Gaussian distribution (κ = 3.0):  k_SSI ≈ 2.98 · σ
For Laplace distribution (κ = 6.0):   k_SSI ≈ 3.30 · σ
For Heavy-tail / Student-t (κ = 12.0): k_SSI ≈ 3.76 · σ
```

---

## 🛠️ Usage Example

```python
from factory.ssi_quantization import StressStrengthInterferenceCalibrator

# 1. Initialize SSI Calibrator
calibrator = StressStrengthInterferenceCalibrator(group_size=128, n_bits=4)

# 2. Compute closed-form optimal scales and quantize W
W_q, scales = calibrator.quantize_w4(weight_tensor, mode="closed_form_ssi")

# 3. Dequantize for inference / verification
W_rec = calibrator.dequantize_w4(W_q, scales, original_shape=(D_in, D_out))
```
