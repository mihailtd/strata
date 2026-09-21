# Rule: LoRA Dynamic $\alpha$ Scaling, Geometric Early Stopping & Calibration Invariants

## Core Principles
The representation manifold of smaller models ($d_{model} \le 2048$) is significantly more vulnerable to displacement than large models ($d_{model} \ge 4096$). All LoRA adapter training, post-train calibration, runtime weight-folding, and benchmark evaluations MUST strictly adhere to the physical perturbation laws established in this repository.

---

## 1. Dimension-Proportional Dynamic $\alpha$ Scaling Law
* **Physical Law**:
  $$\text{Perturbation Magnitude} = \frac{\|\Delta W\|}{\|W\|} = \frac{\alpha}{r} \frac{\|B \cdot A\|}{\|W\|}$$
  Because $\|W\|_F \propto d_{model}$, applying a static $\alpha=128$ at $r=8$ (nominal scaling $16.0$) to small models causes catastrophic representation collapse ($\frac{\|\Delta W\|}{\|W\|} \ge 0.100$).
* **Mandatory Standard**:
  All training scripts and configs MUST compute initial $\alpha$ proportionally to base model hidden dimension $d_{model}$:
  $$\alpha^*(d_{model}) = \max\left(16, \text{round}\left(128 \times \frac{d_{model}}{4096} / 16\right) \times 16\right)$$
  - **Qwen 3.5 0.8B** ($d=1024$): $\alpha = 32$ (nominal scale $4.0$)
  - **Qwen 3.5 2B** ($d=2048$): $\alpha = 64$ (nominal scale $8.0$)
  - **Qwen 3.5 4B** ($d=2560$): $\alpha = 80$ (nominal scale $10.0$)
  - **Qwen 3.5 9B** ($d=4096$): $\alpha = 128$ (nominal scale $16.0$)
  - **Qwen 3.5 27B** ($d=5120$): $\alpha = 160$ (nominal scale $20.0$)
* **Rank Invariant**:
  Rank $r=8$ is mathematically sufficient across all tiers. Never increase or decrease rank to address over-steering; rank governs update *subspace rank*, while $\frac{\alpha}{r}$ governs *amplitude*.

---

## 2. Mandatory Geometric Early Stopping (The Goldilocks Band)
* **The Danger of Fixed Steps**:
  Fixed step counts (e.g. 150 steps) result in unpredictable perturbation depths across different corpus sizes, domains, and model capacities.
* **Mandatory Standard**:
  1. All training runs MUST activate `GoldilocksStoppingCallback` by default with `--stop-at-dw-over-w 0.065`.
  2. Training MUST terminate when Frobenius ratio $\frac{\|\Delta W\|}{\|W\|}$ reaches the target ($0.065 \pm 0.015$), comfortably within the Goldilocks band $[0.035, 0.100]$.
  3. Never disable geometric early stopping in production or fleet training runs.

---

## 3. Mandatory Automated Post-Training Calibration
* **Zero-Uncalibrated-Evaluation Invariant**:
  No evaluation harness, scorecard generator, or agent benchmark may evaluate an adapter directly with raw training defaults without post-training calibration.
* **Mandatory Pipeline**:
  1. Immediately following weight saving, training orchestrators MUST call `calibrate_adapter_alpha(out_dir, base_model=model, apply=True)`.
  2. The calibrator verifies IEEE 754 precision bounds ($\text{merge\_rel\_err} \le 5.0\%$) and selects the optimal deployment $\alpha_{\text{opt}}$ within the target perturbation window $[0.040, 0.085]$.
  3. The chosen $\alpha_{\text{opt}}$ MUST be written directly into `adapter_config.json` before releasing the artifact for inference or benchmarking.
  4. Offline calibrators MUST use tier-specific baseline Frobenius projection norms ($0.8\text{B}: 205.56, 2\text{B}: 323.40, 4\text{B}: 501.78, 9\text{B}: 882.05, 27\text{B}: 1419.00$) rather than defaulting to 4B norms.

---

## 4. Serving-Time Fractional Scale Attenuation (`adapter@scale`)
* **Zero-Retraining Scaling**:
  All inference engines (`runtime-next`, `runtime-triton`, `runtime-ipwf`) MUST support dynamic scale attenuation at request time (e.g., `adapter_name@0.25` or `adapter_scale: 0.50`), computing effective low-rank delta as:
  $$W_{\text{active}} \leftarrow W_0 + \left(\text{scale} \cdot \frac{\alpha}{r}\right) (B \cdot A)$$
* **Pristine Weight Restore Invariant**:
  When altering the scale factor of an already resident adapter in weight-folding engines, the runtime MUST fully restore pristine weights $W_0$ before re-folding with the new scale factor, preventing double-accumulation ($W_0 + s_1 dW + s_2 dW$).
