# ⭐ Architecture Comparison: Stock LoRA vs. `id_kron` (Controlled Head-to-Head)

> **Tier Classification**: **⭐ Industry Standard** (Controlled Architecture Evaluation) / **🔥 Applied Practice** (Matched Effective Scaling Discipline)  
> **Concept Origin**: **Controlled micro-architecture parameter sweeps to isolate intrinsic adapter capacity from scaling artifacts.**

---

### Classification Breakdown: What is Standard vs. What is Innovative
* **⭐ Industry Standard Baseline**: Running controlled A/B evaluation across fine-tuning architectures (Standard LoRA vs Kronecker-product `id_kron`).
* **🔥 Our Applied Practice Discipline**: Enforcing **Matched Effective Scaling ($\frac{\alpha}{r_{\text{total}}}$)**. An earlier retracted audit falsely reported `id_kron` at $r=64$ (49.8M parameters) when the loaded adapter was actually $r_{\text{total}}=8$ (6.26M parameters) and compared mismatched scalings ($32.0$ vs $2.0$). We rectified this by sweeping both architectures across identical scalings ($\frac{\alpha}{r} \in [0.25, 0.5, 1.0, 2.0, 8.0, 16.0, 32.0]$) at matched steps, learning rates, and dataset splits.

---

## 1. Controlled Parameters

| Parameter | Value |
| :--- | :--- |
| **Model Backbone** | `Qwen/Qwen3.5-4B` in `bfloat16` |
| **Training Steps & LR** | 150 steps, lr = 2e-4, batch 2, grad-accum 2 |
| **Dataset & Rubric** | Astral dataset (815 records), identical eval questions & rubric |
| **Architectures Compared** | **Stock LoRA** ($r=8$, 10.62M params) vs. **`id_kron`** ($r_{\text{in}}=8, r_{\text{out}}=1$, 6.26M params) vs. **`id_kron`** ($r_{\text{in}}=16, r_{\text{out}}=1$, 12.42M params) |

---

## 2. Empirical Findings: Stability & Loss Across Scaling Range

| Effective Scaling ($\alpha / r$) | Stock LoRA Loss ($r=8$) | `id_kron` Loss ($r_{\text{total}}=8$) | `id_kron` Loss ($r_{\text{total}}=16$) | Architectural Behavior |
| :---: | :---: | :---: | :---: | :--- |
| **0.25** | 1.0646 | 1.0357 | 1.0337 | `id_kron` slightly lower training loss |
| **0.50** | 1.0502 | 1.0444 | 1.0574 | Competitive parity |
| **1.00** | 1.0371 | 1.0918 | — | LoRA pulls ahead in stability |
| **2.00** | 1.0274 | 1.2067 | 1.2799 | `id_kron` loss begins to climb |
| **8.00** | 1.0290 | **7.6679 (Diverged ❌)** | **7.2853 (Diverged ❌)** | `id_kron` completely diverges at high scaling |
| **16.00** | 1.0556 | **8.0602 (Diverged ❌)** | **7.0332 (Diverged ❌)** | LoRA remains rock-solid; `id_kron` broken |
| **32.00** | 1.1464 | **7.7288 (Diverged ❌)** | **7.9349 (Diverged ❌)** | Stock LoRA handles extreme 128x scaling range |

* **Architectural Takeaway**: Stock LoRA exhibits exceptional hyperparameter stability across a 128x scaling range ($0.25 \dots 32.0$), while Kronecker-product factorization (`id_kron`) becomes numerically unstable above scaling 2.0 at standard learning rates.

---

## 3. Scripts
- **`eval_controlled_headtohead.py`**: Executes the matched-scaling evaluation across architectures and outputs metrics to `results/controlled_headtohead_astral.json`.
  ```bash
  uv run --env-file .env python benchmarks/factory/architecture_comparison/eval_controlled_headtohead.py
  ```
