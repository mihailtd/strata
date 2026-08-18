# 🔥 Per-Adapter Dynamic $\alpha$-Calibration & ⭐ Stock LoRA ($r=8$, Dynamic $\alpha$)

> **Tier Classification**: **🔥 Applied Practice**  
> **Concept Origin**: **Decoupling training rank from runtime deployment scaling via post-train matrix factor linearity, bounded by the `bfloat16` precision floor and held-out capability retention.**

---

### Classification Breakdown: What is Standard vs. What is Innovative
* **⭐ Industry Standard Baseline**: Standard LoRA fine-tuning treats $\alpha$ as a static, fixed hyperparameter chosen before training (e.g. hardcoding $\alpha=16$ or $\frac{\alpha}{r} = 2.0$), assuming all tasks and adapters behave uniformly.
* **🔥 Our Innovative Applied Practice**: **Per-Adapter Dynamic $\alpha$-Calibration**. Because frozen low-rank factor updates are strictly linear ($\Delta W = \frac{\alpha}{r} (B \times A) \propto \alpha$), deployment $\alpha$ is a zero-cost runtime knob. By measuring $\|\Delta W\| / \|W\|$ once post-training, the lower bound is derived analytically in zero GPU time via the Mantissa Inverse Law ($merge\_rel\_err \approx \frac{0.167}{\|dW\|/\|W\|}$), and the upper bound is determined by evaluating the trained adapter across the held-out capability gate without retraining.

---

## 1. Why Fixed $\alpha$ Fails: The Physical Perturbation Axis

In standard LoRA, setting $\alpha=128$ at $r=8$ (nominal scaling $16$) or $\alpha=32$ at $r=64$ (nominal scaling $0.5$) means nothing by itself. The hardware floating-point mantissa and the base model's representation manifold respond exclusively to the **physical perturbation magnitude**:

$$\text{Perturbation Magnitude} = \frac{\|\Delta W\|}{\|W\|} = \frac{\alpha}{r} \frac{\|B \times A\|}{\|W\|}$$

Because $\|B \times A\|$ is learned by SGD during training, it varies across domains, step counts, and corpora. A fixed $\alpha$ in a training config applies one static number to a variable that is only knowable **after** training.

---

## 2. The Two Bounding Equations (The Goldilocks Zone)

```
                      THE ADMISSIBLE OPERATING WINDOW
                      
   0 -----------------[ α_min  ======  α_opt  ======  α_max ]------------------> α
       UNDERFLOW ZONE                                          OVERWRITE ZONE
    (Mantissa Truncation)                                   (Catastrophic Forgetting)
    merge_rel_err > 5%                                      Held-out regression < -0.01
```

### 1. Lower Bound: Analytic Precision Floor ($\alpha_{\min}$)
* **Governed by**: IEEE 754 `bfloat16` mantissa ULP resolution.
* **Formula**:
  $$\alpha_{\min} \ge \frac{r \cdot 0.167}{\left( \frac{\|\Delta W_{\text{ref}}\|}{\|W\|} \cdot \frac{r}{\alpha_{\text{ref}}} \right) \cdot \text{Err}_{\text{target}}}$$
* For our $r=8$ PostgreSQL v3 adapter ($\|\Delta W\|/\|W\| = 0.0750$ at $\alpha=128$), setting a $5.0\%$ merge error tolerance yields $\alpha_{\min} = 64$.

### 2. Upper Bound: Empirical Narrowing Ceiling ($\alpha_{\max}$)
* **Governed by**: Out-of-domain / held-out task retention.
* **Mechanism**: When $\frac{\alpha}{r}$ is too high, the adapter overrides base activations on constructs not present in the training set (e.g. `__slots__`, `functools.partial`).
* **Evaluation**: Rewriting `lora_alpha` in `adapter_config.json` allows evaluating $\alpha \in [32, 64, 96, 128]$ over the held-out benchmark in **1 train + $N$ quick evaluations**, instead of $N$ expensive retraining runs.

---

## 3. Calibration Workflow & Tooling

To calibrate a newly trained adapter:

```bash
uv run --env-file .env python benchmarks/factory/geometry/dynamic_alpha_calibration/calibrate_expert_alpha.py \
    --adapter results/adapters/m2_postgresql_r8a128_v3 \
    --alphas 32 64 96 128 \
    --apply
```

### Output Telemetry:
```
==========================================================================
   alpha   scaling   merge_err%   held-out   admissible
      32       4.0        8.91%     0.2667           no (below floor)
      64       8.0        4.46%     0.4667          yes (optimal)
      96      12.0        2.97%     0.4000          yes
     128      16.0        2.23%     0.3444          yes (narrowed)

  alpha_opt = 64  (held-out 0.4667, merge_err 4.46%)
  APPLIED lora_alpha=64 to results/adapters/m2_postgresql_r8a128_v3/adapter_config.json
```
