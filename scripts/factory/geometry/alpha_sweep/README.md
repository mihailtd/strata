# 🔥 Hyperparameter Scale Calibration ($\alpha$ Sweep)

This module is responsible for analyzing how LoRA adapter weights behave when they are mathematically merged (folded) directly into the base model's 16-bit (`bfloat16`) native weights.

## How it Works
When you fold an adapter, you add its weights ($dW$) directly to the base weights ($W$). Because standard LoRA adapters have very small magnitudes ($|dW| \ll |W|$), the hardware floating-point math often rounds the addition to zero (truncation). The model effectively "forgets" the adapter during folding.

This script (`benchmark_alpha_absorption_sweep.py`) solves this by:
1. **Sweeping Scales:** It takes adapters trained identically but with different $\alpha$ hyperparameter values (e.g., 16, 32, 64, 128, 256).
2. **Measuring Truncation:** For each scale, it calculates the mathematical truncation error (`|dW|/|W|`).
3. **Evaluating Degradation:** It runs a real-world task evaluation (like answering PostgreSQL questions) to measure if the theoretical truncation error translates to a real drop in AI intelligence.

## ⚠️ Workflow Integration: When to Run This?

> [!IMPORTANT]
> This is a **Foundational R&D Step**, not a daily training task. Do not run this before training every new domain!

### When you MUST run this script:
You must run this calibration sweep **before** setting up a new factory pipeline. Specifically, run it if you:
1. Switch to a new base model (e.g., upgrading from Qwen 4B to 8B).
2. Change the native precision type (e.g., migrating from `bfloat16` down to `fp8`).
3. Change the target LoRA rank (e.g., moving from $r=8$ to $r=32$).

In these scenarios, the floating-point truncation math completely changes, and you must use this sweep to scientifically prove what the new optimal $\alpha$ multiplier should be to prevent precision loss.

### When you SKIP this script:
For day-to-day domain training (e.g., training a new Financial or Astral adapter), **do not run this script**. Instead, you take the optimal $\alpha$ value previously discovered by the sweep (currently $\alpha=128$ for our Qwen 4B / $r=8$ architecture) and **hardcode** it into your standard unified training script (`CURRENT_m2`). 

The factory pipeline blindly trusts the sweep's conclusion, allowing you to train daily adapters rapidly without stopping to recalibrate!
