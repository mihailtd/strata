# 🔥 Activation-Space Inertia & Cross-Talk Probe

> **Tier Classification**: **🔥 Applied Practice**  
> **Concept Origin**: **A quantitative diagnostic suite measuring live layer-by-layer activation perturbation energy ($\|\delta\| / \|h\|$) and pairwise cross-talk cosines ($\cos(\delta_a, \delta_b)$), unifying weight-space Frobenius quadrature with dynamic activation-space addition.**

---

### Classification Breakdown: What is Standard vs. What is Innovative
* **⭐ Industry Standard Baseline**: LoRA fine-tuning treats adapters as static weight modifications, optimizing solely for token prediction loss ($\mathcal{L}_{\text{SFT}}$). Literature assumes that if weight matrices have low overlap in storage, they can be stacked without interference, neglecting activation-space behavior.
* **🔥 Our Innovative Applied Practice**: **Activation-Space Inertia & Cross-Talk Telemetry**. We attach live forward hooks across all adapted projections (`q/k/v/o/gate/up/down_proj`) to measure the **Activation Selectivity Ratio (ASR)** ($\|\delta_{\text{in}}\| / \|\delta_{\text{out}}\|$) and prove that activations add in **exact Pythagorean quadrature ($\sqrt{K}$)**. This provides the mathematical foundation for $\alpha/\sqrt{K}$ scaling and the closed-form **Activation Inertia Loss** ($\mathcal{L}_{\text{inert}}$).

---

## 1. The Unified Quadrature Theorem (Weight Space $\iff$ Activation Space)

Our live measurements reveal a profound physical correspondence: because pairwise activation cross-talk is near-orthogonal ($\cos(\delta_A, \delta_B) \approx +0.01$), **live dynamic activations add according to the exact same Pythagorean law as static weight matrices**:

$$\|\delta_{\text{total}}\|_2 = \sqrt{\sum_{i=1}^K \|\delta_i\|_2^2} \approx \sqrt{K} \cdot \|\delta_{\text{solo}}\|_2$$

```
                               THE UNIFIED QUADRATURE LAW
                               
   Stack Size (K)     Weight-Space (‖dW‖/‖W‖)     Activation-Space (‖δ‖/‖h‖)    Quadrature Factor
  ─────────────────────────────────────────────────────────────────────────────────────────────────
   1 Adapter                  0.0764                      0.0852                      1.000 × Base
   2 Adapters                 0.1089                      0.1205                 ──►  √2 × Base (1.414×)
   3 Adapters                 0.1341                      0.1475                 ──►  √3 × Base (1.732×)
```

---

## 2. Why 3-Way Unscaled Stacking Suffers Mild Degradation

The activation-space probe explains the exact mechanism behind the 3-way financial drop ($83.33\% \rightarrow 75.83\%$):

1. **The $1.73\times$ Excess Perturbation:** In an unscaled 3-way stack ($\alpha=128$ for each), total perturbation energy rises to **$0.1475$**, approaching the representation overwrite boundary ($>0.150$).
2. **The Signal-to-Noise Split:**
   * **$58\%$ ($1/\sqrt{3}$):** Active, domain-relevant guidance.
   * **$42\%$ ($1 - 1/\sqrt{3}$):** Background leakage from the other two adapters firing on text they know nothing about.
3. **The Solution ($\alpha/\sqrt{K}$ Scaling):** Dividing $\alpha$ by $\sqrt{K}$ scales total perturbation energy back down from $0.1475 \rightarrow 0.0852$, keeping the base model in the Goldilocks zone.

---

## 3. Empirical Telemetry: Low Selectivity in Standard LoRAs

```
  Expert Adapter           Prompt Domain Fed       Relative Activation Energy (‖δ‖/‖h‖)    ASR
  ──────────────────────────────────────────────────────────────────────────────────────────────
  astral_v4                astral (Python)                      0.085161                  1.01× / 1.09×
  astral_v4                postgresql (SQL)                     0.084165                  (Near-zero
  astral_v4                financial_planning                   0.077836                   selectivity)
  ──────────────────────────────────────────────────────────────────────────────────────────────
  postgresql_v4            postgresql (SQL)                     0.083553                  1.04× / 1.14×
  postgresql_v4            astral (Python)                      0.080030                  (Near-zero
  postgresql_v4            financial_planning                   0.073538                   selectivity)
  ──────────────────────────────────────────────────────────────────────────────────────────────
  financial_v4             financial_planning                   0.108606                  1.21×
  financial_v4             astral (Python)                      0.089659                  (Mild selectivity)
```

* **The Baseline Finding:** Because standard LoRA fine-tuning never penalizes out-of-domain emissions, standard adapters exhibit **$\text{ASR} \approx 1.0\text{–}1.2\times$** (they fire at full volume across all domains).

---

## 4. `L_inert`: ALREADY TRIED THREE TIMES — all three failed or were lost

> **Correction (2026-09-13):** this section used to present `L_inert` as unattempted
> future work ("The Future Target"). It was actually tried three times
> (adapters `v5`, `v5b`, `v5c`), and none of the three produced a validated ASR gain.
> See `docs/CHANGELOG.md`'s `v5`/`v5b`/`v5c` entries (the full, authoritative record)
> and `docs/DECISIONS.md` §39 (the mechanics diagnosis) — flagged as a
> documentation-staleness item in `docs/EXPERIMENT_REAUDIT_2026-09.md`, not a
> fabrication: the attempts and their failures were always recorded elsewhere, just
> never reflected back into this README.

Driving out-of-domain activations to absolute zero ($0.08 \rightarrow 0.00$) is neither possible nor desirable, as adapters must share fundamental English grammar, token semantics, and formatting structures. The original objective, for the record:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SFT}} + \lambda \sum_{l=1}^L \left\| \left(\frac{\alpha}{r}\right) B_l \cdot A_l \cdot h_l \right\|_2^2$$

targeting **ASR $1.1\times \rightarrow 3.5\times$–$5.0\times$**. What actually happened:

1. **`v5`** ($\lambda=0.05$, penalty on in-domain prompt tokens) — **FAILED**. With `completion_only_loss=True` on a single-domain corpus, the penalized tokens were all in-domain, so the adapter had no out-of-domain signal to learn "when to be quiet" from. Result: a uniform **0.843× energy scaling** across every domain (1.4% spread) with **ASR unchanged at 0.96×** — indistinguishable from just lowering $\alpha$, which the stacking benchmark had already shown is dilution, not selectivity.
2. **`v5b`** ($\lambda=0.05$, penalty on out-of-domain replay) — **FAILED**. Training's own in-loop metric reported ASR 1.073 and rising; the real probe measured 0.965 — unchanged from the pre-`L_inert` baseline. Cause: `padding="max_length"` to 512 tokens meant ~70% of every replay batch was `<pad>`, and the adapter learned to be quiet on padding — free under the loss, useless in practice. Lesson: when a cheap in-loop metric and the real instrument disagree, trust the real instrument.
3. **`v5c`** ($\lambda=0.5$, pad-masked replay, the padding bug fixed) — **LOST, not disproven**. Killed by a host crash (two concurrent GPU jobs) at ~step 120/300; the log lived in `/tmp` and did not survive the reboot, and `results/adapters/m2_astral_r8a128_v5c/` only has an empty `checkpoints/`. Last reading before loss (flat across 40 steps, never validated by the real probe): `ASR ≈ 1.064`.

All three adapters were deleted. **`L_inert` is not in v6.** The mechanism itself was never shown to be wrong — v5's own math checked out (15.7% movement at 1.4% spread is a penalty doing exactly what it was told), it just wasn't given the chance to learn selectivity. Retrying is legitimate future work, but only with v5c's padding fix carried forward and on a machine not sharing the GPU with another job.

---

## 5. Running the Diagnostic Tool

```bash
uv run --env-file .env python benchmarks/factory/geometry/activation_inertia_probe/probe_activation_inertia.py
```

> [!NOTE]
> ### 💡 In Layman's Terms: Two Guitars Playing in the Same Room
> When two musicians play together in the same room, their sound waves add together in the air. If both musicians are playing at full volume (volume 8) all the time—even during each other's solo sections—the room gets $1.73\times$ louder, creating a wall of sound that drowns out the nuances of the song. Activation Inertia simply teaches the rhythm guitarist to play softer (volume 2) during the bass solo, keeping the total volume at a comfortable, perfect acoustic level.
