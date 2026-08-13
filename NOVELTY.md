# NOVELTY

Honest tiering of what this repo contributes. Companion to `CURRENT.md` (what is
live) and `SYSTEM.md` (hardware/software).

**Tag by what was invented, not by how hard it was to build or how large the
number is.** Difficult engineering on a known technique is ⭐ or 🔥. A big
speedup does not raise the tier — it means the baseline was worse than people
assumed, which is worth publishing, tagged accurately.

Overclaiming novelty fails the same way overclaiming a benchmark does.

| tag | meaning |
| :--- | :--- |
| 🚀 **Genuine Discovery** | New knowledge, not derivable from existing theory before running the experiment |
| 🔥 **Applied Practice** | Established concept adapted to an unsolved problem in this stack |
| ⭐ **Industry Standard** | Standard pattern, used correctly |
| ❌ **Closed** | Falsified. **Not a contribution at any tier** — the *measurement* may be, the hypothesis is not |

---

## 🚀 Genuine Discovery

Deliberately short. Most of this repo is excellent engineering on known
techniques, and that is not a criticism.

| finding | why it qualifies | evidence |
| :--- | :--- | :--- |
| **Chunked and single-token kernels disagree numerically on this stack** | Not predictable from docs or theory. Makes token-exactness against plain greedy *unreachable* for any multi-token verification — a hard constraint on what speculative decoding can promise here, not a bug to fix. | Divergence at tokens 2–4 of 32 on 3/3 prompts with zero speculation; spec-vs-greedy 80.0% against a *non-speculative* control of 83.3%, determinism 100%, n=60 |
| **A native checkpoint MTP head degrades under domain fine-tuning** | Draft-model-matches-target is known in the abstract; that a *shipped* MTP head loses acceptance monotonically in adapter strength under standard next-token loss is a specific, non-obvious empirical result with a clear mechanism (a draft head must *agree with* its backbone, not be fluent). | τ 2.456 → 1.531–1.988 across 5 adapters and 4 scalings, every CI excluding zero, n=160/condition |

---

## 🔥 Applied Practice

| item | what was adapted | evidence / caveat |
| :--- | :--- | :--- |
| **`activate_many()`** — multi-expert additive folding with restore | Additive delta composition applied to runtime expert stacking. `peft.merge_and_unload` does not do multi-adapter additive folding *with* restore. Fold cost stays flat in N. | Mechanism verified (Pythagorean norm to 4 s.f., additive to 7.3e-3 in bf16). **Tag the capability, not a benefit** — stacking showed resolved interference (`ast+fin` −10.96pp). |
| **Pristine-buffer restore via bit-exact `copy_`** | Standard backup-before-mutate, but the *reason* is non-obvious: subtract-the-delta accumulates bf16 drift across swaps and silently corrupts a long-running expert-swapping server. | `max_drift()` verification; the buffer alone is ⭐, the discipline is 🔥 |
| **bf16 merge-absorption calibration** | The 1/x *form* is elementary floating point (rounding perturbs by ≈ε·‖W‖ regardless of dW, so relative to ‖dW‖ it is ε/(‖dW‖/‖W‖)) — derivable in one line before running anything. What is ours is the measured constant and its validation. | Product constant to **2.3% across a 16× range**; fitted k = 0.22 × bf16 eps, within ~1.5× of the uniform-rounding prediction. **Not** a discovered law. |
| **`fla` (Triton) unblocking speculation on gfx1100** | Known library, undocumented platform. The `fla` / `causal_conv1d` conflation blocked this for a long time — they are two deps with independent fallbacks. | 2.84× penalty → 1.17–1.40×; nobody documents ROCm speculative decoding on consumer AMD |
| **Loading the shipped `mtp.*` tensors** | The checkpoint ships a 15-tensor MTP head that `transformers` never loads. Reading it directly, with the correct `[embedding; hidden]` fuse order, makes speculation possible with no draft model. | Wrong concat order gave 0% accuracy; correct order 70% |
| **Measurement discipline** | Re-measure baselines in-process; paired bootstrap CIs with the decision rule fixed *in advance*; adapter content-hashing (`ADAPTER_MANIFEST.json`). | Caught four separate false results in one day. Arguably the repo's most transferable output. |

---

## ⭐ Industry Standard

Used correctly, but off-the-shelf. Large measured wins here reflect weak
baselines, not novelty.

| item | note |
| :--- | :--- |
| **In-place weight folding** (backbone) | `peft` ships `merge_and_unload`. The **1.83×** is real but it measures peft's *wrapper* costing 1.87× bare — folded decode is 29.2 ms against a 28.7 ms bare model, i.e. folding itself is free. |
| Batching | **~6–7×** at B=8, the largest lever measured. Standard practice. |
| Liger fused kernels | Third-party library, applied correctly (patch before `from_pretrained`; `rope=False` is a blanket opt-out, **not** hybrid-architecture detection). |
| CUDA graph replay, FastAPI REST, LoRA/DoRA/LoKr/id_kron | Standard. id_kron ≈ stock LoRA at matched scaling. |

---

## ❌ Closed — not a contribution at any tier

| hypothesis | why it died |
| :--- | :--- |
| **Adapting/folding the MTP head** | Falsified — see 🚀 above. The *measurement* is a contribution; the hypothesis is not. Retrying needs a distillation objective, not tuning. |
| **Folding the MTP head for latency** | Nothing is wrapped (bare checkpoint tensors, no peft layer), and drafting is ~27% of a round → Amdahl ceiling ~1.14×. |
| **Hybrid Radix caching as a priority** | Prefill is 5.9% at 8k, not 54%. The original figure was ~33 s of Triton JIT inside the timed region. |
| **Monotonic stacking decay / N≤2 rule** | Not monotonic — a third expert moved astral back *above* solo. One resolved cell cannot support a stack-size rule. |
| **In-domain speculation penalty** | Artifact of id_kron adapters mislabelled as stock LoRA. All 9 cells win; router is all-True. |
| **Stacking retention >100%** | Degenerate statistic — financial's solo gain is 0.83 question-equivalents at n=20. |

---

## Rule of thumb

Ask: *could someone have written this down before running the experiment?*

- **Yes, from theory** → ⭐ (or 🔥 if adapted to a genuinely new problem)
- **No, and it changes what we believe** → 🚀
- **No, and it changes what we believe *is false*** → ❌ for the hypothesis, 🔥 for the measurement
