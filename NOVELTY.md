# NOVELTY

Honest tiering of what this repo contributes. Companion to `CURRENT.md` (what is
live) and `SYSTEM.md` (hardware/software).

## THE GOVERNING RULE — read before tagging anything

**Novelty of APPLICATION counts. You do not have to invent the concept.**

🔥 Applied Practice is earned by either of:

1. **First application of an established concept to THIS class of system** —
   e.g. borrowing a technique from optimizers, HPC, databases or classical
   numerics and using it for LLM runtime expert swapping, where it has not been
   used before. The concept being old is irrelevant; the *pairing* is the work.
2. **A non-obvious use of a known technique** — something a competent engineer
   would not reach for by default, where the reason it works is the insight.

What does NOT earn a tier:

* Size of the speedup. A big number means the baseline was weak, not that the
  method is novel. Tag the method; report the number separately.
* Difficulty. Hard debugging on a standard technique is still ⭐.
* Re-using a library exactly as its authors intended for the purpose they
  intended (`peft.merge_and_unload` for merging LoRA → ⭐, not 🔥).

**Honesty guard on "first".** "Nobody has done this before" is a claim about the
literature and ecosystem that this repo cannot verify. Write **"first known
application in this stack — not verified against published work"** rather than
"novel". The claim stays defensible and still gets the credit.

Overclaiming novelty fails the same way overclaiming a benchmark does — and this
project has already paid that cost. But *under*-claiming a genuine first
application is also a failure. Both are errors.

| tag | meaning |
| :--- | :--- |
| 🚀 **Genuine Discovery** | New knowledge, not derivable from existing theory before running the experiment |
| 🔥 **Applied Practice** | Established concept applied to this class of system for the first known time, OR used in a non-obvious way. **Inventing it is not required.** |
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
| **Pristine State Buffer + bit-exact `copy_` restore** | Master-weights discipline (keep an unmutated reference copy, never reconstruct by inverse arithmetic) carried from classical mixed-precision optimizers into **runtime LLM expert swapping** — **first known application in this stack -- not verified against published work**. Non-obvious in the criterion-2 sense too: the default engineering reflex is subtract-the-delta, which silently accumulates bf16 drift and corrupts a long-running swapping server. | `max_drift() = 0.00e+00` verified in situ across both domains. Caveat: drift 0 is expected *by construction* (`copy_` does no arithmetic) and would also read 0 if `pristine` aliased the live tensor — it is a clone (`detach().clone()`), and the functional proof is that scores return to base (astral 56.91% folded → 6.04% restored). |
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

Two questions, in order:

1. *Could someone have written this down from theory before running it?*
   - **No, and it changes what we believe** → 🚀
   - **No, and it changes what we believe is FALSE** → ❌ for the hypothesis, 🔥 for the measurement

2. If yes from theory — *has this concept been used for THIS class of system before?*
   - **No, or only in a non-obvious form** → 🔥 (say "first known application in
     this stack -- not verified against published work")
   - **Yes, and a library does exactly this for exactly this purpose** → ⭐

The second question is the one that is easy to get wrong in BOTH directions.
Under-claiming a genuine first application is as much an error as overclaiming a
discovery.
- **No, and it changes what we believe *is false*** → ❌ for the hypothesis, 🔥 for the measurement
