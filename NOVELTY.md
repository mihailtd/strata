# NOVELTY

Honest tiering of what this repo contributes. Companion to `CURRENT.md` (what is
live) and `SYSTEM.md` (hardware/software).

## THE GOVERNING RULE — read before tagging anything

**Novelty of APPLICATION counts. You do not have to invent the concept.**

**THE OPERATIONAL TEST: would vLLM / SGLang / Ollama / llama.cpp /
TensorRT-LLM do this?**

If the mainstream serving stacks do not do it, and it works, that is 🔥 — you
found something they missed or applied a known idea in a way they did not think
of. That is the bar. It is concrete and checkable, unlike "is this novel".

🔥 Applied Practice is earned by either of:

1. **The serving stacks do not do this.** An established concept applied to this
   class of system in a way the mainstream engines do not. The concept being old
   is irrelevant; the *pairing* is the work.
2. **A non-obvious use of a known technique** — something a competent engineer
   would not reach for by default, where the reason it works is the insight.

**Distinguish "missed it" from "chose differently".** If a serving stack solves
the same problem another way on purpose (e.g. vLLM serves multi-LoRA *unmerged*
with batched adapter kernels rather than folding-and-restoring, because it
optimises multi-tenant throughput over single-stream latency), that is a
different design point, not an oversight. Both can be right. Claim 🔥 for
"they do not do this and it works", not for "we picked a different tradeoff".

What does NOT earn a tier:

* Size of the speedup. A big number means the baseline was weak, not that the
  method is novel. Tag the method; report the number separately.
* Difficulty. Hard debugging on a standard technique is still ⭐.
* Re-using a library exactly as its authors intended for the purpose they
  intended (`peft.merge_and_unload` for merging LoRA → ⭐, not 🔥).

**Honesty guard.** "None of the serving stacks do this" is checkable — but only
by actually reading their code, which has NOT been done from this repo. Every
🔥 claim below is therefore marked with what is known versus what still needs
verifying against the real vLLM / SGLang / Ollama / llama.cpp sources. Write
**"not verified against current upstream"** until someone checks. The claim stays
defensible and still gets the credit.

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

Assessed against the operational test. **VERIFY column = whether "the serving
stacks do not do this" has actually been checked against upstream source. None
of it has been, from this repo.**

| item | why it passes the test | verify |
| :--- | :--- | :--- |
| **Loading the shipped `mtp.*` tensors** | Qwen3.5 ships a 15-tensor MTP draft head in the checkpoint that `transformers` **never loads**. Reading it straight out of the safetensors shards and building a working draft head from it — correct `[embedding; hidden]` fuse order, 3D mRoPE, its own KV cache — turns a dead payload into the only working speculation path here. Strongest candidate: the weights ship to everyone and go unused. | Does vLLM/SGLang have a Qwen3.5 MTP path yet? **UNCHECKED** |
| **Recurrent-state snapshot/restore for speculation** | Speculation on a stateful model is *refused* by `transformers` ("not supported with stateful models") because 24/32 layers carry a recurrent state that cannot be truncated like a KV cache. Snapshotting the fixed-size 52.5 MB state and restoring on partial acceptance converts an impossible rollback into a copy. Measured: every built-in speculation path fails on this architecture; this one works at 1.32–1.39×. | Do any engines speculate on GatedDeltaNet-style hybrids? **UNCHECKED** |
| **Pristine State Buffer + bit-exact `copy_` restore** | Master-weights discipline (keep an unmutated reference copy; never reconstruct by inverse arithmetic) carried from mixed-precision optimizers into runtime expert swapping. Non-obvious independently: the default reflex is subtract-the-delta, which silently accumulates bf16 drift and corrupts a long-running swapping server. | vLLM serves multi-LoRA **unmerged** (batched adapter kernels) — that is a *different design point*, not an oversight, so this is 🔥 for the fold-and-restore approach, not for beating them. `max_drift = 0.00e+00`, though 0 is expected by construction and would also read 0 under aliasing; functional proof is scores returning to base (56.91% → 6.04%). |
| **`activate_many()`** — multi-expert additive folding with restore | N experts folded into one set of weights, cost flat in N, restorable. `peft.add_weighted_adapter` combines adapters *offline into a new adapter*; this is a runtime fold with a restore path. | Partial — offline multi-adapter merging is standard. The runtime fold+restore framing is the delta. **UNCHECKED** |
| **`fla` (Triton) unblocking speculation on gfx1100** | Platform enablement: `fla` and `causal_conv1d` are two independent deps with separate fallbacks, and conflating them blocked this for a long time. Flattened verification 2.84× → 1.19×. | Consumer-AMD ROCm speculative decoding is poorly covered, but this is using `fla` **as intended**. Closer to ⭐ + platform work than a missed idea. |
| **Measurement discipline** | Re-measure baselines in-process; paired bootstrap CIs with the decision rule fixed *in advance*; adapter content-hashing to detect an adapter mutating under a published result. Caught four false results in one day. | Not a competitor comparison — internal methodology. Arguably the most transferable output regardless. |
| **bf16 merge-absorption calibration** | The 1/x form is elementary floating point, derivable in one line beforehand. What is ours is the measured constant (0.22 × bf16 eps) and validating it holds to 2.3% across a 16× range. | Not a missed idea. Kept at 🔥 for the calibration, **not** as a discovered law. |

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
