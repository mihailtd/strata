# Does speculative decoding produce worse answers?

**Verdict: no.** `+0.19pp` on the repo's own scorer, 95% CI `[−0.77, +1.17]`,
n=100 paired triples. The 2.20× is not paid for in quality.

This settles the question `docs/DECISIONS.md` §7 had been carrying as "genuinely
open" — the only unresolved doubt attached to the engine's largest measured win.

```
uv run --env-file .env python \
    benchmarks/runtime/speculative/speculation_quality/benchmark_speculative_quality.py
```

Results: [`results/speculative_quality.json`](../../../../results/speculative_quality.json)

---

## Why this needed an experiment at all

Greedy speculative decoding is *supposed* to be lossless. A draft token is
committed only when it equals `argmax` of the target model's logits, so in exact
arithmetic the output **is** the greedy output.

It is not lossless here, and not because of a bug. Plain decode steps the
GatedDeltaNet layers with the **single-token recurrent** kernel; verification runs
the **multi-token chunked** kernel. Those disagree in the last bits, so `argmax`
flips on near-ties. Every speculation benchmark in this repo therefore carries an
"exact %" column that never reaches 100.

That column is not answerable by making it bigger. It is **saturated by numerics**:
a decoder doing zero speculation already diverges at a comparable rate. So
exact-match can never isolate the effect of speculation, and the only way to answer
"is the different text *worse*?" is to score the text.

The stakes were concrete. Adapter stacking was retired for being a
**speed win that cost accuracy** (−10.96pp for 10.4ms, §4). If speculation had the
same shape, it was shipping by default in every benchmark headline.

## Design: three arms, because two cannot attribute

Comparing autoregressive against speculative confounds two causes. A third arm
separates them:

| arm | token choice via | drafts accepted | what it is |
| :--- | :--- | :--- | :--- |
| **A** `autoregressive` | single-token recurrent kernel | — | the reference path |
| **C** `forced_reject` | chunked verifier | **0 (forced)** | the chunked kernel *without* speculation |
| **B** `speculative` | chunked verifier | real | the shipped decoder, K=4 |

Arm C is the whole point. It runs the identical speculative machinery — same draft
head, same `K+1`-wide chunk, same state rollback — but pins `n_acc = 0`, so every
emitted token is the chunked verifier's own `argmax` and no draft is ever
committed.

- **A vs C** → the chunked kernel alone (pure numerics)
- **C vs B** → the effect of *accepting drafts* (speculation proper)
- **A vs B** → the shipped comparison, what a user receives

Had this been run with two arms, it would have reported "speculation changes
output" and been unable to say whether that mattered or why.

## Method

- **Prompts and scorer are the repo's canonical ones**, imported from
  `evaluate_folded_vs_wrapped.py` rather than redefined — `DOMAINS` and
  `score_question`. Rubric coverage (`expects`) where the eval data supplies one,
  else the good/bad term ratio. This makes the numbers directly comparable to the
  stacking result that set the precedent.
- **n = 100** paired triples: astral 40, postgresql 40, financial_planning 20.
- **256 new tokens**, matching the canonical eval. The speculation matrix uses 32,
  which is too short for a rubric to be covered and cannot score quality.
- **One pass per arm.** Greedy decoding is deterministic — confirmed independently
  by `tau_spread = 0.0000` across repeats in the speculation matrix. Repeats would
  buy nothing; the uncertainty is over **prompts**, so the CIs are a paired
  bootstrap over prompts (20k resamples).
- **Two scorings per arm:** full answers, and every arm truncated to the shortest
  arm's token count. See the retraction below for why that second one exists.

## Results

Pooled, n=100:

| arm | mean score | mean tokens |
| :--- | ---: | ---: |
| A autoregressive | 61.08% | 67.1 |
| C forced-reject | 62.10% | 70.0 |
| B speculative | 61.27% | 70.9 |

| contrast | isolates | Δ (full) | 95% CI | Δ (len-matched) | 95% CI |
| :--- | :--- | ---: | :--- | ---: | :--- |
| A → C | chunked kernel alone | +1.02pp | [−0.98, +3.69] | +1.85pp | [−0.67, +5.23] |
| C → B | accepting drafts | −0.83pp | [−3.50, +1.17] | −0.83pp | [−3.50, +1.17] |
| **A → B** | **shipped** | **+0.19pp** | **[−0.77, +1.17]** | **+1.02pp** | **[−0.62, +3.38]** |

Nothing significant anywhere, on either metric, in any domain.

**Per domain** (τ = accepted drafts per verification step):

| domain | τ | exact vs A: C | exact vs A: B | shipped Δ |
| :--- | ---: | ---: | ---: | :--- |
| astral | 2.33 | 85% | 80% | +0.47pp, CI [−1.93, +2.92] |
| postgresql | 1.87 | 68% | 68% | +0.00pp, CI [+0.00, +0.00] |
| financial_planning | 1.67 | 55% | 40% | +0.00pp, CI [+0.00, +0.00] |

## What it means

**The divergence is the kernel, confirmed directly.** Arm C — zero drafts
accepted — diverges from A on **28%** of prompts; arm B diverges on **33%**.
Accepting drafts adds ~5pp to a 28pp baseline. §7 had argued this from a 3-prompt
control; this is the measurement.

**Divergence is real and score-invisible.** 33/100 prompts emitted different text
under B than under A. **30 of those 33 scored identically.** Median first
divergence: token 28 (B), token 33 (C). Two greedy paths through numerically
different kernels produce different, equally valid text — which is what §7
claimed and what this confirms.

**The null is tight, not underpowered.** ±1pp on the same metric and eval sets
where stacking measured −10.96pp [−21.50, −1.62]. The worst case consistent with
this data is ~1pp of harm against a 2.20× throughput gain.

**No quality term is needed in the speculation gate.** `speedup > 1.0 AND τ ≥ 1.39
AND resolved` stands as-is.

### Scope — what this does *not* say

- It shows the text is not worse **on the repo's own scorer** (rubric coverage /
  good-bad ratio). That metric killed stacking, so the comparison is
  apples-to-apples, but it is coarse and **no human judgement was collected.**
- It does **not** license the speculative decoder for production. `server.py` has
  no draft path; the 2.20× is a benchmark result, not something the engine
  currently does.
- τ here (1.67–2.33) sits below the ~2.8 break-even that `mtp_draft.py` measured
  for this rig. This benchmark answers *quality*, not whether speculation pays.

---

## ⚠️ One retracted run — read this before writing another quality benchmark

`results/speculative_quality_RETRACTED_eos_overrun.json` reported **"+5.71pp
pooled, speculation IMPROVES quality, SIGNIFICANT"** (+11.18pp on astral). That
was a harness bug, and it is kept rather than deleted because the failure mode is
easy to repeat.

A speculative step commits up to `n_acc+1` tokens at once. Checking EOS on
`toks[-1]` at the top of the loop misses an EOS that lands **mid-block**, so
generation runs on:

| astral | mean tokens |
| :--- | ---: |
| A autoregressive | 46.1 |
| C forced-reject | 45.8 |
| B speculative | **119.2** ← 2.6× |

Arms A and C commit exactly one token per iteration and were never affected —
which is the only reason the asymmetry was visible at all. Then the good/bad term
ratio did the rest: it rewards length, and it scores `0.0` when an answer contains
no domain term whatsoever, so a rambling arm can convert `0.0` into `92.0` without
being more correct (prompt `neg_05`: A=0.0, C=0.0, B=92.0).

**The tell was internal to the data.** `financial_planning` is the only domain
scored by rubric coverage instead of the ratio, and it was the only domain
reporting exactly 0.00pp. The entire "win" lived in the length-sensitive metric.

Two guards are now in the harness, and belong in any arm-vs-arm quality benchmark
here:

1. **Report mean token count per arm.** A quality difference accompanied by a
   large length difference is a length difference until proven otherwise.
2. **Score length-matched as well as full**, truncating every arm to the shortest
   arm's token count for that prompt.

A result that says a speed optimisation makes output *better* should be treated as
a bug report against the harness, not a finding.
