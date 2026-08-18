# Qwen3.5-0.8B as a draft model — closed, and it isolates the real bottleneck

## Why it was reopened

`CURRENT.md` recorded the 0.8B draft path as **FAILED**, but that failure was a
library refusal, not a measurement:

```
ValueError: assisted generation is not supported with stateful models,
            such as Qwen3_5ForCausalLM
```

`transformers` declines because it cannot roll back the 24 GatedDeltaNet
recurrent states by truncation. Our own loop rolls them back by in-place copy, so
that reason no longer applies to us — the question deserved real numbers.

Compatibility is not the issue. Both models share a **byte-identical tokenizer**
and `vocab_size=248320`, and both fit (14.4 GB together, under the 22 GB cap).

## Acceptance is good

`probe_draft_acceptance.py` needs no speculative machinery: under greedy decoding
the target's own continuation is exactly what a verifier would emit, so
acceptance can be simulated with plain forward passes.

| target | tau (K=4) | alpha by draft index |
| :--- | ---: | :--- |
| base 4B | 2.089 | 0.745, 0.568, 0.432, 0.344 |
| **astral-folded 4B** | **2.323** | 0.776, 0.630, 0.505, 0.411 |

The 0.8B drafts far better than the shipped MTP head (tau 2.32 vs 1.65). Note the
folded target is **easier** to draft for, not harder — the adapter makes output
more stereotyped. That runs opposite to §23's matched-adapter reasoning.

## Cost kills it

`probe_cost_units.py` and `probe_draft_graphed.py`, measured on the paths that
would actually run:

| component | ms | units (1 unit = one width-1 target token) |
| :--- | ---: | ---: |
| baseline, width-1 graph replay on 4B | 35.51 | 1.00 |
| verify, width-5 graph replay on 4B | 31.27 | **0.88** |
| commit re-forward, width-3 graph replay | 31.78 | **0.89** |
| draft, 4x eager 0.8B | 99.51 | 2.80 |
| draft, 4x **graph-captured** 0.8B | 66.23 | 1.86 |

**Projected 0.913x** with the graphed drafter — break-even tau is 2.64 and we
measured 2.323. Tuning K does not rescue it: K=3 -> 0.917x, K=2 -> 0.888x,
K=6 -> 0.857x.

### The reason, and it is physics

The 0.8B has **5.3x fewer parameters but only 2.14x lower latency** (16.56 vs
35.51 ms/token, both graph-captured). Batch-1 decode here is **depth-bound**, not
bandwidth-bound, and the 0.8B is **24 layers against the 4B's 32** — only 25%
fewer sequential steps. It is too *deep* to be a drafter no matter how small its
weights are.

This is the bar the retired 0.8B work already set: *"Speculation needs a draft
that is several times cheaper AND ~70% accurate."* The 0.8B has the accuracy and
not the cheapness.

## What this isolates

The verify is nearly **free** — 0.88 units for a 5-wide chunk, because graph
replay makes width almost costless. The hybrid tax is not the chunked verify.

It is the **commit re-forward: 31.78 ms, 0.89 units, ~25-40% of every step**, a
full forward that produces zero new tokens and exists only to recover the
recurrent state after exactly `n_acc+1` tokens.

Everything else the commit re-forward appears to provide is already available
from the verify pass: attention KV for the accepted prefix is causally correct
and needs no rewrite, and the hidden state at position `n_acc` is already in the
verify output. **Only the SSM state is missing.**

⚠️ Removing it is NOT free and is NOT untried — see §18, which built recurrent
state replay and measured **1.008x**. §18 predates bucketed graph capture (§25),
so its replacement cost was 24 eager per-layer launches — the same depth/launch
physics that killed the 0.8B here. That is what makes it worth re-measuring, not
an assumption that it will work.

## Run it

```bash
uv run --env-file .env python benchmarks/runtime/speculative/draft_model_0_8b/probe_draft_acceptance.py
uv run --env-file .env python benchmarks/runtime/speculative/draft_model_0_8b/probe_draft_graphed.py
```
