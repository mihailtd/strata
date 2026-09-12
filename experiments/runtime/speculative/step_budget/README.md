# Where the speculative step actually goes — and why "do less work" never helps here

## The budget

512 tokens, chat-template prompt, astral folded, every phase device-synced.
**99.8% accounted**, tau=1.803, step = 118.00 ms, 23.71 tok/s.

| phase | ms/step | % |
| :--- | ---: | ---: |
| verify (width-5 graph replay) | 34.34 | 29.1% |
| commit re-forward | 27.16 | 23.0% |
| head_draft (4x MTP head, eager) | 23.91 | 20.3% |
| head_prefill | 11.69 | 9.9% |
| accept_sync (`.item()` x8) | 8.24 | 7.0% |
| bookkeep (`torch.cat`) | 7.04 | 6.0% |
| snapshot | 4.71 | 4.0% |
| cat_hids | 0.66 | 0.6% |

**The commit re-forward is 23% of the step, not the whole story.** Removing it
entirely leaves ~90.8 ms/step -> 30.9 tok/s -> only **1.10x** over the 28.16 tok/s
graph baseline. It cannot deliver the 1.36-1.64x it appears to promise, because
56.5 ms/step is not model compute at all.

## The law this rig obeys

**Latency here is kernel-LAUNCH bound, not work bound.** Four independent
measurements, all pointing the same way:

| change | work removed | speed gained |
| :--- | :--- | ---: |
| 0.8B drafter instead of 4B (§27) | 5.3x fewer params | **2.14x** |
| §18 recurrent state replay | a whole forward | **1.008x** |
| incremental head_prefill (`probe_incremental_head.py`) | **180x** less work | **1.001x** |
| graph-capturing head_draft | none — same math, fewer launches | **-34% on that phase** |

Only the last one moves anything, and it is the only one that removes *launches*
rather than *work*. `head_draft` costing 23.91 ms for four SINGLE-layer forwards,
against 34 ms for a full 32-layer graph-replayed forward, is the clearest
statement of it: one layer costs 18% of a 32-layer model, so ~all of it is
overhead.

**Design consequence:** on this hardware, optimise by collapsing launches (CUDA
graph capture), never by reducing arithmetic. `probe_incremental_head.py` is kept
precisely because it is a clean null result — it is token-identical to the current
path and 1.001x.

## Attempted and NOT achieved

`probe_graphed_head_ATTEMPTED.py` captures a width-1 graph for the head's draft
step over a StaticCache. The speed prediction held — `head_draft` 24.02 -> 15.74 ms,
step 114.48 -> 101.77 ms — but the graphed head does not draft **equivalently**:

| attempt | tau | throughput |
| :--- | ---: | ---: |
| eager head (reference) | 1.803 | 1.000x |
| graphed, no attention mask | 1.311 | 0.927x |
| graphed, device-built mask | 0.910 | 0.706x |

Adding a mask over the static cache made it **worse**, so "the head attends to
unwritten slots" is refuted as the explanation. The tau loss costs several times
more than the ~5 ms saved. **Do not re-run this without first understanding why a
StaticCache changes the head's drafting** — the speed win is real and small, the
equivalence bug is real and large.

## Run it

```bash
N=512 uv run --env-file .env python benchmarks/runtime/speculative/step_budget/probe_step_budget.py
```
