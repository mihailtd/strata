# Graph-capturing the MTP draft head — exact in isolation, diverges in the loop

## Why anyone would try

§28: this rig is kernel-LAUNCH bound. `head_draft` costs ~20-25 ms/step for FOUR
single-layer forwards, against 34 ms for a full 32-layer graph-replayed forward.
One layer costing ~18% of a 32-layer model is almost entirely dispatch overhead,
and graph capture is the only thing that removes it.

Measured per-replay floor, stable across three independent attempts:

| attempt | head_draft eager | graphed | ms per replay |
| :--- | ---: | ---: | ---: |
| 1 | 24.02 | 15.74 | 3.94 |
| 2 | 20.63 | 15.42 | 3.86 |
| 3 | 19.85 | 15.32 | 3.83 |

**~3.85 ms per graph replay is a hard floor here.** Four replays cost ~15.4 ms, so
an UNROLLED K-step graph (one replay for all K drafts) should land near 4 ms —
a ~4x cut on that phase. That is the strongest argument for unrolling, and it is
measurement, not projection.

## What is proven correct

`probe_head_equiv.py` — no graph, no generation, logits compared directly:

| arm | max\|dlogit\| vs shipped path | argmax |
| :--- | ---: | ---: |
| A DynamicCache (reference) | 0.00000 | 4/4 |
| B StaticCache, `mask=None` | **0.40625** | 4/4 |
| C StaticCache, explicit additive mask | **0.00000** | 4/4 |

`_run_layer` hardcodes `attention_mask=None`, which is correct over a tight
DynamicCache and **wrong** over a 2048-slot StaticCache — it attends to unwritten
slots. With an explicit mask the StaticCache path is **bit-exact**.

`probe_graphed_head_v2.py` carries a numeric gate that must pass before any
generation runs. It **passes**: the captured graph is bit-exact (0.00000, 4/4) for
a draft chain from a fresh cache, using a shape-(1,) position buffer, a
materialised (1,1,1,max_seq) mask, and §26 counter pinning.

**So capture, mask, positions and counters are all verified correct.**

## What is NOT solved

Across a generation, tau collapses **1.803 -> 0.910** (throughput 0.694x) even
though every single step is bit-exact in isolation. The fault is cache LIFETIME.

`probe_lockstep.py` drives one trajectory and computes the head's draft logits
both ways at every step:

```
step   pos  hfilled  prev_nacc  stale_tail  max|dlogit|
   0    53       52         -1           0      0.00000
   1    58       57          4          -1     14.78906   <- diverges
```

Divergence is immediate and large, on the **first full accept** (`prev_n_acc=4`).
`stale_tail=-1` means `extend()` had already rewritten past everything the prior
draft wrote, so **live rejected speculative entries are ruled out**.

The open lead: a full accept is the one path where `new_h` comes from the verify
replay's hidden states rather than the commit re-forward. Untested.

## Hypotheses tested and REFUTED — do not repeat these

1. shared CUDA graph memory pool -> private pools changed nothing
2. head attends to unwritten StaticCache slots -> true in isolation, but masking
   made the generation WORSE (tau 0.910)
3. graph wiring (0-dim position scalar, in-graph mask) -> rewired to the bit-exact
   formulation, gate passes, generation still 0.910
4. stale speculative cache entries -> `stale_tail=-1` at the divergence

## If someone picks this up

Start from `probe_graphed_head_v2.py`'s passing gate — the single-step formulation
is known good. Diff the head's K/V tensors position-by-position between the
rebuilt and static caches at lockstep step 1, rather than reasoning about it.
The prize is ~4 ms/step from unrolling plus whatever the rest of the stack allows;
the risk is that tau is worth several times more than the milliseconds saved.

```bash
uv run --env-file .env python benchmarks/runtime/speculative/graphed_draft_head/probe_head_equiv.py
N=160 uv run --env-file .env python benchmarks/runtime/speculative/graphed_draft_head/probe_lockstep.py
```
