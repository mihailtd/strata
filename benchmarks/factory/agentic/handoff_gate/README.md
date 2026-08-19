# The handoff gate — does multi-expert routing beat the generalist?

The agentic blueprint (autonomous mid-turn routing, 6–8 experts, self-correction)
rests on one unverified assumption: **specialised experts beat the base model on
realistic multi-step tasks.** §21 tested a single swap scored on a TERM LIST, and
term-list scoring is what produced two blind evals here already (§9, §16).

This gate answers it before any infrastructure gets built.

## Design

Three arms on identical 3-step agent conversations (2 database steps, then a
Python step, so every arm makes a cross-domain handoff mid-conversation):

| arm | routing |
| :--- | :--- |
| A | base 4B only, never swaps |
| B | ORACLE — expert hand-assigned per step (upper bound) |
| C | SELF — the model picks its own expert (what would ship) |

Scored **objectively** wherever possible: SQL must parse as Postgres via
`sqlglot`, Python via `ast.parse`. A small pre-registered structural rubric adds
the rest, and the prompts are audited so no rubric term leaks into the prompt
that elicits it.

Decision rule, fixed before running:

    B <= A       -> blueprint dead, experts do not help
    B > A ~= C   -> experts work, ROUTING is the hard problem
    C > A        -> build everything

Outcome: v1 hit the first branch, v2 hits the third in DIRECTION but not with
significance -- see the re-run below.

## Result (5 tasks, 15 paired steps)

| arm | score | vs A | parse rate | routing acc | mean swap |
| :--- | ---: | ---: | ---: | ---: | ---: |
| A base | **0.867** | — | 0.800 | — | — |
| B oracle | 0.790 | −0.077 | 0.800 | — | 14.3 ms |
| C self | 0.790 | −0.077 | 0.800 | **1.000** | 20.7 ms |

Paired bootstrap: **95% CI [−0.200, +0.043]** — not significant. The honest claim
is **no measured benefit from expert routing**, not that experts hurt.

**Routing is not the problem.** 15/15 correct, and C matches B exactly.

### Where the experts differ

| check | A base | experts |
| :--- | ---: | ---: |
| `ann_method` (USING hnsw/ivfflat) | **1.000** | 0.400 |
| `cosine_opclass` (vector_cosine_ops) | **1.000** | 0.200 |
| `route_decorator` (@app.get) | 0.600 | **1.000** |

The astral expert genuinely helps on FastAPI structure. The postgresql expert is
**worse at pgvector than base** — the one thing it should own.

### Why, and it is not a coverage gap

`data/postgresql/training_data.jsonl` mentions hnsw/ivfflat/pgvector/vector_cosine
**205 times**. The expert was trained on this material. But the data is
book-derived recitation Q&A:

> "What is the context or motivation behind Vibhor Kumar and Marc writing a
> definitive guide to PostgreSQL and AI?"

That teaches the model to **talk about** PostgreSQL, not to **write** it — the
exact failure §9 diagnosed for the financial expert, whose applied-examples
rebuild moved it +0.00pp → +35.00pp. postgresql scores +27.92pp on its own eval
(§16) because that eval is recitation-shaped too, matching its training
distribution.

## RE-RUN after rebuilding the postgresql corpus — the gate FLIPPED

`scripts/corpus/build_postgresql_applied_examples.py` replaced the recitation corpus
(14.8% applied, 10.9% author biography) with 741 records at 54.1% applied, every
generated SQL answer verified by `sqlglot`. Retrained 4:18, loss 1.527 -> 0.804.

Same instrument, same prompts, only the adapter changed:

| arm | v1 adapter | v2 adapter |
| :--- | ---: | ---: |
| A base | 0.867 | 0.867 (identical — clean control) |
| B oracle | 0.790 (**−0.077**) | **0.907 (+0.040)** |
| C self | 0.790 (−0.077) | **0.907 (+0.040)** |
| parse rate | 0.800 | **0.867** |

The pgvector regression is **gone**, and this part is categorical, not noise —
consistent across all 15 steps:

| check | base | v1 expert | v2 expert |
| :--- | ---: | ---: | ---: |
| `ann_method` | 1.000 | 0.400 | **1.000** |
| `cosine_opclass` | 1.000 | 0.200 | **1.000** |

⚠️ **The net win is NOT significant.** Paired bootstrap: mean **+0.0400**, 95% CI
**[−0.0133, +0.1067]**. 3 steps improved, 1 regressed, **11 tied**.

The blocker is a **CEILING EFFECT**: base already scores 1.000 on 8 of the 10
checks, so there is almost no headroom for an expert to demonstrate value. Before
this gate can settle the blueprint, the tasks need to be hard enough that base
does NOT saturate them.

## ⚠️ Instrument note — the first run was invalid

Run 1 scored `parses=0/3` on **every** arm while the rubric still scored 0.3–0.55,
because Qwen3.5's thinking mode burns **>1024 tokens** and never closes `<think>`.
At `max_new=1024` there were 13 code fences, all draft attempts *inside* the
reasoning block, ending mid-sentence on "Wait, I recall that". The rubric was
matching keywords in prose. Fixed with `enable_thinking=False`, which is also the
right product choice: an agent that deliberates 1000+ tokens per tool call is
unusable regardless of adapter quality.

## Run it

```bash
MAX_NEW=384 uv run --with sqlglot --env-file .env python \
  benchmarks/factory/agentic/handoff_gate/bench_handoff_gate.py
```
