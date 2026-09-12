# Chained held-out benchmark — the experts are rubric-followers

## What this settles

Earlier gates showed the experts beating base on tasks whose machinery matched
their training families one-to-one (HNSW indexes, RRF queries, `@mcp.tool()`,
PEP 723). That measures **rubric-following**, not capability — the corpora were
built around the same constructs the benchmarks scored.

This benchmark removes that confound two ways:

1. **Held-out machinery.** Every construct was verified to occur **zero** times
   across both training corpora. Constructs found IN the corpora were rejected:
   WITH RECURSIVE (4), window `OVER()` (169), LISTEN/NOTIFY (20), MATERIALIZED
   VIEW (68), GROUPING SETS (62), TRIGGER/plpgsql (33), JSONB (41), SKIP LOCKED
   (11), argparse (86), asynccontextmanager (86), TypedDict (32), pytest (22),
   contextlib (43), itertools (47), Click (11), dataclass (10).
2. **Ground-truth scoring, not regex.** SQL is executed against a seeded
   database and the **returned rows** compared to a reference solution's rows;
   Python is executed and **stdout** compared. Emitting the right keywords earns
   nothing. Prompts never name the construct — the requirement is stated
   semantically, so any correct approach scores.

15 chained tasks x 3 turns, prior turns fed back as conversation context.

## Result

| arm | mean | edge vs base | 95% CI | significant |
| :--- | ---: | ---: | :--- | :--- |
| A base | **0.4667** | — | — | — |
| B oracle | 0.2333 | **−0.2333** | [−0.3556, −0.1111] | **yes** |
| C self | 0.2333 | −0.2333 | [−0.3556, −0.1111] | yes |

Routing accuracy **1.000** over 44 decisions — routing is not the issue, again.

### Where the loss lives

| construct | base | experts |
| :--- | ---: | ---: |
| `DISTINCT ON` | **0.750** | 0.000 |
| `LEAD` | **0.500** | 0.000 |
| `FILTER (WHERE)` | **0.250** | 0.000 |
| `__slots__` | **1.000** | 0.000 |
| `functools.partial` | **1.000** | 0.000 |
| `percentile_cont` | 0.500 | **1.000** |
| `LAG`, `WITH ORDINALITY`, `Protocol`, `singledispatch` | — | tied |
| `LATERAL`, `array_agg`, `unnest`, `date_trunc`, `TaskGroup` | 0.000 | 0.000 |

## What it means

**The adapters trade general capability for trained-distribution performance.**
They beat base where the corpus matches (+0.040 handoff gate, large edges on the
applied gate) and are **significantly worse** where it does not. This is the same
narrowing that made the v1 postgresql adapter worse at pgvector than base, and
the v1 astral adapter worse at Python — the rebuild moved the target, it did not
remove the mechanism.

Practically:

* production tasks that look like the training rubric -> experts help
* production needing general SQL/Python -> experts **hurt**; route to base
* "a 4B that punches above its weight via experts" is **not** supported outside
  the trained distribution

Both v2 adapters are rank 8 / **alpha 128** — a scaling of 16, which is
aggressive. The most testable next lever is corpus-mixing (retain general data in
the adapter training) or lowering alpha, then re-running THIS benchmark. It is
now the instrument that can tell narrowing from capability.

## Run it

```bash
MAX_NEW=384 uv run --env-file .env python \
  benchmarks/factory/agentic/chained_holdout/bench_chained_holdout.py
```

⚠️ Reference solutions are validated before any model runs; a broken reference is
dropped loudly (1 of 45 was — `COLLATE "und-x-icu"` is unavailable in pglite)
rather than silently scoring every arm zero.
