# scripts/corpus/ — build the training data

**There is one builder per domain. The version it targets lives in
`runtime.canon`, not in the filename.** Files named `*_v3_corpus.py` used to
sit here next to unversioned ones; that is precisely how a stale builder gets run.

## Order

```text
1. build_<domain>_corpus.py       ->  data/<domain>/training_data_v3.jsonl
2. reserve_eval_constructs.py     ->  data/<domain>/training_data_v4.jsonl   <- TRAIN ON THIS
```

`v4 = v3 minus the reserved evaluation families.` The builders emit v3; the
reservation step produces what you actually train on. Never train on v3.

| script | domain |
| :--- | :--- |
| `build_astral_corpus.py` | astral — uv / ruff / modern Python |
| `build_postgresql_corpus.py` | postgresql |
| `build_financial_corpus.py` | financial planning |
| `build_duckdb_corpus.py` | duckdb (needs `extract_duckdb_epubs.py` first) |
| `extract_duckdb_epubs.py` | epub → text, feeds the duckdb builder |
| `build_*_applied_examples.py` | applied/agentic examples folded into the corpora |
| `build_financial_speculation_prompts.py` | prompts for the speculative-decoding matrix |

## reserve_eval_constructs.py — do not skip it

It removes the reserved evaluation families from the corpora **and then verifies
zero residual occurrences**. Both halves matter.

Removal alone is worthless. It once reported `Protocol` still leaking at **103
hits** after the `py_protocol_slots` family was removed, because `modern_typing`
also emitted it. Trusting removal would have produced a held-out gate that looked
clean and silently was not.

The earlier approach — scanning a corpus for constructs that happen to be absent —
failed harder: the next corpus revision added **32 of the gate's 45 steps (71%)**,
because `DISTINCT ON` / `LATERAL` / `TaskGroup` are the obvious contents of an
"Advanced PostgreSQL" and a "Modern Python" chapter. Reservation must be an
explicit list that lives next to the data, not an inference from absence.

**Any new generator must be re-run through this script**, or the gate stops being
held out without anyone noticing.

## Adding a domain

1. Write `build_<domain>_corpus.py`, emit `data/<domain>/training_data_v3.jsonl`.
2. Add its reserved families to `RESERVED` in `reserve_eval_constructs.py`.
3. Run it with `--write` and confirm every construct reports **CLEAN**.
4. Add the domain to `DOMAINS` in `src/runtime/canon.py`.
