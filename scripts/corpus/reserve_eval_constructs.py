"""Reserve specific generator families for EVALUATION ONLY, by removing them from
the training corpora — then PROVE the reservation held.

THE PROBLEM THIS FIXES
----------------------
The held-out gate (`benchmarks/factory/agentic/chained_holdout/`) chose its test
constructs by scanning the **v2** corpora for zero occurrences. The v3 corpora then
added exactly those constructs, because they are the obvious contents of an
"Advanced PostgreSQL Features" and a "Modern Python" chapter:

    postgresql v3   g_adv_distinct_on        -> DISTINCT ON        (4 gate steps)
                    g_adv_filter_clause      -> FILTER (WHERE)     (4)
                    g_adv_ordinality_unnest  -> WITH ORDINALITY    (2) + unnest (2)
                    g_adv_lateral_join       -> LATERAL            (3)
                    g_adv_percentiles        -> percentile_cont    (2)
    astral v3       gen_func_partial_curry   -> functools.partial  (3)
                    gen_func_singledispatch  -> singledispatch     (3)
                    gen_py_protocol_slots    -> Protocol, __slots__(6)
                    gen_py_taskgroup         -> asyncio.TaskGroup  (3)

That is 32 of the gate's 45 steps — 71% — trained directly, with the construct
NAMED in several prompts. Scores would rise and mean nothing.

Scanning for absence is fragile: it is only valid until the next corpus revision,
and the corpus author cannot know which constructs are reserved. So invert it.
**Reserve by REMOVAL, and make the reservation an explicit, auditable list that
lives next to the data.** Then contamination is impossible by construction rather
than by scan, and adding new corpus material can never silently break the gate.

WHICH KIND OF RESERVATION -- read this before adding a domain
-------------------------------------------------------------
    Reserve CONSTRUCTS when they are PERIPHERAL to the domain.
    Reserve INSTANCES  when they ARE the domain.

astral is peripheral: remove `TaskGroup` and `singledispatch` and a corpus about
modern Python survives. Holding them out is a real generalization test.

duckdb is central: `read_parquet` / `COLUMNS()` / `GROUP BY ALL` ARE DuckDB, spread
across 11 of 14 families. Reserving them drops 90.3% of the corpus -- and it fights
the goal, because we train that expert to REACH FOR DuckDB. Hiding the tool cannot
teach a bias toward the tool. See docs/WHY_EXPERTS.md.

Applying the construct rule to a central construct fails one of two ways:
  1. silent no-op   -- names do not match, 0 dropped, verification says LEAKING
  2. destroyed corpus -- names do match, 90% of the data disappears

WHAT THIS DOES
--------------
1. Drops every record whose `meta.family` is in RESERVED from each v3 corpus.
2. Writes `training_data_v4.jsonl` alongside it.
3. VERIFIES the outcome: counts each reserved construct's textual signature in the
   filtered corpus and reports which gate constructs are genuinely clean. A
   construct that still appears (because an unreserved generator also emits it) is
   reported as STILL LEAKING rather than silently assumed safe.

The verification step is the point. Removal without verification is the same
mistake as scanning: an assumption dressed up as a guarantee.

    uv run python scripts/corpus/reserve_eval_constructs.py            # report only
    uv run python scripts/corpus/reserve_eval_constructs.py --write    # emit v4 corpora
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from gnn_experiment.canon import REPO_ROOT as REPO  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
# --------------------------------------------------------------------------
# The reservation. Every family here is EXCLUDED from training and belongs to
# the evaluation set alone. Keep this list in sync with the gate's constructs.
# --------------------------------------------------------------------------
RESERVED: dict[str, dict[str, list[str]]] = {
    "postgresql": {
        "families": [
            "adv_distinct_on",
            "adv_filter_clause",
            "adv_ordinality_unnest",
            "adv_lateral_join",
            "adv_percentiles",
        ],
        # textual signatures used to VERIFY the reservation, not to perform it
        "signatures": [
            r"distinct\s+on",
            r"\bfilter\s*\(\s*where",
            r"with\s+ordinality",
            r"\bunnest\s*\(",
            r"\blateral\b",
            r"percentile_cont",
            r"array_agg",
            r"date_trunc",
            r"\blag\s*\(",
            r"\blead\s*\(",
            r"\bcollate\b",
        ],
    },
    "astral": {
        "families": [
            "func_partial_curry",
            "func_singledispatch",
            "py_protocol_slots",
            "py_taskgroup",
        ],
        "signatures": [
            r"functools\.partial",
            r"\bpartial\s*\(",
            r"singledispatch",
            # Protocol is NOT reservable: `modern_typing` also emits it (43 records,
            # 103 hits after filtering py_protocol_slots). That family is valuable
            # general typing content, so the GATE drops Protocol instead and uses
            # cached_property, verified clean below.
            r"__slots__",
            r"TaskGroup",
            r"cached_property",
        ],
    },
    # ------------------------------------------------------------------
    # duckdb is DELIBERATELY ABSENT. Do not add it back.
    #
    # read_parquet / COLUMNS() / GROUP BY ALL / FROM-first ARE what DuckDB is.
    # They are emitted by 11 of the corpus's 14 families; reserving them drops
    # 1465 of 1622 records (90.3%), leaving 157 -- not a trainable corpus.
    #
    # More importantly it fights the goal. We train this expert to be BIASED
    # toward DuckDB on problems where DuckDB is the right tool. You cannot bias a
    # model toward a tool by hiding the tool from it. Construct-in-training is a
    # PREREQUISITE for that bias, not a contamination of it.
    #
    # DuckDB uses INSTANCE reservation instead: hold out specific problems,
    # schemas and table shapes, keep the constructs. See docs/WHY_EXPERTS.md.
    # ------------------------------------------------------------------
}

CORPORA = {
    "postgresql": REPO / "data/postgresql/training_data_v3.jsonl",
    "astral": REPO / "data/astral/training_data_v3.jsonl",
    "duckdb": REPO / "data/duckdb/training_data_v4.jsonl",
}


def record_text(rec: dict) -> str:
    if "text" in rec and rec["text"]:
        return rec["text"]
    msgs = rec.get("messages") or []
    return "\n".join(m.get("content", "") for m in msgs)


def family_of(rec: dict) -> str:
    m = rec.get("meta") or {}
    return str(m.get("family") or m.get("source") or "?")


def main() -> None:
    failures: list[str] = []
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="emit training_data_v4.jsonl (default: report only)")
    args = ap.parse_args()

    for domain, path in CORPORA.items():
        if domain not in RESERVED:
            # Not a bug and not an oversight: this domain uses INSTANCE
            # reservation, not construct reservation. Printed rather than skipped
            # silently so the policy stays visible in the output.
            print("=" * 78)
            print(f" {domain}: {path.name}")
            print("=" * 78)
            print("  INSTANCE-RESERVED, not construct-reserved -- intentionally skipped.")
            print("  Its constructs ARE the domain; reserving them would drop ~90% of the")
            print("  corpus AND defeat the purpose, since we train this expert to REACH FOR")
            print("  the tool. Hold out problems/schemas instead. See docs/WHY_EXPERTS.md\n")
            continue
        spec = RESERVED[domain]
        print("=" * 78)
        print(f" {domain}: {path.name}")
        print("=" * 78)
        if not path.exists():
            print(f"  MISSING: {path}")
            continue

        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        fams = Counter(family_of(r) for r in rows)
        reserved_set = set(spec["families"])
        missing_fams = reserved_set - set(fams)
        if missing_fams:
            # HARD FAILURE, not a warning. A reserved family whose name does not
            # match anything in the corpus drops zero records while the script
            # still prints a reassuring summary -- which is exactly what happened
            # to duckdb: three aspirational names, 0 records dropped, and every
            # construct still present in the hundreds. A warning was not enough.
            print(f"  ✗ FATAL: reserved families not present in corpus: {sorted(missing_fams)}")
            print(f"    present families: {sorted(fams)}")
            print("    The reservation would silently do NOTHING. Fix the names or")
            print("    remove them. Never leave a reservation that cannot bind.")
            failures.append(f"{domain}: unmatched reserved families {sorted(missing_fams)}")
            continue

        kept = [r for r in rows if family_of(r) not in reserved_set]
        dropped = len(rows) - len(kept)
        print(f"  records {len(rows)} -> {len(kept)}   dropped {dropped} "
              f"({dropped / max(1, len(rows)) * 100:.1f}%) as RESERVED FOR EVAL")
        for f in sorted(reserved_set):
            print(f"    reserved family {f:26s} {fams.get(f, 0):5d} records")

        # ---- VERIFY. Removal alone proves nothing; another generator may emit
        # the same construct.
        body = "\n".join(record_text(r) for r in kept).lower()
        print(f"  {'construct signature':28s} {'hits after filter':>18s}   verdict")
        clean, leaking = [], []
        for sig in spec["signatures"]:
            n = len(re.findall(sig, body, re.I))
            (clean if n == 0 else leaking).append((sig, n))
            print(f"    {sig:26s} {n:18d}   "
                  f"{'CLEAN — usable for eval' if n == 0 else 'STILL LEAKING'}")
        print(f"\n  -> {len(clean)} construct(s) genuinely reserved, "
              f"{len(leaking)} still present in training")
        if leaking:
            print("     the leaking ones must NOT be used as gate constructs, or the")
            print("     generator that emits them must also be reserved")

        if args.write:
            out = path.with_name("training_data_v4.jsonl")
            out.write_text("\n".join(json.dumps(r) for r in kept) + "\n")
            print(f"  WROTE {out} ({len(kept)} records)")
        print()

    # ---- CROSS-CORPUS LEAKAGE AUDIT -- DIRECTIONAL
    #
    # The two directions are NOT equally bad, and treating them the same buries
    # the real finding under noise.
    #
    #   DuckDB constructs in the POSTGRES corpus  -> BREAKS THINGS. Postgres has
    #     no COLUMNS(), GROUP BY ALL, QUALIFY or EXCLUDE. A pg expert that learned
    #     them emits SQL that does not parse.
    #
    #   Postgres constructs in the DUCKDB corpus  -> mostly harmless. DuckDB
    #     deliberately targets PostgreSQL syntax compatibility: DISTINCT ON,
    #     FILTER (WHERE), LATERAL, unnest, date_trunc, lag/lead all work there.
    #
    # So only flag a construct when it is INVALID in the receiving engine.
    print("=" * 78)
    print(" CROSS-CORPUS LEAKAGE AUDIT (directional: invalid-in-receiver only)")
    print("=" * 78)

    # construct -> engines it is VALID in
    VALIDITY = {
        r"COLUMNS\s*\(":        {"duckdb"},
        r"read_parquet":         {"duckdb"},
        r"GROUP\s+BY\s+ALL":     {"duckdb"},
        r"\bQUALIFY\b":          {"duckdb"},
        r"\.pl\s*\(\s*\)":       {"duckdb"},
        r"hive_partitioning":    {"duckdb"},
        # valid in BOTH -- DuckDB implements these on purpose
        r"distinct\s+on":        {"duckdb", "postgresql"},
        r"\bfilter\s*\(\s*where": {"duckdb", "postgresql"},
        r"\blateral\b":          {"duckdb", "postgresql"},
        r"\bunnest\s*\(":        {"duckdb", "postgresql"},
        r"date_trunc":           {"duckdb", "postgresql"},
        r"\blag\s*\(":           {"duckdb", "postgresql"},
        r"\blead\s*\(":          {"duckdb", "postgresql"},
        # postgres-only
        r"percentile_cont":      {"postgresql"},
        r"\bcollate\b":          {"postgresql"},
        r"with\s+ordinality":    {"postgresql"},
    }
    ENGINE_CORPORA = {
        "postgresql": REPO / "data/postgresql/training_data_v4.jsonl",
        "duckdb": REPO / "data/duckdb/training_data_v4.jsonl",
    }

    bodies = {}
    for eng, path in ENGINE_CORPORA.items():
        if path.exists():
            bodies[eng] = "\n".join(
                record_text(json.loads(l)) for l in path.read_text().splitlines() if l.strip()
            ).lower()

    real, benign = [], 0
    for eng, body in bodies.items():
        for sig, valid_in in VALIDITY.items():
            if eng in valid_in:
                continue                      # native here, not a leak
            hits = len(re.findall(sig, body, re.I))
            if hits:
                if valid_in & set(bodies):    # belongs to another engine we track
                    real.append((eng, sig, hits))
                else:
                    benign += 1

    if real:
        print("  INVALID-IN-RECEIVER (these produce output that will not run):")
        for eng, sig, hits in sorted(real, key=lambda x: -x[2]):
            print(f"     {sig:26s} {hits:5d} hits in {eng:12s} <- NOT VALID THERE")
        print("\n  These are the ones to fix. An expert trained on them emits broken SQL.")
    else:
        print("  CLEAN: no corpus contains a construct invalid in its own engine.")

    print("\n  (Constructs valid in BOTH engines are not reported. DuckDB implements")
    print("   DISTINCT ON / FILTER / LATERAL / unnest / date_trunc / lag / lead by")
    print("   design, so their presence in that corpus is correct, not leakage.)")
    print()

    if failures:
        print("\n" + "=" * 78)
        for f in failures:
            print(f"  FAILED  {f}")
        raise SystemExit(1)

    print("Reserved-for-eval lists are the contract. Any future corpus revision must")
    print("re-run this script; a family added upstream that emits a reserved construct")
    print("shows up as STILL LEAKING instead of silently invalidating the gate.")


if __name__ == "__main__":
    main()
