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

    uv run python scripts/reserve_eval_constructs.py            # report only
    uv run python scripts/reserve_eval_constructs.py --write    # emit v4 corpora
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

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
}

CORPORA = {
    "postgresql": REPO / "data/postgresql/training_data_v3.jsonl",
    "astral": REPO / "data/astral/training_data_v3.jsonl",
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="emit training_data_v4.jsonl (default: report only)")
    args = ap.parse_args()

    for domain, path in CORPORA.items():
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
            print(f"  ⚠ reserved families not present in corpus: {sorted(missing_fams)}")
            print("    (family naming may differ -- reservation would silently do nothing)")

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

    print("Reserved-for-eval lists are the contract. Any future corpus revision must")
    print("re-run this script; a family added upstream that emits a reserved construct")
    print("shows up as STILL LEAKING instead of silently invalidating the gate.")


if __name__ == "__main__":
    main()
