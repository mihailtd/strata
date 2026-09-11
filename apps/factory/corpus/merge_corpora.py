"""Merge domain corpora into a single training set, to test whether STACKING earns
its complexity.

THE QUESTION THIS ANSWERS
------------------------
The architecture's cost is real: N adapters, a folding engine, a router, and an
interference problem that took a week to characterise. Its benefit only exists if
stacking beats the obvious alternative -- **train one adapter on the union of the
corpora and ship that**.

Nobody has ever run that comparison here. Until it exists, every stacking result
is a measurement of something whose necessity is unestablished.

    ast+pg+duck (stacked)   72.75 postgres / 59.29 astral / 74.67 duckdb
    merged adapter          ???

If the merged adapter matches those numbers, the router, the folding engine, and
the whole interference investigation are unnecessary for N=3. If it does NOT --
if the domains degrade each other inside one adapter the way they do not when
folded separately -- that is the first hard evidence that stacking is required,
and it is the result the architecture has been missing.

WHY THE EVAL SETS ARE **NOT** MERGED
------------------------------------
It is tempting to merge the eval files too, but that destroys the comparison.
Each domain is scored by its own rubric (astral by uv/ruff term ratio, postgresql
by pgvector term ratio, duckdb by `expects` regexes). Merging them forces one
rubric onto records it does not fit, and the result is not comparable to the
stacking matrix.

Keeping the evals per-domain means the merged adapter's row drops straight into
the existing matrix next to `ast+pg+duck` with no reinterpretation.

    uv run python scripts/corpus/merge_corpora.py            # report
    uv run python scripts/corpus/merge_corpora.py --write    # emit merged corpora
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter

from runtime_common.canon import REPO_ROOT

# Merged sets to build. financial is deliberately excluded from both: its expert
# is worth +0.83pp over base and it does -36.67pp of collateral damage, so adding
# it would test the merge against the one domain already known to poison a stack.
MERGES: dict[str, list[str]] = {
    # the two closely-related SQL dialects -- the pair that produced the biggest
    # result in the matrix (pg+duck scored 88.76 on postgres vs 62.96 solo)
    "merged_sql": ["postgresql", "duckdb"],
    # + astral, matching the best all-round stacked condition (ast+pg+duck)
    "merged_all": ["postgresql", "duckdb", "astral"],
}

SOURCES = {
    "postgresql": "apps/factory/data/postgresql/training_data_v4.jsonl",
    "duckdb": "apps/factory/data/duckdb/training_data_v4.jsonl",
    "astral": "apps/factory/data/astral/training_data_v4.jsonl",
}


def load(rel: str) -> list[dict]:
    path = REPO_ROOT / rel
    if not path.exists():
        raise FileNotFoundError(f"missing corpus: {path}")
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="emit the merged corpora")
    ap.add_argument("--seed", type=int, default=0,
                    help="shuffle seed. Fixed so the merge is reproducible -- an "
                         "unseeded shuffle makes two 'identical' corpora train "
                         "differently and the difference looks like a real effect.")
    args = ap.parse_args()

    for name, domains in MERGES.items():
        print("=" * 78)
        print(f" {name}  <-  {' + '.join(domains)}")
        print("=" * 78)

        rows: list[dict] = []
        for d in domains:
            recs = load(SOURCES[d])
            for r in recs:
                # provenance survives the merge. Without it you cannot later ask
                # "did the postgres half degrade?" of a merged adapter's corpus.
                meta = dict(r.get("meta") or {})
                meta["merged_from"] = d
                r["meta"] = meta
            rows.extend(recs)
            print(f"   {d:12s} {len(recs):5d} records")

        # Shuffle so the domains interleave. Concatenated blocks mean the last
        # domain in the file gets the final gradient steps and is over-weighted
        # by the cosine schedule -- an ordering artifact that would look exactly
        # like "that domain merged better".
        random.Random(args.seed).shuffle(rows)

        print(f"   {'TOTAL':12s} {len(rows):5d} records (shuffled, seed={args.seed})")
        prov = Counter(r["meta"]["merged_from"] for r in rows)
        print("   provenance: " + "  ".join(f"{k}={v}" for k, v in sorted(prov.items())))

        out = REPO_ROOT / f"apps/factory/data/{name}/training_data_v4.jsonl"
        if args.write:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            print(f"   WROTE {out}")
        else:
            print(f"   (dry run -- would write {out})")
        print()

    print("Eval sets are NOT merged. Each domain keeps its own rubric, so a merged")
    print("adapter's scores drop straight into the stacking matrix next to")
    print("ast+pg+duck with no reinterpretation. See this file's docstring.")


if __name__ == "__main__":
    main()
