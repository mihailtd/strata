"""Replace the postgresql expert's recitation corpus with APPLIED, verified SQL.

THE DEFECT THIS FIXES
---------------------
The handoff gate measured the postgresql expert as WORSE than base at the one
thing it should own -- pgvector index DDL:

    check                             base     expert
    ann_method (USING hnsw/ivfflat)   1.000     0.400
    cosine_opclass (vector_cosine_ops) 1.000    0.200

It is NOT a coverage gap: `training_data.jsonl` mentions hnsw/ivfflat/pgvector
205 times. It is a SHAPE problem, measured over its 411 records:

    about-style ("What is ...", "How does ...")   350   85.2%
    write-style ("Write ...", "Generate ...")      61   14.8%
    book/author metadata                           45   10.9%

Records like "What is Marc Linster's background and what is his current focus
regarding PostgreSQL and AI?" teach author biography, not PostgreSQL. The corpus
trains a model to TALK ABOUT Postgres, never to WRITE it -- the same failure §9
found in the financial expert, whose applied-examples rebuild moved it
+0.00pp -> +35.00pp.

WHAT THIS PRODUCES
------------------
`data/postgresql/training_data_v2.jsonl` =
    the original corpus MINUS book/author metadata
  + generated APPLIED examples: a concrete schema and goal in, correct SQL out

WHY THIS ONE CAN BE TRUSTED MORE THAN THE FINANCIAL REBUILD
-----------------------------------------------------------
Every generated answer is parsed with `sqlglot` in the Postgres dialect before it
is written. A record that does not parse is a hard failure, not a warning. The
financial rebuild had no such gate -- it could only assert diversity.

`build_financial_planning_dataset.py` records what happens without diversity
guards: 940 records with ONE distinct question prefix produced an adapter that
scored BELOW base (60% -> 35%). So this asserts, and refuses to write, unless:
    * >= 12 distinct question phrasings
    * >= 20 distinct schemas
    * no single template is more than 12% of the generated set

    uv run --with sqlglot python scripts/build_postgresql_applied_examples.py
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import sqlglot

REJECTS: list = []
REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data/postgresql/training_data.jsonl"
OUT = REPO / "data/postgresql/training_data_v2.jsonl"

AUTHOR_NOISE = re.compile(
    r"\b(vibhor|kumar|marc|linster|foreword|the authors?|this book|the book|"
    r"definitive guide|according to the (text|passage|documentation)|"
    r"what is the context or motivation|background and what is|"
    r"key experts mentioned|who (is|are) the|expertise of|has architected)\b", re.I)

# ---------------------------------------------------------------- schemas
SCHEMAS = [
    ("products", "product catalogue", "embedding", 1024, [("title", "text"), ("price", "numeric"), ("in_stock", "boolean")]),
    ("documents", "document store", "embedding", 1536, [("body", "text"), ("tenant_id", "integer"), ("created_at", "timestamptz")]),
    ("tickets", "support desk", "embedding", 768, [("subject", "text"), ("status", "text"), ("priority", "integer")]),
    ("papers", "research library", "embedding", 768, [("abstract", "text"), ("year", "integer"), ("venue", "text")]),
    ("images", "media library", "embedding", 512, [("caption", "text"), ("album_id", "integer")]),
    ("messages", "chat history", "embedding", 1024, [("body", "text"), ("channel_id", "bigint"), ("sent_at", "timestamptz")]),
    ("listings", "marketplace", "embedding", 768, [("headline", "text"), ("city", "text"), ("price", "numeric")]),
    ("recipes", "recipe index", "embedding", 384, [("instructions", "text"), ("cuisine", "text")]),
    ("resumes", "talent pool", "embedding", 1024, [("summary", "text"), ("years_exp", "integer")]),
    ("incidents", "ops log", "embedding", 768, [("description", "text"), ("severity", "integer"), ("resolved", "boolean")]),
    ("faqs", "help centre", "embedding", 384, [("answer", "text"), ("locale", "text")]),
    ("audio_clips", "audio archive", "embedding", 512, [("transcript", "text"), ("duration_s", "numeric")]),
    ("contracts", "legal store", "embedding", 1536, [("clause_text", "text"), ("party", "text"), ("signed_on", "date")]),
    ("reviews", "review corpus", "embedding", 768, [("content", "text"), ("stars", "integer")]),
    ("places", "geo index", "embedding", 512, [("name", "text"), ("country", "text")]),
    ("courses", "learning catalogue", "embedding", 768, [("outline", "text"), ("level", "text")]),
    ("bugs", "issue tracker", "embedding", 1024, [("report", "text"), ("component", "text"), ("open", "boolean")]),
    ("emails", "mail archive", "embedding", 1536, [("body", "text"), ("folder", "text")]),
    ("songs", "music catalogue", "embedding", 384, [("lyrics", "text"), ("genre", "text")]),
    ("policies", "insurance docs", "embedding", 1024, [("text_body", "text"), ("region", "text")]),
    ("alerts", "monitoring", "embedding", 768, [("payload", "text"), ("service", "text"), ("acknowledged", "boolean")]),
    ("notes", "personal wiki", "embedding", 384, [("content", "text"), ("notebook", "text")]),
]

METRICS = [("cosine", "vector_cosine_ops", "<=>"),
           ("L2 / euclidean", "vector_l2_ops", "<->"),
           ("inner product", "vector_ip_ops", "<#>")]


def ddl(t, vcol, dim, cols):
    body = ",\n    ".join([f"{c} {ty}" for c, ty in cols])
    return (f"CREATE TABLE {t} (\n    id bigserial PRIMARY KEY,\n    {body},\n"
            f"    {vcol} vector({dim})\n);")


# ------------------------------------------------------- generators
def g_hnsw(rng, s):
    t, dom, v, dim, cols = s
    metric, opclass, _ = rng.choice(METRICS)
    m, ef = rng.choice([(16, 64), (16, 128), (24, 100), (32, 200)])
    q = rng.choice([
        f"Here is our {dom} table:\n\n{ddl(t, v, dim, cols)}\n\nAdd an HNSW index for {metric} similarity.",
        f"We store {dim}-dimensional vectors in {t}.{v}. Build an HNSW index tuned for {metric} search, and explain the two build parameters.",
        f"Our {dom} needs approximate nearest-neighbour search on {t}.{v} using {metric} distance. Write the index DDL.",
    ])
    a = (f"```sql\nCREATE INDEX {t}_{v}_hnsw_idx\n    ON {t}\n    USING hnsw ({v} {opclass})\n"
         f"    WITH (m = {m}, ef_construction = {ef});\n```\n\n"
         f"`{opclass}` must match the operator you query with, or the planner will "
         f"ignore the index. `m = {m}` sets the graph connectivity per node and "
         f"`ef_construction = {ef}` the build-time candidate list -- higher values "
         f"give better recall at the cost of build time and index size. Build the "
         f"index after bulk loading, not before.")
    return q, a, "hnsw_index"


def g_ivfflat(rng, s):
    t, dom, v, dim, cols = s
    metric, opclass, _ = rng.choice(METRICS)
    rows = rng.choice([50_000, 200_000, 1_000_000, 5_000_000])
    lists = max(10, int((rows ** 0.5) // 10 * 10)) if rows < 1_000_000 else rows // 1000
    q = rng.choice([
        f"{t} has about {rows:,} rows in our {dom}. Write an IVFFlat index for {metric} search and say how you picked the list count.",
        f"Create an IVFFlat index on {t}.{v} for {metric} distance, sized for {rows:,} rows.",
    ])
    a = (f"```sql\nCREATE INDEX {t}_{v}_ivfflat_idx\n    ON {t}\n    USING ivfflat ({v} {opclass})\n"
         f"    WITH (lists = {lists});\n```\n\n"
         f"A common starting point is `rows / 1000` for up to a million rows and "
         f"`sqrt(rows)` beyond that, giving {lists} here. IVFFlat must be built on a "
         f"populated table -- it clusters existing data, so an index built on an "
         f"empty table gives poor recall. Tune recall at query time with "
         f"`SET ivfflat.probes = ...;`.")
    return q, a, "ivfflat_index"


def g_knn(rng, s):
    t, dom, v, dim, cols = s
    metric, opclass, op = rng.choice(METRICS)
    k = rng.choice([5, 10, 20, 50])
    q = rng.choice([
        f"Write the query returning the {k} nearest rows in {t} to a parameter $1, using {metric} distance, that can actually use the ANN index.",
        f"For our {dom}, fetch the top {k} matches from {t} for an input embedding $1 ({metric}).",
        f"Show the {k}-nearest-neighbour query over {t}.{v} with {metric} distance.",
    ])
    a = (f"```sql\nSELECT id, {v} {op} $1 AS distance\nFROM {t}\nORDER BY {v} {op} $1\nLIMIT {k};\n```\n\n"
         f"The `ORDER BY` must use the same `{op}` operator the index was built for. "
         f"Ordering by a computed alias or wrapping the column in a function makes "
         f"the index unusable and forces a sequential scan.")
    return q, a, "knn_query"


def g_filtered(rng, s):
    t, dom, v, dim, cols = s
    scalar = next((c for c, ty in cols if ty in ("integer", "bigint", "boolean", "text")), "id")
    metric, opclass, op = rng.choice(METRICS)
    q = rng.choice([
        f"In {t} I need the 10 nearest neighbours to $1 but only among rows where {scalar} = $2. What are the options and their tradeoffs?",
        f"Write a filtered vector search over {t}: restrict on {scalar}, then rank by {metric} similarity.",
    ])
    a = (f"```sql\n-- Option A: partial index, when the filter values are few and known\n"
         f"CREATE INDEX {t}_{v}_hnsw_{scalar}_idx\n    ON {t}\n    USING hnsw ({v} {opclass})\n"
         f"    WHERE {scalar} = $2;\n\n"
         f"-- Option B: plain post-filter, when the filter is selective\n"
         f"SELECT id\nFROM {t}\nWHERE {scalar} = $2\nORDER BY {v} {op} $1\nLIMIT 10;\n```\n\n"
         f"Option B is the honest default, but be aware the ANN index returns a "
         f"fixed candidate list before the filter is applied, so a highly selective "
         f"`{scalar}` can leave fewer than 10 rows. Raise `hnsw.ef_search` to widen "
         f"the candidate pool, or use a partial index per filter value when the set "
         f"is small and stable.")
    return q, a, "filtered_search"


def g_efsearch(rng, s):
    t, dom, v, dim, cols = s
    ef = rng.choice([40, 100, 200, 400])
    q = rng.choice([
        f"Recall is too low on our {dom} searches over {t}. How do we trade latency for recall at query time without rebuilding the index?",
        f"Tune HNSW query-time recall for {t} to about {ef} candidates and show it applies per session.",
    ])
    a = (f"```sql\nSET hnsw.ef_search = {ef};\n\nSELECT id\nFROM {t}\nORDER BY {v} <=> $1\nLIMIT 10;\n```\n\n"
         f"`hnsw.ef_search` is the query-time candidate list and defaults to 40. "
         f"Raising it improves recall and increases latency roughly linearly; it "
         f"needs no reindex. It is a session GUC, so set it on the connection or "
         f"wrap it in a transaction with `SET LOCAL`. For IVFFlat the equivalent "
         f"knob is `ivfflat.probes`.")
    return q, a, "query_time_recall"


def g_hybrid(rng, s):
    t, dom, v, dim, cols = s
    txt = next((c for c, ty in cols if ty == "text"), "body")
    q = rng.choice([
        f"Combine full-text and vector search over {t} in our {dom} and fuse the two rankings.",
        f"Write a hybrid search over {t}: keyword match on {txt} plus embedding similarity on {v}.",
    ])
    a = (f"```sql\nWITH semantic AS (\n    SELECT id, ROW_NUMBER() OVER (ORDER BY {v} <=> $1) AS rank\n"
         f"    FROM {t}\n    ORDER BY {v} <=> $1\n    LIMIT 50\n),\nkeyword AS (\n"
         f"    SELECT id, ROW_NUMBER() OVER (ORDER BY ts_rank_cd(to_tsvector('english', {txt}), "
         f"plainto_tsquery('english', $2)) DESC) AS rank\n    FROM {t}\n"
         f"    WHERE to_tsvector('english', {txt}) @@ plainto_tsquery('english', $2)\n    LIMIT 50\n)\n"
         f"SELECT COALESCE(s.id, k.id) AS id,\n"
         f"       COALESCE(1.0 / (60 + s.rank), 0.0) + COALESCE(1.0 / (60 + k.rank), 0.0) AS score\n"
         f"FROM semantic s\nFULL OUTER JOIN keyword k USING (id)\nORDER BY score DESC\nLIMIT 10;\n```\n\n"
         f"This is Reciprocal Rank Fusion: each list contributes `1 / (60 + rank)`, "
         f"so the two scores never have to be calibrated against each other. Add a "
         f"GIN index on `to_tsvector('english', {txt})` or the keyword arm will scan.")
    return q, a, "hybrid_search"


def g_explain(rng, s):
    t, dom, v, dim, cols = s
    q = rng.choice([
        f"Our {dom} vector query on {t} got slow after a bulk load. How do I confirm whether the ANN index is being used?",
        f"How do I check the planner is choosing the vector index for a query on {t}, and what do I look for?",
    ])
    a = (f"```sql\nEXPLAIN (ANALYZE, BUFFERS)\nSELECT id\nFROM {t}\nORDER BY {v} <=> $1\nLIMIT 10;\n```\n\n"
         f"Look for `Index Scan using {t}_{v}_hnsw_idx`. If you see `Seq Scan` "
         f"followed by a `Sort`, the index was not used -- the usual causes are an "
         f"operator that does not match the index opclass, an `ORDER BY` on an "
         f"alias rather than the expression, or a missing `LIMIT`. After a bulk "
         f"load run `ANALYZE {t};` so the planner has current statistics.")
    return q, a, "explain_plan"


def g_upsert(rng, s):
    t, dom, v, dim, cols = s
    key = next((c for c, ty in cols if ty == "text"), "id")
    q = rng.choice([
        f"Write an idempotent upsert for {t} that refreshes the {v} column when a row already exists.",
        f"Our {dom} re-embeds rows nightly. Write the upsert that writes {t}.{v} without creating duplicates.",
    ])
    a = (f"```sql\nINSERT INTO {t} ({key}, {v})\nVALUES ($1, $2)\n"
         f"ON CONFLICT ({key}) DO UPDATE\n    SET {v} = EXCLUDED.{v}\n"
         f"WHERE {t}.{v} IS DISTINCT FROM EXCLUDED.{v};\n```\n\n"
         f"The `WHERE` on the DO UPDATE arm skips writes when the embedding has not "
         f"changed, which avoids dead tuples and needless index churn. `ON CONFLICT` "
         f"requires a unique constraint on `{key}`.")
    return q, a, "upsert"


def g_dim_migration(rng, s):
    t, dom, v, dim, cols = s
    new = rng.choice([d for d in (384, 512, 768, 1024, 1536) if d != dim])
    q = rng.choice([
        f"We are moving {t} from a {dim}-dim model to a {new}-dim one. Write the migration with no downtime for reads.",
        f"Change {t}.{v} from vector({dim}) to vector({new}) safely.",
    ])
    a = (f"```sql\nALTER TABLE {t} ADD COLUMN {v}_new vector({new});\n\n"
         f"-- backfill in batches from the application, then:\n"
         f"CREATE INDEX CONCURRENTLY {t}_{v}_new_hnsw_idx\n    ON {t}\n"
         f"    USING hnsw ({v}_new vector_cosine_ops);\n\n"
         f"BEGIN;\nALTER TABLE {t} DROP COLUMN {v};\n"
         f"ALTER TABLE {t} RENAME COLUMN {v}_new TO {v};\nCOMMIT;\n```\n\n"
         f"A vector column's dimension is part of its type, so it cannot be widened "
         f"in place -- you add, backfill, index concurrently, then swap. "
         f"`CREATE INDEX CONCURRENTLY` avoids holding a write lock but cannot run "
         f"inside a transaction block.")
    return q, a, "dim_migration"


GENERATORS = [g_hnsw, g_ivfflat, g_knn, g_filtered, g_efsearch,
              g_hybrid, g_explain, g_upsert, g_dim_migration]


def sql_blocks_parse(answer: str) -> bool:
    """Every ```sql block must parse as Postgres. Hard gate."""
    blocks = re.findall(r"```sql\s*(.*?)```", answer, re.S)
    if not blocks:
        return False
    for b in blocks:
        stmts = [x for x in re.split(r";\s*\n", b) if x.strip() and not x.strip().startswith("--")]
        for st in stmts:
            st = "\n".join(l for l in st.splitlines() if not l.strip().startswith("--")).strip()
            if not st:
                continue
            # sqlglot knows pgvector's <=> and <-> but NOT <#> (negative inner
            # product). That is a parser gap, not invalid SQL, so it is normalised
            # for the syntax check only -- the emitted training data keeps <#>.
            st = st.replace("<#>", "<->")
            try:
                if not sqlglot.parse(st + ";", dialect="postgres"):
                    REJECTS.append(("empty", st))
                    return False
            except Exception as ex:
                REJECTS.append((type(ex).__name__, st))
                return False
    return True


def main() -> None:
    rng = random.Random(20260818)

    kept, dropped = [], 0
    for line in SRC.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if AUTHOR_NOISE.search(d["messages"][0]["content"]):
            dropped += 1
            continue
        kept.append(d)

    by_fam: dict[str, list] = {}
    rejected = 0
    seen_q = set()
    for s in SCHEMAS:
        for g in GENERATORS:
            for _ in range(3):
                q, a, fam = g(rng, s)
                if not sql_blocks_parse(a):
                    rejected += 1
                    continue
                if q in seen_q:
                    continue
                seen_q.add(q)
                by_fam.setdefault(fam, []).append(
                    {"messages": [{"role": "user", "content": q},
                                  {"role": "assistant", "content": a}],
                     "meta": {"source": "applied_generated", "family": fam,
                              "table": s[0]},
                     # the trainer reads `text` (dataset_text_field="text"), so it
                     # must match the corpus's existing "### Question / ### Answer"
                     # framing exactly or the new records train a different format
                     "text": f"### Question:\n{q}\n\n### Answer:\n{a}"})
    # EQUALISE: every family contributes the same count, so no template can
    # dominate the set however many variants a generator happens to produce.
    per = min(len(v) for v in by_fam.values())
    made = []
    for fam in sorted(by_fam):
        rng.shuffle(by_fam[fam])
        made.extend(by_fam[fam][:per])
    fams = Counter(m["meta"]["family"] for m in made)

    prefixes = {m["messages"][0]["content"].split()[0].lower() for m in made}
    tables = {m["meta"]["table"] for m in made}
    top_share = max(fams.values()) / max(1, len(made))

    print(f"  original            {len(kept) + dropped}")
    print(f"  dropped as noise    {dropped}  (book/author metadata)")
    print(f"  kept                {len(kept)}")
    print(f"  generated applied   {len(made)}  (rejected by sqlglot: {rejected})")
    print(f"  distinct phrasings  {len(prefixes)}")
    print(f"  distinct schemas    {len(tables)}")
    print(f"  largest family      {top_share:.1%}  {fams.most_common(3)}")

    assert len(prefixes) >= 12, f"only {len(prefixes)} distinct question phrasings"
    assert len(tables) >= 20, f"only {len(tables)} distinct schemas"
    assert top_share <= 0.12, f"one family is {top_share:.1%} of the set"
    if rejected:
        print("  --- rejected samples ---")
        for kind, st in REJECTS[:4]:
            print(f"   [{kind}] {st[:160]!r}")
    assert rejected == 0, f"{rejected} generated answers failed sqlglot"

    out = kept + made
    rng.shuffle(out)
    OUT.write_text("\n".join(json.dumps(r) for r in out) + "\n")
    applied = len(made) + sum(
        1 for r in kept
        if re.search(r"\bwrite\b|\bcreate a\b|\bgenerate\b|\bimplement\b", r["messages"][0]["content"], re.I))
    print(f"  WROTE {OUT}  ({len(out)} records, ~{applied/len(out)*100:.1f}% applied)")


if __name__ == "__main__":
    main()
