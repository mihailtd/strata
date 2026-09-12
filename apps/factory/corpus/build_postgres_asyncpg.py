"""PostgreSQL: the asyncpg driver surface, which the corpus does not contain at all.

WHAT THIS FIXES -- measured
---------------------------
apps/factory/data/postgresql/training_data_v5.jsonl, 1368 records:

    asyncpg        0
    create_pool    0
    pgvector      60
    <=>          117

So the expert knows the vector OPERATOR and nothing about the DRIVER. Asked for an
asyncpg pool with pgvector similarity search, v6 postgresql solo produced 130 tokens
containing, in one answer:

    ```sql                      <- a Python block labelled as SQL
    asyncpg.create_pool(...)    <- no await, on a coroutine
    '... LIMIT %s', query_vec   <- $1 and %s placeholders mixed in one query
    cursor = await conn.execute(...)   <- asyncpg has no cursor from execute()
    "<=> computes the cosine similarity"  <- it is a DISTANCE

Base, with no adapter, wrote 1743 tokens of correct pooled asyncpg with timeouts and
error handling. The expert made it strictly worse. That is the cost of an expert that
answers confidently outside what it was taught, and it is the single strongest
argument in this repo for the shelved silence-loss line.

DESIGN NOTE -- ~40% of these carry a `**Not X**` rejection, matching the share the
postgresql corpus already runs at (32.3%). NOT the 76% that python_modern runs at:
that is the ratio that turned python_modern into a template emitter.

    uv run python scripts/corpus/build_postgres_asyncpg.py [--write]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

from runtime_common.canon import REPO_ROOT

ANSWER_MARKER = "\n\n### Answer:\n"
GEN = "postgres_asyncpg_v1"

TABLES = [
    "documents",
    "products",
    "articles",
    "tickets",
    "listings",
    "profiles",
    "posts",
    "assets",
    "reviews",
    "incidents",
]
VEC_COLS = ["embedding", "vector", "embed", "content_vector", "doc_embedding"]
DIMS = [384, 768, 1024, 1536]
KS = [5, 10, 20, 50]
POOL_MIN, POOL_MAX = [2, 4, 5], [10, 20, 32, 50]
CTX = [
    "",
    " in a FastAPI service",
    " behind an internal API",
    " in a worker process",
    " under load from a batch job",
    " in a service that also serves web traffic",
    " for a recommendation endpoint",
    " in a long-running consumer",
    " where connections keep being exhausted",
    " on a read replica",
]
PHRASE = [
    "{task}?",
    "How do I {task}?",
    "Need to {task}.",
    "{task} -- show me the code.",
    "What's the right way to {task}?",
    "{task}. Code please.",
    "Can you {task}?",
]

OPS = [
    ("<=>", "cosine distance", "vector_cosine_ops"),
    ("<->", "L2 distance", "vector_l2_ops"),
    ("<#>", "negative inner product", "vector_ip_ops"),
]

# Answer openers. Deliberately varied: python_modern's corpus opens 23.6% of its
# answers with "Here is the ..." and the adapter now emits that template on every
# question regardless of topic.
# Lexical variation. The dedup key strips digits, so a family that varies only
# min_size/max_size collapses to ONE record -- the first build of this file kept 1 of
# 120 pool_lifespan records for exactly that reason. Rationale phrasing is what
# survives normalisation.
FN = ["fetch_rows", "load_page", "list_active", "query_recent", "select_batch"]
FILT = [
    ("active = $1", "True"),
    ("status = $1", '"open"'),
    ("owner_id = $1", "owner_id"),
    ("tenant_id = $1", "tenant_id"),
    ("created_at > $1", "since"),
]
ALT_POOL = [
    "Build it once in `lifespan`, not per request: a pool created per request is a "
    "pool of one that never gets reused, which is slower than no pool at all",
    "One pool for the process. Creating one per request defeats the point — you pay "
    "the connect handshake every time and hold nothing open between calls",
    "Construct it at startup and close it at shutdown. A per-request pool never "
    "amortises its own connections, so it costs more than going without",
]
ALT_ACQUIRE = [
    "`async with pool.acquire()` returns the connection to the pool even if the query raises",
    "the `async with` form releases the connection on the way out, exception or not",
    "acquiring inside `async with` guarantees the connection goes back even when the query blows up",
]
ALT_PARAM = [
    "asyncpg uses **numeric** placeholders — `$1`, `$2` — and binds them server-side as a prepared statement",
    "placeholders in asyncpg are positional and numeric (`$1`, `$2`); the values are "
    "bound server-side, never interpolated",
    "asyncpg speaks the binary protocol, so parameters are `$1`-style and are sent separately from the statement text",
]
ALT_TXN = [
    "`conn.transaction()` is an async context manager that commits on clean exit and rolls back on any exception",
    "the `transaction()` context manager commits when the block ends normally and rolls back if anything raises",
    "wrap both writes in `conn.transaction()` — it commits at the end of the block and rolls back on any exception",
]
ALT_COPY = [
    "`copy_records_to_table` uses the binary COPY protocol — one stream instead of one round trip per row",
    "COPY streams the whole batch in a single protocol message rather than a statement per record",
    "`copy_records_to_table` pushes the rows as one binary stream; there is no per-row round trip",
]
ALT_SYNC = [
    "it is a blocking driver, so a call inside `async def` stalls the whole event loop "
    "and every other in-flight request with it",
    "psycopg2 blocks the thread, and in an async handler that thread is the event loop "
    "— every concurrent request stops with it",
    "a synchronous driver inside `async def` parks the event loop, so the concurrency "
    "you wrote the handler for disappears",
]


def rec(family: str, question: str, answer: str) -> dict:
    return {
        "text": f"### Question:\n{question}{ANSWER_MARKER}{answer}",
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
        "meta": {"family": family, "tool": "postgresql", "gen": GEN},
    }


def build(rng: random.Random, n_per: int) -> list[dict]:
    out: list[dict] = []

    def add(fam: str, task: str, ans: str) -> None:
        q = rng.choice(PHRASE).format(task=task + rng.choice(CTX))
        out.append(rec(fam, q[0].upper() + q[1:], ans))

    for _ in range(n_per):
        t = rng.choice(TABLES)
        vc = rng.choice(VEC_COLS)
        dim = rng.choice(DIMS)
        k = rng.choice(KS)
        lo, hi = rng.choice(POOL_MIN), rng.choice(POOL_MAX)
        op, opname, opclass = rng.choice(OPS)
        filt = rng.choice(FILT)

        add(
            "apg_pool_lifespan",
            "create an asyncpg connection pool once and reuse it",
            f"```python\nfrom contextlib import asynccontextmanager\n\n"
            f"import asyncpg\nfrom fastapi import FastAPI\n\n\n"
            f"@asynccontextmanager\nasync def lifespan(app: FastAPI):\n"
            f"    app.state.pool = await asyncpg.create_pool(\n"
            f"        dsn=settings.database_url,\n        min_size={lo},\n"
            f"        max_size={hi},\n        command_timeout={rng.choice([5, 10, 30])},\n"
            f"        max_inactive_connection_lifetime={rng.choice([300, 600])},\n    )\n"
            f"    try:\n        yield\n    finally:\n        await app.state.pool.close()\n\n\n"
            f"app = FastAPI(lifespan=lifespan)\n```\n\n"
            f"`create_pool` is a coroutine — it must be awaited. {rng.choice(ALT_POOL)}.",
        )

        add(
            "apg_acquire",
            f"run a query against {t} using the pool",
            f"```python\nasync def {rng.choice(FN)}(pool: asyncpg.Pool, limit: int = {k}):\n"
            f"    async with pool.acquire() as conn:\n"
            f"        return await conn.fetch(\n"
            f'            "SELECT id, title FROM {t} WHERE {filt[0]} LIMIT $2",\n'
            f"            {filt[1]},\n            limit,\n        )\n```\n\n"
            f"{rng.choice(ALT_ACQUIRE)}. `fetch` gives you a list of `Record`; use `fetchrow` "
            f"for one row and `fetchval` for a single scalar.\n\n"
            f"**Not `conn.execute()` then a cursor** — in asyncpg `execute` returns a "
            f"status string, not a result set. There is no `cursor` object to fetch from.",
        )

        add(
            "apg_params",
            f"pass user input into a query on {t} safely",
            f"```python\nasync with pool.acquire() as conn:\n"
            f"    rows = await conn.fetch(\n"
            f'        "SELECT id FROM {t} WHERE owner_id = $1 AND status = ANY($2)",\n'
            f'        owner_id,\n        ["open", "pending"],\n    )\n```\n\n'
            f"{rng.choice(ALT_PARAM)}.\n\n"
            f"**Not `%s`** — that is psycopg2's style. Mixing the two in one query is a "
            f"syntax error, and f-stringing the value in is an injection.",
        )

        add(
            "apg_pgvector_search",
            f"find the {k} most similar rows in {t} by {opname}",
            f"```python\nasync def similar(pool: asyncpg.Pool, query_vec: list[float], "
            f"k: int = {k}):\n    async with pool.acquire() as conn:\n"
            f"        return await conn.fetch(\n"
            f'            """\n            SELECT id, title, {vc} {op} $1 AS distance\n'
            f"            FROM {t}\n            ORDER BY {vc} {op} $1\n"
            f'            LIMIT $2\n            """,\n'
            f"            str(query_vec),\n            k,\n        )\n```\n\n"
            f"`{op}` is **{opname}**, not similarity — smaller is closer, so `ORDER BY` "
            f"ascending is correct. For cosine similarity report `1 - (a <=> b)`.\n\n"
            f"Order by the operator expression itself, not by the aliased column: only "
            f"the operator form can use the index.",
        )

        add(
            "apg_pgvector_index",
            f"make similarity search on {t}.{vc} use an index",
            f"```sql\nCREATE EXTENSION IF NOT EXISTS vector;\n\n"
            f"ALTER TABLE {t} ADD COLUMN {vc} vector({dim});\n\n"
            f"CREATE INDEX CONCURRENTLY {t}_{vc}_hnsw\n"
            f"    ON {t} USING hnsw ({vc} {opclass})\n"
            f"    WITH (m = 16, ef_construction = 64);\n```\n\n"
            f"```sql\nSET hnsw.ef_search = {rng.choice([40, 64, 100])};\n```\n\n"
            f"The operator class has to match the operator you query with — an "
            f"`{opclass}` index does nothing for a query using a different operator.\n\n"
            f"**Not a separate vector database** — filtering by tenant, status or date "
            f"alongside the vector is one index scan here, and a two-system join anywhere else.",
        )

        add(
            "apg_transaction",
            f"insert into {t} and its audit table atomically",
            f"```python\nasync with pool.acquire() as conn:\n"
            f"    async with conn.transaction():\n"
            f"        row_id = await conn.fetchval(\n"
            f'            "INSERT INTO {t} (title, owner_id) VALUES ($1, $2) RETURNING id",\n'
            f"            title,\n            owner_id,\n        )\n"
            f"        await conn.execute(\n"
            f'            "INSERT INTO {t}_audit (row_id, action) VALUES ($1, $2)",\n'
            f'            row_id,\n            "created",\n        )\n```\n\n'
            f"{rng.choice(ALT_TXN)}. `RETURNING id` saves the extra round trip a "
            f"select-after-insert would cost.",
        )

        add(
            "apg_copy_bulk",
            f"load {rng.choice(['50k', '200k', '1M'])} rows into {t} quickly",
            f"```python\nasync with pool.acquire() as conn:\n"
            f"    await conn.copy_records_to_table(\n"
            f'        "{t}",\n        records=rows,\n'
            f'        columns=["id", "title", "owner_id"],\n    )\n```\n\n'
            f"{rng.choice(ALT_COPY)}.\n\n"
            f"**Not `executemany` in a loop** — that is still one statement per record, "
            f"and at this size it is the difference between seconds and many minutes.",
        )

        add(
            "apg_no_sync_driver",
            f"query {t} from an async handler",
            f"```python\nasync with pool.acquire() as conn:\n"
            f'    row = await conn.fetchrow("SELECT * FROM {t} WHERE id = $1", row_id)\n```\n\n'
            f"**Not psycopg2** — {rng.choice(ALT_SYNC)}. asyncpg is native async; psycopg3 in "
            f"async mode is the other valid choice.",
        )

        add(
            "apg_scalar",
            f"get a single count from {t} without unpacking a row",
            f"```python\nasync with pool.acquire() as conn:\n"
            f"    total = await conn.fetchval(\n"
            f'        "SELECT count(*) FROM {t} WHERE {filt[0]}",\n        {filt[1]},\n    )\n```\n\n'
            f"`fetchval` returns the first column of the first row directly. "
            f"{rng.choice(['`fetchrow` gives the whole Record when you need more than one column', 'reach for `fetchrow` when you want several columns from a single row', 'if you need more than one column, `fetchrow` returns the Record'])}.",
        )

        add(
            "apg_prepared",
            f"reuse one hot query against {t} across many calls",
            f"```python\nasync with pool.acquire() as conn:\n"
            f"    stmt = await conn.prepare(\n"
            f'        "SELECT id, title FROM {t} WHERE {filt[0]} LIMIT $2"\n    )\n'
            f"    for batch in batches:\n"
            f"        rows = await stmt.fetch({filt[1]}, {k})\n```\n\n"
            f"{rng.choice(['A prepared statement is parsed and planned once, then executed with new parameters', 'preparing once means the parse and plan happen a single time and every execution reuses them', 'the statement is planned on first prepare; each later fetch only sends parameters'])}. "
            f"asyncpg also caches prepared statements per connection automatically, so "
            f"this is worth doing explicitly only when you hold the connection yourself.",
        )

        add(
            "apg_jsonb",
            f"read a JSONB column from {t} as a Python dict",
            f"```python\nimport json\n\nasync with pool.acquire() as conn:\n"
            f'    await conn.set_type_codec(\n        "jsonb",\n'
            f"        encoder=json.dumps,\n        decoder=json.loads,\n"
            f'        schema="pg_catalog",\n    )\n'
            f"    row = await conn.fetchrow(\n"
            f'        "SELECT metadata FROM {t} WHERE id = $1", row_id\n    )\n'
            f'    meta = row["metadata"]\n```\n\n'
            f"{rng.choice(['Without a codec asyncpg hands back the raw JSON string', 'asyncpg returns jsonb as text unless you register a codec for it', 'the default is a str; the codec is what turns it into a dict'])}. "
            f"Register it on the pool with `init=` so every connection gets it.",
        )

    return out


def report(rows: list[dict]) -> None:
    ans = [r["messages"][1]["content"] for r in rows]

    def norm(s: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9\s]", " ", re.sub(r"\d+", "0", s.lower())).split())

    print(f"  records          {len(rows)}")
    print(f"  unique questions {len(set(r['messages'][0]['content'] for r in rows)) / len(rows):6.1%}")
    print(f"  unique answers   {len(set(norm(a) for a in ans)) / len(rows):6.1%}")
    print(f"  emits code       {sum(bool(re.search(r'```(python|sql)', a)) for a in ans) / len(rows):6.1%}")
    print(
        f"  '**Not' share    {sum('**Not ' in a for a in ans) / len(rows):6.1%}   "
        f"(target ~35%, NOT python_modern's 76%)"
    )
    print("\n  families:")
    for f, c in Counter(r["meta"]["family"] for r in rows).most_common():
        print(f"    {f:26s} {c:5d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--n-per", type=int, default=120)
    ap.add_argument("--cap", type=int, default=60, help="max records per family")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = build(random.Random(args.seed), args.n_per)
    # Dedup, then CAP per family. Without the cap the two highest-variety families
    # supplied 60% of the survivors and dragged the rejection share to 72% -- the
    # ratio that turned python_modern into a template emitter.
    seen, uniq, per_fam = set(), [], Counter()
    for r in rows:
        k = " ".join(re.sub(r"[^a-z0-9\s]", " ", re.sub(r"\d+", "0", r["messages"][1]["content"].lower())).split())
        fam = r["meta"]["family"]
        if k in seen or per_fam[fam] >= args.cap:
            continue
        seen.add(k)
        per_fam[fam] += 1
        uniq.append(r)
    print("=" * 78)
    print(" POSTGRESQL: asyncpg driver surface + pgvector, correctly parameterised")
    print("=" * 78)
    print(f"  generated {len(rows)}, {len(uniq)} survive answer-dedup ({len(uniq) / len(rows):.1%})\n")
    rows = uniq
    report(rows)

    if args.write:
        out = REPO_ROOT / "apps/factory/data/postgresql/training_data_asyncpg.jsonl"
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"\n  WROTE {out.relative_to(REPO_ROOT)}")
    else:
        print("\n  (dry run -- pass --write)")


if __name__ == "__main__":
    main()
