"""Disposition corpora for every domain: teach the model WHAT TO REACH FOR.

THE SHAPE, AND WHY IT DIFFERS FROM WHAT WE HAD
----------------------------------------------
The existing corpora teach capability -- the question names the answer:

    Q: Convert a legacy SERIAL primary key to GENERATED ALWAYS AS IDENTITY and
       explain the advantage.

A model can answer that without ever choosing anything. Real requests do not arrive
pre-solved. So every record here is:

    SITUATION (names no tool, no construct)
       -> the RIGHT approach, as code
       -> the COMMON WRONG approach, named and rejected, with the reason

The rejection half is the part that carries the bias. "Use pgvector" teaches a fact;
"use pgvector, NOT a separate vector database, because a join against your own rows
is free and a network hop is not" teaches a preference -- and a preference is what
survives into a situation the corpus never showed.

    astral         -> uv/ruff/ty commands            (build_astral_commands.py)
    python_modern  -> a STYLE of writing Python, not Python syntax
    python_web     -> FastAPI over Django/Flask, plus service best practice
    duckdb         -> reach for DuckDB on file/analytical work instead of pandas
    postgresql     -> antipatterns + modern patterns (pgvector, JSONB, IDENTITY)

    uv run python scripts/corpus/build_disposition_corpus.py
    uv run python scripts/corpus/build_disposition_corpus.py --write
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

from runtime_common.canon import REPO_ROOT

MARK = "\n\n### Answer:\n"

PHRASE_TRAIN = [
    "{s}",
    "{s} What's the right approach?",
    "How should I handle this: {s}",
    "{s} How would you do it?",
    "Working on this: {s} Suggestions?",
    "{s} What do you recommend?",
    "{s} Show me how.",
    "{s} What's the idiomatic way?",
    "{s} Best way to do this?",
    "Question for you: {s}",
    "{s} Is there a better way?",
    "{s} How do people normally solve this?",
    "{s} What should I write?",
    "Stuck on this. {s}",
]
PHRASE_EVAL = ["A colleague asked me about this: {s} Thoughts?", "{s} What would you reach for?"]

# Situational context. Without it each situation repeats verbatim across
# instances -- the duplication defect the audit found in financial (79.9% dup
# answers) and duckdb (51.0%), and which the first build of this file reproduced
# at 26.6% unique questions.
CTX_TRAIN = [
    "",
    "",
    " This is in a service that handles a few thousand requests a second.",
    " It's a small internal tool, so simplicity matters more than throughput.",
    " We're migrating an older codebase and want to stop repeating the old pattern.",
    " This runs in CI on every commit.",
    " The team is new to this stack, so readability matters.",
    " It's part of a nightly batch job.",
    " This is going into a library other teams depend on.",
    " We got burned by this in production last quarter.",
    " Code review flagged it and I want to understand the reasoning.",
    " The data volume has roughly tripled since this was written.",
]
CTX_EVAL = [
    " This is for a proof of concept we need to demo next week.",
    " It sits behind an internal admin panel used by a handful of people.",
]

# (family, situation, right-code, wrong-approach, why-not)
BANKS: dict[str, list[tuple[str, str, str, str, str]]] = {
    # ------------------------------------------------------------ python_modern
    "python_modern": [
        (
            "style_pathlib",
            "I need to build a file path, check it exists, and read it.",
            "```python\nfrom pathlib import Path\n\np = Path(base) / 'data' / 'in.csv'\nif p.exists():\n    text = p.read_text()\n```",
            "`os.path.join` + `open()`",
            "`pathlib` carries the path as an object rather than a string, so joins, suffix changes and existence checks compose instead of nesting.",
        ),
        (
            "style_dataclass",
            "I'm passing a dict of user fields around between functions and typos keep slipping through.",
            "```python\nfrom dataclasses import dataclass\n\n@dataclass(frozen=True, slots=True)\nclass User:\n    id: int\n    email: str\n    active: bool = True\n```",
            "a plain `dict`",
            "a dict has no schema, so a typo is a runtime `KeyError` instead of a type error. `frozen=True` makes it hashable and safe to share; `slots=True` cuts memory.",
        ),
        (
            "style_comprehension",
            "I'm building a list by looping and appending, filtering as I go.",
            "```python\nactive = [u.email for u in users if u.active]\n```",
            "`result = []` then `for ... : result.append(...)`",
            "the comprehension states the transformation as one expression, so there is no partially-built list to reason about.",
        ),
        (
            "style_enumerate",
            "I need both the index and the value while iterating.",
            "```python\nfor i, row in enumerate(rows, start=1):\n    ...\n```",
            "`for i in range(len(rows))` then `rows[i]`",
            "`range(len(...))` indexes a second time on every access and breaks on any non-sequence iterable.",
        ),
        (
            "style_cache",
            "The same expensive lookup runs repeatedly with the same arguments.",
            "```python\nfrom functools import cache\n\n@cache\ndef resolve(sym: str) -> Def:\n    ...\n```",
            "a module-level `_memo = {}` dict",
            "a hand-rolled memo dict is not thread-safe, never evicts, and has to be cleared by hand in tests. `@cache` gives you `.cache_clear()`.",
        ),
        (
            "style_contextmanager",
            "I need setup and guaranteed teardown around a block.",
            "```python\nfrom contextlib import contextmanager\n\n@contextmanager\ndef span(name: str):\n    t = start(name)\n    try:\n        yield t\n    finally:\n        t.close()\n```",
            "`try:` / `finally:` repeated at each call site",
            "the context manager writes the teardown once. Repeating `finally` is how one call site ends up missing it.",
        ),
        (
            "style_match",
            "I'm dispatching on the shape of an incoming event payload.",
            "```python\nmatch event:\n    case {'type': 'click', 'pos': (x, y)}:\n        ...\n    case {'type': 'key', 'code': str(c)}:\n        ...\n    case _:\n        raise ValueError(event)\n```",
            "a chain of `if event['type'] == ...` checks",
            "`match` destructures and validates in one step, and the `case _` arm makes the unhandled path explicit instead of silently falling through.",
        ),
        (
            "style_protocol",
            "I want to accept anything with a `.read()` method without forcing a base class.",
            "```python\nfrom typing import Protocol\n\nclass Readable(Protocol):\n    def read(self) -> bytes: ...\n\ndef load(src: Readable) -> bytes:\n    return src.read()\n```",
            "an ABC that callers must inherit from",
            "a Protocol is structural: existing types satisfy it without modification, so you can accept third-party objects you do not control.",
        ),
        (
            "style_taskgroup",
            "I need to run several async calls concurrently and fail if any fails.",
            "```python\nimport asyncio\n\nasync with asyncio.TaskGroup() as tg:\n    a = tg.create_task(fetch(x))\n    b = tg.create_task(fetch(y))\n```",
            "`asyncio.gather(...)` without `return_exceptions`",
            "`TaskGroup` cancels siblings when one task raises. Bare `gather` leaves the others running and orphaned.",
        ),
        (
            "style_no_mutable_default",
            "My function takes an optional list argument and callers report state leaking between calls.",
            "```python\ndef add(item, into: list | None = None) -> list:\n    if into is None:\n        into = []\n    into.append(item)\n    return into\n```",
            "`def add(item, into=[])`",
            "a mutable default is evaluated once at definition, so every call shares the same list. This is the leak.",
        ),
        (
            "style_strenum",
            "I have a set of fixed string states passed around as raw strings.",
            "```python\nfrom enum import StrEnum\n\nclass State(StrEnum):\n    PENDING = 'pending'\n    DONE = 'done'\n```",
            "bare string constants",
            "`StrEnum` still compares equal to its string value, so serialisation is unchanged, but typos become attribute errors at import time.",
        ),
        (
            "style_itertools",
            "I need to process a large file in fixed-size chunks without loading it all.",
            "```python\nfrom itertools import batched\n\nfor chunk in batched(rows, 1000):\n    handle(chunk)\n```",
            "slicing a fully-materialised list",
            "`batched` consumes the iterator lazily, so peak memory is one chunk rather than the whole file.",
        ),
    ],
    # -------------------------------------------------------------- python_web
    "python_web": [
        (
            "web_choose_fastapi",
            "I need to expose a few JSON endpoints over an existing async data layer.",
            "```python\nfrom fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/items/{item_id}')\nasync def read_item(item_id: int) -> Item:\n    return await store.get(item_id)\n```",
            "Django or Flask",
            "Django brings an ORM, admin and template layer you will not use, and Flask is sync-first so an async data layer would block its worker. FastAPI is async-native and generates the OpenAPI schema from the annotations.",
        ),
        (
            "web_pydantic_model",
            "Request bodies arrive as dicts and I'm validating fields by hand.",
            "```python\nfrom pydantic import BaseModel, Field\n\nclass CreateOrder(BaseModel):\n    sku: str\n    qty: int = Field(gt=0)\n\n@app.post('/orders')\nasync def create(order: CreateOrder) -> OrderOut:\n    ...\n```",
            "reading `request.json()` and checking keys manually",
            "the model validates, coerces and documents in one declaration. Hand-written checks drift from the docs immediately.",
        ),
        (
            "web_response_model",
            "My endpoint returns the ORM object directly and internal fields are leaking to clients.",
            "```python\n@app.get('/users/{uid}', response_model=UserPublic)\nasync def get_user(uid: int):\n    return await db.user(uid)\n```",
            "returning the ORM row and hoping it serialises safely",
            "`response_model` filters the output to declared fields, so a new internal column cannot silently become public.",
        ),
        (
            "web_depends",
            "Several endpoints need a database session and I'm reaching for a module-level global.",
            "```python\nfrom fastapi import Depends\n\nasync def get_db():\n    async with Session() as s:\n        yield s\n\n@app.get('/x')\nasync def handler(db = Depends(get_db)):\n    ...\n```",
            "a global session object",
            "`Depends` scopes the session per request and makes it trivially overridable in tests. A global session is shared across concurrent requests.",
        ),
        (
            "web_no_blocking",
            "One async endpoint makes a `requests.get()` call and throughput collapses under load.",
            "```python\nimport httpx\n\n@app.get('/proxy')\nasync def proxy():\n    async with httpx.AsyncClient() as c:\n        r = await c.get(url)\n    return r.json()\n```",
            "`requests.get()` inside `async def`",
            "a sync call blocks the event loop, so every other request on that worker stalls. If a library is sync-only, push it to `run_in_threadpool`.",
        ),
        (
            "web_lifespan",
            "I need to open a connection pool at startup and close it at shutdown.",
            "```python\nfrom contextlib import asynccontextmanager\n\n@asynccontextmanager\nasync def lifespan(app: FastAPI):\n    app.state.pool = await make_pool()\n    yield\n    await app.state.pool.close()\n\napp = FastAPI(lifespan=lifespan)\n```",
            "`@app.on_event('startup')`",
            "the event decorators are deprecated, and `lifespan` keeps startup and its matching shutdown in one function instead of two that can drift apart.",
        ),
        (
            "web_router",
            "The single app file has grown past a thousand lines of endpoints.",
            "```python\nfrom fastapi import APIRouter\n\nrouter = APIRouter(prefix='/orders', tags=['orders'])\n\n@router.get('/')\nasync def list_orders(): ...\n\napp.include_router(router)\n```",
            "adding more `@app.get` decorators to one module",
            "`APIRouter` splits by resource and gives each group its own prefix, dependencies and tags.",
        ),
        (
            "web_httpexception",
            "Handlers return `{'error': ...}` dicts with a 200 status when something goes wrong.",
            "```python\nfrom fastapi import HTTPException\n\nif item is None:\n    raise HTTPException(status_code=404, detail='item not found')\n```",
            "returning an error dict with HTTP 200",
            "clients and proxies route on the status code. A 200 with an error body means retries, caches and monitoring all treat the failure as success.",
        ),
        (
            "web_background",
            "The request has to send an email, and the client is waiting for it.",
            "```python\nfrom fastapi import BackgroundTasks\n\n@app.post('/signup')\nasync def signup(bg: BackgroundTasks):\n    bg.add_task(send_welcome, user.email)\n    return {'ok': True}\n```",
            "spawning a raw `threading.Thread`",
            "`BackgroundTasks` runs after the response is sent and is tied to the request lifecycle. A bare thread outlives it and is invisible to shutdown.",
        ),
        (
            "web_async_driver",
            "The service is async but the Postgres client is psycopg2.",
            "```python\nimport asyncpg\n\npool = await asyncpg.create_pool(dsn, min_size=2, max_size=10)\nrows = await pool.fetch('SELECT ...')\n```",
            "psycopg2 inside `async def`",
            "psycopg2 is blocking. Under concurrency it serialises the whole worker -- use asyncpg, or psycopg 3 in async mode.",
        ),
    ],
    # ------------------------------------------------------------------ duckdb
    "duckdb": [
        (
            "dd_reach_files",
            "I have a directory of Parquet files partitioned by date and need revenue per category for last quarter.",
            "```sql\nSELECT category, sum(revenue) AS rev\nFROM read_parquet('data/**/*.parquet', hive_partitioning := true)\nWHERE dt >= DATE '2026-04-01'\nGROUP BY ALL;\n```",
            "globbing the files and concatenating DataFrames in pandas",
            "the scan pushes the partition filter and column projection down to the files, so only the needed row groups are read. pandas loads every file fully into memory first.",
        ),
        (
            "dd_bigger_than_memory",
            "A 20GB CSV will not fit in memory and I need per-segment summary statistics.",
            "```sql\nSELECT segment, count(*), avg(value), median(value)\nFROM read_csv('big.csv')\nGROUP BY ALL;\n```",
            "`pd.read_csv(..., chunksize=...)` and manual aggregation per chunk",
            "the engine streams the file and spills to disk if needed. Chunked aggregation means reimplementing the group-by yourself and getting the merge step wrong for medians.",
        ),
        (
            "dd_exclude",
            "This table has 90 columns and I need all of them except two internal ones.",
            "```sql\nSELECT * EXCLUDE (internal_id, trace_id) FROM events;\n```",
            "typing out the other 88 column names",
            "the enumerated list silently goes stale the moment a column is added.",
        ),
        (
            "dd_columns_regex",
            "I want to sum every column whose name ends in `_amount`.",
            "```sql\nSELECT sum(COLUMNS('.*_amount$')) FROM ledger;\n```",
            "writing one `sum()` per column",
            "`COLUMNS` applies the aggregate across a matched set, so new matching columns are picked up automatically.",
        ),
        (
            "dd_qualify",
            "I need the top 3 rows per category by revenue.",
            "```sql\nSELECT *\nFROM sales\nQUALIFY row_number() OVER (PARTITION BY category ORDER BY revenue DESC) <= 3;\n```",
            "wrapping the window function in a subquery just to filter on it",
            "`QUALIFY` filters on a window result directly -- the subquery exists only to give the window a name.",
        ),
        (
            "dd_attach_pg",
            "I need to join a local CSV against a table that lives in Postgres.",
            "```sql\nINSTALL postgres; LOAD postgres;\nATTACH 'dbname=app host=localhost' AS pg (TYPE POSTGRES);\n\nSELECT c.*, u.plan\nFROM read_csv('signups.csv') c\nJOIN pg.public.users u USING (user_id);\n```",
            "exporting the Postgres table to CSV, or loading both into pandas",
            "the attach queries Postgres in place, so there is no export step to schedule and no stale copy to reconcile.",
        ),
        (
            "dd_polars_zerocopy",
            "I want the query result in a Polars DataFrame without an extra copy.",
            "```python\ndf = duckdb.sql('SELECT * FROM read_parquet(\"x.parquet\")').pl()\n```",
            "`.df()` to pandas then `pl.from_pandas(...)`",
            "`.pl()` hands over Arrow buffers directly. The pandas round-trip materialises the whole frame twice and loses the Arrow types.",
        ),
        (
            "dd_sample",
            "I want to eyeball a random 1% of a very large table.",
            "```sql\nSELECT * FROM events USING SAMPLE 1%;\n```",
            "`WHERE random() < 0.01`",
            "the predicate still scans every row. `USING SAMPLE` samples during the scan.",
        ),
        (
            "dd_group_by_all",
            "I'm grouping by six columns and keep forgetting to keep the GROUP BY list in sync with the SELECT.",
            "```sql\nSELECT region, channel, sku, dt, currency, tier, sum(rev)\nFROM sales\nGROUP BY ALL;\n```",
            "restating all six columns in the GROUP BY",
            "`GROUP BY ALL` derives the grouping from the non-aggregated select list, so the two cannot drift.",
        ),
    ],
    # -------------------------------------------------------------- postgresql
    "postgresql": [
        (
            "pg_reach_pgvector",
            "I have a few million documents and need to retrieve them by semantic similarity.",
            "```sql\nCREATE EXTENSION IF NOT EXISTS vector;\nALTER TABLE docs ADD COLUMN embedding vector(768);\nCREATE INDEX ON docs USING hnsw (embedding vector_cosine_ops);\n\nSELECT id, body FROM docs ORDER BY embedding <=> $1 LIMIT 10;\n```",
            "adding a dedicated vector database alongside Postgres",
            "a separate store means a second system to keep in sync and no way to filter or join against your own rows in the same query. `WHERE tenant_id = $2 ORDER BY embedding <=> $1` is one index scan here and a two-system dance anywhere else.",
        ),
        (
            "pg_upsert",
            "I'm inserting a batch where some rows may already exist.",
            "```sql\nINSERT INTO items (sku, qty) VALUES ($1, $2)\nON CONFLICT (sku) DO UPDATE SET qty = excluded.qty;\n```",
            "SELECT first, then INSERT or UPDATE from the application",
            "the check-then-write pattern is a race: two workers both see 'missing' and both insert. `ON CONFLICT` resolves it inside one statement.",
        ),
        (
            "pg_identity",
            "I need an auto-incrementing primary key on a new table.",
            "```sql\nCREATE TABLE events (\n    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,\n    payload jsonb NOT NULL\n);\n```",
            "`SERIAL`",
            "`SERIAL` is a legacy pseudo-type that creates a loosely-owned sequence and lets callers insert explicit ids, which desynchronises the sequence and produces duplicate-key errors later. `GENERATED ALWAYS` refuses them.",
        ),
        (
            "pg_not_in_null",
            "My `NOT IN (SELECT ...)` filter silently returns zero rows sometimes.",
            "```sql\nSELECT * FROM a\nWHERE NOT EXISTS (SELECT 1 FROM b WHERE b.a_id = a.id);\n```",
            "`WHERE id NOT IN (SELECT a_id FROM b)`",
            "if the subquery yields a single NULL, `NOT IN` is never true and the whole result is empty. `NOT EXISTS` has no NULL semantics problem and usually plans as an anti-join.",
        ),
        (
            "pg_sargable",
            "A query filtering on a date column got slow and the index is not being used.",
            "```sql\nCREATE INDEX ON orders (created_at);\nSELECT * FROM orders WHERE created_at >= $1 AND created_at < $2;\n```",
            "`WHERE date(created_at) = $1` or `WHERE created_at::date = $1`",
            "wrapping the column in a function makes the predicate non-sargable, so the b-tree cannot be used. Keep the column bare and put the range on the bounds -- or index the expression itself.",
        ),
        (
            "pg_jsonb",
            "Attributes vary per product type and I was going to build an attribute table.",
            "```sql\nALTER TABLE products ADD COLUMN attrs jsonb NOT NULL DEFAULT '{}';\nCREATE INDEX ON products USING gin (attrs jsonb_path_ops);\n\nSELECT * FROM products WHERE attrs @> '{\"color\":\"red\"}';\n```",
            "an entity-attribute-value table",
            "EAV needs a self-join per attribute and loses types. `jsonb` with a GIN index keeps the containment query to one index scan.",
        ),
        (
            "pg_queue_skip_locked",
            "Several workers pull jobs from a table and keep grabbing the same row.",
            "```sql\nUPDATE jobs SET state = 'running'\nWHERE id = (\n    SELECT id FROM jobs WHERE state = 'queued'\n    ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1\n)\nRETURNING *;\n```",
            "SELECT a job, then UPDATE it in a second statement",
            "without `SKIP LOCKED` the workers serialise on the same row or double-process it. This claims and returns the job atomically.",
        ),
        (
            "pg_partition",
            "A time-series table has hundreds of millions of rows and range queries and deletes are both slow.",
            "```sql\nCREATE TABLE events (id bigint, ts timestamptz NOT NULL, ...)\n  PARTITION BY RANGE (ts);\nCREATE TABLE events_2026_04 PARTITION OF events\n  FOR VALUES FROM ('2026-04-01') TO ('2026-05-01');\nCREATE INDEX ON events USING brin (ts);\n```",
            "moving to a different database, or sharding in the application",
            "range partitioning prunes whole partitions at plan time and turns retention into `DROP TABLE` instead of a long `DELETE`. BRIN suits naturally-ordered timestamps at a fraction of a b-tree's size.",
        ),
        (
            "pg_count_filter",
            "I need counts of several mutually-exclusive states and I'm running one query per state.",
            "```sql\nSELECT\n  count(*) FILTER (WHERE state = 'active')   AS active,\n  count(*) FILTER (WHERE state = 'inactive') AS inactive\nFROM users;\n```",
            "one query per state, or aggregating in the application",
            "`FILTER` computes every bucket in a single pass over the table instead of one scan per state.",
        ),
        (
            "pg_explain",
            "A query that used to be fast degraded after the table grew and I'm guessing at causes.",
            "```sql\nEXPLAIN (ANALYZE, BUFFERS) SELECT ...;\n```",
            "adding timing logs around the call in application code",
            "application timing tells you it is slow, not why. `ANALYZE` gives actual vs estimated rows -- a large gap points at stale statistics -- and `BUFFERS` separates cache misses from CPU.",
        ),
        (
            "pg_text_search",
            "I need keyword search over a description column and was about to add Elasticsearch.",
            "```sql\nALTER TABLE items ADD COLUMN tsv tsvector\n  GENERATED ALWAYS AS (to_tsvector('english', description)) STORED;\nCREATE INDEX ON items USING gin (tsv);\n\nSELECT * FROM items WHERE tsv @@ websearch_to_tsquery('english', $1);\n```",
            "adding Elasticsearch for text search",
            "a generated `tsvector` column stays in sync automatically and the search joins against your existing rows. A second search cluster needs indexing pipelines and reconciliation.",
        ),
    ],
}


# Additional TRAIN situations, appended after the initial build so each domain
# covers more of its preference surface.
BANKS["python_modern"] += [
    (
        "style_zip_strict",
        "I'm iterating two lists in parallel and a length mismatch went unnoticed.",
        "```python\nfor name, score in zip(names, scores, strict=True):\n    ...\n```",
        "plain `zip(...)`",
        "bare `zip` stops at the shorter sequence and silently drops the tail. `strict=True` raises instead.",
    ),
    (
        "style_fstring",
        "I'm building log messages by concatenating strings and `.format()` calls.",
        "```python\nlogger.info('order %s failed after %d retries', order_id, n)\n```",
        "f-strings inside logging calls, or `'a' + str(b) + 'c'`",
        "the %-style form defers formatting until the record is actually emitted, so a filtered-out DEBUG line costs nothing. Concatenation also breaks on non-str types.",
    ),
    (
        "style_pathlib_glob",
        "I need every .json file under a directory tree.",
        "```python\nfor p in Path(root).rglob('*.json'):\n    ...\n```",
        "`os.walk` with manual suffix checks",
        "`rglob` yields Path objects directly, so there is no join step and no string suffix comparison to get wrong.",
    ),
    (
        "style_typed_dict_narrow",
        "A function returns different shapes depending on a flag argument.",
        "```python\nfrom typing import overload\n\n@overload\ndef load(raw: Literal[True]) -> bytes: ...\n@overload\ndef load(raw: Literal[False]) -> str: ...\n```",
        "returning `bytes | str` and letting callers check",
        "`@overload` lets the type checker pick the right branch at each call site instead of forcing an isinstance check downstream.",
    ),
    (
        "style_no_bare_except",
        "Errors are being swallowed somewhere and we cannot tell where.",
        "```python\ntry:\n    parse(payload)\nexcept (ValueError, KeyError) as exc:\n    logger.warning('bad payload: %s', exc)\n    raise\n```",
        "`except:` or `except Exception: pass`",
        "a bare except also catches `KeyboardInterrupt` and `SystemExit`, and swallowing without re-raising hides the failure from every caller.",
    ),
    (
        "style_generator",
        "A function builds a large list that the caller only iterates once.",
        "```python\ndef parse_rows(f) -> Iterator[Row]:\n    for line in f:\n        yield Row.from_line(line)\n```",
        "building and returning the whole list",
        "the generator holds one row at a time. Returning a list means peak memory scales with the file.",
    ),
    (
        "style_datetime_tz",
        "Timestamps compare incorrectly between services in different regions.",
        "```python\nfrom datetime import datetime, UTC\n\nnow = datetime.now(UTC)\n```",
        "`datetime.utcnow()`",
        "`utcnow()` returns a NAIVE datetime that claims no timezone, so comparing it against an aware one raises, and comparing two of them silently assumes local time.",
    ),
    (
        "style_slots_perf",
        "We hold millions of small objects in memory and the process keeps getting OOM-killed.",
        "```python\n@dataclass(slots=True)\nclass Point:\n    x: float\n    y: float\n```",
        "a normal class with a per-instance `__dict__`",
        "`slots=True` drops the per-instance dict, which is the dominant cost at that object count.",
    ),
]

BANKS["python_web"] += [
    (
        "web_pagination",
        "A list endpoint returns every row and the payload has grown to megabytes.",
        "```python\n@app.get('/items')\nasync def list_items(limit: int = Query(50, le=200), cursor: str | None = None):\n    ...\n```",
        "returning the full table and letting the client slice it",
        "an unbounded list endpoint is a denial-of-service waiting for the table to grow. Cap the limit server-side and paginate by cursor rather than offset.",
    ),
    (
        "web_settings",
        "Config is read from `os.environ` scattered through the modules.",
        "```python\nfrom pydantic_settings import BaseSettings\n\nclass Settings(BaseSettings):\n    database_url: str\n    log_level: str = 'INFO'\n\nsettings = Settings()\n```",
        "`os.environ['X']` at each use site",
        "the settings model validates every variable once at startup, so a missing or malformed value fails immediately instead of at the first request that touches it.",
    ),
    (
        "web_status_code",
        "A create endpoint returns 200 and clients cannot tell creation from update.",
        "```python\n@app.post('/items', status_code=201)\nasync def create(item: ItemIn) -> ItemOut:\n    ...\n```",
        "leaving the default 200 on a create",
        "201 with a Location is the contract for creation; clients and caches key off it.",
    ),
    (
        "web_health",
        "The orchestrator restarts the container while it is still warming up.",
        "```python\n@app.get('/healthz')\nasync def healthz():\n    return {'status': 'ok'}\n\n@app.get('/readyz')\nasync def readyz():\n    await pool.fetchval('SELECT 1')\n    return {'status': 'ready'}\n```",
        "a single endpoint used for both liveness and readiness",
        "liveness answers 'is the process alive', readiness answers 'can it serve'. Merging them means a slow dependency triggers a restart loop.",
    ),
    (
        "web_test_client",
        "Tests spin up a real server on a port and are flaky in CI.",
        "```python\nfrom httpx import ASGITransport, AsyncClient\n\nasync with AsyncClient(transport=ASGITransport(app=app), base_url='http://t') as c:\n    r = await c.get('/items')\n```",
        "starting uvicorn in a thread and hitting localhost",
        "the ASGI transport calls the app in-process: no port to collide, no startup race, and dependency overrides still apply.",
    ),
    (
        "web_dep_override",
        "Tests need a stub database and the code reaches for monkeypatching.",
        "```python\napp.dependency_overrides[get_db] = lambda: FakeDB()\n```",
        "monkeypatching the module-level session",
        "the override is scoped to the app and reverts cleanly. Monkeypatching leaks between tests when one forgets to undo it.",
    ),
    (
        "web_streaming",
        "An endpoint builds a 200MB CSV in memory before responding.",
        "```python\nfrom fastapi.responses import StreamingResponse\n\nreturn StreamingResponse(row_iter(), media_type='text/csv')\n```",
        "assembling the whole body then returning it",
        "streaming holds one chunk at a time, so memory is flat regardless of result size and the client starts receiving immediately.",
    ),
    (
        "web_no_orm_in_route",
        "Route handlers contain query construction and business rules inline.",
        "```python\n@app.get('/orders/{oid}')\nasync def get_order(oid: int, svc: OrderService = Depends(get_service)):\n    return await svc.fetch(oid)\n```",
        "putting the query and the rules directly in the handler",
        "the handler should translate HTTP to a call and back. Logic in the route cannot be tested or reused without going through HTTP.",
    ),
]

BANKS["duckdb"] += [
    (
        "dd_json_files",
        "I have a folder of newline-delimited JSON logs and need counts by level.",
        "```sql\nSELECT level, count(*)\nFROM read_json_auto('logs/*.ndjson')\nGROUP BY ALL;\n```",
        "looping the files in Python and parsing each line with `json.loads`",
        "the reader infers the schema once and scans in C++. Per-line `json.loads` in Python is the slowest possible path.",
    ),
    (
        "dd_pivot",
        "I need a wide report with one column per month.",
        "```sql\nPIVOT sales ON month USING sum(revenue) GROUP BY region;\n```",
        "`df.pivot_table(...)` after loading everything",
        "`PIVOT` runs inside the scan, so the wide result is materialised once rather than after a full load.",
    ),
    (
        "dd_asof",
        "I need to match each trade to the most recent quote at or before its timestamp.",
        "```sql\nSELECT *\nFROM trades t\nASOF JOIN quotes q ON t.sym = q.sym AND t.ts >= q.ts;\n```",
        "a correlated subquery, or `merge_asof` after loading both sides",
        "`ASOF JOIN` is a single ordered pass. The correlated subquery re-scans quotes per trade.",
    ),
    (
        "dd_export_parquet",
        "A downstream job needs the query result as Parquet.",
        "```sql\nCOPY (SELECT * FROM report) TO 'out.parquet' (FORMAT PARQUET, COMPRESSION ZSTD);\n```",
        "writing CSV, or round-tripping through pandas `to_parquet`",
        "`COPY` writes directly from the query result and keeps the column types. CSV loses them and costs a re-parse downstream.",
    ),
    (
        "dd_httpfs_s3",
        "The Parquet files live in S3 rather than on local disk.",
        "```sql\nINSTALL httpfs; LOAD httpfs;\nSET s3_region = 'eu-west-1';\n\nSELECT count(*) FROM read_parquet('s3://bucket/y=2026/**/*.parquet');\n```",
        "downloading the objects first with boto3",
        "the scan issues ranged GETs and reads only the row groups the filter needs. Downloading fetches whole objects.",
    ),
    (
        "dd_describe",
        "I have an unfamiliar Parquet file and need to know its schema and ranges.",
        "```sql\nDESCRIBE SELECT * FROM read_parquet('f.parquet');\nSUMMARIZE SELECT * FROM read_parquet('f.parquet');\n```",
        "loading it into pandas and calling `.info()` / `.describe()`",
        "`SUMMARIZE` reads column statistics rather than the data, so it answers immediately on files far larger than memory.",
    ),
]

BANKS["postgresql"] += [
    (
        "pg_uuid_v7",
        "I need globally unique primary keys but random UUIDs are hurting insert performance.",
        "```sql\nCREATE TABLE events (\n    id uuid PRIMARY KEY DEFAULT uuidv7(),\n    ...\n);\n```",
        "`uuid_generate_v4()` as the primary key",
        "v4 is random, so every insert lands in a different b-tree page and the index write amplifies. v7 is time-ordered and appends.",
    ),
    (
        "pg_timestamptz",
        "Timestamps written from two regions compare inconsistently.",
        "```sql\nALTER TABLE events ALTER COLUMN created_at TYPE timestamptz;\n```",
        "`timestamp` without time zone",
        "`timestamp` stores no offset, so the same instant written from two regions is stored as two different values. `timestamptz` normalises to UTC on write.",
    ),
    (
        "pg_index_concurrently",
        "I need an index on a large live table without blocking writes.",
        "```sql\nCREATE INDEX CONCURRENTLY idx_orders_created ON orders (created_at);\n```",
        "a plain `CREATE INDEX` during a maintenance window",
        "a plain create takes an exclusive lock for the whole build. `CONCURRENTLY` takes two shorter passes and lets writes continue.",
    ),
    (
        "pg_covering_index",
        "A hot query reads three columns and the index only helps the lookup.",
        "```sql\nCREATE INDEX ON orders (customer_id) INCLUDE (status, total);\n```",
        "adding the extra columns to the index key",
        "`INCLUDE` stores them as payload so the query is index-only, without widening the key or changing sort order.",
    ),
    (
        "pg_advisory_lock",
        "Two schedulers occasionally run the same nightly job.",
        "```sql\nSELECT pg_try_advisory_lock(hashtext('nightly-rollup'));\n```",
        "a `locks` table with an INSERT and a cleanup job",
        "the advisory lock is released automatically when the session ends, so a crashed worker cannot leave the job wedged.",
    ),
    (
        "pg_ctid_batch",
        "A one-shot UPDATE over 50 million rows holds a transaction open for hours.",
        "```sql\nWITH batch AS (\n  SELECT ctid FROM events WHERE migrated IS false LIMIT 10000\n)\nUPDATE events e SET migrated = true\nFROM batch b WHERE e.ctid = b.ctid;\n```",
        "one `UPDATE` covering the whole table",
        "the single statement bloats the table, blocks autovacuum, and cannot be resumed. Batching keeps each transaction short.",
    ),
]


# financial_planning. Not a tool-selection domain, but the strongest disposition
# domain we have: the wrong answers are famous ones, and the +0.83pp / -36.67pp
# measurement that made this expert look worthless was taken on a corpus that is
# 79.9% duplicate -- which says more about the data than the domain.
BANKS["financial_planning"] = [
    (
        "fin_match_first",
        "I have some spare income each month and I'm deciding where to put it. My employer matches retirement contributions up to 5%.",
        "Contribute at least enough to capture the full 5% employer match before anything else.",
        "opening a brokerage account first, or paying down a low-rate loan first",
        "the match is an immediate 100% return on the matched portion. No investment and no debt at a normal rate competes with that, so it is the first claim on the money.",
    ),
    (
        "fin_debt_order",
        "I'm carrying a credit card balance and also want to start investing.",
        "Clear the card first, then invest.",
        "investing while carrying high-interest revolving debt",
        "a card at 20% is a guaranteed 20% cost. Paying it is a risk-free return that no diversified portfolio reliably matches.",
    ),
    (
        "fin_emergency",
        "I want to put everything I have into the market to catch the upside.",
        "Hold three to six months of expenses in cash first, then invest the rest.",
        "investing the full balance with no cash buffer",
        "without a buffer the first unexpected expense forces a sale at whatever the market happens to be doing, which converts a temporary drop into a permanent loss.",
    ),
    (
        "fin_index",
        "I'm trying to decide which individual companies to buy for my retirement account.",
        "Buy a broad low-cost index fund covering the whole market.",
        "picking individual stocks, or an actively managed fund",
        "concentration adds risk that is not compensated with higher expected return, and fees compound against you. The decision that reliably matters is cost and diversification, not selection.",
    ),
    (
        "fin_expense_ratio",
        "Two funds track the same index. One charges 0.03% and one charges 0.80%, and the expensive one has better recent returns.",
        "Take the 0.03% fund.",
        "choosing on recent performance",
        "the fee is certain and compounds every year; recent outperformance between two funds tracking the same index is noise. Over decades the fee difference dominates.",
    ),
    (
        "fin_whole_life",
        "An adviser is recommending a permanent life policy as a way to build wealth tax-free.",
        "Buy term life for the coverage you need and invest the difference in low-cost index funds.",
        "permanent/whole life as an investment vehicle",
        "it bundles insurance with a high-fee investment and pays the seller a large commission. Separating the two gets you more coverage and better returns for less.",
    ),
    (
        "fin_timing",
        "The market looks expensive right now so I'm holding cash until it drops.",
        "Invest on a fixed schedule regardless of the level.",
        "waiting for a better entry point",
        "you need two correct calls -- when to exit and when to return -- and missing a handful of the strongest days accounts for most of the long-run difference in outcomes.",
    ),
    (
        "fin_401k_cashout",
        "I'm changing jobs and the old retirement account balance is small enough to just take as cash.",
        "Roll it into the new employer's plan or an IRA.",
        "cashing it out",
        "the withdrawal is taxed as income and usually carries an early-withdrawal penalty, and the balance stops compounding. A direct rollover costs nothing.",
    ),
    (
        "fin_tax_advantaged",
        "I'm saving for retirement and using a normal taxable brokerage account.",
        "Fill the tax-advantaged accounts first, then use taxable for the overflow.",
        "using taxable accounts while tax-advantaged space is unused",
        "the sheltered accounts remove the annual drag of tax on dividends and gains. The space does not carry forward -- unused contribution room is gone.",
    ),
    (
        "fin_roth_vs_trad",
        "I'm early in my career and deciding between pre-tax and after-tax retirement contributions.",
        "Favour the after-tax (Roth) option while your marginal rate is low.",
        "defaulting to pre-tax because it lowers this year's bill",
        "the comparison is your rate now versus your rate in retirement. Early-career income is usually the lowest it will be, so paying the tax now buys tax-free growth later.",
    ),
    (
        "fin_lifestyle",
        "I got a substantial raise and I'm looking at a bigger apartment and a newer car.",
        "Direct a fixed share of the raise to savings automatically before adjusting spending.",
        "letting spending expand to match the new income",
        "fixed commitments are hard to unwind, so raises absorbed by lifestyle permanently raise the number you need to retire while leaving your savings rate unchanged.",
    ),
    (
        "fin_diversify_employer",
        "Most of my net worth is in my employer's stock through grants and the discount plan.",
        "Sell down to a small share of net worth and reinvest in a broad fund.",
        "holding a concentrated position because you know the company",
        "your salary already depends on that company. Holding its stock too means one event takes both your income and your savings at the same time.",
    ),
]


# Entity names substituted consistently across question AND answer per instance.
# Without this each situation has ONE fixed answer repeated ~26 times (26.6% unique
# answers measured), which teaches the string rather than the preference. Varying
# the identifiers keeps the pattern identical while making every record textually
# distinct.
ENTITIES = {
    "docs": ["docs", "articles", "manuals", "tickets", "notes", "pages"],
    "orders": ["orders", "bookings", "invoices", "shipments", "claims"],
    "events": ["events", "telemetry", "audit_log", "clicks", "readings"],
    "users": ["users", "accounts", "members", "customers", "subscribers"],
    "items": ["items", "products", "parts", "listings", "assets"],
    "sales": ["sales", "revenue_lines", "transactions", "postings"],
    "jobs": ["jobs", "tasks", "work_items", "queue_entries"],
    "trades": ["trades", "fills", "executions"],
    "quotes": ["quotes", "ticks", "prices"],
}


def vary(text: str, rng: random.Random, mapping: dict[str, str]) -> str:
    for base, repl in mapping.items():
        if base != repl:
            text = re.sub(rf"\b{base}\b", repl, text)
    return text


def rec(domain: str, family: str, q: str, a: str) -> dict:
    return {
        "text": f"### Question:\n{q}{MARK}{a}",
        "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
        "meta": {"family": family, "domain": domain, "gen": "disposition_v1"},
    }


# THE DUPLICATION THAT MADE python_modern EMIT A DATACLASS ON EVERY QUESTION
# --------------------------------------------------------------------------
# vary() only substitutes ENTITY NAMES (docs -> articles). Most answers never
# contain one, so n_per=26 instances of a bank entry produced 26 BYTE-IDENTICAL
# answers. Measured on the shipped corpus: python_modern's 520 disposition records
# were 24 unique answers repeated ~22x each. The adapter did not see 520 examples
# of good judgment, it saw 24 -- and one of them was the frozen dataclass.
#
# So the rejection SURFACE is varied too. Varying it is not a loss of signal: the
# behaviour being taught is "name the thing you rejected and why", not the literal
# string "**Not ". A single literal is precisely what the adapter latched onto.
REJECT_FORMS = [
    "**Not {wrong}** — {why}",
    "**Not {wrong}.** {why}",
    "Avoid {wrong} here — {why}",
    "**{wrong} is the wrong reach** — {why}",
    "Worth saying what this is *not*: {wrong}. {why}",
    "**Not {wrong}** — {why} That is the whole reason to prefer the above.",
]
WHY_LEAD = ["", "", "", "The reason is simple: ", "Concretely: ", "In practice, "]


def build(domain: str, entries, phrases, n_per: int, rng: random.Random, ctxs: list[str]) -> list[dict]:
    out = []
    for _ in range(n_per):
        for fam, sit, right, wrong, why in entries:
            mapping = {b: rng.choice(opts) for b, opts in ENTITIES.items()}
            q = rng.choice(phrases).format(s=vary(sit, rng, mapping)) + rng.choice(ctxs)
            w = rng.choice(WHY_LEAD) + why
            rej = rng.choice(REJECT_FORMS).format(wrong=wrong, why=w)
            # Rejection usually follows the answer, but not always -- a fixed slot is
            # one more thing to memorise instead of learn.
            body = f"{rej}\n\n{right}" if rng.random() < 0.18 else f"{right}\n\n{rej}"
            out.append(rec(domain, fam, q, vary(body, rng, mapping)))
    return out


CODE = re.compile(r"```")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--n-per", type=int, default=26)
    ap.add_argument("--cap", type=int, default=40, help="max records per family")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 78)
    print(" DISPOSITION CORPORA -- situation -> right approach + rejected alternative")
    print("=" * 78)
    for domain, entries in BANKS.items():
        rng = random.Random(args.seed)
        train = build(domain, entries, PHRASE_TRAIN, args.n_per, rng, CTX_TRAIN)
        # held-out-PHRASING sample, reported only. The real eval set lives in
        # build_disposition_evals.py and uses held-out SITUATIONS, which is the
        # stronger test -- a preference is only shown when it fires on a case the
        # corpus never contained.
        ev = build(domain, entries, PHRASE_EVAL, 1, random.Random(args.seed + 99), CTX_EVAL)
        # Dedup on the SAME normalised key the merge uses, then cap per family.
        # Reporting a count that the merge then collapses is how 520 records became
        # 24 without anyone noticing.
        seen, uniq, per_fam = set(), [], Counter()
        for r in train:
            k = " ".join(re.sub(r"[^a-z0-9\s]", " ", re.sub(r"\d+", "0", r["messages"][1]["content"].lower())).split())
            fam = r["meta"]["family"]
            if k in seen or per_fam[fam] >= args.cap:
                continue
            seen.add(k)
            per_fam[fam] += 1
            uniq.append(r)
        raw_n = len(train)
        train = uniq
        uq = len({r["messages"][0]["content"] for r in train}) * 100.0 / len(train)
        code = sum(bool(CODE.search(r["messages"][1]["content"])) for r in train)
        rej = sum(
            bool(re.search(r"\*\*Not |Avoid |wrong reach|is \*not\*", r["messages"][1]["content"])) for r in train
        )
        overlap = len({r["messages"][0]["content"] for r in train} & {r["messages"][0]["content"] for r in ev})
        print(
            f"\n  {domain:14s} {len(train):5d} train (of {raw_n} generated, "
            f"{len(train) / raw_n:.0%} survive dedup) / {len(ev):3d} eval   "
            f"{len(entries)} situations"
        )
        print(
            f"    code in answer {code * 100.0 / len(train):5.1f}%   "
            f"names a rejected alternative {rej * 100.0 / len(train):5.1f}%   "
            f"unique questions {uq:5.1f}%   train/eval overlap {overlap}"
        )
        if args.write:
            d = REPO_ROOT / f"apps/factory/data/{domain}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "training_data_disposition.jsonl").write_text("\n".join(json.dumps(r) for r in train) + "\n")
            print(f"    WROTE apps/factory/data/{domain}/training_data_disposition.jsonl")

    if not args.write:
        print("\n  (dry run -- pass --write)")


if __name__ == "__main__":
    main()
