"""python_modern / python_web: capability records, to break the single-form monoculture.

THE DEFECT THIS DILUTES -- measured end to end
-----------------------------------------------
                     records   '**Not X**' shape   "Here is the ..." opener
    python_modern      681          76.4%                  23.6%
    python_web         609          76.8%                  23.2%
    postgresql        1368          32.3%                   1.8%
    duckdb            1184          32.9%                   0.0%
    astral            1610           0.0%                   2.3%

Three quarters of what those two adapters ever saw is one answer shape, generated
from 20 situations at ~26 instances each. docs/CORPUS_DESIGN.md states the rule this
violates -- a corpus teaches a FORM -- and the adapter now obeys it literally:

    Q: "Add ruff and ty as dev dependencies, then format and lint the whole codebase."
    A: ```python
       @dataclass(frozen=True, slots=True)
       class Point: ...
       **Not a normal class with mutable defaults** -- frozen=True makes it hashable

That exact sentence appears ZERO times in any corpus in this repo. It is not
regurgitation. python_modern learned the SHAPE `<code>\\n\\n**Not X** -- <why>` plus a
handful of content attractors, and now composes novel text into it for any question.
Deleting rows cannot fix a shape; only adding other shapes can.

WHAT THIS EMITS
---------------
Capability records -- situation to code, NO rejection clause, and deliberately varied
openers. At ~800 per domain the disposition share drops from ~76% to ~35%, which is
where postgresql and duckdb sit and neither of them does this.

    uv run python scripts/corpus/build_python_capability.py [--write]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

from runtime_common.canon import REPO_ROOT

ANSWER_MARKER = "\n\n### Answer:\n"
GEN = "python_capability_v1"

# --------------------------------------------------------------------------- pools
ENTITY = ["Invoice", "Shipment", "Reading", "Ticket", "Session", "Booking", "Payment",
          "Device", "Alert", "Subscription", "Batch", "Route"]
FIELD = ["amount", "quantity", "score", "latency_ms", "weight", "priority", "balance"]
NAME = ["records", "items", "rows", "entries", "payloads", "chunks", "events"]
ROOT = ["data", "exports", "uploads", "archive", "inbox", "staging"]
EXT = ["csv", "json", "parquet", "log", "xml", "yaml"]
SVC = ["billing", "catalog", "identity", "shipping", "metering", "notifications"]
CTX = ["", " in a batch job", " in a long-running worker", " in a CLI tool",
       " in a service that runs for weeks", " under a tight memory budget",
       " where the input can be empty", " in code three people maintain",
       " in a module that gets imported at startup", " in a hot path"]
PHRASE = ["{task}?", "How do I {task}?", "Need to {task}.", "{task} -- show me.",
          "What's the cleanest way to {task}?", "{task}. Code please.",
          "Can you {task}?", "Write the code to {task}.", "{task} -- what would you write?"]

# Openers. python_modern opens 23.6% of its answers with "Here is the ...", and the
# adapter now emits that string on questions about ruff. These are deliberately
# heterogeneous, and most of them are empty.
OPEN = ["", "", "", "", "", "", "",
        "Straightforward version:\n\n", "This is the shape I'd reach for:\n\n",
        "Short version:\n\n", "", "", ""]


def rec(domain: str, family: str, question: str, answer: str) -> dict:
    return {
        "text": f"### Question:\n{question}{ANSWER_MARKER}{answer}",
        "messages": [{"role": "user", "content": question},
                     {"role": "assistant", "content": answer}],
        "meta": {"family": family, "tool": domain, "gen": GEN},
    }


def build_modern(rng: random.Random, n_per: int) -> list[dict]:
    out: list[dict] = []

    def add(fam: str, task: str, ans: str) -> None:
        q = rng.choice(PHRASE).format(task=task + rng.choice(CTX))
        out.append(rec("python_modern", fam, q[0].upper() + q[1:],
                       rng.choice(OPEN) + ans))

    for _ in range(n_per):
        E = rng.choice(ENTITY); f = rng.choice(FIELD); nm = rng.choice(NAME)
        root = rng.choice(ROOT); ext = rng.choice(EXT); low = E.lower()

        add("pm_pathlib_walk",
            f"collect every .{ext} under {root}/ and total their sizes",
            f"```python\nfrom pathlib import Path\n\n\n"
            f"def total_bytes(root: Path) -> int:\n"
            f"    return sum(p.stat().st_size for p in root.rglob('*.{ext}'))\n\n\n"
            f"paths = sorted(Path('{root}').rglob('*.{ext}'))\n```\n\n"
            f"{rng.choice(['`rglob` walks recursively and yields `Path` objects, so `.stat()` and `.suffix` are there without a second import', '`rglob` returns Paths, which means no `os.path.join` and no string surgery downstream', 'each yielded item is a Path, so size, suffix and parent are attribute access rather than string parsing'])}.")

        add("pm_pathlib_rw",
            f"read a {ext} file and write a transformed copy next to it",
            f"```python\nfrom pathlib import Path\n\n\n"
            f"src = Path('{root}/{low}.{ext}')\n"
            f"dst = src.with_suffix('.clean.{ext}')\n\n"
            f"dst.write_text(src.read_text(encoding='utf-8').strip(), encoding='utf-8')\n```\n\n"
            f"{rng.choice(['`with_suffix` and `with_name` build the sibling path without touching separators', '`with_suffix` keeps the directory and stem and swaps only the extension', 'deriving the destination with `with_suffix` avoids rebuilding the path by hand'])}. "
            f"Pass `encoding` explicitly — the platform default is not the same everywhere.")

        add("pm_dataclass_domain",
            f"model a {E} that must not change after construction",
            f"```python\nfrom dataclasses import dataclass\nfrom datetime import datetime\n\n\n"
            f"@dataclass(frozen=True, slots=True)\nclass {E}:\n"
            f"    id: int\n    {f}: float\n    created_at: datetime\n\n"
            f"    @property\n    def is_large(self) -> bool:\n"
            f"        return self.{f} > {rng.choice([100, 500, 1000, 10_000])}\n```\n\n"
            f"{rng.choice(['`frozen=True` makes instances hashable so they work as dict keys and set members', '`slots=True` drops the per-instance `__dict__`, which matters once you hold millions of them', 'frozen gives you hashability and safe sharing across threads; slots cuts the memory per instance'])}.")

        add("pm_dataclass_kwonly",
            f"stop callers mixing up the positional arguments of {E}",
            f"```python\nfrom dataclasses import dataclass, field\n\n\n"
            f"@dataclass(frozen=True, slots=True, kw_only=True)\nclass {E}:\n"
            f"    id: int\n    {f}: float = 0.0\n"
            f"    tags: tuple[str, ...] = ()\n    meta: dict[str, str] = field(default_factory=dict)\n```\n\n"
            f"`kw_only=True` forces `{E}(id=1, {f}=2.0)` at every call site, and it also "
            f"lets a defaulted field precede a required one.\n\n"
            f"`field(default_factory=dict)` is required for mutable defaults — a bare "
            f"`{{}}` would be shared by every instance.")

        add("pm_match",
            f"dispatch on the shape of an incoming {low} message",
            f"```python\ndef handle(msg: dict) -> str:\n    match msg:\n"
            f"        case {{'type': 'created', 'id': int(id)}}:\n"
            f"            return f'{low} {{id}} created'\n"
            f"        case {{'type': 'updated', '{f}': float() | int() as v}}:\n"
            f"            return f'{f} now {{v}}'\n"
            f"        case {{'type': str(kind)}}:\n"
            f"            return f'unhandled: {{kind}}'\n"
            f"        case _:\n            raise ValueError(f'not a message: {{msg!r}}')\n```\n\n"
            f"{rng.choice(['The patterns destructure and type-check in one step -- no chain of `isinstance` plus `msg[\"id\"]`', 'each case binds and validates at once, which an if/elif chain has to do in two separate moves', 'capture patterns like `int(id)` check the type and bind the name together'])}.")

        add("pm_cache",
            f"avoid recomputing an expensive {low} lookup",
            f"```python\nfrom functools import cache\n\n\n"
            f"@cache\ndef {low}_rate(region: str, tier: str) -> float:\n"
            f"    return _load_table()[region][tier]\n```\n\n"
            f"{rng.choice(['`@cache` is unbounded; use `@lru_cache(maxsize=1024)` when the key space is open-ended', 'reach for `@lru_cache(maxsize=...)` instead if the arguments come from user input', 'unbounded caching is fine for a fixed key space and a leak for anything user-supplied'])}. "
            f"Arguments have to be hashable, which is one more reason the domain types "
            f"above are frozen.")

        add("pm_taskgroup",
            f"fetch {rng.choice([3, 5, 8])} {nm} concurrently and fail fast if any fails",
            f"```python\nimport asyncio\n\n\n"
            f"async def fetch_all(ids: list[int]) -> list[dict]:\n"
            f"    async with asyncio.TaskGroup() as tg:\n"
            f"        tasks = [tg.create_task(fetch_one(i)) for i in ids]\n"
            f"    return [t.result() for t in tasks]\n```\n\n"
            f"{rng.choice(['`TaskGroup` cancels the siblings when one task raises and re-raises as an ExceptionGroup', 'if one child fails the group cancels the rest -- no orphaned tasks running past the error', 'the group waits for every task and propagates failures as an ExceptionGroup instead of swallowing them'])}. "
            f"Catch it with `except* ValueError:` when you need to handle one kind.")

        add("pm_strenum",
            f"replace the bare string statuses on {E}",
            f"```python\nfrom enum import StrEnum\n\n\n"
            f"class {E}Status(StrEnum):\n    PENDING = 'pending'\n"
            f"    ACTIVE = 'active'\n    CLOSED = 'closed'\n\n\n"
            f"# compares equal to the string, and serialises as one\n"
            f"assert {E}Status.ACTIVE == 'active'\n"
            f"json.dumps({{'status': {E}Status.ACTIVE}})\n```\n\n"
            f"{rng.choice(['`StrEnum` members are real strings, so existing comparisons and JSON keep working while typos become AttributeError', 'you get the type checking of an enum without breaking any code that expects a str', 'a mistyped member fails at import; a mistyped string literal fails in production'])}.")

        add("pm_zip_strict",
            f"pair up {nm} with their computed {f} values",
            f"```python\npairs = list(zip({nm}, {f}s, strict=True))\n```\n\n"
            f"{rng.choice(['`strict=True` raises when the lengths differ instead of silently truncating to the shorter one', 'without `strict=True` a length mismatch just drops the tail and you get a quietly short result', 'the strict flag turns a silent data-loss bug into a ValueError at the zip'])}.")

        add("pm_utc",
            f"stamp each {E} with the time it was received",
            f"```python\nfrom datetime import UTC, datetime\n\n\n"
            f"received_at = datetime.now(UTC)\n"
            f"iso = received_at.isoformat()          # '2026-08-20T09:14:22.481+00:00'\n```\n\n"
            f"{rng.choice(['`datetime.now(UTC)` is timezone-aware; `utcnow()` returns a naive object that claims nothing about its zone and is deprecated', 'always attach the zone -- a naive UTC timestamp compares wrong against an aware one and raises', '`utcnow()` gives you a naive datetime holding UTC, which is the shape that causes the comparison bugs'])}.")

        add("pm_generator",
            f"process a {rng.choice(['2 GB', '10 GB', '40 GB'])} {ext} export without loading it",
            f"```python\nfrom collections.abc import Iterator\nfrom pathlib import Path\n\n\n"
            f"def parsed(path: Path) -> Iterator[dict]:\n"
            f"    with path.open(encoding='utf-8') as fh:\n"
            f"        for line in fh:\n            if line.strip():\n"
            f"                yield json.loads(line)\n\n\n"
            f"total = sum(r['{f}'] for r in parsed(Path('{root}/{low}.{ext}')))\n```\n\n"
            f"{rng.choice(['Memory stays flat at one record because nothing is materialised', 'the generator holds one row at a time, so peak memory is independent of file size', 'nothing accumulates -- the sum consumes the stream as it is produced'])}. "
            f"`itertools.islice` gives you a bounded batch when a downstream API wants one.")

        add("pm_contextmanager",
            f"make sure the {rng.choice(SVC)} client is closed even on error",
            f"```python\nfrom contextlib import contextmanager\nfrom collections.abc import Iterator\n\n\n"
            f"@contextmanager\ndef client(url: str) -> Iterator[Client]:\n"
            f"    c = Client(url)\n    try:\n        yield c\n    finally:\n        c.close()\n\n\n"
            f"with client(url) as c:\n    c.send(payload)\n```\n\n"
            f"{rng.choice(['The `finally` runs on the exception path too, which a plain close-after-use does not', 'wrapping the yield in try/finally is what makes it exception-safe', 'without the finally, an exception in the body leaks the connection'])}.")

        add("pm_protocol",
            f"accept anything that can persist a {E} without importing the class",
            f"```python\nfrom typing import Protocol\n\n\n"
            f"class {E}Store(Protocol):\n"
            f"    def save(self, item: {E}) -> int: ...\n"
            f"    def get(self, id: int) -> {E} | None: ...\n\n\n"
            f"def sync(store: {E}Store, items: list[{E}]) -> None:\n"
            f"    for item in items:\n        store.save(item)\n```\n\n"
            f"{rng.choice(['A Protocol is structural -- any object with those methods satisfies it, no base class and no registration', 'callers do not inherit from anything; matching the methods is enough', 'this keeps the test double independent of the production class entirely'])}.")

        add("pm_exception_group",
            f"report every invalid {low} in a batch instead of only the first",
            f"```python\ndef validate_all(items: list[dict]) -> None:\n    errors = []\n"
            f"    for i, item in enumerate(items):\n        try:\n"
            f"            validate(item)\n        except ValueError as exc:\n"
            f"            errors.append(ValueError(f'row {{i}}: {{exc}}'))\n"
            f"    if errors:\n        raise ExceptionGroup('invalid {low}s', errors)\n```\n\n"
            f"{rng.choice(['Callers pick out one kind with `except* ValueError` and the rest keep propagating', 'an ExceptionGroup carries all of them, so the caller sees the whole batch in one traceback', 'the group preserves every failure rather than discarding all but the first'])}.")

        add("pm_itertools_batch",
            f"send {nm} to an API that accepts at most {rng.choice([100, 250, 500])} at a time",
            f"```python\nfrom itertools import batched\n\n\n"
            f"for chunk in batched({nm}, {rng.choice([100, 250, 500])}):\n"
            f"    api.post(list(chunk))\n```\n\n"
            f"{rng.choice(['`itertools.batched` handles the ragged final chunk, which the manual slice loop usually gets wrong', 'no index arithmetic, and the last partial batch comes through correctly', 'it works on any iterable, not just something with a length'])}.")

    return out


def build_web(rng: random.Random, n_per: int) -> list[dict]:
    out: list[dict] = []

    def add(fam: str, task: str, ans: str) -> None:
        q = rng.choice(PHRASE).format(task=task + rng.choice(CTX))
        out.append(rec("python_web", fam, q[0].upper() + q[1:], rng.choice(OPEN) + ans))

    for _ in range(n_per):
        E = rng.choice(ENTITY); low = E.lower(); f = rng.choice(FIELD)
        svc = rng.choice(SVC); plural = low + "s"

        add("pw_endpoint_response_model",
            f"expose a read endpoint for {E} that cannot leak internal fields",
            f"```python\nfrom fastapi import APIRouter, HTTPException\nfrom pydantic import BaseModel\n\n"
            f"router = APIRouter(prefix='/{plural}', tags=['{plural}'])\n\n\n"
            f"class {E}Out(BaseModel):\n    id: int\n    {f}: float\n\n\n"
            f"@router.get('/{{item_id}}', response_model={E}Out)\n"
            f"async def get_{low}(item_id: int) -> {E}Out:\n"
            f"    row = await repo.get(item_id)\n    if row is None:\n"
            f"        raise HTTPException(status_code=404, detail='{low} not found')\n"
            f"    return row\n```\n\n"
            f"{rng.choice(['`response_model` filters the outgoing payload to declared fields, so an added internal column cannot escape later', 'the response model is the contract -- anything not on it is dropped on the way out', 'returning the ORM row directly is safe because the response model decides what serialises'])}.")

        add("pw_pydantic_validation",
            f"validate the {E} create payload at the edge",
            f"```python\nfrom pydantic import BaseModel, Field, field_validator\n\n\n"
            f"class {E}In(BaseModel):\n"
            f"    {f}: float = Field(gt=0, le={rng.choice([1000, 10_000, 100_000])})\n"
            f"    label: str = Field(min_length=1, max_length=120)\n"
            f"    tags: list[str] = Field(default_factory=list, max_length=10)\n\n"
            f"    @field_validator('label')\n    @classmethod\n"
            f"    def strip_label(cls, v: str) -> str:\n        return v.strip()\n```\n\n"
            f"{rng.choice(['A bad payload becomes a 422 with a field-level error before your handler runs', 'the constraints live in the type, so every route that accepts this model gets the same checks', 'validation at the boundary means the handler body can assume the data is already sane'])}.")

        add("pw_depends",
            f"give every {svc} route a database session without a global",
            f"```python\nfrom typing import Annotated\nfrom fastapi import Depends, Request\n\n\n"
            f"async def get_pool(request: Request) -> asyncpg.Pool:\n"
            f"    return request.app.state.pool\n\n\n"
            f"Pool = Annotated[asyncpg.Pool, Depends(get_pool)]\n\n\n"
            f"@router.get('/{plural}')\nasync def list_{plural}(pool: Pool) -> list[{E}Out]:\n"
            f"    async with pool.acquire() as conn:\n"
            f"        return await conn.fetch('SELECT id, {f} FROM {plural}')\n```\n\n"
            f"{rng.choice(['The `Annotated` alias means the dependency is declared once and reused as a type', 'a module-level global cannot be overridden in tests; a dependency can', 'this is the seam `dependency_overrides` hooks into during testing'])}.")

        add("pw_lifespan",
            f"open the {svc} connection pool at startup and close it cleanly",
            f"```python\nfrom contextlib import asynccontextmanager\nfrom fastapi import FastAPI\n\n\n"
            f"@asynccontextmanager\nasync def lifespan(app: FastAPI):\n"
            f"    app.state.pool = await asyncpg.create_pool(settings.database_url)\n"
            f"    app.state.http = httpx.AsyncClient(timeout=10.0)\n"
            f"    try:\n        yield\n    finally:\n"
            f"        await app.state.http.aclose()\n        await app.state.pool.close()\n\n\n"
            f"app = FastAPI(lifespan=lifespan)\n```\n\n"
            f"{rng.choice(['`lifespan` replaced `@app.on_event`, which is deprecated and had no way to share state between the two halves', 'the try/finally guarantees shutdown runs even if startup partially failed', 'one context manager owns both ends, so there is no matching pair of handlers to keep in sync'])}.")

        add("pw_error_shape",
            f"return a proper error when a {E} is missing",
            f"```python\nfrom fastapi import HTTPException, status\n\n\n"
            f"row = await repo.get(item_id)\nif row is None:\n"
            f"    raise HTTPException(\n"
            f"        status_code=status.HTTP_404_NOT_FOUND,\n"
            f"        detail=f'{low} {{item_id}} not found',\n    )\n```\n\n"
            f"{rng.choice(['A 200 carrying an error body means every client has to parse the body to find out whether it worked', 'the status code is what proxies, retries and monitoring read -- putting the failure only in the body hides it from all of them', 'clients and middleware branch on the status line, so the failure has to live there'])}.")

        add("pw_async_http",
            f"call the {svc} service from inside an async handler",
            f"```python\n@router.post('/{plural}/sync')\n"
            f"async def sync_{plural}(client: HttpClient) -> dict:\n"
            f"    resp = await client.get('https://{svc}.internal/v1/{plural}')\n"
            f"    resp.raise_for_status()\n    return resp.json()\n```\n\n"
            f"{rng.choice(['`httpx.AsyncClient` is awaited, so the loop keeps serving other requests while this one waits on the network', 'a blocking call here would park the event loop and stall every concurrent request', 'reuse one client from app state -- constructing one per request throws away connection pooling'])}.")

        add("pw_router_split",
            f"split the {svc} routes out of main.py",
            f"```python\n# {svc}/routes.py\nfrom fastapi import APIRouter\n\n"
            f"router = APIRouter(prefix='/{plural}', tags=['{svc}'])\n\n\n"
            f"@router.get('')\nasync def list_{plural}() -> list[{E}Out]:\n    ...\n\n\n"
            f"# main.py\napp.include_router(router)\n```\n\n"
            f"{rng.choice(['The prefix and tags are declared once and apply to every route in the module', 'dependencies passed to APIRouter apply to the whole group, which is how you gate a whole area behind auth', 'per-router tags are what keep the generated OpenAPI page readable'])}.")

        add("pw_streaming",
            f"return a large {plural} export without buffering it",
            f"```python\nfrom fastapi.responses import StreamingResponse\n\n\n"
            f"async def rows() -> AsyncIterator[bytes]:\n"
            f"    async with pool.acquire() as conn:\n"
            f"        async for r in conn.cursor('SELECT id, {f} FROM {plural}'):\n"
            f"            yield f'{{r[\"id\"]}},{{r[\"{f}\"]}}\\n'.encode()\n\n\n"
            f"@router.get('/{plural}/export')\nasync def export() -> StreamingResponse:\n"
            f"    return StreamingResponse(rows(), media_type='text/csv')\n```\n\n"
            f"{rng.choice(['The first bytes reach the client before the query has finished, and peak memory is one row', 'nothing is assembled in memory -- the response is produced as the cursor advances', 'a server-side cursor plus a streaming response keeps memory flat regardless of row count'])}.")

        add("pw_background",
            f"send the {low} confirmation without making the caller wait",
            f"```python\nfrom fastapi import BackgroundTasks\n\n\n"
            f"@router.post('/{plural}', status_code=201)\nasync def create_{low}(\n"
            f"    payload: {E}In,\n    background: BackgroundTasks,\n) -> {E}Out:\n"
            f"    row = await repo.create(payload)\n"
            f"    background.add_task(send_confirmation, row.id)\n    return row\n```\n\n"
            f"{rng.choice(['The task runs after the response is sent, so the client is not held for the email round trip', 'BackgroundTasks is right for short work -- anything retryable belongs in a real queue', 'this is in-process and dies with the worker; use a broker when the work must survive a restart'])}.")

        add("pw_testing",
            f"test the {plural} endpoints without binding a port",
            f"```python\nimport httpx\nimport pytest\n\n\n"
            f"@pytest.fixture\nasync def client():\n"
            f"    transport = httpx.ASGITransport(app=app)\n"
            f"    async with httpx.AsyncClient(transport=transport,\n"
            f"                                 base_url='http://test') as c:\n        yield c\n\n\n"
            f"async def test_list_{plural}(client):\n"
            f"    resp = await client.get('/{plural}')\n    assert resp.status_code == 200\n```\n\n"
            f"{rng.choice(['ASGITransport calls the app in-process -- no socket, no port, and the whole middleware stack still runs', 'requests go straight into the ASGI callable, so tests stay fast and parallel-safe', 'nothing binds, so the suite has no port collisions and no startup race'])}.")

        add("pw_overrides",
            f"swap the real {svc} repository for a fake in tests",
            f"```python\nasync def fake_pool():\n    return FakePool()\n\n\n"
            f"app.dependency_overrides[get_pool] = fake_pool\ntry:\n"
            f"    resp = await client.get('/{plural}')\nfinally:\n"
            f"    app.dependency_overrides.clear()\n```\n\n"
            f"{rng.choice(['`dependency_overrides` replaces the dependency at its declared seam, so no import path is patched', 'the substitution is by function object, which survives a refactor of where the module lives', 'monkeypatching an import breaks when the module moves; overriding the dependency does not'])}.")

        add("pw_pagination",
            f"paginate {plural} in a way that survives inserts",
            f"```python\n@router.get('/{plural}')\nasync def list_{plural}(\n"
            f"    pool: Pool,\n    after: int | None = None,\n"
            f"    limit: int = Query(50, le=200),\n) -> {E}Page:\n"
            f"    rows = await pool.fetch(\n"
            f"        'SELECT id, {f} FROM {plural} WHERE ($1::int IS NULL OR id > $1) '\n"
            f"        'ORDER BY id LIMIT $2',\n        after,\n        limit,\n    )\n"
            f"    return {E}Page(items=rows, next=rows[-1]['id'] if rows else None)\n```\n\n"
            f"{rng.choice(['Cursor pagination anchors on the last id, so a row inserted mid-scan cannot shift the window', 'OFFSET makes the database count and discard rows it already read, and a concurrent insert duplicates one', 'keyset pagination keeps a constant cost per page no matter how deep the client goes'])}.")

        add("pw_settings",
            f"read {svc} configuration from the environment with validation",
            f"```python\nfrom pydantic import Field\nfrom pydantic_settings import BaseSettings, SettingsConfigDict\n\n\n"
            f"class Settings(BaseSettings):\n"
            f"    model_config = SettingsConfigDict(env_file='.env', env_prefix='{svc.upper()}_')\n\n"
            f"    database_url: str\n    timeout_s: float = Field(10.0, gt=0)\n"
            f"    debug: bool = False\n\n\nsettings = Settings()\n```\n\n"
            f"{rng.choice(['A missing or unparseable variable fails at import, not on the first request that needs it', 'types are coerced and validated once at startup instead of scattered os.environ reads', 'the process refuses to start misconfigured, which is the behaviour you want in a deploy'])}.")

        add("pw_upload",
            f"accept a {rng.choice(['csv', 'zip', 'parquet'])} upload of {plural} without reading it all",
            f"```python\nfrom fastapi import File, UploadFile\n\n\n"
            f"@router.post('/{plural}/import')\n"
            f"async def import_{plural}(file: UploadFile = File(...)) -> dict:\n"
            f"    total = 0\n    while chunk := await file.read(1 << 20):\n"
            f"        total += handle_chunk(chunk)\n    return {{'rows': total}}\n```\n\n"
            f"{rng.choice(['`UploadFile` spools to disk past a threshold, so a large upload does not sit in RAM', 'reading in fixed chunks keeps memory bounded regardless of what the client sends', 'the walrus loop stops on the empty read without a separate break condition'])}.")

    return out


def finalise(rows: list[dict], cap: int) -> list[dict]:
    seen, uniq, per_fam = set(), [], Counter()
    for r in rows:
        k = " ".join(re.sub(r"[^a-z0-9\s]", " ",
                            re.sub(r"\d+", "0", r["messages"][1]["content"].lower())).split())
        fam = r["meta"]["family"]
        if k in seen or per_fam[fam] >= cap:
            continue
        seen.add(k); per_fam[fam] += 1; uniq.append(r)
    return uniq


def report(name: str, rows: list[dict]) -> None:
    ans = [r["messages"][1]["content"] for r in rows]
    n = max(1, len(rows))
    print(f"\n  {name}: {len(rows)} records")
    print(f"    unique questions {len(set(r['messages'][0]['content'] for r in rows))/n:6.1%}")
    print(f"    unique answers   100.0%  (deduped)")
    print(f"    emits code       {sum('```python' in a for a in ans)/n:6.1%}")
    print(f"    '**Not' share    {sum('**Not ' in a for a in ans)/n:6.1%}   (capability records: should be ~0)")
    print(f"    'Here is the'    {sum(a.startswith('Here is the') for a in ans)/n:6.1%}")
    print(f"    families         {len(set(r['meta']['family'] for r in rows))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--n-per", type=int, default=180)
    ap.add_argument("--cap", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    mod = finalise(build_modern(rng, args.n_per), args.cap)
    web = finalise(build_web(rng, args.n_per), args.cap)

    print("=" * 78)
    print(" PYTHON CAPABILITY RECORDS -- diluting the 76% single-form monoculture")
    print("=" * 78)
    report("python_modern", mod)
    report("python_web", web)

    for name, rows in (("python_modern", mod), ("python_web", web)):
        base = REPO_ROOT / "apps" / "factory" / "data" / name / "training_data_v5.jsonl"
        n_old = sum(1 for _ in base.open()) if base.exists() else 0
        n_disp = sum("**Not " in l for l in base.open()) if base.exists() else 0
        after = n_disp / max(1, n_old + len(rows))
        print(f"\n  {name}: disposition share {n_disp/max(1,n_old):.1%} -> {after:.1%} "
              f"after merge ({n_old} + {len(rows)} records)")

    if args.write:
        for name, rows in (("python_modern", mod), ("python_web", web)):
            out = REPO_ROOT / "apps" / "factory" / "data" / name / "training_data_capability.jsonl"
            out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            print(f"  WROTE {out.relative_to(REPO_ROOT)}")
    else:
        print("\n  (dry run -- pass --write)")


if __name__ == "__main__":
    main()
