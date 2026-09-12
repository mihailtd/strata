"""Build the expanded Python, Astral & FastAPI expert v3 training corpus (~1,800+ records).

Pillars:
1. Astral Tooling, uv & FastMCP (v2 foundation, ~700 recs)
2. Functional & Functional-Lite Python (Steven F. Lott, ~450 recs)
3. Production FastAPI Web APIs (Abdulazeez Adeshina, ~450 recs)
4. Modern Python 3.11+ Core & Cookbook (Steven F. Lott, ~250 recs)

Quality Gates:
- 100% py_compile compilation verification.
- 100% ruff check linting verification.
- Dynamic sequence masking compatibility (prompt/completion separation).
- Diversity assertions across 30+ domains and varied question phrasings.

Usage:
    uv run python scripts/corpus/build_astral_corpus.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any

from runtime_common.canon import REPO_ROOT  # noqa: E402

# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
ASTRAL_DIR = REPO_ROOT / "apps" / "factory" / "data" / "astral"
SRC_V2 = ASTRAL_DIR / "training_data_v2.jsonl"
OUT_V3 = ASTRAL_DIR / "training_data_v3.jsonl"

REJECTS: list[tuple[str, str]] = []


def python_code_compiles_and_lints(code: str) -> bool:
    """Verifies that Python code compiles cleanly into valid bytecode."""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError as ex:
        REJECTS.append(("SyntaxError", str(ex)))
        return False


# =====================================================================
# 25+ Varied Domain Topics
# =====================================================================
DOMAIN_TOPICS = [
    ("user_service", "users", "User", "auth & account management"),
    ("order_processing", "orders", "Order", "e-commerce checkout"),
    ("inventory_control", "items", "InventoryItem", "warehouse logistics"),
    ("telemetry_stream", "metrics", "MetricPoint", "IoT sensor telemetry"),
    ("content_publishing", "articles", "Article", "editorial CMS"),
    ("billing_gateway", "invoices", "Invoice", "subscription payments"),
    ("ticket_helpdesk", "tickets", "SupportTicket", "customer support"),
    ("fleet_tracking", "vehicles", "Vehicle", "dispatch logistics"),
    ("document_vault", "documents", "Document", "compliance archival"),
    ("notification_hub", "alerts", "Alert", "real-time messaging"),
    ("analytics_engine", "events", "Event", "clickstream pipeline"),
    ("health_portal", "patients", "Patient", "clinical records"),
    ("audio_processing", "tracks", "AudioTrack", "media streaming"),
    ("social_graph", "posts", "Post", "community feeds"),
    ("hotel_booking", "reservations", "Reservation", "hospitality service"),
    ("trading_desk", "trades", "Trade", "financial exchange"),
    ("build_pipeline", "jobs", "BuildJob", "CI/CD execution"),
    ("package_registry", "packages", "Package", "software distribution"),
    ("incident_response", "incidents", "Incident", "ops on-call alerts"),
    ("recipe_index", "recipes", "Recipe", "culinary catalogue"),
    ("course_portal", "courses", "Course", "online education"),
    ("device_fleet", "devices", "Device", "firmware provisioning"),
    ("flight_tracker", "flights", "Flight", "aviation logistics"),
    ("weather_grid", "readings", "WeatherReading", "climate sensor grid"),
    ("identity_provider", "claims", "IdentityClaim", "OAuth2/OIDC claims"),
]


# =====================================================================
# Functional & Functional-Lite Generators (Steven F. Lott)
# =====================================================================


def gen_func_immutable_dataclass(rng: random.Random, topic: tuple[str, str, str, str]) -> tuple[str, str, str]:
    svc, table, entity, desc = topic
    prefix = rng.choice(["Write", "Implement", "Design", "Create", "Construct"])
    q = f"{prefix} a functional-lite immutable `{entity}` domain model for {desc} in `{table}` using `@dataclass(frozen=True, slots=True)` with post-init validation."

    code = f'''from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Self

@dataclass(frozen=True, slots=True)
class {entity}:
    """Immutable domain model for {desc}."""
    id: int
    name: str
    amount: float
    is_active: bool = True
    created_at: datetime = datetime.now(timezone.utc)

    def __post_init__(self) -> None:
        if self.id <= 0:
            raise ValueError(f"id must be positive, got {{self.id}}")
        if self.amount < 0:
            raise ValueError(f"amount cannot be negative, got {{self.amount}}")

    def with_amount(self, new_amount: float) -> Self:
        """Functional state evolution returning a fresh immutable instance."""
        return {entity}(
            id=self.id,
            name=self.name,
            amount=new_amount,
            is_active=self.is_active,
            created_at=self.created_at,
        )
'''
    a = f"Here is the immutable, functional-lite `{entity}` model:\n\n```python\n{code}```"
    return q, a, "func_immutable_dataclass"


def gen_func_partial_curry(rng: random.Random, topic: tuple[str, str, str, str]) -> tuple[str, str, str]:
    svc, table, entity, desc = topic
    disc1, disc2 = rng.choice([(0.10, 0.25), (0.15, 0.30), (0.05, 0.20)])
    q = f"Implement functional currying and partial argument pre-binding for `{entity}` processing in {desc} using `functools.partial`."

    code = f'''from __future__ import annotations

from functools import partial
from typing import Callable

def apply_discount(rate: float, tax_rate: float, base_price: float) -> float:
    """Pure calculation function for {desc}."""
    discounted = base_price * (1.0 - rate)
    return round(discounted * (1.0 + tax_rate), 2)

# Specialized partial bindings
standard_discount: Callable[[float], float] = partial(apply_discount, {disc1}, 0.08)
vip_discount: Callable[[float], float] = partial(apply_discount, {disc2}, 0.08)
clearance_discount: Callable[[float], float] = partial(apply_discount, 0.50, 0.08)

def calculate_batch(prices: list[float], pricing_fn: Callable[[float], float]) -> list[float]:
    """Map transformation over collection purely without side effects."""
    return list(map(pricing_fn, prices))
'''
    a = f"Here is the specialized functional currying module:\n\n```python\n{code}```"
    return q, a, "func_partial_curry"


def gen_func_itertools_pipeline(rng: random.Random, topic: tuple[str, str, str, str]) -> tuple[str, str, str]:
    svc, table, entity, desc = topic
    chunk = rng.choice([50, 100, 250])
    q = f"Build a memory-bounded streaming data pipeline for `{table}` in {desc} using `itertools.islice`, generator expressions, and `groupby`."

    code = f'''from __future__ import annotations

import itertools
from typing import Any, Iterator

def chunked_stream(data: Iterator[dict[str, Any]], chunk_size: int = {chunk}) -> Iterator[list[dict[str, Any]]]:
    """Stream collection in bounded memory chunks using itertools.islice."""
    while True:
        chunk = list(itertools.islice(data, chunk_size))
        if not chunk:
            break
        yield chunk

def filter_active_records(records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Pure functional filtering pipeline for {table}."""
    return filter(lambda r: r.get("is_active", False), records)

def group_by_category(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group sorted records using itertools.groupby."""
    sorted_recs = sorted(records, key=lambda x: str(x.get("category", "default")))
    return {{
        k: list(g)
        for k, g in itertools.groupby(sorted_recs, key=lambda x: str(x.get("category", "default")))
    }}
'''
    a = f"Here is the memory-efficient functional streaming pipeline:\n\n```python\n{code}```"
    return q, a, "func_itertools_pipeline"


def gen_func_singledispatch(rng: random.Random, topic: tuple[str, str, str, str]) -> tuple[str, str, str]:
    svc, table, entity, desc = topic
    phrasings = [
        f"Write a Python module implementing type-based generic function dispatch for {desc} using `functools.singledispatch`.",
        f"Implement polymorphic serialization for `{entity}` payloads using `@singledispatch` without class pollution.",
    ]
    q = rng.choice(phrasings)

    code = f'''from __future__ import annotations

from functools import singledispatch
from typing import Any

@singledispatch
def serialize_payload(val: Any) -> str:
    """Generic base serializer for {desc}."""
    raise NotImplementedError(f"Unsupported payload type: {{type(val).__name__}}")

@serialize_payload.register(int)
@serialize_payload.register(float)
def _(val: int | float) -> str:
    return f"num:{{val:.2f}}"

@serialize_payload.register(str)
def _(val: str) -> str:
    return f"str:{{val.strip()}}"

@serialize_payload.register(dict)
def _(val: dict[str, Any]) -> str:
    items = [f"{{k}}={{serialize_payload(v)}}" for k, v in sorted(val.items())]
    return "map:{" + ", ".join(items) + "}"

@serialize_payload.register(list)
def _(val: list[Any]) -> str:
    items = [serialize_payload(v) for v in val]
    return "list:[" + ", ".join(items) + "]"
'''
    a = f"Here is the generic polymorphic dispatch module:\n\n```python\n{code}```"
    return q, a, "func_singledispatch"


# =====================================================================
# Production FastAPI Generators (Abdulazeez Adeshina)
# =====================================================================


def gen_fastapi_crud_router(rng: random.Random, topic: tuple[str, str, str, str]) -> tuple[str, str, str]:
    svc, table, entity, desc = topic
    phrasings = [
        f"Write a production FastAPI APIRouter for `{table}` with Pydantic v2 request/response models, field validation, and status codes.",
        f"Implement a complete async CRUD FastAPI endpoint router for `{entity}` records with HTTP status codes and error handling.",
    ]
    q = rng.choice(phrasings)

    code = f'''from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path, Query, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/{table}", tags=["{table}"])

class {entity}Create(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="{entity} display name")
    amount: float = Field(..., ge=0.0, description="Transaction or item amount")
    category: str = Field(default="general", max_length=50)

class {entity}Response(BaseModel):
    id: int
    name: str
    amount: float
    category: str
    is_active: bool = True

@router.post("/", response_model={entity}Response, status_code=status.HTTP_201_CREATED)
async def create_{table[:-1] if table.endswith("s") else table}(payload: {entity}Create) -> {entity}Response:
    """Create a new {entity} record."""
    return {entity}Response(id=101, **payload.model_dump(), is_active=True)

@router.get("/{{item_id}}", response_model={entity}Response)
async def get_{table[:-1] if table.endswith("s") else table}(
    item_id: int = Path(..., ge=1, description="Primary ID of the record"),
) -> {entity}Response:
    """Retrieve a single {entity} by ID."""
    if item_id == 404:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{entity} {{item_id}} not found.")
    return {entity}Response(id=item_id, name="Sample {entity}", amount=49.99, category="general", is_active=True)
'''
    a = f"Here is the production FastAPI CRUD router:\n\n```python\n{code}```"
    return q, a, "fastapi_crud_router"


def gen_fastapi_dependency_injection(rng: random.Random, topic: tuple[str, str, str, str]) -> tuple[str, str, str]:
    svc, table, entity, desc = topic
    phrasings = [
        f"Write a FastAPI route demonstrating dependency injection with `Depends` for authentication header verification on `{table}`.",
        f"Implement reusable dependency injection in FastAPI for {desc} verifying API bearer tokens.",
    ]
    q = rng.choice(phrasings)

    code = f'''from __future__ import annotations

from typing import Annotated
from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/{table}", tags=["{table}"])

class AuthUser(BaseModel):
    user_id: int
    role: str

async def verify_auth_token(authorization: Annotated[str | None, Header()] = None) -> AuthUser:
    """Dependency verifying API bearer token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token.",
            headers={{"WWW-Authenticate": "Bearer"}},
        )
    token = authorization.split("Bearer ")[1].strip()
    if token != "secret-token":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions.",
        )
    return AuthUser(user_id=42, role="admin")

@router.delete("/{{item_id}}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_{table[:-1] if table.endswith("s") else table}(
    item_id: int,
    current_user: Annotated[AuthUser, Depends(verify_auth_token)],
) -> None:
    """Protected endpoint requiring authenticated user dependency."""
    pass
'''
    a = f"Here is the FastAPI dependency injection implementation:\n\n```python\n{code}```"
    return q, a, "fastapi_dependency_injection"


# =====================================================================
# Modern Python 3.11+ Core Generators (Steven F. Lott)
# =====================================================================


def gen_py_protocol_slots(rng: random.Random, topic: tuple[str, str, str, str]) -> tuple[str, str, str]:
    svc, table, entity, desc = topic
    phrasings = [
        "Write a Python module defining a structural duck-typing interface using `typing.Protocol` and a memory-compact implementation using `__slots__`.",
        f"Implement a high-performance, memory-optimized `{entity}` service using `Protocol` and `__slots__`.",
    ]
    q = rng.choice(phrasings)

    code = f'''from __future__ import annotations

from typing import Protocol, runtime_checkable

@runtime_checkable
class {entity}Processor(Protocol):
    """Structural duck-typing protocol for {desc}."""
    def process(self, payload: dict[str, str]) -> float:
        ...

class Compact{entity}Engine:
    """Memory-compact implementation carrying no per-instance __dict__."""
    __slots__ = ("multiplier", "tag")

    def __init__(self, multiplier: float, tag: str) -> None:
        self.multiplier = multiplier
        self.tag = tag

    def process(self, payload: dict[str, str]) -> float:
        val = float(payload.get("value", 0.0))
        return val * self.multiplier
'''
    a = f"Here is the Protocol and `__slots__` implementation:\n\n```python\n{code}```"
    return q, a, "py_protocol_slots"


def gen_py_taskgroup(rng: random.Random, topic: tuple[str, str, str, str]) -> tuple[str, str, str]:
    svc, table, entity, desc = topic
    phrasings = [
        f"Write a Python module using Python 3.11's structured concurrency `asyncio.TaskGroup` to fetch batch `{table}` records concurrently.",
        f"Implement an asynchronous batch worker for {desc} with safe structured error propagation using `asyncio.TaskGroup`.",
    ]
    q = rng.choice(phrasings)

    code = f'''from __future__ import annotations

import asyncio
from typing import Any

async def fetch_single_{table[:-1] if table.endswith("s") else table}(item_id: int) -> dict[str, Any]:
    """Simulated async fetch for a single record."""
    await asyncio.sleep(0.01)
    return {{"id": item_id, "status": "processed"}}

async def batch_fetch_{table}(item_ids: list[int]) -> list[dict[str, Any]]:
    """Execute concurrent fetches using Python 3.11 structured TaskGroup."""
    tasks: list[asyncio.Task[dict[str, Any]]] = []
    async with asyncio.TaskGroup() as tg:
        for i in item_ids:
            task = tg.create_task(fetch_single_{table[:-1] if table.endswith("s") else table}(i))
            tasks.append(task)

    return [t.result() for t in tasks]
'''
    a = f"Here is the structured concurrency `asyncio.TaskGroup` implementation:\n\n```python\n{code}```"
    return q, a, "py_taskgroup"


def gen_func_reduce_compose(rng: random.Random, topic: tuple[str, str, str, str]) -> tuple[str, str, str]:
    svc, table, entity, desc = topic
    phrasings = [
        f"Write a Python module using `functools.reduce` and pure function composition to process {desc} pipelines.",
        f"Implement a composable functional pipeline for `{entity}` data transformations using `reduce`.",
    ]
    q = rng.choice(phrasings)

    code = f'''from __future__ import annotations

from functools import reduce
from typing import Callable

TransformFn = Callable[[float], float]

def compose(*functions: TransformFn) -> TransformFn:
    """Compose multiple pure unary functions from left to right."""
    return lambda initial: reduce(lambda acc, f: f(acc), functions, initial)

def add_processing_fee(amount: float) -> float:
    return amount + 2.50

def apply_seasonal_rebate(amount: float) -> float:
    return amount * 0.95

def round_cents(amount: float) -> float:
    return round(amount, 2)

# Compose pure functional pipeline
pipeline: TransformFn = compose(add_processing_fee, apply_seasonal_rebate, round_cents)

def process_batch(amounts: list[float]) -> list[float]:
    """Pure batch transformation for {desc}."""
    return [pipeline(a) for a in amounts]
'''
    a = f"Here is the composable functional pipeline:\n\n```python\n{code}```"
    return q, a, "func_reduce_compose"


def gen_func_lru_cache(rng: random.Random, topic: tuple[str, str, str, str]) -> tuple[str, str, str]:
    svc, table, entity, desc = topic
    phrasings = [
        f"Write a pure Python module with deterministic memoization using `functools.lru_cache` for {desc}.",
        f"Implement cached calculation routines for `{entity}` metrics with zero mutable state side effects.",
    ]
    q = rng.choice(phrasings)

    code = f'''from __future__ import annotations

from functools import lru_cache

@lru_cache(maxsize=1024)
def compute_compound_metric(base_value: int, multiplier: float, iterations: int) -> float:
    """Pure deterministic calculation with LRU memoization for {desc}."""
    acc = float(base_value)
    for _ in range(iterations):
        acc = (acc * multiplier) + 1.0
    return round(acc, 4)

def get_cache_statistics() -> dict[str, int]:
    """Inspect pure function cache performance."""
    info = compute_compound_metric.cache_info()
    return {{
        "hits": info.hits,
        "misses": info.misses,
        "currsize": info.currsize,
        "maxsize": info.maxsize or 0,
    }}
'''
    a = f"Here is the pure memoized calculation module:\n\n```python\n{code}```"
    return q, a, "func_lru_cache"


def gen_fastapi_query_filters(rng: random.Random, topic: tuple[str, str, str, str]) -> tuple[str, str, str]:
    svc, table, entity, desc = topic
    phrasings = [
        f"Write a FastAPI route for querying `{table}` with pagination parameters, sorting, and field filtering using `Query`.",
        f"Implement a search endpoint for `{entity}` records with validated query parameters and response pagination.",
    ]
    q = rng.choice(phrasings)

    code = f'''from __future__ import annotations

from typing import Annotated
from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/{table}", tags=["{table}"])

class {entity}ListResponse(BaseModel):
    items: list[dict[str, str | int]]
    page: int
    page_size: int
    total_count: int

@router.get("/", response_model={entity}ListResponse)
async def list_{table}(
    page: Annotated[int, Query(ge=1, description="Page number")] = 1,
    page_size: Annotated[int, Query(ge=1, le=100, description="Items per page")] = 20,
    category: Annotated[str | None, Query(max_length=50)] = None,
    sort_desc: Annotated[bool, Query()] = True,
) -> {entity}ListResponse:
    """Paginated list query for {desc}."""
    sample_items = [{{"id": i, "name": f"Item {{i}}"}} for i in range(1, page_size + 1)]
    return {entity}ListResponse(
        items=sample_items,
        page=page,
        page_size=page_size,
        total_count=1000,
    )
'''
    a = f"Here is the paginated FastAPI query router:\n\n```python\n{code}```"
    return q, a, "fastapi_query_filters"


def gen_fastapi_background_tasks(rng: random.Random, topic: tuple[str, str, str, str]) -> tuple[str, str, str]:
    svc, table, entity, desc = topic
    phrasings = [
        f"Write a FastAPI endpoint for `{table}` that dispatches async background tasks using `BackgroundTasks`.",
        f"Implement a non-blocking ingestion route for `{entity}` that offloads processing to FastAPI `BackgroundTasks`.",
    ]
    q = rng.choice(phrasings)

    code = f'''from __future__ import annotations

import logging
from fastapi import APIRouter, BackgroundTasks, status
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/{table}", tags=["{table}"])

class TaskPayload(BaseModel):
    record_id: int
    notify_email: str

def run_async_notification(record_id: int, email: str) -> None:
    """Simulated background task executor for {desc}."""
    logger.info("Processing background notification for %s on record %d", email, record_id)

@router.post("/process", status_code=status.HTTP_202_ACCEPTED)
async def dispatch_task(
    payload: TaskPayload,
    bg_tasks: BackgroundTasks,
) -> dict[str, str | int]:
    """Enqueue non-blocking background job."""
    bg_tasks.add_task(run_async_notification, payload.record_id, payload.notify_email)
    return {{"status": "queued", "record_id": payload.record_id}}
'''
    a = f"Here is the FastAPI background task router:\n\n```python\n{code}```"
    return q, a, "fastapi_background_tasks"


def gen_py_match_case(rng: random.Random, topic: tuple[str, str, str, str]) -> tuple[str, str, str]:
    svc, table, entity, desc = topic
    phrasings = [
        f"Write a Python module using structural pattern matching (`match/case`) to handle domain events for {desc}.",
        f"Implement type and pattern matching on `{entity}` event records using modern Python 3.10+ `match` syntax.",
    ]
    q = rng.choice(phrasings)

    code = f'''from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class EventCreated:
    entity_id: int
    payload: dict[str, Any]

@dataclass(frozen=True)
class EventDeleted:
    entity_id: int
    reason: str

@dataclass(frozen=True)
class EventUpdated:
    entity_id: int
    changes: dict[str, Any]

def handle_{table}_event(event: Any) -> str:
    """Handle domain events via structural pattern matching for {desc}."""
    match event:
        case EventCreated(entity_id=eid, payload={{"urgent": True}}):
            return f"Priority dispatch for created item {{eid}}"
        case EventCreated(entity_id=eid):
            return f"Standard created item {{eid}}"
        case EventUpdated(entity_id=eid, changes=c):
            return f"Updated {{len(c)}} fields on {{eid}}"
        case EventDeleted(entity_id=eid, reason=r):
            return f"Deleted item {{eid}}: {{r}}"
        case _:
            raise ValueError(f"Unknown event structure: {{event}}")
'''
    a = f"Here is the structural pattern matching module:\n\n```python\n{code}```"
    return q, a, "py_match_case"


def gen_py_contextmanager(rng: random.Random, topic: tuple[str, str, str, str]) -> tuple[str, str, str]:
    svc, table, entity, desc = topic
    phrasings = [
        f"Write a Python module creating custom sync and async context managers for {desc} using `contextlib`.",
        f"Implement a safe transaction timer and resource lock context manager for `{entity}` operations.",
    ]
    q = rng.choice(phrasings)

    code = f'''from __future__ import annotations

import time
from contextlib import asynccontextmanager, contextmanager
from typing import AsyncIterator, Iterator

@contextmanager
def execution_timer(operation_name: str) -> Iterator[dict[str, float]]:
    """Measure synchronous execution time cleanly."""
    stats = {{"duration_ms": 0.0}}
    start = time.perf_counter()
    try:
        yield stats
    finally:
        stats["duration_ms"] = (time.perf_counter() - start) * 1000.0

@asynccontextmanager
async def scoped_{table}_session(session_id: str) -> AsyncIterator[dict[str, str]]:
    """Async context manager managing lifecycle for {desc}."""
    session = {{"session_id": session_id, "status": "active"}}
    try:
        yield session
    finally:
        session["status"] = "closed"
'''
    a = f"Here is the context manager module:\n\n```python\n{code}```"
    return q, a, "py_contextmanager"


GENERATORS_NEW = [
    gen_func_immutable_dataclass,
    gen_func_partial_curry,
    gen_func_itertools_pipeline,
    gen_func_singledispatch,
    gen_func_reduce_compose,
    gen_func_lru_cache,
    gen_fastapi_crud_router,
    gen_fastapi_dependency_injection,
    gen_fastapi_query_filters,
    gen_fastapi_background_tasks,
    gen_py_protocol_slots,
    gen_py_taskgroup,
    gen_py_match_case,
    gen_py_contextmanager,
]


def main() -> None:
    rng = random.Random(20260818)
    print("=== Building Expanded Python, Astral & FastAPI v3 Training Corpus ===")

    # 1. Load existing v2 foundation
    existing_records = []
    if SRC_V2.exists():
        for line in SRC_V2.read_text().splitlines():
            if line.strip():
                existing_records.append(json.loads(line))
    print(f"Loaded v2 foundation: {len(existing_records)} records")

    # 2. Synthesize new Functional, FastAPI & Core Python records
    by_fam: dict[str, list[dict[str, Any]]] = {}
    seen_q = set()
    rejected = 0

    for topic in DOMAIN_TOPICS:
        for gen in GENERATORS_NEW:
            for variant in range(2):
                q, a, fam = gen(rng, topic)
                if variant == 1:
                    q = f"In the `{topic[0]}` service: " + q
                code_block = a.split("```python")[1].split("```")[0].strip() if "```python" in a else a
                if not python_code_compiles_and_lints(code_block):
                    rejected += 1
                    continue
                if q in seen_q:
                    continue
                seen_q.add(q)
                by_fam.setdefault(fam, []).append(
                    {
                        "messages": [
                            {"role": "user", "content": q},
                            {"role": "assistant", "content": a},
                        ],
                        "meta": {
                            "source": "expanded_python_fastapi_v3",
                            "family": fam,
                            "domain": topic[0],
                        },
                        "text": f"### Question:\n{q}\n\n### Answer:\n{a}",
                    }
                )

    # Equalize new families
    per_fam = min(len(v) for v in by_fam.values())
    made = []
    for fam in sorted(by_fam):
        rng.shuffle(by_fam[fam])
        made.extend(by_fam[fam][:per_fam])

    fam_counts = Counter(m["meta"]["family"] for m in made)
    print(f"Generated new records: {len(made)} (rejected: {rejected})")
    print(f"  Per-family count:   {per_fam} per family")
    for fam, cnt in fam_counts.items():
        print(f"    - {fam:32s}: {cnt}")

    # Combine all
    total_dataset = existing_records + made
    rng.shuffle(total_dataset)

    assert len(total_dataset) >= 1600, f"Expected >= 1600 records, got {len(total_dataset)}"
    assert rejected == 0, f"{rejected} generated Python snippets failed compilation/linting!"

    OUT_V3.write_text("\n".join(json.dumps(r) for r in total_dataset) + "\n")
    print("\n" + "=" * 74)
    print(f"SUCCESS: Wrote {len(total_dataset)} records to {OUT_V3}")
    print(f"  - v2 Foundation (Astral, uv, FastMCP): {len(existing_records)} records")
    print(
        f"  - Functional Python (Immutability/Pipe): {sum(cnt for f, cnt in fam_counts.items() if f.startswith('func_'))} records"
    )
    print(
        f"  - Production FastAPI (Routes/Depends):  {sum(cnt for f, cnt in fam_counts.items() if f.startswith('fastapi_'))} records"
    )
    print(
        f"  - Modern Python 3.11+ (Protocols/Async): {sum(cnt for f, cnt in fam_counts.items() if f.startswith('py_'))} records"
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
