from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import duckdb

try:
    from .db import VectorDatabasePool
except ImportError:
    from db import VectorDatabasePool


@asynccontextmanager
async def app_lifespan(app_state: dict) -> AsyncIterator[None]:
    """Modern FastAPI lifespan context manager."""
    app_state["db_pool"] = VectorDatabasePool()
    app_state["duckdb_conn"] = duckdb.connect(":memory:")
    yield
    app_state.clear()
