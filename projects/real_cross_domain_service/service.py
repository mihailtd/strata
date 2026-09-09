from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import duckdb

try:
    from .models import RankedAnalysis, VectorRecord
except ImportError:
    from models import RankedAnalysis, VectorRecord


class VectorDatabasePool:
    """Simulated asyncpg connection pool."""

    async def fetch_cosine_nearest(self, query_embedding: Sequence[float], limit: int = 5) -> list[dict]:
        # PostgreSQL pgvector syntax query
        _sql_statement = """
        SELECT item_id, embedding, (embedding <=> $1) AS cosine_distance
        FROM items_vectors
        ORDER BY embedding <=> $1
        LIMIT $2;
        """
        # Simulated mock records for in-memory execution
        return [
            {"item_id": "v1", "embedding": [0.1, 0.2, 0.3], "similarity_score": 0.95},
            {"item_id": "v2", "embedding": [0.2, 0.3, 0.4], "similarity_score": 0.88},
            {"item_id": "v3", "embedding": [0.3, 0.4, 0.5], "similarity_score": 0.76},
        ]


@asynccontextmanager
async def app_lifespan(app_state: dict) -> AsyncIterator[None]:
    """Modern FastAPI lifespan context manager."""
    app_state["db_pool"] = VectorDatabasePool()
    app_state["duckdb_conn"] = duckdb.connect(":memory:")
    yield
    app_state.clear()


def analyze_with_duckdb(records: list[dict], conn: duckdb.DuckDBPyConnection) -> RankedAnalysis[VectorRecord]:
    """Runs DuckDB SQL with native QUALIFY window clause."""
    conn.execute("CREATE OR REPLACE TABLE raw_items (item_id VARCHAR, score DOUBLE);")
    for r in records:
        conn.execute("INSERT INTO raw_items VALUES (?, ?);", [r["item_id"], r["similarity_score"]])

    query = """
    SELECT 
        item_id,
        score,
        ROW_NUMBER() OVER (ORDER BY score DESC) as rank_pos
    FROM raw_items
    QUALIFY rank_pos <= 2;
    """
    df = conn.execute(query).fetchdf()

    top_records = [
        VectorRecord(item_id=row["item_id"], embedding=[0.1, 0.2, 0.3], similarity_score=float(row["score"]))
        for _, row in df.iterrows()
    ]
    return RankedAnalysis(total_records=len(records), top_items=top_records)
