from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

PACKAGE_DIR = Path(__file__).resolve().parent.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from analytics import analyze_with_duckdb  # noqa: E402
from main import app_lifespan  # noqa: E402
from models import RankedAnalysis, VectorRecord  # noqa: E402


@pytest.mark.asyncio
async def test_lifespan_and_vector_query():
    state = {}
    async with app_lifespan(state):
        assert "db_pool" in state
        pool = state["db_pool"]
        records = await pool.fetch_cosine_nearest([0.1, 0.2, 0.3], limit=3)
        assert len(records) == 3
        assert records[0]["similarity_score"] == 0.96


def test_duckdb_qualify_pipeline():
    conn = duckdb.connect(":memory:")
    sample = [
        {"item_id": "a", "similarity_score": 0.98},
        {"item_id": "b", "similarity_score": 0.85},
        {"item_id": "c", "similarity_score": 0.40},
    ]
    res: RankedAnalysis[VectorRecord] = analyze_with_duckdb(sample, conn)
    assert res.total_records == 3
    assert len(res.top_items) == 2
    assert res.top_items[0].item_id == "a"
    assert res.top_items[0].similarity_score == 0.98
