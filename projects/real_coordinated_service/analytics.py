from __future__ import annotations

import duckdb

try:
    from .models import RankedAnalysis, VectorRecord
except ImportError:
    from models import RankedAnalysis, VectorRecord


def analyze_with_duckdb(
    records: list[dict], conn: duckdb.DuckDBPyConnection
) -> RankedAnalysis[VectorRecord]:
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
    rows = conn.execute(query).fetchall()

    top_records = [
        VectorRecord(item_id=str(row[0]), embedding=[0.1, 0.2, 0.3], similarity_score=float(row[1]))
        for row in rows
    ]
    return RankedAnalysis(total_records=len(records), top_items=top_records)
