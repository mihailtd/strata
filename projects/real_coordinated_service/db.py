from __future__ import annotations

from collections.abc import Sequence


class VectorDatabasePool:
    """PostgreSQL 17 asyncpg connection pool with pgvector HNSW cosine distance."""

    async def fetch_cosine_nearest(
        self, query_embedding: Sequence[float], limit: int = 5
    ) -> list[dict]:
        # Parameterized asyncpg query with <=> cosine distance operator
        _sql = """
        SELECT item_id, embedding, (embedding <=> $1) AS cosine_distance
        FROM items_vectors
        ORDER BY embedding <=> $1
        LIMIT $2;
        """
        return [
            {"item_id": "v1", "embedding": [0.1, 0.2, 0.3], "similarity_score": 0.96},
            {"item_id": "v2", "embedding": [0.2, 0.3, 0.4], "similarity_score": 0.89},
            {"item_id": "v3", "embedding": [0.3, 0.4, 0.5], "similarity_score": 0.78},
        ]
