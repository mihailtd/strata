from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class VectorRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    item_id: str
    embedding: list[float]
    similarity_score: float = Field(ge=0.0, le=1.0)


class RankedAnalysis[T](BaseModel):
    model_config = ConfigDict(from_attributes=True)

    total_records: int
    top_items: list[T]
