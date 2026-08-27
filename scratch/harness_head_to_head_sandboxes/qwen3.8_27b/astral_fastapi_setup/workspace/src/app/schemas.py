"""Pydantic models for request/response bodies (API contracts)."""
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response body for the liveness probe."""

    status: str
    app: str
    version: str
    time: datetime


class ItemCreate(BaseModel):
    """Payload for creating an item."""

    name: str = Field(min_length=1, max_length=200, examples=["Coffee"])
    price: float = Field(gt=0, examples=[4.5])
    sku: str | None = Field(default=None, max_length=64)


class Item(ItemCreate):
    """A stored item (creation payload + server-assigned fields)."""

    id: UUID
    created_at: datetime


class ErrorModel(BaseModel):
    """Standard error envelope returned for handled failures."""

    error: str
    message: str
