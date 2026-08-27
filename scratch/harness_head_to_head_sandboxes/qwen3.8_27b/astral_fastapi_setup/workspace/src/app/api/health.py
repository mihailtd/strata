"""Liveness/readiness endpoint."""
from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def healthcheck(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Return service liveness + identity."""
    return HealthResponse(
        status="ok",
        app=settings.name,
        version=settings.version,
        time=datetime.now(UTC),
    )
