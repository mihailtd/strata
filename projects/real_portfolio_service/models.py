from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PortfolioAsset(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)

    asset_id: str
    symbol: str
    weight: float = Field(gt=0.0, le=1.0)
    expected_return: float


class RiskMetrics(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    var_95: float
    cvar_95: float
    volatility: float


class ResponseEnvelope[T](BaseModel):
    status: str = "success"
    data: T
