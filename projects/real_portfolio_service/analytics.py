from __future__ import annotations

from collections.abc import Sequence

import duckdb
import numpy as np

try:
    from .models import RiskMetrics
except ImportError:
    from models import RiskMetrics


def compute_var_cvar(returns: Sequence[float], confidence: float = 0.95) -> RiskMetrics:
    """Compute Historical Value at Risk (VaR) and Conditional VaR (Expected Shortfall)."""
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) == 0:
        raise ValueError("Returns sequence cannot be empty")

    alpha = 1.0 - confidence
    var_threshold = -float(np.percentile(arr, alpha * 100.0))

    tail_losses = -arr[arr <= -var_threshold]
    cvar_val = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_threshold
    volatility = float(np.std(arr))

    return RiskMetrics(
        var_95=round(var_threshold, 6),
        cvar_95=round(cvar_val, 6),
        volatility=round(volatility, 6),
    )


def query_top_portfolio_assets(conn: duckdb.DuckDBPyConnection) -> list[dict]:
    """Run DuckDB SQL analytics using the QUALIFY window clause."""
    query = """
    SELECT 
        asset_id,
        symbol,
        weight,
        ROW_NUMBER() OVER (ORDER BY weight DESC) as rank
    FROM (
        SELECT 'a1' as asset_id, 'AAPL' as symbol, 0.40 as weight
        UNION ALL
        SELECT 'a2' as asset_id, 'NVDA' as symbol, 0.35 as weight
        UNION ALL
        SELECT 'a3' as asset_id, 'MSFT' as symbol, 0.25 as weight
    )
    QUALIFY rank <= 2;
    """
    df = conn.execute(query).fetchdf()
    return df.to_dict(orient="records")
