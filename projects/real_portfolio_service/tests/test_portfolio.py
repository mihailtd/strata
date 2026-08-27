import sys
from pathlib import Path

import duckdb
import numpy as np

# Ensure local package path
PACKAGE_DIR = Path(__file__).resolve().parent.parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from analytics import compute_var_cvar, query_top_portfolio_assets  # noqa: E402
from models import PortfolioAsset, ResponseEnvelope, RiskMetrics  # noqa: E402


def test_models_pep695():
    asset = PortfolioAsset(asset_id="1", symbol="BTC", weight=0.5, expected_return=0.12)
    assert asset.symbol == "BTC"

    envelope: ResponseEnvelope[PortfolioAsset] = ResponseEnvelope(data=asset)
    assert envelope.status == "success"
    assert envelope.data.asset_id == "1"


def test_var_cvar_computation():
    np.random.seed(42)
    simulated_returns = np.random.normal(0.001, 0.02, 1000).tolist()

    metrics = compute_var_cvar(simulated_returns, confidence=0.95)
    assert isinstance(metrics, RiskMetrics)
    assert metrics.var_95 > 0.0
    assert metrics.cvar_95 >= metrics.var_95
    assert metrics.volatility > 0.0


def test_duckdb_qualify_query():
    conn = duckdb.connect(":memory:")
    top_assets = query_top_portfolio_assets(conn)
    assert len(top_assets) == 2
    assert top_assets[0]["symbol"] == "AAPL"
    assert top_assets[1]["symbol"] == "NVDA"
