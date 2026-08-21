"""Unit tests for Server on-demand engine load/unload lifecycle and API endpoints."""

import pytest
from runtime.server import model_state, get_engine_status, trigger_engine_unload


@pytest.mark.asyncio
async def test_engine_status_standby():
    model_state.clear()
    status = await get_engine_status()
    assert status.loaded is False
    assert status.vram_allocated_gb == 0.0


@pytest.mark.asyncio
async def test_engine_status_loaded():
    model_state["base_model"] = "mock_model"
    model_state["active_team"] = ["astral", "python_modern"]
    status = await get_engine_status()
    assert status.loaded is True
    assert status.active_team == ["astral", "python_modern"]
    model_state.clear()


@pytest.mark.asyncio
async def test_trigger_engine_unload():
    model_state["base_model"] = "mock_model"
    res = await trigger_engine_unload()
    assert res["status"] == "unloaded"
    assert "base_model" not in model_state


@pytest.mark.asyncio
async def test_causal_dag_endpoints():
    from runtime.notears_causal_scheduler import NotearsCausalScheduler
    from runtime.server import get_causal_dag, fit_causal_dag

    model_state["causal_scheduler"] = NotearsCausalScheduler()
    dag = await get_causal_dag()
    assert dag["fitted"] is True
    assert len(dag["edges"]) > 0

    fit_res = await fit_causal_dag()
    assert fit_res["fitted"] is True
    model_state.clear()
