import os

import pytest
import torch
from fastapi.testclient import TestClient

os.environ["GPU_EXCLUSIVE_ACTION"] = "warn"

from runtime.server import app, load_inference_engine, model_state


@pytest.mark.asyncio
async def test_multi_agent_pipeline_api():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Ensure engine is loaded
    if model_state.get("base_model") is None:
        await load_inference_engine()

    client = TestClient(app)

    # 1. Test tensor_handoff mode
    req_body = {
        "turns": [
            {
                "expert": "postgresql",
                "instruction": "Design a table 'items' with id bigint and embedding vector(128).",
            },
            {
                "expert": "astral",
                "instruction": "Write an async python function to insert into this table.",
            },
        ],
        "mode": "tensor_handoff",
        "max_new_tokens": 32,
        "temperature": 0.0,
    }

    resp = client.post("/api/multi_agent/run_pipeline", json=req_body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["mode"] == "tensor_handoff"
    assert len(data["steps"]) == 2
    assert data["steps"][0]["expert"] == "postgresql"
    assert data["steps"][1]["expert"] == "astral"
    assert data["steps"][1]["state_size_mb"] > 0
    assert data["steps"][1]["handoff_ms"] == 0.05
    assert len(data["steps"][0]["output_text"]) > 0
    assert len(data["steps"][1]["output_text"]) > 0

    # 2. Test both_side_by_side mode
    req_body["mode"] = "both_side_by_side"
    resp_both = client.post("/api/multi_agent/run_pipeline", json=req_body)
    assert resp_both.status_code == 200, resp_both.text
    data_both = resp_both.json()
    assert data_both["mode"] == "both_side_by_side"
    assert "tensor_arm" in data_both
    assert "text_arm" in data_both
    assert "comparison" in data_both
    assert "prefill_speedup" in data_both["comparison"]
    assert "tokens_saved" in data_both["comparison"]
