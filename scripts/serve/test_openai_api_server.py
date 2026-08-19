"""Integration Test Suite for IMB OpenAI-Compatible FastAPI REST API Server."""

import json
import sys
from pathlib import Path

import torch
from fastapi.testclient import TestClient

from gnn_experiment.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT / "src"))

from gnn_experiment.server import app, model_state  # noqa: E402
from gnn_experiment.utils.logger import log_benchmark_metric  # noqa: E402


def run_openai_api_server_tests():
    print("==================================================")
    print(" IMB OpenAI-Compatible REST API Integration Tests")
    print("==================================================")

    # Use TestClient with lifespan context manager trigger
    with TestClient(app) as client:
        # Check Health
        res_health = client.get("/health")
        assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
        print(f"  Health Check Status   : {res_health.json()}")

        # Check CUDA Graph Capture Count
        graph_decoder = model_state["graph_decoder"]

        assert graph_decoder.capture_count == 1, f"CUDA Graph Capture Guard Failed: Count={graph_decoder.capture_count}"
        assert graph_decoder._is_locked is True, "Graph Decoder must remain locked!"
        print(f"  CUDA Graph Capture    : {graph_decoder.capture_count} (Expected: 1, Lock=True)")

        # --- Test 1: GET /v1/models ---
        print("\n--- Test 1: GET /v1/models ---")
        res_models = client.get("/v1/models")
        assert res_models.status_code == 200, f"/v1/models failed: {res_models.text}"
        data_models = res_models.json()
        model_ids = [m["id"] for m in data_models["data"]]
        print(f"  Available Models      : {model_ids}")
        assert "financial_planning" in model_ids, "Missing financial_planning model in /v1/models"
        assert "postgresql" in model_ids, "Missing postgresql model in /v1/models"
        assert "astral" in model_ids, "Missing astral model in /v1/models"

        # --- Test 2: POST /v1/chat/completions (Financial Expert) ---
        print("\n--- Test 2: POST /v1/chat/completions (Financial Expert) ---")
        payload_fin = {
            "model": "financial_planning",
            "messages": [{"role": "user", "content": "Formulate a sequence-of-returns risk strategy for retirement."}],
            "max_tokens": 64,
            "stream": False,
        }
        res_fin = client.post("/v1/chat/completions", json=payload_fin)
        assert res_fin.status_code == 200, f"Financial chat completion failed: {res_fin.text}"
        data_fin = res_fin.json()
        text_fin = data_fin["choices"][0]["message"]["content"]
        print(f"  Financial Response    : {text_fin[:120]}...")
        assert len(text_fin) > 10, "Empty response from financial expert!"

        # --- Test 3: POST /v1/chat/completions (PostgreSQL Expert - In-Place Swap) ---
        print("\n--- Test 3: POST /v1/chat/completions (PostgreSQL Expert) ---")
        vram_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        payload_pg = {
            "model": "postgresql",
            "messages": [
                {"role": "user", "content": "Design a PostgreSQL schema for storing client embeddings using pgvector."}
            ],
            "max_tokens": 64,
            "stream": False,
        }
        res_pg = client.post("/v1/chat/completions", json=payload_pg)
        assert res_pg.status_code == 200, f"PostgreSQL chat completion failed: {res_pg.text}"
        vram_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        vram_churn_bytes = abs(vram_after - vram_before)

        data_pg = res_pg.json()
        text_pg = data_pg["choices"][0]["message"]["content"]
        print(f"  PostgreSQL Response   : {text_pg[:120]}...")
        print(f"  VRAM Allocation Churn : {vram_churn_bytes} bytes")
        assert len(text_pg) > 10, "Empty response from postgresql expert!"

        # --- Test 4: POST /v1/chat/completions (Astral Expert - In-Place Swap) ---
        print("\n--- Test 4: POST /v1/chat/completions (Astral Expert) ---")
        payload_astral = {
            "model": "astral",
            "messages": [
                {
                    "role": "user",
                    "content": "Write a Python FastAPI similarity search endpoint using uv and ruff standards.",
                }
            ],
            "max_tokens": 64,
            "stream": False,
        }
        res_astral = client.post("/v1/chat/completions", json=payload_astral)
        assert res_astral.status_code == 200, f"Astral chat completion failed: {res_astral.text}"
        data_astral = res_astral.json()
        text_astral = data_astral["choices"][0]["message"]["content"]
        print(f"  Astral Response       : {text_astral[:120]}...")
        assert len(text_astral) > 10, "Empty response from astral expert!"

        # --- Test 5: POST /v1/chat/completions (SSE Streaming stream=True) ---
        print("\n--- Test 5: POST /v1/chat/completions (SSE Streaming stream=True) ---")
        payload_stream = {
            "model": "postgresql",
            "messages": [{"role": "user", "content": "Explain pgvector index creation."}],
            "max_tokens": 32,
            "stream": True,
        }
        res_stream = client.post("/v1/chat/completions", json=payload_stream)
        assert res_stream.status_code == 200, f"Streaming failed: {res_stream.text}"
        assert "text/event-stream" in res_stream.headers["content-type"], "Invalid Content-Type for SSE!"

        stream_chunks = []
        has_done = False
        for line in res_stream.iter_lines():
            line_str = line.strip()
            if not line_str:
                continue
            if line_str == "data: [DONE]":
                has_done = True
                break
            if line_str.startswith("data: "):
                json_str = line_str[6:]
                chunk_data = json.loads(json_str)
                delta = chunk_data["choices"][0]["delta"]
                if "content" in delta and delta["content"]:
                    stream_chunks.append(delta["content"])

        streamed_text = "".join(stream_chunks)
        print(f"  Streamed Tokens Count : {len(stream_chunks)} chunks")
        print(f"  Streamed Text Snippet : {streamed_text[:100]}...")
        assert len(stream_chunks) > 0, "No SSE tokens received!"
        assert has_done is True, "Missing data: [DONE] SSE termination marker!"

        # Final Capture Count Check
        assert graph_decoder.capture_count == 1, f"CUDA Graph Capture Guard Failed: Count={graph_decoder.capture_count}"
        assert graph_decoder._is_locked is True, "Graph Decoder must remain locked!"

        print("\n==================================================")
        print(" IMB OpenAI REST Server Integration Test Summary")
        print("==================================================")
        print(" GET /v1/models                  : PASSED")
        print(" POST /v1/chat/completions (Fin) : PASSED")
        print(" POST /v1/chat/completions (PG)  : PASSED")
        print(" POST /v1/chat/completions (Ast) : PASSED")
        print(" POST SSE Streaming (stream=true): PASSED")
        print(f" CUDA Graph Capture Count Guard  : PASSED (Capture Count = {graph_decoder.capture_count})")
        print(" Single-Capture Runtime Lock    : PASSED (Lock = True)")

        log_benchmark_metric(
            {
                "experiment": "openai_api_server_integration",
                "status": "PASSED",
                "capture_count": graph_decoder.capture_count,
                "stream_chunks": len(stream_chunks),
            },
            filepath="results/openai_api_server_runs.jsonl",
        )


if __name__ == "__main__":
    run_openai_api_server_tests()
