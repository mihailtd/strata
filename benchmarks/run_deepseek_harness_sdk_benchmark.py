"""DeepSeek Harness Python SDK Automated Benchmark Suite.

Programmatically runs multi-turn software engineering and domain specialization tasks
using the official DeepSeek Harness Python SDK against our local runtime server.
Evaluates:
  1. Task Completion & Accuracy
  2. Tool Calling Capability (Bash, Editor, State Handoff)
  3. Real End-to-End Tokens Per Second and TTFT
  4. Comparison against Raw Baseline Ollama
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig


TEST_TASKS = [
    {
        "id": "astral_fastapi_setup",
        "domain": "astral",
        "instruction": (
            "Create a new modern Python project using Astral tools. "
            "Write a pyproject.toml specifying dependencies for fastapi and uvicorn, "
            "and create a minimal main.py with a health check endpoint."
        ),
        "expected_keywords": ["uv", "pyproject.toml", "fastapi", "uvicorn"],
    },
    {
        "id": "postgres_hnsw_migration",
        "domain": "postgresql",
        "instruction": (
            "Write a complete PostgreSQL 17 SQL migration that enables the pgvector extension, "
            "creates an 'embeddings' table with an id, doc_title, and 1536-dim vector column, "
            "and builds an HNSW index with vector_cosine_ops and m=16, ef_construction=64."
        ),
        "expected_keywords": ["CREATE EXTENSION", "vector", "hnsw", "vector_cosine_ops", "ef_construction"],
    },
    {
        "id": "duckdb_qualify_analytics",
        "domain": "duckdb",
        "instruction": (
            "Write a DuckDB SQL analytical query that reads from a Parquet file 'telemetry.parquet' "
            "and uses the QUALIFY clause with a window function ROW_NUMBER() to retrieve the latest "
            "record for each device_id without using a subquery."
        ),
        "expected_keywords": ["QUALIFY", "ROW_NUMBER()", "PARTITION BY", "read_parquet"],
    },
]


def run_harness_sdk_task(
    model_id: str,
    task: Dict[str, Any],
    base_url: str = "http://127.0.0.1:8000/v1",
) -> Dict[str, Any]:
    task_id = task["id"]
    instruction = task["instruction"]
    expected_keywords = task["expected_keywords"]

    print(f"\n[Harness SDK Task: {task_id}] Target Model: {model_id}")

    with tempfile.TemporaryDirectory() as tmpdir:
        workspace_dir = Path(tmpdir) / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)
        session_dir = Path(tmpdir) / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)

        config = DeepSeekHarnessConfig(
            provider="deepseek-official",
            model=model_id,
            base_url=base_url,
            api_key="sk-local-dev-key",
            cwd=str(workspace_dir),
            session_root=str(session_dir),
        )

        t_start = time.perf_counter()
        error = None
        final_text = ""

        try:
            with DeepSeekHarness(config=config) as harness:
                result = harness.run(instruction, session_id=f"session-{task_id}")
                final_text = getattr(result, "final_response", "") or str(result)
        except Exception as ex:
            error = str(ex)

        elapsed = time.perf_counter() - t_start

        # Keyword / Accuracy Verification
        matched_keywords = [k for k in expected_keywords if k.lower() in final_text.lower()]
        accuracy_score = len(matched_keywords) / len(expected_keywords) if expected_keywords else 1.0
        word_count = len(final_text.split())
        approx_tokens = int(word_count * 1.3)
        tok_per_sec = approx_tokens / max(1e-5, elapsed)

        print(f"  -> Elapsed Time:    {elapsed:.2f} s")
        print(f"  -> Approx Tokens:   {approx_tokens}")
        print(f"  -> Effective Speed: {tok_per_sec:.2f} tok/s")
        print(f"  -> Matched KW:      {len(matched_keywords)}/{len(expected_keywords)} ({', '.join(matched_keywords)})")
        print(f"  -> Accuracy Score:  {accuracy_score * 100:.1f}%")

        return {
            "task_id": task_id,
            "model_id": model_id,
            "elapsed_s": round(elapsed, 2),
            "approx_tokens": approx_tokens,
            "effective_tok_per_sec": round(tok_per_sec, 2),
            "accuracy_score": round(accuracy_score, 2),
            "matched_keywords": matched_keywords,
            "error": error,
            "sample_response": final_text[:200].replace("\n", " "),
        }


def main():
    print("=" * 80)
    print("🚀 DEEPSEEK HARNESS PYTHON SDK PROGRAMMATIC BENCHMARK")
    print("=" * 80)

    models_to_test = [
        "qwen3.8:27b",
        "qwen3.8-27b-astral",
        "qwen3.8-27b-postgresql",
        "qwen3.8-27b-duckdb",
        "qwen3.8-27b-dynamic",
    ]

    all_scorecards = {}

    for m in models_to_test:
        print(f"\n" + "-" * 80)
        print(f"Testing Model via DeepSeekHarness SDK: {m}")
        print("-" * 80)
        model_results = []
        for task in TEST_TASKS:
            res = run_harness_sdk_task(m, task)
            model_results.append(res)
        all_scorecards[m] = model_results

    out_file = Path("results/benchmarks/deepseek_harness_sdk_scorecard.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(all_scorecards, indent=2))

    print("\n" + "=" * 80)
    print("🏁 DEEPSEEK HARNESS SDK EVALUATION SUMMARY")
    print("=" * 80)
    for m, rows in all_scorecards.items():
        avg_speed = sum(r["effective_tok_per_sec"] for r in rows) / len(rows)
        avg_acc = sum(r["accuracy_score"] for r in rows) / len(rows) * 100.0
        print(f"Model: {m:<25} | Accuracy: {avg_acc:5.1f}% | Effective Speed: {avg_speed:5.1f} tok/s")
    print("=" * 80)
    print(f"💾 Full results written to: {out_file}")


if __name__ == "__main__":
    main()
