"""DeepSeek Harness Python SDK Head-to-Head Comparative Benchmark.

Compares Baseline 27B vs Supercharged 27B Specialist LoRAs across
multi-turn SWE and domain engineering challenges.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig

BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-local-imb-key")
TEMP_ROOT = Path("scratch/harness_head_to_head_sandboxes")

TEST_TASKS = [
    {
        "task_id": "astral_fastapi_setup",
        "domain": "astral",
        "instruction": (
            "Create a modern Python backend project using Astral tooling. "
            "Initialize pyproject.toml with uv, add fastapi and uvicorn dependencies, "
            "and create a standard project layout with src/."
        ),
        "expected_keywords": ["uv", "pyproject.toml", "fastapi", "uvicorn"],
    },
    {
        "task_id": "postgres_hnsw_migration",
        "domain": "postgresql",
        "instruction": (
            "Write a complete PostgreSQL 17 SQL migration that enables the vector extension, "
            "adds a 1536-dimensional embedding column to an items table, and creates an HNSW index "
            "using vector_cosine_ops with m=16 and ef_construction=64."
        ),
        "expected_keywords": [
            "CREATE EXTENSION",
            "vector",
            "hnsw",
            "vector_cosine_ops",
            "ef_construction",
        ],
    },
    {
        "task_id": "duckdb_qualify_analytics",
        "domain": "duckdb",
        "instruction": (
            "Write a DuckDB SQL query that scans customer orders from a Parquet dataset 's3://bucket/orders.parquet', "
            "calculates row numbers per customer ordered by order_date descending, "
            "and filters for the most recent order per customer using the QUALIFY window clause."
        ),
        "expected_keywords": ["QUALIFY", "ROW_NUMBER()", "PARTITION BY", "read_parquet"],
    },
]


def run_single_eval(model_id: str, task: dict) -> dict:
    task_id = task["task_id"]
    instruction = task["instruction"]
    expected_keywords = task["expected_keywords"]

    workspace_dir = TEMP_ROOT / model_id.replace(":", "_") / task_id / "workspace"
    session_dir = TEMP_ROOT / model_id.replace(":", "_") / task_id / "session"

    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    if session_dir.exists():
        shutil.rmtree(session_dir)

    workspace_dir.mkdir(parents=True, exist_ok=True)
    session_dir.mkdir(parents=True, exist_ok=True)

    config = DeepSeekHarnessConfig(
        provider="deepseek-official",
        base_url=BASE_URL,
        model=model_id,
        api_key=API_KEY,
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

    matched_keywords = [k for k in expected_keywords if k.lower() in final_text.lower()]
    accuracy = len(matched_keywords) / len(expected_keywords) if expected_keywords else 1.0
    approx_tokens = int(len(final_text.split()) * 1.3)
    speed = approx_tokens / max(1e-5, elapsed)

    print(f"[{model_id}] -> Task: {task_id:<25} | Acc: {accuracy*100:5.1f}% | Time: {elapsed:5.2f}s | Speed: {speed:5.1f} tok/s", flush=True)

    return {
        "task_id": task_id,
        "model_id": model_id,
        "elapsed_s": round(elapsed, 2),
        "approx_tokens": approx_tokens,
        "effective_tok_per_sec": round(speed, 2),
        "accuracy_score": round(accuracy, 2),
        "matched_keywords": matched_keywords,
        "error": error,
    }


def main():
    print("=" * 80)
    print("🥊 DEEPSEEK HARNESS PYTHON SDK HEAD-TO-HEAD COMPARATIVE BENCHMARK")
    print("=" * 80)

    arms = {
        "baseline_27b": "qwen3.8-27b-base",
        "specialist_27b": "qwen3.8:27b",
    }

    results = {}
    for arm_name, model_id in arms.items():
        print(f"\n--- Running Arm: {arm_name} ({model_id}) ---")
        arm_res = []
        for task in TEST_TASKS:
            res = run_single_eval(model_id, task)
            arm_res.append(res)
        results[arm_name] = arm_res

    out_path = Path("results/benchmarks/deepseek_harness_head_to_head_scorecard.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 80)
    print("🏆 FINAL HEAD-TO-HEAD COMPARATIVE SCORECARD")
    print("=" * 80)
    base_acc = sum(r["accuracy_score"] for r in results["baseline_27b"]) / len(TEST_TASKS) * 100.0
    spec_acc = sum(r["accuracy_score"] for r in results["specialist_27b"]) / len(TEST_TASKS) * 100.0
    base_speed = sum(r["effective_tok_per_sec"] for r in results["baseline_27b"]) / len(TEST_TASKS)
    spec_speed = sum(r["effective_tok_per_sec"] for r in results["specialist_27b"]) / len(TEST_TASKS)

    print(f"Baseline 27B Arm:   Accuracy: {base_acc:5.1f}% | Avg Effective Speed: {base_speed:5.1f} tok/s")
    print(f"Specialist 27B Arm: Accuracy: {spec_acc:5.1f}% | Avg Effective Speed: {spec_speed:5.1f} tok/s")
    print(f"Advantage:          Accuracy Lead: +{spec_acc - base_acc:.1f}%")
    print("=" * 80)
    print(f"💾 Full results saved to: {out_path}")


if __name__ == "__main__":
    main()
