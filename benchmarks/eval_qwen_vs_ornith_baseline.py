"""Head-to-Head Baseline Evaluation: Qwen 3.8 27B vs Ornith-1.5 35B MoE.

Tests both models without any LoRAs across the 6 core engineering domains
to evaluate:
1. Modern idioms vs legacy anti-patterns (FastAPI lifespan, Pydantic v2, Python 3.12 syntax, asyncpg HNSW, DuckDB QUALIFY, Astral uv).
2. Speed, latency, and token efficiency.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

TEST_PROMPTS = [
    {
        "domain": "Python Web / FastAPI",
        "id": "fastapi_modern",
        "prompt": "Write a complete production FastAPI application with a database connection pool, a lifespan context manager for startup/shutdown, and Pydantic v2 response models for a User profile.",
        "rubric": {
            "modern_indicators": ["lifespan", "asynccontextmanager", "ConfigDict", "Annotated", "Depends"],
            "legacy_anti_patterns": ["on_event(\"startup\")", "on_event(\"shutdown\")", "class Config:", "orm_mode"],
        }
    },
    {
        "domain": "PostgreSQL & Vector",
        "id": "postgres_vector",
        "prompt": "Write a Python asyncpg function that queries an items table with an HNSW cosine similarity search on a 1536-dim vector, using parameterized queries and returning the top 5 closest items.",
        "rubric": {
            "modern_indicators": ["asyncpg", "<=>", "$1", "create_pool", "acquire"],
            "legacy_anti_patterns": ["psycopg2", "<->", "f\"SELECT", "%s"],
        }
    },
    {
        "domain": "Astral & Tooling",
        "id": "astral_tooling",
        "prompt": "Provide a modern pyproject.toml configuration using Astral uv workspace, strict ruff linter configuration with rule selection, and formatting rules.",
        "rubric": {
            "modern_indicators": ["[tool.uv", "[tool.ruff", "[tool.ruff.lint]", "select =", "target-version"],
            "legacy_anti_patterns": ["[tool.poetry]", "setup.py", "flake8", "isort"],
        }
    },
    {
        "domain": "Python Modern Syntax",
        "id": "python_modern",
        "prompt": "Demonstrate modern Python 3.12 generic classes and generic type aliases using the PEP 695 type parameter syntax.",
        "rubric": {
            "modern_indicators": ["class Box[T]:", "type Number =", "def func[T]", "[T, U]"],
            "legacy_anti_patterns": ["TypeVar(", "Union[", "Optional["],
        }
    },
    {
        "domain": "DuckDB Analytics",
        "id": "duckdb_analytics",
        "prompt": "Write a DuckDB SQL query that reads partitioned Parquet files from 'data/*.parquet', computes window rankings per user, and filters the top record per user using the QUALIFY clause.",
        "rubric": {
            "modern_indicators": ["read_parquet", "QUALIFY", "ROW_NUMBER()", "PARTITION BY"],
            "legacy_anti_patterns": ["SELECT * FROM (SELECT", "WHERE row_num = 1"],
        }
    },
    {
        "domain": "Financial Modeling",
        "id": "financial_risk",
        "prompt": "Implement a Python function to compute Value at Risk (VaR) and Conditional VaR (Expected Shortfall) at 95% confidence from a vector of portfolio asset returns.",
        "rubric": {
            "modern_indicators": ["var", "cvar", "expected_shortfall", "percentile", "numpy", "returns"],
            "legacy_anti_patterns": [],
        }
    }
]


def query_model(model_name: str, prompt: str, max_tokens: int = 400) -> dict:
    url = "http://127.0.0.1:11434/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a principal software engineer. Write clean, production-grade, state-of-the-art code using the latest modern language and framework standards."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    t_start = time.perf_counter()
    t_first = None
    chunks = 0
    full_text_list = []

    with urllib.request.urlopen(req, timeout=90) as resp:
        for line in resp:
            s = line.decode("utf-8").strip()
            if not s.startswith("data: ") or s == "data: [DONE]":
                continue
            try:
                chunk = json.loads(s[6:])
            except Exception:
                continue
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
            if content:
                if t_first is None:
                    t_first = time.perf_counter()
                chunks += 1
                full_text_list.append(content)

    t_end = time.perf_counter()
    full_text = "".join(full_text_list)
    ttft_ms = ((t_first - t_start) * 1000.0) if t_first else 0.0
    decode_s = t_end - (t_first if t_first else t_start)
    tok_s = chunks / max(1e-5, decode_s)

    return {
        "text": full_text,
        "chunks": chunks,
        "ttft_ms": round(ttft_ms, 1),
        "decode_s": round(decode_s, 2),
        "tok_s": round(tok_s, 2),
    }


def grade_response(text: str, rubric: dict) -> dict:
    modern_hits = [k for k in rubric.get("modern_indicators", []) if k in text]
    legacy_hits = [k for k in rubric.get("legacy_anti_patterns", []) if k in text]
    
    score = len(modern_hits) - len(legacy_hits)
    return {
        "modern_hits": modern_hits,
        "legacy_hits": legacy_hits,
        "score": score,
        "is_modern": len(modern_hits) >= len(rubric.get("modern_indicators", [])) * 0.5 and len(legacy_hits) == 0
    }


def main():
    print("=" * 90)
    print("⚖️ BASELINE HEAD-TO-HEAD EVALUATION: QWEN 3.8 27B vs ORNITH-1.5 35B MoE")
    print("   Testing whether stock base models follow modern standards without LoRAs")
    print("=" * 90, flush=True)

    models = [
        ("qwen3.8:27b", "Qwen 3.8 27B (Dense)"),
        ("ornith-1.5:35b", "Ornith-1.5 35B (MoE)"),
    ]

    all_results = {}

    for model_id, model_label in models:
        print(f"\n▶ EVALUATING: {model_label} [{model_id}]")
        print("-" * 90, flush=True)
        model_evals = []

        # Warmup
        query_model(model_id, "Hi", max_tokens=10)

        for i, test in enumerate(TEST_PROMPTS, 1):
            res = query_model(model_id, test["prompt"])
            grade = grade_response(res["text"], test["rubric"])
            
            print(f"  [{i}/6] {test['domain']:<25} | Speed: {res['tok_s']:5.1f} tok/s | TTFT: {res['ttft_ms']:5.1f}ms | Modern: {'✅' if grade['is_modern'] else '⚠️'} (Hits: {grade['modern_hits']}, Anti-patterns: {grade['legacy_hits']})", flush=True)
            
            model_evals.append({
                "test_id": test["id"],
                "domain": test["domain"],
                "speed": res,
                "grade": grade,
                "response_sample": res["text"][:300].replace("\n", " "),
            })

        all_results[model_id] = model_evals

    # Comparison summary
    print("\n" + "=" * 90)
    print("🏆 FINAL COMPARATIVE ARCHITECTURAL & QUALITY SUMMARY")
    print("=" * 90)
    print(f"{'Domain':<25} | {'Qwen 27B Speed & Quality':<30} | {'Ornith 35B Speed & Quality':<30}")
    print("-" * 90)

    qwen_evals = all_results["qwen3.8:27b"]
    ornith_evals = all_results["ornith-1.5:35b"]

    for q, o in zip(qwen_evals, ornith_evals):
        q_str = f"{q['speed']['tok_s']:5.1f} t/s | {'✅ Modern' if q['grade']['is_modern'] else '⚠️ Legacy'}"
        o_str = f"{o['speed']['tok_s']:5.1f} t/s | {'✅ Modern' if o['grade']['is_modern'] else '⚠️ Legacy'}"
        print(f"{q['domain']:<25} | {q_str:<30} | {o_str:<30}")

    print("=" * 90)

    out_file = Path("results/benchmarks/qwen_vs_ornith_baseline_comparison.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"💾 Full evaluation details saved to: {out_file}")


if __name__ == "__main__":
    main()
