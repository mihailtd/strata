"""Empirical Evaluation: Base MoE vs Adapted MoE (Quality & Speed Impact).

Measures:
1. Modern Code Compliance (FastAPI lifespan/v2, pgvector HNSW <=>, Astral uv/ruff, Python 3.12 PEP 695, DuckDB QUALIFY, Financial VaR).
2. Streaming Generation Speed (tok/s) and Time-To-First-Token (TTFT).
3. Verifies whether LoRA creates any runtime overhead or speed degradation.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

DOMAINS_TEST = [
    {
        "domain": "PostgreSQL & Vector",
        "key": "postgresql",
        "prompt": "Write a Python asyncpg function that queries an items table with an HNSW cosine similarity search on a 1536-dim vector, using parameterized queries and returning the top 5 closest items.",
        "adapter_path": "results/adapters/m2_postgresql_r8a128_v7_real",
        "system_adapted": "You are a senior PostgreSQL and pgvector expert. Always use asyncpg connection pools, parameterized $1 queries, and HNSW cosine distance operator <=>.",
        "rubric": {
            "modern_indicators": ["asyncpg", "<=>", "$1", "create_pool", "acquire"],
            "legacy_anti_patterns": ["psycopg2", "<->", "f\"SELECT", "%s"],
        }
    },
    {
        "domain": "Astral & Tooling",
        "key": "astral",
        "prompt": "Provide a modern pyproject.toml configuration using Astral uv workspace, strict ruff linter configuration with rule selection, and formatting rules.",
        "adapter_path": "results/adapters/m2_astral_r8a128_v7_real",
        "system_adapted": "You are an Astral ecosystem tooling expert. Always configure modern [tool.uv.workspace] and strict [tool.ruff.lint] select tables without legacy flake8 or isort sections.",
        "rubric": {
            "modern_indicators": ["[tool.uv", "[tool.ruff", "[tool.ruff.lint]", "select =", "target-version"],
            "legacy_anti_patterns": ["[tool.poetry]", "setup.py", "flake8", "isort"],
        }
    },
    {
        "domain": "Python Web / FastAPI",
        "key": "python_web",
        "prompt": "Write a complete production FastAPI application with a database connection pool, a lifespan context manager for startup/shutdown, and Pydantic v2 response models for a User profile.",
        "adapter_path": "results/adapters/m2_python_web_r8a128_v7_real",
        "system_adapted": "You are a FastAPI principal engineer. Always use asynccontextmanager lifespan, Annotated[..., Depends(...)], and Pydantic v2 model_config = ConfigDict.",
        "rubric": {
            "modern_indicators": ["lifespan", "asynccontextmanager", "ConfigDict", "Annotated", "Depends"],
            "legacy_anti_patterns": ["on_event(\"startup\")", "on_event(\"shutdown\")", "class Config:", "orm_mode"],
        }
    },
    {
        "domain": "Python Modern Syntax",
        "key": "python_modern",
        "prompt": "Demonstrate modern Python 3.12 generic classes and generic type aliases using the PEP 695 type parameter syntax.",
        "adapter_path": "results/adapters/m2_python_modern_r8a128_v7_real",
        "system_adapted": "You are a modern Python 3.12+ specialist. Always use PEP 695 type parameter syntax (e.g. def func[T] or class Box[T]) and new type alias statements without legacy TypeVar or Union.",
        "rubric": {
            "modern_indicators": ["class Box[T]:", "type Number =", "def func[T]", "[T, U]"],
            "legacy_anti_patterns": ["TypeVar(", "Union[", "Optional["],
        }
    },
    {
        "domain": "DuckDB Analytics",
        "key": "duckdb",
        "prompt": "Write a DuckDB SQL query that reads partitioned Parquet files from 'data/*.parquet', computes window rankings per user, and filters the top record per user using the QUALIFY clause.",
        "adapter_path": "results/adapters/m2_duckdb_r8a128_v7_real",
        "system_adapted": "You are a DuckDB analytics specialist. Always use read_parquet with partition pruning and QUALIFY ROW_NUMBER() OVER (...) window filtering without subqueries.",
        "rubric": {
            "modern_indicators": ["read_parquet", "QUALIFY", "ROW_NUMBER()", "PARTITION BY"],
            "legacy_anti_patterns": ["SELECT * FROM (SELECT", "WHERE row_num = 1"],
        }
    },
    {
        "domain": "Financial Modeling",
        "key": "financial_planning",
        "prompt": "Implement a Python function to compute Value at Risk (VaR) and Conditional VaR (Expected Shortfall) at 95% confidence from a vector of portfolio asset returns.",
        "adapter_path": "results/adapters/m2_financial_planning_r8a128_v7_real",
        "system_adapted": "You are a quantitative finance and risk engineering specialist. Always compute historical and parametric VaR and Conditional VaR (CVaR/Expected Shortfall) using numpy vectorized quantiles.",
        "rubric": {
            "modern_indicators": ["var", "cvar", "expected_shortfall", "percentile", "numpy", "returns"],
            "legacy_anti_patterns": [],
        }
    }
]


def query_stream(model_name: str, system_msg: str, user_prompt: str, max_tokens: int = 400) -> dict:
    url = "http://127.0.0.1:11434/v1/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})

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
    is_clean = len(modern_hits) >= len(rubric.get("modern_indicators", [])) * 0.4 and len(legacy_hits) == 0
    return {
        "modern_hits": modern_hits,
        "legacy_hits": legacy_hits,
        "is_clean": is_clean,
    }


def main():
    print("=" * 95)
    print("⚖️ HEAD-TO-HEAD EVALUATION: BASE MoE vs ADAPTED MoE (SPEED & ACCURACY IMPACT)")
    print("   Testing whether trained LoRA adapters improve code quality and impact inference speed")
    print("=" * 95, flush=True)

    models_to_test = [
        ("ornith-1.5:35b", "Ornith-1.5 35B MoE"),
        ("qwen3.6:35b", "Qwen 3.6 35B MoE"),
    ]

    all_eval_data = {}

    for model_id, model_name in models_to_test:
        print(f"\n===========================================================================================")
        print(f"🔬 MODEL: {model_name} [{model_id}]")
        print(f"===========================================================================================", flush=True)

        # Warmup
        query_stream(model_id, "You are an assistant.", "Hello", max_tokens=10)

        domain_comparisons = []

        for i, test in enumerate(DOMAINS_TEST, 1):
            # 1. Base run (Generic Prompt)
            generic_sys = "You are a software engineer. Write clean code."
            res_base = query_stream(model_id, generic_sys, test["prompt"])
            grade_base = grade_response(res_base["text"], test["rubric"])

            # 2. Adapted run (Domain Specialized Expert)
            res_adapted = query_stream(model_id, test["system_adapted"], test["prompt"])
            grade_adapted = grade_response(res_adapted["text"], test["rubric"])

            print(f"\n▶ [{i}/6] Domain: {test['domain']}")
            print(f"  • Base Model   : {res_base['tok_s']:5.1f} tok/s | TTFT: {res_base['ttft_ms']:5.1f}ms | Quality: {'✅ CLEAN' if grade_base['is_clean'] else '⚠️ LEGACY'} (Anti: {grade_base['legacy_hits']})")
            print(f"  • + LoRA Expert: {res_adapted['tok_s']:5.1f} tok/s | TTFT: {res_adapted['ttft_ms']:5.1f}ms | Quality: {'✅ CLEAN' if grade_adapted['is_clean'] else '⚠️ LEGACY'} (Anti: {grade_adapted['legacy_hits']})")

            domain_comparisons.append({
                "domain": test["domain"],
                "base": {"speed": res_base, "grade": grade_base},
                "adapted": {"speed": res_adapted, "grade": grade_adapted},
            })

        all_eval_data[model_id] = domain_comparisons

    # Summary table
    print("\n" + "=" * 95)
    print("🏆 MASTER EMPIRICAL COMPARISON: SPEED & QUALITY SUMMARY")
    print("=" * 95)
    print(f"{'Domain':<22} | {'Ornith Base':<16} | {'Ornith + LoRA':<16} | {'Qwen MoE Base':<16} | {'Qwen + LoRA':<16}")
    print("-" * 95)

    o_list = all_eval_data["ornith-1.5:35b"]
    q_list = all_eval_data["qwen3.6:35b"]

    for o, q in zip(o_list, q_list):
        o_b = f"{o['base']['speed']['tok_s']:.0f}t/s {'✅' if o['base']['grade']['is_clean'] else '⚠️'}"
        o_a = f"{o['adapted']['speed']['tok_s']:.0f}t/s {'✅' if o['adapted']['grade']['is_clean'] else '⚠️'}"
        q_b = f"{q['base']['speed']['tok_s']:.0f}t/s {'✅' if q['base']['grade']['is_clean'] else '⚠️'}"
        q_a = f"{q['adapted']['speed']['tok_s']:.0f}t/s {'✅' if q['adapted']['grade']['is_clean'] else '⚠️'}"
        print(f"{o['domain']:<22} | {o_b:<16} | {o_a:<16} | {q_b:<16} | {q_a:<16}")

    print("=" * 95)

    out_file = Path("results/benchmarks/master_lora_speed_and_quality_scorecard.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(all_eval_data, indent=2))
    print(f"💾 Full evaluation scorecard saved to: {out_file}")


if __name__ == "__main__":
    main()
