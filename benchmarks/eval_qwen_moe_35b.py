"""Evaluation of Official Qwen 3.6 35B MoE (qwen35moe architecture)

Compares official Qwen 3.6 35B MoE against Qwen 3.8 27B Dense and Ornith-1.5 35B MoE.
"""

import json
import time
import urllib.request
from pathlib import Path

TEST_PROMPTS = [
    {
        "domain": "Python Web / FastAPI",
        "prompt": "Write a complete production FastAPI application with a database connection pool, a lifespan context manager for startup/shutdown, and Pydantic v2 response models for a User profile.",
        "rubric": {
            "modern_indicators": ["lifespan", "asynccontextmanager", "ConfigDict", "Annotated", "Depends"],
            "legacy_anti_patterns": ["on_event(\"startup\")", "on_event(\"shutdown\")", "class Config:", "orm_mode"],
        }
    },
    {
        "domain": "PostgreSQL & Vector",
        "prompt": "Write a Python asyncpg function that queries an items table with an HNSW cosine similarity search on a 1536-dim vector, using parameterized queries and returning the top 5 closest items.",
        "rubric": {
            "modern_indicators": ["asyncpg", "<=>", "$1", "create_pool", "acquire"],
            "legacy_anti_patterns": ["psycopg2", "<->", "f\"SELECT", "%s"],
        }
    },
    {
        "domain": "Astral & Tooling",
        "prompt": "Provide a modern pyproject.toml configuration using Astral uv workspace, strict ruff linter configuration with rule selection, and formatting rules.",
        "rubric": {
            "modern_indicators": ["[tool.uv", "[tool.ruff", "[tool.ruff.lint]", "select =", "target-version"],
            "legacy_anti_patterns": ["[tool.poetry]", "setup.py", "flake8", "isort"],
        }
    },
    {
        "domain": "Python Modern Syntax",
        "prompt": "Demonstrate modern Python 3.12 generic classes and generic type aliases using the PEP 695 type parameter syntax.",
        "rubric": {
            "modern_indicators": ["class Box[T]:", "type Number =", "def func[T]", "[T, U]"],
            "legacy_anti_patterns": ["TypeVar(", "Union[", "Optional["],
        }
    },
    {
        "domain": "DuckDB Analytics",
        "prompt": "Write a DuckDB SQL query that reads partitioned Parquet files from 'data/*.parquet', computes window rankings per user, and filters the top record per user using the QUALIFY clause.",
        "rubric": {
            "modern_indicators": ["read_parquet", "QUALIFY", "ROW_NUMBER()", "PARTITION BY"],
            "legacy_anti_patterns": ["SELECT * FROM (SELECT", "WHERE row_num = 1"],
        }
    },
    {
        "domain": "Financial Modeling",
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
    return {
        "modern_hits": modern_hits,
        "legacy_hits": legacy_hits,
        "is_modern": len(modern_hits) >= len(rubric.get("modern_indicators", [])) * 0.5 and len(legacy_hits) == 0
    }


def main():
    print("=" * 90)
    print("⚡ TESTING OFFICIAL QWEN 3.6 35B MoE (qwen35moe architecture)")
    print("=" * 90)

    # Warmup
    query_model("qwen3.6:35b", "Hi", max_tokens=10)

    results = []
    for i, t in enumerate(TEST_PROMPTS, 1):
        res = query_model("qwen3.6:35b", t["prompt"])
        grade = grade_response(res["text"], t["rubric"])
        print(f"  [{i}/6] {t['domain']:<25} | Speed: {res['tok_s']:5.1f} tok/s | TTFT: {res['ttft_ms']:5.1f}ms | Modern: {'✅' if grade['is_modern'] else '⚠️'} (Hits: {grade['modern_hits']}, Anti: {grade['legacy_hits']})", flush=True)
        results.append({
            "domain": t["domain"],
            "speed": res,
            "grade": grade,
        })

    avg_speed = sum(r["speed"]["tok_s"] for r in results) / len(results)
    modern_count = sum(1 for r in results if r["grade"]["is_modern"])

    print("\n" + "=" * 90)
    print(f"Qwen 3.6 35B MoE Average Speed: {avg_speed:5.2f} tok/s | Modern Compliance: {modern_count}/6 ({modern_count/6*100:.1f}%)")
    print("=" * 90)

    out_file = Path("results/benchmarks/qwen3_6_35b_moe_eval.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    print(f"💾 Results saved to: {out_file}")


if __name__ == "__main__":
    main()
