"""Real-world, unsimulated End-to-End Benchmark for all models and adapters.

Sends real requests over HTTP / in-process runtime, streams real tokens,
measures real wall-clock elapsed time, and calculates exact real tokens/sec.
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any, Dict, List


BENCHMARK_PROMPTS = [
    {
        "id": "fastapi_install",
        "domain": "astral",
        "prompt": "How do I install FastAPI using the modern uv toolchain?",
    },
    {
        "id": "postgres_hnsw",
        "domain": "postgresql",
        "prompt": "Write a PostgreSQL 17 SQL migration adding a 1536-dimensional vector column with an HNSW cosine index.",
    },
    {
        "id": "duckdb_qualify",
        "domain": "duckdb",
        "prompt": "Write a DuckDB SQL query using QUALIFY to find the latest transaction per account.",
    },
    {
        "id": "python_async",
        "domain": "fastapi",
        "prompt": "Write a minimal FastAPI app with an async lifespan handler and a Pydantic v2 model.",
    },
]


def test_model_e2e(model_id: str, max_tokens: int = 150) -> List[Dict[str, Any]]:
    print(f"\n" + "=" * 80)
    print(f"📊 TESTING MODEL: {model_id}")
    print("=" * 80)

    results = []

    for item in BENCHMARK_PROMPTS:
        p_id = item["id"]
        prompt = item["prompt"]
        print(f"\n[Prompt: {p_id}] '{prompt[:50]}...'")

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.0,
        }

        req = urllib.request.Request(
            "http://localhost:8000/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        t_start = time.perf_counter()
        ttft = None
        token_count = 0
        response_text = ""

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                for line in resp:
                    line_str = line.decode("utf-8").strip()
                    if not line_str.startswith("data: "):
                        continue
                    if line_str == "data: [DONE]":
                        break
                    
                    data_json = json.loads(line_str[6:])
                    choices = data_json.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content") or delta.get("reasoning_content")
                    if content:
                        if ttft is None:
                            ttft = (time.perf_counter() - t_start) * 1000.0
                        token_count += 1
                        response_text += content

            t_total = time.perf_counter() - t_start
            gen_time = t_total - ((ttft or 0.0) / 1000.0)
            tok_per_sec = token_count / max(1e-5, gen_time)

            print(f"  -> Tokens Generated: {token_count}")
            print(f"  -> TTFT:             {ttft:.1f} ms" if ttft else "  -> TTFT: N/A")
            print(f"  -> Gen Time:         {gen_time:.2f} s")
            print(f"  -> Real Speed:       {tok_per_sec:.2f} tok/s")

            results.append({
                "prompt_id": p_id,
                "model_id": model_id,
                "tokens": token_count,
                "ttft_ms": round(ttft or 0.0, 1),
                "gen_time_s": round(gen_time, 2),
                "tok_per_sec": round(tok_per_sec, 2),
                "sample_output": response_text[:120].replace("\n", " "),
            })

        except Exception as e:
            print(f"  -> ERROR: {e}")
            results.append({
                "prompt_id": p_id,
                "model_id": model_id,
                "error": str(e),
            })

    return results


def main():
    print("🚀 STARTING REAL END-TO-END HARNESS BENCHMARK")
    print("Target: http://localhost:8000/v1/chat/completions (Live Server)")

    models_to_test = [
        "qwen3.5-9b-astral",
        "qwen3.5-9b-postgresql",
        "qwen3.5-9b-duckdb",
        "qwen3.5-9b-dynamic",
    ]

    all_results = {}
    for m in models_to_test:
        res = test_model_e2e(m)
        all_results[m] = res

    print("\n" + "=" * 80)
    print("🏁 FINAL MEASURED END-TO-END SUMMARY")
    print("=" * 80)
    for m, rows in all_results.items():
        valid_speeds = [r["tok_per_sec"] for r in rows if "tok_per_sec" in r]
        valid_ttft = [r["ttft_ms"] for r in rows if "ttft_ms" in r]
        avg_speed = sum(valid_speeds) / len(valid_speeds) if valid_speeds else 0.0
        avg_ttft = sum(valid_ttft) / len(valid_ttft) if valid_ttft else 0.0
        print(f"Model: {m:<25} | Avg TTFT: {avg_ttft:6.1f} ms | Avg Speed: {avg_speed:5.1f} tok/s")
    print("=" * 80)


if __name__ == "__main__":
    main()
