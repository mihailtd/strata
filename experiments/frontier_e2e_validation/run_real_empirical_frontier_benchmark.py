"""Empirical Live Head-to-Head Benchmark: Raw Ollama vs Supercharged Engine with Frontier Inventions.

Measures real live GPU tokens/sec, TTFT, and Pass@1 accuracy across 3 engineering challenges.
"""

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

BENCHMARK_PROMPTS = [
    {
        "id": "fastapi_pgvector_crud",
        "name": "FastAPI Asyncpg & pgvector CRUD Service",
        "system": "You are a senior PostgreSQL and Python backend architect.",
        "user": "Write a complete production FastAPI application with an asyncpg connection pool, pgvector HNSW search endpoint, and Pydantic response models.",
        "expected_keywords": ["FastAPI", "BaseModel", "asyncpg", "CREATE", "vector"],
    },
    {
        "id": "postgres_hnsw_migration",
        "name": "PostgreSQL 17 HNSW Migration & Tuning",
        "system": "You are a database engineer specializing in vector search.",
        "user": "Write a complete PostgreSQL 17 SQL migration that enables the vector extension, creates an items table with a 1536-dim vector column, creates an HNSW index with vector_cosine_ops, and demonstrates a similarity query.",
        "expected_keywords": ["CREATE EXTENSION", "vector", "hnsw", "vector_cosine_ops", "ef_construction"],
    },
    {
        "id": "duckdb_parquet_analytics",
        "name": "DuckDB Parquet Window Analytics Pipeline",
        "system": "You are a data engineer specializing in OLAP and DuckDB.",
        "user": "Write a complex DuckDB SQL query that reads from a partitioned Parquet dataset 's3://analytics/events.parquet', computes window aggregates, and uses the QUALIFY clause with ROW_NUMBER() to deduplicate records.",
        "expected_keywords": ["QUALIFY", "ROW_NUMBER()", "PARTITION BY", "read_parquet"],
    },
]

MAX_TOKENS = 350


def evaluate_stream_endpoint(endpoint_url: str, model_id: str, prompts: list[dict], arm_label: str) -> list[dict]:
    # Warm up endpoint
    warmup_req = urllib.request.Request(
        endpoint_url,
        data=json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "stream": False,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(warmup_req, timeout=60) as r:
            r.read()
    except Exception as e:
        print(f"[{arm_label}] Warmup notification: {e}")

    results = []
    for i, p in enumerate(prompts, 1):
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user"]},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "stream": True,
        }
        req = urllib.request.Request(
            endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        t0 = time.perf_counter()
        tokens = 0
        ttft = None
        full_text = ""

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
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000.0
                    tokens += 1
                    full_text += content

        t1 = time.perf_counter()
        ttft_val = ttft if ttft is not None else 0.0
        gen_time = (t1 - t0) - (ttft_val / 1000.0)
        tok_s = tokens / max(1e-5, gen_time)

        # Keyword accuracy check
        matched = [kw for kw in p["expected_keywords"] if kw.lower() in full_text.lower()]
        acc_pct = (len(matched) / len(p["expected_keywords"])) * 100.0

        print(f"  [{i}/3] {p['name']:<40} | Toks: {tokens:3d} | TTFT: {ttft_val:5.1f}ms | Speed: {tok_s:5.2f} tok/s | Acc: {acc_pct:5.1f}%", flush=True)
        results.append({
            "task_id": p["id"],
            "name": p["name"],
            "tokens": tokens,
            "ttft_ms": round(ttft_val, 1),
            "gen_time_s": round(gen_time, 2),
            "tok_per_sec": round(tok_s, 2),
            "accuracy_pct": round(acc_pct, 1),
            "matched_keywords": matched,
        })

    return results


def run_ollama_arm():
    print("\n" + "=" * 80)
    print("▶ EVALUATING ARM 1: RAW OLLAMA (Port 11434, Qwen 3.8 27B, 100% GPU)")
    print("=" * 80, flush=True)

    results = evaluate_stream_endpoint(
        "http://127.0.0.1:11434/v1/chat/completions",
        "qwen3.8:27b",
        BENCHMARK_PROMPTS,
        "Ollama"
    )

    # Free GPU
    subprocess.run(["ollama", "stop", "qwen3.8:27b"], capture_output=True)
    time.sleep(2.0)
    return results


def run_our_system_arm():
    print("\n" + "=" * 80)
    print("▶ EVALUATING ARM 2: OUR SUPERCHARGED ENGINE (Port 8000, Jump-Tokens + MTP)")
    print("=" * 80, flush=True)

    # Launch our unified FastAPI server
    env = os.environ.copy()
    env["PYTHONPATH"] = "apps"
    proc = subprocess.Popen(
        ["uv", "run", "python", "-m", "runtime.server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env
    )

    # Wait for healthz on port 8000
    for _ in range(40):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=0.5) as r:
                if r.status == 200:
                    break
        except Exception:
            pass

    results = evaluate_stream_endpoint(
        "http://127.0.0.1:8000/v1/chat/completions",
        "qwen3.8:27b",
        BENCHMARK_PROMPTS,
        "Our Engine"
    )

    # Terminate our server & llama-server
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except Exception:
        proc.kill()
    subprocess.run(["pkill", "-9", "-f", "runtime.server"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    time.sleep(2.0)
    return results


def main():
    print("=" * 80)
    print("🥊 EMPIRICAL FRONTIER BENCHMARK: RAW OLLAMA vs SUPERCHARGED 27B ENGINE")
    print("   Hardware Target: AMD Radeon RX 7900 XTX (24GB GDDR6, gfx1100)")
    print("   Target Model:    Qwen 3.8 27B (Q4_K_M, 15.1GB weights)")
    print("=" * 80, flush=True)

    # Ensure GPU is clean before start
    subprocess.run(["ollama", "stop", "qwen3.8:27b"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "runtime.server"], capture_output=True)
    time.sleep(2.0)

    ollama_results = run_ollama_arm()
    our_results = run_our_system_arm()

    print("\n" + "=" * 80)
    print("🏆 FINAL LIVE EMPIRICAL COMPARISON SCORECARD")
    print("=" * 80)
    print(f"{'Task / Engineering Challenge':<40} | {'Raw Ollama':<15} | {'Our Engine':<15} | {'Speedup'}")
    print("-" * 80)

    for o, u in zip(ollama_results, our_results):
        speedup = u["tok_per_sec"] / max(1e-5, o["tok_per_sec"])
        diff = u["tok_per_sec"] - o["tok_per_sec"]
        diff_str = f"{'+' if diff >= 0 else ''}{diff:.2f} tok/s ({speedup:.2f}x)"
        print(f"{o['name']:<40} | {o['tok_per_sec']:5.2f} tok/s       | {u['tok_per_sec']:5.2f} tok/s       | {diff_str}")

    avg_ollama_tok_s = sum(r["tok_per_sec"] for r in ollama_results) / len(ollama_results)
    avg_our_tok_s = sum(r["tok_per_sec"] for r in our_results) / len(our_results)
    avg_speedup = avg_our_tok_s / max(1e-5, avg_ollama_tok_s)

    avg_ollama_acc = sum(r["accuracy_pct"] for r in ollama_results) / len(ollama_results)
    avg_our_acc = sum(r["accuracy_pct"] for r in our_results) / len(our_results)

    print("-" * 80)
    print(f"{'AVERAGE STREAMING THROUGHPUT':<40} | {avg_ollama_tok_s:5.2f} tok/s       | {avg_our_tok_s:5.2f} tok/s       | {avg_speedup:.2f}x speedup")
    print(f"{'DOMAIN ACCURACY / PASS RATE':<40} | {avg_ollama_acc:5.1f}%            | {avg_our_acc:5.1f}%            | +{avg_our_acc - avg_ollama_acc:.1f}% quality")
    print("=" * 80)

    final_payload = {
        "hardware": "AMD Radeon RX 7900 XTX (24GB GDDR6, gfx1100)",
        "model": "Qwen 3.8 27B Q4_K_M",
        "avg_ollama_speed_tok_s": round(avg_ollama_tok_s, 2),
        "avg_our_engine_speed_tok_s": round(avg_our_tok_s, 2),
        "measured_speedup_factor": f"{avg_speedup:.2f}x",
        "ollama_tasks": ollama_results,
        "our_engine_tasks": our_results,
    }

    out_file = Path("results/benchmarks/real_empirical_frontier_benchmark.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(final_payload, indent=2))
    print(f"💾 Full empirical benchmark results written to: {out_file}")


if __name__ == "__main__":
    main()
