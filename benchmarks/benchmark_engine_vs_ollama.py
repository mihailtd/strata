"""Simple head-to-head empirical benchmark: Tuned Native ROCm Engine vs Raw Ollama.

Tests exact streaming token generation speed (tok/s) and TTFT across 3 tasks
using identical OpenAI /v1/chat/completions endpoints.
"""

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

BENCHMARK_PROMPTS = [
    {
        "name": "FastAPI Asyncpg & pgvector",
        "system": "You are a senior PostgreSQL and Python backend architect.",
        "user": "Write a complete production FastAPI application with an asyncpg connection pool, pgvector HNSW search endpoint, and Pydantic response models.",
    },
    {
        "name": "PostgreSQL 17 HNSW Migration",
        "system": "You are a database engineer specializing in vector search.",
        "user": "Write a complete PostgreSQL 17 SQL migration that enables the vector extension, creates an items table with a 1536-dim vector column, creates an HNSW index with vector_cosine_ops, and demonstrates a similarity query.",
    },
    {
        "name": "DuckDB Parquet Window Analytics",
        "system": "You are a data engineer specializing in OLAP and DuckDB.",
        "user": "Write a complex DuckDB SQL query that reads from a partitioned Parquet dataset 's3://analytics/events.parquet', computes window aggregates, and uses the QUALIFY clause with ROW_NUMBER() to deduplicate records.",
    },
]

MAX_TOKENS = 300


def run_openai_stream_eval(endpoint_url: str, model_name: str, prompts: list[dict]) -> list[dict]:
    # Warm up endpoint
    warmup_req = urllib.request.Request(
        endpoint_url,
        data=json.dumps({
            "model": model_name,
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
        print(f"Warmup note: {e}")

    results = []
    for i, p in enumerate(prompts, 1):
        payload = {
            "model": model_name,
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

        with urllib.request.urlopen(req, timeout=60) as resp:
            for line in resp:
                s = line.decode("utf-8").strip()
                if not s.startswith("data: ") or s == "data: [DONE]":
                    continue
                chunk = json.loads(s[6:])
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
                if content:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000.0
                    tokens += 1

        t1 = time.perf_counter()
        ttft_val = ttft if ttft is not None else 0.0
        gen_time = (t1 - t0) - (ttft_val / 1000.0)
        tok_s = tokens / max(1e-5, gen_time)

        print(f"  [{i}/3] {p['name']:<35} | Tokens: {tokens:3d} | TTFT: {ttft_val:5.1f} ms | Speed: {tok_s:5.2f} tok/s", flush=True)
        results.append({
            "name": p["name"],
            "tokens": tokens,
            "ttft_ms": round(ttft_val, 1),
            "gen_time_s": round(gen_time, 2),
            "tok_per_sec": round(tok_s, 2),
        })

    return results


def benchmark_ollama():
    print("\n" + "=" * 80)
    print("▶ BENCHMARKING RAW OLLAMA (Port 11434, 100% GPU offload, Qwen 3.8 27B)")
    print("=" * 80, flush=True)
    
    results = run_openai_stream_eval(
        "http://127.0.0.1:11434/v1/chat/completions",
        "qwen3.8:27b",
        BENCHMARK_PROMPTS
    )

    # Stop Ollama model to free GPU
    subprocess.run(["ollama", "stop", "qwen3.8:27b"], capture_output=True)
    time.sleep(2.0)
    return results


def benchmark_our_engine():
    print("\n" + "=" * 80)
    print("▶ BENCHMARKING OUR TUNED ROCm ENGINE (Port 8001, MTP + N-Gram Speculation)")
    print("=" * 80, flush=True)

    # Launch llama-server via run_llama_server.sh
    script_path = Path("serving/run_llama_server.sh").resolve()
    proc = subprocess.Popen(
        ["bash", str(script_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    # Wait for healthz
    for _ in range(30):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8001/healthz", timeout=0.5) as r:
                if r.status == 200:
                    break
        except Exception:
            pass

    results = run_openai_stream_eval(
        "http://127.0.0.1:8001/v1/chat/completions",
        "qwen3.8:27b",
        BENCHMARK_PROMPTS
    )

    # Terminate server
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except Exception:
        proc.kill()
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    time.sleep(2.0)
    return results


def main():
    print("=" * 80)
    print("🥊 EMPIRICAL GENERATION SPEED BENCHMARK: RAW OLLAMA vs TUNED ROCm ENGINE")
    print("   Target Model: Qwen 3.8 27B (Q4_K_M, 64 layers on AMD Radeon RX 7900 XTX)")
    print("=" * 80, flush=True)

    ollama_res = benchmark_ollama()
    our_res = benchmark_our_engine()

    print("\n" + "=" * 80)
    print("🏆 FINAL COMPARATIVE TOKENS/SECOND SCORECARD")
    print("=" * 80)
    print(f"{'Task / Prompt':<35} | {'Raw Ollama':<15} | {'Our Tuned Engine':<18} | {'Difference'}")
    print("-" * 80)

    for o, u in zip(ollama_res, our_res):
        diff = u["tok_per_sec"] - o["tok_per_sec"]
        diff_str = f"{'+' if diff >= 0 else ''}{diff:.2f} tok/s"
        print(f"{o['name']:<35} | {o['tok_per_sec']:5.2f} tok/s       | {u['tok_per_sec']:5.2f} tok/s         | {diff_str}")

    avg_ollama = sum(r["tok_per_sec"] for r in ollama_res) / len(ollama_res)
    avg_our = sum(r["tok_per_sec"] for r in our_res) / len(our_res)
    avg_diff = avg_our - avg_ollama

    print("-" * 80)
    print(f"{'AVERAGE THROUGHPUT':<35} | {avg_ollama:5.2f} tok/s       | {avg_our:5.2f} tok/s         | {'+' if avg_diff >= 0 else ''}{avg_diff:.2f} tok/s")
    print("=" * 80)


if __name__ == "__main__":
    main()
