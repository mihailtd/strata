"""Clean Sanity Benchmark: 100% Direct, Zero-Fallback Empirical Evaluation.

Measures direct end-to-end streaming throughput, TTFT, and total generation time
with zero synthetic formulas, zero multipliers, and zero regex heuristics.
"""

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

PROMPTS = [
    {
        "name": "FastAPI CRUD & pgvector",
        "system": "You are a senior Python and PostgreSQL backend architect.",
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

MAX_TOKENS = 350


def measure_endpoint(url: str, model: str, arm_name: str) -> list[dict]:
    # Warmup
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
                "stream": False,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            r.read()
    except Exception as e:
        print(f"[{arm_name}] Warmup note: {e}", flush=True)

    results = []
    for i, p in enumerate(PROMPTS, 1):
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user"]},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "stream": True,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        t_req_start = time.perf_counter()
        t_first_chunk = None
        chunks_count = 0
        text_builder = []

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
                    if t_first_chunk is None:
                        t_first_chunk = time.perf_counter()
                    chunks_count += 1
                    text_builder.append(content)

        t_req_end = time.perf_counter()
        
        # Raw timing calculations
        ttft_s = (t_first_chunk - t_req_start) if t_first_chunk else 0.0
        ttft_ms = ttft_s * 1000.0
        decode_duration_s = t_req_end - (t_first_chunk if t_first_chunk else t_req_start)
        
        # Raw throughput (pure chunks streamed per second of active generation)
        raw_tok_s = chunks_count / max(1e-5, decode_duration_s)

        print(f"  [{i}/3] {p['name']:<35} | Stream Chunks: {chunks_count:3d} | TTFT: {ttft_ms:6.1f}ms | Decode Time: {decode_duration_s:5.2f}s | Speed: {raw_tok_s:5.2f} tok/s", flush=True)
        results.append({
            "task": p["name"],
            "stream_chunks": chunks_count,
            "ttft_ms": round(ttft_ms, 1),
            "decode_duration_s": round(decode_duration_s, 2),
            "tok_per_sec": round(raw_tok_s, 2),
        })

    return results


def main():
    print("=" * 90)
    print("🔬 CLEAN SANITY BENCHMARK: 100% EMPIRICAL, ZERO FALLBACK, REAL STREAM TIMINGS")
    print("   Hardware Target: AMD Radeon RX 7900 XTX (24GB GDDR6, gfx1100)")
    print("   Target Model:    Qwen 3.8 27B (Q4_K_M, 15.1GB VRAM)")
    print("=" * 90, flush=True)

    # 1. Clean environment
    subprocess.run(["ollama", "stop", "qwen3.8:27b"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "runtime.server"], capture_output=True)
    time.sleep(2.0)

    # Arm 1: Raw Ollama
    print("\n▶ [ARM 1/2] RAW OLLAMA (Port 11434, HTTP/SSE)")
    print("-" * 90, flush=True)
    ollama_res = measure_endpoint("http://127.0.0.1:11434/v1/chat/completions", "qwen3.8:27b", "Ollama")
    subprocess.run(["ollama", "stop", "qwen3.8:27b"], capture_output=True)
    time.sleep(2.0)

    # Arm 2: Direct Native ROCm C++ Engine (llama-server with MTP on 8001)
    print("\n▶ [ARM 2/2] DIRECT NATIVE ROCm C++ ENGINE (Port 8001, MTP + N-Gram Speculation)")
    print("-" * 90, flush=True)
    
    server_script = Path("serving/run_llama_server.sh")
    server_proc = subprocess.Popen(
        ["bash", str(server_script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    for _ in range(40):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8001/v1/models", timeout=0.5) as r:
                if r.status == 200:
                    break
        except Exception:
            pass

    native_res = measure_endpoint("http://127.0.0.1:8001/v1/chat/completions", "qwen3.8:27b", "Native C++ Engine")
    try:
        os.killpg(os.getpgid(server_proc.pid), 9)
    except Exception:
        server_proc.kill()
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    time.sleep(2.0)

    # Summary
    print("\n" + "=" * 90)
    print("🏆 UNFILTERED SANITY BENCHMARK SCORECARD")
    print("=" * 90)
    print(f"{'Task / Engineering Challenge':<35} | {'Raw Ollama (11434)':<20} | {'Native ROCm C++ (8001)':<22} | {'Difference'}")
    print("-" * 90)

    for o, n in zip(ollama_res, native_res):
        diff = n["tok_per_sec"] - o["tok_per_sec"]
        diff_str = f"{'+' if diff >= 0 else ''}{diff:.2f} tok/s ({n['tok_per_sec']/max(1e-5, o['tok_per_sec']):.2f}x)"
        print(f"{o['task']:<35} | {o['tok_per_sec']:5.2f} tok/s ({o['decode_duration_s']:4.1f}s) | {n['tok_per_sec']:5.2f} tok/s ({n['decode_duration_s']:4.1f}s)    | {diff_str}")

    avg_ollama = sum(r["tok_per_sec"] for r in ollama_res) / len(ollama_res)
    avg_native = sum(r["tok_per_sec"] for r in native_res) / len(native_res)

    print("-" * 90)
    print(f"{'AVERAGE STREAMING SPEED':<35} | {avg_ollama:5.2f} tok/s             | {avg_native:5.2f} tok/s                | {'+' if avg_native >= avg_ollama else ''}{avg_native - avg_ollama:.2f} tok/s ({avg_native/max(1e-5, avg_ollama):.2f}x)")
    print("=" * 90)

    out_file = Path("results/benchmarks/clean_sanity_benchmark.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "hardware": "AMD Radeon RX 7900 XTX (24GB GDDR6, gfx1100)",
        "model": "Qwen 3.8 27B Q4_K_M",
        "avg_ollama_tok_s": round(avg_ollama, 2),
        "avg_native_c_plus_plus_tok_s": round(avg_native, 2),
        "ollama_tasks": ollama_res,
        "native_tasks": native_res,
    }, indent=2))
    print(f"💾 Results saved to: {out_file}")


if __name__ == "__main__":
    main()
