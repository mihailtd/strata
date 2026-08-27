"""Empirical 3-Way Benchmark: Raw Ollama vs Python HTTP Proxy vs Direct Native Harness Plugin.

Tests exact streaming speed, TTFT, and latency across all 3 architectures on AMD Radeon RX 7900 XTX.
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
    },
    {
        "id": "postgres_hnsw_migration",
        "name": "PostgreSQL 17 HNSW Migration & Tuning",
        "system": "You are a database engineer specializing in vector search.",
        "user": "Write a complete PostgreSQL 17 SQL migration that enables the vector extension, creates an items table with a 1536-dim vector column, creates an HNSW index with vector_cosine_ops, and demonstrates a similarity query.",
    },
    {
        "id": "duckdb_parquet_analytics",
        "name": "DuckDB Parquet Window Analytics Pipeline",
        "system": "You are a data engineer specializing in OLAP and DuckDB.",
        "user": "Write a complex DuckDB SQL query that reads from a partitioned Parquet dataset 's3://analytics/events.parquet', computes window aggregates, and uses the QUALIFY clause with ROW_NUMBER() to deduplicate records.",
    },
]

MAX_TOKENS = 350


def eval_http_stream(endpoint_url: str, model_id: str, prompts: list[dict], arm_label: str) -> list[dict]:
    # Warm up
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
        print(f"[{arm_label}] Warmup note: {e}")

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

        t1 = time.perf_counter()
        ttft_val = ttft if ttft is not None else 0.0
        gen_time = (t1 - t0) - (ttft_val / 1000.0)
        tok_s = tokens / max(1e-5, gen_time)

        print(f"  [{i}/3] {p['name']:<40} | Toks: {tokens:3d} | TTFT: {ttft_val:5.1f}ms | Speed: {tok_s:5.2f} tok/s", flush=True)
        results.append({
            "task_id": p["id"],
            "name": p["name"],
            "tokens": tokens,
            "ttft_ms": round(ttft_val, 1),
            "tok_per_sec": round(tok_s, 2),
        })
    return results


def run_ollama_arm():
    print("\n" + "=" * 80)
    print("▶ [ARM 1/3] RAW OLLAMA (Port 11434, HTTP/SSE, 100% GPU)")
    print("=" * 80, flush=True)
    res = eval_http_stream("http://127.0.0.1:11434/v1/chat/completions", "qwen3.8:27b", BENCHMARK_PROMPTS, "Ollama")
    subprocess.run(["ollama", "stop", "qwen3.8:27b"], capture_output=True)
    time.sleep(2.0)
    return res


def run_python_proxy_arm():
    print("\n" + "=" * 80)
    print("▶ [ARM 2/3] OUR PYTHON HTTP PROXY (Port 8000, FastAPI + SSE)")
    print("=" * 80, flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    proc = subprocess.Popen(
        ["uv", "run", "python", "-m", "runtime.server"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env
    )
    for _ in range(40):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=0.5) as r:
                if r.status == 200:
                    break
        except Exception:
            pass

    res = eval_http_stream("http://127.0.0.1:8000/v1/chat/completions", "qwen3.8:27b", BENCHMARK_PROMPTS, "Python Proxy")
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except Exception:
        proc.kill()
    subprocess.run(["pkill", "-9", "-f", "runtime.server"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    time.sleep(2.0)
    return res


def run_direct_native_plugin_arm():
    print("\n" + "=" * 80)
    print("▶ [ARM 3/3] DIRECT NATIVE HARNESS PLUGIN (In-Process C-FFI, Zero HTTP, Jump-Tokens)")
    print("=" * 80, flush=True)

    # Add src to pythonpath
    import sys
    sys.path.insert(0, "src")
    from harness.native_plugin.harness_provider import NativeHarnessDirectProvider

    provider = NativeHarnessDirectProvider(model_id="qwen3.8:27b", enable_jump_tokens=True)

    results = []
    for i, p in enumerate(BENCHMARK_PROMPTS, 1):
        step_res = provider.run_agent_step(
            instruction=p["user"],
            system_prompt=p["system"],
            max_tokens=MAX_TOKENS
        )
        print(f"  [{i}/3] {p['name']:<40} | Toks: {step_res['tokens_generated']:3d} | TTFT: {step_res['ttft_ms']:5.1f}ms | Speed: {step_res['tok_per_sec']:5.2f} tok/s", flush=True)
        results.append({
            "task_id": p["id"],
            "name": p["name"],
            "tokens": step_res["tokens_generated"],
            "ttft_ms": step_res["ttft_ms"],
            "tok_per_sec": step_res["tok_per_sec"],
        })

    return results


def main():
    print("=" * 80)
    print("🥊 3-WAY EMPIRICAL COMPARISON: OLLAMA vs PYTHON PROXY vs DIRECT NATIVE PLUGIN")
    print("   Hardware: AMD Radeon RX 7900 XTX (24GB GDDR6, gfx1100)")
    print("   Model:    Qwen 3.8 27B (Q4_K_M, 15.1GB weights)")
    print("=" * 80, flush=True)

    subprocess.run(["ollama", "stop", "qwen3.8:27b"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "runtime.server"], capture_output=True)
    time.sleep(2.0)

    ollama_res = run_ollama_arm()
    proxy_res = run_python_proxy_arm()
    direct_res = run_direct_native_plugin_arm()

    print("\n" + "=" * 90)
    print("🏆 FINAL 3-WAY EMPIRICAL ARCHITECTURE COMPARISON SCORECARD")
    print("=" * 90)
    print(f"{'Task / Challenge':<35} | {'Raw Ollama':<12} | {'Python Proxy':<14} | {'Direct Plugin':<15} | {'Speedup vs Ollama'}")
    print("-" * 90)

    for o, p, d in zip(ollama_res, proxy_res, direct_res):
        speedup = d["tok_per_sec"] / max(1e-5, o["tok_per_sec"])
        diff_str = f"{speedup:.2f}x ({'+' if d['tok_per_sec'] >= o['tok_per_sec'] else ''}{d['tok_per_sec'] - o['tok_per_sec']:.1f} tok/s)"
        print(f"{o['name']:<35} | {o['tok_per_sec']:5.1f} tok/s  | {p['tok_per_sec']:5.1f} tok/s    | {d['tok_per_sec']:5.1f} tok/s       | {diff_str}")

    avg_ollama = sum(r["tok_per_sec"] for r in ollama_res) / len(ollama_res)
    avg_proxy = sum(r["tok_per_sec"] for r in proxy_res) / len(proxy_res)
    avg_direct = sum(r["tok_per_sec"] for r in direct_res) / len(direct_res)
    total_speedup = avg_direct / max(1e-5, avg_ollama)

    print("-" * 90)
    print(f"{'AVERAGE STREAMING THROUGHPUT':<35} | {avg_ollama:5.1f} tok/s  | {avg_proxy:5.1f} tok/s    | {avg_direct:5.1f} tok/s       | 🚀 {total_speedup:.2f}x speedup")
    print("=" * 90)

    final_payload = {
        "hardware": "AMD Radeon RX 7900 XTX (24GB GDDR6, gfx1100)",
        "model": "Qwen 3.8 27B Q4_K_M",
        "avg_ollama_tok_s": round(avg_ollama, 2),
        "avg_python_proxy_tok_s": round(avg_proxy, 2),
        "avg_direct_plugin_tok_s": round(avg_direct, 2),
        "measured_speedup_factor": f"{total_speedup:.2f}x",
        "tasks": {
            "ollama": ollama_res,
            "python_proxy": proxy_res,
            "direct_plugin": direct_res,
        }
    }

    out_file = Path("results/benchmarks/direct_native_plugin_scorecard.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(final_payload, indent=2))
    print(f"💾 Full results saved to: {out_file}")


if __name__ == "__main__":
    main()
