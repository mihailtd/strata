"""End-to-End Real-World Evaluation: Dual-Engine 27B & KV Cache Optimization.

Benchmarks our real use cases across domain expert prompts comparing:
1. ROCm C++ HIP GGUF Engine (llama-server with Q8_0 KV cache)
2. Pure Native Triton Engine (with PreallocatedKVCache in BF16 and Q8_0)
3. Direct comparison of our custom runtime: Before (NaiveCatBF16) vs Now (PreallocatedBF16 & PreallocatedQ8)
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime.native_27b_engine import (
    EngineConfig27B,
    Native27BEngine,
    PreallocatedKVCache,
    Qwen35FullAttentionBlock,
)


REAL_USECASE_PROMPTS = [
    {
        "id": "astral_uv_packaging",
        "domain": "Astral Python Toolchain",
        "prompt": "How do I initialize a new Python project using uv with pyproject.toml and add FastAPI and Pydantic dependencies?",
    },
    {
        "id": "postgres_pgvector_hnsw",
        "domain": "PostgreSQL 17 & Vector DB",
        "prompt": "Write a PostgreSQL 17 SQL migration adding a 1536-dimensional vector column with an HNSW cosine index.",
    },
    {
        "id": "duckdb_qualify_window",
        "domain": "DuckDB Vectorized Analytics",
        "prompt": "Write a DuckDB query to parse a Parquet file, compute 7-day rolling average revenue per customer, and filter using QUALIFY.",
    },
    {
        "id": "fastapi_async_lifespan",
        "domain": "FastAPI Web Architecture",
        "prompt": "Write a minimal FastAPI app with an async lifespan handler, dependency injection, and Pydantic v2 schemas.",
    },
]


def test_http_endpoint(model_id: str, prompt: str, max_tokens: int = 48) -> Dict[str, Any]:
    """Tests real HTTP completion via the local runtime server."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    req = urllib.request.Request(
        "http://127.0.0.1:8000/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    t_start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = time.perf_counter() - t_start
            usage = data.get("usage", {})
            choices = data.get("choices") or [{}]
            c0 = choices[0] if len(choices) > 0 and choices[0] is not None else {}
            msg = c0.get("message") or {} if isinstance(c0, dict) else {}
            content = msg.get("content", "") or msg.get("reasoning_content", "") or ""

            return {
                "success": True,
                "elapsed_s": round(elapsed, 2),
                "completion_tokens": usage.get("completion_tokens", max_tokens),
                "tokens_per_second": usage.get("tokens_per_second", 0.0),
                "generation_time_ms": usage.get("generation_time_ms", round(elapsed * 1000.0, 1)),
                "sample_output": content[:120].replace("\n", " "),
            }
    except Exception as e:
        return {"success": False, "error": str(e)}


def benchmark_runtime_kv_evolution(
    tokens_to_decode: int = 128,
) -> Dict[str, Any]:
    """Direct head-to-head empirical test of our custom runtime attention layers:
    Compares NaiveCatBF16 (Before) vs PreallocatedBF16 (Now) vs PreallocatedQ8 (Now).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_path = Path("models/qwen3.8-27b-triton/layer_3.pt")

    if not weights_path.exists():
        return {"error": "layer_3.pt not found"}

    layer = Qwen35FullAttentionBlock(layer_idx=3, device=device)
    layer_dict = torch.load(weights_path, map_location=device, weights_only=False)
    layer.load_weights(layer_dict)

    # Precompute RoPE
    inv_freq = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=device) / 64))
    t = torch.arange(0, tokens_to_decode + 64, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos_table = torch.cos(freqs).to(torch.bfloat16).unsqueeze(0).unsqueeze(0)
    sin_table = torch.sin(freqs).to(torch.bfloat16).unsqueeze(0).unsqueeze(0)

    torch.manual_seed(42)
    token_stream = torch.randn(tokens_to_decode, 1, 5120, dtype=torch.bfloat16, device=device)

    # 1. Baseline: Naive Cat
    naive_k = None
    naive_v = None
    t0 = time.perf_counter()
    for step in range(tokens_to_decode):
        x = token_stream[step : step + 1]
        cos_sin = (cos_table[:, :, step : step + 1, :], sin_table[:, :, step : step + 1, :])
        out, (naive_k, naive_v) = layer(x, kv_cache=(naive_k, naive_v) if naive_k is not None else None, cos_sin=cos_sin)
    torch.cuda.synchronize(device)
    naive_time_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Optimized: Preallocated BF16
    prealloc_bf16 = PreallocatedKVCache(
        batch_size=1, num_heads=4, head_dim=256, max_seq_len=tokens_to_decode + 16, mode="bf16", device=device
    )
    t0 = time.perf_counter()
    for step in range(tokens_to_decode):
        x = token_stream[step : step + 1]
        cos_sin = (cos_table[:, :, step : step + 1, :], sin_table[:, :, step : step + 1, :])
        out, _ = layer(x, kv_cache=prealloc_bf16, cos_sin=cos_sin)
    torch.cuda.synchronize(device)
    bf16_time_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Optimized: Preallocated Q8_0
    prealloc_q8 = PreallocatedKVCache(
        batch_size=1, num_heads=4, head_dim=256, max_seq_len=tokens_to_decode + 16, mode="q8_0", device=device
    )
    t0 = time.perf_counter()
    for step in range(tokens_to_decode):
        x = token_stream[step : step + 1]
        cos_sin = (cos_table[:, :, step : step + 1, :], sin_table[:, :, step : step + 1, :])
        out, _ = layer(x, kv_cache=prealloc_q8, cos_sin=cos_sin)
    torch.cuda.synchronize(device)
    q8_time_ms = (time.perf_counter() - t0) * 1000.0

    mem_bf16_kb = prealloc_bf16.get_memory_bytes() / 1024
    mem_q8_kb = prealloc_q8.get_memory_bytes() / 1024

    return {
        "tokens_evaluated": tokens_to_decode,
        "naive_cat_time_ms": round(naive_time_ms, 2),
        "prealloc_bf16_time_ms": round(bf16_time_ms, 2),
        "prealloc_q8_time_ms": round(q8_time_ms, 2),
        "speedup_bf16_vs_naive": round(naive_time_ms / max(1e-5, bf16_time_ms), 2),
        "vram_bf16_kb": round(mem_bf16_kb, 1),
        "vram_q8_kb": round(mem_q8_kb, 1),
        "vram_saved_percent": round((1 - mem_q8_kb / mem_bf16_kb) * 100, 1),
    }


def main():
    print("=" * 80)
    print("      REAL END-TO-END EVALUATION: DUAL-ENGINE 27B & KV CACHE EVOLUTION      ")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Target Server: http://127.0.0.1:8000\n")

    # Part 1: Real Use Case Domain Prompts via HTTP Server
    print("--- [PART 1] Real Use Case End-to-End Query Evaluation ---")
    models_to_test = ["qwen3.8:27b", "qwen3.8:27b-triton"]
    http_results = {}

    for model in models_to_test:
        print(f"\nEvaluating Model: {model}")
        model_results = []
        for item in REAL_USECASE_PROMPTS:
            pid = item["id"]
            domain = item["domain"]
            prompt = item["prompt"]
            print(f"  -> Domain: [{domain}] ... ", end="", flush=True)

            res = test_http_endpoint(model, prompt, max_tokens=32)
            if res.get("success"):
                print(f"DONE ({res['elapsed_s']}s | {res['tokens_per_second']:.2f} tok/s)")
            else:
                print(f"FAILED ({res.get('error')})")
            res["prompt_id"] = pid
            res["domain"] = domain
            model_results.append(res)
        http_results[model] = model_results

    # Part 2: Direct Runtime Attention Layer Evolution Benchmark
    print("\n--- [PART 2] Custom Runtime Layer Evolution (Before vs Now) ---")
    print("Running 256 decode steps comparing NaiveCatBF16 vs PreallocatedBF16 vs PreallocatedQ8...")
    runtime_eval = benchmark_runtime_kv_evolution(tokens_to_decode=256)

    print("\n================================================================================")
    print("                           SUMMARY OF EMPIRICAL METRICS                        ")
    print("================================================================================")

    # HTTP Summary Table
    print("\n1. Real-World Use Case Latencies (32-token completions via HTTP API):")
    print(f"{'Domain':<30} | {'C++ HIP (Port 8001)':<20} | {'Native Triton (Pure ROCm)':<25}")
    print("-" * 80)
    for i, item in enumerate(REAL_USECASE_PROMPTS):
        domain = item["domain"]
        r_cpp = http_results["qwen3.8:27b"][i]
        r_tri = http_results["qwen3.8:27b-triton"][i]
        t_cpp = f"{r_cpp.get('elapsed_s', 'N/A')}s ({r_cpp.get('tokens_per_second', 0):.1f} tok/s)" if r_cpp.get("success") else "Error"
        t_tri = f"{r_tri.get('elapsed_s', 'N/A')}s ({r_tri.get('tokens_per_second', 0):.1f} tok/s)" if r_tri.get("success") else "Error"
        print(f"{domain:<30} | {t_cpp:<20} | {t_tri:<25}")

    # Runtime KV Cache Evolution Table
    print("\n2. Custom Runtime Attention Evolution (256 decode steps):")
    print(f"  - Baseline (Before, NaiveCatBF16):  {runtime_eval.get('naive_cat_time_ms')} ms")
    print(f"  - Optimized (Now, PreallocatedBF16): {runtime_eval.get('prealloc_bf16_time_ms')} ms  (Speedup: {runtime_eval.get('speedup_bf16_vs_naive')}x)")
    print(f"  - Optimized (Now, PreallocatedQ8):   {runtime_eval.get('prealloc_q8_time_ms')} ms")
    print(f"  - Memory per layer at 256 tokens:    {runtime_eval.get('vram_bf16_kb')} KB (BF16) -> {runtime_eval.get('vram_q8_kb')} KB (Q8_0)  ({runtime_eval.get('vram_saved_percent')}% saved)")

    # Save to disk
    out_dir = Path("results/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "e2e_real_usecase_eval.json"
    full_data = {
        "http_benchmarks": http_results,
        "runtime_layer_evolution": runtime_eval,
    }
    with open(out_file, "w") as f:
        json.dump(full_data, f, indent=2)
    print(f"\nSaved complete benchmark data to: {out_file}")


if __name__ == "__main__":
    main()
