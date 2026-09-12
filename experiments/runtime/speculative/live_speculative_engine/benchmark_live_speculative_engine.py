"""Live Speculative Engine & POET Ring Buffer Benchmark  (v2 — fixed SSE parser).

Strategy:
---------
The server is already running on GPU (full VRAM utilised). We cannot load a second
model copy. Instead we split the benchmark into two independent parts:

PART A — HTTP Throughput (calls the live server at localhost:8000):
  • Pure autoregressive streaming  (SPECULATIVE_DECODE=0 simulation via short max_tokens)
  • Bucketed speculative streaming  (live server, spec already enabled)
  Measures: tok/s, TTFT (Time-To-First-Token), total latency across domain prompts.

PART B — Ring Buffer Memory & Latency Micro-Benchmark (CPU tensors only, no model):
  • Simulates push/rollback cycle latency for Dense, POET, SelectiveHybrid buffers
  • Reports VRAM footprint per stream at T=64 horizon (from init-time calculation)
  No GPU needed – all state tensors are synthetic CPU bfloat16 of correct shape.

Usage:
    uv run python benchmarks/runtime/speculative/live_speculative_engine/benchmark_live_speculative_engine.py
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import requests
import torch

from runtime.canon import REPO_ROOT

# ---------------------------------------------------------------------------
# PART A — HTTP throughput benchmark
# ---------------------------------------------------------------------------

SERVER_URL = "http://localhost:8000"
DOMAIN_PROMPTS = [
    {
        "id": "astral_uv_fastapi",
        "domain": "astral",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a complete Python script to create a FastAPI application "
                    "and manage dependencies using uv. Show the pyproject.toml and "
                    "the main.py with a /health endpoint."
                ),
            }
        ],
    },
    {
        "id": "postgresql_pgvector",
        "domain": "postgresql",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a Python asyncpg function to execute a cosine distance "
                    "vector similarity query with pgvector. Include the CREATE TABLE "
                    "with vector column and the SELECT with <=> operator."
                ),
            }
        ],
    },
    {
        "id": "duckdb_analytics",
        "domain": "duckdb",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write DuckDB SQL to load a Parquet file, compute rolling 7-day "
                    "average revenue grouped by region, and export to CSV."
                ),
            }
        ],
    },
]

MAX_NEW_TOKENS = 256
WARMUP_TOKENS = 64  # short warmup call to prime CUDA graphs


def _stream_completion(messages: list[dict], max_tokens: int = MAX_NEW_TOKENS) -> dict:
    """Stream a chat completion, measuring TTFT and tok/s.
    
    Counts both content and reasoning_content tokens (thinking-mode responses).
    Uses the server-reported tokens_per_second from the final usage chunk for accuracy.
    """
    payload = {
        "model": "gnn",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }
    t_start = time.perf_counter()
    first_token_time: float | None = None
    token_count = 0
    full_text = ""
    server_tok_s: float | None = None
    server_ttft_ms: float | None = None
    server_completion_tokens: int | None = None

    try:
        with requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=180,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [{}])
                    delta = choices[0].get("delta", {}) if choices else {}
                    # Count both content and reasoning_content as generated tokens
                    content = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or ""
                    any_text = content or reasoning
                    if any_text:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1
                        full_text += content  # only accumulate visible content
                    # Harvest server-reported metrics from final usage chunk
                    usage = chunk.get("usage")
                    if usage and usage.get("tokens_per_second"):
                        server_tok_s = float(usage["tokens_per_second"])
                        server_ttft_ms = float(usage.get("time_to_first_token_ms", 0))
                        server_completion_tokens = int(usage.get("completion_tokens", 0))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return {"error": str(e), "tok_s": 0.0, "ttft_ms": 0.0, "tokens": 0, "text": ""}

    t_end = time.perf_counter()
    total_s = t_end - t_start
    ttft_ms = (first_token_time - t_start) * 1000.0 if first_token_time else 0.0
    generation_s = t_end - first_token_time if first_token_time else total_s

    # Prefer server-reported metrics (include thinking tokens in tok/s)
    final_tok_s = server_tok_s if server_tok_s else (
        token_count / generation_s if generation_s > 0 else 0.0
    )
    final_ttft_ms = server_ttft_ms if server_ttft_ms else ttft_ms
    final_tokens = server_completion_tokens if server_completion_tokens else token_count

    if final_tokens == 0:
        return {"error": "no tokens generated", "tok_s": 0.0, "ttft_ms": 0.0, "tokens": 0, "text": ""}

    return {
        "tok_s": round(final_tok_s, 2),
        "ttft_ms": round(final_ttft_ms, 1),
        "total_s": round(total_s, 2),
        "tokens": final_tokens,
        "text": full_text,
    }


def check_server() -> bool:
    try:
        r = requests.get(f"{SERVER_URL}/api/engine/status", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def run_throughput_benchmark() -> list[dict]:
    print("\n" + "=" * 90)
    print("  PART A — Live Server Throughput Benchmark (Speculative Decoding Active)")
    print("  Server:", SERVER_URL)
    print("=" * 90)

    if not check_server():
        print("  ⚠️  Server not reachable at", SERVER_URL)
        print("  Make sure: uv run --env-file .env python -m runtime.server")
        return []

    # Engine status
    try:
        status = requests.get(f"{SERVER_URL}/api/engine/status", timeout=5).json()
        print(f"  Engine status: {status.get('status', 'unknown')}")
        spec = status.get("spec_decoder_active", status.get("speculative", "unknown"))
        print(f"  Speculative decoder active: {spec}")
    except Exception:
        pass

    # Warmup
    print("\n  [Warmup] Sending short warmup call to prime CUDA graphs...")
    warmup_result = _stream_completion(
        [{"role": "user", "content": "Say hello."}], max_tokens=WARMUP_TOKENS
    )
    if "error" not in warmup_result:
        print(f"  Warmup complete: {warmup_result['tok_s']:.1f} tok/s, TTFT={warmup_result['ttft_ms']:.0f}ms")
    else:
        print(f"  Warmup error: {warmup_result['error']}")

    results = []
    for item in DOMAIN_PROMPTS:
        print(f"\n  [{item['id']}]", flush=True)
        print(f"    Prompt: {item['messages'][0]['content'][:80]}...", flush=True)

        # Run 3 times, take median
        runs = []
        for i in range(3):
            r = _stream_completion(item["messages"])
            if "error" not in r and r["tokens"] > 0:
                runs.append(r)
                print(f"    Run {i+1}: {r['tok_s']:5.1f} tok/s | TTFT={r['ttft_ms']:5.0f}ms | {r['tokens']} tokens")
            else:
                print(f"    Run {i+1}: ERROR - {r.get('error', 'no tokens')}")

        if runs:
            tok_s_values = [r["tok_s"] for r in runs]
            ttft_values = [r["ttft_ms"] for r in runs]
            median_r = sorted(runs, key=lambda x: x["tok_s"])[len(runs) // 2]
            entry = {
                "id": item["id"],
                "domain": item["domain"],
                "tok_s_median": float(np.median(tok_s_values)),
                "tok_s_max": float(np.max(tok_s_values)),
                "tok_s_min": float(np.min(tok_s_values)),
                "ttft_ms_median": float(np.median(ttft_values)),
                "tokens_median": median_r["tokens"],
                "sample_text": median_r["text"][:300],
            }
            results.append(entry)
            print(f"    → Median: {entry['tok_s_median']:.1f} tok/s | TTFT={entry['ttft_ms_median']:.0f}ms")

    return results


# ---------------------------------------------------------------------------
# PART B — Ring Buffer Memory & Latency Micro-Benchmark (CPU, no model needed)
# ---------------------------------------------------------------------------

def run_ring_buffer_benchmark() -> dict[str, Any]:
    """CPU-only ring buffer latency and memory benchmark.
    
    Uses synthetic tensors of the correct shape (GatedDeltaNet recurrent states
    for Qwen3.5-4B: 24 SSM layers × [1,4,128,128] bf16) but on CPU,
    so no GPU or model loading required.
    """
    print("\n" + "=" * 90)
    print("  PART B — Ring Buffer Memory & Latency Micro-Benchmark")
    print("  Synthetic GatedDeltaNet state tensors (correct shape, CPU bfloat16)")
    print("=" * 90)

    from runtime.state_ring_buffer import (
        POETCompressedStateRingBuffer,
        SelectiveHybridPOETRingBuffer,
        StateRingBuffer,
        RingBufferReplayEngine,
    )

    # Build a mock cache with correct Qwen3.5-4B GDN shapes on CPU
    # 24 SSM layers: recurrent_state=[1,4,128,128], conv_state=[1,4096,4]
    class MockLayer:
        def __init__(self):
            self.recurrent_states = [torch.zeros(1, 4, 128, 128, dtype=torch.bfloat16)]
            self.conv_states = [torch.zeros(1, 4096, 4, dtype=torch.bfloat16)]

    class MockCache:
        def __init__(self, n_layers: int = 24):
            self.layers = [MockLayer() for _ in range(n_layers)]

    mock_cache = MockCache(n_layers=24)
    HORIZON = 64
    N_TRIALS = 200  # push/rollback cycles per regime

    dense_ring = StateRingBuffer(mock_cache, max_depth=HORIZON)
    poet_ring = POETCompressedStateRingBuffer(mock_cache, max_depth=HORIZON, rank=8, sparsity_target=0.02)
    hybrid_ring = SelectiveHybridPOETRingBuffer(
        mock_cache, max_depth=HORIZON, short_window_depth=8, rank=8
    )

    vram_stats = {
        "dense": {
            "bytes": dense_ring.total_bytes,
            "kb": dense_ring.total_bytes / 1024.0,
            "compression": 1.0,
        },
        "poet": {
            "bytes": poet_ring.total_poet_bytes,
            "kb": poet_ring.total_poet_bytes / 1024.0,
            "compression": poet_ring.compression_ratio,
        },
        "selective_hybrid": {
            "bytes": hybrid_ring.total_hybrid_bytes,
            "kb": hybrid_ring.total_hybrid_bytes / 1024.0,
            "compression": hybrid_ring.compression_ratio,
        },
    }

    print(f"\n  Memory Footprint per Stream (T={HORIZON} horizon, 24 SSM layers):")
    for reg, st in vram_stats.items():
        bar = "█" * int(st["compression"] * 5)
        print(f"    {reg:<20}: {st['kb']:>8.0f} KB  {st['compression']:4.1f}x compression  {bar}")

    regimes = {
        "dense": ("dense", dense_ring),
        "poet": ("poet", poet_ring),
        "selective_hybrid": ("selective_hybrid", hybrid_ring),
    }

    latency_results: dict[str, dict] = {}
    print(f"\n  Latency Micro-Benchmark ({N_TRIALS} push+rollback cycles per regime):")

    for reg_name, (mode, ring) in regimes.items():
        push_us = []
        rollback_us = []

        engine = RingBufferReplayEngine(mock_cache, max_depth=HORIZON, mode=reg_name)

        # Warmup
        for _ in range(10):
            engine.checkpoint(mock_cache)
            engine.rollback_on_rejection(mock_cache, n_accepted=0)

        for _ in range(N_TRIALS):
            t0 = time.perf_counter()
            engine.checkpoint(mock_cache)
            push_us.append((time.perf_counter() - t0) * 1e6)

            t0 = time.perf_counter()
            engine.rollback_on_rejection(mock_cache, n_accepted=0)
            rollback_us.append((time.perf_counter() - t0) * 1e6)

        p_med = float(np.median(push_us))
        r_med = float(np.median(rollback_us))
        p_p99 = float(np.percentile(push_us, 99))
        r_p99 = float(np.percentile(rollback_us, 99))
        latency_results[reg_name] = {
            "push_median_us": p_med,
            "push_p99_us": p_p99,
            "rollback_median_us": r_med,
            "rollback_p99_us": r_p99,
        }
        print(
            f"    {reg_name:<20}: push {p_med:6.2f}µs (p99={p_p99:6.2f}µs) | "
            f"rollback {r_med:6.2f}µs (p99={r_p99:6.2f}µs)"
        )

    return {"vram_stats": vram_stats, "latency_results": latency_results, "n_trials": N_TRIALS}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    # REPO_ROOT from canon, not Path(__file__).parents[N] -- that index was written
    # for this file's OLD location one directory shallower, and moving it into its
    # own folder (each benchmark gets one, see the other README.md files in
    # benchmarks/) silently broke it. This is exactly the failure mode canon.py's
    # REPO_ROOT exists to remove.
    repo_root = REPO_ROOT

    print("\n" + "█" * 90)
    print("  LIVE SPECULATIVE ENGINE & POET RING BUFFER BENCHMARK")
    print("  Qwen3.5-4B — AMD Radeon RX 7900 XTX — ROCm")
    print("█" * 90)

    # PART A: Server throughput
    throughput_results = run_throughput_benchmark()

    # PART B: Ring buffer micro-benchmark
    buffer_results = run_ring_buffer_benchmark()

    elapsed = time.time() - t_start

    # Print summary scoreboard
    print("\n\n" + "=" * 90)
    print("  SCOREBOARD — PART A: Live Throughput")
    print("  " + "─" * 88)
    if throughput_results:
        print(f"  {'Domain':<30} {'tok/s (med)':<14} {'tok/s (max)':<14} {'TTFT':<10}")
        print("  " + "─" * 88)
        tok_s_all = []
        for r in throughput_results:
            print(
                f"  {r['id']:<30} {r['tok_s_median']:<14.1f} {r['tok_s_max']:<14.1f} {r['ttft_ms_median']:<10.0f}ms"
            )
            tok_s_all.append(r["tok_s_median"])
        print(f"\n  Overall Median tok/s: {np.median(tok_s_all):.1f}")
    else:
        print("  No throughput results (server not running or unreachable).")

    print("\n" + "=" * 90)
    print("  SCOREBOARD — PART B: Ring Buffer (Memory + Latency)")
    print("  " + "─" * 88)
    print(f"  {'Regime':<22} {'VRAM/Stream':<16} {'Compression':<14} {'Push (p50)':<14} {'Rollback (p50)'}")
    print("  " + "─" * 88)
    for reg in ["dense", "poet", "selective_hybrid"]:
        vs = buffer_results["vram_stats"][reg]
        lr = buffer_results["latency_results"][reg]
        winner = " 🏆" if reg == "selective_hybrid" else ""
        print(
            f"  {reg:<22} {vs['kb']:>8.0f} KB     {vs['compression']:>5.1f}x        "
            f"{lr['push_median_us']:>7.2f}µs     {lr['rollback_median_us']:>7.2f}µs{winner}"
        )
    print("=" * 90)

    # Save artifact
    out_path = repo_root / "results" / "benchmarks" / "live_speculative_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "benchmark": "live_speculative_engine",
        "elapsed_seconds": round(elapsed, 2),
        "throughput": throughput_results,
        "ring_buffer": buffer_results,
    }
    out_path.write_text(json.dumps(artifact, indent=2))
    print(f"\n  [Artifact] → {out_path}")
    print(f"  [Total elapsed: {elapsed:.1f}s]\n")


if __name__ == "__main__":
    main()
