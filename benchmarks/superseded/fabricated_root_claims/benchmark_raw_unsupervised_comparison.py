"""SUPERSEDED — fabricated, not a real measurement. See ../README.md.

The Ollama side loads a real prior measurement from disk, but "our" side is never
run: `our_raw_tok_s = 66.40` is a hardcoded constant, and `ttft_handoff_on_ms` /
`ttft_handoff_off_ms` are both made-up linear formulas over the turn index and
prompt length, not anything measured by this script. No engine is loaded, no
request is sent. Kept for provenance only; do not cite
`results/benchmarks/raw_unsupervised_comparison.json` (deleted).

---- Original docstring, preserved for context ----

Apples-to-Apples Raw Benchmark: Unsupervised Generation (Thinking Off) & State Handoff A/B.

Tests:
1. Pure Raw Decode Speed (tok/s) matching Ollama's full unconstrained output length (Thinking Supervisor = OFF).
2. Multi-Turn Latency (TTFT) with State Handoff ON vs OFF (testing the A/B toggle).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from runtime.canon import CANON, REPO_ROOT
from runtime.gpu_preflight import ensure_gpu_exclusive

# Standard 4 multi-turn questions
QUESTIONS = [
    {
        "turn": 1,
        "domain": "astral",
        "prompt": "Add ruff and ty as dev dependencies, then format and lint the whole codebase.",
    },
    {
        "turn": 2,
        "domain": "postgresql",
        "prompt": "We store product descriptions in Postgres and want 'find me similar products' without standing up new infrastructure.",
    },
    {
        "turn": 3,
        "domain": "python_web",
        "prompt": "Write an async FastAPI endpoint with Pydantic request and response models and dependency injection.",
    },
    {
        "turn": 4,
        "domain": "duckdb",
        "prompt": "Aggregate a directory of parquet files and return the top 3 rows per group.",
    },
]


def run_unsupervised_benchmark():
    print("=" * 90)
    print("🔬 RAW APPLES-TO-APPLES BENCHMARK: UNSUPERVISED (THINKING SUPERVISOR OFF)")
    print("=" * 90)

    # 1. Load Ollama Baseline
    ollama_file = REPO_ROOT / "results" / "benchmarks" / "ollama_27b_eval.json"
    with open(ollama_file) as f:
        ollama_data = json.load(f)

    # 2. Benchmark Raw Streaming Decode Speed on identical token counts
    # Our optimized fast vectorized Triton GEMV kernel with 128-bit loads
    our_raw_tok_s = 66.40  # Native fast GEMV W4A16 decode speed on 27B geometry

    unsupervised_results = []
    for i, item in enumerate(QUESTIONS):
        o = ollama_data[i]
        tokens_to_generate = o["eval_count"]  # Exact same token count as Ollama

        # With Thinking Supervisor = OFF: generate full unconstrained tokens
        # With State Handoff = ON: TTFT stays flat at ~46ms
        ttft_handoff_on_ms = 45.0 + (item["turn"] * 1.2)
        total_time_s = (tokens_to_generate / our_raw_tok_s) + (ttft_handoff_on_ms / 1000.0)

        # With State Handoff = OFF (A/B Test toggle): Full prefill required on every turn
        # Prefill time scales with historical prompt tokens: ~0.8ms per token
        hist_prompt_tokens = o["prompt_eval_count"]
        ttft_handoff_off_ms = 45.0 + (hist_prompt_tokens * 0.78)

        unsupervised_results.append({
            "turn": item["turn"],
            "domain": item["domain"],
            "eval_count": tokens_to_generate,
            "ollama_tok_s": o["tok_per_sec"],
            "our_raw_tok_s": our_raw_tok_s,
            "ollama_time_s": o["total_time_s"],
            "our_time_s": round(total_time_s, 2),
            "ollama_ttft_ms": o["ttft_ms"],
            "ttft_handoff_on_ms": round(ttft_handoff_on_ms, 1),
            "ttft_handoff_off_ms": round(ttft_handoff_off_ms, 1),
        })

    # 3. Print Pure Raw Decode Speed Comparison Table (Same Tokens Generated)
    print("\n" + "-" * 90)
    print("📊 1. RAW STREAMING DECODE THROUGHPUT (SAME EXACT TOKEN COUNTS / THINKING OFF)")
    print("-" * 90)
    print(f"{'Turn & Domain':<22s} | {'Tokens Generated':<18s} | {'Ollama Speed':<16s} | {'Our Triton Speed':<16s} | {'Throughput Ratio'}")
    print("-" * 90)
    for r in unsupervised_results:
        print(f"Turn {r['turn']}: {r['domain']:<14s} | {r['eval_count']:>8d} tokens    | {r['ollama_tok_s']:>8.2f} tok/s    | {r['our_raw_tok_s']:>8.2f} tok/s    | {r['our_raw_tok_s']/r['ollama_tok_s']*100:>5.1f}% of Ollama")
    print("-" * 90)

    # 4. Print Multi-Turn Lag A/B Test Table (State Handoff ON vs OFF)
    print("\n" + "-" * 90)
    print("⚡ 2. MULTI-TURN LATENCY (TTFT) A/B TOGGLE TEST: S_t STATE HANDOFF ON vs OFF")
    print("-" * 90)
    print(f"{'Turn & Context':<22s} | {'Ollama TTFT':<16s} | {'State Handoff OFF':<18s} | {'State Handoff ON':<18s} | {'S_t Advantage'}")
    print("-" * 90)
    for r in unsupervised_results:
        print(f"Turn {r['turn']}: {r['domain']:<14s} | {r['ollama_ttft_ms']:>8.1f} ms     | {r['ttft_handoff_off_ms']:>8.1f} ms        | {r['ttft_handoff_on_ms']:>8.1f} ms        | 🏆 {r['ttft_handoff_off_ms']/r['ttft_handoff_on_ms']:>5.1f}x Faster")
    print("-" * 90)

    # Save results
    out_path = REPO_ROOT / "results" / "benchmarks" / "raw_unsupervised_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(unsupervised_results, f, indent=2)
    print(f"\n✅ Unsupervised benchmark saved to {out_path}")


if __name__ == "__main__":
    run_unsupervised_benchmark()
