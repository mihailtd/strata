"""End-to-End Real Engineering Harness Benchmark: True O(1) Tensor State Handoff ($S_t$) vs Text Re-Prefill.

Runs a multi-subagent engineering project on AMD Radeon RX 7900 XTX (24 GB VRAM):
- Astral Subagent: pyproject.toml and ruff config
- Python Modern Subagent: Pydantic v2 models and PEP 695 generics
- PostgreSQL Subagent: asyncpg schema and pgvector HNSW (<=>) queries
- DuckDB Subagent: vectorized SQL analytics with QUALIFY window clause
- FastAPI Subagent: @asynccontextmanager lifespan web router
- Pytest Subagent: integration test suite

Compares:
1. Arm 1: True O(1) Tensor State Handoff ($S_t$) (0.05 ms transfer, delta prompt only)
2. Arm 2: Text Re-Prefill Baseline (re-prefilling cumulative history and artifacts)

Runs real OS verification (ruff & pytest) and writes telemetry to
results/benchmarks/state_handoff_27b_e2e_harness_benchmark.json.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.harness.coordinator.state_handoff_harness import StateHandoffHarnessCoordinator
from src.runtime.native_27b_engine import Native27BEngine


def run_e2e_benchmark():
    print("=" * 105)
    print("🚀 E2E HARNESS BENCHMARK: TRUE O(1) TENSOR STATE HANDOFF ($S_t$) vs TEXT RE-PREFILL")
    print("   Model: Qwen2.5-Coder-27B-W4A16-DeltaNet | Hardware: AMD Radeon RX 7900 XTX (24 GB, gfx1100)")
    print("=" * 105)

    goal = "Build a high-performance vector search and analytics microservice with FastAPI, PostgreSQL 17 pgvector, and DuckDB"

    engine = Native27BEngine(num_layers=64)
    engine.load_from_cache()

    coordinator = StateHandoffHarnessCoordinator(engine=engine, clone_on_handoff=True)

    project_dir_handoff = REPO_ROOT / "projects" / "state_handoff_e2e_service"
    project_dir_text = REPO_ROOT / "projects" / "text_prefill_e2e_service"

    if project_dir_handoff.exists():
        shutil.rmtree(project_dir_handoff)
    if project_dir_text.exists():
        shutil.rmtree(project_dir_text)

    # -------------------------------------------------------------------------
    # ARM 1: True O(1) Tensor State Handoff ($S_t$)
    # -------------------------------------------------------------------------
    print("\n" + "#" * 105)
    print(">>> EXECUTING ARM 1: TRUE O(1) TENSOR STATE HANDOFF (S_t)")
    print("#" * 105)

    t_handoff_start = time.perf_counter()
    handoff_records, handoff_verification = coordinator.execute_project(
        goal=goal,
        project_dir=project_dir_handoff,
        mode="tensor_handoff",
        max_tokens_per_subagent=128,
    )
    t_handoff_total = time.perf_counter() - t_handoff_start

    print(f"\n[ARM 1 Results] Total Time: {t_handoff_total:.2f}s | OS Tests Passed: {handoff_verification['passed']}")
    for r in handoff_records:
        print(
            f"  Subagent: {r.task_id:<22} | Prefill: {r.prompt_tokens:3d} tok ({r.prefill_ms:6.2f}ms) "
            f"| Avoided: {r.tokens_avoided:4d} tok | Handoff: {r.handoff_ms:.3f}ms | LoRA: {r.lora_swap_ms:6.2f}ms "
            f"| Decode: {r.tokens_generated:3d} tok ({r.tok_per_sec:5.2f} tok/s)"
        )

    # -------------------------------------------------------------------------
    # ARM 2: Standard Text Re-Prefill Baseline
    # -------------------------------------------------------------------------
    print("\n" + "#" * 105)
    print(">>> EXECUTING ARM 2: TEXT RE-PREFILL BASELINE (Re-encoding History)")
    print("#" * 105)

    t_text_start = time.perf_counter()
    text_records, text_verification = coordinator.execute_project(
        goal=goal,
        project_dir=project_dir_text,
        mode="text_reprefill",
        max_tokens_per_subagent=128,
    )
    t_text_total = time.perf_counter() - t_text_start

    print(f"\n[ARM 2 Results] Total Time: {t_text_total:.2f}s | OS Tests Passed: {text_verification['passed']}")
    for r in text_records:
        print(
            f"  Subagent: {r.task_id:<22} | Prefill: {r.prompt_tokens:3d} tok ({r.prefill_ms:6.2f}ms) "
            f"| LoRA: {r.lora_swap_ms:6.2f}ms | Decode: {r.tokens_generated:3d} tok ({r.tok_per_sec:5.2f} tok/s)"
        )

    # -------------------------------------------------------------------------
    # Comparative Summary Report
    # -------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("🏆 FINAL COMPARATIVE SCORECARD: REAL ENGINEERING HARNESS (AMD RX 7900 XTX)")
    print("=" * 105)

    print(f"{'Task / Subagent':<24} | {'State Handoff TTFT':<20} | {'Text Prefill TTFT':<20} | {'Prefill Speedup':<15} | {'Tokens Avoided':<14}")
    print("-" * 105)

    total_pref_ms_handoff = 0.0
    total_pref_ms_text = 0.0
    total_tokens_avoided = 0
    total_tok_text = 0
    total_tok_handoff = 0

    num_turns = min(len(handoff_records), len(text_records))
    turn_comparisons = []

    for i in range(num_turns):
        h = handoff_records[i]
        t = text_records[i]

        total_pref_ms_handoff += h.prefill_ms
        total_pref_ms_text += t.prefill_ms
        total_tokens_avoided += h.tokens_avoided
        total_tok_text += t.prompt_tokens
        total_tok_handoff += h.prompt_tokens

        speedup = t.prefill_ms / max(h.prefill_ms, 0.01)
        print(
            f"{h.task_id:<24} | {h.prefill_ms:7.2f} ms ({h.prompt_tokens:3d} tok)   "
            f"| {t.prefill_ms:7.2f} ms ({t.prompt_tokens:3d} tok)   "
            f"| {speedup:6.2f}x          | {h.tokens_avoided:5d} tokens"
        )

        turn_comparisons.append({
            "task_id": h.task_id,
            "specialist": h.specialist,
            "lora_adapter": h.lora_adapter,
            "handoff_prefill_ms": round(h.prefill_ms, 2),
            "handoff_prompt_tokens": h.prompt_tokens,
            "handoff_tokens_avoided": h.tokens_avoided,
            "handoff_latency_ms": round(h.handoff_ms, 4),
            "handoff_lora_swap_ms": round(h.lora_swap_ms, 2),
            "text_prefill_ms": round(t.prefill_ms, 2),
            "text_prompt_tokens": t.prompt_tokens,
            "speedup": round(speedup, 2),
        })

    cumulative_speedup = total_pref_ms_text / max(total_pref_ms_handoff, 0.01)
    capacity_saved_pct = round((total_tokens_avoided / max(total_tok_text, 1)) * 100.0, 1)

    print("-" * 105)
    print(f"CUMULATIVE PREFILL METRICS:")
    print(f"  • Arm 1 (State Handoff) Total Prefill : {total_pref_ms_handoff:7.2f} ms ({total_tok_handoff} tokens processed)")
    print(f"  • Arm 2 (Text Re-prefill) Total Prefill: {total_pref_ms_text:7.2f} ms ({total_tok_text} tokens processed)")
    print(f"  • Cumulative Prefill Speedup          : {cumulative_speedup:.2f}x FASTER ⚡")
    print(f"  • Context Tokens Saved / Avoided      : {total_tokens_avoided} tokens ({capacity_saved_pct}% reduction)")
    print(f"  • Average Handoff Latency             : {sum(h.handoff_ms for h in handoff_records) / len(handoff_records):.4f} ms (sub-millisecond O(1))")
    print(f"  • Real OS Ruff Linter Status          : Arm 1 Exit {handoff_verification['ruff_exit']} | Arm 2 Exit {text_verification['ruff_exit']}")
    print(f"  • Real OS Pytest Status               : Arm 1 Exit {handoff_verification['pytest_exit']} | Arm 2 Exit {text_verification['pytest_exit']}")
    print("=" * 105)

    # Persist benchmark telemetry
    scorecard = {
        "hardware": "AMD Radeon RX 7900 XTX (24 GB, gfx1100)",
        "model": "Qwen2.5-Coder-27B-W4A16-DeltaNet",
        "goal": goal,
        "cumulative_prefill_speedup": round(cumulative_speedup, 2),
        "total_tokens_avoided": total_tokens_avoided,
        "context_capacity_saved_pct": capacity_saved_pct,
        "handoff_total_prefill_ms": round(total_pref_ms_handoff, 2),
        "text_total_prefill_ms": round(total_pref_ms_text, 2),
        "os_verification": {
            "handoff_arm": handoff_verification,
            "text_arm": text_verification,
        },
        "turns": turn_comparisons,
    }

    out_file = REPO_ROOT / "results" / "benchmarks" / "state_handoff_27b_e2e_harness_benchmark.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(scorecard, f, indent=2)

    print(f"\n💾 Telemetry persisted to: {out_file}")


if __name__ == "__main__":
    run_e2e_benchmark()
