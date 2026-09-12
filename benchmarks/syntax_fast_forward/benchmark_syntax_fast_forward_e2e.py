"""Stage 4: Real End-to-End Evaluation & Dual A/B Comparison for Syntax Fast-Forwarding.

Conducts a 3-arm head-to-head benchmark on AMD Radeon RX 7900 XTX (24 GB VRAM):
- Arm 1: Pure Greedy Baseline (K=1)
- Arm 2: Previous Version: Linear Speculation (N-Gram + Neural MTP, no syntax trie)
- Arm 3: Our Engine: Syntax Fast-Forward Speculation (Syntax Trie + N-Gram + Neural MTP)

Evaluated on 4 authentic software engineering code generation tasks (FastAPI, PostgreSQL, DuckDB, Python).
Saves results to results/benchmarks/syntax_fast_forward_e2e_scorecard.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime import server
from runtime.native_27b_engine import Native27BEngine
from runtime.syntax_drafter import SyntaxTrieDrafter

TASKS = [
    {
        "id": "e2e_01_fastapi",
        "domain": "FastAPI & Pydantic",
        "prompt": "# FastAPI vector search microservice\nfrom fastapi import ",
        "budget": 64,
    },
    {
        "id": "e2e_02_postgresql",
        "domain": "PostgreSQL 17 Asyncpg",
        "prompt": "import asyncpg\n\nasync def execute_query(pool, query):\n    async with pool.acquire() as ",
        "budget": 64,
    },
    {
        "id": "e2e_03_duckdb",
        "domain": "DuckDB Parquet OLAP",
        "prompt": (
            "import duckdb\n\ndef run_analytics(con):\n"
            "    query = \"\"\"\n        SELECT * FROM read_parquet('orders.parquet')\n"
            "        QUALIFY ROW_NUMBER() OVER ("
        ),
        "budget": 64,
    },
    {
        "id": "e2e_04_python_boilerplate",
        "domain": "Python Architecture",
        "prompt": "#!/usr/bin/env python3\nimport sys\nfrom typing import ",
        "budget": 64,
    },
]


def run_e2e_benchmark():
    print("=" * 80)
    print("🏆 Stage 4: Real End-to-End Dual A/B Comparison (AMD Radeon RX 7900 XTX)")
    print("=" * 80)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Target Device: {device}")

    # Initialize engine via server singleton
    t0 = time.perf_counter()
    engine: Native27BEngine = server.get_native_triton_27b_engine(num_layers=64)
    tok = server.get_27b_tokenizer()
    print(f"[*] Engine ready in {(time.perf_counter() - t0):.2f}s")

    # Cache original syntax drafter
    active_syntax_drafter = engine.syntax_drafter
    if active_syntax_drafter is None:
        active_syntax_drafter = SyntaxTrieDrafter(tok)
        engine.syntax_drafter = active_syntax_drafter

    arms = [
        {"id": "arm_1_greedy", "name": "Pure Greedy Baseline (K=1)"},
        {"id": "arm_2_previous_linear", "name": "Previous Version: Linear MTP + N-Gram"},
        {"id": "arm_3_syntax_fast_forward", "name": "Our Engine: Syntax Fast-Forward + N-Gram + MTP"},
    ]

    results_by_arm: dict[str, list[dict[str, Any]]] = {arm["id"]: [] for arm in arms}

    for task in TASKS:
        prompt_ids = tok.encode(task["prompt"])
        budget = task["budget"]
        print("\n" + "-" * 70)
        print(f"📋 Task: [{task['domain']}] {task['id']} (Prompt: {len(prompt_ids)} tok, Budget: {budget} tok)")
        print("-" * 70)

        for arm in arms:
            arm_id = arm["id"]
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated()

            t_start = time.perf_counter()

            if arm_id == "arm_1_greedy":
                gen_tokens = engine.generate(
                    prompt_ids, max_new_tokens=budget, temperature=0.0, use_hip_graph=False
                )
            elif arm_id == "arm_2_previous_linear":
                # Disable syntax drafter for Arm 2 to measure baseline linear speculation
                engine.syntax_drafter = None
                gen_tokens = engine.generate_speculative(
                    prompt_ids, max_new_tokens=budget, draft_k=3, use_mtp=True, use_hip_graph=False
                )
                engine.syntax_drafter = active_syntax_drafter  # Restore
            elif arm_id == "arm_3_syntax_fast_forward":
                # Active syntax drafter
                engine.syntax_drafter = active_syntax_drafter
                gen_tokens = engine.generate_speculative(
                    prompt_ids, max_new_tokens=budget, draft_k=3, use_mtp=True, use_hip_graph=False
                )

            torch.cuda.synchronize()
            elapsed_s = time.perf_counter() - t_start
            mem_after = torch.cuda.memory_allocated()
            mem_delta_mb = round((mem_after - mem_before) / (1024 * 1024), 2)

            num_tokens = len(gen_tokens)
            tok_s = round(num_tokens / max(1e-4, elapsed_s), 1)
            decoded_snippet = tok.decode(gen_tokens[:16]).replace("\n", "\\n")

            entry = {
                "task_id": task["id"],
                "domain": task["domain"],
                "tokens_generated": num_tokens,
                "wall_clock_s": round(elapsed_s, 2),
                "throughput_tok_s": tok_s,
                "mem_delta_mb": mem_delta_mb,
                "snippet": decoded_snippet,
            }
            results_by_arm[arm_id].append(entry)

            print(
                f"  [{arm['name'][:30]:<30}] {num_tokens} tokens in {elapsed_s:.2f}s "
                f"| {tok_s} tok/s | Churn: {mem_delta_mb} MB | Output: {decoded_snippet!r}..."
            )

    # Compute summary statistics
    summary: dict[str, Any] = {}
    for arm in arms:
        arm_id = arm["id"]
        runs = results_by_arm[arm_id]
        total_tokens = sum(r["tokens_generated"] for r in runs)
        total_time = sum(r["wall_clock_s"] for r in runs)
        avg_tok_s = round(total_tokens / max(1e-4, total_time), 1)
        summary[arm_id] = {
            "name": arm["name"],
            "total_tokens": total_tokens,
            "total_wall_clock_s": round(total_time, 2),
            "avg_throughput_tok_s": avg_tok_s,
        }

    greedy_tok_s = summary["arm_1_greedy"]["avg_throughput_tok_s"]
    linear_tok_s = summary["arm_2_previous_linear"]["avg_throughput_tok_s"]
    syntax_tok_s = summary["arm_3_syntax_fast_forward"]["avg_throughput_tok_s"]

    summary["arm_1_greedy"]["speedup_vs_greedy"] = "1.00x"
    summary["arm_2_previous_linear"]["speedup_vs_greedy"] = f"{round(linear_tok_s / max(1e-4, greedy_tok_s), 2)}x"
    summary["arm_3_syntax_fast_forward"]["speedup_vs_greedy"] = f"{round(syntax_tok_s / max(1e-4, greedy_tok_s), 2)}x"
    summary["arm_3_syntax_fast_forward"]["speedup_vs_previous_linear"] = (
        f"{round(syntax_tok_s / max(1e-4, linear_tok_s), 2)}x"
    )

    scorecard = {
        "benchmark": "syntax_fast_forward_e2e_scorecard",
        "hardware": "AMD Radeon RX 7900 XTX (gfx1100)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "results_by_arm": results_by_arm,
    }

    out_file = Path("results/benchmarks/syntax_fast_forward_e2e_scorecard.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(scorecard, f, indent=2)

    print("\n" + "=" * 80)
    print("📊 FINAL DUAL A/B BENCHMARK SCORECARD")
    print("=" * 80)
    sp_greedy_prev = summary['arm_2_previous_linear']['speedup_vs_greedy']
    sp_greedy_syn = summary['arm_3_syntax_fast_forward']['speedup_vs_greedy']
    sp_prev_syn = summary['arm_3_syntax_fast_forward']['speedup_vs_previous_linear']

    print(f"1. Pure Greedy Baseline:                {greedy_tok_s:.1f} tok/s (1.00x)")
    print(f"2. Previous Version (Linear MTP):       {linear_tok_s:.1f} tok/s ({sp_greedy_prev} vs greedy)")
    print(f"3. Our Engine (+ Syntax Fast-Forward):  {syntax_tok_s:.1f} tok/s")
    print(f"   ({sp_greedy_syn} vs greedy, {sp_prev_syn} vs previous)")
    print(f"[*] Scorecard saved to: {out_file}")
    print("=" * 80)

    return scorecard


if __name__ == "__main__":
    run_e2e_benchmark()
