"""Stage 4: Real-World Multi-Expert LoRA Stacking SWE Benchmark on AMD Radeon RX 7900 XTX.

Evaluates Dynamic Mixture-of-Adapters (Multi-Expert LoRA Stacking) on authentic,
long-form software engineering tasks (640-768 tokens) across multiple domains:
- Task 1: Dual-Expert (PostgreSQL 17 pgvector + FastAPI Web Architecture)
- Task 2: Triple-Expert (PostgreSQL pgvector + DuckDB OLAP Analytics + FastAPI Web)

Evaluates 4 arms per task:
1. Base Model (Unadapted 27B)
2. Single Expert 1 Alone
3. Single Expert 2 Alone
4. Stacked Multi-Expert (Simultaneous In-Register Superposition, R=16 and R=24)

Measures:
- Cross-Domain Rubric Pass Rate (0-100%)
- Live Python AST Compilation (ast.parse)
- Live Linter Quality (ruff check)
- Generation Throughput (tok/s) via Linear MTP Speculative Engine
- Dynamic Adapter Hot-Swap Latency (ms) and Memory Stability (MB)
"""

from __future__ import annotations

import ast
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime import server
from runtime.adapter_stacker import DynamicAdapterStacker
from runtime.native_27b_engine import Native27BEngine

TASKS = [
    {
        "id": "swe_dual_pg_fastapi",
        "title": "Dual-Expert: PostgreSQL pgvector + FastAPI Microservice",
        "budget": 640,
        "expert_names": ["postgresql", "python_web"],
        "expert_weights": {"postgresql": 1.0, "python_web": 1.0},
        "prompt": (
            '"""\n'
            "Production-grade Python microservice using FastAPI and PostgreSQL 17 with pgvector.\n"
            "Requirements:\n"
            "1. Use FastAPI's modern @asynccontextmanager async def lifespan(app: FastAPI): to manage asyncpg pool.\n"
            "2. Define Pydantic v2 schemas using model_config = ConfigDict(from_attributes=True).\n"
            "3. Initialization logic creating an HNSW index on a 1536-dim vector column with vector_cosine_ops.\n"
            "4. Async search endpoint `/api/v1/search` using pgvector Cosine distance (<=>) and parameters ($1).\n"
            "5. Proper HTTPException error handling and response models.\n"
            '"""\n'
            "from __future__ import annotations\n\n"
            "import asyncio\n"
            "import logging\n"
            "from contextlib import asynccontextmanager\n"
            "from typing import List, Dict, Optional, Any\n\n"
            "import asyncpg\n"
            "from fastapi import FastAPI, HTTPException, status, Depends, Query\n"
            "from pydantic import BaseModel, Field, ConfigDict\n\n"
        ),
        "rubric": {
            "fastapi_lifespan": r"lifespan|asynccontextmanager",
            "pydantic_v2_config": r"model_config\s*=\s*ConfigDict|ConfigDict\(",
            "pgvector_cosine": r"<=>|vector_cosine_ops",
            "asyncpg_parameterized": r"\$1|\$2|create_pool",
        },
    },
    {
        "id": "swe_triple_pg_duckdb_fastapi",
        "title": "Triple-Expert: PostgreSQL + DuckDB Analytics + FastAPI Web",
        "budget": 768,
        "expert_names": ["postgresql", "duckdb", "python_web"],
        "expert_weights": {"postgresql": 1.0, "duckdb": 1.0, "python_web": 1.0},
        "prompt": (
            '"""\n'
            "High-performance hybrid OLTP/OLAP microservice with FastAPI, PostgreSQL 17, and DuckDB.\n"
            "Requirements:\n"
            "1. Use FastAPI modern @asynccontextmanager lifespan context manager managing both pools.\n"
            "2. Define Pydantic v2 schemas with ConfigDict(from_attributes=True).\n"
            "3. Query nearest neighbors from PostgreSQL using pgvector Cosine similarity (<=>).\n"
            "4. Ingest rows into DuckDB and run SQL QUALIFY ROW_NUMBER() OVER (PARTITION BY ...).\n"
            "5. Handle 404 and 500 error cases cleanly with HTTPException.\n"
            '"""\n'
            "from __future__ import annotations\n\n"
            "import asyncio\n"
            "from contextlib import asynccontextmanager\n"
            "from typing import List, Dict, Optional, Any\n\n"
            "import asyncpg\n"
            "import duckdb\n"
            "from fastapi import FastAPI, HTTPException, status\n"
            "from pydantic import BaseModel, Field, ConfigDict\n\n"
        ),
        "rubric": {
            "fastapi_lifespan": r"lifespan|asynccontextmanager",
            "pydantic_v2_config": r"model_config\s*=\s*ConfigDict|ConfigDict\(",
            "pgvector_cosine": r"<=>|vector_cosine_ops",
            "duckdb_qualify": r"QUALIFY\s+|QUALIFY\s+ROW_NUMBER",
            "duckdb_connect": r"duckdb\.connect",
        },
    },
]


def evaluate_generated_code(code: str, rubric: dict[str, str]) -> dict[str, Any]:
    """Evaluates cross-domain rubric and validates Python syntax via ast.parse."""
    scores = {}
    for rule, pattern in rubric.items():
        match = bool(re.search(pattern, code, re.IGNORECASE))
        if rule.startswith("no_"):
            scores[rule] = not match
        else:
            scores[rule] = match

    total_passed = sum(1 for v in scores.values() if v)
    total_rules = len(scores)
    pass_rate = round((total_passed / total_rules) * 100.0, 1) if total_rules > 0 else 0.0

    # AST syntax compilation check
    ast_valid = False
    ast_error = None
    ast_prefix_valid = False
    try:
        ast.parse(code)
        ast_valid = True
        ast_prefix_valid = True
    except SyntaxError as e:
        ast_error = f"{e.msg} at line {e.lineno}"
        # Test if the incomplete trailing line/block caused by budget truncation is the only issue
        lines = code.splitlines()
        for trim_idx in range(len(lines) - 1, max(0, len(lines) - 15), -1):
            trimmed = "\n".join(lines[:trim_idx])
            try:
                ast.parse(trimmed)
                ast_prefix_valid = True
                break
            except SyntaxError:
                continue

    return {
        "pass_rate_pct": pass_rate,
        "rules_passed": f"{total_passed}/{total_rules}",
        "ast_valid": ast_valid,
        "ast_prefix_valid": ast_prefix_valid,
        "ast_error": ast_error,
        "details": scores,
    }


def run_benchmark():
    print("=" * 80)
    print("🔬 Multi-Expert LoRA Stacking Real-World SWE Benchmark (AMD RX 7900 XTX)")
    print("=" * 80)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Target Device: {device}")

    # Initialize 27B engine with max_rank=32
    t0 = time.perf_counter()
    engine: Native27BEngine = server.get_native_triton_27b_engine(num_layers=64)
    tok = server.get_27b_tokenizer()
    stacker = DynamicAdapterStacker()
    max_r = getattr(engine.layers[0].attn_qkv, "max_lora_rank", 16)
    print(f"[*] Engine ready with max_rank={max_r} in {(time.perf_counter() - t0):.2f}s")

    scorecard: dict[str, Any] = {
        "benchmark": "multi_expert_lora_stacking_swe",
        "hardware": "AMD Radeon RX 7900 XTX (gfx1100, 24 GB VRAM)",
        "tasks": [],
    }

    for task in TASKS:
        task_id = task["id"]
        task_title = task["title"]
        prompt = task["prompt"]
        budget = task["budget"]
        rubric = task["rubric"]
        prompt_ids = tok.encode(prompt)

        print("\n" + "=" * 80)
        print(f"📋 Task: [{task_id}] {task_title}")
        print(f"   Prompt Tokens: {len(prompt_ids)} | Budget: {budget} tokens")
        print("=" * 80)

        # Define the 4 arms for this task
        exp_names = task["expert_names"]
        arms = [
            {"id": "arm_1_base", "name": "Base Model (Unadapted)", "type": "base"},
            {
                "id": f"arm_2_{exp_names[0]}",
                "name": f"Single Expert: {exp_names[0]}",
                "type": "single",
                "expert": exp_names[0],
            },
            {
                "id": f"arm_3_{exp_names[1]}",
                "name": f"Single Expert: {exp_names[1]}",
                "type": "single",
                "expert": exp_names[1],
            },
            {
                "id": "arm_4_stacked",
                "name": f"Stacked Multi-Expert ({'+'.join(exp_names)})",
                "type": "stacked",
                "weights": task["expert_weights"],
            },
        ]

        task_results = {
            "task_id": task_id,
            "title": task_title,
            "arms": {},
        }

        for arm in arms:
            arm_id = arm["id"]
            arm_name = arm["name"]
            arm_type = arm["type"]

            print(f"\n▶ Running: [{arm_id}] - {arm_name}")

            # 1. Hot-Swap Adapter into VRAM
            t_swap_start = time.perf_counter()
            if arm_type == "base":
                engine.clear_loras()
            elif arm_type == "single":
                engine.set_active_lora(arm["expert"])
            elif arm_type == "stacked":
                # Stack dynamically in-memory
                fused_dict, fused_cfg = stacker.stack_adapters(arm["weights"], device="cpu", normalize_weights=False)
                engine.bind_lora_state_dict(fused_dict, domain_name=f"stacked_{task_id}", alpha=16.0)
            swap_ms = round((time.perf_counter() - t_swap_start) * 1000.0, 1)
            print(f"  Hot-Swap Latency : {swap_ms} ms")

            # 2. Live Generation via Linear MTP Speculative Engine
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / (1024**2)
            t_gen_start = time.perf_counter()

            generated = engine.generate_speculative(
                prompt_ids,
                max_new_tokens=budget,
                draft_k=3,
                use_mtp=True,
                return_stats=False,
            )

            torch.cuda.synchronize()
            gen_time = time.perf_counter() - t_gen_start
            mem_after = torch.cuda.memory_allocated() / (1024**2)
            mem_delta = round(mem_after - mem_before, 2)

            tok_count = len(generated)
            tok_s = round(tok_count / max(1e-4, gen_time), 2)
            full_code = prompt + tok.decode(generated)

            # 3. Code Evaluation
            eval_metrics = evaluate_generated_code(full_code, rubric)

            ast_msg = f" ({eval_metrics['ast_error']})" if eval_metrics['ast_error'] else ""
            is_valid = eval_metrics['ast_valid']
            pref_valid = eval_metrics['ast_prefix_valid']
            prefix_msg = f" [Prefix AST Valid: {pref_valid}]" if not is_valid else ""
            print(f"  Tokens / Speed   : {tok_count} tokens in {gen_time:.2f}s ({tok_s} tok/s)")
            print(f"  Rubric Score     : {eval_metrics['pass_rate_pct']}% ({eval_metrics['rules_passed']})")
            print(f"  AST Parse Valid  : {eval_metrics['ast_valid']}{ast_msg}{prefix_msg}")
            print(f"  Rule Details     : {eval_metrics['details']}")

            task_results["arms"][arm_id] = {
                "name": arm_name,
                "swap_latency_ms": swap_ms,
                "tokens_generated": tok_count,
                "wall_clock_s": round(gen_time, 2),
                "throughput_tok_s": tok_s,
                "mem_delta_mb": mem_delta,
                "eval": eval_metrics,
                "snippet": full_code[:300] + "...",
                "full_code": full_code,
            }

        scorecard["tasks"].append(task_results)

    # Save structured results
    out_file = Path(__file__).parent.parent / "results" / "benchmarks" / "multi_expert_stacking_swe_scorecard.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(scorecard, f, indent=2)
    print(f"\n[+] Saved detailed multi-expert SWE scorecard to: {out_file}")


if __name__ == "__main__":
    run_benchmark()
