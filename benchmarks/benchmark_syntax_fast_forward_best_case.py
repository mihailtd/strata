"""Stage 4: Best-Case 1K-2K Token End-to-End Benchmark for Syntax Fast-Forwarding.

Evaluates a real 1,024-token software engineering generation task on AMD Radeon RX 7900 XTX:
- Arm 1: Pure Greedy Baseline (K=1)
- Arm 2: Previous Version: Linear MTP + N-Gram (no syntax drafter)
- Arm 3: Our Engine: Syntax Fast-Forward + N-Gram + Neural MTP

Measures real wall-clock time, tok/s throughput, and granular draft acceptance rates
by drafter source (Syntax Trie vs N-Gram vs Neural MTP).
Saves results to results/benchmarks/syntax_fast_forward_best_case_scorecard.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime import server
from runtime.native_27b_engine import Native27BEngine
from runtime.syntax_drafter import SyntaxTrieDrafter

# Prompt designed to generate a comprehensive enterprise FastAPI service and PyTest suite.
# The structural repetition of endpoints, schemas, error handling, and test assertions provides
# the optimal realistic scenario for Deterministic AST & Syntax Fast-Forwarding.
BENCHMARK_PROMPT = (
    '"""\n'
    "Production-grade FastAPI Inventory & Billing Microservice and Async PyTest Suite.\n"
    "Requirements:\n"
    "1. Pydantic request/response models with ConfigDict(from_attributes=True) and Field validation.\n"
    "2. APIRouter with endpoints for item creation, retrieval, listing, update, and deletion.\n"
    "3. Proper HTTPException error handling (404 Not Found, 400 Bad Request, 201 Created).\n"
    "4. Database interaction using asyncpg connection pool.\n"
    "5. Comprehensive Async PyTest test cases testing every endpoint, checking status codes, "
    "JSON responses, and error flows.\n"
    '"""\n'
    "from __future__ import annotations\n\n"
    "import asyncio\n"
    "import logging\n"
    "from typing import List, Dict, Optional, Any\n\n"
    "import asyncpg\n"
    "import pytest\n"
    "from fastapi import FastAPI, APIRouter, HTTPException, status, Depends\n"
    "from pydantic import BaseModel, Field, ConfigDict\n\n"
)

TARGET_TOKENS = 1024


def run_best_case_benchmark():
    print("=" * 80)
    print(f"🚀 Best-Case 1K-Token Speculative Benchmark ({TARGET_TOKENS} tokens)")
    print("   Target Device: AMD Radeon RX 7900 XTX (Navi 31 / gfx1100, 24 GB VRAM)")
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

    prompt_ids = tok.encode(BENCHMARK_PROMPT)
    print(f"[*] Benchmark Prompt Length: {len(prompt_ids)} tokens")

    arms = [
        {"id": "arm_1_greedy", "name": "Pure Greedy Baseline (K=1)"},
        {"id": "arm_2_previous_linear", "name": "Previous Version: Linear MTP + N-Gram"},
        {"id": "arm_3_syntax_fast_forward", "name": "Our Engine: Syntax Fast-Forward + N-Gram + MTP"},
    ]

    scorecard: dict[str, Any] = {
        "benchmark": "syntax_fast_forward_best_case_1k",
        "hardware": "AMD Radeon RX 7900 XTX (gfx1100, 24 GB VRAM)",
        "prompt_tokens": len(prompt_ids),
        "target_tokens": TARGET_TOKENS,
        "arms": {},
    }

    arm_outputs: dict[str, list[int]] = {}

    for arm in arms:
        arm_id = arm["id"]
        arm_name = arm["name"]
        print("\n" + "-" * 75)
        print(f"▶ Running Arm: [{arm_id}] - {arm_name}")
        print("-" * 75)

        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / (1024**2)

        t_start = time.perf_counter()
        stats_out: dict[str, Any] = {}

        if arm_id == "arm_1_greedy":
            engine.syntax_drafter = None
            generated = engine.generate(
                prompt_ids,
                max_new_tokens=TARGET_TOKENS,
                temperature=0.0,
                use_hip_graph=True,
            )
        elif arm_id == "arm_2_previous_linear":
            engine.syntax_drafter = None
            generated, stats_out = engine.generate_speculative(
                prompt_ids,
                max_new_tokens=TARGET_TOKENS,
                draft_k=3,
                use_mtp=True,
                return_stats=True,
            )
        elif arm_id == "arm_3_syntax_fast_forward":
            engine.syntax_drafter = active_syntax_drafter
            generated, stats_out = engine.generate_speculative(
                prompt_ids,
                max_new_tokens=TARGET_TOKENS,
                draft_k=3,
                use_mtp=True,
                return_stats=True,
            )
        else:
            raise ValueError(f"Unknown arm: {arm_id}")

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        mem_after = torch.cuda.memory_allocated() / (1024**2)
        mem_delta = mem_after - mem_before

        tok_count = len(generated)
        throughput = tok_count / elapsed if elapsed > 0 else 0.0
        arm_outputs[arm_id] = generated

        print(f"  Generated Tokens : {tok_count}")
        print(f"  Wall-Clock Time  : {elapsed:.2f} s")
        print(f"  Throughput       : {throughput:.2f} tok/s")
        print(f"  Memory Delta     : {mem_delta:.2f} MB")

        if stats_out:
            print("  [Drafter Breakdown]")
            print(f"    Total Steps         : {stats_out.get('total_steps', 0)}")
            s_prop = stats_out.get("syntax_drafts_proposed", 0)
            s_acc = stats_out.get("syntax_tokens_accepted", 0)
            s_rate = (s_acc / s_prop * 100) if s_prop > 0 else 0.0
            print(f"    Syntax Trie Drafts  : {s_prop} proposed, {s_acc} accepted ({s_rate:.1f}%)")

            n_prop = stats_out.get("ngram_drafts_proposed", 0)
            n_acc = stats_out.get("ngram_tokens_accepted", 0)
            n_rate = (n_acc / n_prop * 100) if n_prop > 0 else 0.0
            print(f"    N-Gram Drafts       : {n_prop} proposed, {n_acc} accepted ({n_rate:.1f}%)")

            m_prop = stats_out.get("mtp_drafts_proposed", 0)
            m_acc = stats_out.get("mtp_tokens_accepted", 0)
            m_rate = (m_acc / m_prop * 100) if m_prop > 0 else 0.0
            print(f"    Neural MTP Drafts   : {m_prop} proposed, {m_acc} accepted ({m_rate:.1f}%)")
            print(f"    Single-Token Steps  : {stats_out.get('single_token_steps', 0)}")

        scorecard["arms"][arm_id] = {
            "name": arm_name,
            "tokens_generated": tok_count,
            "wall_clock_s": round(elapsed, 2),
            "throughput_tok_s": round(throughput, 2),
            "mem_delta_mb": round(mem_delta, 2),
            "stats": stats_out,
            "snippet": tok.decode(generated[:30]),
        }

    # Speedup calculations
    t_greedy = scorecard["arms"]["arm_1_greedy"]["throughput_tok_s"]
    t_linear = scorecard["arms"]["arm_2_previous_linear"]["throughput_tok_s"]
    t_syntax = scorecard["arms"]["arm_3_syntax_fast_forward"]["throughput_tok_s"]

    scorecard["arms"]["arm_1_greedy"]["speedup_vs_greedy"] = "1.00x"
    scorecard["arms"]["arm_2_previous_linear"]["speedup_vs_greedy"] = f"{t_linear / t_greedy:.2f}x"
    scorecard["arms"]["arm_3_syntax_fast_forward"]["speedup_vs_greedy"] = f"{t_syntax / t_greedy:.2f}x"
    scorecard["arms"]["arm_3_syntax_fast_forward"]["speedup_vs_previous_linear"] = f"{t_syntax / t_linear:.2f}x"

    # Parity check
    greedy_toks = arm_outputs["arm_1_greedy"]
    linear_toks = arm_outputs["arm_2_previous_linear"]
    syntax_toks = arm_outputs["arm_3_syntax_fast_forward"]

    linear_match = (linear_toks == greedy_toks)
    syntax_match = (syntax_toks == greedy_toks)
    speculative_cross_match = (syntax_toks == linear_toks)

    scorecard["parity"] = {
        "linear_matches_greedy": linear_match,
        "syntax_matches_greedy": syntax_match,
        "syntax_matches_linear": speculative_cross_match,
    }

    print("\n" + "=" * 80)
    print("🏁 SUMMARY SCORECARD")
    print("=" * 80)
    print(f"  Arm 1 (Pure Greedy)      : {t_greedy:.1f} tok/s (1.00x)")
    print(f"  Arm 2 (Linear MTP+NGram) : {t_linear:.1f} tok/s ({t_linear/t_greedy:.2f}x vs Greedy)")
    print(
        f"  Arm 3 (Syntax Fast-Fwd)  : {t_syntax:.1f} tok/s "
        f"({t_syntax/t_greedy:.2f}x vs Greedy, {t_syntax/t_linear:.2f}x vs Linear)"
    )
    print(f"  Parity (Linear==Greedy)  : {linear_match}")
    print(f"  Parity (Syntax==Greedy)  : {syntax_match}")
    print(f"  Cross (Syntax==Linear)   : {speculative_cross_match}")

    out_path = Path(__file__).parent.parent / "results" / "benchmarks" / "syntax_fast_forward_best_case_scorecard.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scorecard, f, indent=2)
    print(f"\n[+] Saved detailed scorecard to: {out_path}")


if __name__ == "__main__":
    run_best_case_benchmark()
