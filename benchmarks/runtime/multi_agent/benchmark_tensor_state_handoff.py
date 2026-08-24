"""Empirical Benchmark: Tensor-Level Recurrent State Handoff ($S_t$) vs Text Re-Prefill.

Audits handoff latency, prefill speedup, token efficiency, and downstream code fidelity
across multi-agent task pipelines at varying context history horizons (T=128..2048).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.runtime.canon import CANON, configure_deterministic_attention
from src.runtime.novel_peft import FoldableExpert, WeightFoldingEngine
from src.runtime.state_handoff import (
    AgentHandoffSession,
    RecurrentStateSnapshot,
    capture_recurrent_state,
)

PIPELINES = [
    {
        "name": "Database Schema -> FastMCP Server -> Pytest Suite",
        "turns": [
            {
                "expert": "postgresql",
                "instruction": "Design a PostgreSQL schema for a multi-tenant vector database with 'tenants', 'documents', and 'embeddings' (vector(1536)). Include an HNSW index.",
            },
            {
                "expert": "astral",
                "instruction": "Write an asynchronous FastMCP Python server that connects via asyncpg to insert documents and perform cosine similarity search on embeddings.",
            },
            {
                "expert": "astral",
                "instruction": "Write a pytest async test suite with fixtures to verify tenant isolation and vector search accuracy.",
            },
        ],
    },
    {
        "name": "Product Catalog -> Recommendation API -> Audit Logger",
        "turns": [
            {
                "expert": "postgresql",
                "instruction": "Design a PostgreSQL table 'products' with price, categories (text[]), and item_embedding vector(384).",
            },
            {
                "expert": "astral",
                "instruction": "Write a FastAPI async endpoint using Pydantic v2 to return top-5 recommended products given a user query vector.",
            },
            {
                "expert": "astral",
                "instruction": "Write a lightweight audit logging middleware that records request latency and memory diffs.",
            },
        ],
    },
]


def run_benchmark(horizons: list[int], max_new_tokens: int = CANON.MAX_NEW_TOKENS, n_repeats: int = 5) -> dict[str, Any]:
    configure_deterministic_attention()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_id = "Qwen/Qwen3.5-4B"
    print(f"[Benchmark] Loading {model_id} in native bfloat16 on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    # Load Domain Experts
    exp_pg = FoldableExpert.from_dir(REPO_ROOT / "results/adapters/m2_postgresql_r8a128_v4", "postgresql")
    exp_astral = FoldableExpert.from_dir(REPO_ROOT / "results/adapters/m2_astral_r8a128_v4", "astral")
    engine = WeightFoldingEngine(model, [exp_pg, exp_astral], keep_pristine=True)
    experts = {"postgresql": exp_pg, "astral": exp_astral}

    session = AgentHandoffSession(model, tokenizer, engine, experts)

    # Pre-warm GPU kernels
    print("[Benchmark] Warming GPU kernels...")
    warm_inp = tokenizer("warmup query", return_tensors="pt").to(device)
    with torch.no_grad():
        w_out = model(**warm_inp, use_cache=True)
        model(torch.tensor([[100]], device=device), past_key_values=w_out.past_key_values, use_cache=True)
    torch.cuda.synchronize()

    benchmark_results: dict[str, Any] = {
        "device": torch.cuda.get_device_name(0),
        "model_id": model_id,
        "horizons": horizons,
        "pipelines": [],
        "scaling_summary": {},
    }

    print("\n" + "=" * 95)
    print(" EXECUTING TENSOR-LEVEL RECURRENT STATE HANDOFF BENCHMARK ($S_t$)")
    print("=" * 95)

    for p_idx, pipeline in enumerate(PIPELINES):
        p_name = pipeline["name"]
        print(f"\n--- Pipeline {p_idx+1}: {p_name} ---")
        p_data = {"name": p_name, "turns_data": []}

        # Initial turn (PostgreSQL Expert)
        turn_1_cfg = pipeline["turns"][0]
        res_1 = session.execute_turn(
            expert_name=turn_1_cfg["expert"],
            instruction=turn_1_cfg["instruction"],
            max_new_tokens=max_new_tokens,
        )

        state_st = res_1.state_snapshot
        turn_1_text = res_1.full_output_text

        print(f"Turn 1 ({turn_1_cfg['expert']}): Generated {res_1.generated_tokens} tok | State size: {state_st.total_mb:.2f} MB")

        for turn_idx in range(1, len(pipeline["turns"])):
            turn_cfg = pipeline["turns"][turn_idx]
            exp_name = turn_cfg["expert"]
            instruction = turn_cfg["instruction"]

            print(f"\nEvaluating Turn {turn_idx+1} ({exp_name}) Handoff across Context Horizons ({n_repeats} repeats, alternating order)...")

            for target_horizon in horizons:
                # Prepare Arm A inputs: Full Text Re-Prefill
                padding_tokens = max(0, target_horizon - res_1.prompt_tokens - res_1.generated_tokens)
                synthetic_history = turn_1_text + ("\n-- comment context\n" * (padding_tokens // 4))

                arm_a_prompt = f"<|im_start|>user\n{turn_1_cfg['instruction']}<|im_end|>\n<|im_start|>assistant\n{synthetic_history}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
                arm_a_inputs = tokenizer(arm_a_prompt, return_tensors="pt").to(device)
                actual_tokens_a = arm_a_inputs.input_ids.shape[1]

                # Prepare Arm B inputs: Tensor State Handoff
                short_prompt = f"<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
                arm_b_inputs = tokenizer(short_prompt, return_tensors="pt").to(device)
                actual_tokens_b = arm_b_inputs.input_ids.shape[1]

                arm_a_times: list[float] = []
                arm_b_times: list[float] = []

                for repeat_i in range(n_repeats):
                    # Alternate order: even repeats = A-then-B, odd repeats = B-then-A
                    a_first = (repeat_i % 2 == 0)

                    def run_arm_a() -> float:
                        engine.activate(experts[exp_name])
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            model(**arm_a_inputs, use_cache=True)
                            torch.cuda.synchronize()
                        return (time.perf_counter() - t0) * 1000.0

                    def run_arm_b() -> float:
                        working_state = state_st.clone()
                        cache_b = working_state.create_cache()
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            model(**arm_b_inputs, past_key_values=cache_b, use_cache=True)
                            torch.cuda.synchronize()
                        return (time.perf_counter() - t0) * 1000.0

                    if a_first:
                        arm_a_times.append(run_arm_a())
                        arm_b_times.append(run_arm_b())
                    else:
                        arm_b_times.append(run_arm_b())
                        arm_a_times.append(run_arm_a())

                # Compute median + IQR
                arm_a_times.sort()
                arm_b_times.sort()
                median_a = arm_a_times[len(arm_a_times) // 2]
                median_b = arm_b_times[len(arm_b_times) // 2]

                def iqr(vals: list[float]) -> tuple[float, float]:
                    q1 = vals[max(0, len(vals) // 4)]
                    q3 = vals[min(len(vals) - 1, 3 * len(vals) // 4)]
                    return q1, q3

                iqr_a = iqr(arm_a_times)
                iqr_b = iqr(arm_b_times)

                # Run Arm C once for the human summary
                res_c = session.execute_turn(
                    expert_name=exp_name,
                    instruction=instruction,
                    state_handoff=state_st,
                    generate_human_summary=True,
                    max_new_tokens=max_new_tokens,
                )

                prefill_speedup = median_a / max(0.001, median_b)
                tokens_saved = actual_tokens_a - actual_tokens_b
                token_savings_pct = (tokens_saved / actual_tokens_a) * 100.0

                print(
                    f"  Horizon ~{target_horizon:4d} tok | "
                    f"Arm A median: {median_a:6.2f} ms [{iqr_a[0]:.1f}–{iqr_a[1]:.1f}] ({actual_tokens_a:4d} tok) | "
                    f"Arm B median: {median_b:6.2f} ms [{iqr_b[0]:.1f}–{iqr_b[1]:.1f}] ({actual_tokens_b:2d} tok) | "
                    f"Speedup: {prefill_speedup:5.2f}x | Saved: {tokens_saved:4d} tok ({token_savings_pct:4.1f}%)"
                )

                turn_data = {
                    "pipeline": p_name,
                    "turn": turn_idx + 1,
                    "target_horizon": target_horizon,
                    "n_repeats": n_repeats,
                    "arm_a_tokens": actual_tokens_a,
                    "arm_a_median_ms": round(median_a, 3),
                    "arm_a_iqr_ms": [round(iqr_a[0], 3), round(iqr_a[1], 3)],
                    "arm_a_all_ms": [round(v, 3) for v in arm_a_times],
                    "arm_b_tokens": actual_tokens_b,
                    "arm_b_median_ms": round(median_b, 3),
                    "arm_b_iqr_ms": [round(iqr_b[0], 3), round(iqr_b[1], 3)],
                    "arm_b_all_ms": [round(v, 3) for v in arm_b_times],
                    "prefill_speedup": round(prefill_speedup, 2),
                    "tokens_saved": tokens_saved,
                    "token_savings_pct": round(token_savings_pct, 1),
                    "human_summary": res_c.human_summary,
                }
                p_data["turns_data"].append(turn_data)

        benchmark_results["pipelines"].append(p_data)

    out_file = REPO_ROOT / "results/benchmarks/tensor_state_handoff_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\n[Persisted] Full benchmark report saved to {out_file}")
    return benchmark_results


def main():
    parser = argparse.ArgumentParser(description="Tensor-Level Recurrent State Handoff Benchmark")
    parser.add_argument("--horizons", nargs="+", type=int, default=[128, 512, 1024, 2048])
    parser.add_argument("--max_new_tokens", type=int, default=CANON.MAX_NEW_TOKENS)
    parser.add_argument("--n_repeats", type=int, default=5, help="Number of repeats per cell (alternating arm order)")
    args = parser.parse_args()

    run_benchmark(horizons=args.horizons, max_new_tokens=args.max_new_tokens, n_repeats=args.n_repeats)


if __name__ == "__main__":
    main()

