"""Empirical Benchmark: True O(1) Tensor State Handoff vs Text Re-prefill on Native 27B Triton Engine.

Executes a realistic 3-agent software engineering collaboration pipeline on AMD Radeon RX 7900 XTX:
- Turn 1: Database Architect (postgresql LoRA)
- Turn 2: Async Tooling Engineer (astral LoRA)
- Turn 3: API Engineer (python_web LoRA)

Empirically compares TTFT / Prefill latency, tokens re-computed vs avoided, LoRA swap time,
and decode throughput under:
1. Tensor-Level Recurrent State Handoff (0.05 ms transfer, 151 MB recurrent state bundle)
2. Text Re-prefill Baseline (re-encoding & re-prefilling conversation history)
"""

import json
import os
import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "apps"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.native_27b_engine import Native27BEngine
from runtime.state_handoff_27b import AgentTurn, StateHandoffSession
from runtime.server import get_27b_tokenizer


def run_benchmark():
    print("=" * 80)
    print("BENCHMARK: True O(1) Tensor State Handoff vs Text Re-prefill (27B Triton Engine)")
    print("Hardware: AMD Radeon RX 7900 XTX (24 GB GDDR6, gfx1100)")
    print("=" * 80)

    # 1. Initialize 27B Engine
    engine = Native27BEngine(num_layers=64)
    engine.load_from_cache()
    tokenizer = get_27b_tokenizer()

    pipeline_turns = [
        AgentTurn(
            agent_id="database_architect",
            role="Database Architect",
            expert_lora="postgresql",
            instruction=(
                "Design a high-throughput PostgreSQL schema for a distributed analytics event log. "
                "Include partition by range on created_at, a JSONB payload with GIN index, and an optimized BRIN index."
            ),
            max_new_tokens=128,
            temperature=0.0,
        ),
        AgentTurn(
            agent_id="tooling_engineer",
            role="Async Tooling Engineer",
            expert_lora="astral",
            instruction=(
                "Based on the PostgreSQL event log schema designed above, write an asynchronous Python ingestion worker "
                "using asyncio and asyncpg. Follow Astral uv and ruff production guidelines with structured logging."
            ),
            max_new_tokens=128,
            temperature=0.0,
        ),
        AgentTurn(
            agent_id="api_engineer",
            role="API Backend Engineer",
            expert_lora="python_web",
            instruction=(
                "Implement a FastAPI router that connects to this database and ingestion pipeline, exposing an SSE stream "
                "for live event monitoring and a POST endpoint for batch ingestion with Pydantic v2 validation."
            ),
            max_new_tokens=128,
            temperature=0.0,
        ),
    ]

    # -------------------------------------------------------------
    # ARM 1: True O(1) Tensor State Handoff
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(">>> EXECUTING ARM 1: TRUE O(1) TENSOR STATE HANDOFF (S_t)")
    print("#" * 80)

    handoff_session = StateHandoffSession(engine=engine)
    handoff_results = []

    for idx, turn in enumerate(pipeline_turns):
        print(f"\n--- [ARM 1 - Turn {idx+1}] Agent: {turn.role} (LoRA: {turn.expert_lora}) ---")
        res = handoff_session.execute_turn(turn)
        handoff_results.append(res)
        print(f"  Prefill: {res.prefill_tokens} tokens in {res.prefill_ms:.2f} ms ({res.tokens_avoided} tokens avoided)")
        print(f"  Handoff Latency: {res.handoff_ms:.3f} ms | LoRA Swap: {res.lora_swap_ms:.2f} ms")
        print(f"  Decode:  {res.tokens_generated} tokens in {res.decode_ms:.2f} ms ({res.tok_per_sec:.2f} tok/s)")
        print(f"  Output Sample:\n{res.output_text[:160]}...\n")

    # -------------------------------------------------------------
    # ARM 2: Standard Text Re-prefill Baseline
    # -------------------------------------------------------------
    print("\n" + "#" * 80)
    print(">>> EXECUTING ARM 2: TEXT RE-PREFILL BASELINE (Re-encoding History)")
    print("#" * 80)

    text_results = []
    history_prompt = ""

    for idx, turn in enumerate(pipeline_turns):
        print(f"\n--- [ARM 2 - Turn {idx+1}] Agent: {turn.role} (LoRA: {turn.expert_lora}) ---")
        t_turn_start = time.perf_counter()

        # LoRA hot swap
        t_lora0 = time.perf_counter()
        engine.set_active_lora(turn.expert_lora)
        lora_swap_ms = (time.perf_counter() - t_lora0) * 1000.0

        # Build full conversation history
        if idx == 0:
            history_prompt = f"<|im_start|>user\n{turn.instruction}<|im_end|>\n<|im_start|>assistant\n"
        else:
            history_prompt += f"<|im_end|>\n<|im_start|>user\n{turn.instruction}<|im_end|>\n<|im_start|>assistant\n"

        prompt_tokens = tokenizer.encode(history_prompt)
        prefill_toks = len(prompt_tokens)

        # Full prompt prefill from scratch
        t_pref0 = time.perf_counter()
        l_first, state_dict = engine.forward_prompt(prompt_tokens)
        prefill_ms = (time.perf_counter() - t_pref0) * 1000.0

        # Decode
        first_token = int(torch.argmax(l_first[0, :]).item())
        gen_tokens = [first_token]
        curr_token = first_token
        pos = prefill_toks
        if engine.hip_graph_captured:
            engine.sync_states_to_graphs(state_dict)

        t_dec0 = time.perf_counter()
        for _ in range(turn.max_new_tokens - 1):
            if first_token in engine.STOP_TOKEN_IDS or turn.max_new_tokens <= 1:
                break
            logits, state_dict = engine.forward_token(curr_token, state_dict, pos=pos, use_graph=True)
            next_token = int(torch.argmax(logits[0, :]).item())
            gen_tokens.append(next_token)
            if next_token in engine.STOP_TOKEN_IDS:
                break
            curr_token = next_token
            pos += 1
        decode_ms = (time.perf_counter() - t_dec0) * 1000.0
        total_ms = (time.perf_counter() - t_turn_start) * 1000.0

        output_text = tokenizer.decode(gen_tokens)
        history_prompt += output_text
        tok_s = (len(gen_tokens) / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0

        res_dict = {
            "agent_id": turn.agent_id,
            "role": turn.role,
            "expert_lora": turn.expert_lora,
            "output_text": output_text,
            "tokens_generated": len(gen_tokens),
            "prefill_tokens": prefill_toks,
            "prefill_ms": prefill_ms,
            "decode_ms": decode_ms,
            "lora_swap_ms": lora_swap_ms,
            "total_ms": total_ms,
            "tok_per_sec": tok_s,
        }
        text_results.append(res_dict)

        print(f"  Prefill: {prefill_toks} tokens in {prefill_ms:.2f} ms (re-prefilled full history)")
        print(f"  LoRA Swap: {lora_swap_ms:.2f} ms")
        print(f"  Decode:  {len(gen_tokens)} tokens in {decode_ms:.2f} ms ({tok_s:.2f} tok/s)")
        print(f"  Output Sample:\n{output_text[:160]}...\n")

    # -------------------------------------------------------------
    # Comparison & Summary Report
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EMPIRICAL COMPARISON & SPEEDUP SUMMARY")
    print("=" * 80)

    print(f"{'Turn':<6} | {'Agent / Role':<24} | {'State Handoff TTFT':<18} | {'Text Prefill TTFT':<18} | {'Speedup':<8}")
    print("-" * 80)

    total_pref_ms_handoff = 0.0
    total_pref_ms_text = 0.0
    total_tokens_avoided = 0

    for i in range(len(pipeline_turns)):
        h = handoff_results[i]
        t = text_results[i]
        total_pref_ms_handoff += h.prefill_ms
        total_pref_ms_text += t["prefill_ms"]
        total_tokens_avoided += h.tokens_avoided

        speedup = t["prefill_ms"] / max(h.prefill_ms, 0.01)
        role = pipeline_turns[i].role
        print(f"Turn {i+1:<2} | {role:<24} | {h.prefill_ms:7.2f} ms ({h.prefill_tokens} tok) | {t['prefill_ms']:7.2f} ms ({t['prefill_tokens']} tok) | {speedup:6.2f}x")

    cumulative_speedup = total_pref_ms_text / max(total_pref_ms_handoff, 0.01)
    print("-" * 80)
    print(f"CUMULATIVE PREFILL LATENCY:")
    print(f"  State Handoff:       {total_pref_ms_handoff:.2f} ms (Tokens re-computed: {sum(h.prefill_tokens for h in handoff_results)})")
    print(f"  Text Re-prefill:     {total_pref_ms_text:.2f} ms (Tokens re-computed: {sum(t['prefill_tokens'] for t in text_results)})")
    print(f"  Tokens Saved/Avoided: {total_tokens_avoided} tokens")
    print(f"  Net Prefill Speedup: {cumulative_speedup:.2f}x")
    print(f"  Average Handoff Overhead: {sum(h.handoff_ms for h in handoff_results) / len(handoff_results):.4f} ms")
    print(f"  Average LoRA Swap Latency: {sum(h.lora_swap_ms for h in handoff_results) / len(handoff_results):.2f} ms")
    print("=" * 80)

    # Save benchmark results to json
    results_payload = {
        "hardware": "AMD Radeon RX 7900 XTX (24 GB, gfx1100)",
        "model": "Qwen2.5-Coder-27B-W4A16-DeltaNet",
        "cumulative_prefill_speedup": round(cumulative_speedup, 2),
        "total_tokens_avoided": total_tokens_avoided,
        "turns": [
            {
                "turn": i + 1,
                "role": pipeline_turns[i].role,
                "expert": pipeline_turns[i].expert_lora,
                "handoff_prefill_ms": round(handoff_results[i].prefill_ms, 2),
                "handoff_prefill_tokens": handoff_results[i].prefill_tokens,
                "handoff_tokens_avoided": handoff_results[i].tokens_avoided,
                "handoff_overhead_ms": round(handoff_results[i].handoff_ms, 4),
                "lora_swap_ms": round(handoff_results[i].lora_swap_ms, 2),
                "text_prefill_ms": round(text_results[i]["prefill_ms"], 2),
                "text_prefill_tokens": text_results[i]["prefill_tokens"],
                "speedup": round(text_results[i]["prefill_ms"] / max(handoff_results[i].prefill_ms, 0.01), 2),
            }
            for i in range(len(pipeline_turns))
        ],
    }

    out_path = Path("benchmarks/state_handoff_27b_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\n[Artifact Saved] Benchmark telemetry saved to {out_path}")


if __name__ == "__main__":
    run_benchmark()
