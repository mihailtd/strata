#!/usr/bin/env python3
"""Run 4-Turn Multi-Turn Benchmark against Ollama running qwen3.8:27b."""

import json
import time
import requests

OLLAMA_API_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "qwen3.8:27b"

PROMPTS = [
    {
        "id": "turn_1_astral",
        "role": "user",
        "content": "Add ruff and ty as dev dependencies, then format and lint the whole codebase.",
    },
    {
        "id": "turn_2_postgres",
        "role": "user",
        "content": "We store product descriptions in Postgres and want 'find me similar products' without standing up new infrastructure.",
    },
    {
        "id": "turn_3_fastapi",
        "role": "user",
        "content": "Write an async FastAPI endpoint with Pydantic request and response models and dependency injection.",
    },
    {
        "id": "turn_4_duckdb",
        "role": "user",
        "content": "Aggregate a directory of parquet files and return the top 3 rows per group.",
    },
]

def run_ollama_multiturn():
    print("=" * 80)
    print(f"🚀 RUNNING OLLAMA BENCHMARK: {MODEL_NAME}")
    print("=" * 80)

    messages = []
    results = []

    for i, p in enumerate(PROMPTS, start=1):
        messages.append({"role": "user", "content": p["content"]})
        print(f"\n--- Turn {i}: {p['id']} ---")
        print(f"User: {p['content']}")

        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": True,
        }

        t_start = time.perf_counter()
        t_first = None
        response_text = ""
        eval_count = 0
        prompt_eval_count = 0
        prompt_eval_duration_ns = 0
        eval_duration_ns = 0

        try:
            res = requests.post(OLLAMA_API_URL, json=payload, stream=True, timeout=120)
            res.raise_for_status()

            for line in res.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line.decode("utf-8"))
                if t_first is None:
                    t_first = time.perf_counter()
                
                msg = chunk.get("message", {})
                content = msg.get("content", "")
                response_text += content

                if chunk.get("done", False):
                    eval_count = chunk.get("eval_count", 0)
                    prompt_eval_count = chunk.get("prompt_eval_count", 0)
                    prompt_eval_duration_ns = chunk.get("prompt_eval_duration", 0)
                    eval_duration_ns = chunk.get("eval_duration", 0)

            t_end = time.perf_counter()
            ttft_ms = (t_first - t_start) * 1000.0 if t_first else 0.0
            total_time_s = t_end - t_start
            tok_per_sec = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else (eval_count / total_time_s)

            print(f"\nResponse (First 300 chars): {response_text[:300]}...")
            print(f"📊 Telemetry: TTFT={ttft_ms:.1f}ms | Generated={eval_count} tokens | Prompt={prompt_eval_count} tokens | Speed={tok_per_sec:.2f} tok/s")

            results.append({
                "turn": i,
                "prompt": p["content"],
                "response": response_text,
                "ttft_ms": ttft_ms,
                "eval_count": eval_count,
                "prompt_eval_count": prompt_eval_count,
                "tok_per_sec": tok_per_sec,
                "total_time_s": total_time_s,
            })

            # Append assistant response for true multi-turn context retention
            messages.append({"role": "assistant", "content": response_text})

        except Exception as e:
            print(f"❌ Error on Turn {i}: {e}")
            break

    # Save results
    with open("results/benchmarks/ollama_27b_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 80)
    print("✅ Benchmark Complete! Saved to results/benchmarks/ollama_27b_eval.json")
    print("=" * 80)

if __name__ == "__main__":
    run_ollama_multiturn()
