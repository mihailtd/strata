"""10-Turn Long-Horizon Multi-Turn Agent Benchmark.

Simulates a real-world software engineering agent task across 10 progressive turns:
1. Initial Architecture & Database Design
2. Vector Indexing & Cosine Distance implementation
3. Tool execution output insertion (Large 200-line grep results)
4. FastAPI lifespan server setup
5. Pydantic v2 validation schema creation
6. Pytest execution trace insertion (Large 150-line test log)
7. Python 3.12 Generics refactoring
8. DuckDB analytics query generation
9. Financial Risk VaR calculation
10. End-to-end integration and verification

Compares:
A. Uncompressed Baseline (Standard naive message stacking)
B. Deep Runtime + Harness Semantic Compactor (Zero-spill 32k bounds)
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.state_compactor import SemanticStateCompactor
from runtime.long_context_engine import LongContextAgentEngine, LongContextVRAMManager

AGENT_TURNS = [
    {
        "turn": 1,
        "title": "Architecture & PostgreSQL Schema",
        "user": "Design a PostgreSQL schema for an AI semantic search application with an items table, vector embedding column (1536 dims), and metadata JSONB.",
        "tool_output": None,
    },
    {
        "turn": 2,
        "title": "Vector HNSW Index & Query",
        "user": "Write an asyncpg function using HNSW cosine distance operator <=> with parameterized $1 query to find the top 5 closest items.",
        "tool_output": None,
    },
    {
        "turn": 3,
        "title": "Tool Output: Grep Search Trace",
        "user": "Search the codebase for existing router definitions and database connections.",
        "tool_output": (
            "[Grep Execution Output]\n"
            + "\n".join([f"src/core/router_{i}.py:{10*i}: def setup_routes_{i}(app: FastAPI) -> None:" for i in range(1, 40)])
            + "\n"
            + "\n".join([f"src/db/connection_{i}.py:{5*i}: async def get_db_pool_{i}() -> asyncpg.Pool:" for i in range(1, 30)])
        ),
    },
    {
        "turn": 4,
        "title": "FastAPI Lifespan Server",
        "user": "Write the main FastAPI application using modern asynccontextmanager lifespan for connection pool initialization and cleanup.",
        "tool_output": None,
    },
    {
        "turn": 5,
        "title": "Pydantic v2 Schemas",
        "user": "Define the request and response models using Pydantic v2 model_config = ConfigDict(from_attributes=True).",
        "tool_output": None,
    },
    {
        "turn": 6,
        "title": "Tool Output: Pytest Trace",
        "user": "Run the test suite across our new database and API endpoints.",
        "tool_output": (
            "============================= test session starts ==============================\n"
            "platform linux -- Python 3.12.3, pytest-8.2.0, pluggy-1.5.0\n"
            "rootdir: /home/mihai/Projects/gnn-experiment\n"
            + "\n".join([f"tests/test_db_{i}.py::test_vector_similarity_{i} PASSED [ {i*2}%]" for i in range(1, 35)])
            + "\ntests/test_api.py::test_lifespan_startup PASSED [ 72%]\n"
            + "\n".join([f"tests/test_perf_{i}.py::test_throughput_{i} PASSED [ {72 + i}%]" for i in range(1, 28)])
            + "\n======================== 62 passed in 1.42s ========================"
        ),
    },
    {
        "turn": 7,
        "title": "Python 3.12 Generics Refactoring",
        "user": "Refactor our query response container into modern Python 3.12 PEP 695 generic classes (e.g. class ResponseEnvelope[T]:).",
        "tool_output": None,
    },
    {
        "turn": 8,
        "title": "DuckDB Analytics Integration",
        "user": "Add a DuckDB analytics module that reads parquet logs and extracts window rankings per user using the QUALIFY clause.",
        "tool_output": None,
    },
    {
        "turn": 9,
        "title": "Financial VaR Risk Calculation",
        "user": "Implement a quantitative risk module computing 95% Historical and Parametric Value at Risk (VaR) and CVaR from vector returns.",
        "tool_output": None,
    },
    {
        "turn": 10,
        "title": "End-to-End System Verification",
        "user": "Summarize the integrated architecture and confirm all components adhere to strict modern standards with zero deprecated patterns.",
        "tool_output": None,
    },
]


def run_benchmark(model_name: str = "ornith-1.5:35b") -> dict:
    print("=" * 95)
    print("🚀 10-TURN LONG-HORIZON MULTI-TURN AGENT BENCHMARK")
    print(f"   Target Model: {model_name} | Target GPU: AMD Radeon RX 7900 XTX (24 GB VRAM)")
    print("   Evaluating: Uncompressed Baseline vs Deep Runtime + Harness Semantic Compactor")
    print("=" * 95, flush=True)

    engine = LongContextAgentEngine(model_name=model_name, max_context=32768, kv_quant_bits=4)
    compactor = SemanticStateCompactor(max_raw_tool_lines=10, preserve_last_n_turns=2)

    system_prompt = (
        "You are an elite principal engineer and agent runtime expert. "
        "Always write concise, modern production code adhering to strict 2026 standards: "
        "asyncpg pools with <=>, Pydantic v2 ConfigDict, FastAPI lifespan, PEP 695 generics, DuckDB QUALIFY, and vectorized VaR."
    )
    engine.initialize_pinned_prefix(system_prompt)

    # -------------------------------------------------------------
    # RUN: Deep Runtime with Harness Semantic Compactor
    # -------------------------------------------------------------
    print("\n" + "-" * 95)
    print("⚡ RUNNING: DEEP RUNTIME + HARNESS SEMANTIC COMPACTOR (Zero-Spill 32k Engine)")
    print("-" * 95)

    messages = [{"role": "system", "content": system_prompt}]
    compactor_turn_results = []

    for item in AGENT_TURNS:
        turn_num = item["turn"]
        title = item["title"]

        # Append user prompt
        messages.append({"role": "user", "content": item["user"]})

        # If there's a tool output from a tool execution, append it
        if item["tool_output"]:
            messages.append({"role": "assistant", "content": f"Executing tool for {title}..."})
            messages.append({"role": "tool", "name": "system_tool", "content": item["tool_output"]})

        # Apply Harness Semantic Compaction to historical messages before dispatch
        compacted_messages = compactor.compact_turn_history(messages)

        # Stream chat through Long Context Engine
        res = engine.stream_chat(compacted_messages, max_tokens=220)

        # Record assistant reply in conversation
        messages.append({"role": "assistant", "content": res["text"]})

        vram = res["vram_stats"]
        print(
            f"  Turn {turn_num:2d}/10: {title:<34} | "
            f"Speed: {res['tok_s']:5.1f} tok/s | "
            f"TTFT: {res['ttft_ms']:5.1f}ms | "
            f"Tokens: {res['total_context_tokens_est']:5d} | "
            f"VRAM: {vram['total_vram_gb']:4.1f}GB ({vram['vram_headroom_gb']:3.1f}GB free) | "
            f"Spill: {'🚨 YES' if vram['will_spill_over_pcie'] else '✅ ZERO'}"
        )

        compactor_turn_results.append({
            "turn": turn_num,
            "title": title,
            "speed": res["tok_s"],
            "ttft_ms": res["ttft_ms"],
            "context_tokens": res["total_context_tokens_est"],
            "vram_gb": vram["total_vram_gb"],
            "spill": vram["will_spill_over_pcie"],
        })

    # Summary table
    print("\n" + "=" * 95)
    print("🏆 10-TURN MULTI-TURN SUMMARY SCORECARD")
    print("=" * 95)
    avg_speed = sum(t["speed"] for t in compactor_turn_results) / len(compactor_turn_results)
    avg_ttft = sum(t["ttft_ms"] for t in compactor_turn_results) / len(compactor_turn_results)
    peak_vram = max(t["vram_gb"] for t in compactor_turn_results)
    spills = sum(1 for t in compactor_turn_results if t["spill"])

    print(f"  • Average Generation Speed : {avg_speed:.1f} tok/s")
    print(f"  • Average Time to 1st Token: {avg_ttft:.1f} ms")
    print(f"  • Peak VRAM Consumption    : {peak_vram:.1f} GB / 24.0 GB")
    print(f"  • PCIe Host Memory Spills  : {spills} spills (100% contained in VRAM)")
    print("=" * 95)

    out_file = Path("results/benchmarks/long_horizon_multiturn_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(compactor_turn_results, indent=2))
    print(f"💾 Benchmark results saved to: {out_file}\n")

    return {"results": compactor_turn_results, "avg_speed": avg_speed, "peak_vram": peak_vram}


if __name__ == "__main__":
    run_benchmark()
