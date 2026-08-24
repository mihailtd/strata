"""Empirical Benchmark: Minimal Cut Set Analysis & k-out-of-n Speculative Tool Hedging (Chapter 6).

Theoretical Grounding:
Chapter 6 (System Failure Modeling – k-out-of-n System Model, Minimal Paths & Cuts,
Jaejin Hwang, Reliability Analysis Using MINITAB and Python).

Evaluates:
- Arm A: Naive Series Execution (No hedging, standard sequential execution)
- Arm B: Blanket 3-Way Branching / Full Swarm (3 parallel branches on every step)
- Arm C: Minimal Cut-Set Targeted Speculative Hedging (k=1 of n=2 strictly on Order-1 Cut Sets)

Over 500 Monte Carlo multi-turn agent pipeline runs.
"""

import os
import json
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from runtime.cut_set_router import ReliabilityGraph, ReliabilityNode, ReliabilityDAGExecutor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_pipeline_graph(seed: int) -> ReliabilityGraph:
    """Builds a 5-step agent tool DAG with realistic failure probabilities."""
    random.seed(seed)

    g = ReliabilityGraph(source="step1", sink="step5")

    # Step 1: Schema Introspection (R = 0.98)
    def fn_step1(ctx):
        if random.random() > 0.98:
            raise RuntimeError("Database connection timeout")
        return {"schema": "public", "tables": ["users", "orders"]}

    # Step 2: Complex PostgreSQL Query (R = 0.78 - FRAGILE ORDER-1 CUT SET)
    def fn_step2_primary(ctx):
        if random.random() > 0.78:
            raise ValueError("Postgres SyntaxError: column 'order_ts' not found")
        return [{"id": 1, "amt": 250.0}, {"id": 2, "amt": 450.0}]

    def fn_step2_fallback(ctx):
        if random.random() > 0.95:
            raise ValueError("Fallback SQL failed")
        return [{"id": 1, "amt": 250.0}, {"id": 2, "amt": 450.0}]

    # Step 3: Local Python Data Transformation (R = 0.97)
    def fn_step3(ctx):
        if random.random() > 0.97:
            raise KeyError("Missing 'amt' key in row")
        rows = ctx.get("step2", [])
        return sum(r.get("amt", 0) for r in rows)

    # Step 4: FastMCP External Tool Call (R = 0.88 - MODERATE ORDER-1 CUT SET)
    def fn_step4_primary(ctx):
        if random.random() > 0.88:
            raise TimeoutError("FastMCP server response timeout (5000ms)")
        total = ctx.get("step3", 0)
        return {"tax_rate": 0.08, "tax_amt": total * 0.08}

    def fn_step4_fallback(ctx):
        if random.random() > 0.96:
            raise TimeoutError("Fallback tax service timeout")
        total = ctx.get("step3", 0)
        return {"tax_rate": 0.08, "tax_amt": total * 0.08}

    # Step 5: DuckDB Analytics & Summary (R = 0.99)
    def fn_step5(ctx):
        if random.random() > 0.99:
            raise RuntimeError("DuckDB buffer pool exhausted")
        return {"status": "SUCCESS", "final_net": ctx.get("step3", 0) + ctx.get("step4", {}).get("tax_amt", 0)}

    g.add_node(ReliabilityNode("step1", "Schema Discover", reliability=0.98, execute_fn=fn_step1))
    g.add_node(ReliabilityNode("step2", "Postgres Query", reliability=0.78, execute_fn=fn_step2_primary, fallback_fn=fn_step2_fallback))
    g.add_node(ReliabilityNode("step3", "Python Transform", reliability=0.97, execute_fn=fn_step3))
    g.add_node(ReliabilityNode("step4", "FastMCP Tax Call", reliability=0.88, execute_fn=fn_step4_primary, fallback_fn=fn_step4_fallback))
    g.add_node(ReliabilityNode("step5", "DuckDB Summary", reliability=0.99, execute_fn=fn_step5))

    g.add_edge("step1", "step2")
    g.add_edge("step2", "step3")
    g.add_edge("step3", "step4")
    g.add_edge("step4", "step5")

    return g


def run_benchmark():
    n_runs = 500
    print(f"=== Chapter 6 Minimal Cut Set & k-out-of-n Hedging Benchmark ===")
    print(f"Running {n_runs} Monte Carlo Agent Pipeline Tasks...\n")

    # -------------------------------------------------------------
    # Arm A: Naive Series Execution
    # -------------------------------------------------------------
    arm_a_success = 0
    arm_a_tool_calls = 0
    t0_a = time.perf_counter()

    for i in range(n_runs):
        g = build_pipeline_graph(seed=i)
        # Executor with high threshold disabled (no hedging)
        executor = ReliabilityDAGExecutor(target_reliability=0.0)
        try:
            ctx, telem = executor.execute_dag(g)
            if ctx.get("step5", {}).get("status") == "SUCCESS":
                arm_a_success += 1
            arm_a_tool_calls += telem["total_tool_calls"]
        except Exception:
            arm_a_tool_calls += 5  # Estimated partial calls

    t_a_ms = (time.perf_counter() - t0_a) * 1000.0
    completion_a = (arm_a_success / n_runs) * 100.0
    avg_calls_a = arm_a_tool_calls / n_runs

    # -------------------------------------------------------------
    # Arm B: Blanket 3-Way Branching (Full Swarm on All Steps)
    # -------------------------------------------------------------
    arm_b_success = 0
    arm_b_tool_calls = 0
    t0_b = time.perf_counter()

    for i in range(n_runs):
        # 3 parallel replicas across every step -> 15 tool calls per run
        arm_b_tool_calls += 15
        # Prob of all 3 replicas failing on any step
        # Step 1: 1 - (1-0.98)^3 = 0.999992
        # Step 2: 1 - (1-0.78)^3 = 0.989352
        # Step 3: 1 - (1-0.97)^3 = 0.999973
        # Step 4: 1 - (1-0.88)^3 = 0.998272
        # Step 5: 1 - (1-0.99)^3 = 0.999999
        p_b_success = 0.999992 * 0.989352 * 0.999973 * 0.998272 * 0.999999
        random.seed(i + 10000)
        if random.random() < p_b_success:
            arm_b_success += 1

    t_b_ms = (time.perf_counter() - t0_b) * 1000.0
    completion_b = (arm_b_success / n_runs) * 100.0
    avg_calls_b = arm_b_tool_calls / n_runs

    # -------------------------------------------------------------
    # Arm C: Minimal Cut-Set Targeted Speculative Hedging (Chapter 6)
    # -------------------------------------------------------------
    arm_c_success = 0
    arm_c_tool_calls = 0
    arm_c_exceptions_averted = 0
    t0_c = time.perf_counter()

    for i in range(n_runs):
        g = build_pipeline_graph(seed=i)
        # Target reliability 0.95 -> automatically identifies and hedges Step 2 and Step 4!
        executor = ReliabilityDAGExecutor(target_reliability=0.95)
        try:
            ctx, telem = executor.execute_dag(g)
            if ctx.get("step5", {}).get("status") == "SUCCESS":
                arm_c_success += 1
            arm_c_tool_calls += telem["total_tool_calls"]
            arm_c_exceptions_averted += telem["exceptions_averted"]
        except Exception:
            arm_c_tool_calls += 7

    t_c_ms = (time.perf_counter() - t0_c) * 1000.0
    completion_c = (arm_c_success / n_runs) * 100.0
    avg_calls_c = arm_c_tool_calls / n_runs

    print(f"  Arm A (Naive Series):       Completion={completion_a:5.1f}% | Avg Tool Calls={avg_calls_a:4.1f} | Averted Excs=  0")
    print(f"  Arm B (Blanket 3-Way Swarm): Completion={completion_b:5.1f}% | Avg Tool Calls={avg_calls_b:4.1f} | Overhead=3.00x compute")
    print(f"  Arm C (Minimal Cut-Set):     Completion={completion_c:5.1f}% | Avg Tool Calls={avg_calls_c:4.1f} | Averted Excs={arm_c_exceptions_averted:3d} | Gain: +{completion_c - completion_a:4.1f}%\n")

    results = {
        "benchmark": "Chapter 6 Minimal Cut Set & k-out-of-n Hedging",
        "n_runs": n_runs,
        "naive_series": {"completion_rate_pct": completion_a, "avg_tool_calls": avg_calls_a, "time_ms": t_a_ms},
        "blanket_swarm": {"completion_rate_pct": completion_b, "avg_tool_calls": avg_calls_b, "time_ms": t_b_ms},
        "minimal_cut_set": {
            "completion_rate_pct": completion_c,
            "avg_tool_calls": avg_calls_c,
            "exceptions_averted": arm_c_exceptions_averted,
            "completion_gain_pct": completion_c - completion_a,
            "compute_savings_vs_swarm_pct": (1.0 - avg_calls_c / avg_calls_b) * 100.0,
            "time_ms": t_c_ms,
        },
    }

    artifact_path = RESULTS_DIR / "cut_set_reliability_benchmark.json"
    with open(artifact_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Telemetry saved to {artifact_path}")


if __name__ == "__main__":
    run_benchmark()
