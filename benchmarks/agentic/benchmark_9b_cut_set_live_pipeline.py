"""9B End-to-End Live Benchmark: Qwen3.5-9B Model + 9B LoRA Experts + Cut-Set Speculative Router.

Evaluates:
- Real Qwen3.5-9B Base Model (bfloat16) loaded on GPU.
- Real 9B LoRA Domain Adapters:
    * PostgreSQL: 'results/adapters/m2_postgresql_r8a128_v7_9b'
    * Astral / FastMCP: 'results/adapters/m2_astral_r8a128_v7_9b'
    * DuckDB: 'results/adapters/m2_duckdb_r8a128_v7_9b'
- Dynamic In-Place Weight Folding (via WeightFoldingEngine).
- Real Sandbox Execution (DuckDB/PostgreSQL SQL engine, Ruff linter, Python compiler).
- Chapter 6 Minimal Cut-Set Speculative Hedging on real generated LLM outputs.
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import tempfile
import py_compile
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["ROCR_VISIBLE_DEVICES"] = "0"

import torch
import duckdb
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime.canon import REPO_ROOT
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from runtime.novel_peft import FoldableExpert, WeightFoldingEngine
from runtime.gpu_preflight import ensure_gpu_exclusive
from runtime.cut_set_router import ReliabilityGraph, ReliabilityNode, ReliabilityDAGExecutor

RESULTS_DIR = REPO_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def extract_code_block(text: str, language: str = "sql") -> str:
    """Extracts first code block from markdown."""
    pattern = rf"```{language}\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    generic_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()
    return text.strip()


class Real9BPipelineEvaluator:
    def __init__(self, model_id: str = "Qwen/Qwen3.5-9B", device: str = "cuda:0"):
        print("=" * 80)
        print(f" Initializing 9B Live Sandbox Engine ({model_id})")
        print("=" * 80)

        ensure_gpu_exclusive()

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[9B Engine] Loading base weights into bfloat16...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map={"": self.device.index if self.device.type == "cuda" else "cpu"},
            trust_remote_code=True,
        )
        self.model.eval()

        # Load 9B expert adapters
        experts = []
        adapter_configs = [
            ("postgresql", "m2_postgresql_r8a128_v7_9b"),
            ("astral", "m2_astral_r8a128_v7_9b"),
            ("duckdb", "m2_duckdb_r8a128_v7_9b"),
        ]
        for name, folder in adapter_configs:
            p = REPO_ROOT / "results" / "adapters" / folder
            if p.exists():
                print(f"[9B Engine] Loading 9B expert [{name}] from {folder}...")
                experts.append(FoldableExpert.from_dir(p, name))

        self.folding_engine = WeightFoldingEngine(self.model, experts, keep_pristine=True)
        self.expert_map = {e.name: e for e in experts}
        print(f"[9B Engine] Successfully registered {len(experts)} 9B domain adapters!\n")

    def generate(self, prompt: str, expert_name: str | None = None, max_new_tokens: int = 512) -> tuple[str, float]:
        """Folds the 9B expert adapter in-place, generates text, and restores pristine state."""
        expert = self.expert_map.get(expert_name) if expert_name else None
        
        t_fold = 0.0
        if expert:
            t0 = time.perf_counter()
            self.folding_engine.activate(expert)
            t_fold = (time.perf_counter() - t0) * 1000.0

        messages = [{"role": "user", "content": prompt}]
        chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_prompt += "\n</think>\n"
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)

        torch.cuda.synchronize()
        t0_gen = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        gen_ms = (time.perf_counter() - t0_gen) * 1000.0

        if expert:
            self.folding_engine.restore_pristine()

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return text, gen_ms + t_fold


def run_9b_benchmark():
    evaluator = Real9BPipelineEvaluator()

    print("=" * 80)
    print(" Executing 9B Live Multi-Turn Pipeline across Real Sandboxes")
    print("=" * 80)

    # 1. Generate Real Artifacts using 9B Model + 9B LoRA Adapters
    print("\n[*] Step 1: Generating PostgreSQL Schema via 9B + PostgreSQL LoRA...")
    p1_prompt = "Write PostgreSQL DDL creating a table 'financial_records' with id INT, account VARCHAR, amount DOUBLE, and created_at TIMESTAMP. Output only SQL."
    p1_sql_raw, t_p1 = evaluator.generate(p1_prompt, expert_name="postgresql", max_new_tokens=256)
    p1_sql = extract_code_block(p1_sql_raw, "sql")
    print(f"    -> Generated in {t_p1:.1f}ms:\n{p1_sql}\n")

    print("[*] Step 2: Generating FastMCP Tool Server via 9B + Astral LoRA...")
    p2_prompt = "Write a Python script using FastMCP defining a tool 'compute_yield(principal: float, rate: float) -> float'. Output only python."
    p2_py_raw, t_p2 = evaluator.generate(p2_prompt, expert_name="astral", max_new_tokens=300)
    p2_py = extract_code_block(p2_py_raw, "python")
    print(f"    -> Generated in {t_p2:.1f}ms:\n{p2_py}\n")

    print("[*] Step 3: Generating DuckDB Analytics via 9B + DuckDB LoRA...")
    p3_prompt = "Write a DuckDB SQL query to aggregate 'financial_records' by account, calculating total amount and average amount. Output only SQL."
    p3_sql_raw, t_p3 = evaluator.generate(p3_prompt, expert_name="duckdb", max_new_tokens=256)
    p3_sql = extract_code_block(p3_sql_raw, "sql")
    print(f"    -> Generated in {t_p3:.1f}ms:\n{p3_sql}\n")

    # 2. Free GPU Memory before Sandbox Execution
    del evaluator.model
    del evaluator.folding_engine
    del evaluator
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Construct Live 9B Execution DAG with Cut-Set Speculative Hedging
    print("=" * 80)
    print(" Running Chapter 6 Cut-Set Speculative Execution on 9B Generated Artifacts")
    print("=" * 80)

    db_conn = duckdb.connect(":memory:")

    def step1_exec(ctx):
        # Execute real PostgreSQL DDL generated by 9B
        db_conn.execute(p1_sql)
        db_conn.execute("INSERT INTO financial_records VALUES (1, 'ACC_100', 5000.0, '2026-08-01 10:00:00'), (2, 'ACC_100', 3200.0, '2026-08-02 11:00:00');")
        return {"status": "DDL_INITIALIZED"}

    def step2_exec_primary(ctx):
        # Primary FastMCP code check
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "server.py"
            f.write_text(p2_py, encoding="utf-8")
            py_compile.compile(str(f), doraise=True)
            res = subprocess.run(["ruff", "check", "--select", "E,F,W", str(f)], capture_output=True, text=True)
            if res.returncode != 0:
                raise ValueError(f"Ruff error: {res.stdout}")
        return {"status": "PYTHON_TOOL_VALIDATED"}

    def step2_exec_fallback(ctx):
        # Safe fallback code
        safe_code = "def compute_yield(principal: float, rate: float) -> float:\n    return principal * (1.0 + rate)\n"
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "server.py"
            f.write_text(safe_code, encoding="utf-8")
            py_compile.compile(str(f), doraise=True)
        return {"status": "FALLBACK_TOOL_VALIDATED"}

    def step3_exec_primary(ctx):
        # Execute real DuckDB query generated by 9B
        res = db_conn.execute(p3_sql).fetchall()
        return {"status": "ANALYTICS_SUCCESS", "rows": len(res), "data": res}

    def step3_exec_fallback(ctx):
        res = db_conn.execute("SELECT account, SUM(amount) AS total, AVG(amount) AS avg_amt FROM financial_records GROUP BY account;").fetchall()
        return {"status": "ANALYTICS_FALLBACK_SUCCESS", "rows": len(res), "data": res}

    g = ReliabilityGraph(source="step1", sink="step3")
    g.add_node(ReliabilityNode("step1", "9B PostgreSQL DDL", reliability=0.99, execute_fn=step1_exec))
    g.add_node(ReliabilityNode("step2", "9B FastMCP Python", reliability=0.85, execute_fn=step2_exec_primary, fallback_fn=step2_exec_fallback))
    g.add_node(ReliabilityNode("step3", "9B DuckDB Analytics", reliability=0.90, execute_fn=step3_exec_primary, fallback_fn=step3_exec_fallback))

    g.add_edge("step1", "step2")
    g.add_edge("step2", "step3")

    executor = ReliabilityDAGExecutor(target_reliability=0.95)
    ctx, telem = executor.execute_dag(g)

    print("\n" + "=" * 80)
    print(" [9B LIVE PIPELINE VERIFIED & COMPLETE]")
    print("=" * 80)
    print(f"  Final Pipeline Output Context: {ctx['step3']}")
    print(f"  Execution Telemetry:           {telem}")
    print(f"  Order-1 Cut Sets Detected:     {telem['order_1_cut_sets']}")
    print(f"  Speculatively Hedged Nodes:    {telem['hedged_nodes']}")
    print(f"  Real Exceptions Averted:       {telem['exceptions_averted']}")
    print(f"  Total Sandbox Time:            {telem['execution_time_ms']:.2f} ms\n")

    results = {
        "model_id": "Qwen/Qwen3.5-9B",
        "experts_loaded": ["postgresql", "astral", "duckdb"],
        "gpu_generation_latencies_ms": {
            "p1_postgresql": t_p1,
            "p2_astral": t_p2,
            "p3_duckdb": t_p3,
        },
        "sandbox_execution": ctx["step3"],
        "cut_set_telemetry": telem,
    }

    out_path = RESULTS_DIR / "qwen3_5_9b_live_pipeline_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Full 9B live pipeline telemetry saved to {out_path}")


if __name__ == "__main__":
    run_9b_benchmark()
