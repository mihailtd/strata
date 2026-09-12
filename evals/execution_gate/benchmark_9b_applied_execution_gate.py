"""Deterministic Applied Toolchain & Execution Benchmark for Qwen3.5-9B.

Evaluates whether Domain Expert LoRA weight folding makes Qwen3.5-9B measurably
smarter on real-world executable tasks:
  - Arm A: Base Qwen3.5-9B (Zero-Prompt / Parametric Knowledge)
  - Arm B: Qwen3.5-9B + Folded Domain Expert (In-Place BLAS Tensor Fusion)
  - Arm C: Base Qwen3.5-9B (Prompted Generalist Baseline)

Evaluated against live execution environments:
  - Database: py-pglite (in-process PostgreSQL with pgvector extension & HNSW)
  - Tooling: ruff check (PEP 723 metadata, strict type annotations, asyncpg)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Force GPU 0 exclusive device isolation
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HIP_VISIBLE_DEVICES"] = "0"
os.environ["ROCR_VISIBLE_DEVICES"] = "0"

import torch

from runtime.canon import REPO_ROOT
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from runtime.canon import (
    CANON,
    configure_deterministic_attention,
    validate_kv_cache_precision,
)
from runtime.novel_peft import FoldableExpert, WeightFoldingEngine, set_hard_vram_cap
from runtime.gpu_preflight import ensure_gpu_exclusive


# ---------------------------------------------------------------------------
# Sandbox Verifiers
# ---------------------------------------------------------------------------

def extract_code_block(text: str, language: str = "sql") -> str:
    """Extracts first matching code block from markdown."""
    pattern = rf"```{language}\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    generic_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()
    return text.strip()


def run_pglite_sql_test(dsn: str, sql_script: str, test_queries: list[str]) -> dict[str, Any]:
    """Executes SQL migration and queries against persistent py-pglite."""
    import psycopg
    import uuid

    schema_name = f"test_{uuid.uuid4().hex[:8]}"
    t0 = time.perf_counter()
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(f"CREATE SCHEMA {schema_name};")
            conn.execute(f"SET search_path TO {schema_name}, public;")
            clean_lines = [line for line in sql_script.splitlines() if not line.strip().startswith("--")]
            clean_script = "\n".join(clean_lines)
            statements = [s.strip() for s in clean_script.split(";") if s.strip()]
            for stmt in statements:
                conn.execute(stmt)
            
            results = []
            for query in test_queries:
                cur = conn.execute(query)
                rows = cur.fetchall() if cur.description else []
                results.append(rows)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": True,
            "error": None,
            "elapsed_ms": round(elapsed_ms, 2),
            "results_count": sum(len(r) for r in results),
        }
    except Exception as ex:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "success": False,
            "error": f"{type(ex).__name__}: {str(ex)[:200]}",
            "elapsed_ms": round(elapsed_ms, 2),
            "results_count": 0,
        }


def run_ruff_linter_test(python_code: str) -> dict[str, Any]:
    """Runs ruff check and syntax parsing on Python code."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(python_code)
        tmp_path = f.name

    try:
        proc_check = subprocess.run(
            ["ruff", "check", "--output-format", "json", tmp_path],
            capture_output=True,
            text=True,
        )
        violations = 0
        if proc_check.stdout.strip():
            try:
                data = json.loads(proc_check.stdout)
                violations = len(data)
            except Exception:
                violations = 1

        # Check Python compilation syntax
        proc_compile = subprocess.run(
            [sys.executable, "-m", "py_compile", tmp_path],
            capture_output=True,
            text=True,
        )
        syntax_valid = (proc_compile.returncode == 0)

        return {
            "syntax_valid": syntax_valid,
            "ruff_violations": violations,
            "ruff_clean": violations == 0,
            "syntax_error": proc_compile.stderr.strip() if not syntax_valid else None,
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Problem Benchmark Definitions
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    id: str
    title: str
    target_expert: str
    prompt_unassisted: str
    prompt_prompted: str
    eval_type: str  # "sql" | "python"
    test_queries: list[str] = field(default_factory=list)


BENCHMARK_PROBLEMS = [
    Problem(
        id="p1_pgvector_hnsw_hybrid",
        title="PostgreSQL 17 pgvector HNSW Migration",
        target_expert="postgresql",
        prompt_unassisted=(
            "Write a production PostgreSQL DDL migration script. Create an 'articles' table with an id (UUID primary key), "
            "title (TEXT), content (TEXT), and a 1536-dimensional embedding vector. Create an HNSW index on the embedding column "
            "optimized for cosine distance with m=16, ef_construction=64. Insert one sample row. Output only SQL."
        ),
        prompt_prompted=(
            "You are a Senior PostgreSQL DBA. Create an 'articles' table with extension vector, id UUID PRIMARY KEY DEFAULT gen_random_uuid(), "
            "title TEXT NOT NULL, content TEXT NOT NULL, embedding vector(1536). Add an HNSW index using vector_cosine_ops with m=16 and ef_construction=64. "
            "Insert one sample row with a 1536-dim vector. Output only SQL in a ```sql block."
        ),
        eval_type="sql",
        test_queries=[
            "SELECT count(*) FROM articles;",
            "SELECT id, title FROM articles ORDER BY embedding <=> (SELECT embedding FROM articles LIMIT 1) LIMIT 1;",
        ],
    ),
    Problem(
        id="p2_fastmcp_asyncpg_service",
        title="Modern Python FastMCP Tool Server with asyncpg",
        target_expert="astral",
        prompt_unassisted=(
            "Write a production Python tool server script using FastMCP and asyncpg. It should expose a tool 'search_db' "
            "accepting query: str and limit: int = 5. Include PEP 723 inline script metadata for uv dependencies (fastmcp, asyncpg). "
            "Ensure modern type annotations and clean formatting. Output only the Python block."
        ),
        prompt_prompted=(
            "Write a single-file Python script using FastMCP and asyncpg. Use PEP 723 inline script metadata at the top: "
            "# /// script\n# dependencies = [\"fastmcp\", \"asyncpg\"]\n# ///\n"
            "Expose @mcp.tool() search_db(query: str, limit: int = 5) -> list[dict[str, Any]]. Output only Python in a ```python block."
        ),
        eval_type="python",
    ),
]


# ---------------------------------------------------------------------------
# Benchmark Engine
# ---------------------------------------------------------------------------

class AppliedExecutionGate9B:
    def __init__(self, model_id: str = "Qwen/Qwen3.5-9B", device: str = "cuda:0"):
        print("=" * 90)
        print(f" Applied Execution Gate Benchmark: {model_id}")
        print("=" * 90)

        ensure_gpu_exclusive()

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[Engine] Loading base model ({model_id}) in bfloat16...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map={"": self.device.index if self.device.type == "cuda" else "cpu"},
            trust_remote_code=True,
        )
        self.model.eval()

        # Load 9B experts
        experts = []
        for d, folder in [("postgresql", "m2_postgresql_r8a128_v7_9b"), ("astral", "m2_astral_r8a128_v7_9b")]:
            adapter_dir = REPO_ROOT / "results" / "adapters" / folder
            if adapter_dir.exists():
                print(f"[Engine] Registering 9B expert [{d}] from {folder}...")
                experts.append(FoldableExpert.from_dir(adapter_dir, d))

        self.folding_engine = WeightFoldingEngine(self.model, experts, keep_pristine=True)
        self.expert_map = {e.name: e for e in experts}

    def generate(self, prompt: str, max_new_tokens: int = 512) -> tuple[str, float]:
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        chat_prompt += "\n</think>\n"
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        gen_ms = (time.perf_counter() - t0) * 1000.0

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return text, gen_ms

    def generate_all_arms(self) -> dict[str, Any]:
        print("\n" + "=" * 90)
        print(" PHASE 1: GENERATING RESPONSES ACROSS ARMS A, B, C ON 9B GPU")
        print("=" * 90)

        raw_records = []
        for prob in BENCHMARK_PROBLEMS:
            print(f"\n--- Generating for Problem: {prob.id} ({prob.title}) ---")
            
            # Arm A: Base 9B Unassisted
            self.folding_engine.restore_pristine()
            text_a, ms_a = self.generate(prob.prompt_unassisted)
            print(f"  Arm A (Base 9B Unassisted): {ms_a:.1f} ms | Output length: {len(text_a)} chars")

            # Arm B: Base 9B + Folded Domain Expert
            expert = self.expert_map.get(prob.target_expert)
            if expert:
                self.folding_engine.activate(expert)
                text_b, ms_b = self.generate(prob.prompt_unassisted)
                self.folding_engine.restore_pristine()
            else:
                text_b, ms_b = text_a, ms_a
            print(f"  Arm B (9B + Folded Expert):  {ms_b:.1f} ms | Output length: {len(text_b)} chars")

            # Arm C: Base 9B Prompted
            self.folding_engine.restore_pristine()
            text_c, ms_c = self.generate(prob.prompt_prompted)
            print(f"  Arm C (Base 9B Prompted):   {ms_c:.1f} ms | Output length: {len(text_c)} chars")

            raw_records.append({
                "id": prob.id,
                "title": prob.title,
                "eval_type": prob.eval_type,
                "test_queries": prob.test_queries,
                "text_a": text_a, "ms_a": ms_a,
                "text_b": text_b, "ms_b": ms_b,
                "text_c": text_c, "ms_c": ms_c,
            })

        return {
            "model_id": "Qwen/Qwen3.5-9B",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "problems": raw_records,
        }


def main():
    # 1. GPU Generation Phase
    engine = AppliedExecutionGate9B()
    raw_data = engine.generate_all_arms()

    # Free GPU memory before Sandbox phase
    del engine.model
    del engine.folding_engine
    del engine
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    raw_file = REPO_ROOT / "results" / "benchmarks" / "raw_9b_completions.json"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_file, "w") as f:
        json.dump(raw_data, f, indent=2)

    # 2. Run Clean Subprocess Sandbox Verification
    out_file = REPO_ROOT / "results" / "benchmarks" / "benchmark_9b_applied_execution_gate.json"
    verifier_script = REPO_ROOT / "benchmarks" / "verify_sandbox_completions.py"
    cmd = [sys.executable, str(verifier_script), str(raw_file), str(out_file)]
    subprocess.run(cmd, check=True)

    print("\n" + "=" * 90)
    print(f" [9B APPLIED BENCHMARK COMPLETE] Audited results in {out_file}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
