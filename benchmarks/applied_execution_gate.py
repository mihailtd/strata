"""Deterministic Applied Toolchain & Execution Benchmark.

Compares Base 4B Generalist vs. Specialized Multi-Expert Swarm across:
  1. Regime A: Zero-Prompt Unassisted (Default Modernity & Parametric Instincts)
  2. Regime B: Explicitly Prompted (Prompting Tax, Syntax Drift & Verification)

Evaluation Engines:
  - Database Runtime: py-pglite[extensions] (in-process PostgreSQL with real pgvector execution)
  - Code & Linter: ruff check / ruff format / uv run (deterministic static analysis)

Usage:
  uv run --env-file .env python benchmarks/applied_execution_gate.py [--smoke] [--regime {unassisted,prompted,both}]
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

import torch

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from runtime.canon import CANON, adapter_path
from runtime.novel_peft import FoldableExpert, WeightFoldingEngine, set_hard_vram_cap
from runtime.fused_norm import inject_exact_rmsnorm, fold_rmsnorm_into_linear
from runtime.gpu_preflight import ensure_gpu_exclusive

# ---------------------------------------------------------------------------
# Evaluation Sandbox Tools
# ---------------------------------------------------------------------------

def extract_code_block(text: str, language: str = "sql") -> str:
    """Extracts the first matching code block from markdown text."""
    pattern = rf"```{language}\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    generic_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()
    return text.strip()


def run_pglite_sql_test(dsn: str, sql_script: str, test_queries: list[str]) -> dict[str, Any]:
    """Executes SQL migration and queries against persistent py-pglite in an isolated schema."""
    import psycopg, uuid

    schema_name = f"test_{uuid.uuid4().hex[:8]}"
    t0 = time.perf_counter()
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(f"CREATE SCHEMA {schema_name};")
            conn.execute(f"SET search_path TO {schema_name}, public;")
            # Remove comment-only lines before splitting statements by semicolon
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
    """Runs ruff check and format against the generated Python code."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(python_code)
        tmp_path = f.name

    try:
        proc_check = subprocess.run(
            ["ruff", "check", "--select", "ALL", "--output-format", "json", tmp_path],
            capture_output=True,
            text=True,
        )
        violations = 0
        violation_types = []
        if proc_check.stdout.strip():
            try:
                data = json.loads(proc_check.stdout)
                violations = len(data)
                violation_types = [d.get("code", "") for d in data[:5]]
            except Exception:
                violations = len(proc_check.stdout.splitlines())

        proc_fmt = subprocess.run(
            ["ruff", "format", "--check", tmp_path],
            capture_output=True,
            text=True,
        )
        fmt_clean = proc_fmt.returncode == 0

        proc_compile = subprocess.run(
            [sys.executable, "-m", "py_compile", tmp_path],
            capture_output=True,
            text=True,
        )
        compiles = proc_compile.returncode == 0

        return {
            "compiles": compiles,
            "violations": violations,
            "violation_samples": violation_types,
            "formatted": fmt_clean,
            "linter_score": round(1.0 / (1.0 + 0.1 * violations), 3) if compiles else 0.0,
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Benchmark Problem Definitions
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    id: str
    title: str
    target_expert: str
    prompt_unassisted: str
    prompt_prompted: str
    eval_type: str  # 'sql', 'python', 'multi_turn'
    test_queries: list[str] = field(default_factory=list)
    seed_sql: str = ""


BENCHMARK_PROBLEMS = [
    Problem(
        id="p1_hnsw_hybrid_migration",
        title="Multi-Tenant HNSW Vector Index Migration",
        target_expert="postgresql",
        prompt_unassisted=(
            "We are building a multi-tenant knowledge base. Write a complete PostgreSQL migration statement "
            "that creates a table 'document_chunks' with tenant_id (uuid), chunk_id (bigserial pk), "
            "content (text), and a 1536-dimensional vector embedding. Build an approximate-nearest-neighbor "
            "index on the embedding column optimized for cosine similarity. Output only the SQL block."
        ),
        prompt_prompted=(
            "Using PostgreSQL 18 and pgvector, write a migration statement that creates table 'document_chunks' "
            "(tenant_id uuid, chunk_id bigserial primary key, content text, embedding vector(1536)). "
            "Create an HNSW index on embedding using vector_cosine_ops with m=16 and ef_construction=64. "
            "Output only the SQL in a ```sql block."
        ),
        eval_type="sql",
        test_queries=[
            "INSERT INTO document_chunks (tenant_id, content, embedding) VALUES ('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'Test chunk', array_fill(0.01::real, ARRAY[1536])::vector);",
            "SELECT chunk_id, content, embedding <=> array_fill(0.01::real, ARRAY[1536])::vector AS dist FROM document_chunks ORDER BY dist LIMIT 1;",
        ],
    ),
    Problem(
        id="p2_rrf_hybrid_cte_search",
        title="Hybrid Reciprocal Rank Fusion Search Query",
        target_expert="postgresql",
        seed_sql=(
            "CREATE TABLE IF NOT EXISTS articles (id serial primary key, title text, content text, embedding vector(3));\n"
            "INSERT INTO articles (title, content, embedding) VALUES "
            "('Postgres AI Guide', 'Deep dive into vectors and AI indexing', '[0.1, 0.2, 0.3]'), "
            "('Python Microservices', 'FastAPI and modern async tooling', '[0.9, 0.1, 0.0]'), "
            "('Database Tuning', 'Optimizer plans and pgvector indexing', '[0.1, 0.2, 0.4]');"
        ),
        prompt_unassisted=(
            "Given the articles table (id serial, title text, content text, embedding vector(3)), write an "
            "optimized hybrid search SQL query that finds the most relevant articles for a query vector '[0.1, 0.2, 0.3]' "
            "and keyword 'indexing'. Combine vector distance and title matching. Output only the SQL query."
        ),
        prompt_prompted=(
            "Given articles (id, title, content, embedding vector(3)), write a PostgreSQL query combining "
            "pgvector cosine distance (embedding <=> '[0.1, 0.2, 0.3]') and keyword search (title ILIKE '%indexing%') "
            "using CTEs. Return id, title, and combined ranking score ordered by relevance. Output only SQL."
        ),
        eval_type="sql",
        test_queries=[
            # Query itself is executed against seed table
        ],
    ),
    Problem(
        id="p3_fastmcp_asyncpg_service",
        title="FastMCP AsyncPG Document Search Server",
        target_expert="astral",
        prompt_unassisted=(
            "Write a production-ready Python tool server that exposes a tool 'search_knowledge_base' accepting "
            "query_text (str) and limit (int = 5). It connects to PostgreSQL using connection pooling to query "
            "the articles table. Ensure modern dependency management, strict type annotations, and clean formatting. "
            "Output only the Python script in a ```python block."
        ),
        prompt_prompted=(
            "Write a single-file Python service using FastMCP and asyncpg. Expose an @mcp.tool() 'search_knowledge_base' "
            "(query_text: str, limit: int = 5) -> list[dict[str, Any]]. Use PEP 723 inline script metadata for uv dependencies "
            "(fastmcp, asyncpg, pydantic>=2.0). Ensure strict type annotations and ruff compliance. Output only the Python block."
        ),
        eval_type="python",
    ),
]


# ---------------------------------------------------------------------------
# Benchmark Engine & Arm Execution
# ---------------------------------------------------------------------------

class AppliedExecutionGate:
    def __init__(self, model_id: str = "Qwen/Qwen3.5-4B", device: str = "cuda:0"):
        print("==================================================")
        print(" Applied Toolchain & Execution Benchmark Engine")
        print("==================================================")
        # PRE-FLIGHT EXCLUSIVITY GUARD: Fail fast if another job is holding VRAM
        ensure_gpu_exclusive()

        set_hard_vram_cap(22.0)
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

        # In-place optimizations: FlashNorm folding
        inject_exact_rmsnorm(self.model)
        fold_rmsnorm_into_linear(self.model, fold_weights=True)

        # Load available experts
        adapters_dir = REPO_ROOT / "results" / "adapters"
        experts = []
        
        # PostgreSQL expert (priority to v4 completion-only loss adapter)
        pg_dirs = [adapter_path("postgresql")]   # canon: raises, never falls back
        for pg_dir in pg_dirs:
            if pg_dir.exists():
                print(f"[Engine] Registering PostgreSQL expert from {pg_dir.name}...")
                experts.append(FoldableExpert.from_dir(pg_dir, "postgresql"))
                break

        # Astral expert (priority to v4 completion-only loss adapter)
        astral_dirs = [adapter_path("astral")]   # canon: raises, never falls back
        for ast_dir in astral_dirs:
            if ast_dir.exists():
                print(f"[Engine] Registering Astral expert from {ast_dir.name}...")
                experts.append(FoldableExpert.from_dir(ast_dir, "astral"))
                break

        self.folding_engine = WeightFoldingEngine(self.model, experts, keep_pristine=True)
        self.expert_map = {e.name: e for e in experts}
        print(f"[Engine] Initialized WeightFoldingEngine with {len(experts)} domain experts.")

        # Initialize persistent PGlite server for 3ms per-test validation
        try:
            from py_pglite import PGliteConfig, PGliteManager
            import psycopg
            self.pglite_config = PGliteConfig(extensions=["pgvector"])
            self.pglite_mgr = PGliteManager(config=self.pglite_config)
            self.pglite_mgr.__enter__()
            self.pglite_dsn = self.pglite_mgr.get_dsn()
            with psycopg.connect(self.pglite_dsn, autocommit=True) as conn:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("[Engine] Initialized persistent PGlite server with pgvector enabled.\n")
        except Exception as ex:
            print(f"[Engine] Warning: Failed to start persistent PGlite: {ex}\n")
            self.pglite_mgr = None
            self.pglite_dsn = None

    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        expert_name: str | None = None,
        max_new_tokens: int = CANON.MAX_NEW_TOKENS,
    ) -> tuple[str, float, float]:
        """Generates response with in-place expert folding and non-thinking stop tokens."""
        if expert_name and expert_name in self.expert_map:
            self.folding_engine.activate(self.expert_map[expert_name])
        else:
            self.folding_engine.restore()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a principal software architect. Provide direct, production-grade, executable "
                    "code blocks with zero conversational filler or unclosed thinking blocks."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Suppress long deliberation loops by initiating direct answer
        prompt_text += "<think>\n</think>\n"

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        prompt_num_toks = inputs.input_ids.shape[1]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                self.tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            ],
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t0

        gen_tokens = outputs[0, prompt_num_toks:].tolist()
        gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        tok_s = len(gen_tokens) / max(1e-5, elapsed_s)

        return gen_text, elapsed_s, tok_s

    def evaluate_problem(
        self,
        problem: Problem,
        regime: str,  # 'unassisted' or 'prompted'
    ) -> dict[str, Any]:
        """Runs problem across Arm A (Base), Arm B (Oracle), and Arm C (Autonomous)."""
        prompt = problem.prompt_unassisted if regime == "unassisted" else problem.prompt_prompted
        print(f"\n--- Problem: {problem.title} [{regime.upper()}] ---")

        results = {}
        arms = [
            ("Arm A (Base 4B)", None),
            ("Arm B (Oracle Expert)", problem.target_expert),
            ("Arm C (Autonomous Router)", self._route_expert(prompt, problem.target_expert)),
        ]

        for arm_name, expert in arms:
            text, elapsed, tok_s = self.generate_text(prompt, expert_name=expert)
            
            exec_score = 0.0
            linter_score = 1.0
            modernity_score = 0.0
            exec_detail = ""

            if problem.eval_type == "sql":
                sql_block = extract_code_block(text, "sql")
                
                # Check modernity indicators (strictly based on actual modern vector practices)
                has_pgvector = bool(re.search(r"\b(vector|hnsw|ivfflat|<=>|<->|<#>)\b", sql_block, re.I))
                has_opclass = bool(re.search(r"\b(vector_cosine_ops|vector_l2_ops|vector_ip_ops)\b", sql_block, re.I))
                modernity_score = (0.5 if has_pgvector else 0.0) + (0.5 if has_opclass else 0.0)

                # Live py-pglite database execution (substitute $1 if raw parameter was used)
                test_sql = (problem.seed_sql + "\n" + sql_block).strip()
                # If query contains $1 parameter placeholder, bind it for testing
                test_sql_bound = re.sub(r"\$1\b", "'[0.1, 0.2, 0.3]'::vector", test_sql)
                res = run_pglite_sql_test(self.pglite_dsn, test_sql_bound, problem.test_queries)
                if res["success"]:
                    exec_score = 1.0
                    exec_detail = f"Executed clean ({res['elapsed_ms']}ms)"
                else:
                    exec_score = 0.0
                    exec_detail = f"Failed: {res['error']}"

                # ZERO-FLOOR: SQL score is 70% execution + 30% vector modernity (no free linter points)
                linter_score = 0.0
                composite = round(0.70 * exec_score + 0.30 * modernity_score, 3)

            elif problem.eval_type == "python":
                py_block = extract_code_block(text, "python")
                
                # Check modernity indicators
                has_uv = bool(re.search(r"(#\s*///\s*script|dependencies\s*=|uv\b)", py_block, re.I))
                has_fastmcp = bool(re.search(r"\b(fastmcp|mcp\.tool|FastMCP)\b", py_block, re.I))
                has_typing = bool(re.search(r"\blist\[|\bdict\[|\|\s*None\b", py_block)) and not bool(re.search(r"\btyping\.List\b", py_block))
                modernity_score = (0.4 if has_uv else 0.0) + (0.3 if has_fastmcp else 0.0) + (0.3 if has_typing else 0.0)

                # Ruff & Python Linter execution
                res = run_ruff_linter_test(py_block)
                exec_score = 1.0 if res["compiles"] else 0.0
                linter_score = res["linter_score"]
                exec_detail = f"Compiles={res['compiles']}, Ruff violations={res['violations']}"

                # ZERO-FLOOR: Python score is 50% compilation + 30% linter + 20% modernity
                composite = round(0.50 * exec_score + 0.30 * linter_score + 0.20 * modernity_score, 3)

            results[arm_name] = {
                "expert": expert or "base",
                "composite_score": composite,
                "exec_score": exec_score,
                "linter_score": linter_score,
                "modernity_score": modernity_score,
                "exec_detail": exec_detail,
                "elapsed_s": round(elapsed, 2),
                "tok_s": round(tok_s, 1),
                "chars": len(text),
                "generated_text": text,
            }
            print(f"  {arm_name:28s} | Score: {composite:.3f} (Exec: {exec_score}, Linter: {linter_score:.2f}, Mod: {modernity_score:.2f}) | {exec_detail}")

        return results

    def _route_expert(self, prompt: str, oracle_target: str) -> str:
        """Deterministic intent router for Arm C. Checks Python/Astral specific terms first."""
        prompt_lower = prompt.lower()
        # Python / Tooling domain checks first
        if any(w in prompt_lower for w in ["fastmcp", "mcp.tool", "asyncpg", "fastapi", "python", "script", "pydantic", "ruff", "tool server"]):
            return "astral"
        # Database / SQL domain checks second
        if any(w in prompt_lower for w in ["create table", "pgvector", "hnsw", "vector_cosine", "migration", "cte", "select", "sql"]):
            return "postgresql"
        return oracle_target


def bootstrap_ci(diffs: list[float], n_resamples: int = 10000) -> tuple[float, float, float]:
    """Computes paired bootstrap mean and 95% confidence interval."""
    import numpy as np
    if not diffs:
        return 0.0, 0.0, 0.0
    arr = np.array(diffs)
    mean = float(np.mean(arr))
    if len(arr) == 1:
        return mean, mean, mean
    boot_means = [np.mean(np.random.choice(arr, size=len(arr), replace=True)) for _ in range(n_resamples)]
    low = float(np.percentile(boot_means, 2.5))
    high = float(np.percentile(boot_means, 97.5))
    return mean, low, high


def run_applied_execution_gate(regime: str = "both", smoke: bool = False):
    gate = AppliedExecutionGate()
    problems = BENCHMARK_PROBLEMS[:1] if smoke else BENCHMARK_PROBLEMS

    regimes_to_run = ["unassisted", "prompted"] if regime == "both" else [regime]
    all_summary = {}

    for r in regimes_to_run:
        print(f"\n==================================================")
        print(f" REGIME: {r.upper()} EVALUATION")
        print(f"==================================================")
        regime_results = []
        for p in problems:
            res = gate.evaluate_problem(p, regime=r)
            regime_results.append(res)
        all_summary[r] = regime_results

    # Print Final Summary Table
    print("\n" + "=" * 95)
    print(" DETERMINISTIC APPLIED TOOLCHAIN BENCHMARK SUMMARY (ZERO-FLOOR SCORING)")
    print("=" * 95)
    
    table_data = {}
    for r, res_list in all_summary.items():
        print(f"\n--- REGIME: {r.upper()} ---")
        print(f"{'Problem ID':28s} | {'Arm A (Base)':15s} | {'Arm B (Oracle)':15s} | {'Arm C (Auto)':15s} | {'Edge (B-A)':10s} | {'Edge (C-A)':10s}")
        print("-" * 105)
        
        base_scores, oracle_scores, auto_scores = [], [], []
        diffs_ba, diffs_ca = [], []
        for idx, res in enumerate(res_list):
            p_id = problems[idx].id
            s_a = res["Arm A (Base 4B)"]["composite_score"]
            s_b = res["Arm B (Oracle Expert)"]["composite_score"]
            s_c = res["Arm C (Autonomous Router)"]["composite_score"]
            edge_ba = s_b - s_a
            edge_ca = s_c - s_a
            
            base_scores.append(s_a)
            oracle_scores.append(s_b)
            auto_scores.append(s_c)
            diffs_ba.append(edge_ba)
            diffs_ca.append(edge_ca)
            
            print(f"{p_id:28s} | {s_a:15.3f} | {s_b:15.3f} | {s_c:15.3f} | {edge_ba:+10.3f} | {edge_ca:+10.3f}")

        mean_a = sum(base_scores) / len(base_scores)
        mean_b = sum(oracle_scores) / len(oracle_scores)
        mean_c = sum(auto_scores) / len(auto_scores)
        
        mean_ba, low_ba, high_ba = bootstrap_ci(diffs_ba)
        mean_ca, low_ca, high_ca = bootstrap_ci(diffs_ca)
        
        print("-" * 105)
        print(f"{'MEAN COMPOSITE SCORE':28s} | {mean_a:15.3f} | {mean_b:15.3f} | {mean_c:15.3f} | {mean_ba:+10.3f} | {mean_ca:+10.3f}")
        print(f"95% CI (B-A): [{low_ba:+.3f}, {high_ba:+.3f}] | 95% CI (C-A): [{low_ca:+.3f}, {high_ca:+.3f}]")
        print("=" * 105)

        table_data[r] = {
            "mean_base": mean_a,
            "mean_oracle": mean_b,
            "mean_auto": mean_c,
            "edge_oracle_vs_base": mean_ba,
            "ci_oracle_vs_base": [low_ba, high_ba],
            "edge_auto_vs_base": mean_ca,
            "ci_auto_vs_base": [low_ca, high_ca],
            "problems": [
                {
                    "problem_id": problems[i].id,
                    "scores": {k: {sk: sv for sk, sv in v.items() if sk != "generated_text"} for k, v in res_list[i].items()}
                }
                for i in range(len(problems))
            ]
        }

    # Persist JSON report
    out_dir = REPO_ROOT / "results" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "applied_execution_gate_results.json"
    with open(out_file, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "regimes": table_data,
        }, f, indent=2)
    print(f"\n[Persisted] Full auditable report written to {out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic Applied Execution Benchmark")
    parser.add_argument("--smoke", action="store_true", help="Run 1 problem smoke test")
    parser.add_argument("--regime", choices=["unassisted", "prompted", "both"], default="both")
    args = parser.parse_args()

    run_applied_execution_gate(regime=args.regime, smoke=args.smoke)
