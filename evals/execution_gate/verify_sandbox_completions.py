"""Sandbox Execution Verifier for Applied Toolchain Benchmarks.

Runs in a clean standalone process (without PyTorch C++ bindings) to execute SQL
against py-pglite and lint Python against ruff.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from runtime.canon import REPO_ROOT


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
    """Executes SQL migration and queries against persistent py-pglite in an isolated schema."""
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


def main():
    if len(sys.argv) < 3:
        print("Usage: python verify_sandbox_completions.py <raw_json> <out_json>")
        sys.exit(1)

    raw_file = Path(sys.argv[1])
    out_file = Path(sys.argv[2])

    with open(raw_file) as f:
        raw_data = json.load(f)

    print("\n" + "=" * 90)
    print(" DETERMINISTIC APPLIED EXECUTION IN SANDBOX (PGlite & Ruff)")
    print("=" * 90)

    from py_pglite import PGliteConfig, PGliteManager
    import psycopg

    pg_config = PGliteConfig(extensions=["pgvector"])
    results = []

    with PGliteManager(config=pg_config) as pg_mgr:
        dsn = pg_mgr.get_dsn()
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print(f"[Sandbox] PGlite server operational at {dsn}\n")

        for prob in raw_data["problems"]:
            p_id = prob["id"]
            p_title = prob["title"]
            eval_type = prob["eval_type"]
            test_queries = prob.get("test_queries", [])

            def eval_text(text: str) -> dict[str, Any]:
                if eval_type == "sql":
                    code = extract_code_block(text, "sql")
                    res = run_pglite_sql_test(dsn, code, test_queries)
                    score = 1.0 if res["success"] and res["results_count"] > 0 else (0.5 if res["success"] else 0.0)
                    return {
                        "score": score,
                        "success": res["success"],
                        "results_count": res.get("results_count", 0),
                        "error": res.get("error"),
                        "extracted_code": code[:120] + "...",
                    }
                elif eval_type == "python":
                    code = extract_code_block(text, "python")
                    res = run_ruff_linter_test(code)
                    score = 0.0
                    if res["syntax_valid"]:
                        score += 0.5
                        if res["ruff_clean"]:
                            score += 0.5
                        else:
                            score += max(0.0, 0.5 - (res["ruff_violations"] * 0.05))
                    return {
                        "score": round(score, 3),
                        "syntax_valid": res["syntax_valid"],
                        "ruff_violations": res["ruff_violations"],
                        "extracted_code": code[:120] + "...",
                    }
                return {"score": 0.0}

            eval_a = eval_text(prob["text_a"])
            eval_b = eval_text(prob["text_b"])
            eval_c = eval_text(prob["text_c"])

            print(f"--- Results for Problem: {p_id} ({p_title}) ---")
            print(f"  Arm A (Base 9B Unassisted): Score = {eval_a['score']:.2f} | Error: {eval_a.get('error')}")
            print(f"  Arm B (9B + Folded Expert):  Score = {eval_b['score']:.2f} [Delta: {eval_b['score'] - eval_a['score']:+.2f}]")
            print(f"  Arm C (Base 9B Prompted):   Score = {eval_c['score']:.2f}")
            print("-" * 90)

            results.append({
                "problem_id": p_id,
                "title": p_title,
                "arm_a_base": eval_a,
                "arm_b_folded_expert": eval_b,
                "arm_c_prompted": eval_c,
                "expert_edge": round(eval_b["score"] - eval_a["score"], 3),
            })

    report = {
        "model_id": raw_data.get("model_id", "Qwen/Qwen3.5-9B"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "problems": results,
    }

    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Persisted] Auditable execution report written to {out_file}\n")


if __name__ == "__main__":
    main()
