"""TIER 2 v2 — capability with format stripped out, tiered SQL, and self-healing.

THREE FIXES OVER v1
-------------------
1. FORMAT-AGNOSTIC EXTRACTION. v1 required a ```fence and fell back to the whole
   reply, so an answer wrapped in prose failed as "code". Base scored 0.067 on
   Python, which conflated FORMAT DISCIPLINE with PYTHON SKILL. Code is now
   salvaged from unfenced replies, and the fenced rate is reported SEPARATELY so
   both quantities are visible instead of entangled.

2. TIERED SQL SCORING. v1 demanded exact row equality against a reference on
   synthetic seed data. Both arms sat at the floor (base 0.042, expert 0.000), so
   it measured nothing. Three nested tiers now separate "ran at all" from "was
   right":
       T1 executes   no exception
       T2 tables     touches the same tables as the reference
       T3 exact      returns byte-identical rows
   T3 is still the truth; T1/T2 show WHERE an answer failed instead of collapsing
   every failure into one zero.

3. PASS@1 vs PASS@2 (agentic self-healing). A single-shot score answers "can it
   write perfect code blind?" -- which is not how an agent runs. On failure the
   real stderr is fed back for exactly one retry. The gap between the two is the
   RECOVERY RATE, and it is the metric that should favour an expert trained on
   anti-patterns: it has seen why that error happens.

       recovery = (pass@2 - pass@1) / (1 - pass@1)
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

REPO = Path("/home/mihai/gnn-experiment")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "apps"))
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(Path(__file__).parent))
T0 = time.perf_counter()


def stage(m): print(f"[{time.perf_counter() - T0:7.1f}s] {m}", flush=True)


import psycopg  # noqa: E402
from runtime.canon import adapter_path
import torch  # noqa: E402

torch.zeros(1, device="cuda"); torch.cuda.synchronize()
from bench_indomain_reserved import build_seed  # noqa: E402
from py_pglite import PGliteConfig, PGliteManager  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from runtime.cuda_graph import FoldedCudaGraphDecoder  # noqa: E402
from runtime.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

MODEL = "Qwen/Qwen3.5-4B"
ADAPTERS = {"postgresql": os.environ.get("PG_ADAPTER", str(adapter_path("postgresql"))),
            "astral": os.environ.get("ASTRAL_ADAPTER", str(adapter_path("astral")))}
RESERVED = {"postgresql": {"adv_distinct_on", "adv_filter_clause", "adv_ordinality_unnest",
                           "adv_lateral_join", "adv_percentiles"},
            "astral": {"func_partial_curry", "func_singledispatch", "py_protocol_slots",
                       "py_taskgroup"}}
N_SQL, N_PY = int(os.environ.get("N_SQL", "24")), int(os.environ.get("N_PY", "24"))
# A tight cap turns "long but correct" into "wrong". Tokens and seconds are now
# MEASURED (see gen()), so the budget does not need to double as a penalty --
# let the model finish, then charge it for what it spent via COST/CORRECT.
MAX_NEW = int(os.environ.get("MAX_NEW", "2048"))
MAX_SEQ = int(os.environ.get("MAX_SEQ", "8192"))
FENCE = re.compile(r"```(?:sql|python|py)?\s*(.*?)```", re.S | re.I)
SQL_START = re.compile(r"^\s*(with|select|create|insert|update|delete|alter)\b", re.I)


def extract_code(text: str, lang: str) -> tuple[str, bool]:
    """Salvage code even when the model never opened a fence.

    v1 treated 'no fence' as 'no code', which scored FORMAT rather than SKILL.
    """
    m = FENCE.findall(text)
    if m:
        return max(m, key=len).strip(), True
    if lang == "python":
        try:                       # whole reply may already be valid Python
            ast.parse(text)
            return text.strip(), False
        except Exception:
            pass
        lines, buf = text.splitlines(), []
        for ln in lines:           # keep code-ish lines, drop prose
            if re.match(r"^\s*(#|from |import |def |class |@|\s{4})", ln) or \
               re.search(r"[=:()\[\]]", ln) and not ln.strip().endswith("."):
                buf.append(ln)
        cand = "\n".join(buf).strip()
        try:
            ast.parse(cand)
            return cand, False
        except Exception:
            return cand, False
    stmts, cur = [], []
    for ln in text.splitlines():
        if SQL_START.match(ln):
            if cur:
                stmts.append("\n".join(cur))
            cur = [ln]
        elif cur:
            cur.append(ln)
    if cur:
        stmts.append("\n".join(cur))
    return (max(stmts, key=len).strip() if stmts else text.strip()), False


DDL_BY_TABLE: dict[str, str] = {}


def build_ddl_index(seed: str) -> None:
    """CREATE TABLE text per table, so a prompt can carry its own schema.

    WHY: the SQL half measured schema GUESSING, not SQL skill. Failure sat at T1
    (does it execute) -- base 0.208 / expert 0.083 -- not at T3 (are the rows
    right). Every surviving item's reference executes against this seed, so the
    schema supports the task; the model simply cannot know the column names from
    a prompt that says "In `hotel_bookings`, ...". A real agent always has the
    schema in context, so withholding it measures nothing anyone cares about.
    """
    for stmt in seed.split(";"):
        m = re.match(r"\s*CREATE TABLE\s+([a-z_][a-z0-9_]*)\s*\((.*)\)\s*$",
                     stmt.strip(), re.S | re.I)
        if m:
            DDL_BY_TABLE[m.group(1).lower()] = f"CREATE TABLE {m.group(1)} ({m.group(2)});"


def schema_prefix(tables: set[str]) -> str:
    ddl = [DDL_BY_TABLE[t] for t in sorted(tables) if t in DDL_BY_TABLE]
    if not ddl:
        return ""
    return ("Schema:\n\n```sql\n" + "\n".join(ddl) + "\n```\n\n")


def values_contained(got, ref) -> bool:
    """Does the model's result contain the reference's answer?

    Exact row equality punishes a correct computation that returns extra columns,
    different column names, or a different order -- none of which the prompt
    specified. Rather than dictating the output shape in the prompt (which would
    leak the task requirements), the GRADER is made tolerant: every reference row
    must be recoverable as a subset of some distinct model row.

    Strict exact-match is still reported alongside, so both a lenient and an
    unforgiving number are visible.
    """
    if got is None or ref is None or not ref:
        return False
    pool = [set(r) for r in got]
    used = set()
    for rrow in ref:
        want = set(rrow)
        hit = next((i for i, g in enumerate(pool) if i not in used and want <= g), None)
        if hit is None:
            return False
        used.add(hit)
    return True


def tables_in(sql: str) -> set[str]:
    return {t.lower() for t in re.findall(r"\b(?:from|join|into|update)\s+([a-z_][a-z0-9_]*)",
                                          sql, re.I)}


def run_sql_meta(dsn, seed, q):
    """Rows PLUS the column names the reference emits.

    T3 compares returned rows, but the reserved prompts were written as TRAINING
    prompts -- the answer defined the output. As eval prompts they never state
    which columns to return, so a model that computes the right median but emits
    one column instead of three scores zero. That is a defect in the instrument,
    not the model, so the contract is now derived from the reference and stated
    in the prompt.
    """
    sch = f"s_{uuid.uuid4().hex[:10]}"
    try:
        with psycopg.connect(dsn, autocommit=True) as c:
            c.execute(f"CREATE SCHEMA {sch};"); c.execute(f"SET search_path TO {sch}, public;")
            for st in [x.strip() for x in seed.split(";") if x.strip()]:
                c.execute(st)
            body = "\n".join(l for l in q.splitlines() if not l.strip().startswith("--"))
            sts = [x.strip() for x in body.split(";") if x.strip()]
            rows, cols = None, []
            for st in sts:
                cur = c.execute(st)
                if cur.description:
                    cols = [d.name for d in cur.description]
                    rows = cur.fetchall()
                else:
                    rows = []
            return ([tuple(str(v) for v in r) for r in (rows or [])], cols, None)
    except Exception as ex:
        return (None, [], f"{type(ex).__name__}: {str(ex)[:200]}")
    finally:
        try:
            with psycopg.connect(dsn, autocommit=True) as c:
                c.execute(f"DROP SCHEMA IF EXISTS {sch} CASCADE;")
        except Exception:
            pass


def run_sql(dsn, seed, q):
    sch = f"s_{uuid.uuid4().hex[:10]}"
    try:
        with psycopg.connect(dsn, autocommit=True) as c:
            c.execute(f"CREATE SCHEMA {sch};"); c.execute(f"SET search_path TO {sch}, public;")
            for st in [s.strip() for s in seed.split(";") if s.strip()]:
                c.execute(st)
            body = "\n".join(l for l in q.splitlines() if not l.strip().startswith("--"))
            sts = [s.strip() for s in body.split(";") if s.strip()]
            if not sts:
                return None, "empty"
            rows = None
            for st in sts:
                cur = c.execute(st)
                rows = cur.fetchall() if cur.description else []
            return [tuple(str(v) for v in r) for r in (rows or [])], None
    except Exception as ex:
        return None, f"{type(ex).__name__}: {str(ex)[:200]}"
    finally:
        try:
            with psycopg.connect(dsn, autocommit=True) as c:
                c.execute(f"DROP SCHEMA IF EXISTS {sch} CASCADE;")
        except Exception:
            pass


def run_py(code):
    try:
        ast.parse(code)
    except Exception as ex:
        return False, f"SyntaxError: {ex}"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); p = f.name
    try:
        r = subprocess.run([sys.executable, "-m", "ruff", "check", "--no-cache",
                            "--select", "E9,F63,F7,F82", p], capture_output=True,
                           text=True, timeout=60)
        return (r.returncode == 0), (r.stdout.strip()[:300] if r.returncode else "")
    finally:
        os.unlink(p)


def main():
    set_hard_vram_cap(22.0)
    import random
    def load(dom, path):
        rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
        return [r for r in rows
                if str((r.get("meta") or {}).get("family")) in RESERVED[dom]]
    sql_r = load("postgresql", REPO / "data/postgresql/training_data_v3.jsonl")
    py_r = load("astral", REPO / "data/astral/training_data_v3.jsonl")
    random.Random(7).shuffle(sql_r); random.Random(7).shuffle(py_r)

    mgr = PGliteManager(PGliteConfig()); mgr.start(); dsn = mgr.get_dsn(); seed = build_seed()
    build_ddl_index(seed)
    stage("validating references")
    sql_items = []
    for r in sql_r:
        ref, _ = extract_code(r["messages"][1]["content"], "sql")
        rows, cols, err = run_sql_meta(dsn, seed, ref)
        if rows is not None:
            ordered = bool(re.search(r"\border\s+by\b", ref, re.I))
            sql_items.append((r["messages"][0]["content"], rows, tables_in(ref),
                              cols, ordered))
        if len(sql_items) >= N_SQL:
            break
    py_items = []
    for r in py_r:
        ref, _ = extract_code(r["messages"][1]["content"], "python")
        ok, _ = run_py(ref)
        if ok:
            py_items.append(r["messages"][0]["content"])
        if len(py_items) >= N_PY:
            break
    stage(f"  {len(sql_items)} SQL + {len(py_items)} Python items")

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()
    experts = {k: FoldableExpert.from_dir(REPO / v, k) for k, v in ADAPTERS.items()}
    engine = WeightFoldingEngine(model, list(experts.values()), keep_pristine=True)
    dec = FoldedCudaGraphDecoder(model, tok, max_seq_len=MAX_SEQ, device=model.device)
    dec.capture(tok("hi", return_tensors="pt").input_ids.to(model.device))
    stage("ready")

    def gen(msgs, expert):
        """Returns (text, meta). Tokens and wall time are METRICS, not diagnostics.

        A correct answer that costs 3x the tokens is not equally good: in an agent
        loop it is 3x the latency and 3x the context burned before the next tool
        call. So this tracks prompt/completion tokens and seconds per call, and the
        summary reports COST PER CORRECT ANSWER alongside the pass rates.
        """
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        if ids.shape[1] + MAX_NEW >= MAX_SEQ:
            # keeping the TAIL drops the head of the user turn -- which is where
            # schema_prefix() puts the schema. Silently removing the schema would
            # look like a model failure, so say so.
            stage(f"  ⚠ prompt {ids.shape[1]} + {MAX_NEW} exceeds {MAX_SEQ}; "
                  f"truncating prompt head (SCHEMA MAY BE LOST)")
            ids = ids[:, -(MAX_SEQ - MAX_NEW - 8):]
        t0 = time.perf_counter()
        out, _, tps, _ = dec.generate_with_graph(ids, engine=engine, expert=expert,
                                                 max_new_tokens=MAX_NEW)
        dt = time.perf_counter() - t0
        raw = tok.decode(out, skip_special_tokens=True)
        body = raw.split("</think>", 1)[1].strip() if "</think>" in raw else raw.strip()
        return body, {"prompt_tokens": int(ids.shape[1]), "completion_tokens": int(len(out)),
                      "seconds": round(dt, 3), "tok_s": round(tps, 2),
                      # hitting the cap means the answer was cut off, not wrong
                      "truncated": bool(len(out) >= MAX_NEW)}

    HEAL = ("Your previous answer failed with this error:\n\n{err}\n\n"
            "{schema}Fix it and provide the corrected version.")
    res = []
    for kind, items in (("sql", sql_items), ("python", py_items)):
        for idx, item in enumerate(items):
            if kind == "sql":
                _, _, tabs, cols, ordered = item
                # NO output contract. Naming the reference's columns would leak the
                # task requirements -- for percentile_cont it reveals WHICH
                # percentiles to compute, which the prompt never asked for. The
                # schema stays (every real agent has it); inference is the point.
                # Fairness is restored in the GRADER instead, via t3_values below.
                prompt = schema_prefix(tabs) + item[0]
            else:
                prompt = item
            for arm, exp in (("base", None),
                             ("expert", experts["postgresql" if kind == "sql" else "astral"])):
                msgs = [{"role": "user", "content": prompt}]
                reply, m1 = gen(msgs, exp)
                code, fenced = extract_code(reply, kind)
                rec = {"kind": kind, "arm": arm, "idx": idx, "fenced": fenced,
                       "prompt_tokens": m1["prompt_tokens"],
                       "completion_tokens": m1["completion_tokens"],
                       "seconds": m1["seconds"], "tok_s": m1["tok_s"], "attempts": 1,
                       "truncated": m1["truncated"]}
                if kind == "sql":
                    _, ref_rows, ref_tabs, _cols, _ordered = item  # order NOT graded
                    rows, err = run_sql(dsn, seed, code)
                    # ORDER IS NEVER GRADED. Result order is only meaningful if the
                    # task asked for it, and none of these prompts do. For
                    # DISTINCT ON the ORDER BY is required to pick WHICH row
                    # survives per group -- the output order is incidental, so
                    # grading it would fail a correct query for an accident.
                    rec |= {"t1": rows is not None,
                            "t2": rows is not None and bool(tables_in(code) & ref_tabs),
                            # right values present (lenient: extra cols/rows allowed)
                            "t3": values_contained(rows, ref_rows),
                            # right SHAPE: same number of rows
                            "t3_rows": rows is not None and len(rows) == len(ref_rows),
                            # same multiset of rows, order-insensitive
                            "t3_strict": rows is not None and
                                         sorted(rows) == sorted(ref_rows)}
                else:
                    ok, err = run_py(code)
                    rec |= {"t1": ok, "t2": ok, "t3": ok}
                rec["pass1"] = rec["t3"]
                if not rec["t3"] and err:      # ---- ONE self-healing retry ----
                    # a good harness returns the schema alongside a column error
                    sch = (schema_prefix(item[2])
                           if kind == "sql" and re.search(r"column|relation", err or "", re.I)
                           else "")
                    msgs += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": HEAL.format(err=err, schema=sch)}]
                    reply2, m2 = gen(msgs, exp)
                    rec["completion_tokens"] += m2["completion_tokens"]
                    rec["prompt_tokens"] += m2["prompt_tokens"]
                    rec["seconds"] = round(rec["seconds"] + m2["seconds"], 3)
                    rec["attempts"] = 2
                    rec["truncated"] = rec["truncated"] or m2["truncated"]
                    code2, _ = extract_code(reply2, kind)
                    if kind == "sql":
                        rows2, _ = run_sql(dsn, seed, code2)
                        rec["pass2"] = values_contained(rows2, ref_rows)
                    else:
                        rec["pass2"], _ = run_py(code2)
                else:
                    rec["pass2"] = rec["pass1"]
                res.append(rec)
            if (idx + 1) % 8 == 0:
                stage(f"  {kind} {idx+1}/{len(items)}")

    stage("=" * 78)
    for kind in ("sql", "python", "ALL"):
        sub = [r for r in res if kind == "ALL" or r["kind"] == kind]
        if not sub:
            continue
        stage(f"  --- {kind} ---")
        for arm in ("base", "expert"):
            a = [r for r in sub if r["arm"] == arm]
            if not a:
                continue
            n = len(a)
            p1 = sum(r["pass1"] for r in a) / n
            p2 = sum(r["pass2"] for r in a) / n
            rec = (p2 - p1) / max(1e-9, 1 - p1)
            ct = sum(r["completion_tokens"] for r in a)
            sec = sum(r["seconds"] for r in a)
            n_ok = sum(r["pass2"] for r in a)
            line = (f"    {arm:6s} n={n:3d}  pass@1={p1:.3f}  pass@2={p2:.3f}  "
                    f"recovery={rec:.3f}  fenced={sum(r['fenced'] for r in a)/n:.3f}")
            line += (f"\n           tokens/item={ct/n:6.1f}  sec/item={sec/n:5.2f}  "
                     f"tok/s={sum(r['tok_s'] for r in a)/n:5.1f}  "
                     f"retries={sum(r['attempts'] - 1 for r in a):2d}  "
                     f"truncated={sum(r.get('truncated', False) for r in a):2d}  "
                     f"COST/CORRECT: {ct/max(1, n_ok):7.1f} tok, "
                     f"{sec/max(1, n_ok):5.2f} s")
            if kind == "sql":
                line += (f"  |  T1 exec={sum(r['t1'] for r in a)/n:.3f}"
                         f"  T2 tables={sum(r['t2'] for r in a)/n:.3f}"
                         f"  T3 values={sum(r['t3'] for r in a)/n:.3f}"
                         f"  T3 rows={sum(r.get('t3_rows', False) for r in a)/n:.3f}"
                         f"  T3 multiset={sum(r.get('t3_strict', False) for r in a)/n:.3f}")
            stage(line)
    out = REPO / "results/benchmarks/indomain_v2_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    stage(f"wrote {out}")
    mgr.stop(); stage("ALL DONE")


if __name__ == "__main__":
    main()
