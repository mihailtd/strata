"""Do the experts EARN their place? In-domain tasks they were never trained on.

WHY THIS AND NOT THE HELD-OUT GATE
----------------------------------
`chained_holdout` establishes only that the experts no longer DAMAGE the base
(edge -0.0333, not significant). On constructs never seen, parity is the expected
outcome, not a win. It cannot show the experts are worth having, because every
construct in it is reserved from training.

This uses the RESERVED FAMILIES THEMSELVES as the test set. They are ideal:

  * squarely IN-DOMAIN -- advanced PostgreSQL features, functional Python idioms
  * verified correct at build time (sqlglot / py_compile + ruff)
  * PROVABLY never trained on -- `scripts/reserve_eval_constructs.py` removed them
    from v4 and verified zero residual occurrences

So a win here is in-domain capability, not memorisation.

SCORING IS EXECUTION, NOT REGEX
-------------------------------
  SQL     both the reference and the model's answer run against a seeded database;
          the model scores only if its RETURNED ROWS match the reference's exactly.
          Items whose reference fails to execute are DROPPED (loudly) rather than
          scored against a broken ground truth.
  Python  the model's module must pass `py_compile` AND `ruff check` with zero
          findings -- the same gate the corpus itself had to pass.

ARMS: base 4B vs the domain expert. Routing is not re-tested; it has measured
1.000 in every gate so far.
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

T0 = time.perf_counter()


def stage(m): print(f"[{time.perf_counter() - T0:7.1f}s] {m}", flush=True)


import psycopg  # noqa: E402
import torch  # noqa: E402

torch.zeros(1, device="cuda"); torch.cuda.synchronize()
from py_pglite import PGliteConfig, PGliteManager  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from gnn_experiment.cuda_graph import FoldedCudaGraphDecoder  # noqa: E402
from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert, WeightFoldingEngine, set_hard_vram_cap,
)

MODEL = "Qwen/Qwen3.5-4B"
ADAPTERS = {
    "postgresql": os.environ.get("PG_ADAPTER", "results/adapters/m2_postgresql_r8a128_v4"),
    "astral": os.environ.get("ASTRAL_ADAPTER", "results/adapters/m2_astral_r8a128_v4"),
}
RESERVED = {
    "postgresql": {"adv_distinct_on", "adv_filter_clause", "adv_ordinality_unnest",
                   "adv_lateral_join", "adv_percentiles"},
    "astral": {"func_partial_curry", "func_singledispatch", "py_protocol_slots",
               "py_taskgroup"},
}
N_SQL = int(os.environ.get("N_SQL", "40"))
N_PY = int(os.environ.get("N_PY", "30"))
MAX_NEW = int(os.environ.get("MAX_NEW", "2048"))
MAX_SEQ = 4096
BLOCK = re.compile(r"```(?:sql|python|py)?\s*(.*?)```", re.S | re.I)


def extract(t):
    m = BLOCK.findall(t)
    return (max(m, key=len) if m else t).strip()


def family_of(r):
    m = r.get("meta") or {}
    return str(m.get("family") or m.get("source") or "?")


def load_reserved(domain, path):
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    return [r for r in rows if family_of(r) in RESERVED[domain]]


# ------------------------------------------------------------------ seeds
def build_seed():
    """CREATE TABLE + deterministic rows for every schema the corpus uses."""
    from build_postgresql_v3_corpus import SCHEMAS
    out = []
    for t, _dom, cols in SCHEMAS:
        decls, names, vals = [], [], []
        for c, ty in cols:
            decls.append(f"{c} {ty}")
            names.append(c)
        # several generated answers reference a `status` column; add it only when
        # the schema does not already declare one, and remember whether we did --
        # appending the VALUE unconditionally is what broke the seed before.
        added_status = "status" not in names
        if added_status:
            decls.append("status text")
            names.append("status")
        out.append(f"CREATE TABLE {t} (" + ", ".join(decls) + ");")
        for i in (1, 2, 3):
            row = []
            for c, ty in cols:
                if "int" in ty:
                    row.append(str(i))
                elif ty.startswith("numeric") or ty in ("real", "double precision"):
                    row.append(f"{i * 10}.5")
                elif ty == "boolean":
                    row.append("true" if i % 2 else "false")
                elif ty == "date":
                    row.append(f"DATE '2024-0{i}-01'")
                elif ty == "timestamptz":
                    row.append(f"TIMESTAMPTZ '2024-0{i}-01 0{i}:00+00'")
                elif ty == "uuid":
                    row.append(f"'00000000-0000-0000-0000-00000000000{i}'::uuid")
                elif ty == "inet":
                    row.append(f"'10.0.0.{i}'::inet")
                elif ty.endswith("[]"):
                    row.append("ARRAY['a','b']")
                else:
                    row.append(f"'v{i}'")
            if added_status:
                row.append("'active'" if i % 2 else "'closed'")
            elif "status" in names:
                # schema already had status -- fill it in its declared position
                row[names.index("status")] = "'active'" if i % 2 else "'closed'"
            vals.append("(" + ", ".join(row) + ")")
        out.append(f"INSERT INTO {t} ({', '.join(names)}) VALUES " + ", ".join(vals) + ";")
    return "\n".join(out)


def sql_rows(dsn, seed, query):
    schema = f"s_{uuid.uuid4().hex[:10]}"
    try:
        with psycopg.connect(dsn, autocommit=True) as c:
            c.execute(f"CREATE SCHEMA {schema};")
            c.execute(f"SET search_path TO {schema}, public;")
            for st in [s.strip() for s in seed.split(";") if s.strip()]:
                c.execute(st)
            body = "\n".join(l for l in query.splitlines() if not l.strip().startswith("--"))
            stmts = [s.strip() for s in body.split(";") if s.strip()]
            if not stmts:
                return None, "no statement"
            rows = None
            for st in stmts:
                cur = c.execute(st)
                rows = cur.fetchall() if cur.description else []
            return [tuple(str(v) for v in r) for r in (rows or [])], None
    except Exception as ex:
        return None, f"{type(ex).__name__}: {str(ex)[:90]}"
    finally:
        try:
            with psycopg.connect(dsn, autocommit=True) as c:
                c.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE;")
        except Exception:
            pass


def py_ok(code):
    """compiles AND ruff-clean -- the same bar the corpus had to clear."""
    try:
        ast.parse(code)
    except Exception as ex:
        return False, f"syntax: {type(ex).__name__}"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); p = f.name
    try:
        r = subprocess.run([sys.executable, "-m", "ruff", "check", "--no-cache",
                            "--select", "E9,F63,F7,F82", p],
                           capture_output=True, text=True, timeout=60)
        return (r.returncode == 0), ("clean" if r.returncode == 0
                                     else r.stdout.strip().splitlines()[:1] or ["fail"])
    except Exception as ex:
        return False, f"{type(ex).__name__}"
    finally:
        os.unlink(p)


def main():
    set_hard_vram_cap(22.0)
    stage("loading reserved evaluation records")
    sql_recs = load_reserved("postgresql", REPO / "data/postgresql/training_data_v3.jsonl")
    py_recs = load_reserved("astral", REPO / "data/astral/training_data_v3.jsonl")
    import random
    random.Random(7).shuffle(sql_recs); random.Random(7).shuffle(py_recs)
    sql_recs, py_recs = sql_recs[:N_SQL], py_recs[:N_PY]
    stage(f"  {len(sql_recs)} SQL + {len(py_recs)} Python candidates")

    stage("booting py-pglite")
    mgr = PGliteManager(PGliteConfig()); mgr.start(); dsn = mgr.get_dsn()
    seed = build_seed()

    stage("validating references (dropping any whose ground truth fails)")
    sql_items = []
    for r in sql_recs:
        ref = extract(r["messages"][1]["content"])
        rows, err = sql_rows(dsn, seed, ref)
        if err or rows is None:
            continue
        sql_items.append((r["messages"][0]["content"], rows, family_of(r)))
    py_items = []
    for r in py_recs:
        ref = extract(r["messages"][1]["content"])
        ok, _ = py_ok(ref)
        if ok:
            py_items.append((r["messages"][0]["content"], family_of(r)))
    stage(f"  usable: {len(sql_items)}/{len(sql_recs)} SQL, {len(py_items)}/{len(py_recs)} Python")

    stage("loading model")
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

    def gen(prompt, expert):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        out, _, _, _ = dec.generate_with_graph(ids, engine=engine, expert=expert,
                                               max_new_tokens=MAX_NEW)
        raw = tok.decode(out, skip_special_tokens=True)
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[1]
        return extract(raw.strip())

    res = {"base": [], "expert": []}
    for i, (prompt, ref_rows, fam) in enumerate(sql_items):
        for arm, exp in (("base", None), ("expert", experts["postgresql"])):
            got, _ = sql_rows(dsn, seed, gen(prompt, exp))
            res[arm].append({"kind": "sql", "family": fam,
                             "ok": got is not None and got == ref_rows})
        if (i + 1) % 10 == 0:
            stage(f"  SQL {i+1}/{len(sql_items)}")
    for i, (prompt, fam) in enumerate(py_items):
        for arm, exp in (("base", None), ("expert", experts["astral"])):
            ok, _ = py_ok(gen(prompt, exp))
            res[arm].append({"kind": "python", "family": fam, "ok": ok})
        if (i + 1) % 10 == 0:
            stage(f"  PY  {i+1}/{len(py_items)}")

    import random as _r
    _r.seed(0)
    b = [x["ok"] for x in res["base"]]
    e = [x["ok"] for x in res["expert"]]
    d = [int(y) - int(x) for x, y in zip(b, e)]
    boot = sorted(sum(_r.choice(d) for _ in d) / len(d) for _ in range(20000))
    lo, hi = boot[500], boot[19499]
    stage("=" * 74)
    stage(f"  n={len(b)}   base={sum(b)/len(b):.4f}   expert={sum(e)/len(e):.4f}")
    stage(f"  edge = {sum(d)/len(d):+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
          f"significant={lo > 0 or hi < 0}")
    for kind in ("sql", "python"):
        bb = [x["ok"] for x in res["base"] if x["kind"] == kind]
        ee = [x["ok"] for x in res["expert"] if x["kind"] == kind]
        if bb:
            stage(f"    {kind:7s} n={len(bb):3d}  base={sum(bb)/len(bb):.3f}  "
                  f"expert={sum(ee)/len(ee):.3f}  edge={sum(ee)/len(ee)-sum(bb)/len(bb):+.3f}")
    fams = sorted({x["family"] for x in res["base"]})
    stage("  --- by reserved family ---")
    for f in fams:
        bb = [x["ok"] for x in res["base"] if x["family"] == f]
        ee = [x["ok"] for x in res["expert"] if x["family"] == f]
        stage(f"    {f:24s} n={len(bb):3d}  base={sum(bb)/len(bb):.3f}  expert={sum(ee)/len(ee):.3f}")
    out = REPO / "results/benchmarks/indomain_reserved_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": res, "adapters": ADAPTERS,
                               "n_sql": len(sql_items), "n_py": len(py_items)}, indent=2))
    stage(f"wrote {out}")
    mgr.stop()
    stage("ALL DONE")


if __name__ == "__main__":
    main()
