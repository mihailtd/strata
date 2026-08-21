"""Chained multi-turn handoff benchmark on HELD-OUT constructs, scored by execution.

WHAT THIS ANSWERS
-----------------
Earlier gates showed experts beating base on tasks whose machinery matched the
training families one-to-one (HNSW, RRF, @mcp.tool(), PEP 723). That measures
rubric-following, not capability. This uses ONLY constructs verified absent from
both training corpora, and scores by GROUND TRUTH:

    SQL     executed against a seeded database; RETURNED ROWS compared to a
            reference solution's rows
    Python  executed in a subprocess; STDOUT compared to expected

No regex rubric, so emitting the right keywords earns nothing. Prompts never name
the construct -- they state the requirement semantically, so any correct approach
scores and the test is capability, not instruction-following.

ARMS (identical 3-turn conversations, prior turns fed back as context)
    A  base 4B, never swaps
    B  oracle routing, expert hand-assigned per step
    C  self routing, the model picks its own expert

    experts beat base HERE  -> generalisation is real
    edge collapses to zero  -> the experts are rubric-followers, which is still
                               shippable if production looks like the rubric, but
                               it is a different claim
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

REPO_ROOT = Path("/home/mihai/gnn-experiment")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

T0 = time.perf_counter()


def stage(m): print(f"[{time.perf_counter() - T0:7.1f}s] {m}", flush=True)


import psycopg  # noqa: E402
import torch  # noqa: E402

torch.zeros(1, device="cuda"); torch.cuda.synchronize()
from py_pglite import PGliteConfig, PGliteManager  # noqa: E402
from tasks import TASKS  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from runtime.canon import CANON, adapter_path  # noqa: E402
from runtime.cuda_graph import FoldedCudaGraphDecoder  # noqa: E402
from runtime.novel_peft import (  # noqa: E402
    FoldableExpert, WeightFoldingEngine, set_hard_vram_cap,
)

ADAPTERS = {
    # canon: no "use v4 if it exists else v2" fallback -- that is how a stale
    # adapter silently survives a corpus rebuild. adapter_path raises instead.
    "postgresql": str(adapter_path("postgresql")),
    "astral": str(adapter_path("astral")),
}
MAX_NEW = CANON.MAX_NEW_TOKENS
MAX_SEQ = 8192
BLOCK = re.compile(r"```(?:sql|python|py)?\s*(.*?)```", re.S | re.I)


def extract(text):
    m = BLOCK.findall(text)
    return (max(m, key=len) if m else text).strip()


# ------------------------------------------------------------------ execution
def sql_rows(dsn, seed, query):
    """Run seed + one query in a throwaway schema; return normalised rows."""
    schema = f"s_{uuid.uuid4().hex[:10]}"
    try:
        with psycopg.connect(dsn, autocommit=True) as c:
            c.execute(f"CREATE SCHEMA {schema};")
            c.execute(f"SET search_path TO {schema}, public;")
            for stmt in [s.strip() for s in seed.split(";") if s.strip()]:
                c.execute(stmt)
            body = "\n".join(l for l in query.splitlines()
                             if not l.strip().startswith("--"))
            stmts = [s.strip() for s in body.split(";") if s.strip()]
            if not stmts:
                return None, "no statement"
            rows = None
            for st in stmts:
                cur = c.execute(st)
                rows = cur.fetchall() if cur.description else []
            return [tuple(str(v) for v in r) for r in (rows or [])], None
    except Exception as ex:
        return None, f"{type(ex).__name__}: {str(ex)[:120]}"
    finally:
        try:
            with psycopg.connect(dsn, autocommit=True) as c:
                c.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE;")
        except Exception:
            pass


def py_stdout(code):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code); path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=25)
        if p.returncode != 0:
            return None, (p.stderr.strip().splitlines() or ["failed"])[-1][:120]
        return p.stdout.strip(), None
    except subprocess.TimeoutExpired:
        return None, "timeout"
    except Exception as ex:
        return None, f"{type(ex).__name__}"
    finally:
        os.unlink(path)


def rows_match(got, exp, ordered):
    if got is None or exp is None:
        return False
    if len(got) != len(exp):
        return False
    return (got == exp) if ordered else (sorted(got) == sorted(exp))


# ------------------------------------------------------------------- the run
def main():
    set_hard_vram_cap(22.0)
    stage("booting py-pglite")
    mgr = PGliteManager(PGliteConfig())
    mgr.start()
    dsn = mgr.get_dsn()
    stage("pglite up")

    # ---- ground truth: run every reference BEFORE any model work -----------
    stage("validating reference solutions")
    truth, dropped = {}, []
    for t in TASKS:
        for i, s in enumerate(t.steps):
            key = (t.id, i)
            if s.kind == "sql":
                rows, err = sql_rows(dsn, t.seed_sql, s.reference)
                if err or rows is None:
                    dropped.append((key, err)); continue
                truth[key] = rows
            else:
                truth[key] = s.reference
    stage(f"references OK: {len(truth)}/{sum(len(t.steps) for t in TASKS)}")
    for k, e in dropped:
        stage(f"  DROPPED {k}: {e}")

    stage("loading model")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()
    experts = {k: FoldableExpert.from_dir(REPO_ROOT / v, k) for k, v in ADAPTERS.items()}
    engine = WeightFoldingEngine(model, list(experts.values()), keep_pristine=True)
    dec = FoldedCudaGraphDecoder(model, tok, max_seq_len=MAX_SEQ, device=model.device)
    dec.capture(tok("hi", return_tensors="pt").input_ids.to(model.device))
    stage(f"ready  alloc={torch.cuda.memory_allocated()/2**30:.2f}G")

    def gen(msgs, expert):
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
        ids = tok(text, return_tensors="pt").input_ids.to(model.device)
        if ids.shape[1] + MAX_NEW >= MAX_SEQ:
            ids = ids[:, -(MAX_SEQ - MAX_NEW - 8):]
        out, _, _, swap = dec.generate_with_graph(ids, engine=engine, expert=expert,
                                                  max_new_tokens=MAX_NEW)
        raw = tok.decode(out, skip_special_tokens=True)
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[1]
        return raw.strip(), swap

    def route(prompt):
        p = prompt.lower()
        if any(w in p for w in ("```python", "python module", "python block",
                                "coroutine", "class ", "callable")):
            return "astral"
        return "postgresql"

    results = {a: {} for a in ("A base", "B oracle", "C self")}
    routes = []
    for t in TASKS:
        for arm in ("A base", "B oracle", "C self"):
            msgs, per = [], []
            for i, s in enumerate(t.steps):
                key = (t.id, i)
                if key not in truth:
                    continue
                if arm == "A base":
                    exp_obj = None
                elif arm == "B oracle":
                    exp_obj = experts[s.domain]
                else:
                    pick = route(s.prompt)
                    routes.append({"task": t.id, "step": i, "true": s.domain, "picked": pick})
                    exp_obj = experts.get(pick)
                msgs.append({"role": "user", "content": s.prompt})
                txt, _ = gen(msgs, exp_obj)
                msgs.append({"role": "assistant", "content": txt})
                code = extract(txt)
                if s.kind == "sql":
                    got, err = sql_rows(dsn, t.seed_sql, code)
                    ok = rows_match(got, truth[key], s.order_matters)
                else:
                    got, err = py_stdout(code)
                    ok = (got is not None and got == truth[key])
                per.append({"task": t.id, "step": i, "construct": s.construct,
                            "kind": s.kind, "ok": bool(ok), "err": err})
            results[arm][t.id] = per
            score = sum(p["ok"] for p in per) / max(1, len(per))
            stage(f"  {t.id:22s} {arm:9s} {score:.3f}  "
                  f"({''.join('1' if p['ok'] else '0' for p in per)})")

    # --------------------------------------------------------------- summary
    def task_scores(arm):
        return [sum(p["ok"] for p in v) / max(1, len(v)) for v in results[arm].values()]

    import random
    random.seed(0)
    A = task_scores("A base")
    stage("=" * 76)
    for arm in ("A base", "B oracle", "C self"):
        S = task_scores(arm)
        mean = sum(S) / len(S)
        line = f"  {arm:9s} mean={mean:.4f}"
        if arm != "A base":
            d = [x - y for x, y in zip(S, A)]
            boot = sorted(sum(random.choice(d) for _ in d) / len(d) for _ in range(20000))
            lo, hi = boot[500], boot[19499]
            line += (f"   edge={sum(d)/len(d):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
                     f"  significant={lo > 0 or hi < 0}")
        stage(line)
    if routes:
        acc = sum(r["picked"] == r["true"] for r in routes) / len(routes)
        stage(f"  routing accuracy: {acc:.3f} over {len(routes)} decisions")

    # per-construct breakdown -- where does the edge live?
    stage("  --- pass rate by held-out construct ---")
    cons = sorted({p["construct"] for arm in results for v in results[arm].values() for p in v})
    stage(f"  {'construct':24s} " + "  ".join(f"{a:>9s}" for a in results))
    for c in cons:
        row = []
        for arm in results:
            vals = [p["ok"] for v in results[arm].values() for p in v if p["construct"] == c]
            row.append(sum(vals) / len(vals) if vals else float("nan"))
        stage(f"  {c:24s} " + "  ".join(f"{v:9.3f}" for v in row))

    out = REPO_ROOT / "results/benchmarks/chained_holdout_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results, "routes": routes,
                               "adapters": ADAPTERS, "dropped": [str(d) for d in dropped]},
                              indent=2, default=str))
    stage(f"wrote {out}")
    mgr.stop()
    stage("ALL DONE")


if __name__ == "__main__":
    main()
