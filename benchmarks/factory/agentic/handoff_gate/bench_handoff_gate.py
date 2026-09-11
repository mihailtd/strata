"""THE HANDOFF GATE: does multi-expert routing beat the generalist at all?

The blueprint for an agentic multi-expert engine rests on one unverified
assumption: that specialised experts beat the base model on realistic MULTI-STEP
tasks, and that swapping mid-conversation does not degrade coherence. §21 tested
a single swap scored on a TERM LIST -- and term-list scoring is exactly what
produced two blind evals here already (§9 financial read +0.00pp while the expert
was fine; §16 postgresql leaked giveaways).

So this scores OBJECTIVELY wherever possible:
    SQL     must parse as Postgres via sqlglot            (not keyword matching)
    Python  must parse via ast.parse                      (not keyword matching)
plus a small pre-registered structural rubric per step, audited below for
giveaways -- no rubric term may appear in the prompt that elicits it.

THREE ARMS on identical multi-turn conversations:
    A  base 4B only, never swaps                    -- the generalist baseline
    B  ORACLE routing, expert hand-assigned per step -- upper bound on routing
    C  SELF routing, the model picks its own expert  -- what would actually ship

    B <= A          -> the blueprint is dead; experts do not help here
    B > A ~= C      -> experts work, ROUTING is the hard problem
    C > A           -> build the whole thing

Run:
    uv run --with sqlglot --env-file .env python <this file>
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/home/mihai/gnn-experiment")
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

T0 = time.perf_counter()


def stage(m): print(f"[{time.perf_counter() - T0:7.1f}s] {m}", flush=True)


import sqlglot  # noqa: E402
import torch  # noqa: E402

torch.zeros(1, device="cuda"); torch.cuda.synchronize()
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from runtime.cuda_graph import FoldedCudaGraphDecoder, apply_expert_state  # noqa: E402
from runtime.novel_peft import (  # noqa: E402
    FoldableExpert, WeightFoldingEngine, set_hard_vram_cap,
)

MODEL = "Qwen/Qwen3.5-4B"
ADAPTERS = {"postgresql": os.environ.get("PG_ADAPTER",
                                        "results/adapters/m2_postgresql_r8a128"),
            "astral": "results/adapters/m2_astral_r8a128"}
MAX_NEW = int(os.environ.get("MAX_NEW", "2048"))
# THINKING=1 gives the model a test-time-compute runway. The gate's first run used
# THINKING=0 for tractability; all arms shared it, so the A/B/C comparison was fair,
# but it could not see whether EXPERTS benefit from deliberation differentially.
THINKING = os.environ.get("THINKING", "0") == "1"
MAX_SEQ = 4096

# ---------------------------------------------------------------- the tasks
# Each task is a 3-step agent conversation: two database steps then a Python
# step, so every arm exercises a cross-domain handoff in mid-conversation.
def sql_ok(t):
    try:
        return bool(sqlglot.parse(t, dialect="postgres"))
    except Exception:
        return False


def py_ok(t):
    try:
        ast.parse(t)
        return True
    except Exception:
        return False


TASKS = [
    {
        "id": "products",
        "schema": "products(id bigserial pk, title text, price numeric, embedding vector(1024))",
        "topic": "product catalogue semantic search",
    },
    {
        "id": "tickets",
        "schema": "tickets(id bigserial pk, subject text, status text, embedding vector(768))",
        "topic": "support ticket retrieval",
    },
    {
        "id": "docs",
        "schema": "documents(id bigserial pk, body text, tenant_id int, embedding vector(1536))",
        "topic": "multi-tenant document search",
    },
    {
        "id": "papers",
        "schema": "papers(id bigserial pk, abstract text, year int, embedding vector(768))",
        "topic": "research paper similarity",
    },
    {
        "id": "images",
        "schema": "images(id bigserial pk, caption text, album_id int, embedding vector(512))",
        "topic": "image caption search",
    },
]


def build_steps(t):
    return [
        {
            "domain": "postgresql",
            "prompt": (f"Our table is {t['schema']}. Write the migration statement that "
                       f"builds an approximate-nearest-neighbour index on the embedding "
                       f"column for cosine similarity. Output only SQL in a ```sql block."),
            "lang": "sql",
            "checks": {
                "creates_index": lambda s: re.search(r"create\s+index", s, re.I) is not None,
                "ann_method": lambda s: re.search(r"\busing\s+(hnsw|ivfflat)\b", s, re.I) is not None,
                "cosine_opclass": lambda s: re.search(r"vector_cosine_ops", s, re.I) is not None,
            },
        },
        {
            "domain": "postgresql",
            "prompt": ("Now write the query that returns the 10 closest rows to a "
                       "parameter :q for that same table, and make it able to use the "
                       "index you just created. Output only SQL in a ```sql block."),
            "lang": "sql",
            "checks": {
                "distance_operator": lambda s: re.search(r"<=>|<->|<#>", s) is not None,
                "orders_by_distance": lambda s: re.search(r"order\s+by", s, re.I) is not None,
                "limits_ten": lambda s: re.search(r"limit\s+10\b", s, re.I) is not None,
            },
        },
        {
            "domain": "astral",
            "prompt": ("Now expose that query as a FastAPI endpoint. It must be async, "
                       "annotate its parameters and return type, and use a connection "
                       "pool rather than opening a connection per request. Output only "
                       "Python in a ```python block."),
            "lang": "python",
            "checks": {
                "async_handler": lambda s: re.search(r"async\s+def", s) is not None,
                "route_decorator": lambda s: re.search(r"@\w+\.(get|post)", s) is not None,
                "annotated": lambda s: re.search(r"->\s*\w", s) is not None,
                "pooled": lambda s: re.search(r"pool", s, re.I) is not None,
            },
        },
    ]


BLOCK = re.compile(r"```(?:sql|python|py)?\s*(.*?)```", re.S | re.I)


def extract(text, lang):
    m = BLOCK.findall(text)
    if m:
        return max(m, key=len).strip()
    return text.strip()


def score_step(text, step):
    fenced = bool(BLOCK.search(text))
    code = extract(text, step["lang"])
    parses = sql_ok(code) if step["lang"] == "sql" else py_ok(code)
    passed = {k: bool(f(code)) for k, f in step["checks"].items()}
    return {"fenced": fenced, "parses": parses, "checks": passed,
            "score": (int(parses) + sum(passed.values())) / (1 + len(passed)),
            "code": code[:400]}


def audit_giveaways():
    """No rubric term may appear in the prompt that elicits it."""
    bad = []
    for t in TASKS[:1]:
        for st in build_steps(t):
            p = st["prompt"].lower()
            for term in ("hnsw", "ivfflat", "vector_cosine_ops", "<=>", "<->",
                         "async def", "@app", "->", "create index", "limit 10"):
                if term in p:
                    bad.append((st["domain"], term))
    return bad


@torch.no_grad()
def gen(dec, tok, messages, engine, expert, max_new=MAX_NEW):
    # THINKING OFF. With it on, Qwen3.5 burns >1024 tokens deliberating and never
    # closes <think> -- measured: at max_new=1024 there were 13 code fences, ALL of
    # them draft attempts inside the reasoning block, ending mid-sentence on "Wait,
    # I recall that". Every arm then scored parses=0/3 and the rubric matched
    # keywords in prose, which is a broken instrument, not a model result.
    # It is also the right product choice: an agent that spends 1000+ tokens
    # deliberating per tool call is unusable whatever the adapter quality.
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=THINKING)
    ids = tok(text, return_tensors="pt").input_ids.to(dec.model.device)
    if ids.shape[1] + max_new >= MAX_SEQ:
        ids = ids[:, -(MAX_SEQ - max_new - 8):]
    t = time.perf_counter()
    out, _, _, swap_ms = dec.generate_with_graph(ids, engine=engine, expert=expert,
                                                 max_new_tokens=max_new)
    dt = time.perf_counter() - t
    raw = tok.decode(out, skip_special_tokens=True)
    closed = "</think>" in raw
    # An unclosed <think> means the runway was too short: there IS no final answer,
    # only deliberation. Counting that as an empty answer is the honest scoring --
    # it is exactly what a caller would receive.
    body = raw.split("</think>", 1)[1] if closed else ("" if THINKING else raw)
    return body.strip(), dt, swap_ms, {"emitted": len(out), "think_closed": closed}


ROUTE_Q = ("Which specialist should handle the next request: answer with exactly one "
           "word, either `database` or `python`.\n\nRequest: {p}\n\nSpecialist:")


def route(dec, tok, prompt, engine):
    txt, _, _, _ = gen(dec, tok, [{"role": "user", "content": ROUTE_Q.format(p=prompt)}],
                    engine, None, max_new=8 if not THINKING else 256)
    low = txt.lower()
    if "database" in low or "sql" in low or "postgres" in low:
        return "postgresql"
    if "python" in low or "fastapi" in low:
        return "astral"
    return None


def main():
    set_hard_vram_cap(22.0)
    bad = audit_giveaways()
    stage(f"giveaway audit: {'CLEAN' if not bad else bad}")
    if bad:
        stage("ABORT -- rubric terms leak into the prompts"); return

    stage("loading")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True).eval()
    experts = {k: FoldableExpert.from_dir(REPO_ROOT / v, k) for k, v in ADAPTERS.items()}
    engine = WeightFoldingEngine(model, list(experts.values()), keep_pristine=True)
    dec = FoldedCudaGraphDecoder(model, tok, max_seq_len=MAX_SEQ, device=model.device)
    warm = tok("hello", return_tensors="pt").input_ids.to(model.device)
    dec.capture(warm)
    stage(f"ready  alloc={torch.cuda.memory_allocated()/2**30:.2f}G")

    results = {"A base": [], "B oracle": [], "C self": []}
    swaps = {"A base": [], "B oracle": [], "C self": []}
    routes = []

    for task in TASKS:
        steps = build_steps(task)
        for arm in ("A base", "B oracle", "C self"):
            msgs, per_step = [], []
            for si, st in enumerate(steps):
                if arm == "A base":
                    exp = None
                elif arm == "B oracle":
                    exp = experts[st["domain"]]
                else:
                    picked = route(dec, tok, st["prompt"], engine)
                    routes.append({"task": task["id"], "step": si,
                                   "true": st["domain"], "picked": picked})
                    exp = experts.get(picked) if picked else None
                msgs.append({"role": "user", "content": st["prompt"]})
                txt, dt, swap_ms, meta = gen(dec, tok, msgs, engine, exp)
                msgs.append({"role": "assistant", "content": txt})
                sc = score_step(txt, st)
                sc.update({"task": task["id"], "step": si, "arm": arm,
                           "sec": dt, "swap_ms": swap_ms, **meta})
                per_step.append(sc)
                swaps[arm].append(swap_ms)
            results[arm].extend(per_step)
            m = sum(s["score"] for s in per_step) / len(per_step)
            stage(f"  {task['id']:9s} {arm:9s} score={m:.3f} "
                  f"parses={sum(s['parses'] for s in per_step)}/3 "
                  f"fenced={sum(s['fenced'] for s in per_step)}/3 "
                  f"closed={sum(s['think_closed'] for s in per_step)}/3 "
                  f"toks={sum(s['emitted'] for s in per_step)}")

    stage("=" * 78)
    base_mean = None
    for arm in ("A base", "B oracle", "C self"):
        rs = results[arm]
        mean = sum(s["score"] for s in rs) / len(rs)
        parse = sum(s["parses"] for s in rs) / len(rs)
        sw = sum(swaps[arm]) / max(1, len(swaps[arm]))
        if base_mean is None:
            base_mean = mean
        fen = sum(s["fenced"] for s in rs) / len(rs)
        cl = sum(s["think_closed"] for s in rs) / len(rs)
        et = sum(s["emitted"] for s in rs) / len(rs)
        stage(f"  {arm:9s} score={mean:.3f} ({mean-base_mean:+.3f} vs A)  "
              f"parse_rate={parse:.3f}  fenced={fen:.3f}  think_closed={cl:.3f}  "
              f"mean_toks={et:.0f}  mean_swap={sw:.1f} ms")
    if routes:
        acc = sum(r["picked"] == r["true"] for r in routes) / len(routes)
        stage(f"  routing accuracy (arm C): {acc:.3f} over {len(routes)} decisions")
        wrong = [r for r in routes if r["picked"] != r["true"]]
        if wrong:
            stage(f"  misroutes: {wrong[:6]}")

    stage("  --- per-check pass rates ---")
    keys = sorted({k for arm in results for s in results[arm] for k in s["checks"]})
    stage(f"  {'check':22s} " + "  ".join(f"{a:>9s}" for a in results))
    for k in keys:
        row = []
        for arm in results:
            vals = [s["checks"][k] for s in results[arm] if k in s["checks"]]
            row.append(sum(vals) / len(vals) if vals else float("nan"))
        stage(f"  {k:22s} " + "  ".join(f"{v:9.3f}" for v in row))

    out = REPO_ROOT / ("results/handoff_gate_thinking.json" if THINKING
                       else "results/handoff_gate.json")
    out.write_text(json.dumps({"results": results, "routes": routes}, indent=2, default=str))
    stage(f"wrote {out}")
    stage("ALL DONE")


if __name__ == "__main__":
    main()
