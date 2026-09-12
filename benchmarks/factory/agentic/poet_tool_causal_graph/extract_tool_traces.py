r"""Turn recorded routing traces into the tool->expert DAG, and answer the pre-fold question.

WHAT §47 ESTABLISHED AND WHAT IT DID NOT
-----------------------------------------
§47 fit NOTEARS on `simulate_tool_execution_data()` -- np.random.seed(42), a
hand-built ground-truth DAG, Gaussian noise -- and concluded that POET factor
subtraction destroys causal recall (90% -> 10%) so raw counts are optimal. That is
a real result about the ALGORITHM, and this probe honours it: no low-rank filtering.

It says nothing about this runtime, because no observed trace existed. The open
question is operational, not algorithmic:

    Can the runtime predict the next expert well enough to pre-fold it in the
    background while the current tool is still executing?

Pre-folding buys the morph latency when the guess is right, and costs a wasted fold
plus a stall when it is wrong. So the decision needs a measured hit rate, and this
probe reports it against the break-even implied by the measured morph cost -- not
against an assumed 0.95.

INPUT
-----
    export GNN_TOOL_TRACE=results/logs/tool_trace.jsonl   # then run the server
    uv run python benchmarks/factory/agentic/poet_tool_causal_graph/extract_tool_traces.py

With no trace on disk it reports the designed pipeline structure instead, clearly
labelled: a designed workflow is a PRIOR, not observed behaviour, and fitting a DAG
to your own script and calling it discovery is circular.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from runtime.canon import REPO_ROOT
from runtime.tool_trace import TOOL_PATTERNS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_poet_tool_causal_graph import notears_linear  # noqa: E402

EXPERTS = ["astral", "postgresql", "duckdb", "financial", "python_modern", "python_web"]
MORPH_MS = 1.9        # measured fold cost, benchmarks/runtime/folding
MISS_MS = 1.9         # a wrong pre-fold is paid twice: the wasted fold, then the real one


def load_trace(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def designed_pipelines() -> list[list[str]]:
    """target_expert sequences from BENCHMARK_PIPELINES, read WITHOUT importing it.

    The module pulls in torch/transformers at import; this probe is CPU-only and may
    run while the GPU is busy, so the sequence is lifted textually.
    """
    src = (REPO_ROOT / "benchmarks/multi_turn/multi_turn_execution_benchmark.py").read_text()
    seqs, cur = [], []
    for line in src.splitlines():
        if "pipeline_id=" in line and cur:
            seqs.append(cur); cur = []
        m = re.search(r'target_expert="([a-z_]+)"', line)
        if m:
            cur.append(m.group(1))
    if cur:
        seqs.append(cur)
    return [s for s in seqs if len(s) >= 2]


def transition_report(seqs: list[list[str]], label: str) -> dict:
    """P(next expert | current expert), and what a pre-fold would win."""
    nxt: dict[str, Counter] = defaultdict(Counter)
    n = 0
    for s in seqs:
        for a, b in zip(s, s[1:]):
            nxt[a][b] += 1
            n += 1
    print(f"\n  {label}: {len(seqs)} sequences, {n} transitions")
    if not n:
        return {"transitions": 0}
    print(f"    {'from':16s} {'n':>4s}  {'best next':16s} {'P(next|cur)':>12s}")
    print("    " + "-" * 54)
    hits = 0
    rows = []
    for a, c in sorted(nxt.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(c.values())
        b, cnt = c.most_common(1)[0]
        hits += cnt
        rows.append({"from": a, "n": tot, "best_next": b, "p": cnt / tot})
        print(f"    {a:16s} {tot:4d}  {b:16s} {cnt/tot:11.1%}")
    acc = hits / n
    ev = acc * MORPH_MS - (1 - acc) * MISS_MS
    print(f"\n    top-1 hit rate {acc:.1%} over {n} transitions")
    print(f"    pre-fold expected value: {acc:.3f}*{MORPH_MS} - {1-acc:.3f}*{MISS_MS} "
          f"= {ev:+.2f} ms/transition")
    print(f"    break-even hit rate at these costs: {MISS_MS/(MORPH_MS+MISS_MS):.1%}")
    return {"transitions": n, "top1_hit_rate": acc, "expected_ms": ev, "rows": rows}


def tool_expert_dag(rows: list[dict]) -> dict:
    """NOTEARS over [tool surfaces | next expert] co-occurrence counts. Raw counts (§47)."""
    tools = sorted(TOOL_PATTERNS)
    names = tools + [f"next:{e}" for e in EXPERTS]
    d = len(names)
    per_session: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_session[r.get("session", "default")].append(r)

    obs = []
    for _, rs in per_session.items():
        rs.sort(key=lambda r: r["ts"])
        for cur, nxt in zip(rs, rs[1:]):
            v = np.zeros(d)
            for t in cur.get("tools", []):
                if t in tools:
                    v[tools.index(t)] = 1.0
            p = nxt.get("primary")
            if p in EXPERTS:
                v[len(tools) + EXPERTS.index(p)] = 1.0
            obs.append(v)
    X = np.array(obs) if obs else np.zeros((0, d))
    print(f"\n  NOTEARS design matrix: {X.shape[0]} observations x {d} nodes")
    if X.shape[0] < 4 * d:
        print(f"    TOO FEW. Linear SEM structure learning at d={d} wants roughly")
        print(f"    4*d = {4*d} observations before an edge set means anything.")
        print(f"    Need ~{max(0, 4*d - X.shape[0])} more routed turns with GNN_TOOL_TRACE set.")
        return {"observations": int(X.shape[0]), "nodes": d, "fitted": False}
    W = notears_linear(X - X.mean(0), lambda1=0.05, w_threshold=0.1)
    edges = [(names[i], names[j], float(W[i, j]))
             for i in range(d) for j in range(d) if abs(W[i, j]) > 0.1]
    edges.sort(key=lambda e: -abs(e[2]))
    print(f"    {len(edges)} edges above threshold; strongest tool -> expert:")
    for a, b, w in [e for e in edges if e[1].startswith("next:")][:10]:
        print(f"      {a:18s} -> {b:22s} {w:+.3f}")
    return {"observations": int(X.shape[0]), "nodes": d, "fitted": True,
            "edges": edges[:60]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", default="results/logs/tool_trace.jsonl")
    ap.add_argument("--out", default="results/benchmarks/tool_trace_dag.json")
    args = ap.parse_args()

    print("=" * 84)
    print(" TOOL -> EXPERT TRANSITION STRUCTURE, from recorded routing traces")
    print("=" * 84)

    rows = load_trace(REPO_ROOT / args.trace)
    out: dict = {"trace": args.trace, "events": len(rows)}

    if rows:
        seqs: dict[str, list[str]] = defaultdict(list)
        for r in sorted(rows, key=lambda r: r["ts"]):
            if r.get("primary"):
                seqs[r.get("session", "default")].append(r["primary"])
        out["observed"] = transition_report(list(seqs.values()), "OBSERVED")
        out["dag"] = tool_expert_dag(rows)
    else:
        print(f"\n  No trace at {args.trace}.")
        print("  Set GNN_TOOL_TRACE and run the server; every routed turn appends one line.")

    print("\n" + "-" * 84)
    print("  DESIGNED pipeline structure (a PRIOR, not evidence -- these sequences are")
    print("  written by hand in the benchmark, so recovering them proves nothing about")
    print("  what the runtime does):")
    out["designed"] = transition_report(designed_pipelines(), "DESIGNED")

    p = REPO_ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
