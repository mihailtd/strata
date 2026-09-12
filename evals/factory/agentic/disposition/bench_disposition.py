"""Disposition benchmark -- does the expert REACH FOR the native tool?

Runs each task's tool-neutral prompt against base and against that domain's expert,
then classifies which tool the answer reached for.

    uv run --env-file .env python benchmarks/factory/agentic/disposition/bench_disposition.py

WHAT MAKES THIS DIFFERENT FROM EVERY OTHER BENCHMARK HERE
---------------------------------------------------------
The others ask "can it produce correct output when told what to use". This asks
"does it pick the right thing when told nothing". See docs/WHY_EXPERTS.md.

The headline number is not the mean score -- it is the MANUAL RATE: how often the
model confidently solved the problem with the wrong tool. That outcome passes every
execution-gated test we run while being the exact failure the experts exist to
prevent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from runtime.canon import CANON, REPO_ROOT, adapter_path  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "apps"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks/factory/agentic/disposition"))

from runtime.novel_peft import (  # noqa: E402
    FoldableExpert, WeightFoldingEngine, set_hard_vram_cap,
)
from tasks import TASKS, sanity_check  # noqa: E402


# ---------------------------------------------------------------------------
# THE BASELINE THAT DECIDES WHETHER ANY OF THIS IS WORTH IT.
#
# Before an adapter can justify training, folding, routing and stacking, it has
# to beat five lines of text prepended to the prompt. This is that text.
#
# It is written to be as STRONG as possible, not as a strawman: it names the
# tools explicitly, names the anti-tools explicitly, and covers all three
# domains at once -- which is more than any single adapter is given. If the
# adapters cannot beat this, the architecture is not carrying its weight.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert Python and data engineer. Always prefer modern native tooling.

Python packaging and tooling: use uv (uv add, uv lock, uv sync, uv run, uv init, uv python install) and ruff (ruff check, ruff format). Never use pip, poetry, pipenv, pyenv, virtualenv, black, isort, or flake8.

Analytical data work -- Parquet/CSV files, large scans, aggregations: use DuckDB (read_parquet, read_csv, hive_partitioning, GROUP BY ALL, COLUMNS(), EXCLUDE, QUALIFY, USING SAMPLE, .pl() for zero-copy Polars). Never use pandas loops or chunking for bulk file processing.

Relational and transactional data, and vector search: use PostgreSQL's own features (pgvector with HNSW/IVFFlat, DISTINCT ON, FILTER (WHERE), LATERAL, ON CONFLICT, percentile_cont, range partitioning, BRIN, EXPLAIN ANALYZE). Never recommend a separate vector database.

Answer concisely with the idiomatic command or query."""

VERDICTS = ("NATIVE", "MIXED", "MANUAL", "NEITHER")
POINTS = {"NATIVE": 1.0, "MIXED": 0.5, "MANUAL": 0.0, "NEITHER": 0.0}


def classify(text: str, task: dict) -> tuple[str, list[str], list[str]]:
    """Which tool did it reach for? Returns (verdict, native_hits, manual_hits)."""
    nat = [p for p in task["native"] if re.search(p, text, re.I)]
    man = [p for p in task["manual"] if re.search(p, text, re.I)]
    if nat and not man:
        return "NATIVE", nat, man
    if nat and man:
        return "MIXED", nat, man
    if man:
        return "MANUAL", nat, man
    return "NEITHER", nat, man


@torch.no_grad()
def answer(model, tok, prompt: str, max_new_tokens: int, system: str = "") -> tuple[str, int]:
    # The system text is prepended ahead of the SAME ### Question: framing the
    # experts were trained on, so the only variable between arms is the
    # instruction -- not the prompt format.
    pre = f"{system}\n\n" if system else ""
    ids = tok(f"{pre}### Question:\n{prompt}\n\n### Answer:\n", return_tensors="pt").to(model.device)
    gen = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        stop_strings=["### Question"],
        tokenizer=tok,
    )
    new = gen[0][ids["input_ids"].shape[1]:]
    txt = tok.decode(new, skip_special_tokens=True)
    return re.split(r"#+\s*Question", txt)[0].strip(), int(new.numel())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/benchmarks/disposition_results.json")
    args = ap.parse_args()

    problems = sanity_check()
    if problems:
        # A prompt that names its own tool measures capability, not disposition.
        # Refuse rather than silently produce a number that means something else.
        for p in problems:
            print(f"  TASK DEFECT: {p}")
        raise SystemExit(1)

    set_hard_vram_cap(CANON.VRAM_CAP_GB)
    tok = AutoTokenizer.from_pretrained(CANON.BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        CANON.BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()

    domains = sorted({t["domain"] for t in TASKS})
    experts = {d: FoldableExpert.from_dir(adapter_path(d), d) for d in domains}
    engine = WeightFoldingEngine(model, experts.values(), keep_pristine=True)

    print("=" * 92)
    print(f" DISPOSITION BENCHMARK -- max_new_tokens={CANON.MAX_NEW_TOKENS} greedy "
          f"experts={CANON.ADAPTER_VERSION}")
    print(" does the model REACH FOR the native tool when the prompt names none?")
    print("=" * 92, flush=True)

    rows, out_path = [], REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for i, task in enumerate(TASKS, 1):
        rec = {"id": task["id"], "domain": task["domain"], "prompt": task["prompt"]}
        # 4th arm: they are not alternatives. The system prompt wins on
        # coverage (2 NEITHER vs the expert's 11) and the expert wins on
        # tokens (155 vs 719) and never hedges (0 MIXED vs 8). If those
        # compose, "both" is the config to ship and neither arm alone
        # measures it.
        for arm in ("base", "sysprompt", "expert", "both"):
            if arm in ("expert", "both"):
                engine.activate_many([experts[task["domain"]]])
            else:
                engine.restore()
            sys_txt = SYSTEM_PROMPT if arm in ("sysprompt", "both") else ""
            txt, ntok = answer(model, tok, task["prompt"],
                               CANON.MAX_NEW_TOKENS, system=sys_txt)
            verdict, nat, man = classify(txt, task)
            rec[arm] = {"verdict": verdict, "tokens": ntok,
                        "native_hits": nat, "manual_hits": man, "text": txt[:1200]}
        rows.append(rec)
        print(f" [{i:2d}/{len(TASKS)}] {task['id']:20s} "
              f"base={rec['base']['verdict']:8s} sys={rec['sysprompt']['verdict']:8s} "
              f"expert={rec['expert']['verdict']:8s} both={rec['both']['verdict']:8s}",
              flush=True)

        # persist every item: a long run that only writes at the end loses
        # everything to a crash, which is how a training run was lost once
        out_path.write_text(json.dumps(
            {"config": CANON.stamp() | {"complete": len(rows) == len(TASKS),
                                        "items_done": len(rows)},
             "items": rows}, indent=2))

    engine.restore()

    print("\n" + "=" * 92)
    print(f" {'domain':12s} {"arm":9s} {'NATIVE':>7s} {'MIXED':>7s} {'MANUAL':>7s} "
          f"{'NEITHER':>8s} {'score':>7s} {'tokens':>8s}")
    print("-" * 92)
    summary = {}
    for dom in domains + ["ALL"]:
        sel = rows if dom == "ALL" else [r for r in rows if r["domain"] == dom]
        for arm in ("base", "sysprompt", "expert", "both"):
            c = Counter(r[arm]["verdict"] for r in sel)
            score = sum(POINTS[r[arm]["verdict"]] for r in sel) / max(1, len(sel))
            toks = sum(r[arm]["tokens"] for r in sel) / max(1, len(sel))
            summary[f"{dom}|{arm}"] = {"counts": dict(c), "score": score, "mean_tokens": toks,
                                       "manual_rate": c["MANUAL"] / max(1, len(sel))}
            print(f" {dom:12s} {arm:9s} {c['NATIVE']:7d} {c['MIXED']:7d} {c['MANUAL']:7d} "
                  f"{c['NEITHER']:8d} {score:7.3f} {toks:8.0f}")
        print("-" * 92)

    b, sp, e = summary["ALL|base"], summary["ALL|sysprompt"], summary["ALL|expert"]
    bo = summary["ALL|both"]
    print(f"\n  MANUAL RATE (solved it with the WRONG TOOL -- the failure that every")
    print(f"  execution-gated benchmark scores as a SUCCESS):")
    print(f"     base      {b['manual_rate']:.1%}")
    print(f"     sysprompt {sp['manual_rate']:.1%}   vs base {sp['manual_rate']-b['manual_rate']:+.1%}")
    print(f"     expert    {e['manual_rate']:.1%}   vs base {e['manual_rate']-b['manual_rate']:+.1%}"
          f"   vs sysprompt {e['manual_rate']-sp['manual_rate']:+.1%}   <-- THE ONE THAT MATTERS")
    print(f"\n  disposition score:  base {b['score']:.3f}   sysprompt {sp['score']:.3f}   "
          f"expert {e['score']:.3f}")
    print(f"  expert - sysprompt = {e['score']-sp['score']:+.3f}")
    print(f"  both  {bo['score']:.3f}  ({bo['mean_tokens']:.0f} tok)   "
          f"vs sysprompt {bo['score']-sp['score']:+.3f}   vs expert {bo['score']-e['score']:+.3f}")
    print("\n  score per 100 tokens (the architecture's actual case is cost, not quality):")
    for nm, a in (("base", b), ("sysprompt", sp), ("expert", e), ("both", bo)):
        print(f"     {nm:10s} {a['score']/max(1e-9,a['mean_tokens'])*100:.3f}")
    print("\n  If the expert does not beat the system prompt, five lines of text buy")
    print("  what training, folding, routing and stacking buy -- and the architecture")
    print("  is not carrying its weight. See docs/WHY_EXPERTS.md.")

    out_path.write_text(json.dumps(
        {"config": CANON.stamp() | {"complete": True, "items_done": len(rows)},
         "summary": summary, "items": rows}, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
