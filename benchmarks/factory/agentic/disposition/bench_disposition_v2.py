"""Disposition scorer for the held-out-situation eval sets: base vs v4 vs v6.

WHY A NEW SCORER
----------------
The old disposition benchmark was retired (DECISIONS.md §44) because it counted
tool NAMES. It scored an astral answer NATIVE=1.0 that invented an invalid
`[tool.uv] sources = [...]` schema, recommended git-installing Ruff, and never
mentioned `uv.lock` on a reproducible-builds question.

This one reads BOTH directions from the eval file:

    expects  the right approach appeared
    avoid    the common WRONG approach appeared

`avoid` is the half that matters. "Solved it correctly with the wrong tool" is
scored as a SUCCESS by every execution-gated benchmark in this repo, and it is the
exact failure the experts exist to prevent.

    NATIVE   expects hit, no avoid      1.0
    MIXED    both hit                   0.5   -- knows it, hedges
    MANUAL   avoid only                 0.0   -- THE failure mode
    NEITHER  neither                    0.0   -- did not engage

⚠️ STILL A REGEX SCORER. It cannot tell working code from broken code -- it only
knows which tool was reached for. That is the question it is designed for, and it
is NOT evidence that the emitted code runs. Execution gating is a separate
instrument.

    uv run --env-file .env python benchmarks/factory/agentic/disposition/bench_disposition_v2.py
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from gnn_experiment.canon import CANON, REPO_ROOT  # noqa: E402
from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert, WeightFoldingEngine, set_hard_vram_cap,
)

# domain -> (eval file, v4 adapter, v6 adapter). Naming differs between
# generations for financial, so the paths are explicit rather than derived.
# DELIBERATE v4 paths: this benchmark's whole purpose is comparing generations,
# so it pins both sides explicitly rather than following CANON.ADAPTER_VERSION.
PAIRS = {
    "astral": ("data/astral/evaluation_data_disposition.jsonl",
               "results/adapters/m2_astral_r8a128_v4",
               "results/adapters/m2_astral_r8a128_v6"),
    "postgresql": ("data/postgresql/evaluation_data_disposition.jsonl",
                   "results/adapters/m2_postgresql_r8a128_v4",
                   "results/adapters/m2_postgresql_r8a128_v6"),
    "duckdb": ("data/duckdb/evaluation_data_disposition.jsonl",
               "results/adapters/m2_duckdb_r8a128_v4",
               "results/adapters/m2_duckdb_r8a128_v6"),
    "financial_planning": ("data/financial_planning/evaluation_data_disposition.jsonl",
                           "results/adapters/m2_financial_r8a128_v4",
                           "results/adapters/m2_financial_r8a128_v6"),
}
POINTS = {"NATIVE": 1.0, "MIXED": 0.5, "MANUAL": 0.0, "NEITHER": 0.0}


def classify(text: str, item: dict) -> tuple[str, list, list]:
    exp = [p for p in item.get("expects", []) if re.search(p, text, re.I)]
    avo = [p for p in item.get("avoid", []) if re.search(p, text, re.I)]
    if exp and not avo:
        return "NATIVE", exp, avo
    if exp and avo:
        return "MIXED", exp, avo
    if avo:
        return "MANUAL", exp, avo
    return "NEITHER", exp, avo


@torch.no_grad()
def answer(model, tok, prompt: str, max_new: int) -> tuple[str, int]:
    ids = tok(f"### Question:\n{prompt}\n\n### Answer:\n",
              return_tensors="pt").to(model.device)
    gen = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id or tok.eos_token_id,
                         stop_strings=["### Question"], tokenizer=tok)
    new = gen[0][ids["input_ids"].shape[1]:]
    txt = tok.decode(new, skip_special_tokens=True)
    return re.split(r"#+\s*Question", txt)[0].strip(), int(new.numel())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/benchmarks/disposition_v4_vs_v6.json")
    ap.add_argument("--max-new-tokens", type=int, default=CANON.MAX_NEW_TOKENS)
    args = ap.parse_args()

    set_hard_vram_cap(CANON.VRAM_CAP_GB)
    tok = AutoTokenizer.from_pretrained(CANON.BASE_MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        CANON.BASE_MODEL, dtype=torch.bfloat16, device_map={"": 0},
        trust_remote_code=True)
    model.eval()

    experts, items = {}, {}
    for dom, (ev, p4, p6) in PAIRS.items():
        f = REPO_ROOT / ev
        if not f.exists():
            print(f"  SKIP {dom}: no eval file"); continue
        items[dom] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        for gen, p in (("v4", p4), ("v6", p6)):
            d = REPO_ROOT / p
            if (d / "adapter_model.safetensors").exists():
                experts[(dom, gen)] = FoldableExpert.from_dir(d, f"{dom}_{gen}")
            else:
                print(f"  SKIP {dom} {gen}: {p} has no weights")

    engine = WeightFoldingEngine(model, experts.values(), keep_pristine=True)
    print("=" * 88)
    print(f" DISPOSITION: base vs v4 vs v6   max_new_tokens={args.max_new_tokens} greedy")
    print(" held-out SITUATIONS, trained CONSTRUCTS -- scored on expects AND avoid")
    print("=" * 88, flush=True)

    rows, t0 = [], time.time()
    for dom, its in items.items():
        for it in its:
            rec = {"domain": dom, "id": it["id"], "prompt": it["prompt"]}
            for gen in ("base", "v4", "v6"):
                if gen == "base":
                    engine.restore()
                elif (dom, gen) in experts:
                    engine.activate_many([experts[(dom, gen)]])
                else:
                    continue
                txt, n = answer(model, tok, it["prompt"], args.max_new_tokens)
                v, e, a = classify(txt, it)
                rec[gen] = {"verdict": v, "tokens": n, "expects_hit": e,
                            "avoid_hit": a, "text": txt[:900]}
            rows.append(rec)
            print(f"  [{len(rows):2d}/{sum(len(x) for x in items.values())}] "
                  f"{dom:18s} {it['id']:8s} "
                  + "  ".join(f"{g}={rec[g]['verdict']:7s}"
                              for g in ("base", "v4", "v6") if g in rec), flush=True)
            (REPO_ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
            (REPO_ROOT / args.out).write_text(json.dumps(
                {"config": CANON.stamp() | {"complete": False}, "items": rows}, indent=2))

    engine.restore()
    print("\n" + "=" * 88)
    print(f" {'domain':20s} {'arm':5s} {'NAT':>4s} {'MIX':>4s} {'MAN':>4s} {'NEI':>4s} "
          f"{'score':>7s} {'manual%':>8s} {'tokens':>7s}")
    print("-" * 88)
    summary = {}
    for dom in list(items) + ["ALL"]:
        sel = rows if dom == "ALL" else [r for r in rows if r["domain"] == dom]
        for gen in ("base", "v4", "v6"):
            s = [r[gen] for r in sel if gen in r]
            if not s:
                continue
            c = Counter(x["verdict"] for x in s)
            sc = sum(POINTS[x["verdict"]] for x in s) / len(s)
            summary[f"{dom}|{gen}"] = {"counts": dict(c), "score": sc,
                                       "manual_rate": c["MANUAL"] / len(s),
                                       "mean_tokens": sum(x["tokens"] for x in s) / len(s)}
            print(f" {dom:20s} {gen:5s} {c['NATIVE']:4d} {c['MIXED']:4d} {c['MANUAL']:4d} "
                  f"{c['NEITHER']:4d} {sc:7.3f} {c['MANUAL']/len(s):7.1%} "
                  f"{sum(x['tokens'] for x in s)/len(s):7.0f}")
        print("-" * 88)

    b, v4, v6 = (summary.get("ALL|base"), summary.get("ALL|v4"), summary.get("ALL|v6"))
    if v4 and v6:
        print(f"\n  score      base {b['score']:.3f}   v4 {v4['score']:.3f}   "
              f"v6 {v6['score']:.3f}   (v6-v4 {v6['score']-v4['score']:+.3f})")
        print(f"  MANUAL     base {b['manual_rate']:.1%}   v4 {v4['manual_rate']:.1%}   "
              f"v6 {v6['manual_rate']:.1%}")
        print(f"  tokens     base {b['mean_tokens']:.0f}   v4 {v4['mean_tokens']:.0f}   "
              f"v6 {v6['mean_tokens']:.0f}")
        n = len(rows)
        print(f"\n  n={n} items. A one-item swing is {1/n:.1%} -- differences smaller")
        print("  than a few items are not interpretable at this size.")

    (REPO_ROOT / args.out).write_text(json.dumps(
        {"config": CANON.stamp() | {"complete": True}, "summary": summary,
         "items": rows}, indent=2))
    print(f"\n  {time.time()-t0:.0f}s   wrote {args.out}")


if __name__ == "__main__":
    main()
