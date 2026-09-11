"""[CONSOLIDATED / SUPERSEDED]
NOTE: Use `benchmark_stacked_experts.py --scale-mode [none|sqrt_k|inv_k]` for canonical
stacking scaling evaluations. This script is preserved for historical 3-way benchmark reproducibility.

Targeted 3-Way Scaled Stacking Evaluation Across All Domains.
"""

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
import torch

if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime.canon import REPO_ROOT, adapter_path  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.eval.eval_suite import LEGACY_TERMS, MODERN_TERMS
from runtime.novel_peft import FoldableExpert, WeightFoldingEngine, set_hard_vram_cap

POSTGRES_GOOD = [
    r"\bpgvector\b", r"\bhnsw\b", r"\bivfflat\b", r"\bembeddings?\b",
    r"cosine (distance|similarity)", r"semantic search",
    r"unified (platform|database)", r"vector (column|index|extension)",
]
POSTGRES_BAD = [
    r"\bpinecone\b", r"\bweaviate\b", r"\bmilvus\b", r"\bqdrant\b",
    r"\bchroma(db)?\b", r"\bfaiss\b",
    r"(separate|dedicated|standalone) vector (database|store)", r"\belasticsearch\b",
]

DOMAINS = {
    "financial_planning": ("data/financial_planning/evaluation_data.jsonl", None),
    "astral": ("data/astral/evaluation_data.jsonl", (MODERN_TERMS, LEGACY_TERMS)),
    "postgresql": ("data/postgresql/evaluation_data.jsonl", (POSTGRES_GOOD, POSTGRES_BAD)),
    "duckdb": ("data/duckdb/evaluation_data.jsonl", None),
}

EXPERTS_V4 = {
    "fin": str(adapter_path("financial").relative_to(REPO_ROOT)),
    "ast": str(adapter_path("astral").relative_to(REPO_ROOT)),
    "pg": str(adapter_path("postgresql").relative_to(REPO_ROOT)),
    "duck": str(adapter_path("duckdb").relative_to(REPO_ROOT)),
    # DELIBERATE v4: these are the merged-corpus adapters from the
    # stacking-vs-merging experiment (DECISIONS.md §42). They are a fixed
    # historical control, not a generation that tracks canon.
    "msql": "results/adapters/m2_merged_sql_r8a128_v4",
    "mall": "results/adapters/m2_merged_all_r8a128_v4",
}


def score_one(q, txt, terms):
    if q.get("expects"):
        pats = q["expects"]
        # IGNORECASE: txt is lowercased above, but duckdb's expects carry uppercase
        # SQL keywords (FROM/EXCLUDE/WHERE). Without this 45/150 never match.
        return 100.0 * sum(1 for p in pats if re.search(p, txt, re.I)) / len(pats)
    good, bad = terms
    g = sum(len(re.findall(p, txt)) for p in good)
    b = sum(len(re.findall(p, txt)) for p in bad)
    return (g / max(1, g + b)) * 100.0 if (g + b) else 0.0


@torch.no_grad()
def score(model, tok, questions, terms, max_new_tokens):
    out = []
    for q in questions:
        ids = tok(f"### Question:\n{q['prompt']}\n\n### Answer:\n", return_tensors="pt").to(model.device)
        gen = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            stop_strings=["### Question"],
            tokenizer=tok,
        )
        txt = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        txt = re.split(r"#+\s*Question", txt)[0].strip().lower()
        out.append(score_one(q, txt, terms))
    return out


# Two presets over ONE script -- duplicating it into a second file is how a
# 192-token default once shipped alongside a 2048 one.
PRESETS = {
    "matrix": [
        ("Base Model", []),
        ("Solo Astral", [("ast", 1.0)]),
        ("Solo PostgreSQL", [("pg", 1.0)]),
        ("Solo DuckDB", [("duck", 1.0)]),
        ("ast+pg   [SHIP A]", [("ast", 1.0), ("pg", 1.0)]),
        ("ast+duck [SHIP B]", [("ast", 1.0), ("duck", 1.0)]),
        ("pg+duck  [collide]", [("pg", 1.0), ("duck", 1.0)]),
        ("ast+pg+duck", [("ast", 1.0), ("pg", 1.0), ("duck", 1.0)]),
    ],
    # Does STACKING earn its complexity? One adapter trained on the union vs
    # the stacked equivalent. Only the two merged rows are new -- base, the
    # solos and the stacked rows are already measured, and the harness is
    # greedy/deterministic so they carry over. Base is kept purely as a
    # control: it must reproduce 85.00 / 6.25 / 42.44 / 66.67 exactly, or
    # something changed and the carried-over rows are not comparable.
    "merged": [
        ("Base Model [control]", []),
        ("merged_sql  = pg+duck corpora", [("msql", 1.0)]),
        ("merged_all  = +astral corpus", [("mall", 1.0)]),
    ],
}


def main():
    parser = argparse.ArgumentParser(description="Targeted 3-Way Stacking Scaling Benchmark")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--preset", choices=["matrix", "merged"], default="matrix")
    parser.add_argument("--out", default="results/benchmarks/stacked_3way_scaling_comparison.json")
    args = parser.parse_args()

    set_hard_vram_cap(22.0)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()

    needed = {n for _lbl, spec in PRESETS[args.preset] for n, _m in spec}
    experts = {n: FoldableExpert.from_dir(REPO_ROOT / p, n)
               for n, p in EXPERTS_V4.items() if n in needed}
    engine = WeightFoldingEngine(model, experts.values(), keep_pristine=True)

    qs = {
        d: [json.loads(x) for x in (REPO_ROOT / f).read_text().splitlines() if x.strip()]
        for d, (f, _) in DOMAINS.items()
    }

    conditions = PRESETS[args.preset]

    print("=" * 88)
    print(f" max_new_tokens={args.max_new_tokens}   greedy   experts=v4")
    print(" 4-EXPERT STACKING MATRIX -- relatedness vs interference")
    print("=" * 88)
    print(" " + "Condition".ljust(20) + "".join(d.rjust(13) for d in DOMAINS))
    print("-" * 88)

    results = {}
    per_question = {}

    for label, active_specs in conditions:
        if not active_specs:
            engine.restore()
        else:
            active_experts = []
            orig_scalings = []
            for name, mult in active_specs:
                e = experts[name]
                orig_scalings.append((e, e.scaling))
                e.scaling = e.scaling * mult
                active_experts.append(e)

            engine.activate_many(active_experts)

            for e, orig in orig_scalings:
                e.scaling = orig

        per_q = {d: score(model, tok, qs[d], DOMAINS[d][1], args.max_new_tokens) for d in DOMAINS}
        row = {d: sum(v) / len(v) for d, v in per_q.items()}
        results[label] = row
        per_question[label] = per_q

        # WRITE AFTER EVERY CONDITION, not once at the end. A 2h run that only
        # persists on completion loses everything to a crash -- which is exactly
        # how the v5c training run was lost. Partial results are still results.
        out_path = REPO_ROOT / args.out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "config": {"max_new_tokens": args.max_new_tokens, "model": args.model_name,
                       "experts": EXPERTS_V4, "greedy": True,
                       "complete": False, "conditions_done": len(results)},
            "means": results, "per_question": per_question}, indent=2))
        print(" " + label.ljust(20) + "".join(f"{row[d]:12.2f}%" for d in DOMAINS), flush=True)

    engine.restore()
    print("=" * 88)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        # Record the decode cap IN the artifact. A results file that does not say
        # what cap produced it cannot be compared to any other results file, and a
        # short cap silently reshapes every score.
        "config": {"max_new_tokens": args.max_new_tokens, "model": args.model_name,
                   "experts": EXPERTS_V4, "greedy": True,
                   "complete": True},
        "means": results, "per_question": per_question}, indent=2))
    print(f"\nWrote full benchmark report to: {out_path}\n")


if __name__ == "__main__":
    main()
