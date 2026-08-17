"""How much did 4-bit training (m1) cost, versus bf16+Liger (m2)?

THE QUESTION
------------
Every adapter in this repo up to today was trained against a 4-bit NF4 base
(`load_in_4bit=True`) while every folding/speculation/stacking benchmark loads a
bf16 base -- so adapters learned a correction to quantized weights and were then
folded into unquantized ones. This measures what that seam cost.

It matters beyond history: 56 adapters are still m1. If the seam is large they
need retraining; if it is small they can simply be retired.

WHAT IS AND IS NOT CONTROLLED
-----------------------------
Controlled: rank 8, alpha 128 (scaling 16), same 7 projections, 150 steps, batch
2, grad-accum 2, lr 2e-4, same dataset, same eval questions. The ONLY differences
are the base precision during training (4-bit NF4 vs bf16) and Liger's fused
kernels -- which are bundled, so this measures the methodology change as a whole
and cannot separate quantization from fused kernels. (Liger's kernels were
verified numerically equivalent to stock, so quantization is the likely driver,
but that is inference, not measurement.)

FINANCIAL IS EXCLUDED. `ctl_lora_fin_a128` was overwritten with a bf16 retrain,
so its true m1 counterpart no longer exists. The earlier financial observation
(-5.00pp -> +4.17pp) came from a run whose eval set has since changed and is not
comparable to these numbers.

Reported as absolute pp deltas with 95% paired bootstrap CIs over questions, not
ratios. Scoring is greedy and deterministic here, so the only uncertainty is
which questions were sampled -- exactly what the bootstrap resamples.

    uv run --env-file .env scripts/benchmark_m1_vs_m2_regime.py
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

if torch.cuda.is_available():  # fla's device probe is @cache'd at import
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.eval.eval_suite import LEGACY_TERMS, MODERN_TERMS  # noqa: E402
from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

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

# domain -> (eval file, (good, bad) terms, m1 adapter, m2 adapter)
PAIRS = {
    "astral": (
        "data/astral/evaluation_data.jsonl", (MODERN_TERMS, LEGACY_TERMS),
        "results/adapters/ctl_lora_r8_a128", "results/adapters/m2_astral_r8a128",
    ),
    "postgresql": (
        "data/postgresql/evaluation_data.jsonl", (POSTGRES_GOOD, POSTGRES_BAD),
        "results/adapters/ctl_lora_pg_a128", "results/adapters/m2_postgresql_r8a128",
    ),
}


def score_one(q, txt, terms):
    if q.get("expects"):
        return 100.0 * sum(1 for p in q["expects"] if re.search(p, txt)) / len(q["expects"])
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
            **ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            stop_strings=["### Question"], tokenizer=tok,
        )
        txt = tok.decode(gen[0][ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        txt = re.split(r"#+\s*Question", txt)[0].strip().lower()
        out.append(score_one(q, txt, terms))
    return out


def ci(a, b, rng, B=10000):
    """Paired bootstrap on b - a over questions."""
    n = len(a)
    d = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        d.append(sum(b[i] - a[i] for i in idx) / n)
    d.sort()
    return d[int(0.025 * B)], d[int(0.975 * B)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--out", default="results/m1_vs_m2_regime.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()

    rng = random.Random(0)
    results = {}
    for dom, (evf, terms, m1_rel, m2_rel) in PAIRS.items():
        qs = [json.loads(x) for x in (REPO_ROOT / evf).read_text().splitlines() if x.strip()]
        print(f"\n=== {dom} ({len(qs)} questions) ===")

        experts = {
            "m1": FoldableExpert.from_dir(REPO_ROOT / m1_rel, "m1"),
            "m2": FoldableExpert.from_dir(REPO_ROOT / m2_rel, "m2"),
        }
        engine = WeightFoldingEngine(model, experts.values(), keep_pristine=True)

        engine.restore()
        base = score(model, tok, qs, terms, args.max_new_tokens)
        print(f"  base           {sum(base) / len(base):6.2f}%")

        per = {}
        for name in ("m1", "m2"):
            engine.activate(experts[name])
            per[name] = score(model, tok, qs, terms, args.max_new_tokens)
            print(f"  {name} folded      {sum(per[name]) / len(per[name]):6.2f}%")
        engine.restore()
        drift = engine.max_drift()

        lo, hi = ci(per["m1"], per["m2"], rng)
        d = (sum(per["m2"]) - sum(per["m1"])) / len(qs)
        results[dom] = {
            "n": len(qs),
            "base_pct": sum(base) / len(base),
            "m1_pct": sum(per["m1"]) / len(qs),
            "m2_pct": sum(per["m2"]) / len(qs),
            "m2_minus_m1_pp": d,
            "ci95_lo": lo, "ci95_hi": hi,
            "resolved": bool(lo > 0 or hi < 0),
            "max_drift_after_restore": drift,
            "per_question": {"base": base, **per},
        }
        print(f"  m2 - m1        {d:+6.2f}pp   CI [{lo:+.2f},{hi:+.2f}]   "
              f"{'RESOLVED' if results[dom]['resolved'] else 'not resolvable'}   "
              f"(restore drift {drift:.2e})")

    print("\n" + "=" * 78)
    print(" Cost of m1 (4-bit NF4) vs m2 (bf16 + Liger), same hyperparameters and data")
    print("=" * 78)
    print(f" {'domain':<14} {'base':>8} {'m1':>8} {'m2':>8} {'m2-m1':>9} {'95% CI':>18}  verdict")
    for dom, r in results.items():
        print(f" {dom:<14} {r['base_pct']:7.2f}% {r['m1_pct']:7.2f}% {r['m2_pct']:7.2f}% "
              f"{r['m2_minus_m1_pp']:+8.2f}pp [{r['ci95_lo']:+6.2f},{r['ci95_hi']:+6.2f}]  "
              f"{'RESOLVED' if r['resolved'] else 'not resolvable'}")
    nres = sum(1 for r in results.values() if r["resolved"])
    print(f"\n {nres} of {len(results)} domains show a resolvable regime effect.")
    print(" financial excluded: its m1 adapter was overwritten by a bf16 retrain.")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
