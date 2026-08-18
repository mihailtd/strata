"""Does bf16 merge error actually predict task degradation? An alpha sweep.

THE CLAIM UNDER TEST
--------------------
Folding an adapter into bf16 base weights quantises the delta, and the loss
scales inversely with |dW|/|W|. Measured on two adapters: 0.96% relative error
at |dW|/|W| ~ 0.08, and 7.06% at ~0.012. A synthetic sweep confirms the
functional form (rel_err ~ 0.6 * 2^-8 / ratio) holds across 0.005-0.1.

From that, a decision rule was proposed: fold high-scale adapters, keep
low-scale ones wrapped to avoid truncation. THAT RULE IS NOT YET JUSTIFIED.
The adapter with 7x worse merge fidelity lost at most 0.83pp on task score
across three runs -- inside the noise of a 20-question eval. A rule that costs
+21% decode speed needs better evidence than a weight-space number that has
never been shown to matter at the output.

WHAT THIS MEASURES
------------------
One adapter, one dataset, one training budget, five alphas (16..256, a 16x
scaling range). Everything except alpha is held identical, so alpha is the only
independent variable. For each:

    |dW|/|W|              the ratio that drives absorption
    merge rel_err         ||realised - intended||_F / ||intended||_F in bf16
    absorbed %            delta elements the merge rounded away entirely
    wrapped score         rubric coverage, adapter running as wrappers
    folded score          rubric coverage, adapter merged into weights
    folded - wrapped      the quantity the decision rule is really about

Then: does merge rel_err predict (folded - wrapped)? If it does, the rule is
real and the crossover point is measurable. If it does not, the weight-space
error is a true measurement with no demonstrated consequence, and folding
should stay on by default.

Everything runs in ONE model load so no arm is confounded by a reload.

    uv run --env-file .env scripts/benchmark_alpha_absorption_sweep.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(REPO_ROOT))

from gnn_experiment.eval.eval_suite import LEGACY_TERMS, MODERN_TERMS  # noqa: E402
from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    load_novel_adapter,
    set_hard_vram_cap,
    unwrap_novel_lora,
)

# rank_total is 64 for every adapter here, so scaling = alpha/64 throughout.
#
# Scoring differs by domain and that is deliberate. astral and postgresql have a
# genuine opposing set (uv vs pip, pgvector vs pinecone), so a good/(good+bad)
# ratio is meaningful. financial_planning does not -- its anti-pattern list fired
# ZERO times across 60 measured generations -- so it carries a per-question
# rubric in its eval data instead. `score_one` picks whichever the data supplies.
DOMAINS = {
    "financial_planning": {
        "questions": "data/financial_planning/evaluation_data.jsonl",
        "alphas": {
            16: "results/adapters/financial_planning_id_kron_v2",
            32: "results/adapters/fin_sweep_a32",
            64: "results/adapters/financial_planning_id_kron_v3_a64",
            128: "results/adapters/fin_sweep_a128",
            256: "results/adapters/fin_sweep_a256",
        },
    },
    "astral": {
        "questions": "data/astral/evaluation_data.jsonl",
        "alphas": {a: f"results/adapters/astral_sweep_a{a}" for a in (16, 32, 64, 128, 256)},
    },
    "postgresql": {
        "questions": "data/postgresql/evaluation_data.jsonl",
        "alphas": {a: f"results/adapters/pg_sweep_a{a}" for a in (16, 32, 64, 128, 256)},
    },
}

POSTGRES_GOOD = [
    r"\bpgvector\b", r"\bhnsw\b", r"\bivfflat\b", r"\bembeddings?\b",
    r"cosine (distance|similarity)", r"semantic search", r"unified (platform|database)",
    r"vector (column|index|extension)",
]
POSTGRES_BAD = [
    r"\bpinecone\b", r"\bweaviate\b", r"\bmilvus\b", r"\bqdrant\b",
    r"\bchroma(db)?\b", r"\bfaiss\b",
    r"(separate|dedicated|standalone) vector (database|store)", r"\belasticsearch\b",
]
TERMS = {
    "astral": (MODERN_TERMS, LEGACY_TERMS),
    "postgresql": (POSTGRES_GOOD, POSTGRES_BAD),
}


@torch.no_grad()
def merge_fidelity(expert: FoldableExpert, params: dict) -> dict:
    """Intended delta vs what a bf16 merge actually realises."""
    tot_int = tot_err = 0.0
    absorbed = total = 0
    ratios = []
    for key, (u, v) in expert.factors.items():
        w0 = params[key]
        intended = expert.scaling * (u.float() @ v.float())
        realised = (w0.float() + intended).to(torch.bfloat16).float() - w0.float()
        tot_err += (realised - intended).pow(2).sum().item()
        tot_int += intended.pow(2).sum().item()
        absorbed += ((realised == 0) & (intended != 0)).sum().item()
        total += intended.numel()
        ratios.append((intended.abs().mean() / w0.float().abs().mean()).item())
    return {
        "rel_err": (tot_err / tot_int) ** 0.5 if tot_int else 0.0,
        "absorbed_pct": 100.0 * absorbed / total,
        "dw_over_w": sum(ratios) / len(ratios),
    }


def score_one(q: dict, txt: str, terms) -> float:
    """Per-question rubric coverage if the data supplies one, else term ratio."""
    if q.get("expects"):
        pats = q["expects"]
        return 100.0 * sum(1 for p in pats if re.search(p, txt)) / len(pats)
    good, bad = terms
    g = sum(len(re.findall(p, txt)) for p in good)
    b = sum(len(re.findall(p, txt)) for p in bad)
    return (g / max(1, g + b)) * 100.0 if (g + b) else 0.0


def score(model, tokenizer, questions, max_new_tokens, terms=None):
    """Greedy decode with the stop-string fix, then domain-appropriate scoring."""
    out = []
    for q in questions:
        ids = tokenizer(f"### Question:\n{q['prompt']}\n\n### Answer:\n", return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                stop_strings=["### Question"],
                tokenizer=tokenizer,
            )
        txt = tokenizer.decode(gen[0][ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        txt = re.split(r"#+\s*Question", txt)[0].strip().lower()
        out.append(score_one(q, txt, terms))
    return sum(out) / len(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--domain", default="financial_planning", choices=sorted(DOMAINS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = DOMAINS[args.domain]
    sweep, questions_file = cfg["alphas"], cfg["questions"]
    terms = TERMS.get(args.domain)
    out_path = args.out or f"results/alpha_sweep_{args.domain}.json"
    print(f"=== alpha sweep: {args.domain} ===")

    set_hard_vram_cap(args.vram_cap_gb)
    present = {a: d for a, d in sweep.items() if (REPO_ROOT / d).exists()}
    missing = sorted(set(sweep) - set(present))
    if missing:
        print(f"[warn] missing adapters for alpha={missing}; continuing with {sorted(present)}")

    experts = {a: FoldableExpert.from_dir(REPO_ROOT / d, f"alpha{a}") for a, d in present.items()}
    for a, e in sorted(experts.items()):
        print(f"  alpha={a:4d}  {e}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    questions = [json.loads(x) for x in (REPO_ROOT / questions_file).read_text().splitlines() if x.strip()]

    print(f"\nLoading {args.model_name} (bf16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()
    engine = WeightFoldingEngine(model, experts.values(), keep_pristine=True)
    params = dict(model.named_parameters())

    engine.restore()
    base_score = score(model, tokenizer, questions, args.max_new_tokens, terms)
    print(f"base (no adapter): {base_score:.2f}%\n")

    rows = []
    for a in sorted(experts):
        e = experts[a]
        fid = merge_fidelity(e, params)

        engine.restore()
        load_novel_adapter(model, REPO_ROOT / present[a])
        wrapped = score(model, tokenizer, questions, args.max_new_tokens, terms)
        unwrap_novel_lora(model)

        engine.activate(e)
        folded = score(model, tokenizer, questions, args.max_new_tokens, terms)
        engine.restore()

        row = {
            "alpha": a,
            "scaling": e.scaling,
            "dw_over_w": fid["dw_over_w"],
            "merge_rel_err_pct": 100 * fid["rel_err"],
            "absorbed_pct": fid["absorbed_pct"],
            "wrapped_pct": wrapped,
            "folded_pct": folded,
            "delta_pp": folded - wrapped,
        }
        rows.append(row)
        print(
            f"  alpha={a:4d} s={e.scaling:5.2f}  |dW|/|W|={row['dw_over_w']:.4f}  "
            f"merge_err={row['merge_rel_err_pct']:6.2f}%  absorbed={row['absorbed_pct']:5.2f}%  "
            f"wrapped={wrapped:6.2f}%  folded={folded:6.2f}%  delta={row['delta_pp']:+6.2f}pp",
            flush=True,
        )

    print("\n" + "=" * 78)
    print(" Does merge error predict task degradation?")
    print("=" * 78)
    print(f" base (no adapter): {base_score:.2f}%")
    print(
        f" {'alpha':>6s} {'scaling':>8s} {'|dW|/|W|':>9s} {'merge err':>10s} "
        f"{'wrapped':>9s} {'folded':>8s} {'delta':>8s}"
    )
    for r in rows:
        print(
            f" {r['alpha']:6d} {r['scaling']:8.2f} {r['dw_over_w']:9.4f} {r['merge_rel_err_pct']:9.2f}% "
            f"{r['wrapped_pct']:8.2f}% {r['folded_pct']:7.2f}% {r['delta_pp']:+7.2f}pp"
        )

    if len(rows) >= 3:
        xs = [r["merge_rel_err_pct"] for r in rows]
        ys = [r["delta_pp"] for r in rows]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sx = (sum((x - mx) ** 2 for x in xs) / n) ** 0.5
        sy = (sum((y - my) ** 2 for y in ys) / n) ** 0.5
        corr = (sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / n / (sx * sy)) if sx and sy else 0.0
        print(f"\n corr(merge_err, folded-wrapped) = {corr:+.3f}  over n={n} alphas")
        print(" A rule that says 'do not fold low-scale adapters' needs this to be")
        print(" strongly NEGATIVE (more merge error -> worse folded score).")
        print(f" |max delta| across the whole sweep: {max(abs(r['delta_pp']) for r in rows):.2f}pp")

    out = REPO_ROOT / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"domain": args.domain, "base_pct": base_score, "rows": rows}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
