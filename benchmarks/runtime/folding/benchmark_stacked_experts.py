"""Can several domain experts be folded into one model at once?

THE PREDICTION BEING TESTED
---------------------------
The subspace probe measured every cross-task adapter pair at 1.10-1.28x chance,
i.e. statistically orthogonal. That has only ever been used as a NEGATIVE result
(shared-basis compression cannot work). Read forward it is a prediction:
orthogonal deltas should compose additively, so

    W0 + dW_astral + dW_financial

ought to preserve BOTH behaviours. If it holds you get multi-domain experts
without multi-domain training, and with no swap latency at all.

Direct weight-space confirmation of the premise (measured, one q_proj slot):

    cos(d_financial, d_astral)          = +0.0002      (0 = orthogonal)
    |d_stacked|                         =  4.933
    sqrt(|d_fin|^2 + |d_ast|^2)         =  4.932       Pythagorean to 4 s.f.
    stacked vs sum of individual deltas =  7.3e-03     additive in bf16

The absorption law also favours stacking: summing deltas raises |dW|/|W|, and
merge error scales as ~0.167/(|dW|/|W|), so a stacked delta is represented MORE
faithfully in bf16 than either component alone.

WHAT WOULD FALSIFY IT
---------------------
Each domain's score under a stack should stay close to that domain's score under
its own expert alone. Interference shows up as a drop from single -> pair ->
triple. The base-model column bounds how much of any retained score is just the
base model being competent to begin with.

Scoring matches the rest of the repo: per-question rubric where the eval data
supplies `expects` (financial_planning), else the good/bad term ratio (astral,
postgresql), greedy, with the stop_strings fix.

    uv run --env-file .env scripts/benchmark_stacked_experts.py
"""

import argparse
import itertools
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

from runtime.canon import REPO_ROOT, adapter_path  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.eval.eval_suite import LEGACY_TERMS, MODERN_TERMS  # noqa: E402
from runtime.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

# Default set is id_kron (the *_sweep_* adapters). --experts stock switches to the
# controlled stock-LoRA experts (r=8, alpha=128, scaling 16) so the stacking result
# can be checked for architecture-dependence.
EXPERTS_IDKRON = {
    "fin": "results/adapters/fin_sweep_a32",
    "ast": "results/adapters/astral_sweep_a64",
    "pg": "results/adapters/pg_sweep_a64",
}
EXPERTS_STOCK = {
    "fin": "results/adapters/ctl_lora_fin_a128",
    "ast": "results/adapters/ctl_lora_r8_a128",
    "pg": "results/adapters/ctl_lora_pg_a128",
}
# All-bf16 set. EXPERTS_STOCK is mixed-regime: ctl_lora_r8_a128 and
# ctl_lora_pg_a128 came from export_adapter.py (load_in_4bit=True) while
# ctl_lora_fin_a128 is bf16, so stacking them sums a bf16-trained delta with two
# 4-bit-trained ones into a bf16 base.
EXPERTS_BF16 = {
    "fin": "results/adapters/m2_financial_r8a128",
    "ast": "results/adapters/m2_astral_r8a128",
    "pg": "results/adapters/m2_postgresql_r8a128",
}
EXPERTS_V4 = {
    "fin": str(adapter_path("financial").relative_to(REPO_ROOT)),
    "ast": str(adapter_path("astral").relative_to(REPO_ROOT)),
    "pg": str(adapter_path("postgresql").relative_to(REPO_ROOT)),
}
EXPERTS = EXPERTS_V4  # default: the v4 clean completion set
# which expert "owns" each domain, for the retention comparison
OWNER = {"financial_planning": "fin", "astral": "ast", "postgresql": "pg"}

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
}


def score_one(q, txt, terms):
    if q.get("expects"):
        pats = q["expects"]
        return 100.0 * sum(1 for p in pats if re.search(p, txt)) / len(pats)
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
        txt = tok.decode(gen[0][ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        txt = re.split(r"#+\s*Question", txt)[0].strip().lower()
        out.append(score_one(q, txt, terms))
    return out  # per-question scores; callers average. Needed for bootstrap CIs:
    # scoring is greedy and deterministic (two independent runs returned
    # bit-identical means for every condition not involving the changed adapter),
    # so run-to-run variance is exactly zero and the ONLY uncertainty is which
    # questions were sampled. That is what bootstrapping over these resamples.


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--experts", choices=["idkron", "stock", "bf16", "v4"], default="v4",
                    help="v4 = clean completion v4 set (DEFAULT). bf16/stock/idkron are legacy.")
    ap.add_argument("--scale-mode", choices=["none", "sqrt", "linear"], default="none",
                    help="how to scale alpha when stacking K experts: none (1.0), sqrt (1/sqrt(K)), linear (1/K)")
    # The full power set is 2^N conditions and scores every domain in each. That
    # is mostly wasted: financial (+4.17pp solo) and postgres (+8.67pp) lack the
    # headroom to resolve a stacking effect at any n, while astral (+42.74pp) is
    # the only domain that can carry the question. --nested runs a single chain
    # of growing stack size scored on one domain, which is what the decay
    # question actually needs, at a fraction of the cost.
    ap.add_argument("--nested", action="store_true",
                    help="run base + a nested chain of growing stacks instead of the power set")
    ap.add_argument("--only-domain", default=None,
                    help="score only this domain (e.g. astral); implies its expert leads the chain")
    ap.add_argument("--out", default="results/stacked_experts.json")
    args = ap.parse_args()

    set_hard_vram_cap(args.vram_cap_gb)
    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()

    expert_set = {"stock": EXPERTS_STOCK, "bf16": EXPERTS_BF16, "v4": EXPERTS_V4}.get(args.experts, EXPERTS_IDKRON)
    print(f"expert set: {args.experts} (scale-mode: {args.scale_mode})")
    experts = {n: FoldableExpert.from_dir(REPO_ROOT / p, n) for n, p in expert_set.items()}
    engine = WeightFoldingEngine(model, experts.values(), keep_pristine=True)
    scored = {args.only_domain: DOMAINS[args.only_domain]} if args.only_domain else DOMAINS
    if args.only_domain and args.only_domain not in DOMAINS:
        raise SystemExit(f"unknown domain {args.only_domain}")
    qs = {
        d: [json.loads(x) for x in (REPO_ROOT / f).read_text().splitlines() if x.strip()]
        for d, (f, _) in scored.items()
    }

    names = list(expert_set)
    if args.nested:
        lead = OWNER.get(args.only_domain) if args.only_domain else names[0]
        if lead not in names:
            raise SystemExit(f"--only-domain {args.only_domain} has no expert in this set")
        rest = [n for n in names if n != lead]
        combos = [()] + [tuple([lead] + rest[:i]) for i in range(len(rest) + 1)]
    else:
        combos = [()]
        for r in (1, 2, 3):
            combos += list(itertools.combinations(names, r))

    nq = sum(len(v) for v in qs.values())
    print(f"{len(combos)} conditions x {len(scored)} domain(s) "
          f"({', '.join(scored)}), {nq} questions/condition, {args.max_new_tokens} tokens\n")
    results = {}
    per_question = {}
    for combo in combos:
        if combo:
            active_experts = [experts[n] for n in combo]
            k = len(active_experts)
            if args.scale_mode == "sqrt" and k > 1:
                scale_factor = 1.0 / (k ** 0.5)
            elif args.scale_mode == "linear" and k > 1:
                scale_factor = 1.0 / float(k)
            else:
                scale_factor = 1.0
            
            orig_scalings = [e.scaling for e in active_experts]
            for e, orig in zip(active_experts, orig_scalings):
                e.scaling = orig * scale_factor
            
            engine.activate_many(active_experts)
            
            for e, orig in zip(active_experts, orig_scalings):
                e.scaling = orig
        else:
            engine.restore()
        label = "+".join(combo) if combo else "base"
        per_q = {d: score(model, tok, qs[d], DOMAINS[d][1], args.max_new_tokens) for d in scored}
        row = {d: sum(v) / len(v) for d, v in per_q.items()}
        results[label] = row
        per_question[label] = per_q
        print(f"  {label:16s} " + "  ".join(f"{d[:4]}={row[d]:6.2f}%" for d in scored), flush=True)
    engine.restore()

    # ---- bootstrap CIs on ABSOLUTE pp deltas -------------------------------
    # The retention ratio (stacked-base)/(solo-base) is NOT reported: with a
    # small solo gain the denominator approaches zero and the ratio explodes
    # (measured: 247.7% on postgres off a +8.67pp gain, 160.0% on financial off
    # +4.17pp). Absolute pp deltas have no such failure mode.
    rng = random.Random(0)
    B = 10000
    print("\n" + "=" * 84)
    print(" Does stacking preserve each domain?  (stacked - solo, in pp, 95% bootstrap CI)")
    print("=" * 84)
    print(f" {'condition':14s} {'domain':<20s} {'solo':>8s} {'stacked':>9s} {'delta':>9s} {'95% CI':>18s}  verdict")
    boot = {}
    for label in results:
        if label == "base":
            continue
        for d in scored:
            owner = OWNER[d]
            if owner not in label.split("+"):
                continue
            a = per_question[owner][d]     # solo, per question
            b = per_question[label][d]     # stacked, per question
            n = len(a)
            diffs = []
            for _ in range(B):
                idx = [rng.randrange(n) for _ in range(n)]
                diffs.append(sum(b[i] - a[i] for i in idx) / n)
            diffs.sort()
            lo, hi = diffs[int(0.025 * B)], diffs[int(0.975 * B)]
            delta = sum(b) - sum(a)
            delta /= n
            sig = "RESOLVED" if (lo > 0 or hi < 0) else "not resolvable"
            boot[f"{label}|{d}"] = {
                "solo": sum(a) / n, "stacked": sum(b) / n, "delta_pp": delta,
                "ci95_lo": lo, "ci95_hi": hi, "resolved": bool(lo > 0 or hi < 0), "n": n,
            }
            print(f" {label:14s} {d:<20s} {sum(a) / n:7.2f}% {sum(b) / n:8.2f}% "
                  f"{delta:+8.2f}pp  [{lo:+6.2f},{hi:+6.2f}]  {sig}")
    nres = sum(1 for v in boot.values() if v["resolved"])
    print(f"\n {nres} of {len(boot)} stack/domain cells have a CI excluding zero.")
    print(" Cells whose CI spans 0 cannot distinguish stacking interference from")
    print(" question-sampling noise, regardless of how large the point estimate looks.")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"means": results, "per_question": per_question, "bootstrap": boot}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
