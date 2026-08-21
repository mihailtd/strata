"""Does folding an adapter into the base weights change what it answers?

Folding is only numerically equivalent to wrapping, not bit-identical: the
merged weight W0 + dW is rounded to bf16, whereas the wrapper keeps dW in its
own tensor and adds the delta after the base matmul. Measured on last-position
logits that shows up as max|dlogit| = 0.156 against an adapter effect of 6.0 --
small, but not zero, and logit fingerprints on four prompts are far too coarse
to conclude it doesn't matter. This runs the project's real eval over all 20
questions per domain and reports whether the folded model actually answers
differently.

The headline number here is EXACT STRING MATCH between the wrapped and folded
arms, not the score. Decoding is greedy, so both arms are a deterministic
function of (weights, prompt): if the strings match, the score is identical by
construction and folding is behaviourally free. Any gap in score is only
meaningful once the strings are known to have diverged.

Scoring is per-question rubric coverage where the eval data supplies an
`expects` list (financial_planning), else the good/bad term ratio (astral,
postgresql).

Three arms per domain, one model load, same weights throughout:
    base      engine restored to pristine W0
    wrapped   NovelLoraLinear wrappers, the normal inference path
    folded    W_live = W0 + scaling * (U @ V), wrappers removed

NOTE ON COMPARABILITY: these run in bf16, because folding a bf16 delta into a
4-bit packed weight is not possible. The project's historical adherence tables
were measured in 4-bit NF4 and these numbers are NOT comparable to them. The
comparison that is valid is the one this script makes: arms against each other,
same dtype, same process.

    uv run --env-file .env scripts/evaluate_folded_vs_wrapped.py
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime.canon import REPO_ROOT, adapter_path  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.append(str(REPO_ROOT))

from runtime.eval.eval_suite import LEGACY_TERMS, MODERN_TERMS  # noqa: E402
from runtime.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    load_novel_adapter,
    set_hard_vram_cap,
    unwrap_novel_lora,
)

# Carried over from the deleted scripts/evaluate_financial_planning.py, whose
# folding path was unrunnable but whose term lists are the domain's ground truth.
FINANCIAL_TERMS = [
    r"\bmoney script\b",
    r"\bmoney scripts\b",
    r"\bloss aversion\b",
    r"\brisk capacity\b",
    r"\brisk tolerance\b",
    r"\brisk perception\b",
    r"\bbehavioral finance\b",
    r"\bfinancial therapy\b",
    r"\bfinancial psychology\b",
    r"\bactive listening\b",
    r"\bmotivational interviewing\b",
    r"\bmental accounting\b",
    r"\bsequence of returns\b",
    r"\brecency bias\b",
    r"\bconfirmation bias\b",
    r"\bfinancial flashpoint\b",
]
GENERAL_FILLER_TERMS = [
    r"\bconsult a professional\b",
    r"\bseek advice\b",
    r"\bI am an AI\b",
    r"\bgeneral information\b",
    r"\bas an AI language model\b",
]
POSTGRES_MODERN_TERMS = [
    r"\bpgvector\b",
    r"\bhnsw\b",
    r"\bivfflat\b",
    r"\bembeddings?\b",
    r"cosine (distance|similarity)",
    r"semantic search",
    r"unified (platform|database)",
    r"vector (column|index|extension)",
]
POSTGRES_LEGACY_TERMS = [
    r"\bpinecone\b",
    r"\bweaviate\b",
    r"\bmilvus\b",
    r"\bqdrant\b",
    r"\bchroma(db)?\b",
    r"\bfaiss\b",
    r"(separate|dedicated|standalone) vector (database|store)",
    r"\belasticsearch\b",
]

DOMAINS = {
    "astral": {
        "adapter": "results/adapters/m2_astral_r8a128",
        "questions": "data/astral/evaluation_data.jsonl",
        "good": MODERN_TERMS,
        "bad": LEGACY_TERMS,
    },
    "postgresql": {
        "adapter": "results/adapters/m2_postgresql_r8a128",
        "questions": "data/postgresql/evaluation_data.jsonl",
        "good": POSTGRES_MODERN_TERMS,
        "bad": POSTGRES_LEGACY_TERMS,
    },
    "financial_planning": {
        "adapter": "results/adapters/m2_financial_r8a128",
        "questions": "data/financial_planning/evaluation_data.jsonl",
        "good": FINANCIAL_TERMS,
        "bad": GENERAL_FILLER_TERMS,
    },
    "duckdb": {
        "adapter": str(adapter_path("duckdb").relative_to(REPO_ROOT)),
        "questions": "data/duckdb/evaluation_data.jsonl",
        "good": [],
        "bad": [],
    },
}


def count_matches(text: str, patterns: list[str]) -> int:
    low = text.lower()
    return sum(len(re.findall(p, low)) for p in patterns)


def score_question(q: dict, text: str, good: list[str], bad: list[str]) -> dict:
    """Rubric coverage when the question carries one, else the good/bad ratio.

    A question with an `expects` list is scored on how many of ITS OWN required
    concepts the answer names. That is immune to the two ways the ratio metric
    was gameable -- verbosity (more text, more hits) and repetition (the same
    term counted N times) -- and it is the only option for a domain with no
    genuine opposing set. See scripts/old/build_financial_planning_dataset.py.
    """
    low = text.lower()
    if q.get("expects"):
        pats = q["expects"]
        hit = sum(1 for p in pats if re.search(p, low))
        return {"score_pct": 100.0 * hit / len(pats), "covered": hit, "expected": len(pats)}
    g, b = count_matches(text, good), count_matches(text, bad)
    return {"score_pct": (g / max(1, g + b)) * 100.0 if (g + b) else 0.0, "good_hits": g, "bad_hits": b}


def run_arm(model, tokenizer, questions, good, bad, max_new_tokens):
    """Greedy generation + scoring, matching the project's eval suite."""
    out = []
    for q in questions:
        ids = tokenizer(f"### Question:\n{q['prompt']}\n\n### Answer:\n", return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            gen = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                # Without this the base model runs past its answer and invents a
                # new "### Question:" block, which then gets scored: 13-18/20 base
                # answers did so, supplying 57% of base's financial term hits.
                # Adapters learned to stop, so they were losing to an artifact.
                stop_strings=["### Question"],
                tokenizer=tokenizer,
            )
        dt = time.perf_counter() - t0
        text = tokenizer.decode(gen[0][ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()
        text = re.split(r"#+\s*Question", text)[0].strip()
        out.append({"id": q.get("id"), "response": text, "gen_time_s": dt, **score_question(q, text, good, bad)})
    return out


def summarise(rows):
    s = {"score_pct": sum(r["score_pct"] for r in rows) / len(rows)}
    for k in ("good_hits", "bad_hits", "covered", "expected"):
        if k in rows[0]:
            s[k] = sum(r[k] for r in rows)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS))
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument(
        "--adapter",
        default=None,
        help="Override the adapter dir for the single domain under test (A/B two adapters "
        "on the same harness). Only valid with exactly one --domains entry.",
    )
    ap.add_argument("--out", default="results/folded_vs_wrapped_eval.json")
    args = ap.parse_args()

    if args.adapter:
        if len(args.domains) != 1:
            ap.error("--adapter requires exactly one --domains entry")
        DOMAINS[args.domains[0]] = {**DOMAINS[args.domains[0]], "adapter": args.adapter}

    set_hard_vram_cap(args.vram_cap_gb)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    experts = {d: FoldableExpert.from_dir(REPO_ROOT / DOMAINS[d]["adapter"], name=d) for d in args.domains}
    for e in experts.values():
        print(f"  {e}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nLoading {args.model_name} in {dtype} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    model.eval()

    engine = WeightFoldingEngine(model, experts.values())
    print(f"Engine: {len(engine.slots)} tensors, pristine {engine.pristine_bytes / 1e9:.2f} GB\n")

    results = {"model": args.model_name, "dtype": str(dtype), "max_new_tokens": args.max_new_tokens, "domains": {}}

    for dom in args.domains:
        cfg = DOMAINS[dom]
        questions = [json.loads(x) for x in (REPO_ROOT / cfg["questions"]).read_text().splitlines() if x.strip()]
        print("=" * 72)
        print(f" {dom}  ({len(questions)} questions)")
        print("=" * 72)

        engine.restore()
        print("  base    ...", end="", flush=True)
        base = run_arm(model, tokenizer, questions, cfg["good"], cfg["bad"], args.max_new_tokens)
        print(f" {summarise(base)['score_pct']:.2f}%")

        adapter_path = REPO_ROOT / cfg["adapter"]
        if (adapter_path / "adapter_model.safetensors").exists():
            from peft import PeftModel

            pm = PeftModel.from_pretrained(model, str(adapter_path))
            pm.eval()
            print("  wrapped ...", end="", flush=True)
            wrapped = run_arm(pm, tokenizer, questions, cfg["good"], cfg["bad"], args.max_new_tokens)
            print(f" {summarise(wrapped)['score_pct']:.2f}%")
            pm.unload()
        else:
            load_novel_adapter(model, adapter_path)
            print("  wrapped ...", end="", flush=True)
            wrapped = run_arm(model, tokenizer, questions, cfg["good"], cfg["bad"], args.max_new_tokens)
            print(f" {summarise(wrapped)['score_pct']:.2f}%")
            unwrap_novel_lora(model)

        engine.activate(experts[dom])
        print("  folded  ...", end="", flush=True)
        folded = run_arm(model, tokenizer, questions, cfg["good"], cfg["bad"], args.max_new_tokens)
        print(f" {summarise(folded)['score_pct']:.2f}%")
        engine.restore()

        exact = sum(1 for a, b in zip(wrapped, folded, strict=True) if a["response"] == b["response"])
        # where they diverge, how far in do they first differ?
        prefixes = []
        for a, b in zip(wrapped, folded, strict=True):
            if a["response"] != b["response"]:
                i = next((j for j, (x, y) in enumerate(zip(a["response"], b["response"], strict=False)) if x != y), 0)
                prefixes.append(i)

        s_base, s_wrap, s_fold = summarise(base), summarise(wrapped), summarise(folded)
        print(
            f"\n  score       base {s_base['score_pct']:6.2f}%   "
            f"wrapped {s_wrap['score_pct']:6.2f}%   folded {s_fold['score_pct']:6.2f}%"
        )
        print(f"  folded - wrapped: {s_fold['score_pct'] - s_wrap['score_pct']:+.2f} pp")
        print(f"  exact string match wrapped vs folded: {exact}/{len(questions)}")
        if prefixes:
            print(
                f"  on the {len(prefixes)} that diverged, first difference at char "
                f"{min(prefixes)}-{max(prefixes)} (median {sorted(prefixes)[len(prefixes) // 2]})"
            )
        print()

        results["domains"][dom] = {
            "questions": len(questions),
            "base": s_base,
            "wrapped": s_wrap,
            "folded": s_fold,
            "delta_folded_minus_wrapped_pp": s_fold["score_pct"] - s_wrap["score_pct"],
            "exact_match": exact,
            "divergence_first_char": prefixes,
            "rows": {"base": base, "wrapped": wrapped, "folded": folded},
        }

    print("=" * 72)
    print(" Summary")
    print("=" * 72)
    print(f" {'domain':20s} {'base':>8s} {'wrapped':>9s} {'folded':>8s} {'delta':>8s} {'exact':>8s}")
    for dom, r in results["domains"].items():
        print(
            f" {dom:20s} {r['base']['score_pct']:7.2f}% {r['wrapped']['score_pct']:8.2f}% "
            f"{r['folded']['score_pct']:7.2f}% {r['delta_folded_minus_wrapped_pp']:+7.2f}pp "
            f"{r['exact_match']:4d}/{r['questions']:<3d}"
        )

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
