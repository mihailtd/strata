r"""Does activation-space geodesic distance actually predict real multi-expert stacking damage?

WHY THIS EXISTS
---------------
docs/DECISIONS.md §74 measured real activation-covariance AIRM distance between
6 domain experts (18.5% off-diagonal spread -- real signal, unlike weight-space's
flat 0.8%). Its only evidence that this distance means anything for stacking was
a `GROUND_TRUTH_STACKING` table of 5 hardcoded "measured damage" numbers inherited
from §59 (v4-era). Tracing those 5 numbers back to every stacking-result JSON still
on disk (`stacked_experts_v4_all_clean.json`, `_unscaled.json`, and everything under
results/benchmarks/*stack*) found: only ONE of the 5 (financial+postgresql, -8.50)
plausibly reconstructs (as the mean of both domains' bootstrap deltas, -8.515,
close enough to be the same number under a different rounding/run). The other four
do not match any on-disk artifact at all, and two of them (astral+duckdb,
postgresql+duckdb) reference a domain, duckdb, that no surviving stacking JSON ever
scored. That table cannot be trusted as "measured" any more -- it is unverifiable.

So instead of reusing it, this script BUILDS A FRESH ONE: real weight-level LoRA
stacking (`runtime.novel_peft.WeightFoldingEngine`, the same mechanism
`benchmarks/runtime/folding/benchmark_stacked_experts.py` already validated,
extended from 3 domains to the full 6-domain v7 fleet), scored with real
per-domain rubrics against real held-out eval prompts, with bootstrap CIs -- for
ALL C(6,2)=15 domain pairs, not 5 cherry-picked ones. Then correlates that real,
current n=15 damage matrix against the already-real, already-measured
`results/benchmarks/riemannian_activation_geodesics.json` pairwise AIRM distances
for the identical 6 domains.

METHOD
------
1. Load Qwen/Qwen3.5-4B once, fold in each of the 6 canonical v7 adapters as a
   `FoldableExpert` (no separate model load per condition -- same engine as
   `benchmark_stacked_experts.py`).
2. For each domain, load its real held-out eval set (data/<domain>/evaluation_data*.jsonl)
   and subsample to at most --max-questions-per-domain (fixed seed, disclosed) to
   bound cost; astral/postgresql/duckdb are subsampled, financial/python_modern/
   python_web are small enough to use in full.
3. Score BASE (no adapters), then each of the 6 SOLO experts on its own domain's
   eval set, then each of the 15 PAIRWISE stacks on BOTH member domains' eval sets.
   Scoring: `expects`-only domains (duckdb, financial) score the fraction of
   `expects` regexes matched; `expects`+`avoid` domains (python_modern, python_web)
   score (expects matched + avoid NOT matched) / total; astral/postgresql have no
   `expects` field and fall back to the repo's existing MODERN/LEGACY and
   POSTGRES_GOOD/BAD term-ratio rubrics.
4. Bootstrap 95% CI (B=10000) per (pair, domain) cell on the stacked-vs-solo delta,
   exactly as `benchmark_stacked_experts.py` already established.
5. Collapse each pair's two per-domain deltas into one "pair damage" number (mean
   of both directions) and correlate (Spearman) against that pair's real AIRM
   distance from `riemannian_activation_geodesics.json`, at n=15 -- not n=5.

RUN
    uv run --env-file .env python \
        experiments/factory/geometry/riemannian_metric/benchmark_stacking_geodesic_correlation.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

if torch.cuda.is_available():  # fla's device probe is @cache'd at import
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from runtime.canon import CANON, REPO_ROOT, adapter_path  # noqa: E402

sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "apps"))

from runtime.eval.eval_suite import LEGACY_TERMS, MODERN_TERMS  # noqa: E402
from runtime.gpu_preflight import ensure_gpu_exclusive  # noqa: E402
from runtime.novel_peft import FoldableExpert, WeightFoldingEngine  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "benchmarks"
GEODESICS_JSON = RESULTS_DIR / "riemannian_activation_geodesics.json"

DOMAINS = ["astral", "postgresql", "duckdb", "financial", "python_modern", "python_web"]

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

# (relative eval file, scoring mode). "terms" domains have no `expects` field in
# their eval data and are scored by regex-hit ratio against a fixed good/bad list.
# "expects" domains score the fraction of `expects` patterns matched. "expects_avoid"
# domains (the only real eval data that exists for python_modern/python_web is
# evaluation_data_disposition.jsonl, which also carries `avoid` patterns) score
# (expects matched + avoid NOT matched) / total.
EVAL_SPECS: dict[str, tuple[str, str]] = {
    "astral": ("data/astral/evaluation_data.jsonl", "terms"),
    "postgresql": ("data/postgresql/evaluation_data.jsonl", "terms"),
    "duckdb": ("data/duckdb/evaluation_data.jsonl", "expects"),
    "financial": ("data/financial_planning/evaluation_data.jsonl", "expects"),
    "python_modern": ("data/python_modern/evaluation_data_disposition.jsonl", "expects_avoid"),
    "python_web": ("data/python_web/evaluation_data_disposition.jsonl", "expects_avoid"),
}
TERMS: dict[str, tuple[list[str], list[str]]] = {
    "astral": (MODERN_TERMS, LEGACY_TERMS),
    "postgresql": (POSTGRES_GOOD, POSTGRES_BAD),
}


def load_domain_questions(domain: str, max_q: int, seed: int) -> list[dict]:
    rel_path, _mode = EVAL_SPECS[domain]
    f = REPO_ROOT / rel_path
    qs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
    if len(qs) > max_q:
        rng = random.Random(seed)
        qs = rng.sample(qs, max_q)
    return qs


def score_one(domain: str, q: dict, txt: str) -> float:
    mode = EVAL_SPECS[domain][1]
    if mode == "terms":
        good, bad = TERMS[domain]
        g = sum(len(re.findall(p, txt, re.IGNORECASE)) for p in good)
        b = sum(len(re.findall(p, txt, re.IGNORECASE)) for p in bad)
        return (g / max(1, g + b)) * 100.0 if (g + b) else 0.0
    if mode == "expects":
        pats = q["expects"]
        return 100.0 * sum(1 for p in pats if re.search(p, txt, re.IGNORECASE)) / len(pats)
    if mode == "expects_avoid":
        exp = q.get("expects", [])
        avo = q.get("avoid", [])
        total = len(exp) + len(avo)
        if not total:
            return 0.0
        hits = sum(1 for p in exp if re.search(p, txt, re.IGNORECASE))
        misses = sum(1 for p in avo if not re.search(p, txt, re.IGNORECASE))
        return 100.0 * (hits + misses) / total
    raise ValueError(f"unknown scoring mode {mode!r}")


@torch.no_grad()
def score(model, tok, domain: str, questions: list[dict], max_new_tokens: int) -> list[float]:
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
        out.append(score_one(domain, q, txt))
    return out


def bootstrap_delta(a: list[float], b: list[float], seed: int, B: int = 10000) -> dict[str, float]:
    """a = solo per-question scores, b = stacked per-question scores (same domain, same questions)."""
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    lo, hi = diffs[int(0.025 * B)], diffs[int(0.975 * B)]
    delta = (sum(b) - sum(a)) / n
    return {
        "solo": sum(a) / n, "stacked": sum(b) / n, "delta_pp": delta,
        "ci95_lo": lo, "ci95_hi": hi, "resolved": bool(lo > 0 or hi < 0), "n": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-questions-per-domain", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", default=None,
                     help="comma-separated domain:domain pairs to run (default: all 15)")
    ap.add_argument("--out", default="results/benchmarks/stacking_geodesic_correlation.json")
    args = ap.parse_args()

    ensure_gpu_exclusive()
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    ).eval()

    print(f"Loading {len(DOMAINS)} v7 domain adapters...")
    experts = {d: FoldableExpert.from_dir(adapter_path(d), d) for d in DOMAINS}
    engine = WeightFoldingEngine(model, experts.values(), keep_pristine=True)

    print("Loading real held-out eval questions per domain (subsampled, fixed seed)...")
    questions = {d: load_domain_questions(d, args.max_questions_per_domain, args.seed) for d in DOMAINS}
    for d in DOMAINS:
        print(f"  {d:16s} {len(questions[d]):3d} questions  ({EVAL_SPECS[d][0]})")

    if args.pairs:
        pairs = [tuple(p.split(":")) for p in args.pairs.split(",")]
    else:
        pairs = list(itertools.combinations(DOMAINS, 2))
    print(f"\n{len(pairs)} pairs x up to 2 domains scored each, "
          f"{sum(len(v) for v in questions.values())} questions/condition, "
          f"{args.max_new_tokens} max_new_tokens\n")

    per_question: dict[str, dict[str, list[float]]] = {}

    def run_condition(label: str, active: list[str], score_domains: list[str]) -> None:
        if active:
            engine.activate_many([experts[d] for d in active])
        else:
            engine.restore()
        per_question[label] = {d: score(model, tok, d, questions[d], args.max_new_tokens) for d in score_domains}
        row = {d: sum(v) / len(v) for d, v in per_question[label].items()}
        print(f"  {label:24s} " + "  ".join(f"{d[:4]}={row[d]:6.2f}%" for d in score_domains), flush=True)

    print("[base]")
    run_condition("base", [], DOMAINS)
    engine.restore()

    print("[solo]")
    for d in DOMAINS:
        run_condition(d, [d], [d])
        engine.restore()

    print("[pairwise]")
    for d1, d2 in pairs:
        run_condition(f"{d1}+{d2}", [d1, d2], [d1, d2])
        engine.restore()

    # ---- bootstrap CIs + pair damage --------------------------------------
    print("\n" + "=" * 90)
    print(" Real stacking damage: stacked - solo, per domain per pair (95% bootstrap CI, B=10000)")
    print("=" * 90)
    pair_results: dict[str, Any] = {}
    for d1, d2 in pairs:
        label = f"{d1}+{d2}"
        cell1 = bootstrap_delta(per_question[d1][d1], per_question[label][d1], seed=args.seed)
        cell2 = bootstrap_delta(per_question[d2][d2], per_question[label][d2], seed=args.seed + 1)
        pair_damage = (cell1["delta_pp"] + cell2["delta_pp"]) / 2.0
        pair_results[label] = {d1: cell1, d2: cell2, "pair_damage_pp": pair_damage}
        print(f" {label:24s} {d1}={cell1['delta_pp']:+7.2f}pp [{cell1['ci95_lo']:+6.2f},{cell1['ci95_hi']:+6.2f}]  "
              f"{d2}={cell2['delta_pp']:+7.2f}pp [{cell2['ci95_lo']:+6.2f},{cell2['ci95_hi']:+6.2f}]  "
              f"mean={pair_damage:+7.2f}pp")

    # ---- correlate against the already-real activation-geodesic distances -
    print("\n" + "=" * 90)
    print(" Correlation: real d_R (activation AIRM) vs real stacking damage")
    print("=" * 90)
    geo = json.loads(GEODESICS_JSON.read_text())
    geo_domains = geo["domains"]
    airm = geo["airm"]

    def d_r(d1: str, d2: str) -> float:
        return airm[geo_domains.index(d1)][geo_domains.index(d2)]

    gt_pairs, damages, dists = [], [], []
    for d1, d2 in pairs:
        label = f"{d1}+{d2}"
        damage = pair_results[label]["pair_damage_pp"]
        dist = d_r(d1, d2)
        gt_pairs.append((d1, d2, damage, dist))
        damages.append(damage)
        dists.append(dist)

    print(f"{'Pair':<28} | {'Pair damage (pp)':>18} | {'Activation d_R':>15}")
    print("-" * 68)
    for d1, d2, damage, dist in gt_pairs:
        print(f"{f'{d1} + {d2}':<28} | {damage:>+17.2f} | {dist:>15.4f}")

    from scipy.stats import spearmanr
    try:
        rho, p = spearmanr(damages, dists, alternative="two-sided")
    except TypeError:
        rho, p = spearmanr(damages, dists)
    n = len(pairs)
    print(f"\nSpearman rho(damage, d_R) = {rho:+.4f}  (p={p:.4f}, n={n})")
    print("Interpretation: closer domains (smaller d_R) synergizing (positive damage)")
    print("would show rho < 0. rho > 0 means closer domains interfere MORE, the")
    print("opposite of the intuitive 'similar domains are safer to stack' hypothesis.")

    elapsed = time.time() - t0
    artifact = {
        "config": CANON.stamp() | {
            "model": args.model_name, "max_new_tokens": args.max_new_tokens,
            "max_questions_per_domain": args.max_questions_per_domain, "seed": args.seed,
        },
        "domains": DOMAINS,
        "question_counts": {d: len(questions[d]) for d in DOMAINS},
        "means": {label: {d: sum(v) / len(v) for d, v in scores.items()} for label, scores in per_question.items()},
        "per_question": per_question,
        "pair_damage": pair_results,
        "geodesic_source": str(GEODESICS_JSON.relative_to(REPO_ROOT)),
        "correlation": {"pairs": gt_pairs, "spearman_rho": rho, "p_value": float(p), "n": n},
        "elapsed_s": elapsed,
    }
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, default=str))
    print(f"\n{elapsed:.0f}s total. Wrote {out}")


if __name__ == "__main__":
    main()
