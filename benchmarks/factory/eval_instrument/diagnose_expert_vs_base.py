"""Per-question base-vs-expert diagnosis: is the ADAPTER inert, or is the EVAL blind?

WHY THIS EXISTS
---------------
The stacking matrix showed the financial expert scoring 83.33% on its own domain
and the BASE MODEL scoring 83.33% — an adapter effect of exactly +0.00pp, against
astral's +44.44pp and postgresql's +24.08pp. An aggregate cannot distinguish the
two explanations:

    (1) the adapter is inert       — training did not take
    (2) the eval is blind          — no headroom, or the rubric gives away its
                                     own answers, so base already scores high

They need opposite fixes (retrain vs rewrite the eval), so guessing is expensive.
This runs both arms question-by-question and reports DISCRIMINATION: how many
questions actually separate base from expert, and in which direction.

WHAT IT PRINTS PER QUESTION
    base / expert score, whether the text changed at all, and the rubric terms
    each arm hit — so a "no difference" can be read as "same answer" (adapter
    inert) versus "different answer, same score" (rubric insensitive).

Scoring is the repo's own `score_question`, imported rather than redefined.

    uv run --env-file .env python \
        benchmarks/factory/eval_instrument/diagnose_expert_vs_base.py --domain financial_planning
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

if torch.cuda.is_available():  # fla's device probe is @cache'd at import
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "runtime" / "folding"))

from evaluate_folded_vs_wrapped import DOMAINS, score_question  # noqa: E402

from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)


def generate(model, tok, prompt: str, max_new_tokens: int) -> str:
    ids = tok(f"### Question:\n{prompt}\n\n### Answer:\n", return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            stop_strings=["### Question"],
            tokenizer=tok,
        )
    txt = tok.decode(out[0][ids["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    return re.split(r"#+\s*Question", txt)[0].strip()


def hits(q: dict, text: str) -> list[str]:
    """Which of this question's own rubric terms the answer matched."""
    if not q.get("expects"):
        return []
    low = text.lower()
    return [p for p in q["expects"] if re.search(p, low)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="financial_planning", choices=sorted(DOMAINS))
    ap.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--vram-cap-gb", type=float, default=22.0)
    ap.add_argument("--questions", default=None, help="override the domain's eval file")
    ap.add_argument("--adapter", default=None, help="override the domain's adapter dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = DOMAINS[args.domain]
    qpath = REPO_ROOT / (args.questions or cfg["questions"])
    questions = [json.loads(x) for x in qpath.read_text().splitlines() if x.strip()]

    set_hard_vram_cap(args.vram_cap_gb)
    print("=" * 100)
    print(f"  BASE vs EXPERT, PER QUESTION — {args.domain}  (n={len(questions)}, {qpath.name})")
    print("=" * 100)

    tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    adapter_rel = args.adapter or cfg["adapter"]
    expert = FoldableExpert.from_dir(REPO_ROOT / adapter_rel, args.domain)
    print(f"  adapter: {adapter_rel}")
    engine = WeightFoldingEngine(model, [expert], keep_pristine=True)

    rows = []
    print(f"\n  {'id':<10}{'base':>7}{'expert':>8}{'delta':>8}  {'text':<9} rubric hits (base -> expert)")
    print("  " + "-" * 96)

    for q in questions:
        engine.restore()
        base_txt = generate(model, tok, q["prompt"], args.max_new_tokens)
        engine.activate(expert)
        exp_txt = generate(model, tok, q["prompt"], args.max_new_tokens)

        b = score_question(q, base_txt, cfg["good"], cfg["bad"])
        e = score_question(q, exp_txt, cfg["good"], cfg["bad"])
        hb, he = hits(q, base_txt), hits(q, exp_txt)
        changed = base_txt != exp_txt
        d = e["score_pct"] - b["score_pct"]
        rows.append({
            "id": q.get("id"), "category": q.get("category"),
            "base_score": b["score_pct"], "expert_score": e["score_pct"], "delta": d,
            "text_changed": changed,
            "base_hits": hb, "expert_hits": he,
            "n_expects": len(q.get("expects", [])),
            "base_response": base_txt, "expert_response": exp_txt,
        })
        print(f"  {str(q.get('id')):<10}{b['score_pct']:>7.1f}{e['score_pct']:>8.1f}{d:>+8.1f}"
              f"  {'CHANGED' if changed else 'same':<9} {len(hb)}/{len(q.get('expects', []))}"
              f" -> {len(he)}/{len(q.get('expects', []))}")

    n = len(rows)
    changed = sum(r["text_changed"] for r in rows)
    moved = sum(abs(r["delta"]) > 1e-9 for r in rows)
    better = sum(r["delta"] > 0 for r in rows)
    worse = sum(r["delta"] < 0 for r in rows)
    mean_b = sum(r["base_score"] for r in rows) / n
    mean_e = sum(r["expert_score"] for r in rows) / n

    print("\n" + "=" * 100)
    print("  DIAGNOSIS")
    print("=" * 100)
    print(f"  base mean {mean_b:.2f}%   expert mean {mean_e:.2f}%   delta {mean_e - mean_b:+.2f}pp")
    print(f"  answers that CHANGED text:      {changed}/{n}")
    print(f"  questions that MOVED the score: {moved}/{n}   ({better} up, {worse} down)")
    print()
    if changed == 0:
        print("  => ADAPTER IS INERT. It does not change the text at all. Fix training, not the eval.")
    elif moved == 0:
        print("  => EVAL IS BLIND. The adapter rewrites every answer and the rubric scores")
        print("     none of it differently. Fix the eval instrument, not the adapter.")
    else:
        print(f"  => PARTIAL. The adapter changes {changed}/{n} answers but only {moved} register.")
        print("     Both the eval's sensitivity and the adapter's strength are in question.")

    ceiling = sum(1 for r in rows if r["base_score"] >= 99.9)
    print(f"\n  questions where BASE already scores 100%: {ceiling}/{n}"
          f"  (no headroom — these can never show an adapter effect)")

    out = REPO_ROOT / (args.out or f"results/diagnose_{args.domain}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "domain": args.domain, "questions_file": str(qpath.relative_to(REPO_ROOT)),
        "adapter": adapter_rel,
        "base_mean": mean_b, "expert_mean": mean_e,
        "n_text_changed": changed, "n_score_moved": moved,
        "n_base_at_ceiling": ceiling, "rows": rows,
    }, indent=2))
    print(f"\n  Saved -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
