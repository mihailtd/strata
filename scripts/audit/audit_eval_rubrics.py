"""Static audit of the eval instruments: can these questions detect an adapter at all?

THE BUG THIS GENERALISES
------------------------
`data/financial_planning/evaluation_data.jsonl` scores answers by whether they
contain the terms in each question's `expects` list. 68.2% of those terms appear
VERBATIM IN THE QUESTION, and 9 of 20 questions give away every term they ask
for. So the rubric largely measures whether the model echoes the prompt's own
vocabulary — which any instruction-following model does. The base model scored
83.33% and the domain expert scored 83.33%: an adapter effect of exactly +0.00pp,
against astral's +44.44pp. The instrument was blind, and it took a stacking matrix
to notice.

That is a design flaw with a cheap static signature, so it should never again need
a GPU or a benchmark to catch. This checks every domain, offline, in a second.

WHAT IT MEASURES

  giveaway (rubric domains)   fraction of `expects` patterns that already match
                              the question text. High = the eval rewards echoing.

  giveaway (ratio domains)    fraction of questions whose prompt already contains
                              a "good" term. Same failure in the other metric:
                              astral/postgresql score good/(good+bad) term hits,
                              so a prompt naming the good term hands over points.

  leading questions           prompts that state the expected answer's polarity
                              ("Should ... without ...?"). Not a defect by itself
                              — anti-pattern probes are deliberately leading — but
                              an RLHF'd base model answers these well by reflex,
                              so they are weak discriminators unless paired with a
                              rubric that demands the specific concept.

    uv run python scripts/audit/audit_eval_rubrics.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from gnn_experiment.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "runtime" / "folding"))

from evaluate_folded_vs_wrapped import DOMAINS  # noqa: E402

LEADING = re.compile(
    r"\b(should|is it|does it|can you just|isn't it|only|solely|strictly|without|alone)\b", re.I
)


def audit_domain(name: str, cfg: dict, questions_file: str | None = None) -> dict:
    target = questions_file or cfg["questions"]
    path = Path(target) if Path(target).is_absolute() else REPO_ROOT / target
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    rubric = [r for r in rows if r.get("expects")]

    try:
        rel_path = str(path.relative_to(REPO_ROOT))
    except ValueError:
        rel_path = str(path)

    out: dict = {"file": rel_path, "n": len(rows),
                 "n_with_rubric": len(rubric)}

    if rubric:
        tot = given = 0
        full = []
        for r in rubric:
            q = r["prompt"]
            g = sum(1 for p in r["expects"] if re.search(p, q, re.I))
            tot += len(r["expects"])
            given += g
            if g == len(r["expects"]):
                full.append(r.get("id"))
        out["rubric_terms_total"] = tot
        out["rubric_terms_in_question"] = given
        out["giveaway_pct"] = 100.0 * given / max(1, tot)
        out["fully_given_away_ids"] = full
        out["n_fully_given_away"] = len(full)

    # ratio-scored domains: does the prompt already contain a "good" term?
    good = cfg.get("good") or []
    if good and not rubric:
        n_pre = sum(1 for r in rows if any(re.search(p, r["prompt"], re.I) for p in good))
        out["prompts_containing_a_good_term"] = n_pre
        out["giveaway_pct"] = 100.0 * n_pre / max(1, len(rows))

    out["n_leading"] = sum(1 for r in rows if LEADING.search(r["prompt"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=list(DOMAINS))
    ap.add_argument("--questions", default=None, help="override file (single domain only)")
    ap.add_argument("--out", default="results/eval_rubric_audit.json")
    args = ap.parse_args()

    print("=" * 100)
    print("  EVAL INSTRUMENT AUDIT — can these questions detect an adapter?")
    print("=" * 100)
    print(f"\n  {'domain':<22}{'n':>5}{'rubric':>8}{'giveaway':>11}{'all-given':>11}{'leading':>9}"
          "   metric")
    print("  " + "-" * 96)

    report = {}
    for d in args.domains:
        r = audit_domain(d, DOMAINS[d], args.questions if len(args.domains) == 1 else None)
        report[d] = r
        metric = "rubric coverage" if r["n_with_rubric"] else "good/bad term ratio"
        gp = r.get("giveaway_pct")
        flag = " ⚠️" if gp is not None and gp >= 50 else ""
        print(f"  {d:<22}{r['n']:>5}{r['n_with_rubric']:>8}"
              f"{(f'{gp:.1f}%' if gp is not None else '—'):>11}"
              f"{r.get('n_fully_given_away', '—'):>11}{r['n_leading']:>9}   {metric}{flag}")

    print("\n  giveaway  = rubric terms already present in the question (rubric domains),")
    print("              or prompts already containing a scored 'good' term (ratio domains).")
    print("  ⚠️ at >=50%: the instrument substantially rewards echoing the prompt.")

    worst = max(report.items(), key=lambda kv: kv[1].get("giveaway_pct") or 0)
    if (worst[1].get("giveaway_pct") or 0) >= 50:
        print(f"\n  WORST: {worst[0]} at {worst[1]['giveaway_pct']:.1f}%. "
              f"{worst[1].get('n_fully_given_away', 0)} questions give away EVERY term they ask for:")
        print(f"    {', '.join(str(x) for x in worst[1].get('fully_given_away_ids', []))}")

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
