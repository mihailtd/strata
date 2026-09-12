"""Add APPLIED classification examples to the financial planning training set.

THE DEFECT THIS FIXES
---------------------
The financial expert measured +0.00pp over base on its own eval. Two audits
found two separate causes, and this script addresses the second:

  1. THE EVAL WAS BLIND. 68.2% of its rubric terms appeared verbatim in the
     question and base sat at 100% on 14 of 20 items. Fixed separately by
     `data/financial_planning/evaluation_data_v2.jsonl` (giveaway 0.0%).

  2. THE CORPUS TEACHES RECITATION, NOT APPLICATION. Measured over the 304
     training records:

         "How ..." questions          235
         "What ..." questions          64
         scenario / vignette            5   (1.6%)
         asking to CLASSIFY a client    0   (0.0%)

     So the model learns to define a money script and has never once been asked
     to identify one. On the fixed eval that shows up exactly as predicted: asked
     the definitional question ("what are those inherited beliefs called?") it
     answers "money scripts" correctly, but shown a client vignette it invents
     plausible framework names -- "Financial Identity Framework",
     "Financial Enmeshment", "autobiographical memory" -- instead of applying the
     taxonomy it was trained on. A reciter, not a diagnostician.

WHY GENERATED RATHER THAN PIPELINE-EXTRACTED
--------------------------------------------
`build_financial_planning_dataset.py` records what happened last time training
data was produced by a single hardcoded template: 940 records with **1 distinct
question prefix**, and an adapter that scored BELOW base (60% -> 35%). Question
diversity, not adapter architecture, was the problem.

So diversity is the hard requirement here, and it is asserted at the end of this
script rather than hoped for: distinct question prefixes, distinct openings, and
no near-duplicate of any eval scenario. Labels are guaranteed correct by
construction, which an LLM pipeline cannot promise for a taxonomy this specific.

    uv run python scripts/corpus/build_financial_applied_examples.py
"""

from __future__ import annotations

import argparse
import json
import random
import re

from runtime_common.canon import REPO_ROOT  # noqa: E402

# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
FIN = REPO_ROOT / "apps" / "factory" / "data" / "financial_planning"

# --- The taxonomy the corpus already teaches (Klontz money scripts) -----------
# Symptom pools are deliberately disjoint from the v2 eval vignettes.
SCRIPTS = {
    "money avoidance": {
        "belief": "that money is bad, or that they do not deserve it",
        "symptoms": [
            "ignores account statements for months at a time",
            "gives away bonuses almost as soon as they arrive",
            "describes wealthy people as fundamentally greedy",
            "leaves an inheritance untouched in a current account out of discomfort",
            "underprices their own freelance work despite steady demand",
            "feels a pang of guilt after any purchase above the essentials",
            "avoids opening the pension portal even when prompted",
            "turns down a promotion because the pay rise 'feels wrong'",
        ],
        "action": "explore where the belief that money is corrupting came from, "
        "without challenging it directly in the first session",
    },
    "money status": {
        "belief": "that self-worth equals net worth",
        "symptoms": [
            "upgrades their car every eighteen months regardless of the balance sheet",
            "will not discuss downsizing because of how it would look to colleagues",
            "keeps a private-school place they cannot fund without borrowing",
            "measures a good year purely by whether they out-earned their brother",
            "buys the most expensive option available as a default",
            "conceals a redundancy from friends and keeps up the same spending",
            "insists on picking up every restaurant bill",
            "treats a modest portfolio as a personal failing rather than a stage",
        ],
        "action": "separate identity from balance sheet gently, and anchor goals in values rather than comparison",
    },
    "money worship": {
        "belief": "that more money will solve their problems",
        "symptoms": [
            "keeps moving the retirement number upward each time it is reached",
            "works through every holiday convinced the next bonus is the one that matters",
            "believes a larger portfolio would repair a strained marriage",
            "cannot articulate what the money is actually for",
            "treats every windfall as insufficient within weeks",
            "postpones every enjoyment until an amount that keeps receding",
            "attributes all unhappiness to not having earned enough yet",
            "chases higher-yield products well past their stated risk comfort",
        ],
        "action": "help them define sufficiency explicitly, and connect the plan to what the money is meant to buy",
    },
    "money vigilance": {
        "belief": "that money must be watched closely and never discussed",
        "symptoms": [
            "will not disclose salary even to a long-term partner",
            "keeps far more cash than the plan requires and cannot deploy it",
            "feels acute discomfort when a family member raises finances",
            "checks balances several times a day without acting on them",
            "declines to spend on a long-planned trip despite ample funding",
            "keeps separate accounts hidden from a spouse for security",
            "treats any debt, including a low-rate mortgage, as an emergency",
            "cannot enjoy a purchase without recalculating the buffer afterwards",
        ],
        "action": "acknowledge the protective function of the vigilance before working on the discomfort it creates",
    },
}

# --- Question framings. The last corpus died of having exactly one. ----------
FRAMINGS = [
    "A client {sym}. Which money script is this, and how would you work with it?",
    "{Prof}, {age}, {sym}. Identify the belief pattern and your first move as their planner.",
    "During a review a client {sym}. What does that indicate, and what would you do next?",
    "One of your clients {sym}. Name the underlying script and the appropriate response.",
    "A prospect says they {sym}. What would you note, and why?",
    "{Prof} in their {age}s {sym}. Classify what you are seeing and outline how to approach it.",
    "You notice a client {sym}. What belief is likely driving this, and how should it be handled?",
    "A partner reports they {sym}. What would you explore, and under what heading?",
]

PROFESSIONS = [
    "A surgeon",
    "A software engineer",
    "A schoolteacher",
    "A restaurant owner",
    "A civil servant",
    "A freelance designer",
    "A pharmacist",
    "A logistics manager",
    "A dentist",
    "A university lecturer",
    "A retired engineer",
    "A sales director",
]
AGES = ["34", "41", "47", "52", "58", "63", "29", "45"]

# --- Formative-event (flashpoint) items --------------------------------------
FLASHPOINTS = [
    (
        "watched a parent lose a business when they were eleven",
        "cannot tolerate any business debt, even well-covered borrowing",
    ),
    (
        "remembers bailiffs at the door in their early teens",
        "keeps eighteen months of expenses in cash and will not invest it",
    ),
    (
        "was told repeatedly as a child that the family could not afford anything",
        "underspends badly relative to a very comfortable position",
    ),
    (
        "saw a grandparent's savings wiped out by a bank failure",
        "distrusts every institution and splits deposits across six providers",
    ),
    (
        "grew up with a parent who hid purchases from the other",
        "keeps a private account and has never mentioned it to their spouse",
    ),
    (
        "had university fees withdrawn abruptly after a family argument",
        "over-funds their own children's education at the cost of their pension",
    ),
]

# --- Bias items, using only concepts the corpus actually teaches --------------
BIASES = [
    (
        "loss aversion",
        [
            "demands you sell everything after a single bad quarter",
            "refuses to rebalance because it means realising a paper loss",
            "wants to hold cash until markets 'feel safe again'",
        ],
        "the discomfort of a loss is felt far more strongly than an equivalent gain",
        "reconnect the decision to the plan's horizon and agreed policy, rather than to the current drawdown",
    ),
    (
        "mental accounting",
        [
            "keeps a car fund in a low-interest account while servicing an expensive card balance",
            "treats a tax refund as free money while budgeting salary tightly",
            "refuses to touch an 'education pot' to clear costly short-term debt",
        ],
        "money is treated as belonging to separate, non-interchangeable pots",
        "show the household balance sheet as one pool and net the positions",
    ),
    (
        "recency bias",
        [
            "wants to concentrate into whatever performed best over the last year",
            "abandons a strategy after two weak quarters",
            "assumes the recent trend will simply continue",
        ],
        "recent outcomes are given far more weight than the long record",
        "widen the window under discussion and revisit the written policy",
    ),
    (
        "confirmation bias",
        [
            "reads only commentary that supports a position they already hold",
            "dismisses any analysis that contradicts their favoured holding",
            "seeks a second opinion only until one agrees with them",
        ],
        "evidence that supports an existing belief is sought and contrary evidence discounted",
        "deliberately surface the strongest disconfirming case before deciding",
    ),
]


def pick_distinct(rng: random.Random, framings: list[str], k: int) -> list[str]:
    """k framings that do not share an opening.

    Sampling blind produced two 'A client {sym}...' variants for the same symptom,
    whose first 40 characters are then identical — which is the exact prefix
    collapse that killed the previous corpus. The guard in main() caught it; this
    prevents it.
    """
    by_open: dict[str, list[str]] = {}
    for f in framings:
        by_open.setdefault(f[:9], []).append(f)
    groups = list(by_open.values())
    rng.shuffle(groups)
    return [rng.choice(g) for g in groups[:k]]


def build(seed: int = 20260817) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []

    # Money script classification, ONE framing per symptom.
    #
    # A first version used two framings each, making money scripts 64 of 100
    # applied records. The retrained adapter then reached for the money-script
    # framework indiscriminately: asked about a client keeping a low-interest
    # holiday pot beside a 22% card balance it answered "money vigilance" where
    # the base model had correctly said "mental accounting" (50 -> 0), and it
    # answered a risk tolerance/capacity question with "money avoidance". The
    # class prior was the defect, not the phrasing. Keeping money scripts near
    # 40% of the applied set, with the discriminative categories below carrying
    # real weight, is what stops the framework being applied to everything.
    for script, spec in SCRIPTS.items():
        for sym in spec["symptoms"]:
            for f in pick_distinct(rng, FRAMINGS, 1):
                q = f.format(sym=sym, Prof=rng.choice(PROFESSIONS), age=rng.choice(AGES))
                a = (
                    f"This is a money script, specifically {script} — the belief "
                    f"{spec['belief']}. Money scripts are typically formed early, often in "
                    f"the family of origin, and operate outside conscious awareness, which "
                    f"is why the behaviour persists even when the client knows the "
                    f"arithmetic. As their planner I would {spec['action']}."
                )
                out.append({"q": q, "a": a})

    # flashpoint -> script -> behaviour
    fp_framings = [
        "A client {ev} and now {beh}. What is the technical term for that early "
        "experience, and how does it connect to what you are seeing?",
        "At discovery, a client {ev}. Today they {beh}. How would you describe this chain?",
        "A client {ev}. Their current behaviour is that they {beh}. Name the mechanism at work.",
    ]
    for ev, beh in FLASHPOINTS:
        for f in pick_distinct(rng, fp_framings, 2):
            out.append(
                {
                    "q": f.format(ev=ev, beh=beh),
                    "a": (
                        "That early experience is a financial flashpoint — a significant, often "
                        "painful money event that leaves a lasting impression. Flashpoints shape "
                        "money scripts, the beliefs a client carries about money, and those "
                        "scripts in turn drive present-day financial behaviour. The behaviour is "
                        "not irrational once the flashpoint is understood; it is the script doing "
                        "its protective job. I would explore the flashpoint first, then name the "
                        "script it produced, before discussing any change to the plan."
                    ),
                }
            )

    # bias classification
    bias_framings = [
        "A client {sym}. Which bias is this, and what is the planner's response?",
        "During a review a client {sym}. Name the bias and how you would address it.",
        "A client {sym}. What is driving this, and what would you do?",
    ]
    for name, syms, mech, action in BIASES:
        for sym in syms:
            for f in pick_distinct(rng, bias_framings, 2):
                out.append(
                    {
                        "q": f.format(sym=sym),
                        "a": (
                            f"This is {name}: {mech}. It is a predictable pattern rather than a "
                            f"failure of intelligence, and arguing with the arithmetic rarely "
                            f"shifts it. The response is to {action}, and to have agreed the "
                            f"approach in advance so the conversation is not happening for the "
                            f"first time under pressure."
                        ),
                    }
                )

    # Risk tolerance vs risk capacity. Two distinct concepts the corpus barely
    # separates (risk capacity appears ONCE in 304 records), and the eval item
    # that probes them was answered with "money avoidance" before these existed.
    risk_framings = [
        "A client {sit}. Which two concepts are in tension, and which should govern the plan?",
        "{Sit_cap}. How would you describe the mismatch, and what do you do about it?",
        "During review a client {sit}. Name the distinction at issue.",
    ]
    RISK_SITS = [
        "scores as aggressive on the questionnaire but has no emergency fund and irregular contract income",
        "says they are comfortable with volatility while carrying a large mortgage on a single income",
        "wants maximum growth two years from a planned drawdown",
        "is emotionally unbothered by losses but cannot afford any shortfall against a fixed liability",
        "has ample assets and a long horizon but panics at every market headline",
    ]
    for sit in RISK_SITS:
        for f in pick_distinct(rng, risk_framings, 2):
            out.append(
                {
                    "q": f.format(sit=sit, Sit_cap="A client " + sit),
                    "a": (
                        "Two separate things are in tension here: risk tolerance, which is the "
                        "client's psychological comfort with volatility, and risk capacity, which "
                        "is their financial ability to absorb a loss without damaging the plan. "
                        "Tolerance is a feeling and capacity is arithmetic; a questionnaire "
                        "measures the first and says nothing about the second. Capacity sets the "
                        "ceiling — you cannot take more risk than the plan can survive, however "
                        "comfortable the client feels — and tolerance determines whether they will "
                        "actually stay invested. Where they disagree, plan to the lower of the two "
                        "and address the gap explicitly."
                    ),
                }
            )

    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="apps/factory/data/financial_planning/training_data.jsonl")
    ap.add_argument("--out", default="apps/factory/data/financial_planning/training_data_v2.jsonl")
    # evaluation_data*.jsonl stayed at root data/ (consumed by benchmarks/, not training)
    ap.add_argument("--eval", default="data/financial_planning/evaluation_data_v2.jsonl")
    args = ap.parse_args()

    base_rows = [json.loads(x) for x in (REPO_ROOT / args.base).read_text().splitlines() if x.strip()]
    new = build()
    new_rows = [{"text": f"### Question:\n{r['q']}\n\n### Answer:\n{r['a']}"} for r in new]

    # --- guards, asserted rather than assumed --------------------------------
    qs = [r["q"] for r in new]
    prefixes = {q[:40] for q in qs}
    div = len(prefixes) / len(qs)
    firstword = {q.split()[0].lower() for q in qs}

    ev_rows = [json.loads(x) for x in (REPO_ROOT / args.eval).read_text().splitlines() if x.strip()]

    def toks(s: str) -> set[str]:
        return set(re.findall(r"\w+", s.lower()))

    worst = 0.0
    for e in ev_rows:
        et = toks(e["prompt"])
        for q in qs:
            worst = max(worst, len(et & toks(q)) / max(1, len(et)))

    print("=" * 92)
    print("  APPLIED CLASSIFICATION EXAMPLES — financial planning")
    print("=" * 92)
    print(f"  existing definitional records : {len(base_rows)}")
    print(f"  new applied records           : {len(new_rows)}")
    print(f"  combined                      : {len(base_rows) + len(new_rows)}")
    print()
    print("  DIVERSITY GUARDS (the last corpus died of 1 distinct question prefix)")
    print(f"    distinct 40-char prefixes   : {len(prefixes)}/{len(qs)} = {100 * div:.1f}%")
    print(f"    distinct opening words      : {len(firstword)}  {sorted(firstword)}")
    print(f"    worst eval-overlap (jaccard): {worst:.2f}")

    assert div > 0.9, f"question prefixes not diverse enough: {div:.2f}"
    assert worst < 0.8, f"a generated question is a near-duplicate of an eval item: {worst:.2f}"
    print("\n    ✓ all guards passed")

    out = REPO_ROOT / args.out
    with open(out, "w") as f:
        for r in base_rows + new_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\n  Wrote {len(base_rows) + len(new_rows)} records -> {out.relative_to(REPO_ROOT)}")
    print("\n  Sample generated record:")
    print("  " + new_rows[0]["text"][:300].replace("\n", "\n  "))


if __name__ == "__main__":
    main()
