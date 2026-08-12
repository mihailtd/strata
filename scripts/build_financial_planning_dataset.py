"""Build the Financial Planning EVALUATION set (20 hand-written prompts).

TRAINING DATA IS NOT BUILT HERE ANY MORE. This script used to also emit
`training_data.jsonl` by wrapping raw EPUB paragraphs in one hardcoded
f-string:

    f"In financial planning and behavioral finance ({heading}), explain the "
    f"core principles of: {clean_p[:80]}..."

with the answer being that same paragraph echoed back. No model was involved.
Measured consequence, over the 940 records it produced:

    distinct 40-char question prefixes    1  (0.1%)   vs 97.5% for astral
    questions truncated mid-word        100%          vs 0%    for astral
    answers echoing the question        100%

Trained on that, the adapter learned to answer one stilted template with
book-excerpt prose and never learned the eval's natural questions -- it scored
*below* the base model on its own domain metric (60% -> 35%). Question
diversity, not adapter architecture, was the problem.

Training data now comes from the same 3-stage question/verify/answer pipeline
that produced the astral and postgresql sets, which reads EPUBs directly:

    # 1. start the server (see serving/README.md)
    ./serving/llama.cpp/build/bin/llama-server \\
        -m serving/models/gemma-4-E2B-it-Q8_0.gguf --host 127.0.0.1 --port 8080

    # 2. generate
    uv run scripts/run_datagen.py \\
        --epub "data/financial_planning/Psychology of Financial Plannin - Brad Klontz.epub" \\
        --domain-description "financial psychology, behavioral finance and client communication" \\
        --source-label book \\
        --out data/financial_planning

Run this script only to regenerate the evaluation prompts:

    uv run scripts/build_financial_planning_dataset.py
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FINANCIAL_DIR = REPO_ROOT / "data" / "financial_planning"

# Hand-written, and deliberately NOT derived from the book text: an eval whose
# questions are cut from the training source measures memorisation. Five per
# category, with anti_pattern_correction phrased as leading questions the model
# is expected to push back on rather than agree with.
EVAL_PROMPTS = [
    {
        "id": "fin_01",
        "category": "behavioral_finance",
        "prompt": "What are money scripts in financial therapy, and how do money avoidance and "
        "money status scripts affect client savings behaviors?",
    },
    {
        "id": "fin_02",
        "category": "behavioral_finance",
        "prompt": "How should a financial planner address loss aversion when a client wants to "
        "panic-sell during market volatility?",
    },
    {
        "id": "fin_03",
        "category": "behavioral_finance",
        "prompt": "What is financial flashpoint analysis, and how do childhood experiences shape "
        "adult financial decision-making?",
    },
    {
        "id": "fin_04",
        "category": "behavioral_finance",
        "prompt": "How does confirmation bias impact client investment choices, and what techniques "
        "can a planner use to counteract it?",
    },
    {
        "id": "fin_05",
        "category": "behavioral_finance",
        "prompt": "Explain the difference between financial literacy and financial psychology in client outcomes.",
    },
    {
        "id": "fin_06",
        "category": "client_communication",
        "prompt": "How can a financial planner use active listening and open-ended questions to "
        "uncover a client's core financial values?",
    },
    {
        "id": "fin_07",
        "category": "client_communication",
        "prompt": "What strategies help a financial planner navigate money conflict between married "
        "couples during joint planning sessions?",
    },
    {
        "id": "fin_08",
        "category": "client_communication",
        "prompt": "How should a planner deliver disappointing news, such as a client being unable to "
        "retire at their target age?",
    },
    {
        "id": "fin_09",
        "category": "client_communication",
        "prompt": "What is motivational interviewing in financial counseling, and how does it "
        "encourage long-term behavioral change?",
    },
    {
        "id": "fin_10",
        "category": "client_communication",
        "prompt": "How can a planner establish trust and rapport with a new client who feels anxious "
        "or ashamed about debt?",
    },
    {
        "id": "fin_11",
        "category": "risk_management",
        "prompt": "What is the distinction between a client's risk tolerance, risk capacity, and risk perception?",
    },
    {
        "id": "fin_12",
        "category": "risk_management",
        "prompt": "How does recency bias distort a client's perceived risk during extended bull or bear markets?",
    },
    {
        "id": "fin_13",
        "category": "risk_management",
        "prompt": "Why is sequence of returns risk critical in retirement income planning, "
        "and how can it be mitigated?",
    },
    {
        "id": "fin_14",
        "category": "risk_management",
        "prompt": "How should a financial planner handle mental accounting bias when clients treat "
        "inherited money differently from earned income?",
    },
    {
        "id": "fin_15",
        "category": "risk_management",
        "prompt": "What role does emotional intelligence play in managing portfolio risk during economic crises?",
    },
    {
        "id": "fin_16",
        "category": "anti_pattern_correction",
        "prompt": "Should a financial planner focus strictly on mathematical portfolio optimization "
        "without addressing client emotional reactions?",
    },
    {
        "id": "fin_17",
        "category": "anti_pattern_correction",
        "prompt": "Is giving clients comprehensive financial education alone sufficient to change "
        "destructive spending habits?",
    },
    {
        "id": "fin_18",
        "category": "anti_pattern_correction",
        "prompt": "Should a planner assume a client's risk tolerance questionnaire score remains "
        "constant throughout their lifetime?",
    },
    {
        "id": "fin_19",
        "category": "anti_pattern_correction",
        "prompt": "Is it best to immediately challenge and correct a client's irrational financial "
        "beliefs during the initial consultation?",
    },
    {
        "id": "fin_20",
        "category": "anti_pattern_correction",
        "prompt": "Should financial goals be established solely around numerical target returns "
        "rather than personal life values?",
    },
]


# Per-question rubric: the concepts a correct answer must actually name.
#
# WHY NOT A good/(good+bad) RATIO like astral and postgresql use? Those domains
# have a real opposing set (uv vs pip, pgvector vs pinecone). This one does not
# -- the old GENERAL_FILLER_TERMS list ("consult a professional", "as an AI
# language model") fired ZERO times across 60 measured generations, so the
# ratio collapsed to a binary "did it mention any domain word at all" and
# scored a broken adapter and the base model identically at 35%.
#
# Coverage against a per-question rubric fixes three things at once: it cannot
# be gamed by verbosity (the old metric rewarded longer answers), it cannot be
# gamed by repetition (distinct concepts, not raw hit counts), and it measures
# whether the answer addressed THIS question rather than the domain in general.
#
# Calibrated on 60 stored generations: base 78.3%, the broken template-trained
# adapter 33.3%. Every alternative is domain-specific; generic English like
# "long-term" or a bare "not" was removed because it matched base answers that
# had not actually engaged with the concept.
RUBRIC = {
    "fin_01": [r"money script", r"money avoidance", r"money status"],
    "fin_02": [r"loss aversion", r"(reframe|behavioral coach|stay the course|time horizon|pre[- ]commit)"],
    "fin_03": [r"flashpoint", r"(childhood|early (life|experience)|formative|upbringing)"],
    "fin_04": [
        r"confirmation bias",
        r"(disconfirm|contradict|challenge (their |the )?(assumption|belief)|devil's advocate)",
    ],
    "fin_05": [r"financial literacy", r"financial psychology"],
    "fin_06": [r"active listening", r"open[- ]ended", r"(core )?values"],
    "fin_07": [
        r"(money (conflict|argument|script)|financial (conflict|infidelity))",
        r"(couple|partner|spouse)",
        r"(neutral|facilitat|mediat|common ground)",
    ],
    "fin_08": [
        r"(empath|compassion)",
        r"(rehearse|prepare the client|deliver(ing)? (bad|difficult)|soften|scenario)",
    ],
    "fin_09": [r"motivational interviewing", r"(ambivalen|change talk|intrinsic motivat|autonomy|discrepan)"],
    "fin_10": [r"(trust|rapport)", r"(shame|non[- ]?judg|stigma|vulnerab)"],
    "fin_11": [r"risk tolerance", r"risk capacity", r"risk perception"],
    "fin_12": [r"recency bias", r"(bull|bear) market"],
    "fin_13": [r"sequence of returns", r"(bucket|cash reserve|withdrawal rate|guardrail|glide path|annuit)"],
    "fin_14": [r"mental accounting", r"(inherit|windfall|bequest)"],
    "fin_15": [r"emotional intelligence", r"(self[- ]regulat|self[- ]aware|empath)"],
    # The five anti-pattern questions are leading: a correct answer must
    # contradict the premise, so the second pattern looks for explicit pushback.
    "fin_16": [
        r"(emotion|behavioral|psycholog)",
        r"(shouldn't|should not|is not (sufficient|enough)|insufficient|more than (just )?(math|numbers)|rather than)",
    ],
    "fin_17": [
        r"(behavio|habit|emotion)",
        r"(not (sufficient|enough)|alone is (not|insufficient)|insufficient|shouldn't|should not|knowledge alone)",
    ],
    "fin_18": [
        r"(not (static|constant|fixed)|change(s)? over time|evolve|fluctuat|revisit|reassess)",
        r"(life (event|stage)|over time|periodic|market cycle)",
    ],
    "fin_19": [
        r"(rapport|trust|premature|too (early|soon)|defensive|resistance)",
        r"(shouldn't|should not|avoid|instead|rather than|counterproductive)",
    ],
    "fin_20": [
        r"(core )?values",
        r"(not (solely|just|only)|beyond (just )?|more than (just )?|rather than|in addition to)",
    ],
}


def build_eval_set() -> Path:
    FINANCIAL_DIR.mkdir(parents=True, exist_ok=True)
    missing = {p["id"] for p in EVAL_PROMPTS} ^ set(RUBRIC)
    if missing:
        raise ValueError(f"RUBRIC and EVAL_PROMPTS disagree on ids: {sorted(missing)}")

    out = FINANCIAL_DIR / "evaluation_data.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for p in EVAL_PROMPTS:
            f.write(json.dumps({**p, "expects": RUBRIC[p["id"]]}) + "\n")
    print(f"Wrote {len(EVAL_PROMPTS)} evaluation prompts (with rubrics) to {out}")
    return out


if __name__ == "__main__":
    build_eval_set()
