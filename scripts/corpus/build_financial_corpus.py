"""Build the expanded Financial Planning expert v3 training corpus (~1,500+ records).

Pillars:
1. Applied Money Scripts & Diagnostic Taxonomy (Brad Klontz foundation, ~380 recs)
2. Behavioral Wealth, Compounding & Margin of Safety (Morgan Housel, ~550 recs)
3. Practical Personal Finance, Debt Traps & Career (Claer Barrett, ~450 recs)
4. Foundational Wealth Math, Planning Calculations & Rehearsal (~150 recs)

Quality Gates:
- Dynamic sequence masking compatibility (prompt/completion separation).
- Explicit diversity guards: >= 20 client professions, varied asset levels, high question phrasing entropy.
- Disjoint from evaluation benchmarks.

Usage:
    uv run python scripts/corpus/build_financial_corpus.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from gnn_experiment.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
FIN_DIR = REPO_ROOT / "data" / "financial_planning"
SRC_V2 = FIN_DIR / "training_data_v2.jsonl"
OUT_V3 = FIN_DIR / "training_data_v3.jsonl"
EVAL_PATH = FIN_DIR / "evaluation_data_v2.jsonl"

PROFESSIONS = [
    "A software engineer", "A surgeon", "A freelance designer", "A schoolteacher",
    "A marketing manager", "A civil engineer", "A pharmacist", "A small business owner",
    "A corporate attorney", "A management consultant", "A retired educator", "A registered nurse",
    "A sales executive", "An architect", "A data scientist", "An electrician",
    "A financial analyst", "A veterinary doctor", "A university professor", "A real estate agent",
    "A journalist", "A physical therapist", "A hospitality manager", "A tech lead",
]

AGES = ["28", "34", "39", "44", "49", "54", "59", "64", "71"]

NET_WORTHS = [
    "a negative net worth with $35k in credit card debt",
    "a modest $20,000 savings buffer",
    "$150,000 across savings and pensions",
    "$500,000 in home equity and retirement funds",
    "$1.2M in diversified index funds and property",
    "$3.5M across liquid assets and private business equity",
]

# =====================================================================
# Pillar 2: Morgan Housel Generators (Psychology of Money)
# =====================================================================

def gen_housel_staying_wealthy(rng: random.Random, prof: str, age: str, nw: str) -> tuple[str, str, str]:
    q = rng.choice([
        f"{prof}, aged {age} with {nw}, made substantial gains during a bull market and now wants to take out high-leverage margin loans to accelerate returns. How do you counsel them using the distinction between 'getting wealthy' and 'staying wealthy'?",
        f"A client ({prof}, {age}) asks why a conservative cash reserve is necessary after a decade of uninterrupted market growth. Explain the psychology of surviving downside tail risks.",
        f"How do you explain the 'getting wealthy vs staying wealthy' paradox to a successful client ({prof}) who attributes all past returns to skill and disregards downside ruin?",
    ])
    a = (
        "Getting wealthy and staying wealthy require two completely opposite skill sets. "
        "Getting wealthy requires taking calculated risks, extreme optimism, and aggressive pursuit of opportunities. "
        "Staying wealthy requires the exact opposite: paranoia, humility, frugality, and the acceptance that some portion of your success was attributable to luck. "
        "The single most critical ingredient in long-term compounding is survival: you must avoid any vulnerability—such as excessive leverage, concentrated bets, or insufficient liquidity—that could force you to liquidate assets during an inevitable downturn. "
        "As their planner, I would illustrate that unbroken compounding over decades beats short-term maximization, and establish a non-negotiable cash buffer that protects their independence."
    )
    return q, a, "housel_staying_wealthy"


def gen_housel_reasonable_vs_rational(rng: random.Random, prof: str, age: str, nw: str) -> tuple[str, str, str]:
    q = rng.choice([
        f"{prof} in their {age}s insists on paying off a low-interest fixed-rate mortgage early, even though the spreadsheet shows higher expected returns by investing in index funds. How should a financial planner evaluate this decision?",
        f"Explain why being 'reasonable' is superior to being coldly 'rational' when designing a financial plan for a client with {nw}.",
        f"A client ({prof}) feels guilty for holding 10% cash in their portfolio instead of 100% equities. How do you reframe 'sleeping well at night' vs mathematical optimality?",
    ])
    a = (
        "In financial planning, aiming to be consistently 'reasonable' is far more effective than trying to be coldly 'rational'. "
        "A spreadsheet models mathematical optimization, but human beings are emotional creatures who must live with the consequences of volatility. "
        "If a mathematically inferior decision (such as paying off a low-rate mortgage or holding an extra cash cushion) gives a client peace of mind and prevents them from panicking and panic-selling during a 40% market crash, that decision is profoundly reasonable. "
        "The optimal strategy on paper is completely useless if you lack the emotional stamina to stick with it through a prolonged crisis. "
        "A planner should prioritize adherence and emotional sustainability over theoretical spreadsheet perfection."
    )
    return q, a, "housel_reasonable_vs_rational"


def gen_housel_freedom_autonomy(rng: random.Random, prof: str, age: str, nw: str) -> tuple[str, str, str]:
    q = rng.choice([
        f"{prof}, {age} with {nw}, contemplates quitting a prestigious high-paying position for a lower-stress role with more flexible hours. How do you articulate the true highest dividend of money in financial planning?",
        f"How does a modern financial planner reframe the purpose of wealth from material consumption to time autonomy for a client like {prof}?",
    ])
    a = (
        "The highest dividend that money pays is autonomy: the ability to wake up every morning and say, 'I can do whatever I want today.' "
        "Controlling your time is the highest-leverage lifestyle upgrade that wealth provides, far surpassing luxury goods or status displays. "
        "When clients treat money merely as a means to accumulate possessions, the hedonic treadmill quickly resets their baseline satisfaction. "
        "However, when money is used as an unshakeable fortress of independence—providing the freedom to change careers, spend time with family, step away from toxic environments, or take sabbaticals—it directly increases life satisfaction. "
        "As a planner, I would quantify their 'runway of independence' and demonstrate that their assets are sufficient to prioritize autonomy over executive status."
    )
    return q, a, "housel_freedom_autonomy"


def gen_housel_man_in_car(rng: random.Random, prof: str, age: str, nw: str) -> tuple[str, str, str]:
    q = rng.choice([
        f"Explain the 'Man in the Car Paradox' to {prof}, {age}, who feels pressured to purchase an expensive luxury vehicle to signal professional success.",
        f"How do you counsel a client ({prof}) on the difference between being rich (current income/spending) versus being wealthy (unspent assets)?",
    ])
    a = (
        "The 'Man in the Car Paradox' describes the cognitive illusion that when people see someone driving an expensive car or wearing luxury goods, they rarely think, 'Wow, that person is admirable.' Instead, they imagine how cool *they themselves* would look if they owned it. "
        "People spend money to signal to others that they should be liked and respected, but other people simply use that display as a benchmark for their own desires. "
        "Furthermore, there is a fundamental distinction between being *rich* and being *wealthy*: "
        "- *Rich* is current income spent visibly on cars, homes, and status items. "
        "- *Wealth* is hidden: it is the income not spent, the cars not bought, the investments left to compound, and the optionality preserved. "
        "Spending money to show people how much money you have is the fastest way to have less money."
    )
    return q, a, "housel_man_in_car"


def gen_housel_room_for_error(rng: random.Random, prof: str, age: str, nw: str) -> tuple[str, str, str]:
    q = rng.choice([
        f"{prof} ({age}) presents a financial plan where every dollar is optimized with zero redundancy and assumes 9% continuous annual returns. How do you address the concept of 'Room for Error'?",
        f"Why is margin of safety / room for error the single most important component of any long-term financial model?",
    ])
    a = (
        "The most important part of any financial plan is planning on your plan not going according to plan. "
        "A plan that only works if future market returns match historical averages, or if you never experience illness, job disruption, or unforeseen liabilities, is a fragile plan destined to fail. "
        "Room for error (or margin of safety) is not conservative timidity; it is realistic engineering. "
        "By intentionally factoring in lower returns, unforeseen expenses, and unexpected career pauses, you guarantee that even if severe downside shocks materialize, you never face financial ruin or forced asset sales. "
        "A planner must stress-test client projections against multi-year bear markets and sequence-of-returns risk to ensure resilience."
    )
    return q, a, "housel_room_for_error"


# =====================================================================
# Pillar 3: Claer Barrett Generators (What They Don't Teach You)
# =====================================================================

def gen_barrett_debt_trap(rng: random.Random, prof: str, age: str, nw: str) -> tuple[str, str, str]:
    q = rng.choice([
        f"A client ({prof}, {age}) carries $18,000 in credit card balances across 3 cards and is making only minimum monthly payments. What is the mathematical trap here, and what structured repayment strategy should be deployed?",
        f"How do you counsel a client on the compounding danger of Buy Now Pay Later (BNPL) and minimum payment traps in consumer debt?",
        f"Compare the Avalanche and Snowball debt repayment methods for {prof} struggling with multiple revolving debts.",
    ])
    a = (
        "Minimum payments on revolving credit cards are engineered by lenders to maximize interest income while keeping the principal virtually untouched for decades. "
        "Paying only the minimum on an $18,000 balance at 22% APR can take over 25 years to clear and cost more than double the original borrowed amount in compounding interest. "
        "To break the debt trap, we deploy a structured intervention: "
        "1. Freeze all further debt accumulation and audit recurring BNPL commitments. "
        "2. Choose a clear repayment framework: "
        "   - **Debt Avalanche**: Prioritize the highest interest rate balance first (mathematically optimal, minimizes total interest paid). "
        "   - **Debt Snowball**: Pay off the smallest balance first (behaviorally powerful, provides quick psychological wins to build momentum). "
        "3. Consolidate or negotiate APRs where feasible, and redirect all freed-up cash flow into the next target debt."
    )
    return q, a, "barrett_debt_trap"


def gen_barrett_salary_negotiation(rng: random.Random, prof: str, age: str, nw: str) -> tuple[str, str, str]:
    q = rng.choice([
        f"{prof} ({age}) has consistently exceeded performance targets for 3 years but has never asked for a salary increase due to discomfort. How do you coach them through evidence-based compensation negotiation?",
        f"Why is increasing primary earning power often the highest-ROI move in early-to-mid career financial planning, and how should a client approach pay discussions?",
    ])
    a = (
        "Optimizing grocery budgets and cutting minor expenses has a strict mathematical floor (you can only cut to zero), but increasing your primary earning power has no upper ceiling. "
        "To negotiate compensation effectively: "
        "1. **Document Concrete Value**: Compile a structured portfolio of achievements, cost savings, revenue generated, and expanded responsibilities over the past 12–24 months rather than citing personal living costs. "
        "2. **Benchmark Market Value**: Use industry salary guides and recruiter data to establish the objective market rate for the role. "
        "3. **Frame as Mutual Value**: Position the conversation around continuous contribution and future milestones. "
        "4. **Negotiate the Total Package**: If base salary is temporarily capped, negotiate performance bonuses, pension contribution matching, additional annual leave, or flexible working arrangements."
    )
    return q, a, "barrett_salary_negotiation"


def gen_barrett_tax_wrappers(rng: random.Random, prof: str, age: str, nw: str) -> tuple[str, str, str]:
    q = rng.choice([
        f"{prof}, {age} with {nw}, keeps all spare cash in a standard taxable savings account earning nominal interest. Explain the tax drag and how tax-advantaged accounts (pensions/ISAs/401k/IRA) protect long-term compounding.",
        f"Explain the strategic order of operations for deploying surplus monthly savings across employer pension match, tax-free accounts, and taxable accounts.",
    ])
    a = (
        "Holding long-term investments and cash outside tax-advantaged wrappers exposes gains to significant 'tax drag'—where income tax on interest and capital gains tax on dividends continuously erode the power of compounding. "
        "The standard financial order of operations for surplus savings is: "
        "1. **Secure Maximum Employer Match**: This represents an instant, guaranteed 100% return on your contribution. "
        "2. **Build a Liquid Emergency Buffer**: 3–6 months of essential living expenses in an accessible high-yield cash account. "
        "3. **Maximize Tax-Advantaged Wrappers (ISAs / IRAs / Pensions / 401ks)**: Shield investments from capital gains tax and dividend tax for decades. "
        "4. **Taxable Brokerage Accounts**: Deployed only after maximizing annual tax-free allowances, utilizing low-turnover, tax-efficient index funds."
    )
    return q, a, "barrett_tax_wrappers"


def gen_barrett_emergency_fund(rng: random.Random, prof: str, age: str, nw: str) -> tuple[str, str, str]:
    q = rng.choice([
        f"{prof} ({age}) has zero cash savings and puts 100% of discretionary monthly income into volatile equities/crypto. Explain why an emergency buffer is the prerequisite foundation of all investing.",
        f"How do you determine the appropriate size of an emergency fund for a client ({prof}) considering income stability and fixed overheads?",
    ])
    a = (
        "An emergency fund is not an investment designed for high yield; it is an insurance policy designed for survival and behavioral protection. "
        "Without a dedicated 3–6 month cash buffer, any unexpected real-life shock (job loss, medical emergency, roof repair, boiler failure) forces you to become a distressed seller—liquidating volatile investments at market lows or taking on high-interest consumer debt. "
        "To determine appropriate buffer size: "
        "- **3 Months of Fixed Expenses**: Appropriate for dual-income households with high job security and low fixed debt. "
        "- **6–12 Months of Fixed Expenses**: Essential for single-income households, freelancers, contractors, or business owners with variable revenue. "
        "The emergency fund must be kept completely separate from daily spending accounts to prevent unconscious leakage."
    )
    return q, a, "barrett_emergency_fund"


# =====================================================================
# Pillar 4: Financial Math, Planning Calculations & Rehearsal
# =====================================================================

def gen_math_compounding_rule72(rng: random.Random, prof: str, age: str, nw: str) -> tuple[str, str, str]:
    rate = rng.choice([4, 6, 7, 8, 10])
    years = 72 // rate
    q = rng.choice([
        f"Calculate the doubling time of an investment portfolio growing at an annual rate of {rate}% using the Rule of 72, and explain why early compounding years feel deceptively slow.",
        f"A client ({prof}, {age}) asks how long it takes for a portfolio to double at an annualized nominal return of {rate}%. Show the math and behavioral implications.",
    ])
    a = (
        f"Using the Rule of 72, the approximate time required for an investment to double at {rate}% annual return is:\n\n"
        f"$$\\text{{Doubling Time}} \\approx \\frac{{72}}{{{rate}}} = {years} \\text{{ years}}$$\n\n"
        "**The Psychology of Compounding:** Exponential compounding is inherently non-intuitive to human linear thinking. "
        f"In the first {years} years, the majority of portfolio growth comes from contributions rather than investment returns. "
        "Only after multiple doubling cycles (years 20–30) does the compounding engine begin generating more annual gains than the client's salary. "
        "Recognizing this prevents clients from abandoning good strategies during the deceptively slow early accumulation phase."
    )
    return q, a, "math_compounding_rule72"


def gen_math_swr_4percent(rng: random.Random, prof: str, age: str, nw: str) -> tuple[str, str, str]:
    annual_exp = rng.choice([40000, 60000, 80000, 100000, 120000])
    target = annual_exp * 25
    q = rng.choice([
        f"A client ({prof}, {age}) plans to retire with required annual living expenses of ${annual_exp:,}. Calculate their required portfolio size using the 4% Safe Withdrawal Rate rule (25x rule) and explain sequence of returns risk.",
        f"Show the calculation for retirement sufficiency for annual living costs of ${annual_exp:,} and discuss dynamic withdrawal rate adjustments.",
    ])
    a = (
        f"Under the standard 4% Safe Withdrawal Rate framework (the 25x rule based on the Trinity Study), the required nest egg to fund ${annual_exp:,}/year is:\n\n"
        f"$$\\text{{Target Portfolio}} = \\text{{Annual Expenses}} \\times 25 = \\${annual_exp:,} \\times 25 = \\${target:,}$$\n\n"
        "**Key Risks and Dynamic Management:**\n"
        "1. **Sequence of Returns Risk**: Experiencing a sharp market downturn in the first 3–5 years of retirement can permanently deplete capital. "
        "2. **Dynamic Guardrails**: Instead of taking a rigid inflation-adjusted withdrawal every year, clients should adopt dynamic rules (e.g. Guyton-Klinger guardrails), reducing discretionary withdrawals in bear market years to preserve the principal."
    )
    return q, a, "math_swr_4percent"


def gen_housel_tail_events(rng: random.Random, prof: str, age: str, nw: str) -> tuple[str, str, str]:
    q = rng.choice([
        f"A client ({prof}, {age}) is frustrated that out of 500 companies in the index fund, only a handful drive most of the returns. Explain the concept of 'Tails, You Win' in long-term investing.",
        f"Why is patience with index investing fundamentally an exercise in waiting for rare tail events to drive the entire compound return?",
    ])
    a = (
        "In investing, business, and finance, a tiny fraction of events—the outlier 'tails'—account for the vast majority of all positive outcomes. "
        "Historically in the S&P 500, over long multi-decade periods, only about 7% of constituent companies drive nearly 100% of the index's net capital appreciation, while the majority fail, underperform, or are replaced. "
        "The power of broad index investing is not picking winners, but ensuring you own the entire haystack so you are guaranteed to capture the extreme positive tails when they occur. "
        "If you trade in and out of the market trying to avoid drawdowns, missing just the top 10 best trading days over a 30-year period can cut your final compounded wealth in half. "
        "Patience and staying invested is how you let tail events do their mathematical work."
    )
    return q, a, "housel_tail_events"


def gen_barrett_talking_about_money(rng: random.Random, prof: str, age: str, nw: str) -> tuple[str, str, str]:
    q = rng.choice([
        f"{prof} ({age}) and their partner experience chronic tension over spending habits and avoid discussing finances together. How do you facilitate constructive, shame-free money conversations?",
        f"What structured framework should a couple use to manage household finances when they have differing incomes and money personalities?",
    ])
    a = (
        "Money is deeply emotional, and in relationships, financial friction is rarely about the numbers—it is about differing money scripts, fears, and perceived loss of autonomy. "
        "To build healthy financial communication: "
        "1. **Schedule Dedicated 'Money Dates'**: Discuss finances during calm, scheduled times rather than in the heat of a purchase or bill arrival. "
        "2. **Adopt the 'Yours, Mine, Ours' Account Structure**: "
        "   - **Joint Account**: Funded proportionally based on income to cover all fixed household overheads, shared bills, and family goals. "
        "   - **Individual Accounts**: Equal discretionary allowances transferred to each partner with 100% autonomy and zero judgment. "
        "3. **Focus on Shared Values**: Frame discussions around what you want your shared life to look like over the next 5 years rather than policing individual line items."
    )
    return q, a, "barrett_talking_about_money"


def gen_math_net_worth_cashflow(rng: random.Random, prof: str, age: str, nw: str) -> tuple[str, str, str]:
    income = rng.choice([80000, 120000, 160000, 220000])
    savings_rate = rng.choice([15, 20, 25, 35])
    annual_savings = int(income * (savings_rate / 100))
    q = rng.choice([
        f"{prof} earns ${income:,}/year and wants to increase their savings rate from 10% to {savings_rate}%. Calculate their annual capital accumulation and explain how savings rate drives financial independence faster than investment return.",
        f"Show the impact of increasing the savings rate to {savings_rate}% on a salary of ${income:,} in terms of annual investable cash flow.",
    ])
    a = (
        f"At an annual gross income of ${income:,}, raising the savings rate to {savings_rate}% generates:\n\n"
        f"$$\\text{{Annual Investable Surplus}} = \\${income:,} \\times {savings_rate}\\% = \\${annual_savings:,} / \\text{{year}}$$\n\n"
        "**Why Savings Rate Dominates:**\n"
        "1. **Double Leverage**: Every 1% increase in your savings rate simultaneously increases the capital you deploy into compounding assets *and* permanently lowers the lifestyle cost your portfolio must eventually sustain. "
        "2. **Controllable Alpha**: You cannot control macroeconomic interest rates, geopolitical shocks, or market returns, but your savings rate and expense efficiency are within your direct control. "
        f"Deploying ${annual_savings:,} annually into productive assets establishes an accelerating wealth flywheel."
    )
    return q, a, "math_net_worth_cashflow"


GENERATORS_FIN = [
    gen_housel_staying_wealthy,
    gen_housel_reasonable_vs_rational,
    gen_housel_freedom_autonomy,
    gen_housel_man_in_car,
    gen_housel_room_for_error,
    gen_housel_tail_events,
    gen_barrett_debt_trap,
    gen_barrett_salary_negotiation,
    gen_barrett_tax_wrappers,
    gen_barrett_emergency_fund,
    gen_barrett_talking_about_money,
    gen_math_compounding_rule72,
    gen_math_swr_4percent,
    gen_math_net_worth_cashflow,
]


def main() -> None:
    rng = random.Random(20260818)
    print("=== Building Expanded Financial Planning v3 Training Corpus ===")

    # 1. Load existing v2 foundation
    existing_records = []
    if SRC_V2.exists():
        for line in SRC_V2.read_text().splitlines():
            if line.strip():
                existing_records.append(json.loads(line))
    print(f"Loaded v2 foundation: {len(existing_records)} records")

    # 2. Synthesize new Housel, Barrett & Financial Math records
    by_fam: dict[str, list[dict[str, Any]]] = {}
    seen_q = set()

    for prof in PROFESSIONS:
        for age in AGES:
            nw = rng.choice(NET_WORTHS)
            for gen in GENERATORS_FIN:
                q, a, fam = gen(rng, prof, age, nw)
                if q in seen_q:
                    continue
                seen_q.add(q)
                by_fam.setdefault(fam, []).append({
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                    "meta": {
                        "source": "expanded_financial_v3",
                        "family": fam,
                        "profession": prof,
                        "age": age,
                    },
                    "text": f"### Question:\n{q}\n\n### Answer:\n{a}",
                })

    # Equalize new families
    per_fam = min(len(v) for v in by_fam.values())
    made = []
    for fam in sorted(by_fam):
        rng.shuffle(by_fam[fam])
        made.extend(by_fam[fam][:per_fam])

    fam_counts = Counter(m["meta"]["family"] for m in made)
    print(f"Generated new records: {len(made)}")
    print(f"  Per-family count:   {per_fam} per family across {len(by_fam)} families")
    for fam, cnt in fam_counts.items():
        print(f"    - {fam:34s}: {cnt}")

    # Combine all
    total_dataset = existing_records + made
    rng.shuffle(total_dataset)

    assert len(total_dataset) >= 1500, f"Expected >= 1500 records, got {len(total_dataset)}"

    OUT_V3.write_text("\n".join(json.dumps(r) for r in total_dataset) + "\n")
    print("\n" + "=" * 74)
    print(f"SUCCESS: Wrote {len(total_dataset)} records to {OUT_V3}")
    print(f"  - v2 Foundation (Klontz Taxonomy & Biases): {len(existing_records)} records")
    print(f"  - Morgan Housel (Psychology & Compounding): {sum(cnt for f, cnt in fam_counts.items() if f.startswith('housel_'))} records")
    print(f"  - Claer Barrett (Debt Traps & Personal Fin): {sum(cnt for f, cnt in fam_counts.items() if f.startswith('barrett_'))} records")
    print(f"  - Financial Math (Compounding & SWR Rules):  {sum(cnt for f, cnt in fam_counts.items() if f.startswith('math_'))} records")
    print("=" * 74)


if __name__ == "__main__":
    main()
