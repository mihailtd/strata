# Eval instrument audit — is the adapter inert, or is the eval blind?

The financial expert measured **+0.00pp over base** on its own domain (83.33% vs
83.33%), against astral's +44.44pp and postgresql's +24.08pp. It was read as a
dead adapter for as long as it went unexamined.

**It was not the adapter. The eval could not see.** After fixing the instrument
and then the corpus, the same adapter family measures **+35.00pp, 95% CI
[+15.00, +55.00]**.

| stage | base | expert | delta | 95% CI |
| :--- | ---: | ---: | ---: | :--- |
| v1 eval, v1 corpus | 83.33% | 83.33% | **+0.00pp** | — |
| v2 eval, v1 corpus | 35.00% | 45.00% | +10.00pp | [−2.50, +22.50] *includes 0* |
| v2 eval, v2 corpus | 35.00% | **70.00%** | **+35.00pp** | **[+15.00, +55.00]** |

---

## Two independent defects

### 1. The eval gave away its own answers

`scripts/audit_eval_rubrics.py` measures this statically, in a second, with no GPU:

| domain | giveaway | fully given away | measured adapter effect |
| :--- | ---: | ---: | ---: |
| astral | **5.0%** | — | **+44.44pp** |
| postgresql | **55.0%** ⚠️ | — | **+24.08pp** |
| financial_planning (v1) | **68.2%** ⚠️ | 9/20 | **+0.00pp** |
| financial_planning (v2) | **0.0%** | 0/20 | +35.00pp |

*giveaway* = rubric terms already present in the question. `fin_01` asks "What are
money scripts... how do money avoidance and money status scripts affect..." and
its rubric is `['money script', 'money avoidance', 'money status']` — every term
handed over in the prompt. The rubric measured whether the model echoes the
question, which any instruction-following model does. **Base sat at 100% on 14 of
20 items**, so most of the eval could not register an adapter effect at all.

The ordering is monotonic across three domains with a mechanism that explains it:
the cleanest instrument produced the largest measured effect, the blindest
produced none. **postgresql at 55% implies its +24.08pp is an understatement.**

### 2. The corpus taught recitation, not application

`diagnose_expert_vs_base.py` separated the two hypotheses by running both arms
question-by-question: the adapter **changed 20/20 answers** and the rubric scored
**0** of those changes. Alive adapter, blind instrument.

But the fixed eval then exposed a second, real defect. Measured over the 304
training records:

    "How ..." questions          235
    "What ..." questions          64
    scenario / vignette            5   (1.6%)
    asking to CLASSIFY a client    0   (0.0%)

Asked the definitional question ("what are those inherited beliefs called?") the
expert answered **"money scripts"** correctly. Shown a client vignette it invented
framework names — **"Financial Identity Framework"**, **"Financial Enmeshment"**,
**"autobiographical memory"**. It could define the taxonomy and not apply it.
A reciter, not a diagnostician. money_script questions scored **+0.00pp** while
generic technique scored +25pp.

`scripts/build_financial_applied_examples.py` adds 78 applied classification
records (money-script taxonomy, flashpoint→script→behaviour, bias identification,
risk tolerance vs capacity). All four confabulations were fixed.

---

## Two things that went wrong while fixing it, both caught by guards

**Prefix collapse.** The repo's own history records the last corpus built from a
template: 940 records, **1 distinct question prefix**, adapter scoring *below*
base (60% → 35%). So the generator asserts diversity rather than hoping for it.
It failed twice — 80%, then 86%, against a house standard of ~97% measured from
the three real corpora. Both times the generator was fixed, not the threshold.
The second failure was two framings carrying **40 characters of fixed template
before the variable text** — the original failure mode in miniature. Now 100%.

**Class-prior overfit.** The first rebalance-free version made money scripts 64 of
100 applied records, and the adapter then reached for that framework
indiscriminately: it answered a mental-accounting vignette with "money vigilance"
(**50 → 0**, worse than base) and a risk tolerance/capacity question with "money
avoidance". Rebalancing to 41% money script + 31% bias + 15% flashpoint + 13% risk
fixed both (**→ 100** each). The lesson is that the class balance of applied
examples is itself a hyperparameter.

---

## What is NOT claimed

- **These are not production.** `m2_financial_r8a128`,
  `evaluation_data.jsonl` and `training_data.jsonl` are untouched. The v2 files
  sit alongside as `*_v2`. Swapping the canonical eval would break comparability
  with every historical financial number in the repo — that is a deliberate
  decision for a human, not a side effect of this work.
- **Scope.** The v2 eval is derived from the corpus's own concept inventory, so it
  measures *"did the adapter learn this corpus"* — not *"is this a good financial
  planner."* That is the same standard astral meets (does it say `uv`, not "is
  this good Python advice"), and it is not an independent quality benchmark.
- **The eval and the added training data share an author.** Mitigations: the eval
  was written and run *before* the applied corpus existed; vignettes are disjoint
  (worst overlap 0.58, asserted in the generator); base is measured on the same
  instrument at 35%; and the two categories given **no** new data act as controls.
- **Residual over-application.** `v2_18` still answers an anti-pattern question
  about financial education with "money script, money worship" (−50). The control
  categories also slipped, planner_technique and anti_pattern both +25.0 → +12.5,
  so the applied data bought its gains at some cost elsewhere.

| category | v1 corpus | v2 corpus | n | applied data added? |
| :--- | ---: | ---: | ---: | :--- |
| money_script_taxonomy | +0.0 | **+60.0** | 5 | yes |
| formative_events | +0.0 | **+50.0** | 2 | yes |
| behavioral_bias | +0.0 | **+25.0** | 4 | yes |
| risk_assessment | +0.0 | **+100.0** | 1 | yes (new) |
| planner_technique | +25.0 | +12.5 | 4 | **no — control** |
| anti_pattern | +25.0 | +12.5 | 4 | **no — control** |

---

## Usage

```bash
# static, no GPU — run this on any eval set before trusting a null result
uv run python scripts/audit_eval_rubrics.py

# per-question base vs expert: is the adapter inert, or the eval blind?
uv run --env-file .env python \
    benchmarks/factory/eval_instrument/diagnose_expert_vs_base.py \
    --domain financial_planning \
    --questions data/financial_planning/evaluation_data_v2.jsonl \
    --adapter results/adapters/m2v2_financial_r8a128

# regenerate the applied training records (diversity + overlap guards assert)
uv run python scripts/build_financial_applied_examples.py
```

**Before reporting any adapter as inert, run the rubric audit.** A +0.00pp result
on an instrument with 68% giveaway and 70% of items at ceiling is not a
measurement of the adapter.
