# Adapter Changelog

What changed between adapter versions: techniques added, techniques removed, and
what the measurement said afterwards. Every entry records the **outcome**, not just
the intent — several techniques here were added, measured, and abandoned.

> ⚠️ **Version numbering has a discontinuity.** Adapter `v5` is the *failed L_inert
> experiment*. Corpus `training_data_v5.jsonl` is the *merged disposition corpus*.
> They are unrelated. To avoid compounding it, adapters trained on corpus v5 are
> named **v6**. Adapter version ≠ corpus version from here on; each entry states
> which corpus it used.

| version | corpus | headline | status |
| :--- | :--- | :--- | :--- |
| v1 | v1 | first adapters | superseded |
| v2 | v2 | corpus expansion | superseded — prompt-inclusive loss |
| v3 | v3 | more generators | superseded — contaminated gate |
| v4 | v4 | completion-only loss + verified reservation + replay | superseded by v6 |
| ~~v5~~ | v4 | `L_inert` on in-domain prompt tokens | ❌ failed — **weights deleted** |
| ~~v5b~~ | v4 | `L_inert` on out-of-domain replay, pads unmasked | ❌ failed — **weights deleted** |
| ~~v5c~~ | v4 | `L_inert` pad-masked, λ=0.5 | ⚠️ lost to crash — **weights deleted** |
| v6 | v5 | disposition corpora + dual-criterion stopping | superseded by v7 |
| **v7** | v6 | round-2 corpora + two new domains; *entry reconstructed 2026-09-24* | **CANONICAL** (`CANON.ADAPTER_VERSION`) |
| v8 | v6 | `agentic_coding` only — a re-run of v7; *entry reconstructed* | shipped for `agentic_coding` |
| v9 | **v7** | served chat format + model-written reasoning per record | ❌ **not promoted** — style up, correctness down in 5 of 7 domains (4B) |

> ⚠️ **v7 and v8 shipped without changelog entries.** The two entries below were
> reconstructed on 2026-09-24 from the artifacts themselves (`adapter_config.json`,
> `regime.json`, `geometry_trace.json`, file timestamps) and the §12 measurements in
> `MEASURED_FINDINGS.md`. The training ledger (`results/factory.db`) could **not** be
> used: it has no run matching the served 4B v7 adapters (their weights are dated
> 2026-08-20; the earliest ledger v7 run is 2026-08-24), and several of its rows
> report identical final geometry to eight decimal places across different runs.
> Where a fact could not be recovered, the entry says so.

---

## v9 — served chat format + reasoning in every record *(corpus v7)* ❌ NOT PROMOTED

`CANON.ADAPTER_VERSION` stays **v7**. v9 improves the style axis and costs general
correctness; it fails the "better on both axes" bar.

### What changed from v7 — deliberately ONE thing
Everything a v7 adapter was trained with is kept — r=8, α=128 (scaling 16), lr 2e-4,
completion-only loss, and each size's own stopping rule (4B: first check with
`‖ΔW‖/‖W‖` ≥ 0.071; 9B: 0.075, 150-step cap) — so any difference is attributable
to the one change:

**The training records are now rendered exactly as the served model sees them,
with reasoning in the `<think>` block.**

| | v7 | v9 |
| :--- | :--- | :--- |
| prompt | `### Question:\n{q}\n\n### Answer:\n` | `<\|im_start\|>user\n{q}<\|im_end\|>\n<\|im_start\|>assistant\n<think>\n` |
| target | `{answer}` | `{reasoning}\n</think>\n\n{answer}<\|im_end\|>` |
| reasoning in target | none (0 of 9,918 records) | every record |

The v9 rendering is byte-identical to Qwen3.5's own chat template and to what
`runtime-next` serves (`apps/factory/tests/test_chat_format_and_thinking.py` pins it
against the real tokenizer). v7 was trained in a position the served model is
never in.

### Corpus v7
Corpus v6 plus one reasoning trace per record, written by **Qwen3.5-9B itself**
(`apps/factory/corpus/add_thinking.py`) — same model family, same voice, so the
adapter is not taught to imitate a foreign style. The teacher sees the request and
the known-good answer and writes the reasoning that precedes it: what the task needs,
**which tool or idiom and why for this task** (polars vs pandas, uv vs pip, a
dataclass and a function vs a class), then a plan. Each trace is validated: no code
block, no reference to having seen an answer, 40–320 words.

| domain | source | kept | rejected (why) | median words |
| :--- | ---: | ---: | :--- | ---: |
| agentic_coding | 1193 | 1179 | 14 (answer leak) | 131 |
| python_modern | 1418 | 1384 | 34 (answer leak) | 139 |
| python_web | 1511 | 1478 | 33 (answer leak) | 141 |
| financial | 619 | 616 | 3 (answer leak) | 147 |
| astral | 1396 | 1375 | 20 leak, 1 truncated | 138 |
| duckdb | 1767 | 1743 | 23 leak, 1 too long | 142 |
| postgresql | 2014 | 1977 | 37 (answer leak) | 142 |

Rejected records are left out rather than given an empty think block. Full manifest:
`apps/factory/data/corpus_v7_manifest.json`. Known limitation: traces are
**rationalized** (written knowing the answer) and can state wrong things
confidently — spot-checking found one ("a frozen `TypedDict`", which does not exist).

### Measured — 4B, through the serving path
`evals/factory/compare_adapter_generations.py`: `runtime-next`, real chat template,
thinking on, greedy, adapters swapped per request, every raw completion stored
(`results/benchmarks/adapter_generations_v7_vs_v9_4b.json`). Correctness = HumanEval
pass@1 (164), paired exact McNemar. Style = the held-out disposition items scored on
the **answer after `</think>`** (n = 7–8 per domain — one item is 12.5 points, so
single-domain style moves are weak evidence), and for `agentic_coding` SEARCH/REPLACE
format adherence on its 179 held-out tasks. Baseline arm = the shipped version (v8 for
`agentic_coding`, v7 otherwise).

Base: HumanEval **146/164**, ruff 0.60 findings per passing solution, median
thinking 1,692 chars.

| domain | style base / old / **v9** | HumanEval old → **v9** | paired −/+ | p | thinking chars old → v9 |
| :--- | :---: | ---: | :---: | ---: | ---: |
| agentic_coding | 0.00 / 0.04 / **0.50** | 100 → 100 | −19 / +19 | 1.0 | 77 → 904 |
| python_modern | 0.25 / 0.31 / **0.44** | 110 → 103 | −30 / +23 | 0.41 | 713 → 800 |
| python_web | 0.56 / 0.62 / 0.56 | 129 → **108** | −31 / +10 | **0.002** | 1021 → 782 |
| astral | 0.19 / 0.56 / 0.56 | 140 → **127** | −24 / +11 | **0.04** | 1898 → 814 |
| duckdb | 0.21 / 0.36 / **0.50** | 125 → **111** | −27 / +13 | **0.04** | 1456 → 780 |
| postgresql | 0.44 / 0.44 / 0.50 | 136 → **112** | −35 / +11 | **0.0005** | 1678 → 796 |
| financial | 0.56 / 0.19 / **0.56** | 141 → **122** | −22 / +3 | **0.0002** | 1645 → 776 |

Mean HumanEval across the seven: v7/v8 **125.9** → v9 **111.9** (base 146).

### What it means

- **The format fix works — the adapters finally do what they were trained for.**
  `agentic_coding` produced its SEARCH/REPLACE edits 3.9% → **49.7%** of the time;
  v8 never even closed its think block on 93 of 179 tasks, v9 on 0. Style is up or
  level in six of seven domains.
- **It also makes the collateral damage fully live.** v7's non-code adapters barely
  touched HumanEval (financial 141, astral 140) *because* they were trained off-position
  and were half-inert when served. v9 is fully switched on, in both directions.
- **v9 imposes its trace length on everything.** Every v9 adapter thinks ~800 chars on
  HumanEval — the length of the corpus traces (~140 words) — against base's 1,692.
  Where v7 had left thinking at base length, v9 roughly halved it, and those are the
  domains that lost correctness.
- **Two candidate causes, not yet separated:** (1) the weight change is too large —
  scaling 16 at ~7% of `‖W‖`, which §12e showed matters; (2) the traces are too short,
  so the adapter caps reasoning at half of base. `agentic_coding` is the one domain
  where v9 *lengthened* thinking (77 → 904) and it is also the one that did not lose
  correctness — consistent with (2).
- **Retracted along the way:** an earlier reading of the `agentic_coding` pilot said
  "thinking length is not the lever, magnitude is". The full table does not support
  that; both remain open.

### Measured — 9B (partial: 6 of 7 domains; evaluation paused 2026-09-24)
Same instrument, 9B, base HumanEval **152/164**. To save GPU time the old arm's
HumanEval was run only for `agentic_coding` and `python_modern` (see the retraction
below for why the others were skipped).

| domain | style base / old / **v9** | HumanEval old → **v9** |
| :--- | :---: | ---: |
| agentic_coding | 0.00 / 0.02 / **0.50** | 72 → **113** |
| python_modern | 0.44 / 0.31 / **0.69** | 133 → **113** |
| python_web | 0.50 / 0.44 / **0.62** | — → 115 |
| astral | 0.31 / 0.56 / **0.69** | — → 119 |
| duckdb | 0.29 / 0.29 / **0.50** | — → 131 |
| postgresql | 0.38 / 0.69 / 0.69 | — → 121 |
| financial | *not yet measured* | |

At 9B, v9's style gains are larger and more uniform than at 4B (up in 5 of 6),
and `agentic_coding` gains **41 HumanEval problems** over v8 — the think-collapse
fix pays off more on the bigger model. But every v9 adapter still sits 21–39
problems below base, and `python_modern` loses 20 against its v7.

> ⚠️ **Retraction (2026-09-24).** An earlier version of this entry, the pipeline
> comments and the eval-script help text said six 9B v7 adapters were "effectively
> untrained", because their `regime.json`/`geometry_trace.json` record
> `‖ΔW‖/‖W‖` ≈ 7×10⁻⁶. That was wrong. Measured directly from
> `adapter_model.safetensors`, `m2_python_modern_r8a128_v7_9b` has
> `‖Σ ΔW‖_F` = 45.78 — the same as the normally-trained v8 (45.38) and v9 (45.71)
> — and it scores 133 on HumanEval, not base's 152. **The recorded geometry was a
> measurement error in that training run's telemetry, not an untrained adapter.**
> Consequence: the skipped old-arm HumanEval runs for five domains are real gaps,
> not redundant measurements, and should be run before any 9B verdict.

### Infrastructure built for v9
- `train_expert.py --chat-format qwen` (served-format rendering, `<|im_end|>` stop,
  records without reasoning skipped, `--max-length` 512 → 2048) and
  `--skip-alpha-calibration` (the post-train alpha rewrite would change a second
  variable; the served v7s predate it).
- `runtime-next`: `chat_template_kwargs.enable_thinking` (vLLM/SGLang-compatible),
  used to make the 9B teacher write the rationale directly — with its own reasoning
  on, it spent all 2,048 tokens drafting and emitted nothing (32/32 in the pilot).
- `runtime-next` bug fixed: changing batch size built the new batched decode state
  before freeing the old one, OOMing the final partial batch at 9B.
- The generator resumes after a crash (`--resume`); the pipeline
  (`apps/factory/run_v9_pipeline.sh`) refuses to ship an adapter whose final
  `‖ΔW‖/‖W‖` is below 0.01.
- Evaluation-harness bug found and fixed before any number here was recorded: the
  first run passed pre-cleaned code to `HumanEvalExecutor.execute`, which cleans
  again and strips imports; every completion was re-executed from storage.

### Next — v10
Test cause (1) alone: the same corpus v7 and format, with the `‖ΔW‖/‖W‖` stopping
target halved (0.071 → 0.036). If correctness returns and the style gains hold, v10
ships; the thinking length it produces is also evidence on cause (2).

## v8 — `agentic_coding` re-run on the same corpus *(corpus v6)* — reconstructed

Only `agentic_coding` has a v8 (`m2_agentic_coding_r8a128_v8`, plus `_0_8b`, `_2b`,
`_9b`, `_27b` size variants). Against its v7 the artifacts differ in exactly this:

| | v7 | v8 |
| :--- | ---: | ---: |
| corpus | `agentic_coding/training_data_v6.jsonl` (1193) | same |
| r / α / lr | 8 / 128 / 2e-4 | same |
| stopped at step | 100 | 110 |
| final `‖ΔW‖/‖W‖` | 0.07517 | 0.07522 |
| relative growth at stop | 0.74% | 0.36% |
| weights written | 2026-09-21 12:49 | 2026-09-21 14:00 |

`adapter_config.json` differs only in the *order* of `target_modules`. No document,
commit message or ledger row states why it was re-run. **Treat v8 as a re-roll of
v7, not a new technique.** It is the version §12 measured (verified: the server log
for that run, `results/benchmarks/server_4b_humaneval.log`, pre-loaded
`m2_agentic_coding_r8a128_v8`).

### Measured (§12, 4B, `runtime-next`, HumanEval, 164 problems)

| arm | pass@1 | ruff findings / passing solution |
| :--- | ---: | ---: |
| base | **150** | 0.60 |
| `agentic_coding` v8 @1.0 | 98 (**−52 net**: loses 55, wins 3) | **0.06** |

Never run on a disposition eval.

## v7 — round-2 corpora, two new domains *(corpus v6)* — reconstructed

### Added
- **Corpus v6** (the trainer's `--v7` help text: "round-2 rebuild: astral command
  families, postgres asyncpg, duckdb analytics, python capability records").
  Records per domain: astral 1396, postgresql 2014, duckdb 1767, financial 619,
  python_modern 1418, python_web 1511, and a new **`agentic_coding`** domain, 1193.
- **Size variants**: `_9b` and `_27b` for every domain (`agentic_coding`'s 27B is
  v8), `_0_8b`/`_2b` for `python_modern` (and v8 `agentic_coding`), `_ornith35b` for
  the six non-agentic domains.

### Regime, as recovered from the artifacts
r=8, α=128 (scaling **16**), lr 2e-4, completion-only loss, records rendered as
`### Question: … ### Answer: …`. Every 4B v7 stopped at the **first geometry check
at or above `‖ΔW‖/‖W‖` 0.071** (steps 70–80; `agentic_coding` 100). 0.071 is the only
target consistent with all seven traces (python_modern stopped at 0.0722, which
rules out 0.075).

### ⚠️ Regression against v6, found during reconstruction
v6's entry discarded the fixed geometric target as **under-training** and shipped
dual-criterion stopping (plateau + ceiling). v7 went back to the fixed target: the
trainer's defaults are `--stop-at-dw-over-w 0.065` with `--stop-at-plateau` off, and
at the stop the large corpora were **still growing 7.8–8.2% per 10 steps** —
the exact under-training signature v6 described. Only `agentic_coding` (0.74%) and
`python_modern` (4.2%) were near convergence.

### Measured (§12, 4B, `runtime-next`, 164 HumanEval problems + 32 aider tasks)

| arm | HumanEval pass@1 | ruff / passing | aider (32, 5 turns) |
| :--- | ---: | ---: | ---: |
| base | **150** | 0.60 | **29** |
| `python_modern` v7 @1.0 | 107 (**−43 net**) | **0.20** | 17 |
| `python_modern` v7 @0.5 | — | — | 28 |

**The style training works; correctness pays for it.** §12d traced the cause to the
data: 0 of 1418 records contain reasoning, so with completion-only loss the adapter
learned to answer without thinking (mean thinking 1934 → 823 chars). The other five
domains were never measured on correctness, and no domain was measured on a
disposition eval.

### Known hazards in the serving path, found 2026-09-24
- `runtime-next` resolves adapter names by **substring** (`name.contains(...)`), so
  requesting `m2_x_r8a128_v7` after `m2_x_r8a128_v7_9b` is registered can bind the
  wrong adapter. Several scorecards record only `"adapter": "agentic_coding"`, which
  is ambiguous for exactly this reason.
- Adapter state is **sticky**: a request without an `adapter` field runs under
  whatever the previous request folded in. A "base" arm must name `"base"`.

## v6 — disposition corpora + dual-criterion stopping *(corpus v5)*

> **The v5 adapter line (`v5`/`v5b`/`v5c`) was deleted.** All three L_inert attempts
> failed or were lost; the findings below are the record, the weights had no further
> use. Version numbers are not contiguous on disk, and that is fine — renumbering
> would invalidate every reference in DECISIONS.md and in the probe results.

### ⚠️ First attempt DISCARDED — a fixed 0.075 target measured a WASH against v4

Six adapters were trained at `--stop-at-dw-over-w 0.075`, all stopping at 80–90
steps. Benchmarked on the held-out-situation disposition evals (n=23, base/v4/v6):

| | score | MANUAL | tokens |
| :--- | ---: | ---: | ---: |
| base | 0.087 | 13.0% | 251 |
| v4 | **0.522** | 0.0% | 91 |
| v6 (first attempt) | **0.522** | 4.3% | 264 |

Per domain the changes cancelled: postgres **+0.188**, astral +0.063, duckdb
**−0.285**. Those adapters were deleted; `results/benchmarks/disposition_v4_vs_v6.json`
is retained as the evidence.

Two signals said **under-trained**, not mis-trained:

1. **Everything stopped while still growing at 7% per 10 steps.** The fixed target
   was chosen only because it matched what v4 happened to land on.
2. **Family blending.** Asked how to give two services different interpreters, it
   produced `uv add -r requirements.txt` + `uvx pyenv` — the migration recipe
   grafted onto the interpreter question, with the tool it was trained to avoid.
   That is two shallowly-learned families colliding, not a missing one: it saw the
   `uvcmd_python` family ~9 times, at 0.22 epochs.

### Added (in v6 as shipped)

- **Everything from the first attempt**: disposition corpora across all domains,
  the 836-record uv/ruff/ty command corpus (astral 13.7% → 62.4% command-shaped),
  `python_modern` and `python_web` split out of astral, financial disposition data,
  deduplication (~2600 records removed), held-out-situation eval sets.
- **Dual-criterion stopping**, first-to-fire:

  | | | |
  | :--- | :--- | :--- |
  | ceiling | `‖ΔW‖/‖W‖ ≥ 0.100` | hard stop at the retention ceiling |
  | plateau | `rel_growth < 2%` | converged; adapts per corpus |
  | floor | `≥ 0.035` | never call it converged in the bf16 truncation zone |

  The plateau term is the on-the-fly signal a level threshold cannot express. At
  step 80 the 609–681 record corpora sat at **3.6–4.0%** relative growth while the
  1184–1610 record ones sat at **7.8–8.4%** — genuinely different distances from
  convergence, which a fixed target discards.

  Plateau alone is unsafe: extrapolated, the large corpora reach 2% at
  `‖ΔW‖/‖W‖ ≈ 0.100–0.106`, at or past the ceiling. Both bounds are required.

- **~1.5–1.9× more training.** Expected: astral/postgres/duckdb ~150–165 steps
  (ceiling fires), python_modern/python_web/financial ~120–125 (plateau fires).
- **Financial eval set** — 8 held-out situations. It had 312 training records and
  zero eval items, so it was silently skipped from the first benchmark (n 31 → 23).

### Removed

- **Fixed 150-step training** (0.37–0.99 epochs depending on corpus size).
- **The fixed 0.075 geometric target** as the sole criterion — retained as an
  optional flag, no longer the default path.

### Measured — v6 is now `CANON.ADAPTER_VERSION`

Held-out-situation disposition evals, base / v4 / v6, n=31:

| | score | MANUAL | tokens |
| :--- | ---: | ---: | ---: |
| base | 0.097 | 16.1% | 195 |
| v4 | 0.532 | 0.0% | 123 |
| **v6** | **0.581** | 0.0% | 282 |

Per domain: astral **+0.250**, postgresql **+0.126**, financial **+0.063**,
duckdb −0.285.

⚠️ **The duckdb figure is a SCORING ARTIFACT, not a regression.** The disposition
corpus format is `right approach + **Not X** — why not`, so v6 learned to name the
rejected tool when explaining itself. The scorer's `avoid` patterns then fire on
that mention and downgrade NATIVE to MIXED:

    v6 answer: "... USING SAMPLE;  **Not `WHERE random() < 0.1`** — the predicate
                still scans every row"     <- avoid pattern matches here

156 of 390 duckdb disposition answers name a rejected tool. v6 has 10 MIXED to
v4's 3, concentrated in duckdb. v4 was never trained to explain rejections so it
never trips it. **v6's true margin is larger than +0.048**; the scorer needs to
ignore avoid-matches inside a rejection construction before these numbers are
final.

v6 is 2.3x more verbose (282 vs 123 tokens), partly because it explains its
rejections. That is a deliberate trade, and terseness is recoverable at the prompt
level.

### Runtime Integration: Surgical Stacking as Default Multi-Expert Activation Mode
- Integrated Two-Stage Surgical Stacking into `WeightFoldingEngine.activate_many()` with `scale_mode="surgical"` as default.
- Auto-identifies collision layers (Macro LV-GLasso) and notches conflicting neurons (Micro POET), leaving 100% of Attention and clean MLP modules unattenuated ($\alpha=128$).
- Verified zero inference latency penalty, zero extra VRAM allocation, and exact $L_\infty = 0.00$ restoration via `Pristine State Buffer`.


### Deliberately NOT included

**`L_inert` is not in v6.** Three attempts, two measured failures, one lost. Adding
an unvalidated loss on top of a corpus overhaul makes the result unattributable.
**v7 = v6 + `L_inert`**, if it ever reproduces a real ASR gain.

## v5c — `L_inert` with pad-masked out-of-domain replay ⚠️ LOST

λ=0.5, 300 steps, corpus v4. **Killed by a host crash (machine to BIOS) at ~step
120.** `results/adapters/m2_astral_r8a128_v5c/` contains only an empty
`checkpoints/`. The log lived in `/tmp` and did not survive the reboot.

Last reading before loss: `e_in=0.0672 e_out=0.0632 ASR≈1.064`, flat across 40
steps. Never validated by the probe. Probable cause of the crash: two GPU jobs
running concurrently.

### Fixed in this attempt

- **Pad tokens excluded from the penalty.** v5b padded replay batches to 512 with
  `padding="max_length"`, so ~70% of each batch was `<pad>` and the adapter learned
  to be silent on padding — free, and generalising to nothing.
- **The forward hook made pure.** It branched on mutable trainer state, so gradient
  checkpointing's recomputation saved a different number of tensors and raised
  `CheckpointError`. A forward hook must be a function of its inputs only.

---

## v5b — `L_inert` on out-of-domain replay ❌ FAILED

λ=0.05, 150 steps, corpus v4. Replay drawn from the *other* domains' corpora.

**Training said ASR 1.073 and rising. The probe said 0.965 — unchanged from v4.**

Cause: `padding="max_length"` to 512 tokens meant most of every replay batch was
`<pad>`, and the penalty ran over every position. The adapter learned to be quiet on
padding, which costs the SFT loss nothing and transfers nowhere.

The lesson generalised beyond this adapter: when a cheap in-loop metric and the real
instrument disagree, the in-loop metric is measuring something adjacent.

---

## v5 — `L_inert` on in-domain prompt tokens ❌ FAILED

λ=0.05, corpus v4, astral only. Penalised `‖(BA)h‖` where `labels == -100`.

With `completion_only_loss=True` on a single-domain corpus, those are the **in-domain
prompt tokens** — no out-of-domain token ever entered the penalty, so the adapter
could not learn *when* to be quiet, only *how much*.

Measured energy vs v4:

```
astral 0.844  postgresql 0.837  duckdb 0.842  financial 0.849  general 0.846
```

A clean **uniform 0.843× scaling** with 1.4% spread and ASR unchanged at 0.96 — i.e.
a lower effective α wearing a different name, which the 2048-token stacking
benchmark had already shown is dilution (astral retained 90.3% at α=128, 57.7% at
α/√3).

The mechanism was never the problem — 15.7% movement at 1.4% spread is a penalty
working exactly as written. It was aimed at the wrong tokens.

---

## v4 — completion-only loss + verified reservation + replay ✅ CANONICAL

`CANON.ADAPTER_VERSION = "v4"`. Domains: astral, postgresql, duckdb, financial.

### Added

- **⭐ Response-only completion loss.** Prompt tokens masked to `-100`. Before this,
  **48.4% of every batch was prompt text** the model was being scored on
  reproducing — 55.2% for astral specifically. The trainer now asserts the mask is
  non-empty rather than trusting the flag.
- **Reserve-by-removal, then VERIFY.** `reserve_eval_constructs.py` strips reserved
  families and then counts residual occurrences. Verification is load-bearing: it
  caught `Protocol` still leaking at **103 hits** after its family was removed,
  because `modern_typing` also emitted it. Removal alone proves nothing.
- **General replay / retention mixing** (~10% anchors).
- **duckdb** as a fourth domain.
- **Geometry recording** — `‖dW‖/‖W‖` and predicted merge error written next to the
  adapter.

### Removed

- **The contaminated gate.** v3 trained **32 of the held-out gate's 45 steps (71%)**,
  because `DISTINCT ON` / `LATERAL` / `TaskGroup` are the obvious contents of an
  "Advanced PostgreSQL" and a "Modern Python" chapter. Corpora dropped 214 (postgres)
  and 172 (astral) records.

### Measured

| | v3 | v4 |
| :--- | ---: | ---: |
| held-out gate edge vs base | −0.1000 | **−0.0333** (not significant) |
| α=96 vs α=128 | differ by 0.0445 | **identical** — window widened |
| `‖dW‖/‖W‖` | — | 0.0758 pg / 0.0754 astral |
| predicted merge error | — | ~2.2% |

The α-window widening is the **Null-Space Window Expansion**: replay anchors drive
`(BA)·x_general → 0`, which raises `α_max` and turns α from a knife-edge into a
plateau.

---

## v3 — corpus expansion ⚠️ SUPERSEDED

Expanded generators for astral and postgresql. **Contaminated the held-out gate** —
the reserved constructs were chosen by scanning v2 for absences, and v3 then added
exactly those constructs. Scanning for absence is only valid until the next corpus
revision.

---

## v2 — corpus expansion ⚠️ SUPERSEDED

Larger corpora, prompt-inclusive loss. Never benchmark against v2 except as an
explicit, labelled ablation: it predates the completion-only fix *and* the
contamination removal, so it differs on two axes at once.

---

## v1 — first adapters ⚠️ SUPERSEDED

astral, postgresql, financial. No suffix in `results/adapters/`.

---

## Techniques evaluated and REJECTED (not in any shipping version)

| technique | version tried | outcome |
| :--- | :--- | :--- |
| Weight-space orthogonality penalty (§13) | v3-era | constraint measured **1.000** — already satisfied, penalty had nothing to do |
| α/√K scaling for stacking | v4 eval | **refuted at 2048 tokens** — astral retained 90.3% at α=128 vs 57.7% at α/√3. Dilution, not repair |
| `L_inert` (three variants) | v5, v5b, v5c | no validated selectivity gain; see entries above |
| Four geometric predictors of stacking damage | v4 eval | weight overlap, activation cosine, Merit, Conflict — all near-uniform, none predictive |
| PiSSA / SVD basis projection | v2-era | premise measured the wrong quantity; line closed |
| Merging corpora instead of stacking adapters | v4 | **stacking wins** by 18.6pp astral / 16.6pp postgres (rank-24 control still owed) |
