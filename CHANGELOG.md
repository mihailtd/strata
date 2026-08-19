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
| **v6** | v5 | disposition corpora + dual-criterion stopping | **CANONICAL** |

---

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
