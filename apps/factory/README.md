# factory — The Factory

The repo's own name (see `docs/THE_FACTORY_FINE_TUNING_AND_GEOMETRY.md`) for
its whole PEFT/LoRA fine-tuning discipline: corpus quality, loss/backprop
math, and parameter geometry. This project is the actual implementation —
corpus pipeline, adapter training, quantization geometry — consolidated from
what used to be scattered across `scripts/train/`, `scripts/old/`,
`scripts/corpus/`, and a separate `training/` uv project.

**Independent uv project** — own `pyproject.toml`/`uv.lock`/`.venv`, pinned to
`torch==2.11.0+rocm7.2` (the same matched ROCm wheel trio as everything else
here) because Unsloth requires `torch<2.12.0`. Nothing else in this project
(transformers/peft/trl/liger-kernel) needs anything newer — verified by
actually resolving and running against it, not just reading version floors.

---

## Table of contents

- [What's here](#whats-here)
- [Quick start](#quick-start)
- [How to train one adapter](#how-to-train-one-adapter)
- [How to retrain the whole fleet on a version](#how-to-retrain-the-whole-fleet-on-a-version)
- [How to add a new domain](#how-to-add-a-new-domain)
- [How to update existing training data](#how-to-update-existing-training-data)
- [When to retrain (and when not to)](#when-to-retrain-and-when-not-to)
- [The 27B tier has two trainers, on purpose](#the-27b-tier-has-two-trainers-on-purpose)
- [Version changelog](#version-changelog)
- [Retired approaches](#retired-approaches)
- [Dependency architecture](#dependency-architecture)
- [Known issues](#known-issues)
- [Moon tasks](#moon-tasks)

---

## What's here

```
apps/factory/
  train_expert.py                  the canonical trainer (m2: bf16 base + Liger fused kernels)
  calibrate_expert_alpha.py        per-adapter alpha calibration
  train_mtp_adapters_fleet.py      MTP draft-head micro-adapters, whole v7 fleet
  train_all_9b_experts.py          orchestrator: 9B tier
  train_all_27b_experts.py         orchestrator: 27B tier, QLoRA (wraps train_expert.py)
  train_w4a16_27b_adapters.py      27B tier, independent GGUF/W4A16 loop (matches
                                    what apps/runtime-triton actually serves)
  train_all_ornith_35b_experts.py  Ornith-1.5 35B MoE tier
  leverage_quantization.py         outlier-aware leverage scoring for mixed-precision W4A16
  ssi_quantization.py              closed-form INT4 group-scale calibration (SSI)
  peft_compat.py                   training-only peft monkeypatches
  corpus/                          corpus-building pipeline (produces data/)
  training/                        Unsloth micro-probe benchmark
  data/                            training corpora: raw source material + training_data*.jsonl
                                    (evaluation_data*.jsonl stayed at root data/ -- consumed by
                                    benchmarks/ and apps/runtime's inference-time eval code)
  legacy/                          frozen, superseded -- see legacy/README.md, do not run
  tests/
```

---

## Quick start

```bash
cd apps/factory
uv sync
uv run pytest tests -v          # 15 tests, should all pass, no GPU needed
```

---

## How to train one adapter

```bash
cd apps/factory
uv run --env-file .env python train_expert.py --domain astral
```

That's it for the common case — no version flag needed. `train_expert.py`
defaults to `runtime_common.canon.CANON.ADAPTER_VERSION` (currently `v7`) for
both which corpus it reads and which adapter version it writes. Output lands
at `results/adapters/m2_astral_r8a128_v7`.

Valid `--domain` values: `astral`, `postgresql`, `duckdb`, `financial_planning`,
`python_modern`, `python_web`, plus the merged corpora `merged_sql` and
`merged_all`.

Useful flags (see `train_expert.py --help` for the full set — everything has
a docstring explaining *why*, not just what):

| flag | when to use it |
| :--- | :--- |
| `--out <dir>` | write somewhere other than the canonical `results/adapters/...` path (e.g. for a sweep/ablation you don't want to collide with the real adapter) |
| `--stop-at-dw-over-w 0.075` | stop by geometry (`\|\|dW\|\|/\|\|W\|\|` in the Goldilocks band, see `docs/THE_FACTORY_FINE_TUNING_AND_GEOMETRY.md` Ch.4) instead of a fixed step count -- **prefer this over `--max-steps`** |
| `--qlora` | 4-bit NF4 base -- needed for anything you can't fit in bf16 on 24GB (9B+) |
| `--dataset <path>` | override the domain's default corpus file (recorded in `regime.json` so the adapter never loses track of what trained it) |
| `--seed 42` | already the default; change only for a deliberate multi-seed variance check |
| `--v4` / `--v6` | **legacy opt-in only** -- train against a superseded corpus/adapter generation for a deliberate, labelled ablation. Never use these to "fix" a failure; if `--v7`'s corpus is missing a construct, fix the corpus, don't fall back. |

**After training**, calibrate its deployment alpha (this is a real step, not
optional polish — alpha is scaling, not personality):

```bash
uv run --env-file .env python calibrate_expert_alpha.py \
    --adapter results/adapters/m2_astral_r8a128_v7 \
    --alphas 32 64 96 128
```

---

## How to retrain the whole fleet on a version

Pick the orchestrator matching the model tier:

```bash
# 4B (default), all 6 domains, wraps train_expert.py --v7
uv run --env-file .env python train_all_9b_experts.py            # 9B tier
uv run --env-file .env python train_all_27b_experts.py           # 27B tier, QLoRA
uv run --env-file .env python train_w4a16_27b_adapters.py        # 27B tier, GGUF/W4A16
uv run --env-file .env python train_all_ornith_35b_experts.py    # Ornith-1.5 35B MoE
```

Each orchestrator skips a domain if its adapter directory already exists and
looks complete (`adapter_config.json` + weights) — pass `--force` to retrain
anyway. There is currently no single "retrain absolutely everything, every
tier" command; run the tier(s) you actually need. `train_all_9b_experts.py`
in particular has **no CLI flags at all** — it just runs the full 6-domain
pipeline unconditionally when invoked (careful running it by habit).

There is no `train_expert.py`-style fleet orchestrator for `calibrate_expert_alpha.py`
— re-run it by hand per adapter after a fleet retrain, or write one if you
find yourself doing this often enough to justify it.

---

## How to add a new domain

1. Write `corpus/build_<domain>_corpus.py`, emit
   `apps/factory/data/<domain>/training_data_v3.jsonl` (yes, start at v3 —
   see the [version changelog](#version-changelog) for why the numbering
   starts there).
2. Add the domain's reserved evaluation constructs to `RESERVED` in
   `corpus/reserve_eval_constructs.py`, then run it with `--write` and
   confirm every construct reports **CLEAN**. This is not optional — see
   `corpus/README.md`'s "do not skip it" section; skipping it is how a
   held-out eval gate silently stops being held out.
3. Add the domain to `_GEN_DOMAINS` in `train_expert.py` (and to
   `runtime_common.canon`'s `DOMAINS` tuple if it should be a first-class
   canonical domain, not just an ad hoc training target).
4. Train it: `uv run python train_expert.py --domain <domain>`.
5. If it needs a merged-corpus variant, add it to `merge_corpora.py` /
   `merge_domain_corpora.py`.

## How to update existing training data

1. Regenerate or hand-edit the domain's `corpus/build_<domain>_corpus.py`
   output.
2. **Always re-run `corpus/reserve_eval_constructs.py --write` after.** A
   corpus rebuild that skips this step can silently leak eval-only
   constructs back into training data (this happened once for real — see
   `corpus/README.md`).
3. Bump the corpus version number in the filename (`training_data_v6.jsonl`
   → `training_data_v7.jsonl`, say) — **never overwrite a version in place.**
   An in-place overwrite is how `m2_python_modern_r8a128_v6` got silently
   retrained on the wrong baseline once (see the `_gen_table` comment block
   in `train_expert.py`).
4. Point `train_expert.py`'s `DOMAINS_V{N}` table (or add a new
   `DOMAINS_V{N+1}` via `_gen_table`) at the new corpus version, and bump
   `runtime_common.canon.CANON.ADAPTER_VERSION` to match **only when you are
   ready for it to become the default for everyone** — that one line changes
   what every no-flag `train_expert.py` invocation does, repo-wide.
5. Retrain the affected domain(s), re-run `corpus/audit_corpora.py` (note:
   currently missing `python_modern`/`python_web` entries in its `RUNNABLE`
   dict — a pre-existing bug, add them if you need those domains audited),
   and compare against the previous version's held-out eval numbers before
   trusting the new adapter.

## When to retrain (and when not to)

Retrain a domain when:
- Its corpus changed (new records, a bug fix in generation, a rebalance).
- `CANON.ADAPTER_VERSION` advances and you want that domain on the new
  generation.
- `calibrate_expert_alpha.py` or a benchmark shows it's drifted outside its
  Goldilocks geometry band (`0.035 ≤ ||dW||/||W|| ≤ 0.100`).

Don't retrain to "fix" a bad eval result without first checking whether the
corpus, the eval rubric, or a hyperparameter is the actual cause — a
retrain with the same broken corpus just reproduces the same result with
different noise. Check `docs/DECISIONS.md` first; there's a real chance
whatever you're about to try has already been tried and refuted (see
[Retired approaches](#retired-approaches)).

---

## The 27B tier has two trainers, on purpose

`train_all_27b_experts.py` (QLoRA, wraps the canonical `train_expert.py`,
consistent with the 9B tier) and `train_w4a16_27b_adapters.py` (an
independent loop that loads the GGUF blob directly and trains W4A16,
matching what `apps/runtime-triton` actually serves) used to write to the
*same* output directories via completely different implementations. Their
output suffixes are now disambiguated: `_v7_27b_qlora` vs. `_v7_27b_w4a16`.
Pick whichever matches what you're about to deploy — if in doubt, W4A16 is
the one that actually matches production serving format.

---

## Version changelog

Adapter version and corpus version are **off by one and always have been** —
see the boxed comment in `train_expert.py` for the exact mapping. This table
is the human-readable version.

| adapter | corpus | methodology | notes |
| :--- | :--- | :--- | :--- |
| v2 / v3 | — | m1 (4-bit NF4 base) | **LEGACY.** Predate the completion-only-loss fix (48.4% of every v2 training batch was prompt tokens, not answer tokens) and contamination removal. Never benchmark against these except as an explicit, labelled ablation. |
| v4 | v3 minus reserved eval constructs | m1 → m2 transition | `export_adapter.py`'s m1 (4-bit NF4, no fused kernels) measurably underperforms m2 (bf16 + Liger): financial expert moved from -5.00pp to +4.17pp on its own domain switching methodologies alone. v4 is the last version with an m1 lineage; `train_expert.py` (m2) is the trainer from here on. |
| *(v5 — does not exist as an adapter)* | v5 = merged disposition corpus | — | The number v5 is used for two **unrelated** things: the failed `L_inert` adapter line (weights deleted, never shipped) and the corpus that rebuilt everything as disposition data. Confusing on purpose only in the sense that nobody renumbered it after the fact — don't assume "adapter v5" means anything. |
| v6 | v5 | m2 | Corpora rebuilt as DISPOSITION data (situation → right approach + the rejected alternative named), plus the uv/ruff/ty command corpus, the python_modern/python_web split, deduplication, and dual-criterion geometric stopping. Measured 0.581 vs. v4's 0.532 on held-out disposition evals (n=31). |
| **v7 (current — `CANON.ADAPTER_VERSION`)** | v6 | m2 | Round-2 corpus rebuild: astral command families, postgres asyncpg, duckdb analytics, python capability records. This is what `train_expert.py` targets with no flags. |

**Methodology, separately from version number:**
- **m1** — 4-bit NF4 base, no fused kernels. `legacy/export_adapter.py`,
  `legacy/finetune_novel_adapter.py`. Adapters here learn a correction to
  *quantized* weights, then get folded into bf16 ones — a seam that measures
  worse than training directly in bf16.
- **m2 (current)** — bf16 base + Liger fused kernels (`fused_linear_cross_entropy`,
  `rms_norm`, `swiglu`; rope off). `train_expert.py`. Hyperparameters fixed
  across every domain so experts stay comparable: r=8, alpha=128 (scaling 16),
  7 LoRA projections, 150 steps (or geometric stop), batch 2, grad-accum 2,
  lr 2e-4, cosine schedule, max_length 512.

When the methodology changes again (m3), the convention (see
`train_expert.py`'s own docstring) is: rename this file to
`train_expert_m2.py` and move it into `legacy/`, create a new
`train_expert_CURRENT_m3.py`, and give m3 adapters their own
`m3_<domain>_r<rank>a<alpha>` naming — so methodology is always readable from
both the script name and the adapter name, and results from different
methodologies can never be silently compared.

---

## Retired approaches

Full detail and measurements in `docs/DECISIONS.md`; code in `legacy/`
(frozen, not import-fixed — see `legacy/README.md`).

| approach | files | verdict |
| :--- | :--- | :--- |
| PiSSA / OLoRA / CorDA init | `legacy/compare_init_schemes.py` | §2 — didn't reach target loss faster than stock LoRA init; retired |
| Shared SVD basis across adapters | `legacy/extract_svd_basis.py`, `legacy/project_adapter_to_basis.py` | §3 — line closed |
| Explicit subspace-orthogonality penalty | `legacy/train_with_orthogonality_penalty.py` | §13 — measured overlap ratio was already 1.000 (baseline satisfied the constraint already); the penalty had nothing real to remove, cost +0.026–0.049 loss and 1.9x wall-clock |
| Novel LoRA architectures (Tucker factorization, velocity-masked SFT) | `legacy/finetune_novel_adapter.py` | "most of the claims made for them did not survive benchmarking" |
| m1 methodology (4-bit NF4 base) | `legacy/export_adapter.py` | superseded by m2 (bf16 + Liger), see version table above |

---

## Dependency architecture: two kinds of "shared"

- **`runtime-common`** (canon, gpu_preflight) — a real, formal dependency
  (`[tool.uv.sources]` path dependency). Zero torch dependency on its side,
  so no conflict with this project's `torch==2.11.0` pin.
- **`apps/runtime`** (novel_peft, mtp_draft, w4a16_loader, training_db,
  micro_probe/dataset, utils/logger, eval/eval_suite) — consumed as *source*
  via `_bootstrap.py` (adds `apps/` to `sys.path`), never as a package
  dependency. Two different reasons a module ends up here instead of moving
  into `apps/factory`:

  1. **It's genuinely inference-time code.** `novel_peft.py` (`FoldableExpert`,
     `WeightFoldingEngine`) folds trained adapters into the base model for
     serving; `mtp_draft.py` (`Qwen35MTPDraftHead`) is the draft head the
     speculative-decoding server runs; `w4a16_loader.py` is the same
     quantized-weight loader `apps/runtime/native_27b_engine.py` uses.
     Factory needs these because it trains *against* the exact classes the
     server later runs — they belong in `apps/runtime` forever, not here.
  2. **Something else in `apps/runtime` re-exports it as public API.**
     `micro_probe/dataset.py` and `eval/eval_suite.py` are both re-exported
     by their package's own `__init__.py` (`apps/runtime/micro_probe/__init__.py`,
     `apps/runtime/eval/__init__.py`) and consumed by several benchmarks
     directly — moving them would break those call sites.

  `training_db.py` and `utils/logger.py` are general-purpose infra
  (training-run bookkeeping, structured metric logging) also read/used by
  `apps/runtime/server.py` and various benchmarks — shared because the thing
  they do (log a run, log a metric) isn't specific to training.

  **The one real "could move" candidate**: `apps/runtime/datagen/` (an
  LLM-driven question→verify→answer pipeline that turns raw docs into Q&A
  training data, via a local llama-server). Its only remaining consumer is
  `legacy/run_datagen.py` (retired). It's pinned in `apps/runtime` only
  because `micro_probe/dataset.py` (which can't move, see above) imports two
  of its sibling files (`chunk.py`, `schema.py`). Not worth refactoring for
  dead code today, but if `datagen/` ever gets a live consumer again, that's
  the dependency to cut first.

If you add a new import from `apps/runtime` here, ask: is this genuinely
training-only (→ move it into this project), does something else already
depend on it (→ leave it in `apps/runtime`, consume via `_bootstrap`), or is
it live inference-time code (→ definitely leave it in `apps/runtime`)?

---

## Known issues

- `corpus/audit_corpora.py`'s `RUNNABLE` dict is missing `python_modern` and
  `python_web` keys — auditing either domain crashes with a `KeyError`.
  Pre-existing, not caused by the move into this project.
- `corpus/` itself has been stale since 2026-08-21 relative to `data/`, which
  has files as recent as v6 corpora with no obvious corresponding commit —
  likely generated by uncommitted local runs. Verify against
  `docs/CORPUS_DESIGN.md` before trusting a fresh corpus-pipeline run.
- `legacy/*.py` files are **not** import-fixed — they still reference
  `runtime.*` and `data/<domain>/...` the way they did in their original
  location and will not run without manual repair. This is deliberate (see
  `legacy/README.md`), not an oversight.
- `moon run factory:*` is slow on a cold cache (a known, only partially
  understood moon limitation tied to this project's 16GB `.venv` — see the
  repo's top-level notes). Use direct `uv run`/`pytest` instead.

---

## Running things

```bash
cd apps/factory
uv sync

# Corpus pipeline (stale since 2026-08-21 -- verify against docs/CORPUS_DESIGN.md
# before trusting its output; last known-good corpus version is v6)
uv run python corpus/build_astral_corpus.py
uv run python corpus/audit_corpora.py   # see Known issues -- python_modern/python_web will crash it

# Unsloth micro-probe
uv run --env-file .env python training/micro_probe_bench.py

# Sanity-check a freshly trained adapter by talking to it directly (no server)
uv run python chat.py --adapter results/adapters/m2_astral_r8a128_v7

# Inspect a training run's loss curve (converged? still descending? early-stop point?)
uv run python analyze_loss_curves.py --dir results/loss_curves
```

### Unsloth Studio (local web UI)

```bash
~/.unsloth/studio/unsloth_studio/bin/unsloth studio -p 8888
```

Confirms real GPU use via the startup log line: `Hardware detected: ROCm
(HIP ...) -- AMD Radeon RX 7900 XTX` (vs `Hardware detected: CPU training
backend` if something's misconfigured). Bound to localhost only by default —
pass `-H 0.0.0.0` to expose on the LAN, `--cloudflare` for a public HTTPS
tunnel (off by default, don't enable without knowing what you're exposing).
Other flags: `--disable-tools`, `--parallel N` (decode slots, default 4),
`--password <pw>`. Stop with Ctrl+C or `unsloth studio stop`.

### Metric logging

Benchmark/eval/training runs log structured metrics to zero-overhead JSON
Lines files in `results/` (e.g. `results/micro_probe_runs.jsonl`).

---

## Moon tasks

```bash
moon run factory:sync
moon run factory:test
moon run factory:lint
moon run factory:format
moon run factory:update-deps
```

(See [Known issues](#known-issues) — these are currently slow on a cold
cache; prefer direct `uv run` invocations if that matters to you right now.)
