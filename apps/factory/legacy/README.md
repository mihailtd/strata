# legacy — frozen historical record, do not run

Everything in this directory is superseded. It is kept for provenance —
so a past decision's reasoning and its actual code are in the same place —
not because any of it should be run again.

**These files were NOT import-fixed or path-fixed during the move into
`apps/factory/legacy/`.** They still reference `runtime.*` modules and
`data/<domain>/...` paths the way they did in their original location
(`scripts/old/` or `scripts/train/`). They will not run without manual
path/import repair. If you genuinely need to resurrect one, treat that as
its own deliberate, labelled task — don't casually "just fix the import."

## What's here and why it's retired

From `scripts/train/` (superseded by the current `train_expert.py` /
`train_mtp_adapters_fleet.py`):

- **`export_adapter.py`** — the "m1" methodology: trains against a 4-bit NF4
  base. Measurably worse than m2's bf16-base approach (financial expert:
  -5.00pp vs +4.17pp) — see `train_expert.py`'s own docstring.
- **`train_mtp_adapter.py`** — single-domain, single-invocation MTP adapter
  trainer. Superseded by `train_mtp_adapters_fleet.py`, which trains the
  whole v7 fleet in one run with versioned output paths.
- **`train_all_6_domains_real.py`** — a 30-step smoke-test script, not a
  production trainer. Also has a known domain-naming bug (writes
  `m2_financial_planning_r8a128_v7_real` instead of the canonical
  `m2_financial_*` stem every other trainer uses).

From `scripts/old/` (already retired before this move; see the original
analysis in git history at `scripts/old/README.md` for the full detail —
summarized here):

- **`evaluate_novel_adapter.py`, `evaluate_astral_models.py`,
  `evaluate_postgres_adapter.py`** — eval harnesses; evaluation now lives in
  `benchmarks/`.
- **`compare_init_schemes.py`** — the experiment that retired PiSSA/OLoRA/CorDA
  init schemes (see `docs/DECISIONS.md`).
- **`add_opinionated_sft_data.py`, `build_financial_planning_dataset.py`,
  `clean_financial_training_data.py`** — early corpus work, superseded by
  `apps/factory/corpus/`.
- **`finetune_novel_adapter.py`** — the m1-methodology sibling exploring novel
  LoRA architectures (Tucker factorization, velocity-masked SFT); "most of the
  claims made for them did not survive benchmarking."
- **`train_with_orthogonality_penalty.py`, `extract_svd_basis.py`,
  `project_adapter_to_basis.py`** — the shared-basis/orthogonality-penalty
  experiment line, retired on geometric grounds (`docs/DECISIONS.md`).
- **`run_datagen.py`** — early harness plumbing for LLM-generated datasets via
  a local llama-server.
- **`export_mlflow_data.py`** — one-off MLflow archival utility.

## Provenance

These files moved twice: `scripts/old/*.py` and 3 files from `scripts/train/`
→ `apps/factory/legacy/` (this consolidation). Use `git log --follow` on a
given filename to see its full history through both moves.
