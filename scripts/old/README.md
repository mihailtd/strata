# scripts/old/ — superseded

**Do not run these. Do not copy patterns from them.** Kept for provenance only.

If you are looking for how something is done now, it is in
[`../corpus/`](../corpus/), [`../train/`](../train/), [`../audit/`](../audit/), or
[`../serve/`](../serve/). Evaluation moved to `benchmarks/`.

| script | why it is here |
| :--- | :--- |
| `evaluate_astral_models.py`, `evaluate_novel_adapter.py`, `evaluate_postgres_adapter.py` | evaluation lives in `benchmarks/` now, with execution gating instead of keyword matching |
| `extract_svd_basis.py`, `project_adapter_to_basis.py` | the PiSSA line. The premise measured `W0`'s spectrum concentration rather than ΔW's alignment with it; corrected, and the line closed. |
| `train_with_orthogonality_penalty.py` | weight-space orthogonality penalty. The constraint measured a ratio of **1.000** — already satisfied, so the penalty had nothing to do. The untested variant is activation-space, not weight-space. |
| `compare_init_schemes.py` | init-scheme comparison, superseded by the α-window work |
| `build_financial_planning_dataset.py`, `clean_financial_training_data.py`, `add_opinionated_sft_data.py` | early corpus work, replaced by `../corpus/build_financial_corpus.py` |
| `run_datagen.py`, `export_mlflow_data.py` | early harness plumbing |
| `finetune_novel_adapter.py` | Tucker / velocity adapter variants. Most of the claims made for them did not survive benchmarking. |

## Why they are kept

Deleting them would erase the record of what was tried and why it did not work,
which is how a team re-runs the same dead end a month later. They are archived, not
endorsed.
