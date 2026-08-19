# scripts/

**Version numbers live in `src/gnn_experiment/canon.py`, not in filenames.**

That is the whole point of this layout. `build_astral_v3_corpus.py` sitting next to
`build_astral_corpus.py` is how a stale script gets picked by whoever (human or
agent) reads the directory listing and guesses. There is now exactly one script per
job, and the version it targets is imported.

Anything superseded lives in [`old/`](old/) and is **not** to be used or copied from.

---

## The pipeline, in order

```text
1. corpus/   build the training data
2. train/    train the adapter
3. audit/    verify nothing drifted
4. serve/    run it
```

### 1. `corpus/` — build training data

| script | what it does |
| :--- | :--- |
| `build_astral_corpus.py` | astral (uv / ruff / modern Python) corpus |
| `build_postgresql_corpus.py` | postgresql corpus |
| `build_financial_corpus.py` | financial planning corpus |
| `build_duckdb_corpus.py` | duckdb corpus (from the epubs) |
| `extract_duckdb_epubs.py` | epub → text, feeds `build_duckdb_corpus.py` |
| `build_*_applied_examples.py` | applied/agentic examples folded into the above |
| `build_financial_speculation_prompts.py` | prompts for the speculative-decoding matrix |
| **`reserve_eval_constructs.py`** | **the contamination gate — see below** |

**`reserve_eval_constructs.py` is not optional.** It removes the reserved
evaluation families from the corpora and then *verifies* zero residual
occurrences. Removal without verification is worthless: it once caught `Protocol`
still leaking at 103 hits because an unreserved generator also emitted it.

Any new corpus generator must be re-run through it, or the held-out gate silently
stops being held out.

### 2. `train/` — train adapters

| script | what it does |
| :--- | :--- |
| **`train_expert.py`** | **the trainer.** completion-only loss, retention mixing, records `\|dW\|/\|W\|` and merge error |
| `calibrate_expert_alpha.py` | per-adapter admissible α window |
| `train_mtp_adapter.py` | multi-token-prediction draft head |
| `export_adapter.py` | export/merge an adapter |

### 3. `audit/` — verify before you trust a number

| script | what it does |
| :--- | :--- |
| **`check_canon.py`** | **fails on any benchmark that hard-codes a decode budget < 2048 or a legacy adapter.** Run it before quoting any result. |
| `audit_adapters.py` | adapter inventory and geometry |
| `audit_eval_rubrics.py` | rubric sanity |
| `check_gpu.py` | ROCm / GPU visibility preflight |
| `analyze_loss_curves.py` | training-curve inspection |

### 4. `serve/` — run it

| script | what it does |
| :--- | :--- |
| `run_openai_api_server.py` | OpenAI-compatible server |
| `test_openai_api_server.py` | server smoke test |
| `chat.py` | interactive CLI |

---

## Rules

1. **Never hard-code a decode budget or an adapter version.** Import them:
   ```python
   from gnn_experiment.canon import CANON, adapter_path
   ```
   `check_canon.py` enforces this and exits non-zero on violation.

2. **Never write a fallback chain for adapter paths.** This pattern —
   ```python
   for d in [v4_dir, v2_dir, legacy_dir]:
       if d.exists(): use(d); break     # WRONG
   ```
   is how a stale adapter survives a corpus rebuild unnoticed. `adapter_path()`
   raises `FileNotFoundError` instead. A benchmark that cannot find its adapter
   must crash, not quietly measure the wrong thing.

3. **Stamp every results artifact** with `CANON.stamp()`. A results file that does
   not record what produced it cannot be compared to any other results file.

4. **GPU runs:** always `timeout`, always `python -u` straight to a log file, and
   check the log rather than waiting on the process.

---

## `old/`

Superseded. Kept for provenance only — do not run, do not copy patterns from.

`add_opinionated_sft_data.py`, `build_financial_planning_dataset.py`,
`clean_financial_training_data.py`, `compare_init_schemes.py`,
`evaluate_astral_models.py`, `evaluate_novel_adapter.py`,
`evaluate_postgres_adapter.py`, `export_mlflow_data.py`, `extract_svd_basis.py`,
`finetune_novel_adapter.py`, `project_adapter_to_basis.py`, `run_datagen.py`,
`train_with_orthogonality_penalty.py`

Evaluation moved to `benchmarks/`. The SVD/basis-projection pair belongs to the
PiSSA line, which closed. `train_with_orthogonality_penalty.py` implements the
weight-space orthogonality penalty that measured a ratio of 1.000 — the constraint
was already satisfied, so the penalty had nothing to do.
