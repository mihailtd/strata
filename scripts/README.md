# scripts/

**Version numbers live in `runtime_common.canon.CANON.ADAPTER_VERSION`, not in
filenames.** That is the whole point of this layout: there is exactly one
script per job, and the version it targets is imported, not guessed from a
directory listing.

The corpus-building, adapter-training, and superseded-experiment scripts that
used to live here (`corpus/`, `train/`, `old/`) moved to
[`apps/factory/`](../apps/factory/) — see its README for the full pipeline.
Nothing training-related remains in this directory; only the pieces that
operate on the whole repo (auditing, serving) stay here.

---

## The pipeline, in order

```text
1. apps/factory/corpus/   build the training data
2. apps/factory/          train the adapter
3. scripts/audit/         verify nothing drifted
4. scripts/serve/         run it
```

### `audit/` — verify before you trust a number

| script | what it does |
| :--- | :--- |
| **`check_canon.py`** | **fails on any benchmark that hard-codes a decode budget < 2048 or a legacy adapter.** Run it before quoting any result. |
| `audit_adapters.py` | adapter inventory and geometry |
| `audit_eval_rubrics.py` | rubric sanity |
| `check_gpu.py` | ROCm / GPU visibility preflight |
| `analyze_loss_curves.py` | training-curve inspection |

### `serve/` — run it

| script | what it does |
| :--- | :--- |
| `run_openai_api_server.py` | OpenAI-compatible server |
| `test_openai_api_server.py` | server smoke test |
| `chat.py` | interactive CLI |

---

## Rules

1. **Never hard-code a decode budget or an adapter version.** Import them:
   ```python
   from runtime_common.canon import CANON, adapter_path
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
