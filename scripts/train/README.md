# scripts/train/ — train adapters

## `train_expert.py` is the trainer

Formerly `train_expert_CURRENT_m2.py`. The "CURRENT" in the old name was doing the
job that this directory now does.

```bash
uv run --env-file .env python scripts/train/train_expert.py --domain astral
```

Always `--env-file .env` — a bare `uv run` looks GPU-less and will silently train
on CPU.

### What it does that matters

- **Completion-only loss.** Prompt tokens are masked to `-100`. Before this, 48.4%
  of every batch was prompt text the model was being scored on reproducing —
  55.2% for the astral corpus specifically. The trainer asserts the mask is
  non-empty rather than trusting the flag.
- **Retention mixing** — suppresses `(BA)h` on general constructs, which widens the
  admissible α window.
- **Records geometry**: `|dW|/|W|` and predicted bf16 merge error, written next to
  the adapter. `merge_err ≈ 0.167/(|dW|/|W|)`, so a *bigger* perturbation is
  represented *more* faithfully — precision is not what limits α.

Train on `training_data_v4.jsonl`. Never v3. See [`../corpus/`](../corpus/).

| script | what it does |
| :--- | :--- |
| `train_expert.py` | **the trainer** |
| `calibrate_expert_alpha.py` | per-adapter admissible α window (1 train + N evals) |
| `train_mtp_adapter.py` | multi-token-prediction draft head |
| `export_adapter.py` | export / merge an adapter |

## α, as currently measured

α=128 is canonical (`CANON.LORA_ALPHA`). Measured at 2048 tokens, scaling α **down**
for stacking is dilution, not repair — the 3-way stack retains 90.3% of astral's
gain at α=128, 57.7% at α/√3, and 27.5% at α/3. Do not apply a √K rule.
