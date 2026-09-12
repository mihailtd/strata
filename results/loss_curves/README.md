# Per-step loss curves (diagnostic)

Captured 2026-08-17 to settle whether EMA early stopping or per-domain epoch
budgeting was worth implementing. **Both were rejected** — see
[`docs/DECISIONS.md`](../../docs/DECISIONS.md) §8.

```bash
uv run --env-file .env python scripts/train/train_expert.py \
    --domain astral --logging-steps 1 \
    --out results/loss_curves/adapters/astral \
    --loss-curve-out results/loss_curves/astral.json

uv run python apps/factory/analyze_loss_curves.py
```

`adapters/` holds the throwaway adapters those runs produced. **They are NOT the
live expert set** and are deliberately outside `results/adapters/` so
`audit_adapters.py` and the manifest never see them. The live experts remain
`results/adapters/m2_<domain>_r8a128`, untouched by this run.

## Initialisation-scheme curves (`*_pissa*.json`, `*_olora*.json`, `*_stock_a8.json`)

Added 2026-08-17 to settle whether PiSSA/OLoRA train faster — `docs/DECISIONS.md`
§2 had retired them **without a single run**. They do not: at matched alpha both
are inert (PiSSA −0.009, OLoRA +0.019, noise floor 0.0171), and neither reaches
the stock run's final loss in 150 steps.

```bash
uv run --env-file .env python scripts/train/train_expert.py \
    --domain astral --init-lora-weights pissa --alpha 8 --logging-steps 1 \
    --loss-curve-out results/loss_curves/astral_pissa_a8.json \
    --out results/adapters/probe_pissa_a8_astral

uv run python scripts/old/compare_init_schemes.py       # -> results/init_scheme_comparison.json
```

⚠️ **Always compare at matched alpha.** These schemes interact strongly with
scaling: at α=128 PiSSA reads +0.127 and OLoRA +0.557, which looks damning and is
almost entirely an artefact — 96.6% of OLoRA's deficit vanishes at α=8. The
`astral_stock_a8.json` control exists precisely so the α=8 comparison changes one
variable. Their probe adapters were deleted after analysis (1.3 GB, gitignored);
the curves are the evidence.
