# POET activation cross-talk: real 4-way stacking, real notch filtering

> **Theoretical Reference**: *Regressions in Covariances, Dependencies and Graphs* (Mohsen Pourahmadi & Reza Arabpour), Chapter 7.3 (§7.3.1 Low-Rank Plus Sparse Covariance).
> **Real result artifact**: [`results/benchmarks/poet_4way_stacking_results.json`](../../../../results/benchmarks/poet_4way_stacking_results.json).

## Correction (2026-09-12)

This README used to headline a "1.4×–5.4× cross-talk reduction" claim, computed by `probe_poet_activation_crosstalk.py` from **synthetic `torch.randn` activation inputs** multiplied by real trained v4 LoRA deltas — flagged as Critical #4 in [`docs/EXPERIMENT_REAUDIT_2026-09.md`](../../../../docs/EXPERIMENT_REAUDIT_2026-09.md). That script is retired to [`benchmarks/superseded/poet_activation_crosstalk_fabricated/`](../../../../benchmarks/superseded/poet_activation_crosstalk_fabricated/). One of its four reported ratios (astral vs financial, 0.96×) was even below 1.0 — filtering made the synthetic cross-talk slightly *worse*, a sign this input wasn't exercising anything real.

The cluster's other script, `probe_4way_poet_notch.py`, was **already doing this correctly** — real Qwen3.5-4B, real forward hooks, real v4 adapters — and had already been run, its real result just wasn't the one quoted here. This README now reports that real result instead.

## What this measures

Loads the real Qwen3.5-4B base model and folds all 4 canonical v4 domain experts (astral, postgresql, duckdb, financial) simultaneously via the real `WeightFoldingEngine`. Registers real forward hooks on every `mlp.down_proj`/`self_attn.o_proj` module to capture real hidden-state norms and real per-layer output perturbation, across 5 domains of real prompts (4 domain-specific + 1 general-knowledge control, 4 prompts each). Compares:

- **Raw 4-way stack**: all 4 experts folded unscaled (`scale_mode="none"`), no notch filtering.
- **POET-notched 4-way stack**: same 4 experts, but with the top-15 conflicting output channels per module (computed from real trained factors via `compute_poet_notch_mask_per_layer`, the same real-weight row-correlation math as production's `compute_surgical_notch_masks`) zeroed out.

Metric: relative activation perturbation energy `‖δ‖ / ‖h‖` (output perturbation norm over base hidden-state norm), averaged per domain.

## Real result

| Prompt domain | Raw 4-way energy | POET-notched energy | Reduction |
| :--- | ---: | ---: | ---: |
| astral | 0.5090 | 0.5082 | +0.16% |
| postgresql | 0.5139 | 0.5129 | +0.19% |
| duckdb | 0.5098 | 0.5090 | +0.15% |
| financial | 0.5061 | 0.5050 | +0.22% |
| general | 0.5123 | 0.5114 | +0.18% |

**The real effect is small: 0.15%–0.22% activation-energy reduction from notching**, consistently positive (never negative, unlike the retired script's 0.96× outlier) but nowhere near the fabricated "1.4×–5.4× cross-talk reduction" claim. This is directionally consistent with — and roughly the same order of magnitude as — [`experiments/factory/geometry/surgical_notch_sweep/`](../surgical_notch_sweep/)'s independent real-weight-space finding that the shipped notching mechanism removes a real but modest amount of interference (that probe measures crosstalk/signal *selectivity*, a different quantity from this one's raw energy reduction, but both agree: real, positive, small).

## Run

```bash
uv run --env-file .env python experiments/factory/geometry/poet_activation_crosstalk/probe_4way_poet_notch.py
```

Requires a GPU (loads the real 4B model + 4 real adapters). Saves to `results/benchmarks/poet_4way_stacking_results.json`.
