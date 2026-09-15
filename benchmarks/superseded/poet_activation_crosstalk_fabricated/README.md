# Fabricated POET activation cross-talk probe (milder LV-GLasso variant)

One script (formerly `experiments/factory/geometry/poet_activation_crosstalk/probe_poet_activation_crosstalk.py`) that generated "activation probe inputs" as `X = torch.randn(...)` (its own comment: "Synthetic activation probe inputs"), multiplied by real trained v4 LoRA deltas, and reported the result as measuring "dynamic activation perturbations... on Qwen3.5-4B." Flagged as Critical #4 in `docs/EXPERIMENT_REAUDIT_2026-09.md` — a milder version of the same real-weights/fake-activations pattern as the retired LV-GLasso cluster.

One of the four reported "cross-talk reduction" ratios (`astral` vs `financial`, 0.96×) was below 1.0 — filtering made the synthetic cross-talk slightly *worse* in that pair, which in hindsight should have been a signal that this input wasn't actually exercising the notch mechanism meaningfully.

`poet_crosstalk_fast` and `compute_adapter_delta` are generic, correct math (POET low-rank+sparse decomposition; real LoRA delta reconstruction) — not fabricated, only what was fed into them. Their unit tests (`test_poet_activation_crosstalk.py`, on hand-planted synthetic ground truth, never claimed to represent a real model) still pass and are kept here alongside for provenance, though no longer collected by pytest.

## Replaced by

A sibling script in the same cluster already did this correctly and was **not** retired: [`experiments/factory/geometry/poet_activation_crosstalk/probe_4way_poet_notch.py`](../../../experiments/factory/geometry/poet_activation_crosstalk/probe_4way_poet_notch.py) — real Qwen3.5-4B, real forward hooks capturing real activations, real v4 adapters folded via the real `WeightFoldingEngine`, real prompts across 5 domains (astral, postgresql, duckdb, financial, general). Its real, already-computed result (`results/benchmarks/poet_4way_stacking_results.json`) is a **0.15%–0.22% activation-energy reduction** from notch-filtering a real 4-way stack — nowhere near this script's fabricated "1.4×–5.4× cross-talk reduction." See that cluster's rewritten README for the full real table.
