# LV-GLasso, redone with real forward-pass activations

Stage 1: a real analytical re-run answering the question the retired, fabricated cluster claimed to answer — this time with real model activations, not synthetic noise. See [`docs/EXPERIMENT_REAUDIT_2026-09.md`](../../../../docs/EXPERIMENT_REAUDIT_2026-09.md) Critical #2 and [`docs/DECISIONS.md`](../../../../docs/DECISIONS.md) §68/§72.

## Why this exists

[`benchmarks/superseded/latent_variable_glasso_fabricated/`](../../../../benchmarks/superseded/latent_variable_glasso_fabricated/) built its precision-graph analysis on a synthetic input activation matrix (`torch.randn` shared-drift + noise) multiplied by real trained LoRA weights. [`experiments/factory/geometry/surgical_notch_sweep/`](../surgical_notch_sweep/) answered the practical "does the shipped notch filter work" question for real using pure weight-space math — no activations needed — and found 96/96 MLP modules exceed the conflict gate, contradicting the fabricated "1 of 512" claim. But neither ever ran the model and captured what real per-token forward-pass activation covariance actually looks like, which was the *original* cluster's whole premise. This closes that gap.

## Method — real forward pass, real hooks, no synthetic data anywhere

Follows the same real-hook methodology already validated in [`experiments/factory/geometry/activation_inertia_probe/probe_stacking_merit.py`](../activation_inertia_probe/probe_stacking_merit.py): load the real base model once (no PEFT wrapper — attaching an adapter changes the hidden state, and every adapter must see the *same* h for the comparison to be well-defined), register real forward hooks capturing the real per-token input activation at every LoRA-targeted module common to 4 real, current (v7) adapters (astral, postgresql, duckdb, financial), run real forward passes over real domain-representative prompts (reused verbatim from `probe_stacking_merit.py`, including its deliberate postgresql/duckdb collision prompt), and for each captured real hidden state compute each adapter's real `delta = scale * (h @ A.T) @ B.T` — the identical math the fabricated script used, only the input is now a real captured activation, never `torch.randn(...)`.

`solve_lv_glasso_admm` / `soft_threshold` are copied verbatim from the retired script — that machinery was always real, generic linear algebra; only its input was fabricated.

## Result (2026-09-13, Qwen3.5-4B, 4 adapters × 128 common modules = p=512 features, 16 real prompts, 451 real tokens)

| | Fabricated (retired) claim | `surgical_notch_sweep` (real weight-space) | **This (real activations)** |
| :--- | :--- | :--- | :--- |
| Method | Synthetic activations × real weights | Real weights only, no activations | **Real activations × real weights** |
| rank(L) | 299 | n/a | **282** |
| S sparsity | 99.998% | n/a | **99.9847%** |
| Conflicts found | ~1 edge (of 512 modules) | 96 of 96 MLP modules (100%) | **20 edges (of 130,816 possible pairs)** |

Full edge list (top 15 by \|S\|) in [`results/benchmarks/latent_variable_glasso_real_activations.json`](../../../../results/benchmarks/latent_variable_glasso_real_activations.json). All 20 real conflict edges are at specific `(adapter, layer, module)` pairs — e.g. `astral::L27.self_attn.o_proj ↔ financial::L27.self_attn.o_proj` (S=-1.99, the strongest), several `mlp.down_proj`/`self_attn.o_proj` pairs at scattered layers, both cross-adapter (different domains, same layer/module) and within-adapter (same domain, different layers) edges.

## This does not simply "confirm" or "refute" either prior finding — it answers a different question than `surgical_notch_sweep`, honestly

- **`surgical_notch_sweep`** asks, per MLP weight matrix: *does any output neuron's row have anomalously high crosstalk relative to that matrix's own average?* Answer: yes, for all 96 matrices tested — every module has *some* internally distinguishable channel. That is a per-module, within-module question, and pure weight-space (no activations, no shared-factor removal).
- **This probe** asks, across the whole 512-feature activation-response space: *after removing the dominant shared factor (rank 282), which specific `(adapter, module)` pairs still show statistically distinguishable direct correlation?* Answer: only 20 out of 130,816 possible pairs. That is a cross-feature, shared-factor-controlled question.

Both can be true at once, and the evidence here says they are: every module has *some* internal outlier structure (matches `surgical_notch_sweep`), while *specific pairwise cross-talk that survives controlling for the shared foundation factor* is genuinely rare (rhymes with the ORIGINAL hypothesis's shape — though not its specific fabricated numbers). **Neither weight-space analysis measures what the retired cluster claimed to measure** (real per-token activation covariance) — this probe is the first one that actually does.

## An honest, slightly uncomfortable note: rank(L)=282 is suspiciously close to the fabricated rank(L)=299

This is *not* vindication of the fabrication — the input to that number was still `torch.randn(...)`, unconditionally fake. But it's worth being honest that the real re-run's qualitative shape (a large shared low-rank-ish factor, plus a small sparse residual) landed close to what was claimed, for what is plausibly a real reason: every adapter shares the same base model, tokenizer, and "English + code" foundation, so a large shared factor is exactly what you'd expect from real activations too, independent of whether anyone ever measured it honestly. The specific *fabricated* numbers (299, "1 edge", 99.998%) were never a real measurement — but the *idea* that a shared foundation factor dominates was not itself absurd, just asserted without evidence at the time.

## Honest limitations (what would strengthen or falsify this further)

- **n < p.** 451 real token samples for 512 features means the empirical correlation matrix is rank-deficient by construction (rank ≤ ~450) before any ADMM regularization is applied — `rank(L)=282` sits within that hard ceiling, but more real samples (longer real prompts, more of them, or a real generation loop instead of prompt-only forward passes) would be needed to trust the exact rank value with real confidence, rather than an artifact of sample size. The fabricated original had the same n<p problem — it just doesn't matter as much when the input was fake to begin with.
- **Prompt-only forward pass**, not real autoregressive generation — the retired cluster's premise was about activations during real inference, and a prompt-only forward pass is a real activation but not necessarily representative of what activations look like deep into a real generation.
- **4 adapters (astral, postgresql, duckdb, financial), not the full 6-domain fleet** (python_web, python_modern excluded, matching the original cluster's scope for comparability).
- **Regularization hyperparameters (`lambda1=0.05`, `lambda2=0.10`, `kappa=0.02`) are the retired script's defaults**, not tuned for this real data — a real hyperparameter sweep (e.g. eBIC-selected, matching the rigor `latent_variable_glasso`'s own README once claimed for a different context) could change which specific edges surface, though the overall sparsity level is unlikely to move by orders of magnitude.

## Run

```bash
uv run --env-file .env python experiments/factory/geometry/latent_variable_glasso_real_activations/probe_lv_glasso_real_activations.py
```

Requires a GPU (loads the real 4B model + 4 real v7 adapters). Saves to `results/benchmarks/latent_variable_glasso_real_activations.json`.
