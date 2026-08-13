# Superseded scripts

Kept for provenance only. **Do not cite their results and do not use them for new
work.** Each file's docstring records why it was retired.

| script | why |
| :--- | :--- |
| `benchmark_id_kron_vs_stock_lora.py` | Compared four pre-existing adapters of unknown provenance, reported id_kron at "r=64, 49.8M" when the adapter it loaded was rank_total=8 / 6.26M — inverting its own headline. Replaced by `eval_controlled_headtohead.py`. |
| `benchmark_mtp_folding_sweep.py` | Trained throwaway 30-step adapters inline and never loaded the 150-step ones on disk; n=41 tokens over 4 prompts; scored top-1 rather than accepted prefix; emitted hardcoded string literals as if they were measurements. Replaced by `benchmark_mtp_head_adapter_acceptance.py`, which reached the **same conclusion on sound evidence**: n=160 draft events/condition, the 150-step adapters, accepted-prefix scoring — un-adapted tau 2.456 vs 1.531–1.988 for all five adapters, every CI excluding zero. **Do not retry:** a draft head must agree with its backbone, not be domain-fluent, so this needs a new training objective (distil backbone outputs), not tuning. Note this closes *adapting the MTP head* only — in-place folding itself remains one of the repo's best results (1.83× vs the peft wrapper on the backbone). |
| `train_financial_adapter.py` | Financial-only trainer, and the ONLY script applying Liger kernels — so the financial expert was trained differently from astral/postgres while all three were compared as matched. Replaced by `train_expert_CURRENT_m2.py`. |
