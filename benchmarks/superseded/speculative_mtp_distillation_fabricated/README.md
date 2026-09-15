# Fabricated MTP draft-head distillation recommendation

One script (formerly `experiments/factory/speculative_mtp_distillation/`) that recommended a production training practice a real experiment already contradicts. Flagged as Critical #3 in `docs/EXPERIMENT_REAUDIT_2026-09.md`.

`benchmark_mtp_domain_distillation.py` operates entirely on a `SimulatedMTPDraftHead` (`d_model=512`, `vocab_size=1000`) — nowhere near Qwen3.5-4B's real dimensions (hidden_size=2560, vocab~151936) — with synthetic random teacher weights and random input tokens. It never loads the real model, never touches `apps/runtime/mtp_draft.py`, and never uses a real adapter. Even by its own fabricated yardstick, positional accuracy stayed at 0.8%–1.6% for every arm; the reported "success" (backbone cosine alignment 0.003 → 0.731) is a side-channel metric the Huber distillation loss directly optimizes for by construction, not evidence the draft head actually predicts better.

Despite this, the retired README's "Architectural Integration" section told readers to use `HuberDistillationLoss` when training domain-specialized MTP draft heads for the live engine.

## The real answer already existed and says the opposite

[`docs/DECISIONS.md` §63](../../../docs/DECISIONS.md) ran the real experiment: the real 1-layer EAGLE draft head, real domain fine-tuning, all 6 canonical v7 domains, real GPU throughput measurement. Result: **the stock, non-domain-adapted head beats every domain-tuned variant by 10%–34%** across all 6 domains, because the 1-layer head's limited capacity means any domain fine-tuning — robust loss or not — causes representation collapse. §63's decision: use the stock generalized MTP draft head for all speculative decoding across all domain experts, permanently.

**`HuberDistillationLoss` itself is not fabricated** — it's a real, generic, reusable loss function (`apps/runtime/robust_distill.py`), correctly used elsewhere by the honestly-labeled [`experiments/factory/robust_distillation/`](../../../experiments/factory/robust_distillation/) cluster ("Breakdown-Bounded Loss Functions in Noisy Synthetic SFT" — self-labeled synthetic in its own title). Only this cluster's specific claim — that Huber distillation rescues MTP draft-head domain adaptation for the live engine — is retracted.

**Action taken**: retracted the integration advice; `docs/DECISIONS.md` §63 is the authoritative answer on MTP draft-head domain adaptation. Nothing in production depends on this cluster's numbers (unlike the `weibull_hazard_gating` and `latent_variable_glasso` critical findings), so this was a documentation-only fix — no runtime code changed.
