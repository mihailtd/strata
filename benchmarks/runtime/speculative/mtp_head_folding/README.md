# ❌ CLOSED: In-Place Weight Folding on the Native MTP Head

> **Tier: none — falsified.** A refuted hypothesis is not a contribution at any
> novelty tier. The *measurement* that closed it is 🔥 Applied Practice; the
> hypothesis it tested is dead. See `NOVELTY.md`.

This directory contains the experimental framework measuring what happens when you try to fine-tune a speculative draft head (MTP head) on a specific domain (like Financial or Astral data) and fold those weights in.

## ⚠️ Workflow Integration: When to Use This?

> [!CAUTION]
> **DO NOT USE THIS SCRIPT FOR DAILY TRAINING.** 
> This is a historical R&D benchmark that proved a profound architectural constraint. It is preserved here as generating knowledge, but it is **not** an active component of the production inference engine.

### The Architectural Discovery
This benchmark (`benchmark_mtp_head_adapter_acceptance.py`) measures Speculative Decoding Token Acceptance ($\tau$). 

It **measured** (n=160 draft events/condition, K=6, paired bootstrap, all five adapters' CIs excluding zero) that **you cannot adapt a draft head using standard domain next-token loss**. This is an empirical result on this stack, not a proof. A draft head's only job is to perfectly mimic the main backbone model so that its drafted tokens get accepted. If you make the draft head "fluent" in a domain, it forms its own opinions and disagrees with the backbone, causing the backbone to reject the drafts. This destroys the speculative decoding speedup (e.g., dropping $\tau$ from 2.456 down to 1.531).

### How to Apply This Knowledge
- **Weight Folding** itself remains highly active and incredibly fast (yielding 1.83x speedups) when applied to the **main backbone**.
- However, **adapting the MTP Head** via fine-tuning is a dead end. If you want to train an MTP head for a new domain in the future, this benchmark proved that you must write a completely new training objective based on **Distillation** (training the draft head to copy the backbone's exact logits), *not* standard fine-tuning.
