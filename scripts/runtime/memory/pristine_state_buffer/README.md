# 🔥 Pristine State Buffer

This folder exists to document one of the most critical safety mechanics in the Runtime Engine. 

There are no standalone scripts to execute here because the **Pristine State Buffer** is a live architectural variable running deep inside the engine API (`WeightFoldingEngine` located at `src/gnn_experiment/novel_peft.py`).

## The Problem
When you run a live inference server that constantly swaps between different domains (e.g., answering a Financial question, then an Astral question), you have to mathematically "un-merge" the previous adapter before folding in the new one. 

If you try to "un-merge" an adapter by mathematically subtracting its weights in 16-bit precision (`bfloat16`), you accumulate floating-point truncation drift. Doing this 100 times in a row will silently corrupt the base model's intelligence until it outputs total garbage.

## The Solution: The Pristine Buffer
When `WeightFoldingEngine` boots up, it makes a bit-exact, read-only clone of the base model's weights and stores them in VRAM. This is the **Pristine State Buffer** (`self.pristine`).

Whenever the engine needs to swap to a new domain, it completely skips the subtraction math. Instead, it blasts the exact bytes from the Pristine Buffer directly back into the live model using PyTorch's `copy_()` command. 

This guarantees $L_{\infty}$ zero-drift (0.00% corruption) no matter how many millions of times the server swaps adapters!
