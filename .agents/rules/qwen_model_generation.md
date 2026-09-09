# Rule: Strict Qwen 3.x Model Generation Invariant

## Core Directive
All model loading, training, fine-tuning, benchmarking, evaluation, and kernel development in this repository MUST strictly target **Qwen 3.x generation models only** (e.g., `Qwen3.5-4B`, `Qwen3.5-9B`, `qwen3.8:27b`).

## Invariants & Guardrails
1. **Never Fall Back to Qwen 2.5 or Older Generations**:
   - Do NOT substitute `Qwen2.5-*`, `Qwen2-*`, or older dense transformer checkpoints under any circumstance.
   - Any comparison against older generation models defeats architectural parity and is strictly prohibited.
2. **Explicit User Notification Over Silent Substitution**:
   - If a specific Qwen 3.x checkpoint weight format (e.g. unquantized safetensors vs. GGUF blob) is not directly available on a remote hub, **inform the user immediately**. Never silently switch to an older model family.
3. **Architectural Parity**:
   - Respect Qwen 3.x specific architectural components across all runtime kernels:
     - Gated DeltaNet Linear Attention
     - Recurrent State Tensor $S_t$ ($O(1)$ multi-turn handoff)
     - MTP (Multi-Token Prediction) draft heads
     - W4A16 Triton WMMA register dequantization

4. **Strict Tokenizer Governance (Zero Vocabulary Drift)**:
   - Qwen 3.x models (e.g. Qwen 3.5 / 3.8) have a distinct vocabulary and token IDs from Qwen 2.5 (e.g. token ID 750 vs 727 for keyword `def`).
   - The runtime server (`server.py`) and all evaluation harnesses MUST dynamically resolve the local Qwen 3.x snapshot tokenizer (`~/.cache/huggingface/hub/models--Qwen--Qwen3.5-*/snapshots/*`).
   - NEVER import or fall back to `Qwen/Qwen2.5-Coder-7B-Instruct` or older tokenizers in 27B runtime paths.
