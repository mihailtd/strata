# Rule: The Chicken-and-Egg Runtime Paradox & Custom Kernel Mandate

## Core Principle
Standard inference frameworks present an impossible tradeoff for multi-expert agent workflows:
1. **Ollama / llama.cpp**: Blazing raw speed (110+ tok/s on 35B MoE via C++/HIP assembly), but an **immutable static weight graph**. Swapping a LoRA adapter requires unloading the model and evicting the active KV cache (costing a 2,500–4,500 ms re-prefill freeze per turn).
2. **Naive PyTorch / HuggingFace PEFT**: Native dynamic adapter switching, but **unoptimized ROCm kernels** that run at only 15–25 tok/s (unusable for real-time agent loops).

To resolve this paradox, the system mandates a custom low-level Triton/HIP runtime that matches Ollama's hardware saturation while delivering **sub-18ms in-place LoRA swapping**.

## Architectural Invariants & Guardrails

1. **Sub-18ms Zero-Copy Weight Folding ($W_{\text{live}} \leftarrow W_0 + s \cdot U V$)**:
   - Mutate pre-allocated low-rank working buffers in VRAM in-place without re-allocating model memory or reloading weights from host RAM.
   - Total memory churn per adapter swap MUST be **0 bytes**.

2. **100% KV Cache Retention Across Adapter Swaps**:
   - Swapping a specialist LoRA adapter MUST NEVER invalidate or flush the active KV Cache.
   - The model must continue streaming the subsequent turn immediately with zero prompt re-prefill penalty.

3. **Fused In-Register LoRA Dot Products**:
   - Never compute separate LoRA forward passes ($y = W_0 x + B(Ax)$) that double VRAM bus traffic.
   - Fuse the low-rank rank-8 dot product directly into the 4-bit dequantization loop inside GPU Wave32 vector registers.

4. **128-Bit Vector Coalescing & Fused SwiGLU**:
   - Scribe all global memory loads into 128-bit vector bundles (`int32x4`) to achieve $\ge 70\%$ physical GDDR6 bus bandwidth saturation on AMD RDNA3 hardware.
   - Fuse Gate + Up GEMV with in-register SiLU activation to eliminate intermediate VRAM traffic.

5. **Ultrafast Semantic Routing ($<30\mu\text{s}$)**:
   - Route incoming agent steps to the optimal domain specialist (e.g. `postgresql`, `duckdb`, `fastapi`) using an embedded Riemannian covariance classifier executing in $<30\mu\text{s}$ before the first token is emitted.
