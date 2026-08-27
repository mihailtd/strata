# Rule: 24GB VRAM Bounding & Zero-Spill Long-Horizon Execution

## Core Principle
On consumer 24 GB GPUs (AMD Radeon RX 7900 XTX / NVIDIA RTX 4090), VRAM allocation MUST be strictly bounded to prevent PCIe host memory spilling, which causes catastrophic 10x throughput degradation (collapsing from ~110 tok/s to ~10 tok/s).

## Invariants & Guardrails
1. **Strict 21.8 GB VRAM Ceiling**:
   - Model Weights (4-Bit 35B MoE): ~19.8 GB
   - Desktop / Display OS Overhead: ~1.2 GB
   - Maximum KV Cache Budget: **$\le 1.0\text{ GB}$**
2. **Mandatory 4-Bit / 8-Bit Quantized KV Cache for Long Context (30k+)**:
   - Never allow unquantized FP16 KV cache to grow unbounded beyond 12k tokens.
   - For 30k+ active context, configure Q4/Q8 KV cache (`OLLAMA_KV_CACHE_TYPE=q4_0` / Flash-Attention) to compress 32k tokens into $\approx 0.8\text{ GB}$.
3. **Harness Semantic State Compaction**:
   - Active Turn ($T$): Keep raw tool output in full fidelity for precise code reasoning.
   - Historical Turns ($T-k$): Automatically fold verbose outputs (200-line greps, test logs, file reads) into compact structural digests, reducing token growth by 75–85%.
4. **Context Shift Rolling Buffer**:
   - If total context approaches hardware limits, seamlessly shift out older intermediate conversation turns while locking the initial System Instructions and recent turns in VRAM.
