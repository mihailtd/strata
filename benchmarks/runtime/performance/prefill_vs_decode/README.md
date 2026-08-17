# ⭐ Prefill vs. Decode Latency Share (Amdahl's Law Audit)

> **Tier Classification**: **⭐ Industry Standard** (Latency Profiling) / **🔥 Applied Practice** (Amdahl's Law Optimization Prioritization)  
> **Concept Origin**: **Amdahl's Law execution profiling applied to hybrid linear-attention serving architectures.**

---

### Classification Breakdown: What is Standard vs. What is Innovative
* **⭐ Industry Standard Baseline**: Isolating prompt prefill wall-time from autoregressive token generation across varying context lengths ($T \in [2048, 4096, 8192]$).
* **🔥 Our Applied Practice Discipline**: Debunking the **"54% Prefill Myth"**. Early un-warmed runs falsely reported prefill taking 54% of wall time due to ~33 seconds of un-warmed Triton JIT compilation inside the timed loop. By enforcing per-shape warmups and profiling isolated kernels, we proved that prefill accounts for only **5.9% of wall time at 8k context**, scientifically closing complex Radix KV-caching and proving that **Decode Acceleration (In-Place Weight Folding, Speculative Decoding) governs >94% of user latency**.

---

## 1. Empirical Latency Breakdown ($T_{\text{context}} \in [2048, 4096, 8192]$, 1024 Generated Tokens)

Audited on `Qwen/Qwen3.5-4B` in `bfloat16` on AMD RX 7900 XTX:

| Prompt Context Length ($T_{\text{prompt}}$) | Prefill Wall-Time (s) | Decode Wall-Time (1024 Tokens) | Prefill Share (%) | Decode Share (%) | Architectural Implication |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **2,048 tokens** | 0.49 s | 30.12 s | **1.6%** | **98.4%** | Decode dominates completely |
| **4,096 tokens** | 0.98 s | 30.34 s | **3.1%** | **96.9%** | Decode dominates completely |
| **8,192 tokens** | 1.92 s | 30.61 s | **5.9%** | **94.1%** | Decode governs 94%+ of latency! |

---

## 2. Key Architectural Takeaways

1. **Decode Governs Total User Latency**: Even under massive 8,192-token prompt contexts, prompt prefill takes less than 2 seconds, while decoding 1,024 output tokens takes over 30 seconds.
2. **Why Decode Optimization is King**:
   * Optimizing prefill by 50% saves only **~0.9 seconds**.
   * Optimizing decode by 50% (via our **2.20x Speculative Engine** and **+82.1% In-Place Weight Folding**) saves **over 15 seconds per request**!
3. **Closing Hybrid Radix Caching**: Complex prefix-caching trees (e.g., Radix KV caches) add memory fragmentation and cache eviction overhead for at most a ~5% theoretical gain on this architecture, confirming our engine's strategic focus on decode acceleration.

---

## 3. Scripts
- **`benchmark_prefill_share.py`**: Measures median prefill vs. decode latency across 2k, 4k, and 8k context lengths and saves metrics to `results/prefill_share_benchmark.json`.
  ```bash
  uv run --env-file .env python benchmarks/runtime/performance/prefill_vs_decode/benchmark_prefill_share.py --max-new-tokens 1024
  ```
