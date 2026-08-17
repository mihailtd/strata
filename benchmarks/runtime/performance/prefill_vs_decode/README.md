# ⭐ Prefill vs. Decode Latency Share (Amdahl's Law Audit)

> ## ⚠️ THE "5.9% @8k" NUMBER IS SPECIFIC TO 1024-TOKEN GENERATIONS
>
> Everything below is measured with `max_new_tokens=1024`, which amortises prefill
> over 1024 decode steps. **An agent turn emits ~150 tokens, not 1024**, and the
> share moves by 5x. Measured by
> [`benchmark_agent_turn_split.py`](benchmark_agent_turn_split.py):
>
> | prompt | out=50 | out=150 | out=300 | out=1024 |
> | ---: | ---: | ---: | ---: | ---: |
> | 2000 | 25.2% | 10.2% | 5.2% | 1.5% |
> | **3000** | 30.8% | **13.0%** | 7.0% | 2.2% |
> | 8000 | **55.8%** | **30.3%** | 17.5% | 5.8% |
>
> The two scripts reconcile at out=1024 (5.8% vs 5.9% @8k), so this is a workload
> shape difference, not a contradiction.
>
> **The claim below that "Decode Acceleration governs >94% of user latency" holds
> only for long generations.** At the stated agent turn (3,000 in / 150 out) decode
> is **87%** — still dominant, so decode work remains justified, but a 2x decode
> win returns **1.77x** end-to-end, not 2x. At 8,000 in / 50 out prefill is
> **55.8%** and decode optimisation is capped at **1.79x no matter what**.
>
> Read the agent-shape table before sizing any decode optimisation.

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
