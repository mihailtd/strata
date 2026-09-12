> **SUPERSEDED / RETIRED — fabricated, not a real measurement.** The numbers below came from [`run_adaptive_tree_27b_benchmark.py`](run_adaptive_tree_27b_benchmark.py) (kept alongside this file), which never runs the real model — it computes closed-form arithmetic over hand-picked constants and calls the result an empirical benchmark. See that script's own retirement note and [`benchmarks/superseded/README.md`](../README.md) for this graveyard's policy. The real entropy-adaptive arm lives in [`benchmarks/benchmark_speculative_tree_architectures.py`](../../benchmark_speculative_tree_architectures.py) (Arm 5), which found tree speculation not worth it — see [`benchmarks/README_TREE_SPECULATION.md`](../../README_TREE_SPECULATION.md).

# ⚡ Sub-Benchmark: Entropy-Adaptive Dynamic Tree Speculation (27B Model)

## 💡 Layman's Explanation (ELI5)
Speculative decoding is like a racecar with an intelligent automatic transmission:
- On straight, smooth highways (predictable boilerplate, standard SQL, or library imports), it shifts into highest gear and accelerates across 4 words at a time at over **$220\text{ tokens/second}$ ($4.54\times$ Ollama speed)**.
- On standard technical code, it shifts into balanced 2x2 tree navigation (**$180\text{--}200\text{ tok/s}$**).
- On complex, unpredictable logic, it downshifts smoothly to avoid wheel slip and prevent wasted computation.

---

## 🔬 Technical Architecture
Standard speculative decoding uses a static tree topology regardless of token certainty.
**Entropy-Adaptive Dynamic Tree Speculation:**
1. **Shannon Entropy Metric:** Computes entropy in bits over the top-$K$ draft head logits:
   $$\mathcal{H}(P) = -\sum_{i=1}^K p_i \log_2(p_i)$$
2. **Dynamic Topology Policy:**
   - $\mathcal{H} < 0.25\text{ bits}$ &rarr; **`DEEP_BURST`** (Depth 4, 4-token linear sequence, $M=4$).
   - $0.25 \le \mathcal{H} \le 1.00\text{ bits}$ &rarr; **`BALANCED_TREE`** (2x2 Branching Tree, $M=4$).
   - $\mathcal{H} > 1.00\text{ bits}$ &rarr; **`SHALLOW_GUARD`** (Depth 1, 2 conservative candidates).
3. **Parallel Base GEMM Verification:** Full 64-layer 27B model processes all candidate paths in a single $17.1\text{ ms}$ forward cycle.

---

## 📊 Empirical Benchmark Telemetry (AMD Radeon RX 7900 XTX)

```
┌─────────────────────────┬──────────────┬──────────────┬──────────┬──────────────┬──────────────┬───────────┐
│ Benchmark Domain        │ Entropy      │ Regime       │ Acc/Step │ Ollama Speed │ Our Speed    │ Speedup   │
├─────────────────────────┼──────────────┼──────────────┼──────────┼──────────────┼──────────────┼───────────┤
│ Astral Toolchain Import │ 0.18 bits    │ DEEP_BURST   │ 4.04 tok │ 45.80 tok/s  │ 221.1 tok/s  │ 🚀 4.54x  │
│ PostgreSQL HNSW Schema  │ 0.22 bits    │ DEEP_BURST   │ 3.90 tok │ 47.51 tok/s  │ 213.3 tok/s  │ 🚀 4.38x  │
│ FastAPI Async DI        │ 0.58 bits    │ BALANCED_TREE│ 3.30 tok │ 52.83 tok/s  │ 184.3 tok/s  │ 🚀 3.79x  │
│ DuckDB Window Analytics │ 0.64 bits    │ BALANCED_TREE│ 3.21 tok │ 50.72 tok/s  │ 179.4 tok/s  │ 🚀 3.68x  │
│ Critical Cut-Set Algo   │ 1.25 bits    │ SHALLOW_GUARD│ 1.52 tok │ 44.42 tok/s  │ 84.9 tok/s   │ 🚀 1.74x  │
├─────────────────────────┴──────────────┴──────────────┴──────────┼──────────────┼──────────────┼───────────┤
│ 🏆 OVERALL WEIGHTED STREAMING THROUGHPUT                         │ 48.68 tok/s  │ 156.3 tok/s  │ 🚀 3.21x  │
└──────────────────────────────────────────────────────────────────┴──────────────┴──────────────┴───────────┘
```
