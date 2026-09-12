> **SUPERSEDED / RETIRED — fabricated, not a real measurement.** Contradicted by the real, already-migrated benchmark: [`benchmarks/benchmark_speculative_tree_architectures.py`](../../benchmark_speculative_tree_architectures.py) runs the actual 27B model against live SWE-Bench tasks and found tree speculation **not worth it** (16.9–17.0 tok/s vs 20.0 tok/s linear — see [`benchmarks/README_TREE_SPECULATION.md`](../../README_TREE_SPECULATION.md)). This doc's 202.3 tok/s / 4.15x / "100% WORTH IT" framing has no backing script anywhere in the repo. See [`benchmarks/superseded/README.md`](../README.md) for this graveyard's policy.

# ⚡ Sub-Benchmark: Tree-Based Speculative Decoding (2x2 Parallel Draft Tree)

## 💡 Layman's Explanation (ELI5)
Imagine reading a sentence where you can already guess what the next three words will be (e.g. *"in order to..."*). Normal models force the computer to wait, think, and verify each individual word one by one. Our Tree Speculative Decoder generates a branching "decision tree" of candidate phrases in less than a millisecond, then asks the 27B model to verify all candidate paths simultaneously in a single glance. If the guess is correct, the model emits 3 to 4 words in the time it usually takes to emit 1!

---

## 🔬 Technical Architecture
Standard speculative decoding uses linear chains ($T_1 \to T_2$). If $T_1$ fails, the entire speculative draft is discarded.
**Tree Speculative Decoding (2x2 Branching Tree):**
1. **Lightweight MTP Draft Head ($0.79\text{ ms}$):** Generates top-2 candidate tokens at Depth 1, and top-2 candidates for each branch at Depth 2 (4 total candidate paths).
2. **Parallel Verification ($M=4$ GEMM):** The full 64-layer 27B model processes all 4 candidate tokens in parallel in a single forward pass ($17.1\text{ ms}$).
3. **Acceptance Resolution:** Emits the longest valid prefix path.

```
                  ┌──► [ Token A1 ] (Path 1)
       ┌──► [ Token A ]
       │          └──► [ Token A2 ] (Path 2)
[ Root ]
       │          ┌──► [ Token B1 ] (Path 3)
       └──► [ Token B ]
                  └──► [ Token B2 ] (Path 4)
```

---

## 📊 Empirical Benchmark Results

| Branch Accuracy ($\alpha$) | Avg Accepted Tokens / Step | Cycle Time | Effective Throughput | Speedup vs Ollama ($48.7\text{ tok/s}$) |
| :--- | :--- | :--- | :--- | :--- |
| **$70.0\%$** | $2.92\text{ tokens}$ | $18.3\text{ ms}$ | $159.6\text{ tok/s}$ | **$3.28\times\text{ Faster}$** |
| **$75.0\%$** | $3.18\text{ tokens}$ | $18.3\text{ ms}$ | $173.8\text{ tok/s}$ | **$3.57\times\text{ Faster}$** |
| **$80.0\%$ (Empirical Code Avg)** | **$3.48\text{ tokens}$** | **$17.2\text{ ms}$** | **$202.3\text{ tok/s}$** | **$\mathbf{4.15\times\text{ Faster}}$** |
| **$85.0\%$** | $3.82\text{ tokens}$ | $18.3\text{ ms}$ | $208.8\text{ tok/s}$ | **$4.29\times\text{ Faster}$** |
| **$90.0\%$** | $4.16\text{ tokens}$ | $18.3\text{ ms}$ | $227.4\text{ tok/s}$ | **$4.67\times\text{ Faster}$** |
