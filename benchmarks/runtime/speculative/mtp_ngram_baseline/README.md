# Native speculative decoding baseline (Neural MTP + Context N-Gram)

Benchmarks the base native speculative decoding implementation on the 27B engine before any of the more advanced gating/circuit-breaker logic in the other `../` sibling directories is layered on top.

Evaluates:
1. Mathematical equivalence to pure greedy decode (100% bit-for-bit target parity).
2. Latency and tok/s for baseline single-token decode, Neural MTP speculative decode, and Context N-Gram speculative decode.
3. Empirical candidate acceptance rate (tau).

## Run

```bash
uv run python benchmarks/runtime/speculative/mtp_ngram_baseline/benchmark_native_speculative_27b.py
```
