# MACD speculation circuit-breaker: synthetic control-logic benchmark

A discrete-event simulation over hand-authored token-acceptance workloads (`tau` sequences representing high-alignment boilerplate, low-alignment reasoning, and a mixed transition), driving the **real** `MACDSpeculationCircuitBreaker` (`apps/runtime/macd_speculation_circuit_breaker.py`) through its actual `should_draft()`/`update_acceptance()` decision logic.

This is a stage-1 experiment, not a production throughput claim: latency constants (`eager_step_latency_ms`, `draft_step_latency_ms`, etc.) are fixed assumptions, not measured on live hardware in this run, and no real model is in the loop. What's real is the circuit-breaker's control-flow decisions under those assumed dynamics — whether it correctly trips to raw W=1 decode when acceptance drops and re-engages when it recovers.

Moved here from `benchmarks/` (misclassified — it was sitting next to real end-to-end benchmarks despite being a synthetic control-logic check).

## Run

```bash
uv run python experiments/runtime/speculative/macd_circuit_breaker/benchmark_macd_circuit_breaker.py
```
