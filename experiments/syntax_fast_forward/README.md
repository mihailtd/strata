# Syntax fast-forwarding: stage-1 synthetic benchmark

Stage 1 of the syntax fast-forwarding lifecycle: compiles the real `SyntaxTrieDrafter` (`apps/runtime/syntax_drafter.py`) against the real 27B tokenizer and measures trie compilation latency, microsecond lookup latency distributions (P50/P90/P99), and candidate-continuation precision across scripted code-generation trigger scenarios — no live model decode loop yet.

The real, live-model end-to-end result this graduated into lives in [`benchmarks/syntax_fast_forward/`](../../benchmarks/syntax_fast_forward/README.md) (which found syntax fast-forwarding is not recommended for unconstrained decode — see that README).

## Run

```bash
uv run python experiments/syntax_fast_forward/benchmark_syntax_fast_forward.py
```

Saves results to `results/benchmarks/syntax_fast_forward_benchmark.json`.
