# Execution gate — does folding an expert make the model measurably better at real work?

Real-sandbox task execution scoring (not throughput measurement) for the in-place weight-folding engine's domain experts. `verify_sandbox_completions.py` is the shared execution helper (runs in a clean process without PyTorch bindings, executes SQL against `py-pglite` and lints Python via `ruff`) used by `applied_execution_gate.py` (4B), `benchmark_9b_applied_execution_gate.py` (9B), and `benchmark_multi_expert_stacking_swe.py` (multi-expert stacking on real SWE-style tasks).

See [`evals/README.md`](../README.md) and the repo's [`docs/METHODOLOGY.md`](../../docs/METHODOLOGY.md).
