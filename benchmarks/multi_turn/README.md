# Multi-turn chained handoff benchmarks

Real end-to-end multi-turn benchmarks that chain several tasks through one session, plus their unit tests.

| File | What it measures |
| :--- | :--- |
| `multi_turn_execution_benchmark.py` | Chained multi-turn handoff with held-out constructs (n=15 turns): sequential task handoff across DB schema, service layer, and test client, testing OOD generalization (SQL window functions, recursive CTEs) not seen during adapter training. |
| `multiturn_dashboard_benchmark.py` | Simulates sequential user clicks on the 4 dashboard domain prompts in a single context window, for speculative decoding + LoRA weight-folding under realistic dashboard usage patterns. |
| `test_bootstrap_ci.py` | Unit tests for `multi_turn_execution_benchmark.py`'s paired bootstrap confidence-interval calculation. |
| `test_multi_turn_parser.py` | Unit tests for its markdown code-fence extraction and `Pipeline`/`TurnStep` data structures. |

## Run

```bash
uv run python benchmarks/multi_turn/multi_turn_execution_benchmark.py
uv run python benchmarks/multi_turn/multiturn_dashboard_benchmark.py
uv run pytest benchmarks/multi_turn -v
```
