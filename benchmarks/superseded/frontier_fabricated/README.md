# Fabricated "Frontier Inventions" scorecards

Five scripts (formerly `experiments/frontier/`) presenting four speculative feature ideas as a benchmarked scorecard. Flagged as Critical #5 in `docs/EXPERIMENT_REAUDIT_2026-09.md`. None of the four "inventions" load a model; none import torch. Every reported number (cosine similarity retention, exit probabilities, acceptance rates, "net throughput speedup") is a hand-authored Python literal — these scripts assert a scorecard, they don't test anything.

Per the audit's own judgment, **not worth revisiting in current form** — unlike the other critical findings, nothing shipped depends on any of these, so there was no production default to correct. This is a pure documentation/labeling fix (retire + cross-link), not a re-measurement.

| Script | "Invention" | Real analog, if any |
| :--- | :--- | :--- |
| `benchmark_tree_speculation.py` | Speculative tree decoding (Tree-MTP / Medusa) | **Contradicted by a real measurement**: `benchmarks/tree_speculation_e2e/` ran the real 27B model against live SWE-Bench tasks and found tree speculation NOT worth it (16.9–17.0 tok/s vs 20.0 tok/s linear) — the opposite of this script's claimed "156–202 tok/s, 100% WORTH IT." |
| `benchmark_jump_tokens.py` | Jump-tokens (AST/template macro injection) | **Partial real analog exists and shipped**: `apps/runtime/jump_streamer.py`'s `JumpTokenStreamFilter`, wired live into `apps/runtime/server.py` (not yet vendored into `apps/runtime-ipwf`). Never itself given a dedicated real-model throughput benchmark, so this fabricated script's predictions can't yet be checked against it directly. |
| `benchmark_early_exit.py` | Dynamic entropy early-exit (adaptive depth) | None — never built. |
| `benchmark_block_sparsity.py` | Dynamic subspace weight sparsification (block-sparse GEMV) | None — never built. |
| `run_all_frontier_benchmarks.py` | Orchestrator/aggregator for the above 4 | N/A |

Zero references anywhere in `docs/DECISIONS.md` — these numbers never made it into the repo's decision record, only into these scripts' own output JSONs (deleted).
