# Evals — is it actually correct, not just fast?

Task- and rubric-scored evaluation of a runtime or a trained adapter, as distinct from [`benchmarks/`](../benchmarks/)'s raw performance measurement. See the repo's [`docs/METHODOLOGY.md`](../docs/METHODOLOGY.md) for how this fits alongside `experiments/`/`benchmarks/`, and for the rule of thumb that decides which tree a new script belongs in: *a benchmark's number is a physical measurement; an eval's number is a judgment against a task, a rubric, or a ground truth.*

## What's here

| Directory | Methodology | Scoring |
| :--- | :--- | :--- |
| [`judge/`](judge/) | Agent-as-Judge: multi-turn conversations and hybrid domain challenges, full untruncated transcripts | Human/agent reads a 5-dimension rubric against the transcript — not auto-scored |
| [`domain_rubric/`](domain_rubric/) | Single-shot domain prompts against a fixed keyword/pattern rubric per domain | Automated substring/pattern match against expected indicators |
| [`swe_bench/`](swe_bench/) | Fixed coding tasks across 6 real-world domains, real pytest sandboxes | Pass@1 — real test execution, ground truth |
| [`execution_gate/`](execution_gate/) | Applied toolchain/execution tasks (does folding an expert make the model measurably better at real executable work) | Real sandbox execution (`ruff`, `py-pglite`) |
| [`factory/`](factory/) | Held-out/task-quality evaluation of trained adapters (adherence, attribution, disposition, chained-holdout methodology) | Task-based, mostly against held-out constructs |
| [`dsh_agent/`](dsh_agent/) | Autonomous agentic tool-use: can a DSH-driven agent diagnose and fix a real dependency conflict | Ground truth (`uv lock` exits 0 afterward) |

## Why this many methodologies

Each answers a different question about correctness, not the same question five ways:

- **judge/** — is a multi-turn conversation *good*, in a way no fixed rubric can pin down (does it stay coherent, pick the right expert team, avoid degenerate repetition)?
- **domain_rubric/** — did a single-shot answer hit the right idioms for its domain, cheaply and repeatably?
- **swe_bench/** and **execution_gate/** — does the generated code actually *run* — real pytest, real linter, real database, not a text-similarity proxy for correctness?
- **factory/** — do the *trained adapters themselves* generalize past their training distribution, not just the base model at inference time?
- **dsh_agent/** — can an autonomous agent *use tools* to fix something, not just describe the fix?

## Relationship to `tests/opencode_evals/` (removed)

This repo's first pass at agentic-capability evaluation used the third-party `opencode` CLI, under `tests/opencode_evals/`. That was retired as too heavyweight for local-model use; `dsh_agent/` is its replacement, built on DSH (`apps/harness/`) instead — see `dsh_agent/README.md` for the full history and the mechanical differences.
