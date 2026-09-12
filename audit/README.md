# audit/ — verify before you trust a number

**What this is**: the only 4 tools in the repo that are genuinely repo-wide — none of them is owned by, or specific to, any single `apps/*` project. Each one scans *across* `benchmarks/`, `experiments/`, `evals/`, `results/`, or `data/` — resources shared by every app — rather than checking one app's own correctness. That's the actual test for "does this belong here": if a tool only ever touches one `apps/<project>`, it belongs in that project, not here (this replaced a root-level `scripts/` folder that used to mix both kinds together, which is exactly the kind of thing that gets an agent to run the wrong tool against the wrong target).

**When to run these**: before you cite a benchmark/eval result (`check_canon.py`), before/after a training run (`audit_adapters.py`), when writing or reviewing an eval's grading rubric (`audit_eval_rubrics.py`), and before starting any GPU-heavy work (`check_gpu.py`). None of these run automatically — nothing in this repo currently gates a commit or benchmark run on them; run them yourself at the point above.

**When NOT to reach for this folder**: if you're checking or launching something specific to one runtime or to training, go to that app instead — e.g. `apps/runtime/run_server.py` / `apps/runtime/integration_check.py` / `apps/runtime/ab.py` (that one runtime's server, tests, and comparison tool), or `apps/factory/chat.py` / `apps/factory/analyze_loss_curves.py` (training-adjacent tools). See `apps/RUNTIME.md` and `apps/factory/README.md`.

## `check_canon.py` — run this before quoting any result

```bash
uv run python audit/check_canon.py     # exit 1 on violation
```

Fails on any quality-grading benchmark that hard-codes a decode budget below
**2048** or references a legacy adapter version. Latency probes are exempt by
explicit path — an 8-token decode is the correct measurement there.

### Why it exists

Every one of these was a real wrong conclusion caused by a constant in the wrong place:

| constant | what it produced |
| :--- | :--- |
| `max_new_tokens=64` | a +20.6% "speculative win" that was really **22% slower** |
| `max_new_tokens=448` | a p=0.0018 Python win; at 2048 the same base went 0.292 → **0.917** |
| `max_new_tokens=192` | the entire 3-way α-scaling matrix, while the benchmark it duplicated defaulted to 2048 |
| `max_new_tokens=768` | base "ran out of budget and failed to compile", read as a robustness win |
| adapter fallback chain | v4 silently bypassed for hours because a stale path sorted first |

The pattern never varies: a number small enough to look harmless, somewhere nobody
looks, quietly deciding the result.

| script | what it does |
| :--- | :--- |
| `check_canon.py` | **enforces canon.py** — decode budget and adapter version, across `benchmarks/`, `experiments/`, and `evals/` |
| `audit_adapters.py` | inventory and drift-detection over every trained adapter in `results/adapters/`, shared by all runtimes |
| `audit_eval_rubrics.py` | rubric sanity for eval data under `data/*/evaluation_data*.jsonl` — catches regexes that can never match |
| `check_gpu.py` | ROCm / GPU visibility preflight — hardware-level, not owned by any one runtime |

## Rubrics deserve the same suspicion as caps

`audit_eval_rubrics.py` exists because a grader that lowercases its input and then
matches case-sensitively against `FROM`/`EXCLUDE`/`WHERE` silently scores 30% of
its patterns at zero. That is indistinguishable from a real result.
