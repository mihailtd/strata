# scripts/audit/ — verify before you trust a number

## `check_canon.py` — run this before quoting any result

```bash
uv run python scripts/audit/check_canon.py     # exit 1 on violation
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
| `check_canon.py` | **enforces canon.py** — decode budget and adapter version |
| `audit_adapters.py` | adapter inventory and geometry |
| `audit_eval_rubrics.py` | rubric sanity — catches regexes that can never match |
| `check_gpu.py` | ROCm / GPU visibility preflight |
| `analyze_loss_curves.py` | training-curve inspection |

## Rubrics deserve the same suspicion as caps

`audit_eval_rubrics.py` exists because a grader that lowercases its input and then
matches case-sensitively against `FROM`/`EXCLUDE`/`WHERE` silently scores 30% of
its patterns at zero. That is indistinguishable from a real result.
