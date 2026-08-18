# v4 corpora + reserved-construct gate — results (DRAFT FOR REVIEW, not committed)

## 1. What was broken

The held-out gate picked its test constructs by scanning the **v2** corpora for
zero occurrences. The v3 corpora then trained **32 of the gate's 45 steps (71%)**,
because `DISTINCT ON` / `FILTER` / `LATERAL` / `percentile_cont` /
`singledispatch` / `TaskGroup` / `Protocol` / `__slots__` are the obvious contents
of an "Advanced PostgreSQL Features" and a "Modern Python" chapter.

This was a benchmark-design failure, not a corpus failure: the reserved set was
never written down anywhere the corpus author could see it.

## 2. The fix — reserve by REMOVAL, then verify

`scripts/reserve_eval_constructs.py` holds an explicit reserved-family list,
strips those families from training, and **verifies zero residual occurrences**.
Verification is the load-bearing part: removal alone proves nothing, because an
unreserved generator can emit the same construct.

| corpus | records | dropped | result |
| :--- | ---: | ---: | :--- |
| postgresql v3 → **v4** | 1599 → **1385** | 214 (13.4%) | 11/11 constructs CLEAN |
| astral v3 → **v4** | 1605 → **1433** | 172 (10.7%) | 6/6 constructs CLEAN |

⚠️ Verification caught a real leak: **`Protocol` survived at 103 hits** after
removing `py_protocol_slots`, because `modern_typing` also emits it. That family is
valuable general typing content, so `Protocol` was dropped as a *gate* construct
and replaced with `cached_property` (verified at zero). Trusting removal alone
would have produced a gate that looked clean and silently wasn't.

**Train on `training_data_v4.jsonl`, never v3.**

## 3. Controlled results — same gate, base identical at 0.3889

| adapters | base | oracle | edge | 95% CI | significant |
| :--- | ---: | ---: | ---: | :--- | :--- |
| v3, alpha=128 (control) | 0.3889 | 0.2889 | **−0.1000** | [−0.2111, +0.0000] | no |
| **v4, alpha=128** | 0.3889 | 0.3556 | **−0.0333** | [−0.2000, +0.1333] | **no** |
| v4, alpha=96 | 0.3889 | 0.3556 | −0.0333 | [−0.1889, +0.1111] | no |

Base scored **0.3889 in all three runs** — the harness is deterministic, so the
movement is attributable to the adapters, not to sampling or to the gate change.

**The corpus work cut the regression by 67% (−0.1000 → −0.0333), and narrowing is
no longer statistically detectable.**

⚠️ The earlier −0.2333 → −0.1222 → −0.0778 figures were measured on the OLD gate
(base 0.4667) and are **not** comparable to these. Swapping `Protocol` for
`cached_property` cost base 3 steps, which is why base fell to 0.3889. Only the
two rows above share an instrument.

### alpha no longer discriminates

alpha=96 and alpha=128 give **identical** results on v4, where they differed by
0.0445 on v3. That is the admissible window WIDENING, exactly as predicted:
retention mixing suppresses `(BA)h` on general constructs, raising alpha_max, so
both points now sit inside the window. **No alpha calibration is needed for v4.**

Recorded automatically by the trainer (§32 Factory stage):

| adapter | records | prompt masked | \|dW\|/\|W\| | merge_err |
| :--- | ---: | ---: | ---: | ---: |
| postgres v4 | 1385 | 35.5% | 0.0758 | 2.20% |
| astral v4 | 1433 | **55.2%** | 0.0754 | 2.21% |

astral's corpus was over half prompt text — the completion-only fix matters most
there. Note `|dW|/|W|` barely moved despite nearly doubling the corpus: the
perturbation magnitude is set by rank and alpha, not data volume, so the precision
FLOOR is unchanged while the narrowing CEILING moved.

### Per-construct — experts now win four, lose two

| construct | base | v4 experts |
| :--- | ---: | ---: |
| `unnest` | 0.000 | **0.500** |
| `LATERAL` | 0.000 | **0.333** |
| `asyncio.TaskGroup` | 0.000 | **0.333** |
| `array_agg` | 0.000 | **0.250** |
| `__slots__`, `singledispatch` | 1.000 | 1.000 (tied) |
| `DISTINCT ON` | **0.750** | 0.250 |
| `functools.partial` | **1.000** | 0.000 |

Routing was **1.000 over 44 decisions** in every run. Routing has never been the
problem in any gate.

## 4. What this does and does not establish

**Does:** the −0.2333 regression measured earlier was largely self-inflicted —
prompt-inclusive loss, an over-scaled alpha, and a contaminated corpus. Clean data
plus completion-only loss plus retention mixing removes almost all of it, and the
remaining −0.0333 is indistinguishable from zero at n=15.

**Does not:** show that experts HELP. On constructs they have never seen, parity is
the expected outcome, not a win. The case for having experts at all rests on
**in-domain** performance, which this gate deliberately does not measure — every
construct in it is reserved from training.

**The missing measurement** is therefore an in-domain benchmark using the reserved
families themselves as the test set (they are real, verified-correct SQL/Python
that the experts never saw, but which sit squarely in their domains). That would
show whether the experts earn their place, rather than merely not damaging the base.
