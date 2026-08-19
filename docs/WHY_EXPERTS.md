# Why we train experts — capability AND disposition

> Read this before designing any expert benchmark, and before "cleaning" any
> corpus of the constructs its domain is made of.

## The two things an expert buys us

**1. Capability** — can it produce correct DuckDB / Postgres / `uv` output when
explicitly asked? This is what every benchmark in this repo currently measures.

**2. Disposition** — when handed a problem that *names no tool*, does it reach for
the right one? This is **roughly half the value and currently none of the
measurement.**

Disposition is the thing a base model is worst at and the thing a domain expert is
uniquely good at. It is also the thing that shows up in real use, because in real
use nobody prefixes their question with "write me a DuckDB query."

## The example that makes it concrete

> *"I have 40GB of Parquet files partitioned by date. Give me revenue per
> category for last quarter."*

- **Base Qwen 4B**: writes a Python loop, or reaches for pandas, reads files one
  at a time, aggregates in memory. It may well be *correct*.
- **DuckDB expert**: `read_parquet` with `hive_partitioning`, projection pushdown,
  `GROUP BY ALL`. Correct, and orders of magnitude faster on real data.

**An execution-gated benchmark scores these EQUAL.** Both produce the right
numbers. The entire difference — the difference that matters in production — is
invisible to every instrument we currently run.

That is the gap. Base going off on a tangent and hand-rolling Python when the
perfect tool was sitting right there is a **real, large loss**, and an expert that
prevents it is a **real, large win**, regardless of what pass@1 says.

## What this means for contamination policy

This is where we got it wrong, so it is written down explicitly.

Training the DuckDB expert on `read_parquet`, `COLUMNS()`, `GROUP BY ALL`, and
`FROM`-first syntax is **not leakage. It is the entire point.** You cannot bias a
model toward a tool by hiding the tool from it. Construct-in-training is a
*prerequisite* for disposition, not a contamination of it.

The measured facts that forced this conclusion:

| | |
| :--- | ---: |
| duckdb corpus records | 1622 |
| families emitting a "reserved" construct | 11 of 14 |
| records dropped if all are reserved | **1465 (90.3%)** |
| records left to train on | **157** |

Reserving DuckDB's constructs does not produce a clean expert. It produces no
expert. Compare astral, where `TaskGroup` and `singledispatch` are peripheral —
remove them and a corpus about modern Python survives intact.

### The decision rule

> **Reserve CONSTRUCTS when they are peripheral to the domain.
> Reserve INSTANCES when they ARE the domain.**

- *peripheral* → astral: hold out `functools.partial`, `TaskGroup`. The expert
  never saw them; generalizing to them is a real test.
- *central* → duckdb: hold out **specific problems, schemas, and table shapes**.
  The expert learns `read_parquet`; the benchmark tests whether it applies it to
  data it has never seen.

Applying the construct rule to a central construct gives you one of two failures,
and we hit the first before noticing:

1. a **silent no-op** — the family names don't match, 0 records drop, and the
   verification prints `STILL LEAKING` in the hundreds
2. a **destroyed corpus** — the names do match, and 90% of the training data
   disappears

## The benchmark we are missing

A disposition benchmark is the natural fit, and it is **inherently
contamination-resistant** — which is the elegant part. You are not testing whether
the model knows `read_parquet`. You are testing whether it *volunteers* it when
nothing in the prompt suggested it.

Design:

- **The prompt must not name an engine, a library, or a function.** Describe a
  data problem, nothing more.
- **Score = which tool did it reach for**, not whether the output ran.
- Report base vs expert as a rate over N neutral problems.
- Because the construct is deliberately in training, contamination of the
  construct is irrelevant. What must be held out is the *problem*.

Suggested scoring, in descending value:

| outcome | meaning |
| :--- | :--- |
| reaches for the right engine, idiomatically | the win we are training for |
| reaches for the right engine, clumsily | partial — capability gap, not disposition gap |
| solves it correctly with the wrong tool | **the failure we care most about** — invisible to execution gating |
| fails outright | capability gap |

Note the third row. It is scored as a *success* by every benchmark we run today.

## Standing reminder

When a benchmark result and this document disagree, check what the benchmark is
measuring before concluding the expert is not working. "The expert didn't beat
base on pass@1" and "the expert is not worth having" are different claims, and the
instruments in this repo can currently only speak to the first.

---

## ⚠️ Update: the benchmark this document proposed is BLIND. See DECISIONS.md §44.

The disposition benchmark described above was built, run, and **cannot support a
conclusion in either direction.** Two findings retire it:

**It scores broken answers as perfect.** An astral-expert answer that invented an
invalid `[tool.uv] sources = [...]` schema, recommended git-installing Ruff, and
never mentioned `uv.lock` on a reproducible-builds question scores **NATIVE = 1.0** —
because it contains "uv sync" and lacks "pip". Keyword ratios cannot see correctness.

**The corpus never taught what the benchmark tests.** Of 1433 astral training
answers, **9.8%** contain a runnable uv/ruff command; 50.7% are Python code and
29.4% are pure prose. The adapter writes prose because that is what it was trained
on. The benchmark asked a question its corpus never covered.

So the principle in this document stands — **disposition is half the value and
capability benchmarks cannot see it** — but the instrument proposed for measuring it
does not work. Grade by *parsing what comes out* (does the TOML load? is
`tool.uv.sources` a table? does the backend match `requires`? does `uv.lock` appear?)
rather than by counting tool names.

And fix the corpus first: an adapter cannot reach for `uv add` if 90% of its
training answers never do.
