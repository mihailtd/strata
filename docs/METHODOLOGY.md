# Methodology: how an idea becomes a shipped feature here

This repo has one recurring workflow for turning an optimization idea or a new feature into something real, and five directories that each own one piece of it. This document is the map between them — read it before deciding where a new script belongs, or before assuming a folder's name tells you its status.

## The 3-stage lifecycle

```
1. EXPERIMENT              2. INTEGRATION              3. E2E TESTS & BENCHMARKS
   experiments/                apps/runtime*/               benchmarks/
                                apps/factory/                 evals/
```

**Stage 1 — Experiment.** A new idea starts as a synthetic microbenchmark (isolated kernel/operation timing, no real model in the loop) — does the underlying math or hardware trick work at all, in principle? If that passes, it graduates to a quick empirical test: a small, real-model run that proves the idea can survive contact with production conditions, without yet being wired into anything. Both halves of this stage live in **[`experiments/`](experiments/)**.

**Stage 2 — Integration.** The idea gets wired into one or more of the actual serving/training codebases — `apps/runtime`, `apps/runtime-triton`, `apps/runtime-ipwf`, `apps/factory`, etc. — optionally behind a feature flag so it can be toggled without a redeploy. This stage has no dedicated top-level folder: it *is* the application source. The signal that something has reached this stage is simple and checkable — **does a module under `apps/runtime*` or `apps/factory` actually implement it?**

**Stage 3 — Real end-to-end tests and benchmarks.** Once integrated, the feature gets measured for real: full pipeline, real model, real hardware, no synthetic shortcuts. Pure performance measurement (throughput, latency, memory, hardware utilization) lives in **[`benchmarks/`](benchmarks/)**. Task- or rubric-scored quality/correctness evaluation lives in **[`evals/`](evals/)** — see the next section for why that's a separate axis rather than a 4th stage.

### The classification rule

Given any script that measures something, ask: **does the concept it exercises already have a live module under `apps/runtime*` or `apps/factory`?**

- **Yes → `benchmarks/`** (or `evals/` if it's scoring a task/rubric, not raw performance) — stage 3, the feature is real.
- **No → `experiments/`** — stage 1, still pre-integration, however convincing the synthetic numbers look.

This is a check, not a guess from the filename or folder name. `benchmarks/runtime/statistical/{copula_routing,gee_trajectory}` sound exploratory by name, but `apps/runtime/copula_routing.py` and `apps/runtime/gee_trajectory.py` are real, live modules — so those benchmarks measure a shipped feature and belong in `benchmarks/`, not `experiments/`. Conversely, a script under `benchmarks/` with a `"""Synthetic Benchmark:` docstring prefix and no corresponding `apps/*` module is still stage 1, wherever it happens to be sitting.

## Where evals and tests fit

**Evals** (`evals/`) and **tests** (colocated with whatever they test — see below) are not a 4th and 5th stage bolted onto the end. They cut *across* the 3 stages:

- An eval can score a stage-1 experiment's viability (`experiments/frontier_e2e_validation/run_real_empirical_frontier_benchmark.py`-style "does this idea hold up against a real task" checks) just as easily as it can score a stage-3 integrated feature (`evals/swe_bench/`, `evals/domain_rubric/`).
- A unit test pins the correctness of a specific module regardless of which stage produced it — a geometry probe in `experiments/factory/geometry/` gets exactly the same kind of pinning test as a shipped module in `apps/runtime/tests/`.

The dividing line between a **benchmark** and an **eval** (the blurriest one in this repo) is: *what does the number mean?* A benchmark's number is a physical measurement — tok/s, milliseconds, GB/s, VRAM bytes. An eval's number is a judgment against a task, a rubric, or a ground truth — did it pass, did it match the expected keywords, did `uv lock` exit 0 afterward. A script that does both (measures speed *and* scores task success) belongs in `evals/` — correctness takes priority over performance when they're bundled, because a fast wrong answer is worthless.

## Tests: colocated with their subject, never a flat root pile

`tests/` no longer holds a monolithic flat suite. Every test lives next to what it tests:

- Tests exercising one specific `apps/<project>` module live in `apps/<project>/tests/`.
- Tests pinning a benchmark or experiment probe's correctness live right next to that probe (e.g. `experiments/factory/geometry/latent_variable_glasso/test_latent_variable_glasso.py`), not in a separate tree.
- Repo-wide tooling that isn't owned by one app (e.g. the audit scripts in `audit/`) gets its own `tests/` alongside that tooling.

If you're writing a test and asking "does this go in root `tests/`?" — the answer is no; find what the test actually exercises and put it there instead. See `tests/README.md`.

## Retirement policy

When an experiment fails, or a benchmark's approach gets replaced by a better one, it moves to **[`benchmarks/superseded/`](benchmarks/superseded/)** — one unified graveyard regardless of which of the three trees it came from — with a retirement rationale explaining *why* (not-worth-it, beaten-by-an-alternative, or found-to-be-wrong are all different things worth distinguishing). Never delete a retired idea outright unless it's truly dead code with zero historical value; a documented rejection saves someone from re-proposing and re-measuring the same thing years later.

## Rigor check: is this number real?

Before citing any result from `experiments/`, check whether a script's "measurement" actually calls a model or is a hand-authored constant standing in for one — this repo has had real, repeat instances of the latter presented as the former (see [`docs/EXPERIMENT_REAUDIT_2026-09.md`](docs/EXPERIMENT_REAUDIT_2026-09.md) for a full sweep, including which shipped runtime defaults currently rest on unverified or fabricated data). If you're about to build on top of an `experiments/` finding, check that document first — it may already tell you the number is unverified, or that a cleaner real-data version of the same finding already exists elsewhere in the repo.

## Quick reference: where does my new script go?

1. **Haven't touched `apps/` yet, just testing an idea?** → `experiments/`. Synthetic first if you can get away with it; a small real-model run once the synthetic version looks promising.
2. **It's wired into `apps/runtime*` or `apps/factory` now, and you want a speed/memory/throughput number?** → `benchmarks/`.
3. **It's wired in, and you want to know if it's actually *correct* — a task passes, a rubric scores well, a ground truth matches?** → `evals/`.
4. **It's a regression test pinning one module's behavior?** → colocated with that module, in whichever of the above three trees (or `apps/<project>/tests/`) that module lives in.
5. **It didn't work out, or got replaced?** → `benchmarks/superseded/`, with a one-paragraph rationale.
