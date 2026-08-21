# RETIRED — APSP / Floyd-Warshall VRAM State Routing

**Do not revive this. Read this file first if you are tempted to add graph
pathfinding to expert scheduling.**

Retired 2026-08-17. Replaced by direct cost lookup + SLA-bounded cluster
scheduling in [`src/runtime/router/vram_state_router.py`](../../../src/runtime/router/vram_state_router.py).

---

## What it claimed

That modelling GPU weight configurations as a directed graph and solving
all-pairs shortest path with Floyd-Warshall finds minimum-cost transition
sequences for multi-expert serving. It was tiered 🚀 *Genuine Discovery* with
"Zero-to-One Novelty … invented specifically during this research".

## Why it was wrong

The algorithm never affected a single scheduling decision, and could not have.

**1. The cost model is destination-only.** `WeightFoldingEngine.activate(e)`
writes `W_live = W0 + s*(U@V)` as one fused addmm per slot, reading from the
pristine buffer. It never restores first and never reads the live weights. So
entering state *v* costs the same from every source. Measured
([`results/vram_transition_costs.json`](../../../results/vram_transition_costs.json),
reproduce with [`calibrate_transition_costs.py`](../../runtime/cost_model/calibrate_transition_costs.py)):

| from \ to | astral | postgresql | financial |
| :--- | ---: | ---: | ---: |
| pristine | 17.97 | 17.92 | 17.99 |
| astral | 18.27 | 18.10 | 18.17 |
| postgresql | 18.13 | 18.19 | 18.18 |
| financial | 18.12 | 18.16 | 18.22 |

Spread across sources: **0.23–0.31 ms**. Measurement noise: **0.66–2.12 ms**.
Every row is the same row.

**2. Therefore APSP is provably degenerate.** With `C(u,v) = f(v)` and `f > 0`,
any detour costs `f(k) + f(v) > f(v)`, so the direct edge is always the shortest
path and the solve returns its own input. Run
[`floyd_warshall_apsp.py`](floyd_warshall_apsp.py) — 0 pairs improved, at every
expert count from 1 to 10 (3136 pairs at the largest). This is a property of the
cost model, not of the 3-expert graph that happened to be configured.

**3. The cost term was also numerically inert in the scheduler.** The old scorer
was `oldest_age + urgency_bonus + affinity_bonus − γ·transition_cost_s`. With
γ=0.05 and an 18 ms cost, that term was **0.0009** against a hardcoded
`affinity_bonus = 3.0` — a factor of **3333**. Sweeping γ over
[0, 0.05, 1, 1000] produced byte-identical schedules. All the clustering came
from the affinity constant; none came from the graph.

**4. The decomposed cost model was wrong even where its total was right.** It
assumed `C(u,v) = t_restore(8.0) + t_fold(10.0)`. Measured, `restore()` is
**13.95 ms** and a cross-expert swap is a single fused addmm at **18.14 ms** —
not 13.95 + 17.97 = 31.9. It got 18 ms for cross-expert swaps by coincidence and
the wrong number for every transition involving pristine.

**5. Even granting perfect clustering, there is nothing material to win.**
Measured end-to-end against a live server, swap overhead is **0.86% of wall
clock**. The A/B (30 requests, concurrency 10, alternating repeats) cut GPU swaps
25.5 → 16.0 and swap time 494 → 302 ms, and mean latency was **not significant**
(−62 ms, 95% CI [−1481, +1417]) while P95 got **worse by 3.6 s**.

## What replaced it

`VRAMStateScheduler` — SLA-bounded cluster draining. Under destination-only
costs, total cost is `sum(f(v))` over the **distinct** states entered,
independent of order. So grouping requests by target state is cost-optimal *by
construction* rather than by search, and because ordering is then free, all of
the ordering freedom is spent on deadlines instead. That is the part that
actually earns its place: across the load sweep it beats both FIFO and greedy
affinity on SLA violations (98% → 79.5% at ρ=1.5, 62.5% → 47% at ρ=0.95).

## The generalisable lesson

Before reaching for a graph algorithm, **measure whether the cost function
depends on the variable the algorithm optimises over.** Shortest-path routing
optimises over *paths*; if cost depends only on the destination, there is no path
structure to exploit and the algorithm is decoration. One 30-line calibration
script would have caught this before any of it was built.

The corollary bit harder: because the graph machinery looked like it was
working, nobody checked whether the *execution path* honoured it.
`generate_with_graph` was calling `engine.activate()` unconditionally on every
request — re-folding the same expert even on a cache hit — so the router's entire
output was being discarded downstream. The first honest A/B showed transitions
dropping 25.5 → 7.0 while swap time went *up*, which is the signature of exactly
that bug. Fixed in [`cuda_graph.py`](../../../src/runtime/cuda_graph.py)
(`apply_expert_state`).
