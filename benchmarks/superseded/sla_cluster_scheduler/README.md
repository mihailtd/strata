> # ⚠️ RETIRED — this whole directory is provenance
>
> The scheduler documented below **worked** and beat both baselines on deadline
> misses. It was retired because the engine turned out to be **single-tenant**:
> one agent walking a deterministic tool DAG, one node active at a time, so
> there is nothing to reorder. Revive only if the engine becomes multi-tenant.
> See [`docs/DECISIONS.md`](../../../docs/DECISIONS.md) §6.

# VRAM State Router — SLA-bounded cluster scheduling

> **Tier**: ⭐ Industry Standard, with one measured finding worth keeping.
> **Not** 🚀. An earlier version of this document claimed 🚀 *Genuine Discovery*
> for "Floyd-Warshall All-Pairs Shortest Path applied to GPU memory state
> transitions". That was wrong, and it is instructive *how* it was wrong — see
> [`benchmarks/superseded/apsp_floyd_warshall/`](../apsp_floyd_warshall/).

---

## 1. What this is

A scheduling layer in the FastAPI serving path that reorders concurrent requests
so that requests needing the same expert run back-to-back, subject to a deadline
bound that stops minority domains from starving.

It is **not** a pathfinding system. There is deliberately no graph solver.

## 2. The measured cost model

`WeightFoldingEngine.activate(e)` writes `W_live = W0 + s*(U@V)` as one fused
addmm per slot, reading from the pristine buffer — it never restores first and
never reads the live weights. Measured on RX 7900 XTX / Qwen3.5-4B bf16
([`calibrate_transition_costs.py`](../../runtime/cost_model/calibrate_transition_costs.py) →
[`results/vram_transition_costs.json`](../../../results/vram_transition_costs.json)):

| from \ to | astral | postgresql | financial |
| :--- | ---: | ---: | ---: |
| pristine | 17.97 | 17.92 | 17.99 |
| astral | 18.27 | 18.10 | 18.17 |
| postgresql | 18.13 | 18.19 | 18.18 |
| financial | 18.12 | 18.16 | 18.22 |

Every row is identical: spread across sources 0.23–0.31 ms against 0.66–2.12 ms
of noise. **Cost is destination-only: `C(u,v) = f(v)`.**

| operation | measured |
| :--- | ---: |
| `restore()` (pure pristine copy) | 13.95 ms |
| `activate(e)` (one fused addmm) | 17.97 ms |
| `activate_many([a,b])` (2-expert stack) | 25.86 ms |
| cold adapter reload from disk | 35.4 ms |

Two consequences, both provable and both pinned by tests:

1. **Shortest-path routing is vacuous.** Any detour costs `f(k) + f(v) > f(v)`.
2. **Ordering is free; only clustering pays.** A batch touching a set `S` of
   distinct states costs `sum(f(v) for v in S)` regardless of visit order. So
   grouping by target state is cost-optimal *by construction*, and all of the
   ordering freedom can be spent on deadlines.

## 3. What it actually buys — measured

### Offline sweep (200 requests/arm, real GPU folds, modelled generation)

Run: `benchmark_vram_router.py`. Transitions are real hardware; generation is
charged at a fixed rate so a 200-request sweep finishes in minutes. Every regime
is charged identically.

| ρ | | Legacy | FIFO | Greedy | **Router** |
| :--- | :--- | ---: | ---: | ---: | ---: |
| 0.5 | SLA violations | 16.5% | 14.5% | 13.5% | **13.0%** |
| 0.8 | SLA violations | 48.5% | 43.5% | 38.5% | **33.0%** |
| 0.95 | SLA violations | 71.0% | 62.5% | 51.5% | **47.0%** |
| 1.5 | SLA violations | 98.0% | 98.0% | 96.0% | **79.5%** |

**Latency is not significantly different at any load** — every paired bootstrap
CI contains zero (ρ=1.5: −536.9 ms, 95% CI [−3016, +1922]). Swap overhead is
0.10–1.20% of makespan, so there was never a latency win available.

Greedy affinity wins on raw swap count (11 vs 57 at ρ=1.5) by starving minority
domains: its P95 is 65805 ms vs the router's 58254 ms. The router deliberately
pays more swaps to protect deadlines. **That tradeoff is the contribution.**

### End-to-end A/B against the live server

Run: `benchmark_router_e2e.py`. Both arms hit the same server process, same
loaded model, same warm CUDA graph; arms alternate ABAB so drift is shared.

| Metric | FIFO | Router | Delta |
| :--- | ---: | ---: | ---: |
| GPU state transitions | 25.5 | 16.0 | **−37%** |
| Swap overhead (ms) | 494.4 | 302.4 | **−39%** |
| Mean latency (ms) | 16313.1 | 16251.1 | −62 → **not significant** |
| P95 latency (ms) | 19283.7 | 22872.6 | **+3589 (worse)** |
| Wall clock (s) | 57.57 | 57.35 | −0.23 |

Paired mean latency difference: −62 ms, 95% CI [−1481, +1417]. **Swap overhead
is 0.86% of wall clock** — that is the entire budget any scheduler can compete
for, which is why a 39% cut in it is invisible end-to-end.

## 4. Read this before optimising anything here

**Swap cost is already 0.86% of wall clock.** Any further work on transition
scheduling is bounded by that number. Two specific traps already paid for:

- **Do not add a graph algorithm.** Measure whether the cost function depends on
  the variable the algorithm optimises over. Here it does not.
- **Do not trust a scheduler's counters without checking the execution path
  honours them.** `generate_with_graph` was re-folding on every request even on a
  cache hit, so the router's output was discarded downstream. The signature was
  transitions dropping 25.5 → 7.0 while swap *time went up*. Fixed via
  `apply_expert_state` in [`cuda_graph.py`](../../../src/runtime/cuda_graph.py).

**Stacking is disabled on purpose.** Co-residency is cheaper (25.86 ms vs
2 × 18.14 = 36.28 ms, saving ~10.4 ms), but the accuracy question is already
answered and the answer is no — see
[`CURRENT.md`](../../../CURRENT.md) open question 1: `ast+fin` costs astral
**−10.96pp, 95% CI [−21.50, −1.62]** (resolved loss, n=40). Trading 11 accuracy
points for 10.4 ms — 0.55% of a request — is not a trade worth making.

## 5. Files

| file | what |
| :--- | :--- |
| [`calibrate_transition_costs.py`](../../runtime/cost_model/calibrate_transition_costs.py) | Measures the transition matrix; decides whether cost is destination-only |
| [`benchmark_vram_router.py`](benchmark_vram_router.py) | Offered-load sweep, 4 regimes, real folds, paired CIs |
| [`benchmark_router_e2e.py`](benchmark_router_e2e.py) | Live-server A/B with a real FIFO control arm |
| [`vram_state_router.py`](../../../src/runtime/router/vram_state_router.py) | `TransitionCosts`, `VRAMStateGraph`, `VRAMStateScheduler` |
| [`test_vram_state_router.py`](../../../tests/test_vram_state_router.py) | 12 tests, incl. the properties that keep the solver retired |

## 6. Reproduce

```bash
# 1. Measure the cost model (decides whether pathfinding could ever help)
uv run --env-file .env python benchmarks/runtime/router/vram_state_routing/calibrate_transition_costs.py

# 2. Offline sweep across offered load
uv run --env-file .env python benchmarks/runtime/router/vram_state_routing/benchmark_vram_router.py --num-requests 200

# 3. Live A/B (server must be running)
uv run --env-file .env python scripts/serve/run_openai_api_server.py &
uv run --env-file .env python benchmarks/runtime/router/vram_state_routing/benchmark_router_e2e.py --repeats 2
```

Runtime A/B switch (no second model load required):

```bash
curl -X POST localhost:8000/v1/router/config \
  -H 'Content-Type: application/json' \
  -d '{"enabled": false, "reset_telemetry": true}'   # FIFO control arm
```
