# Live Speculative Engine — throughput against the running server, plus a ring-buffer microbenchmark

> Run: `uv run python benchmarks/runtime/speculative/live_speculative_engine/benchmark_live_speculative_engine.py`
> Needs the server already up (`uv run --env-file .env python -m runtime.server`) — this
> benchmark does not load its own model copy, it calls the live one over HTTP.

## What it measures

**Part A — live throughput.** SSE streaming against `localhost:8000`, reading the
server's own `usage.tokens_per_second`, across a small set of domain prompts
(`astral_uv_fastapi`, `postgresql_pgvector`, `duckdb_analytics`). The "baseline"
(non-speculative) arm is **simulated** by capping `max_tokens` short rather than by
restarting the server with `SPECULATIVE_DECODE=0` — read the script's own docstring
before trusting a speedup ratio computed from these two arms; a short-cap simulation
and an actually-disabled decoder are not guaranteed to differ only in speculation.

**Part B — ring buffer micro-benchmark, CPU only, synthetic tensors.** Push/rollback
latency and VRAM footprint for `StateRingBuffer` (dense), `POETCompressedStateRingBuffer`,
and `SelectiveHybridPOETRingBuffer` (`apps/runtime/state_ring_buffer.py`), using
GatedDeltaNet-shaped bf16 tensors (`[1,4,128,128]` × 24 SSM layers) that were never run
through the actual model.

## ✅ CORRECTED (2026-09-13): Part B's ring buffer IS wired into the live decoder

This section used to say the ring buffer was not wired into `BucketedSpeculativeDecoder`
and that `/api/engine/status` reported `ring_buffer_wired: false`. That was accurate
when written, but the wiring landed the same day and this doc never caught up — a
stale-documentation bug, not a fabrication (see `docs/EXPERIMENT_REAUDIT_2026-09.md`'s
"Documentation staleness" note).

Direct code check today: `apps/runtime/bucketed_speculative.py` imports
`RingBufferReplayEngine` and instantiates it unconditionally in `capture()`
(`self.ring_engine = RingBufferReplayEngine(self.cache, max_depth=64, mode=ring_mode)`,
`ring_mode` from `RING_BUFFER_MODE`), and it is genuinely exercised on the real decode
path — `reset()`, `checkpoint()`, `commit_on_acceptance()`, and
`rollback_on_rejection()` are all called at real accept/reject decision points during
speculative verification, not just constructed and left idle. `server.py` reports
`ring_buffer_wired=ring_engine is not None` via `/api/engine/status`, which is `True`
whenever speculative decoding is active.

So Part B's **2.4× VRAM / 264µs rollback** numbers, measured on `SelectiveHybridPOETRingBuffer`
in isolation, now describe (to first order) what the live decode loop's real rollback
mechanism actually is — though this README has not re-measured the *live* engine's
rollback latency end-to-end to confirm the isolated micro-benchmark numbers transfer
unchanged once real CUDA-graph replay and real speculative traffic are in the loop.
That end-to-end re-measurement is open work; the wiring itself is not.

## What IS live: speculative draft depth (K) is now a runtime control, not a restart

Until 2026-08-20, `SPECULATIVE_K` was read once at server startup
(`load_inference_engine()`) to size `BucketedSpeculativeDecoder`'s captured CUDA
graphs, and `/api/engine/status`'s reported `spec_k` **re-read the env var on every
call** — so it could report a K the live decoder was never built for if the
environment changed after startup. Fixed:

- `POST /api/engine/set_speculative_k {"k": 2|4|8}` rebuilds
  `BucketedSpeculativeDecoder` at the new K and recaptures its graphs. This does
  **not** reload the 4B base model (weights stay resident) — only the draft head's
  graph set is rebuilt, so the cost is graph-capture time, not model-load time. If
  capture at the new K fails, the previous decoder is left serving untouched; nothing
  swaps until the new one has captured cleanly.
- `GET /api/engine/status` now reports `spec_k` from the live decoder object
  (`model_state["spec_decoder_k"]`), plus `spec_k_options: [2, 4, 8]` — the three
  values the dashboard exposes and the only ones this benchmark has ever exercised.
- A **draft K** control sits in the dashboard header next to the engine load/unload
  toggle: click 2 / 4 / 8, watch the swap-status readout, then use the existing chat
  panel to see the live tok/s at that K — that IS the comparison; no separate
  benchmark-runner UI was built, since the chat panel already reports tok/s per
  response and re-running a full 3-arm HTTP benchmark on every click would cost real
  GPU time for a number the chat panel already shows.

`k=2`, `k=4`, `k=8` are the only values wired into the dashboard and this benchmark's
prompt set — chosen because they're what was actually measured, not because larger K
is refused for a principled reason.

## Part A results on record (2026-08-20 walkthrough — NOT independently reproduced)

| Domain | tok/s (med) | tok/s (max) | TTFT |
| :--- | ---: | ---: | ---: |
| `astral_uv_fastapi` | 34.1 | 34.1 | 37 ms |
| `postgresql_pgvector` | 41.3 | 42.7 | 36 ms |
| `duckdb_analytics` | 36.8 | 37.2 | 36 ms |

Recorded at K=2. These numbers came from another session's run against the live
server and have not been re-run or independently checked here — treat them as a
starting point for comparison once K becomes a real, swappable control, not as a
settled baseline. A natural next measurement, now that it's cheap: the same three
prompts at K=4 and K=8, to see whether tok/s actually climbs with draft depth on this
model or whether acceptance rate falls off fast enough to cancel it out.

## Files

- `benchmark_live_speculative_engine.py` — this benchmark
- `apps/runtime/bucketed_speculative.py` — the live decode loop; `BucketedSpeculativeDecoder`
- `apps/runtime/state_ring_buffer.py` — the four ring buffer implementations Part B measures
- `apps/runtime/alpha_calibration.py` + `/api/factory/calibrate_alpha` — unrelated
  feature landed alongside this one; documented in
  `benchmarks/factory/geometry/dynamic_alpha_calibration/README.md`
