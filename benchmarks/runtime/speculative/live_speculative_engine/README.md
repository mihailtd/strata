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
and `SelectiveHybridPOETRingBuffer` (`src/runtime/state_ring_buffer.py`), using
GatedDeltaNet-shaped bf16 tensors (`[1,4,128,128]` × 24 SSM layers) that were never run
through the actual model.

## ⚠️ Part B is not measuring the live server, and right now it can't be

`BucketedSpeculativeDecoder` (`src/runtime/bucketed_speculative.py`) is what
actually runs the live decode loop, and its `_snapshot_ssm()` / `_restore_ssm()` are
hardcoded in-place `copy_()` calls inside the captured CUDA graph region:

```
grep -n "_snapshot_ssm\|_restore_ssm\|RingBufferReplayEngine\|SelectiveHybridPOET" \
    src/runtime/bucketed_speculative.py
```

returns the two `copy_()` sites and nothing else — no reference to
`RingBufferReplayEngine` or `SelectiveHybridPOETRingBuffer` anywhere in that file.
`model_state["ring_buffer_mode"]` (`server.py`, set from the `RING_BUFFER_MODE` env
var) is stored and returned by `GET /api/engine/status`, but **nothing reads it** to
route the live rollback through the compressed buffer. Selecting a mode currently
changes a label, not behavior — confirmed 2026-08-20, `/api/engine/status` now
reports this explicitly as `ring_buffer_wired: false` rather than implying otherwise.

So Part B's **2.4× VRAM / 264µs rollback** numbers are real measurements of the
`SelectiveHybridPOETRingBuffer` class in isolation (also covered by
`tests/test_state_ring_buffer.py`), not of what the live server does when a request
gets rejected mid-draft. Wiring the two together — replacing the graph-capture-region
`copy_()` calls with calls into the ring buffer without breaking capture, which the
file's own comments flag as delicate ("any host sync inside capture is illegal") — is
open work, not done. Do not read this README as claiming the live engine gets the
2.4× VRAM saving; it does not, yet.

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
- `src/runtime/bucketed_speculative.py` — the live decode loop; `BucketedSpeculativeDecoder`
- `src/runtime/state_ring_buffer.py` — the four ring buffer implementations Part B measures
- `src/runtime/alpha_calibration.py` + `/api/factory/calibrate_alpha` — unrelated
  feature landed alongside this one; documented in
  `benchmarks/factory/geometry/dynamic_alpha_calibration/README.md`
