# Bucketed HIP Graph prefill for `runtime-next` — tried, made correct, measured a net loss, not shipped

**Status: not in the live runtime.** This directory preserves a real, complete, working implementation
that was removed from `apps/runtime-next` after a controlled A/B showed it made real HTTP time-to-first-token
(TTFT) *worse*, not better, for this engine's real prompt-length distribution. Kept here (not deleted) because
the engineering — three real, non-obvious HIP-Graph-capture correctness bugs, found and fixed with a
decisive regression test for each — is worth having on hand if this is revisited with a different bucket
design.

## What this was

`runtime-next`'s prefill path already used real batched GEMMs (§96/§103/§108 in `docs/DECISIONS.md`), but
profiling found ~58% of real prefill wall-clock time was host-side kernel-launch dispatch overhead, not GPU
compute (`diagnose_real_quantized_prefill_wall_clock_vs_kernel_time`: 138.8ms wall-clock vs. ~55-60ms of real
rocprofv3-measured GPU time on a 54-token 4B prompt). HIP Graph capture/replay is the standard fix for
per-launch dispatch overhead — record the whole kernel sequence once, replay it cheaply thereafter — but
graph replay requires every kernel launch's *shape* (grid/block size, row counts) to be fixed across replays,
which a real, varying-length prompt is not.

This feature made prefill graph-capturable by **bucketing**: padding every real prompt up to the nearest fixed
size in `PREFILL_BUCKETS = [64, 128, 192, 256]` (all multiples of `GDN_CHUNK_SIZE=64`, so GDN's own internal
chunk padding never needs extra rounding), running the same real prefill pipeline over the full bucket, then
capturing/replaying one `hipGraphExec` per bucket size the first time each is seen.

## Three real correctness bugs found and fixed (kept for reference)

1. **`causal_conv1d_prefill`'s state-handoff bug.** The kernel always snapshotted its trailing conv window at
   the very last processed row — correct when that row was real, wrong once padding rows could follow it (the
   snapshot silently captured padding-derived history, corrupting every subsequent decode step). Fixed by
   snapshotting at the real last row specifically. Caught by a real end-to-end generation test failure, not by
   inspection.
2. **HIP Graph replay does not re-evaluate host-side values baked in at capture time.** This is the deep one.
   `real_num_tokens` (how many of a bucket's rows are real prompt vs. padding) was threaded through as a plain
   host `usize`/`i32` in three places: GDN's zero-pad boundary (a host `if` branch plus a host-computed byte
   offset), the logits row-select (a host-computed pointer offset into the final hidden state), and
   `causal_conv1d_prefill`'s snapshot point. A captured graph freezes all three at whichever real prompt
   happened to trigger capture for a given bucket; every *later* real request replaying that same bucket's
   graph silently reused the *first* request's `real_num_tokens`, corrupting its own logits and GDN state
   regardless of its own actual length. Found via a real, reproduced HTTP sequence (a short warmup request
   captured a bucket; a longer, different real request later replayed it and generated only 2 tokens before an
   incorrect early EOS) — not caught by any test that only replayed with the *same* prompt used to capture,
   which is exactly why bug #3 below exists. Fixed by moving all three to real device-side reads: a
   `real_num_tokens_buf: DeviceBuffer<i32>` written fresh (via a plain, uncaptured host→device copy) before
   *every* real call, read from *inside* three kernels (`zero_from_device_offset_bf16/f32` for the GDN
   zero-fill, made unconditional so the zero-fill node always exists in the graph; `gather_last_real_row_bf16`
   for the logits row-select, gathering into a fixed destination so `rmsnorm`'s input pointer never changes
   across replays; and `causal_conv1d_prefill`'s own snapshot-point comparison).
3. **A real HIP Graph capture stack overflow.** Capturing the whole 32-layer prefill graph (thousands of
   kernel-launch nodes) overflowed the default 8MB thread stack — confirmed via a real `RUST_MIN_STACK`
   experiment, fixed with a dedicated `std::thread::Builder::stack_size(64MB)` scoped thread for the capture
   step specifically.

The decisive test that closes the exact gap bug #2 exposed —
`real_graphed_bucketed_prefill_replay_with_different_real_num_tokens_matches_eager_reference` in
`snapshot_src/model.rs` — captures a bucket's graph with one real prompt, then replays that *same* graph with
a genuinely different-length real prompt sharing the bucket, and checks the result against an independent
eager reference. Every other test in the original implementation only ever replayed with the identical prompt
used to capture, which cannot catch a stale-host-value bug since the "stale" and "real" values are then always
equal.

## Why it wasn't shipped: a real, measured net TTFT regression

Once fully correct, a controlled A/B (same binary, same real HTTP benchmark, same 3 real prompts, only
`server.rs`'s `start_request` toggled between the bucketed+graphed path and the existing eager
`forward_prefill`) found bucketed+graphed prefill *slower*, not faster, at 4B:

| path | TTFT (avg of 3 real tasks) |
| :--- | ---: |
| eager `forward_prefill` (shipped) | **168.8ms** |
| bucketed + HIP-Graph-captured (this feature) | 276.2ms (+64%) |

Root cause, reasoned through and consistent with the magnitude observed: real prompts in this benchmark are
~35-54 tokens, which round up to `bucket=64` — 10-30 wasted padding tokens per real prefill call. Padding cost
is not uniform across the pipeline: causal attention's cost scales roughly with T² for a fresh prefill from
position 0, so padding a ~42-token prompt to 64 tokens (1.52x linear) inflates the attention GEMMs by roughly
1.52² ≈ 2.3x. That compute-side loss was larger than the dispatch-overhead the graph capture was designed to
recover. The bucket granularity (64/128/192/256) was simply too coarse relative to this workload's real
prompt-length distribution.

## How to revisit this

The correctness work above (the device-side `real_num_tokens` pattern, the stack-overflow fix, the decisive
replay-with-different-length test) is reusable as-is. What would need to change to make the *performance* case
work:

- **Finer-grained buckets** (e.g. 16/32/48/64/... instead of 64/128/192/256) to cut average padding waste,
  re-measured against eager with the same controlled A/B methodology before shipping.
- **Or: per-exact-length capture with an LRU cache** instead of fixed buckets, trading a bounded number of
  capture events (one per distinct real prompt length seen recently) for zero padding waste — more graphs to
  manage, no compute inflation.
- Either direction should be validated with the same real, controlled, eager-vs-graphed A/B this write-up
  used — not just kernel-level or unit-level numbers, which is exactly what let the original TTFT regression
  go undetected until a real HTTP benchmark caught it.

## Contents of this directory

- `snapshot_src/model.rs`, `snapshot_src/kernels.rs`, `snapshot_src/server.rs` — the full, working, real
  source files as they stood right before this feature was stripped back out, including all decisive tests
  (`real_bucketed_prefill_matches_real_qwen3_5_4b_greedy_generation`,
  `real_graphed_bucketed_prefill_matches_real_qwen3_5_4b_greedy_generation`,
  `real_graphed_bucketed_prefill_replay_with_different_real_num_tokens_matches_eager_reference`,
  `real_gdn_chunk_forward_prefill_bucketed_padding_matches_real_transformers_function`,
  `diagnose_real_bucketed_vs_eager_prefill_state_divergence`).
- `snapshot_kernels/zero_from_device_offset.hip`, `snapshot_kernels/gather_last_real_row.hip` — the two new
  device-side kernels bug #2's fix required.
- `snapshot_kernels/causal_conv1d_prefill_with_real_num_tokens.hip` — the bug #1/#2-fixed variant of the
  existing `causal_conv1d_prefill` kernel (the live runtime's copy was reverted to its simpler, pre-bucketing
  form, since eager-only prefill never needs the `real_num_tokens != num_tokens` distinction).

See `docs/DECISIONS.md` §117 for the same story in this project's standard decision-log format.
