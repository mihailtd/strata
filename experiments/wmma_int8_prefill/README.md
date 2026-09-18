# RDNA3 WMMA INT8 fragment-layout debugging trail — the probes and debug kernels that found the real bug, not shipped themselves

**Status: not in the live runtime.** The *production* result of this investigation — a real RDNA3 hardware
WMMA INT8 tensor-core prefill kernel, 1.9x-3.3x faster than the scalar kernel it replaced, real end-to-end
TTFT wins at every quantized model size — **is shipped** and lives at
`apps/runtime-next/src/kernels/w4a16_gemm_prefill_wmma_int8.hip`, wired into
`raw::linear_quantized_prefill` in `model.rs`. This directory preserves the isolated probes and debug
kernels that got there: real, working tools that either confirmed a hypothesis, falsified one, or localized
a real bug — kept here (not deleted) because rebuilding this reasoning from scratch would waste the exact
effort this trail already spent, not because they belong in the live runtime.

See `docs/DECISIONS.md` §125/§126 for the same story in this project's standard decision-log format. This
README is the narrative version.

## The problem

`llama.cpp`'s own quantized prefill kernel dispatches to RDNA3's hardware WMMA INT8 tensor cores for `Q4_K`
unconditionally (`ggml_cuda_should_use_mmq`, confirmed by direct source reading). This engine's own W4A16
prefill kernel was still doing scalar VALU dequant+FMA, running at only 10.6% of RDNA3's VALU peak on a real
27B shape — real, substantial headroom, and the real reason quantized TTFT lost to `llama.cpp`/`Ollama` at
4B, 9B, and 27B.

## Attempt 1: `wmma_probe.hip` — dense bf16, FALSIFIED

A minimal single-tile bf16×bf16→f32 WMMA GEMM, fragment-layout formulas hand-derived from `mma.cuh`'s
generic `get_i`/`get_j` template (`load_generic`'s own indexing pattern). Real, disclosed, unresolved
ambiguity going in: `get_j(l)` for the A/B operand tiles returned values in `[0,16)`, too large for a
naturally-packed 16-column row — the concrete hypothesis tested was that RDNA3 expects each bf16 value
physically duplicated into both lanes of its own `half2`.

**Verdict: falsified.** Real GPU output diverged from a real, independent CPU reference by max abs diff 119
— far beyond bf16 rounding noise. A follow-up sorted-multiset check ruled out a simple output-indexing swap
as the cause (diff 4537, not ~0) — the actual dot products computed were genuinely wrong, not just
misplaced. Real, useful negative result: it converted "WMMA might be intricate" from a read-only impression
into a specific, tested, falsified hypothesis.

## Attempt 2: `wmma_int8_probe.hip` — the real INT8 path, CONFIRMED bit-exact

The real breakthrough was reading the *exact* function `llama.cpp`'s own dispatch path actually calls for
this operand shape: `load_ldmatrix`'s RDNA3 branch for `tile<16,8,T,dl>`, not the generic `load_generic`
attempt 1 used. That function carries a real, load-bearing `static_assert(sizeof(t.x) == 32, "bad ne")` — 8
real int32 elements per lane, not the 4 the generic (non-mirrored) template's `ne = I*J/32` formula gives.
This resolved a real contradiction the investigation had been stuck on across a context-compaction boundary:
`mma()`'s RDNA3 branch uses `a_vec[0]`/`a_vec[1]` (two 4-int chunks, needing 8 ints total) against what
looked like a 4-int `A.x` — the generic template's `ne` formula was simply the wrong specialization.

Real, confirmed layout for `tile<16,8,int,DATA_LAYOUT_I_MAJOR_MIRRORED>`: `get_i(l) = tid % 16` (constant in
`l`), `get_j(l) = l`. Each lane's row is `tid % 16`, holding that row's 8 consecutive int32 (32 int8 values)
— genuinely duplicated across the two 16-lane wave halves, not the column-parity split attempt 1 assumed for
a different operand shape.

**Verdict: bit-exact (max abs diff = 0) on the first real run**, against an independent CPU int32 reference.

## Debug kernels: `wmma_int8_probe_4chunk.hip`, `w4a16_gemm_prefill_wmma_int8_debug.hip`

Building the real, production-shaped kernel around the confirmed layout hit a real, large error (max abs
diff 10.88 against real 27B weights, values of magnitude ~0.05-0.5) on the first attempt. Two purpose-built
debug tools localized it:

- `wmma_int8_probe_4chunk.hip`: the same confirmed layout extended to 4 real K=32 chunks (K=128 total) with
  simple hardcoded data. Matched the CPU reference bit-exactly — ruled out the multi-chunk accumulation loop
  as the cause.
- `w4a16_gemm_prefill_wmma_int8_debug.hip`: an instrumented copy of the production kernel, dumping per-thread
  `my_row` and raw fragment/accumulator values to device buffers for host-side inspection.

Along the way, a **real, separate test-data bug** briefly confounded the investigation: a synthetic weight
generator of the form `(i*C + ...) % 16` with `i = row*words_per_row + word` is structurally row-INVARIANT
whenever `words_per_row` is itself a multiple of 16 (`row*words_per_row*C mod 16 == 0` regardless of `row` or
`C`) — every row had identical test data, making the real signal briefly invisible. Not a kernel bug; fixed
by deriving nibbles from `row` and `word` separately (the corrected generator lives in the production test,
`real_w4a16_gemm_prefill_wmma_int8_matches_cpu_reference_and_shipped_kernel`, in the live `kernels.rs`).

With genuinely varying test data, the debug dump found the real bug precisely: `w_scale` was computed from
`my_row` (the row whose weight data *this lane* feeds into the WMMA "A" operand) and applied to all 8 of that
lane's `facc_f[l]` outputs. But RDNA3's WMMA hardware combines all 32 lanes' operand data into one real 16x16
cross product — a lane's own accumulator slot corresponds to the real output row
`row_base + 2*l + lane_id/16`, generally **not** `my_row` at all. Fixed in the production kernel by computing
the real `out_row` per `l` and fetching *that* row's scale directly, never the irrelevant `my_row`-based one.

A related, real, latent correctness hazard was found and fixed at the same time: the WMMA calls were
conditionally skipped per-lane via `if (my_row_valid)`, unsafe for a warp-collective instruction when
validity is non-uniform across a warp (real for `out_features` not a multiple of 128, not exercised by the
27B `down_proj` shape that surfaced the scale bug, but real regardless). Fixed by running the WMMA calls
unconditionally, feeding a safe all-8-nibbles dummy for out-of-bounds rows.

## Contents of this directory

- `snapshot_kernels/wmma_probe.hip` — attempt 1, falsified, dense bf16.
- `snapshot_kernels/wmma_int8_probe.hip` — attempt 2, confirmed bit-exact, the real INT8 fragment layout.
- `snapshot_kernels/wmma_int8_probe_4chunk.hip` — debug tool, ruled out the multi-chunk accumulation loop.
- `snapshot_kernels/w4a16_gemm_prefill_wmma_int8_debug.hip` — debug tool, instrumented dump that localized
  the real scale-indexing bug.
- `snapshot_src/kernels_ffi_and_wrappers_and_tests.rs` — the real FFI declarations, safe wrappers, and test
  functions for all four kernels above, exactly as they stood in `apps/runtime-next/src/kernels.rs` right
  before removal (`diagnose_real_wmma_probe_16x16x16_vs_cpu_reference`,
  `real_wmma_int8_probe_16x16x32_matches_cpu_reference`,
  `diagnose_real_wmma_int8_probe_4chunk_vs_cpu_reference`, `diagnose_real_wmma_int8_debug_dump`,
  `diagnose_real_wmma_int8_minimal_single_group_single_block`).

## What stayed in the live runtime

- `apps/runtime-next/src/kernels/w4a16_gemm_prefill_wmma_int8.hip` — the production kernel.
- `apps/runtime-next/src/model.rs` — `raw::linear_quantized_prefill` dispatches to it.
- `apps/runtime-next/src/kernels.rs` —
  `real_w4a16_gemm_prefill_wmma_int8_matches_cpu_reference_and_shipped_kernel` (the real, permanent
  correctness gate, real 27B weights, real CPU reference + real shipped-kernel comparison) and
  `bench_real_w4a16_gemm_prefill_tile_n16_vs_wmma_int8` (the real, permanent kernel-level benchmark).
