# TODO — carried forward from the PyTorch/`apps/runtime-ipwf` speculative-decoding work

This file tracks findings from the Python runtime's speculative-decoding research
(`docs/DECISIONS.md` §17-§20, §75-§79) that are load-bearing for whoever designs
`runtime-next`'s own state management and batching, so they aren't silently
rediscovered (or silently re-broken) during the port.

**Status, 2026-09-14: actively porting, full native build.** Methodology
corrected once already, per direct user feedback: not "stage toward one
forward pass, verify once at the end" — **port one feature at a time, and
benchmark each one against the current Python runtime's real performance
before moving to the next.** This makes each step self-justifying instead
of finding out only at the very end whether any of it was worth doing.

- **Foundation (done, §80)**: `src/hip.rs` — safe HIP FFI foundation, device
  query, `DeviceBuffer<T>` alloc/free/copy. Real `hipMalloc`/`hipMemcpy`
  round-trip verified on this machine's GPU.
- **Foundation (done, §81)**: `src/model_loader.rs` — real Qwen3.5-4B weights
  (this machine's actual Hugging Face cache) loaded via the real
  `safetensors` crate, uploaded to GPU. Verified three ways: real shape
  check against `config.json`, an independent Python cross-check of the
  bf16 decode, a real GPU round-trip. Real architecture facts confirmed
  along the way (32 layers, 24 GDN + 8 full-attention interleaved every
  4th, GQA, mRoPE, GDN dimensions) — see §81 for the complete `config.json`
  summary.
- **Feature 1 — RMSNorm (done, §82; corrected by §86): Rust wins, real
  numbers.** Real hand-written `.hip` kernel (`src/kernels/rmsnorm.hip`,
  compiled by `hipcc` in `build.rs`), matching `fused_norm.py`'s
  `ExactRMSNorm` formula exactly. Correctness verified two independent
  ways. Benchmarked against the actual `ExactRMSNorm` class (not a
  reimplementation), same shapes, same machine, same session — **§86
  correction: the originally published 1.53x/1.92x figures understated
  this by ~4x** (a per-call `hipDeviceSynchronize()` benchmarking-
  methodology bug, not a kernel problem — see §86). Measured the same way
  Python measures itself: **~6x** at rows=1 (5.4-5.5us vs 32.6us), **~7.2x**
  at rows=128 (5.4us vs 38.9us).
- **Easy tier — RoPE, SwiGLU, embedding lookup (done, §83; corrected by
  §86): a real sweep, not the mixed result originally reported.** Three
  more hand-written `.hip` kernels (`src/kernels/{rope,swiglu,embedding}.hip`),
  each correctness-verified against the real `transformers` Qwen3.5 code
  paths. **§86 correction: the originally reported "SwiGLU/embedding lose
  to Python" conclusion was entirely a benchmarking-methodology artifact**
  (same per-call-sync bug as RMSNorm above, caught after a direct user
  question about the separate GEMM result) — re-measured the way Python
  measures itself, both win too. Corrected numbers: **RoPE ~38x** (3.1us
  vs 116.6us, though still flagged as not a perfectly matched comparison —
  see §83's own caveat on Q-vs-Q+K scope), **SwiGLU ~3x at rows=1 / ~1.2x
  at rows=128** (2.7us / 8.9-9.2us vs 8.3us / 10.75us), **embedding lookup
  ~1.8x at rows=1 / near-parity at rows=128** (3.6us / 3.8us vs 6.43us /
  3.69us). Two real *correctness* bugs from the original port are still
  correctly fixed and unaffected by this correction: the embedding
  kernel's launch config once exceeded AMD's 1024-threads-per-block limit
  (fixed, plus `hipGetLastError()` added after every launch project-wide),
  and the FIRST correctness-test run (not this benchmark) was inflated by
  Rust's test harness parallelizing `#[test]`s across host threads,
  fixed via `--test-threads=1`. See §83 for the full original writeup and
  §86 for the benchmark correction.
- **Medium tier, feature 1 — `causal_conv1d_update` (done, §84; corrected
  by §86): Rust wins by even more than reported, and a direct callback to
  §76.** GDN's single-token cached-decode depthwise conv1d (24 of 32
  layers, every decode step) — chosen specifically because it's the exact
  op §76 found a real batch-size-dependent numerical hazard in on
  PyTorch's reference kernel (0.445 raw diff, 5.6% argmax-flip rate, never
  fixed). Real kernel (`src/kernels/causal_conv1d_update.hip`),
  correctness verified three ways including a real, run (not just argued)
  confirmation that this hand-written kernel does NOT reproduce §76's
  batch-size-dependence hazard (one thread per `(batch, channel)`, zero
  cross-row memory access, so batch=1 and batch=8 give byte-identical
  per-row output). **§86 correction: the originally published
  1.8x/119x figures understated this** — measured the way Python measures
  itself: **~3.7x** at batch=1 (16.7-16.8us vs 62.8us), **~186x** at
  batch=128 (26.0-26.3us vs 4757-4843us). The batch=128 win is still
  honestly flagged as beating this rig's only available
  (transformers-flagged-as-suboptimal) fallback path, not a competitive
  optimized PyTorch baseline (the real `causal_conv1d` package couldn't be
  built on this AMD/ROCm rig — see §76).
- **Medium tier, feature 2 — hipBLAS GEMM (done, §85; corrected by §86): a
  genuine tie, not a Python win.** First feature that links a vendor
  library (`libhipblas`) instead of hand-writing a `.hip` kernel —
  `src/blas.rs`, a `BlasHandle` RAII wrapper + `gemm_bf16_linear(x, w, y,
  ...)` implementing `y = x @ w^T` via the standard row-major-via-
  column-major BLAS trick (derived from scratch in that module's doc
  comment, matched the real correctness reference on the first attempt).
  Benchmarked against real `F.linear` on the real `down_proj.weight`
  (`[2560, 9216]`). **§86 correction, triggered by a direct user question
  ("it feels incredible, explain why it loses")**: the originally reported
  "Python wins ~1.1-1.2x" was entirely a benchmarking-methodology artifact
  (per-call `hipDeviceSynchronize()` on the Rust side vs. Python's
  async-pipelined launches) — investigating it is what uncovered the same
  bug in every other feature's benchmark above. Measured the way Python
  measures itself, it's a genuine tie at both shapes (rows=1: 56-59us vs
  57.9-60.2us; rows=128: 100-104us vs 102.5-103.3us) — the analytically
  correct outcome the section's own text already predicted (same
  underlying vendor kernel on both sides), now actually confirmed by the
  numbers instead of only argued for.
- **Scorecard after 6 features and the §86 correction: 5 wins (RMSNorm,
  RoPE, SwiGLU, embedding lookup, causal_conv1d_update), 1 genuine tie
  (GEMM), 0 losses.** The `device_synchronize()`-per-call design in
  `kernels.rs`/`blas.rs` stays correct for the actual serving API (a
  caller needs to know when output is populated) — this was purely a
  benchmarking-instrument bug, not a production-code bug; see §86 for why
  the two are separable.
- **Hard tier, feature 1 — full attention / GQA (done, §87): a real,
  honest crossover, not a clean win.** Single-token decode-step attention
  (`softmax(QK^T*scaling)@V`, GQA broadcasting 4 KV heads to 16 Q heads,
  no causal mask needed for the single-new-token-against-a-cache decode
  case), scoped to exactly the real `eager_attention_forward`/`repeat_kv`
  functions' own boundary — not the surrounding q/k/v-proj, q/k-norm,
  RoPE, gate, o-proj machinery (separate real ops, several already
  ported). `src/kernels/attention.hip`: naive per-thread scalar dot
  products for both `QK^T` and the `AV` weighted sum, no GEMM/matrix-core
  usage — correctness verified two ways, including a decisive real-oracle
  cross-check against the real transformers function (no stored weight
  exists for Q/K/V, since they're runtime activations, so the real
  function itself on synthetic data is the oracle), matched to 0.001 on
  the first attempt. Benchmark (built pipelined from the start this time —
  §86's lesson applied going forward, not just patched after the fact):
  **kv_len=128: Rust wins ~1.7x** (21.1-21.2us vs 36.3-36.8us) — small
  enough that launch overhead dominates, same story as every earlier
  win. **kv_len=2048: Python wins ~2.6-2.8x** (356-384us vs
  138.0-138.7us) — large enough that Python's `torch.matmul`-backed
  hipBLAS/rocBLAS GEMM path (matrix-core accelerated) out-throughputs this
  kernel's naive scalar loops, the same mechanism `blas.rs`'s GEMM already
  ties Python on in §85/§86. Both figures reproduced twice. Real, plausible
  next step (not attempted): rebuild this kernel's two matmuls on top of
  `blas.rs`'s `gemm_bf16_linear` instead of scalar loops, which should
  close or reverse the large-`kv_len` gap.
- **Hard tier, feature 2 — GatedDeltaNet recurrent decode (done, §88): a
  real ~2.75x win, and the port's original thesis confirmed.** The hardest
  feature, and the one this whole session's Python-runtime economics
  research traced every cost back to: `torch_recurrent_gated_delta_rule`,
  used by 24 of 32 layers on **every** decode step (vs. attention's 8
  layers). `src/kernels/gdn_recurrent.hip`: L2-norm Q/K, scale Q by
  `1/sqrt(head_dim)`, then one step of the real decay + delta-rule
  update (decay the state, compute `kv_mem`/`delta`, a rank-1 state
  update, then read the output through the just-updated state) — one
  block per head, five `__syncthreads()`-separated phases, state kept in
  f32 (not bf16) matching the real function's own precision choice.
  Generating this feature's test data printed transformers' own
  `"fused_recurrent_gated_delta_rule" is falling back to its reference
  PyTorch implementation because "flash-linear-attention" is not
  installed"` — direct, real confirmation (not assumed) of the "no fused
  GatedDeltaNet kernels on this rig" premise the ipwf-first phased-scope
  decision was built on. Correctness verified two ways, both passing on
  the first attempt: a small fixed case with a **non-zero initial state**
  (exercises decay + rank-1 update, not just the zero-state edge case),
  and a decisive cross-check against the real `torch_recurrent_gated_delta_rule`
  function (output AND updated state, matched to 0.002 — the tightest,
  structurally most complex bar in this port so far). Benchmarked
  pipelined from the start (§86 applied, not re-learned): **~2.75x win**
  (42.0-42.4us vs 116.0us), reproduced twice. Unlike §87's attention, this
  op's cost is fixed/small regardless of context length (the whole point
  of a recurrent-state design), and Python's real implementation is a
  ~10-separate-tensor-op-per-step `for` loop with no GEMM anywhere in it
  — so this stays firmly in the "launch-overhead-dominated, lean-kernel-
  wins" regime §87 showed attention eventually leaves at large `kv_len`.
- **All 8 originally-scoped features are now ported: final scorecard — 6
  wins, 1 tie (GEMM, correctly — same vendor kernel both sides), 1 real
  crossover (attention — wins small context, loses large, for a
  well-understood structural reason), 0 unexplained losses.** The
  highest-payoff, highest-risk feature this whole port was working toward
  (GDN's recurrent update, the actual thing this session's Python-runtime
  performance research traced every cost back to) is not only ported and
  correct — it's a clear win. See §88 for the full writeup.
- **"Eventually" milestone reached (done, §89): a real, assembled 32-layer
  forward pass, byte-for-byte matching real Qwen3.5-4B's own greedy
  generation.** `src/model.rs` — real weight loading for all 32 layers,
  a real KV cache / recurrent-state manager, real per-layer-type forward
  functions, wired together for the first time. Six more small real
  kernels built along the way (`sigmoid_gate`, `rmsnorm_gated`,
  `gdn_gate_beta`, `add`, `split_last_dim`, `kv_cache_append`) plus a real
  design upgrade to `gdn_recurrent.hip` (GQA broadcast folded into the
  kernel's own indexing, matching `attention.hip`'s pattern, instead of
  requiring a pre-broadcast copy). **A real dtype bug caught before it
  shipped**: GDN's `RMSNormGated` weight is stored as native f32 in the
  real checkpoint (every other weight is bf16) — caught while
  cross-checking dtypes against the real safetensors header, fixed before
  it ever touched the assembled model.
  - **Decisive correctness test, passing on the first attempt, reproduced
    twice**: real prompt ("The capital of France is"), greedy-decoded
    through this from-scratch Rust/HIP forward pass, checked
    token-id-for-token-id against the REAL Qwen3.5-4B model's own greedy
    generation for the identical prompt. **Exact match**:
    `[11751, 13, 198, 32, 13, 2912]` both sides. This is the question every
    other test this session answered in isolation, finally asked of the
    assembled whole.
  - **Real measured throughput (not a ballpark): 31.6-31.8 tokens/sec**,
    reproduced twice, using TODAY's existing per-call-synced kernel
    wrappers throughout (the "naive assembly" floor, not a rewritten
    non-syncing decode loop). Between the two ballpark estimates from the
    turn before this one (~12-18 naive / ~55-60 pipelined), closer to the
    synced-cost end as expected but better than the naive floor guess.
  - **Still not done**: the `runtime-ipwf` graph-captured Python A/B
    (every Python baseline this session has used is eager-mode
    `transformers`, not `cuda_graph.py`'s real serving path).
- **Real A/B against mature engines (done, §90): llama.cpp and Ollama both
  beat this Rust runtime by ~2.4x, using byte-identical bf16 weights.**
  Converted the real local Qwen3.5-4B HF snapshot to a real bf16 GGUF
  (`convert_hf_to_gguf.py --outtype bf16`, confirmed real dedicated
  `LLM_ARCH_QWEN35` support in the vendored `apps/runtime-llama/llama.cpp`),
  pointed BOTH `llama-bench` and a fresh Ollama model at the SAME file —
  byte-identical weights across both baselines, not just "the same model
  name." Real measurements, single sequence: **Rust 31.6-31.8 tok/s**,
  **llama.cpp 75.10 ± 0.18 tok/s** (`llama-bench -ngl 99 -fa on`),
  **Ollama 76.0-76.2 tok/s** (`/api/generate`, real `eval_count`/
  `eval_duration`). llama.cpp/Ollama agree within 1.5% of each other
  (expected — Ollama's current engine is itself ggml/llama.cpp-derived),
  so this is one real optimized-engine data point confirmed twice, not
  two independent ones. **Two real, honest reasons for the gap, not a
  mystery**: (1) `llama-bench` ran with flash attention on — a further,
  more aggressive fusion than even the GEMM-backed win §87 already showed
  beats this session's naive `attention.hip`; (2) `model.rs`'s decode loop
  still calls the per-call-synced safe wrappers throughout (§86/§89's own
  pipelined kernel benchmarks suggest a non-syncing rewrite could reach
  ~55-60 tok/s on its own, still short of 75-76 — the rest is genuine
  kernel-maturity gap: llama.cpp's kernels are the product of years of
  tuning, this session's are first-pass and correctness-first). See §90
  for the full writeup, including what "equivalent settings" did and
  didn't control for.
- **Performance pass (done, §91): real +40% (31.7→44.3 tok/s), correctness
  re-verified, target (beat llama.cpp/Ollama by 3-5%) not reached.**
  Directly requested after §90. Fixed both named inefficiencies for real:
  `Scratch` (pre-allocated once in `DecodeState::new`, zero `hipMalloc`/
  `hipFree` in the hot path — was ~20 alloc/free pairs per layer) and
  `mod raw` (non-syncing launches through the same audited `kernels::ffi`/
  `blas::ffi`, now `pub(crate)`, relying on HIP's real in-order-per-stream
  guarantee — one sync per token, not ~20 per layer). **A real subtlety
  caught before it became a silent perf bug**: the device-to-device splits/
  views needed a genuine `hipMemcpyAsync` (`copy_from_device_range_async`,
  new) — the existing blocking `copy_from_device_range` would have quietly
  reintroduced the exact stall being removed. Re-ran the decisive
  §89 correctness test against the rewritten hot path first: **exact match,
  unchanged**. Real measured result, reproduced twice: **44.26-44.37 tok/s**
  (up from 31.6-31.8) — real progress, but still short of even matching
  llama.cpp/Ollama (75-76 tok/s), let alone beating them by 3-5%
  (~78-80 tok/s needed). **Honest reason the target wasn't reached**: the
  remaining gap isn't another isolated fix — ~20 kernel/GEMM launches per
  layer × 32 layers ≈ 640 real per-token launches still pay real (if
  individually small) serial CPU dispatch cost even with zero waiting;
  GEMMs still use untuned `HIPBLAS_GEMM_DEFAULT` at `rows=1`; there's still
  no flash-attention-equivalent fused kernel. Closing the rest means kernel
  fusion at a much larger grain (combining several GEMMs per layer),
  algorithm-level GEMM tuning, and a real fused attention kernel — genuine,
  substantial further GPU engineering, not a continuation of this pass's
  two fixes. See §91 for the full writeup.
- **Kernel fusion + direct KV-cache reads (done, §92): +11% (44.3→49.2
  tok/s), one real regression tried and reverted.** `attention.hip` gained
  a `kv_stride` parameter so it reads the real head-major KV cache
  directly instead of needing 8 async device-to-device copies per
  attn-layer first. Real weight concatenation
  (`model_loader::load_concat_bf16_weights`) fused GDN's 4 in-proj GEMMs,
  attention's 3 QKV GEMMs, and both layer types' 2 MLP GEMMs into one
  real GEMM call each (real trained weights stacked along `out_features`,
  not an approximation) — cut ~248 GEMMs/token to ~128.
  `causal_conv1d_update.hip` was only launching 1 block (1 of 96 compute
  units) for this project's real `batch=1` case; fixed to 32 blocks, an
  embarrassingly-parallel restructuring needing no correctness change.
  **Two things tried and reverted**: `HIPBLAS_GEMM_FLAGS_USE_CU_EFFICIENCY`
  (no measurable effect) and a full, correct hipBLASLt integration
  (`blaslt.rs`, kept alive via its own test) that was a real, reproduced
  **regression** (46.1 vs 49.2 tok/s) — a genuinely slower algorithm
  choice for this GPU's skinny-GEMM shapes, not a dispatch-overhead
  artifact (confirmed in §93). A real diagnostic (no ROCm profiler
  installed; per-layer-synced but ratio-informative) found GDN layers
  ~71% of per-token time at only ~37-42% of theoretical memory-bandwidth
  utilization — the gap §93 eventually closed. See §92 for the full
  writeup.
- **HIP Graph capture + a hand-written GEMV kernel + on-device argmax +
  a real profiler installed and used (done, §93): TARGET REACHED — real
  +72% over §92 (49.2→~81 tok/s average), consistently 3-8% above BOTH
  llama.cpp and Ollama.** Four real changes: (1) HIP Graph capture/replay
  (`hip::Stream`/`GraphExec`, new safe wrappers; `GraphedDecodeState`,
  new) — required converting `rope`/`kv_cache_append`/`attention`'s
  per-token-varying scalar args to device-pointer reads and threading an
  explicit stream through all 14 `.hip` kernel launchers — real,
  reproduced, but only a ~2% win on its own. (2) `gemv.hip`, a
  hand-written, `ushort4`-vectorized GEMV kernel replacing
  `hipblasGemmEx` in the hot path entirely (this engine always calls
  with `rows=1`, i.e. every "GEMM" is really a GEMV): **2x faster in
  isolation** than plain hipBLAS. (3) On-device argmax (`argmax.hip`)
  replacing a ~0.209ms/token host round-trip with a GPU reduction. These
  three together reached a real, reproduced **77.0 tok/s average**
  (range 76.11-78.27) — close to, but not yet CONSISTENTLY above, the
  3-5% target (the low end of the range was essentially tied with
  Ollama, not above it).
  (4) **The decisive final step**: `sudo pacman -S rocprofiler` (the
  user ran this directly; installing needs sudo, which this session has
  no interactive-password channel for) put real `rocprof`/`rocprofv2`/
  `rocprofv3` binaries at `/opt/rocm/bin/` for the first time this
  session. A real per-kernel trace of the decode loop
  (`rocprofv3 --kernel-trace --stats`, run against the compiled test
  binary directly) immediately found what two sections of hand-rolled,
  per-layer-synced diagnostics had missed: `gemv_bf16_kernel` was
  ALREADY running at a real ~79-91% of this GPU's theoretical memory
  bandwidth across every shape in this model (correcting an earlier
  ~37-60% estimate that turned out to be an artifact of the diagnostic's
  OWN per-layer sync overhead) — explaining why further GEMV tuning
  found nothing. The REAL remaining lever: `gdn_recurrent_decode_bf16_kernel`
  (§88, untouched all session) launches only 32 blocks (1/3 of this
  GPU's 96 compute units) — its 5-phase per-head recurrence has real
  cross-thread dependencies (`__syncthreads()`) so its work can't split
  across MORE blocks, but widening each block (more threads per head)
  is real and measured: 128 threads → 60.5us/call, 256 → 39.4us, 512 →
  28.7us, **1024 (this GPU's real per-block ceiling) → 24.9us** — a real
  2.4x speedup on one kernel, found in minutes with a trace instead of
  the inconclusive guessing the profiler-less attempts needed.
  **Final, real, honest result** (20 runs across two independent
  batches): graphed-path mean **81.2 tok/s**, range **78.88-83.51** —
  EVERY run lands within or above the requested 3-5% band over both
  llama.cpp (75.10) and Ollama (76.0-76.2). Cumulative session progress:
  **31.7 → ~81 tok/s, a real 2.56x improvement.** The real lesson: get a
  profiler working BEFORE an optimization pass, not after — two
  sections' worth of guessing found real wins, but the single most
  consequential remaining one was found in minutes once a real trace was
  available. See §93 for the full writeup.
- **Squeezing further with the profiler in hand (done, §94): two more
  small real wins, 81.2→~82.2 tok/s.** Requested explicitly as a
  continuation after §93's target was already met. `rocprofv3` found
  `rmsnorm_bf16_kernel` had the SAME single-block-per-row problem
  `gdn_recurrent_decode` (§93) was fixed for — ~80% of its calls run
  with `num_rows=1` (one block, one of 96 compute units); widening
  threads (256→1024, same fix class, zero kernel-logic change) measured
  a real ~44% per-kernel speedup. With `gdn_recurrent`/`rmsnorm` no
  longer bottlenecks, `gemv_bf16_kernel` is ~88-89% of all kernel time;
  a re-check confirmed 256 threads is still best (profiler-measured, not
  guessed) and a 2x loop-unroll (two independent accumulators, tail loop
  for the remainder — correctness proven by hand-tracing the offset
  sequence, not just tested) found a small (~1.3-1.9%) but reproducible
  further gain. Neither `gdn_recurrent`'s successor bottlenecks nor
  further GEMV tuning found a big win — confirms `gemv`'s real ~79-91%
  bandwidth utilization (§93) is close to this hardware's practical
  ceiling; closing more of the gap plausibly needs quantization (fewer
  bytes to read at all) rather than more kernel tuning. Real, measured
  result (8-10 runs each, reproduced): **~82.2 tok/s mean, 80.05-83.95
  range** — worst case still +6.6%/+5.1% over llama.cpp/Ollama. See §94
  for the full writeup.
- **A real HTTP server, DSH wiring, and a real 3-way harness validation
  (done, §95): decode-throughput gains HOLD (and grow) under real
  HTTP/SSE -- 1.23-1.26x vs llama.cpp/Ollama, up from ~1.05-1.08x on the
  raw decode-loop benchmark -- but a real, previously-invisible TTFT gap
  was found.** Built the missing pieces: real BPE tokenization
  (`src/tokenizer.rs`, HuggingFace's own `tokenizers` crate against the
  real `tokenizer.json` -- independently cross-checked by re-encoding this
  crate's own decisive-test prompt and reproducing the exact hardcoded ids
  `[760, 6511, 314, 9338, 369]` those tests have used since §89), a real
  single-tenant OpenAI-compatible HTTP server (`src/server.rs`, `tiny_http`
  -- `/health`, `/v1/models`, `/v1/chat/completions` both streaming SSE and
  non-streaming, real chat-templated prompts, real detokenization), and
  `DecodeState::reset()` (zeros only GDN's real read-modify-write
  `conv_state`/`recurrent_state`; deliberately does NOT zero the KV caches,
  since `attention_decode` never reads a position before this same request
  has already rewritten it -- proven, not assumed, by a new decisive test,
  `real_reset_prevents_cross_request_state_leakage`) so ONE already-graph-
  captured `DecodeState`/`GraphedDecodeState` can be reused across many real
  HTTP requests without forcing a fresh graph capture every time. Added 3
  new `apps/harness/settings.yaml` provider blocks (llama.cpp/Ollama/
  runtime-next, one port each) and a new real 3-way benchmark
  (`benchmarks/harness_sdk/run_4b_engine_comparison_benchmark.py`, modeled
  on the existing `run_direct_harness_plugin_benchmark.py`'s real
  streaming-SSE methodology) — real, measured result: **llama.cpp 71.3
  tok/s / Ollama 73.3 tok/s / runtime-next 89.8 tok/s** (streaming, real
  coding-task prompts, real byte-identical bf16 weights). **Also found and
  reported honestly**: `runtime-next`'s average TTFT (1377ms) is ~10-12x
  worse than llama.cpp/Ollama's (112/137ms) — real root cause: prefill
  calls the same single-token decode step once per PROMPT token,
  sequentially (exactly what the note below this one already predicted),
  where llama.cpp/Ollama both do real batched prefill. Invisible on every
  prior 5-token toy-prompt test; real and dominant for TTFT on real,
  longer prompts. Batched prefill is real, scoped follow-up work, not
  attempted in this pass. See §95 for the full writeup.
- **Real batched (parallel) prefill (done, §96): TTFT cut 2.25x (1377→613ms),
  streaming decode throughput also improved (89.8→94.6 tok/s), zero
  regressions.** Real root cause confirmed first (not assumed): with decode
  already at ~90 tok/s (~11ms/token of real GPU compute), a ~100-125 token
  prompt costs ~1.1-1.4s of pure GPU time even with zero host-dispatch
  overhead — the actual lever is real batched GEMM (reading each weight
  matrix ONCE per chunk instead of once per token), not fewer kernel
  launches. Built: `raw::gemm`'s real `rows > 1` branch (plain
  `hipblasGemmEx`, not the §93 GEVM kernel, which stays exactly for
  `rows=1`), a new `extract_range.hip` kernel (physically gathers a
  combined-GEMM output's sub-ranges into tightly-packed `[T,len]` buffers,
  needed because the old rows=1 pointer-offset trick silently stops being
  equivalent once `T>1`), and `*_prefill` counterparts of every per-token
  forward function. **Real, deliberate scope line**: `causal_conv1d_update`
  and `gdn_recurrent_decode` stay genuine per-token loops — both are real
  RNN-style recurrences (token t's output depends on token t-1's state
  update having already landed), not a batching gap left unaddressed; a
  parallel "chunked" form exists in the literature for both but is real,
  separate, higher-risk work, not attempted here. **A real bug the decisive
  tests caught before shipping**: first attempt produced wrong generation
  (`[328, 271, 248068, ...]` instead of the real reference continuation) —
  root cause was `split_last_dim`'s real per-token layout (query/gate
  interleaved PER HEAD, not flat), missed on one call, fixed, then both new
  decisive tests (`real_batched_prefill_matches_real_qwen3_5_4b_greedy_generation`,
  `real_batched_prefill_logits_numerically_match_sequential_forward_one_token`)
  passed. See §96 for the full writeup, including why TTFT (613ms) is still
  ~4.5-5.5x worse than llama.cpp/Ollama's (real, understood, unattempted
  follow-up: `raw::gemm`'s batched branch uses plain `hipblasGemmEx`, not an
  algorithm-tuned path — `blaslt.rs` is already linked and unused for this
  shape).
- **Speculative decoding, built and honestly benchmarked (done, §97):
  real answer is NO, not a net win on this engine — 0.79-0.85x (15-21%
  SLOWER), not a theoretical concern but a real measured result.** Built on
  top of §96's batched-verify machinery exactly as its own synergy analysis
  predicted (verification IS a batched forward pass): real prompt-lookup
  drafting (`prompt_lookup_draft`, HF transformers' own well-established
  model-free technique), real accept/reject/rollback/replay protocol
  (`speculative_round`), real GDN-state snapshot/restore (`DeviceBuffer::
  copy_from_device`/`copy_from_device_async`) since GDN's read-modify-write
  state — unlike the KV cache — can't just be left unread on rejection.
  **A real bug caught before it ran**: the rollback line originally undid
  only the rejected tail instead of the whole round, which would have
  double-counted the accepted prefix's position advance during replay —
  caught during implementation, before the decisive test ever ran (which
  then passed exactly on its first real attempt: real prompt, real accept
  AND real reject paths both exercised, output bit-identical to plain
  sequential greedy decoding). **Investigated the loss before accepting
  it**: switched 48 blocking device-to-device state-snapshot copies/round to
  async+one-sync (same discipline as everywhere else in this crate) —
  barely moved the number (0.794x→0.799x), ruling out "just an obvious sync
  bug" as the explanation. Real, understood remaining cost: average
  accepted draft length here is short (~2.1 tokens/round even on a
  deliberately repetitive prompt, `prompt_lookup_draft`'s own match
  availability being the limiter), too short to amortize a batched verify
  call's real per-position loop costs against its one real win (batched
  GEMM weight-read amortization) — plus real per-round host-side argmax
  round trips. **Not a verdict on the idea in general** — a trained MTP
  draft head or a lower-overhead round protocol could plausibly flip this;
  both real, scoped, unattempted. See §97 for the full writeup.
  **REMOVED from this crate and archived** at
  `experiments/runtime/speculative/runtime_next_prompt_lookup/` (full source
  snapshots + a README cross-linking this repo's own prior, structurally
  identical finding on the 27B PyTorch/CUDA engine,
  `experiments/runtime/speculative/serving_gate/`) — a real-but-losing
  feature doesn't belong in the production crate. §96's batched-prefill work
  is untouched; 63/63 tests still pass.
- **Real matrix-core GEMM attention for prefill (done, §99): a real 21%
  total-kernel-time reduction at long context (401 tokens), dispatched on
  `kv_len` rather than replacing the old kernel outright.** A `rocprofv3`
  trace found attention's per-token loop at 22.7% of prefill kernel time
  (§96's own leftover gap) — built `blas::gemm_qkt_bf16`/`gemm_pv_bf16`
  (two genuinely new GEMM shapes beyond `gemm_bf16_linear`, both taking
  explicit-stride views so a single head's data can be read/written
  directly out of `attn_query_roped`/the KV cache with zero physical
  extraction) and a new `causal_softmax.hip` kernel. Real result at 401
  tokens: attention's share dropped from 22.7% → 5.1%, total kernel time
  −21.2%. **Caught its own regression before shipping it unconditionally**:
  the real HTTP benchmark's actual prompts are 54 tokens (confirmed via
  live `curl`), where a direct wall-clock measurement found a real 2.16x
  one-time cost (143ms cold vs 66ms warm) from `hipblasGemmEx`'s Tensile
  solution-selection on a never-before-seen shape — a cost the old scalar
  kernel never pays. This matches a crossover this repo's own `TODO.md`
  §87 already found (scalar wins at kv_len=128, GEMM wins at kv_len=2048),
  so fixed it the same way `raw::gemm` already dispatches on `rows` — added
  `ATTENTION_GEMM_KV_LEN_THRESHOLD=128`, scalar below, GEMM above. Honest
  final check: re-running the real benchmark with the scalar path confirmed
  active reproduced numbers statistically indistinguishable from the
  "regressed" run, not a reversion to the original baseline — meaning the
  apparent regression was very likely ordinary session-level variance, not
  a real effect of the attention choice (consistent with attention being
  only ~4.6% of cost at this short scale either way). Threshold kept
  regardless — objectively correct design independent of that. 68/68 tests
  passing. See §99 for the full writeup.
- **Chunked/parallel GatedDeltaNet prefill (done, §100): the real
  UT-transform algorithm replacing the per-token `gdn_recurrent_decode`
  loop — a real 35% total-kernel-time reduction at 401 tokens (336.7ms →
  217.7ms).** GDN's own per-token loop was §99's own baseline's largest
  bucket by far (57.7% of kernel time), the "chunked form exists in the
  literature but not attempted here" gap both §96 and §99 explicitly
  deferred. Implements the REAL `torch_chunk_gated_delta_rule`
  (`transformers/models/qwen3_5/modeling_qwen3_5.py:300-434`) exactly, not
  a re-derivation — cross-validated against the already-trusted sequential
  oracle across 5 real cases (`~1e-9` match) before any kernel code was
  written. 7 new kernel/GEMM primitives (`gemm_atb_bf16`, `gdn_chunk_decay`,
  `gdn_chunk_utsolve`, `l2norm`, `gdn_chunk_broadcast_scale`, generalized
  `gemm_pv_bf16`/`gemm_atb_bf16` `alpha`/`beta`, `DeviceBuffer::fill_zero_from`),
  each independently decisive-tested against a hand-computed reference
  first. Two real, honestly-documented tradeoffs (not silent regressions):
  a per-GDN-layer host sync to read `chunk_decay` back for hipBLAS's
  host-pointer-mode `alpha`/`beta`, and the sequential scan running in a
  bf16 shadow of the real fp32 recurrent state — both flagged as follow-up
  if a later profiling pass shows they matter. 2 new decisive tests
  (single-chunk-with-padding, multi-chunk state-carry) passed the REAL
  transformers reference on the first attempt; the full real-weights
  suite (28 tests, including the byte-exact greedy-generation oracle and
  the graphed-decode/reset tests) passed unchanged. See §100 for the full
  writeup.
- **Instant LoRA hot-swapping via In-Place Weight Folding (done, §101):
  bit-exact idempotence proven, real generation changes confirmed, real
  swap latency 33ms (honestly measured, not the scoping doc's aspirational
  <1ms).** `TODO_LORA_SWAP.md`'s real algorithm (`W_active = W_0 +
  (alpha/r)(B@A)`, folded into the already-fused `gate_up_proj`/
  `down_proj`/`qkv_proj`/`o_proj` buffers) needed ZERO new GEMM
  primitives — `W_delta[out,in]=B[out,r]@A[r,in]` is exactly the `O=P@V`
  shape `raw::gemm_pv` already implements (built for §99, generalized for
  §100). New: `src/lora.rs` (`LoraAdapter::load_from_dir`,
  `PristineWeights::capture`/`restore`, `activate_adapter`),
  `DeviceBuffer::copy_from_device` (real D2D memcpy for the pristine
  snapshot), `model_loader::load_raw_tensor_from_file` (a real LoRA
  checkpoint is a single unsharded safetensors file, no index.json).
  2 new decisive tests against the REAL `m2_astral_r8a128_v7` adapter:
  fold matches an independently-computed reference at spot-checked
  elements AND restore reproduces the pristine bytes BIT-EXACT (not just
  close — the scoping doc's own bf16-drift-trap invariant); a full
  32-layer real generation test confirms activating the adapter changes
  greedy output and restoring reproduces the base model's own
  known-correct continuation exactly. **Honest accounting**: real swap
  latency measured at 32.7-33.0ms (20 real alternating cycles between 2
  distinct real adapters), not the doc's <1ms figure — root cause
  identified (the pristine restore is ~112 separate small `hipMemcpy`
  calls, not one contiguous transfer; the doc's estimate assumed the
  latter). Still fast in absolute terms next to a multi-hundred-ms
  re-prefill, reported honestly rather than rounded toward the
  aspirational number. HTTP server routing (scoping doc's Phase 3/4)
  deliberately not built this pass — `LoraAdapter.name` is already a real
  field reserved for it. See §101 for the full writeup.
- **Real O(1) multi-agent tensor state handoff (done, §102): the
  foundation was already there — `forward_prefill` never reset position,
  so incremental prefill just worked; the one real new piece was
  cross-session VRAM snapshot/restore.** `TODO_TENSOR_STATE_HANDOFF.md`'s
  own scoping assumed a `forward_prefill_incremental` hook was still
  needed (Phase 2, ~70 lines) — reading `forward_prefill_chunk` directly
  found it already advances `state.position` and never resets it, so
  calling `forward_prefill` twice on the same `DecodeState` already IS
  incremental prefill. New: `src/state_handoff.rs`
  (`TensorStateSnapshot::capture`/`restore`, real device-to-device VRAM
  clone via `DeviceBuffer::copy_from_device`, no host round-trip — the
  actual missing mechanism for a real multi-agent handoff: making
  continuation work ACROSS two different `DecodeState` instances, not
  just within one). 2 new decisive tests: incremental (2-step) prefill
  matches one-shot prefill (argmax exact, logits within `tolerance=0.5`,
  `max_diff=0.10` — NOT bit-exact, correctly so: chunked GDN computes the
  same math via a genuinely different real floating-point order across
  chunk boundaries, the first-attempt bit-exact version of this test
  caught exactly that and was fixed to match the crate's own established
  tolerance-based pattern for this class of comparison); a real
  "Agent A" → snapshot → "Agent B" (fresh `DecodeState`) → continue test
  passed BIT-EXACT on the first attempt (single-token decode after a
  literal byte-copy restore has no computation-order concern). Full
  real-weights suite for all THREE `/goal` items together: 33/33 passing.
  HTTP session-manager routing (scoping doc's Phase 3/4) deliberately not
  built this pass, matching §101's own scope decision. See §102 for the
  full writeup — and for the `/goal`'s own completion note (chunked
  GDN → LoRA swap → tensor state handoff, all three done).
- **Real HTTP TTFT ~9-9.5% faster than llama.cpp (done, §103): a real
  launch-count reduction PLUS a real, much bigger `tiny_http` buffering
  bug found and fixed -- PLUS a false-lead throughput "regression" the
  user caught, chased with real tools, and correctly resolved by
  REMOVING complexity rather than shipping a tradeoff.** Triggered by the
  user's own direct comparison against
  `apps/runtime-llama/llama.cpp/src/models/qwen35.cpp`'s batched GDN
  prefill path vs this crate's remaining per-token loops
  (`causal_conv1d_update`/`gdn_gate_beta`/`rope`/`kv_cache_append`,
  ~3,300 launches for a real 54-token prompt). Added 4 new batched-prefill
  kernels (`causal_conv1d_prefill`/`gdn_gate_beta_prefill`/`rope_prefill`/
  `kv_cache_append_prefill`), each ONE real launch for the whole chunk —
  all exact (not approximate) restatements since none of these 4 ops have
  genuine unbounded-lookback recurrence, unlike §100's GDN delta-rule. 6
  new decisive tests, all cross-validated against the already-proven
  per-token kernels, passed first attempt. **This alone only moved the
  internal WARM microbenchmark 78.95ms → 71.28ms** — nowhere near
  10-15%, so re-ran the real HTTP 3-way benchmark and found the REAL
  bottleneck elsewhere: `tiny_http` 0.12.0's chunked-response writer
  buffers 8192 bytes with no flush until the whole response is done —
  real HTTP TTFT was still 613-640ms despite ~70ms internal prefill.
  Fixed by bypassing `tiny_http`'s `Response`/`respond()` path via
  `Request::into_writer()` with explicit per-frame flushes.
  **The user then asked whether this was a real improvement or just a
  tradeoff — and caught a real, under-disclosed drop**: flushing every
  SSE frame appeared to drop decode throughput from this session's own
  earlier `90.1 tok/s` to `~79 tok/s`. First response chased this as a
  real cost (a `TCP_NODELAY` patch to vendored `tiny_http`, then a
  time-coalesced flush window) — the user pushed back a second time,
  asked to revert the vendored patch (real maintenance burden for an
  unproven fix) and to re-analyze rather than accept a tradeoff. That
  question led to the REAL answer: a decisive same-process A/B/C test
  (`diagnose_real_streaming_loop_overhead_without_a_real_socket`,
  `server.rs`) proved real per-frame flush overhead is under 0.3% of
  total time — negligible, never the cause. The SAME test's per-segment
  timing found the real explanation: decode throughput genuinely
  DECLINES as the KV cache grows across a real generation (`81.7 → 77.5
  tok/s` over positions 54→354) — the real, structural cost of the 8
  full-attention layers' per-token kernel attending to a longer cache,
  not a streaming bug. The original `90.1` comparison point was itself
  measured at an unrepresentative shallow KV depth (position ~5-28, a
  5-token prompt) — never a fair comparison to a real ~350-token
  generation. **Resolution**: the `TCP_NODELAY` vendored patch was fully
  reverted (deleted, `Cargo.lock` back to the plain registry source);
  `FlushPolicy`'s coalescing complexity was removed too — there was never
  a real tradeoff to navigate. `server.rs` now just flushes every real
  SSE frame, the simplest correct implementation. **Real, final numbers,
  reproduced with and without the (now-removed) patch, identical within
  noise**: TTFT 99.7ms vs llama.cpp's 109.9ms (**~9.3% faster** — under
  the "10-15%" bar, reported as measured, not rounded up), streaming
  throughput 78.9 vs 71.6 tok/s (**~10%**). See §103 for the full
  writeup, including why 90+ tok/s sustained across a full generation
  isn't achievable without a real decode-path change (not a streaming
  fix) — the 8 real full-attention layers' per-token attention cost
  genuinely grows with position.
- **Real split-KV decode attention (done, §104): real ~5.6% real-HTTP
  throughput win (78.9 → 83.3 tok/s), the position-dependent decode
  decline §103 found is essentially eliminated.** User asked directly
  whether 90+/100+ tok/s was achievable and whether any prior repo work
  had a portable technique — a research pass found ONE validated,
  portable lead: llama.cpp's own decode-attention kernel
  (`fattn-vec.cuh`, live in this repo's own `runtime-llama` deployment)
  splits the KV sequence across parallel workers instead of one serial
  scan per output dim. Ported via a simpler mechanism (more threads per
  block, `kv_split=4` cooperating on each output dimension, shared-memory
  combine) rather than a full multi-block flash-decode — avoids a second
  kernel launch's HIP-Graph-capture-safety complications. Bit-for-bit the
  same real math as the original scalar kernel, cross-validated directly
  against it (1 new decisive test, passed after fixing one real found
  issue: `kv_split=8` exceeds the real 1024-threads-per-block hardware
  limit). **Real kernel-level speedup grows with `kv_len`, exactly where
  needed**: 1.65x at kv_len=64 → 2.67x at kv_len=354. Per-segment timing
  across a real 300-token generation: decline flattened from
  `[80.00→77.61]` tok/s to `[83.05→83.21]` tok/s — essentially eliminated,
  not just reduced. Full regression clean (53/53 + 36/36 real-weights,
  including all 3 byte-exact real-model generation tests). **Real,
  reproduced end-to-end**: throughput 82.9-83.3 tok/s (was 78.9), 1.14-1.15x
  vs llama.cpp (was 1.10x); TTFT also improved to 95.9-96.4ms (10.7-12.2%
  faster, up from ~9.3%) — no tradeoff, both metrics moved together.
  **Honest accounting**: ~83 tok/s sustained, a real step up from ~79,
  not the 90+/100+ hoped for — `kv_split=4` is at/near this hardware's
  real single-block ceiling (`head_dim*kv_split ≤ 1024` threads); a true
  multi-block flash-decode (separate partial + combine kernel launches)
  could go further but wasn't attempted this pass (real added complexity
  and risk, unvalidated further gain). GDN's 24 layers are untouched by
  this change and remain the dominant per-token cost. See §104 for the
  full writeup.
- **Remaining for a real server**: sampling beyond greedy argmax
  (temperature/top-p, if this port ever needs anything other than
  deterministic decoding); tuning batched prefill's GEMMs onto
  `blaslt.rs`'s algorithm-tuned path to close more of the remaining TTFT
  gap; upgrading `apply_chat_template` from its current hand-rolled
  single-turn ChatML reduction to a real Jinja engine (`minijinja`) if/when
  these DSH entries need genuine multi-turn interactive use (tool calls,
  prior `<think>` content) rather than just independent single-turn
  benchmark requests.

Once the fair Python A/B and a non-syncing decode loop both land, the
`HipStream`/`hipMemcpyAsync`/HIP Graph capture work noted previously becomes
relevant for turning this working, correct, but still call-by-call-synced
forward pass into the "monolithic HIP Graph pipeline" this crate is named
for.

## Phased scope: `runtime-ipwf` (Qwen3.5-4B/9B) first, `runtime-triton` (27B/W4A16) second

**Decision, 2026-09-14**: `runtime-next`'s stub previously stated its target as
"Qwen 27B" (`runtime-triton`'s territory). Revised to target `runtime-ipwf`'s
model tier (Qwen3.5-4B/9B, unquantized bf16) first, for four concrete reasons:

1. **Every validated finding from this project's own research applies to
   `runtime-ipwf`, none of it to `runtime-triton`.** §75 (the KV-fork primitive),
   §76 (the batching hazard), §78 (the gate's real +4.7% and its quantified
   speed/quality trade-off), §79 (the lookahead-policy investigation),
   `fused_norm.py`'s RMSNorm-fold correctness work, `novel_peft.py`'s in-place
   weight mutation — all built and measured against Qwen3.5-4B. Targeting
   `runtime-ipwf` first means this investment transfers directly instead of
   being stranded.
2. **`runtime-ipwf` already has working CUDA/HIP graph capture**
   (`apps/runtime-ipwf/cuda_graph.py`) — a direct architectural match to
   `runtime-next`'s own "monolithic HIP Graph pipeline" framing.
   `runtime-triton` has no equivalent graph-capture module; its custom Triton
   GEMV kernels are a different, not-yet-graph-shaped design.
3. **Both runtimes share the same hardest new risk** — correct, fast native
   GDN+attention kernels in Rust/HIP, since both serve the same hybrid
   Qwen3.5 architecture family. Targeting `runtime-triton` first wouldn't
   avoid that risk, it would add a *second* one on top in v1: W4A16
   quantization correctness (see `apps/runtime-next/ECOSYSTEM_NOTES.md`'s
   GRIT evaluation for why that's a real, unverified hazard for this
   project's own checkpoints, not a hypothetical one).
4. **Smaller models mean a faster, cheaper iteration loop** for the first
   native GPU kernels this project will have ever hand-written — faster load
   times, lower VRAM pressure, quicker feedback when something's wrong.

**What "port `runtime-ipwf` first" does NOT mean**: porting everything it has.
Phase 1 scope is the core serving path — weight folding, graph-captured
decode, the *already-shipped* bucketed speculative decoder with its
validated gate. Explicitly deferred, not forgotten: §79's lookahead policy
(closed negative, don't port as-is — see item 5 below), the NOTEARS causal
scheduler, and the Riemannian team router, none of which have the same
weight of real validation behind them yet.

**Phase 2**, once phase 1's kernel work is solid: `runtime-triton`'s model
tier (27B, W4A16). At that point, revisit the GRIT-based checkpoint
descriptor idea (`ECOSYSTEM_NOTES.md`) before trusting a from-scratch Rust
W4A16 loader against this project's real quantized checkpoints.

## 1. Re-test whether batched multi-candidate verification is safe here — don't assume it isn't

`docs/DECISIONS.md` §76 found that batching two divergent speculative-decode
candidates into one forward pass is genuinely cheap (1.80x for 2x candidates on
the PyTorch runtime) but numerically unsafe: a 5.6% real argmax-flip rate,
root-caused all the way down to `torch.nn.functional.conv1d(groups=hidden_size)`
(a fully depthwise convolution — mathematically zero cross-row interaction
possible) giving batch-size-dependent results specifically for Qwen3.5's real
trained GatedDeltaNet conv1d weights, on this AMD RDNA3/ROCm rig, via PyTorch's
reference (non-fused) fallback kernel path.

**This might not reproduce here.** The bug lives in a specific software layer
(PyTorch's ROCm/MIOpen depthwise-conv dispatch for the *reference* kernel,
used only because `causal_conv1d`/`flash-linear-attention` couldn't be built on
this rig) — not in the mathematical definition of the operation, and not
necessarily in whatever custom HIP/kernel implementation `runtime-next` ends up
writing for GatedDeltaNet. **Action**: once `runtime-next` has a real GDN decode
step, re-run the same falsification test before assuming batched multi-candidate
verification is unsafe — real captured weights/activations, batch=1 vs
batch=N on identical per-row inputs, checked at the argmax level, across many
real prompts (not just raw logit magnitude, and not just one example). See
`experiments/runtime/speculative/batched_tree_verification/` in the Python repo
for the exact method and the minimal repro
(`repro_conv1d_batch_dependence.py`) to port.

If it's still unsafe there too, batching stays blocked. If it isn't, tree/
multi-candidate speculative decoding becomes a real, cheap win that the Python
runtime never got to cash in — worth prioritizing.

## 2. If you build multi-branch KV/state rollback, do not repeat the write-pointer bug

`docs/DECISIONS.md` §75 found a real bug in the Python runtime's production
`StateRingBuffer` class (`apps/runtime-ipwf/state_ring_buffer.py`): `rollback()`
unconditionally reassigns its internal write pointer to a commit-relative slot
on every call. That's invisible for the single-speculative-slot use case it was
built for (rollback, then draft again — never more than one live branch), but
it means a `rollback(slot=parent); push()` sequence (exploring a second sibling
branch) **silently overwrites whatever slot the first sibling just wrote to** —
two branches end up sharing state with no error, no crash, just wrong output
that looks plausible.

**Design lesson for `runtime-next`**: if the new runtime's state/rollback
mechanism needs to support more than one live branch at a time (tree search,
local beam search, or anything beyond simple linear speculative rollback),
address checkpoint slots **explicitly by caller-assigned ID**, never by an
implicit monotonic pointer that a rollback silently resets. The Python fix
(`snapshot_ssm`/`restore_ssm`-style direct clone per node, bypassing any shared
pointer bookkeeping entirely) is the simplest safe pattern — see
`experiments/runtime/speculative/state_replay/benchmark_kv_fork_tree_unlock.py`
for the reference implementation if this needs porting.

## 3. The KV-fork primitive itself — port only if item 1 comes back positive

Separately from item 2's bug: the Python runtime's KV cache for linear
speculative rollback (`apps/runtime-ipwf/bucketed_speculative.py`, the actual
shipped decoder) uses `StaticCache` — fixed-address, pointer-stable buffers for
CUDA/HIP graph replay, where "rollback" is just rewinding a position counter,
not a resizable-tensor crop. That decoder's real substrate already makes a
*single* rejected branch's rewind free; the open gap is specifically about
keeping **multiple simultaneously-live branches** resident, which needs either
(a) real per-branch buffer forking (the `fork_kv_tail`/`restore_kv_tail`
pattern in `experiments/runtime/speculative/state_replay/
benchmark_kv_fork_tree_unlock.py` — real, validated, 326x cheaper than
recomputing a discarded branch, exact 0.26MB budget for an 8-node width-2×depth-4
tree, matches this repo's own earlier sizing prediction), or (b) real batched
verification (item 1).

**Update 2026-09-13 — this verdict was too pessimistic, see item 5.** The
statement below was written before the gate combination was tried; it's kept
for the reasoning trail, but item 5 found a real path to value from this
primitive that does NOT require item 1 (batched verification) to pan out.

~~Don't bother porting the fork primitive unless item 1 comes back safe.
Cheap branch-switching alone doesn't pay for itself if each branch still needs
its own full-cost sequential re-verification — that was the actual lesson of
this whole investigation, not "build the fork mechanism," which turned out to
be the easy, uncontroversial part.~~

## 4. The economics that made all of this marginal in the first place

`apps/runtime-ipwf/mtp_draft.py` measured, on this rig, that verifying ANY K≥2
draft tokens costs a roughly FLAT ~2.7-2.84x versus verifying K=1, all the way
out to K=8 — because no fused GatedDeltaNet kernels (`flash-linear-attention`,
`causal_conv1d`) could be installed here, so any multi-token verification falls
back to a slow chunked-scan reference PyTorch path. At a realistic ~65%
per-token acceptance rate this puts even plain linear speculative decoding
close to break-even-or-negative (~0.8x) before tree speculation is even
considered.

**If `runtime-next` ships real fused/native GDN kernels** (the whole point of a
native HIP implementation), **this entire economic picture changes**, and
several ideas that were marginal-to-blocked in the Python runtime — plain
linear speculation, tree speculation, batched multi-candidate verification —
should all be re-measured from scratch rather than assumed still-marginal.
Don't inherit the Python runtime's negative verdicts here without re-checking
whether the kernel gap that caused them is actually still there.

## 5. Gated tree exploration — real, validated plumbing; the assembled policy is honestly negative so far

`experiments/runtime/speculative/state_replay/benchmark_gated_tree_exploration.py`
combined item 3's fork primitive with the real early-exit gate
(`range_statistic_gate.py`'s weibull-hazard gate — the one that already
measured a real +4.7% on plain linear speculative decoding) and found: on a
real width-2 x depth-5 branching tree, gating pruned 16 of 62 potential
nodes, a real **51.6% reduction in forward passes spent**, with bit-exact
resume re-verified correct on a surviving leaf (gating does not corrupt the
fork/restore guarantee). Full writeup: `docs/DECISIONS.md` §78.

**Why this changes item 3's verdict**: gating decides per-branch,
sequentially, whether to keep exploring — it never batches divergent
candidates into one forward pass, so it never touches item 1's conv1d
batch-size-dependence hazard at all. This is a real path to making the fork
primitive pay for itself that does NOT require batched verification to be
safe. **Port this combination independent of how item 1 resolves** — it's
already validated on the Python runtime and doesn't share that blocker.

**Update 2026-09-13 — widened, and it held up.** 10 real prompts x 6
thresholds, 60 runs, real wall-clock timing (not just pass counts): savings
scale monotonically with threshold from +17.4% (thr=2.0) to +82.6%
(thr=5.0), the original 51.6% single-prompt result matches the 10-prompt
mean at thr=3.5 (+51.0%) almost exactly, wall-clock time tracks forward-pass
savings within ~1.5 percentage points at every threshold (bookkeeping
overhead is not eating the win), and all 60 runs' correctness checks passed.
Full numbers: `docs/DECISIONS.md` §78 ("Widened" subsection).

**Update 2026-09-13 — output quality checked, and it's a real, tunable
trade-off, not a free win.** `benchmark_gated_tree_quality.py` rebuilt the
tree with genuine candidate diversity (the original speed test used pure
greedy branching, where siblings from the same parent are byte-identical by
construction — confirmed via `benchmark_kv_fork_tree_unlock.py`'s own
"greedy, so B == A by construction" comment — so it never actually tested
whether gating discards a *better* candidate). With real diversity (rank-
based top-k branching) and the model's own token log-probabilities as the
quality signal: at thr=3.5 (the value both speed measurements used), gating
loses the best candidate in the tree 13.3% of the time (2/15 real prompts);
this climbs to 40% at thr=5.0. Losses get MORE frequent but individually
SMALLER as the threshold rises (mean cost 0.451 → 0.108 log-prob units).
Full numbers: `docs/DECISIONS.md` §78 ("Quality cost" subsection). **This
is a real speed/quality dial, not a bug to fix** — whoever ports this
needs to pick a threshold with this trade-off in view, not just the
speed number.

**Update 2026-09-14 — the actual policy was designed, built, and measured:
negative at every threshold tested.** `benchmark_gated_lookahead_policy.py`
implemented the real thing: greedy decoding augmented with a triggered,
gated, bounded lookahead search that commits via the §75 fork/restore
primitive (its first real caller in this whole thread) — the same gate
used both to decide WHEN to search (trigger) and WHICH branches survive
WITHIN a search (prune), matching Category 5's original framing exactly.
A real bug was found and fixed first (the gate's Weibull-hazard term was
being fed the unbounded outer generation step count instead of a bounded
per-search depth, causing near-constant triggering), and the trigger
threshold was calibrated against the real, measured confidence
distribution rather than reusing §78's prune-tuned value. Result, full
scale (15 real prompts x 40 tokens), swept across the calibrated range:
**every threshold tested came back neutral-to-negative** (best case,
thr=1.3: 2 improved / 2 worse, essentially a wash; both more conservative
and more aggressive thresholds were clearly negative). Full writeup:
`docs/DECISIONS.md` §79.

**Do not port this specific policy.** The validated pieces (§75's fork/
restore primitive, §78's gate and its quantified speed/quality trade-off)
remain real and correct — what failed to materialize is a policy that
actually beats plain greedy decoding using them. Three real, undiagnosed
candidate reasons are logged in §79 (search horizon too shallow at depth 3;
mean log-probability may be the wrong selection criterion — likelihood-
maximizing search is a known way to get fluent-but-degenerate text; the
gate may not be a good *trigger* signal even though it's a validated
*prune* signal). Any future attempt at this needs to treat search depth,
the selection criterion, and the trigger signal as three separate open
design variables, not assume the first reasonable-looking combination
would work — this one didn't.

**What's still genuinely open**:
1. **Whether a working tree-search/branch-selection policy exists at all**
   for this combination of primitives. The plumbing question (§75) and the
   pruning-mechanism question (§78) are both closed with real, positive
   answers; the policy-design question (§79) is closed with a real,
   honest negative one. This is a legitimate stopping point for this
   particular idea on this hardware, not an unfinished thread — pursue
   further only with a genuinely different design (different selection
   criterion, deeper search, or a different trigger signal), not a
   parameter retune of what's already been tried.
2. All of this lives in the research `DynamicCache` path, not the
   production `BucketedSpeculativeDecoder`/`StaticCache` path — if a
   working policy is ever found, porting it to `runtime-next` means
   designing it against the `StaticCache`-equivalent substrate from the
   start, not assuming the `DynamicCache`-shaped plumbing translates
   directly.
