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
- **Tokenizer**: real BPE matching this model (`tokenizer.json` already in
  the same cached snapshot) — reuse a mature crate, not a feature to
  benchmark against Python (correctness matters here, not speed). Bypassed
  for §89's correctness test via a fixed, real, pre-tokenized prompt
  (obtained once via the real tokenizer) — still needed for a
  general-purpose server.
- **Remaining for a real server**: sampling beyond greedy argmax
  (temperature/top-p, if this port ever needs anything other than
  deterministic decoding), the OpenAI-compatible HTTP server DSH can plug
  into (DSH currently points at port 8000/`runtime-triton`; pointing it at
  `runtime-next` instead is a one-line `apps/harness/settings.yaml` change
  once there's something real to point it at), and prefill scoped as
  genuinely parallel (today's prefill is just the same single-token decode
  step called once per prompt token, sequentially — correct, but not fast
  for long prompts).

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
