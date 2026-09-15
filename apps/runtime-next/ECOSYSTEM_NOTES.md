# External ecosystem watch — evaluated for relevance to `runtime-next`

Running log of external crates, blog posts, and techniques brought to this
project's attention and evaluated specifically against `runtime-next`'s scope
("Zero-allocation inference & monolithic HIP Graph pipeline" — see
`src/main.rs`). **Phased target, decided 2026-09-14 (see `TODO.md` "Phased
scope"): Qwen3.5-4B/9B — `runtime-ipwf`'s territory — first, 27B/W4A16 —
`runtime-triton`'s territory — second.** Entries below written while the
stub still said "Qwen 27B" remain valid where they discuss quantization/
W4A16 (still real, just phase 2 now, not phase 1) — noted individually
where that distinction matters. Distinct from `TODO.md`, which tracks
*internal* findings carried forward from the Python runtime's own
experiments — this file tracks *external* tech, so nothing gets
re-evaluated from scratch when it resurfaces.

Each entry gets a one-line verdict up front, the reasoning, and — only for
items judged relevant — a pointer to the corresponding `TODO.md` action item
(added there, not duplicated here).

**Two separate questions per item**: whether the *crate/library itself* is
relevant (usually "no" for CPU-side tooling, since `runtime-next`'s compute
surface is GPU/HIP), and whether any *design technique* it demonstrates is
worth adapting independent of the crate. A "not relevant" crate verdict does
not by itself mean there is nothing to learn from it — see "Techniques worth
adapting" below.

## Directly relevant

### grit-datatype v1.1.0 (evaluated 2026-09-13) — a checkable contract for quantized-tensor decode semantics

**Phase 2 relevance note (added 2026-09-14)**: this entry is about the
W4A16/27B path (`runtime-triton`'s territory), which is now phase 2 per
`TODO.md`'s "Phased scope" — phase 1 targets unquantized Qwen3.5-4B/9B
(`runtime-ipwf`), where no quantization-decode-contract risk exists. Action
item 1 below (running `grit scan` against the existing checkpoints) is
still cheap and worth doing whenever convenient; action item 2 (wiring a
real descriptor into the export/load boundary) should wait for phase 2.

**What it is**: a 64-byte, fixed-layout descriptor (`Grade, Placement, Planes,
Shape`) that pins every decode decision a quantized tensor format normally
leaves implicit in tool source code — element format, group/block size,
scale format, zero-point convention, byte layout — verifiable in O(1) at a
producer/consumer boundary. Ships a read-only `grit scan` auditor that
applies the same checks to files that carry no descriptor at all (GGUF,
safetensors), by inferring the contract and flagging disagreements. Five
zero-dependency implementations (C, C++, Rust, Python, TypeScript) agree
byte-for-byte across 68 conformance vectors and a 96/96 cross-language
fingerprint check. Real field study on 4 popular checkpoints found: a GGUF
whose declared file type disagreed with its actual tensor bytes, and two
4-bit zero-point conventions (`zpc=asis` vs `zpc=minus1`) that are
byte-identical at the container level while decoding to different numbers.

**Verdict: relevant, both the crate and the underlying idea — unlike every
other item logged in this file so far.** This is not a CPU-tooling curiosity;
it targets exactly the risk surface `runtime-next` is about to create.
`w4a16_loader.py`/`triton_w4a16.py` are already vendored *twice*
independently (`apps/runtime` and `apps/runtime-triton`) — group-wise 4-bit
weight quantization (`group_size=128`, packed `int32` words, per-group
`bfloat16` scales; confirmed by reading `apps/runtime-triton/w4a16_loader.py`
directly). Porting this a *third* time, to Rust/HIP, based on reading Python
source and reverse-engineering the checkpoint's byte layout, is precisely
the scenario GRIT's own framing describes: "when a quantized checkpoint
moves between two tools, what promises both sides decode the same numbers?
Nothing." This project has already hit the exact FAILURE SHAPE GRIT targets
— silent, plausible-looking-but-wrong numeric results from an undeclared
convention mismatch — with the real §76 conv1d batch-dependence bug this
session found. A quantization zero-point or group-layout mismatch between
the Python exporter and a from-scratch Rust loader would be the same failure
shape again: no crash, a fluent-but-degraded model, discovered only by
accident or a very lucky benchmark.

**Two concrete, separable action items**:
1. **Cheap, immediate, no port required**: run `grit scan` (`pip install
   grit-datatype`) against this project's actual existing quantized
   checkpoint files (the 27B W4A16 weights) *before* the Rust port begins.
   Read-only, no code changes, ~seconds. If it flags a `file_type_mismatch`
   or `zp_convention_ambiguous`-class finding on files this project already
   depends on, that is worth knowing now rather than discovering it as an
   unexplained quality regression mid-port.
2. **For the port itself**: consider having the Python export/quantization
   side (`apps/factory`'s quantization scripts, or wherever the W4A16
   checkpoints are produced) write a real 64-byte GRIT descriptor alongside
   the checkpoint, and have `runtime-next`'s loader validate against it at
   startup (the `grit-datatype` Rust crate is `no_std`-friendly, zero
   dependencies, `cargo add grit-datatype`) — turning "the Rust loader
   assumes the same zero-point/layout convention the Python writer used"
   from an assumption into a checked, O(1), load-time gate.

**Honest caveat, from the project's own limitations section**: v1's grade
grammar cannot express NF4's codebook or GPTQ act-order permutations. This
project's own scheme (plain group-wise INT4 + per-group scale, no codebook,
no act-order permutation visible in `w4a16_loader.py`) looks like a
plausible fit for v1's grammar, but this has not been verified by actually
running the tool against the real checkpoint — that's exactly action item 1
above, and it should come before committing to action item 2.

**Status**: not started. Action item 1 (run the scanner) is cheap enough to
do at any time and costs nothing if it finds nothing; action item 2 (wire a
real descriptor into the export/load boundary) is a real design decision for
whenever `runtime-next`'s W4A16 loader is actually written, not before.

### "GPU Offload in Rust: Portable, Safe, and Fast" (Drehwald et al., arXiv:2608.13759, evaluated 2026-09-13)

**What it is**: a real research paper (LLNL + University of Toronto),
building safe, cross-vendor GPU offload natively into `rustc` and LLVM —
not a third-party crate, an in-progress upstream compiler extension.
Targets `nvptx64-nvidia-cuda` *and* `amdgcn-amd-amdhsa` (Intel/Apple
planned as their LLVM backends mature). Uses Rust's existing ownership/
mutability annotations (`&T` vs `&mut T`) to automatically derive GPU
data-transfer direction — no separate `#pragma omp target map(...)`-style
annotations needed, unlike OpenMP/SYCL. Evaluated on real hardware,
including a real AMD MI250X, against hand-written CUDA/HIP/RAJA baselines
on the RAJAPerf suite: kernel times land within roughly 30-45% of native
in either direction (kernel-dependent, sometimes faster), a real, honestly
volatile result, not a clean win.

**Verdict: the most directly on-topic material evaluated in this file so
far — genuinely a paper about safe Rust + AMD GPU compute — but NOT
adoptable today.** This is a research prototype: it requires a patched
`rustc` and a specific pinned LLVM revision (23.1.0-rc1) behind unstable
`-Z offload=...` nightly flags, not something `cargo add`-able. Unlike
GRIT (a real, versioned, installable crate) or renew-engine (a real,
clonable, working project), there is nothing here to depend on yet. Its
value is in the design lessons and the real measurement, not the tool
itself.

**Two real, quantified findings that should directly inform `runtime-next`'s
design, independent of ever using this toolchain**:

1. **The "convenient/automatic" interface measured over 400x slower than
   explicit data movement**, on their own benchmarks, because implicit
   host-device synchronization gets triggered by seemingly unrelated
   host-side reads (their own example: a debug `println!` between two
   kernel calls silently forces a round-trip). This is a real, quantified
   version of a risk already flagged in this file from fearless_simd/
   renew-engine's "explicit over implicit" pattern — now with a number
   attached. Directly actionable: whatever `runtime-next`'s own buffer/FFI
   design ends up being, the *hot serving path* must use explicit,
   type-enforced data movement (their Interface C shape, and the
   `Preload`/`PreloadMut` pattern now cross-referenced into the type-state-
   gating entry above) — an "automatic, sync-on-use" convenience API can
   exist for tooling/debugging, but must never be reachable from the
   per-token decode loop.
2. **Data-transfer direction can be derived for free from ordinary
   mutability annotations** (`&T` → host-to-device only, `&mut T` →
   bidirectional) — the type information a `DeviceBuffer<T>`-style API
   would already carry for ordinary Rust reasons tells you everything
   needed to decide transfer direction, with no extra annotation burden.
   Worth designing `runtime-next`'s own buffer types so this falls out
   naturally, the same way this paper's compiler pass does it automatically
   from MIR.

**Also worth knowing, stated honestly**: even this paper's own authors hit
real cross-vendor ABI mismatches on something as basic as a slice type
(x86_64/amdgcn lower it as two scalars; nvptx64 lowers it as a fixed
array) — a sign that safe cross-vendor Rust GPU tooling is still immature
at the primitive-type level. Since `runtime-next` targets AMD only, this
specific mismatch doesn't apply directly, but it's a useful calibration
on how early-stage this whole ecosystem still is. Also: their measured
Rust kernels used slightly higher GPU register counts than the CUDA/HIP
baselines (33 vs. 28 average), attributed to bounds-checking from
explicit thread/block-index array access — a small, real, non-zero safety
tax to expect, not a reason not to pursue this, but worth not promising
zero-overhead safety unconditionally either.

**Why that register-count gap is worth taking seriously, not filing away as
a cosmetic number** (cross-referenced 2026-09-14, from a CPU-SIMD article
otherwise unrelated to this project — see "Improving std::simd::swizzle_dyn"
under "Not relevant" below): that article's own methodology section makes
the mechanism explicit for the CPU case — a "faster" individual operation
that uses more registers can force spills to the stack and net out *slower*
for the whole algorithm, which is why it insists on measuring a full
real algorithm, not the isolated operation, before trusting a win (it
actually builds a base64 codec to check). The GPU analog is occupancy: more
registers per thread means fewer concurrently-resident warps/wavefronts,
which can throttle a kernel's real throughput independent of the kernel's
own instruction count. `runtime-next` should apply the same discipline this
paper's own RAJAPerf benchmark already started (kernel time measured, not
just correctness) if it ever hand-writes kernels: watch register usage and
real occupancy, not just whether the generated code type-checks and passes
correctness tests, before trusting that safe Rust kernel code is
performance-neutral.

**Status**: not adoptable now. Worth a periodic re-check (this is a fast-
moving, actively-published research area) for whether this work — or
`cuda-oxide`, the NVIDIA-led project with a similar safety design the paper
itself compares against — reaches stable, `cargo`-installable form before
`runtime-next`'s FFI/kernel layer is actually written. If it does, it could
replace a hand-rolled unsafe-FFI-wrapper design entirely rather than just
inform it. Tracked here so this doesn't need re-discovering from scratch.

## Techniques worth adapting (independent of crate relevance)

### An enforced, measured "zero-allocation" gate, not just a docstring claim (via renew-engine/renew)

**What it does**: renew (an early-stage, AI-first Rust game engine — Vulkan
rendering, bit-deterministic fixed-timestep simulation, 29 modular crates) has
a `memory` crate providing "arenas, pools, a counting allocator," and states
outright: "the steady-state frame loop is held to zero heap allocations
through the engine's allocators, counted in development builds." Not
aspirational — measured, every build.

**Why it's worth adapting here**: `runtime-next`'s own docstring already
*claims* "zero-allocation inference," but nothing in the stub enforces or
even measures that claim today. Renew's pattern is directly portable: wrap
the global allocator (or the specific arenas/pools used for KV cache and
activation buffers) in a counting layer active in dev/CI builds, and assert
zero allocations across the steady-state per-token decode loop specifically
— the loop that runs once the HIP graph is captured and buffers are
pre-sized (`StaticCache`-equivalent, fixed-address, no resize — the Python
runtime's `BucketedSpeculativeDecoder` already established this is the
right shape, see `TODO.md` item 3). This turns "zero-allocation" from a
docstring assertion into a CI gate that fails the instant a future change
reintroduces a hidden `Vec::push` or similar in the hot path.

**Status**: not started — no allocator instrumentation exists yet, and
there's no hot loop to measure until real HIP integration begins. Worth
deciding on *which* allocator/arena boundary to instrument before writing
the first real buffer-management code, since retrofitting it later is much
more work than designing for it from the start.

### A determinism CLI gate, retargeted at this project's actual known hazard (via renew-engine/renew)

**What it does**: renew ships a `determinism` subcommand that runs a pinned
set of simulations and emits a JSON document of digests; CI runs it on five
targets (Windows/Linux/macOS desktop, Android emulator, iOS simulator) and
diffs them against each other — deliberately *not* against a value committed
to the repo, because "comparing against a value committed in the repository
would prove only that one machine agrees with its own past; comparing
targets against each other is what actually establishes the claim." Full
bit-determinism is achievable there because the simulation uses fixed-point
(Q47.16) arithmetic specifically to avoid float/instruction-selection
variance.

**Why it's worth adapting here, with a real target already in hand**: full
bit-determinism is the wrong goal for `runtime-next` — bf16/fp16 GPU tensor
math is the whole point, and fixed-point arithmetic isn't a real option for
transformer inference. But the renew pattern — *pin a set of real workloads,
run them under varying conditions, diff structured output digests across
runs instead of trusting a single run* — is exactly the falsification test
this project already needed and didn't have, for a bug this session found
the hard way: `docs/DECISIONS.md` §76's real conv1d batch-size-dependence
bug (`F.conv1d(groups=hidden_size)` gives a different per-row numeric result
depending on total batch size, for Qwen3.5's real trained GDN weights, on
this hardware — see `TODO.md` item 1 and
`experiments/runtime/speculative/batched_tree_verification/repro_conv1d_batch_dependence.py`).
That bug was found by ad hoc, one-off investigation; it should have been
caught by a standing gate. The adapted version for `runtime-next`: a
`determinism`-style subcommand that runs a pinned set of real prompts at
varying batch sizes / candidate counts and diffs output-token digests
across them — not for bit-exactness (batch-size invariance for bf16 GPU
math is a *narrower*, specific claim: "the same row's output is the same
regardless of what else is in the batch"), but specifically to catch the
next instance of this exact hazard class before it needs a multi-day manual
bisection to find, the way this one did.

**Status**: worth prioritizing early — this is a gate, not a feature, and
it directly targets a hazard this project has already paid the cost of
discovering manually once. Cheapest to build once real batched-inference
code exists in `runtime-next`; the design (what "digest" means for an LLM
server's output, what workload set to pin) can be scoped now.

### Deny-unsafe-by-default with narrow, documented opt-in crates (via renew-engine/renew)

**What it does**: `unsafe` is denied workspace-wide by default; only three
of renew's crates opt back in (`memory`, `jobs`, `rhi` — precisely the ones
touching raw memory layout, threading primitives, and the GPU API), and CI
requires every remaining `unsafe` block to carry a comment documenting the
invariant that makes it sound. Enforced mechanically, not by convention.

**Why it's worth adapting here**: this is the concrete CI-enforcement
mechanism for the type-state-gating idea already logged above (see the
fearless_simd entry) — that entry describes the *design* (wrap the HIP FFI
boundary in a small number of safe types); this is the *policy* that keeps
it honest over time. Adapted for `runtime-next`: `#![forbid(unsafe_code)]`
at the workspace level, with only the crate(s) directly wrapping HIP/ROCm
FFI calls opting out, and a lint rule (or review checklist item) requiring
every `unsafe` block in those crates to document its invariant inline. This
is cheap to adopt from day one and expensive to retrofit once `unsafe`
blocks are already scattered.

**Status**: directly actionable the moment the crate/workspace layout for
`runtime-next` is decided — this is a `Cargo.toml`/lint-config decision, not
a code-writing one, so it costs nothing to adopt before real code exists.

### Ban panicking shortcuts outside tests (via renew-engine/renew)

**What it does**: `unwrap`, `expect`, and `panic` are denied by lint outside
test code; `todo!`, `unimplemented!`, and `dbg!` are denied everywhere,
including tests.

**Why it's worth adapting here**: a panic inside a monolithic HIP graph
pipeline serving live, possibly-concurrent requests is a much worse failure
mode than a panic in most request-handling code — depending on how graph
capture/replay state is structured, it could plausibly corrupt or strand
in-flight GPU state for requests that have nothing to do with the one that
triggered it, not just fail the one request. Denying `unwrap`/`expect`/
`panic` outside tests forces every fallible operation on the serving path
(tensor shape mismatches, HIP call failures, malformed requests) to be
handled as a real `Result`, not asserted away. This is a cheap, mechanical
clippy/lint configuration, not an architectural decision.

**Status**: directly actionable whenever CI/lint config is first set up for
`runtime-next` — cheapest to adopt before the first real fallible operation
is written, same reasoning as the unsafe-denial item above.

### Named refusals with a regression corpus, for anything parsing untrusted input (via renew-engine/renew)

**What it does**: renew's `REFUSALS.md` names every way externally-sourced
bytes (files the engine didn't write itself) can be malformed, each with a
dedicated test that provokes it and a recorded corpus replayed on every
merge. Their own framing: "a bar with a recorded exception is a bar, and one
with a silent exception is a slogan" — they explicitly name the two readers
that don't yet meet the bar, rather than letting the gap go undiscovered.

**Why it's worth adapting here — and more so than for renew itself**: a game
engine mostly reads its own asset formats, produced by its own tooling. An
LLM inference server reads untrusted network input constantly — the
OpenAI-compatible request body, on every single request, from any client.
This project's own session history already has real examples of the
underlying hazard class this pattern guards against: silent, wrong-looking-
right behavior on malformed or unexpected input (e.g. `training_db.update_airm_metrics`
silently writing `0.0` for keys that didn't exist, found this session; the
old AITER path's bare `except Exception: pass` swallowing a completely inert
fallback, documented in `fused_norm.py`'s own docstring). A `REFUSALS.md`
for `runtime-next`'s request-parsing surface — one entry per way a request
can be malformed (bad JSON, an out-of-range sampling parameter, a truncated
adapter/checkpoint file, an unsupported model alias), each with its own
test and a replayed corpus — would make "we handle malformed input" a
checkable claim instead of an assumption.

**Status**: not urgent while `runtime-next` has no request-parsing surface
yet, but cheap to start as a living document the moment the first real
`/v1/chat/completions`-equivalent handler is written, rather than backfilling
it after the fact.

### A single CLI binary with schema-versioned `--json` on every subcommand (via renew-engine/renew)

**What it does**: every renew capability (`build`, `test`, `bench`, `run`,
`record`, `replay`, `determinism`, `doctor`, ...) is a subcommand of one
binary, and every one accepts `--json` and answers with a document carrying
an explicit `schema_version`, so tooling has a stable contract even as
human-readable output changes freely. Their framing: "a person, a script,
and an agent drive the engine through exactly the same surface, because
there is only one."

**Why it's worth adapting here**: this project's own workflow — this very
session included — is already agent-driven: scripts are run, their JSON
output is read back and reasoned about, largely without a human reading raw
stdout. `runtime-next` becoming a real serving engine is an opportunity to
make that the *designed* interface instead of an incidental one: one binary,
subcommands for `serve` / `bench` / `determinism` / `doctor`, each emitting
a schema-versioned JSON document. This costs little extra over building the
same capabilities as separate ad hoc scripts, and pays off directly for how
this project already operates.

**Status**: a structural decision for whenever `runtime-next`'s CLI surface
is first designed — cheap to decide early, annoying to retrofit across many
already-separate entry points later.

### (Corroboration, not a new idea) "Performance claims arrive with numbers and the configuration that produced them, or they are not made"

Renew states this as a project rule. It is, word for word, the same
discipline this project already enforces on itself (CANON-stamping every
benchmark artifact, the Zero-Mock Invariant against fabricated measurements
— see this session's own repeated real-vs-synthetic investigations). Logged
here only as independent confirmation that this is recognized good practice
in serious systems projects generally, not a house quirk — no action item,
just don't relax this discipline when `runtime-next`'s own benchmarks start
getting written.

### Type-state capability gating to shrink an unsafe FFI surface (via fearless_simd v0.7)

**The technique**: fearless_simd's core design move is encoding *hardware
capability* (which instruction-set level is available: SSE2/AVX2/NEON/etc.)
as a Rust trait/type, so the compiler — not the programmer — proves which
unsafe intrinsics are reachable from which code path. The result: "orders of
magnitude less unsafe" than hand-rolled alternatives, with almost all of it
centralized into a small, auditable core, while everything built on top is
safe generic code.

**Why it's worth adapting here**: `runtime-next` will need a real unsafe FFI
boundary against HIP/ROCm (`hipMalloc`, `hipGraphLaunch`, stream/event
management, etc.). The naive approach — `unsafe { hip_call(...) }` scattered
across the codebase wherever a HIP function is needed — is exactly the
pattern fearless_simd avoids for CPU intrinsics. The adaptable version:
define a small number of safe wrapper types (e.g. a `DeviceBuffer<T>`, a
`HipStream`, a captured `HipGraph`) that hold the *only* unsafe blocks in the
crate, expose safe methods on top, and let ownership/lifetimes (not manual
discipline) enforce things like "a graph capture can't outlive its stream."
This is a design-time decision worth making before the FFI layer is written,
not a refactor to do after the fact.

**Status**: not started — `runtime-next` is still a stub. Worth deciding on
this pattern before the first real `unsafe extern "C"` HIP binding is written.

**Update 2026-09-13 — a concrete, validated realization of this exact idea
now exists, see "GPU Offload in Rust" below.** That paper's `Preload`/
`PreloadMut` types are what this entry was gesturing at in the abstract: a
type that ties a Rust borrow's lifetime to a GPU-resident copy of the data,
where the type itself (not documentation, not caller discipline) makes it
impossible to read stale host data while the device holds the live copy,
and where `drop` *is* the host-sync point. `runtime-next`'s own `DeviceBuffer<T>`
idea above should copy this shape directly rather than reinvent it.

**Update 2026-09-14 — a second, complementary realization, this time for
*inside* a kernel launch rather than host/device data movement: see
"NVIDIA CUDA Rust" below.** `#[launch_contract]` + a `prepare_*` step that
returns an opaque proof token is the same "make the compiler carry the
safety argument" idea applied to launch CONFIGURATION specifically (grid/
block shape validated against both the kernel's own declared contract and
live device limits, once, before any unsafe launch can occur) — worth
copying this shape too if `runtime-next` ever exposes a kernel-launch API,
alongside `Preload`/`PreloadMut` for data movement.

### Compile-time-monomorphized capability dispatch instead of runtime branching (via fearless_simd v0.7)

**The technique**: fearless_simd detects the best available instruction-set
level once (at startup, via multiversioning), then runs fully monomorphized,
generic-but-concrete code for that level with zero further runtime branching
in the hot path — as opposed to checking "which level am I on?" inside a
loop.

**Why it's worth adapting here**: `runtime-next` may eventually need to run
across more than one AMD GPU architecture (this project's own rig is RDNA3/
gfx1100; MI300-class CDNA hardware behaves differently for several kernels
already documented in `docs/DECISIONS.md`, e.g. the AITER findings in
`fused_norm.py`'s docstring — correct only at large shapes, wrong at this
project's shapes). If/when `runtime-next` supports more than one target
architecture, detect the architecture once at startup and dispatch to a
monomorphized, arch-specific code path (via a trait per architecture, the
same shape as fearless_simd's per-instruction-set traits) rather than
branching on architecture inside any hot loop.

**Status**: not yet applicable — single-architecture (RDNA3) is the only
target today. Revisit if/when multi-architecture support is ever scoped in.

**A real failure mode to avoid when this becomes applicable** (cross-
referenced 2026-09-14, from "Improving std::simd::swizzle_dyn" — see "Not
relevant" below): `std::simd`'s own dispatch for one operation
(`swizzle_dyn`) turned out to be un-multiversionable in practice, because
its `#[cfg(target_feature = ...)]` guards get resolved at standard-library
*build* time, before the calling context (and any runtime CPU-detection
wrapper around it) is known — so "compile once per detected capability,
dispatch at startup" silently degrades into "whatever was baked in when
the standard library itself was built," with no error, just a permanently
suboptimal or crashing binary if run on different hardware than expected.
**The lesson, if `runtime-next` ever adds multi-architecture dispatch**:
confirm that whatever mechanism selects the architecture-specific code path
is actually re-checked/re-selected at the point real dispatch happens
(startup detection wired live to the code path taken), not baked in at an
earlier, disconnected compilation stage — verify this the same
disassemble-and-check way logged elsewhere in this file, don't assume a
`#[cfg]`-based or generic-trait-based scheme automatically does the right
thing just because it compiles.

### An explicit, hand-verified baseline instead of trusting the compiler for it (via fearless_simd v0.7)

**The technique**: fearless_simd used to fall back to "scalar code and hope
autovectorization kicks in" when SSE4.2 wasn't available; v0.7 replaced that
with an explicit, hand-written SSE2 implementation, because relying on the
compiler to autovectorize a baseline path turned out to be unpredictable.

**Why it's worth adapting here**: this project has already hit this exact
failure shape once, just with GPU kernels instead of CPU ones — `fused_norm.py`
was built specifically because a "fused" AITER path silently fell through to
a working-but-wrong or working-but-slow implementation with no visible
signal (`except Exception: pass` swallowing the fallback). The general lesson
generalizes directly: for any `runtime-next` code path with a "fast path /
fallback path" shape (a fused kernel vs. a reference implementation), the
fallback must be an explicit, deliberately-written, independently-verified
implementation — never an assumption that some underlying toolchain will
"do the right thing" silently. This is already this project's practice in
the Python runtime (see `fused_norm.py`'s "NO SILENT FALLBACKS" policy); the
fearless_simd write-up is independent confirmation the same discipline
matters in systems-level Rust too, not just this codebase's own scar tissue.

**Status**: already a stated principle for the Python runtime; carry it
forward explicitly as a design principle for `runtime-next` rather than
re-learning it the same way once real fallback paths exist there.

### Explicit wrap/saturate/cheapest choice on narrowing numeric conversions (via fearless_simd v0.7)

**The technique**: fearless_simd's widening/narrowing conversions require the
caller to explicitly choose the behavior (wrap like `as`, saturate, or "do
the cheapest thing the platform offers, if you're sure values fit") rather
than picking one silent default.

**Why it's worth adapting here**: `runtime-next` will very likely need its
own bit-packing/unpacking code for quantized weights (this project already
vendors `w4a16_loader.py`/`triton_w4a16.py` for the 27B path). Quantization
pack/unpack is exactly the kind of narrowing-conversion code where a silent
wrap-vs-saturate choice can produce a plausible-looking but wrong result —
the same *shape* of hazard as this session's real conv1d batch-dependence
bug (§76: correct-looking code, wrong numeric result, invisible without a
targeted check). When `runtime-next` writes its own quantization pack/unpack
routines, make the wrap/saturate/assume-in-bounds choice an explicit,
named decision at each call site, not an implicit default.

**Status**: not yet applicable — no quantization pack/unpack code exists in
`runtime-next` yet. Revisit when the W4A16 path is ported.

### `Drop`-based deterministic GPU memory release, instead of manual VRAM-cap policing (via flodl v0.8.0)

**The technique**: flodl's own pitch is "tensor memory freed by `Drop` the
instant it leaves scope. No GC, no finalizers, no VRAM budget heuristics.
Five phases of PyTorch memory management replaced by `impl Drop for
Tensor`." Rust's ownership model gives it deterministic, immediate resource
release that a GC'd/refcounted host language cannot guarantee.

**Why it's worth adapting here**: the *reason* this stood out is that this
project's own Python runtime has visible scar tissue from exactly the
problem flodl is describing — `set_hard_vram_cap`, `ensure_gpu_exclusive`,
and the `keep_pristine=True` pattern in `WeightFoldingEngine` all exist
because PyTorch's own memory management doesn't release VRAM predictably
enough for a GPU-exclusive, single-tenant server to just trust it. That's a
whole category of manual bookkeeping code that a Rust rewrite gets to
*delete*, not reimplement, if GPU buffer lifetime is tied to Rust's
ownership/`Drop` from the start (a `DeviceBuffer<T>` whose `Drop` frees the
HIP allocation immediately, KV-cache/activation buffers scoped to the
request or session that owns them). This is a structural argument for *why*
`runtime-next` is worth building at all, not just a nice-to-have API detail
— it directly removes a class of workaround the Python side has needed
repeatedly.

**Status**: a design principle to hold from the start of real buffer-
management code — worth stating explicitly now so the temptation to port
the Python side's manual VRAM-cap logic verbatim doesn't creep in as a
"looks familiar, must be necessary" habit.

### An explainable, provenance-annotated resolved-config command (via flodl v0.8.0)

**The technique**: `fdl config show` prints the fully deep-merged
configuration (base manifest + environment overlay) with a per-field
annotation naming exactly which file and line contributed each resolved
value, before a long job runs.

**Why it's worth adapting here**: this project's Python runtime already has
many env-var-driven behaviors scattered across files (`FLASH_NORM_FOLD`,
`STREAM_ORDER_TIMEOUT_S`, `HF_HUB_OFFLINE`, VRAM caps, and more, seen
throughout this session) with no single place to see the fully resolved
configuration or where each value came from. Pairs naturally with the
renew-engine idea already logged above (a single CLI binary, schema-
versioned `--json` output on every subcommand) — a `runtime-next config
show` subcommand that prints the resolved configuration with per-field
provenance would make "why is this server behaving differently in CI vs.
locally" a one-command question instead of a grep-through-env-vars exercise.

**Status**: cheap to design alongside the CLI-surface decision already
logged from renew-engine; no urgency until `runtime-next` has more than a
trivial handful of configuration knobs.

### Python's per-branch/per-loop overhead as a second, independent reason to re-test tree/gating strategies (via flodl's "Trajectory Thesis")

**The claim, as made**: flodl's design manifesto argues that adaptive-
computation architectures (early exit, variable-depth recursion, conditional
branching, tree search) are structurally penalized by Python — every branch
evaluation, loop iteration, or early-exit check costs "~3-5us overhead plus
a CUDA synchronization point" — and that this has produced a real
*selection bias* in ML research toward architectures that are cheap to
express in Python (fixed-depth feedforward, single-pass attention) and away
from ones that aren't (recurrent attention, adaptive depth, tree search),
independent of whether the latter would work better.

**This is not a new idea in the abstract, but it lands on real, freshly-
measured evidence from this exact project, gathered independently of
reading this document.** `docs/DECISIONS.md` §78 (this session, same day)
combined `state_replay`'s KV-fork primitive with a real early-exit gate and
found a genuine 51.6% forward-pass reduction on a branching tree, with
correctness preserved — proof that gated/branching exploration is a real,
working idea on *this* hardware, using *today's* Python runtime. That
result already exists without needing this argument. What the trajectory-
thesis framing *adds* is a second, independent hypothesis for `TODO.md`
item 5's own open question ("does 51.6% forward-pass savings become the
same 51.6% wall-clock savings?"): part of any gap between the two could be
Python-level orchestration cost specific to *how* the branching decision
gets made and acted on (the `copy.deepcopy` per node, dict-based tree
bookkeeping, Python control flow deciding whether to recurse) — a cost
axis that is separate from, and additive to, the already-documented
missing-fused-kernel gap (`TODO.md` item 4). A native Rust implementation
of the same gated-tree logic would remove this second axis regardless of
whether fused GDN kernels ever materialize for the first.

**Status**: strengthens the case already logged in `TODO.md` items 1, 4,
and 5 — file as an additional reason to actually measure wall-clock (not
just forward-pass count) once a real Rust/HIP gated-tree implementation
exists, per item 5's own stated next step, rather than a new standalone
action.

### Open design question (not a technique to adopt): does a "monolithic HIP Graph pipeline" cohere with adaptive/branching control flow?

**The tension, surfaced by reading the trajectory-thesis argument against
`runtime-next`'s own name**: HIP/CUDA graph capture is a strong fit for
*fixed*, replayed-every-time kernel sequences — which is exactly why it's
cheap (zero per-op CPU dispatch overhead once captured). Adaptive
computation, early-exit gating, and tree branching are, by definition,
*input-dependent* control flow — the whole point of `docs/DECISIONS.md`
§78's real 51.6% saving is that different inputs take different numbers of
steps. These two ideas are in genuine tension: a single captured graph
can't natively vary its own shape per request.

**This is a real open question for `runtime-next`'s design, not a resolved
technique** — logged here because the tension only becomes visible by
holding `runtime-next`'s own stated architecture next to the
adaptive-computation ideas this session's `state_replay`/gating work keeps
finding real value in. Possible directions, none evaluated here: capture
multiple graph variants (e.g. per max-depth or per-gate-outcome) and select
between them at replay time (still zero per-op branching cost, more capture
work and VRAM for the extra graphs); or use whatever HIP's current
support is for graphs with conditional/dynamic nodes (unknown maturity on
ROCm as of this evaluation — not checked); or accept that heavily adaptive
paths run outside graph capture entirely, with the capture boundary drawn
around the parts of the pipeline that *are* fixed-shape.

**Status**: worth resolving explicitly when `runtime-next`'s HIP graph
capture strategy is first designed, precisely because `TODO.md` items 1, 4,
and 5 are all pushing toward branching/adaptive strategies being worth
porting — better to have an answer for how they'd coexist with graph
capture before committing to the capture architecture, not after.

### What was left out, and why

The rest of the trajectory-thesis document (Mixture of Strategies,
hierarchical composition of trained sub-networks, "solved layers become
permanent frozen infrastructure," modular multi-team AI development) is
training-architecture and research-philosophy content — genuinely
interesting, but not engineering-actionable for `runtime-next` specifically,
which serves already-trained models rather than composing or training them.
The document's own framing agrees: it calls these "research problems, not
engineering problems." Where any of it might eventually matter is
`apps/factory` (or a hypothetical `factory-next`) — this project's own
`dynamic_team_router.py`/`cut_set_router.py`/multi-expert-stacking work is
already a distant thematic cousin of "mixture of strategies," for whatever
that's worth — but nothing here clears the bar this file holds items to
(a concrete claim traceable to this project's own code or real evidence).
Not logged as an action item anywhere; noted here only so it isn't silently
re-read and re-considered the same way later.

### Verify an optimization claim by disassembling the emitted code, with a deliberate negative control (via "What Rust's +simd128 Actually Changed in My WebAssembly")

**What it is**: a Rust-to-WebAssembly case study. A build script had
`-C target-feature=+simd128` set and a code comment claiming a "SIMD
pre-scan"; that looked like enough evidence to call the build accelerated.
The author tested it instead of trusting it: built the same source twice
(flag on, flag off, compiler otherwise pinned identical), disassembled both
`.wasm` outputs, and counted SIMD instructions *per named function*, not
just aggregated across the binary. Result: one loop (LTTB) produced
byte-identical output with the flag on or off — a deliberate **negative
control**, included specifically to prove the measurement method would
show a difference if one existed. The other module gained 205 real SIMD
instructions — but none of them were in the specific statistics loop the
comment claimed was vectorized; they landed in unrelated sorting/t-digest/
HyperLogLog helpers elsewhere in the same compiled module. The flag did
something real. It did not do the thing the comment said.

**Verdict: not relevant as a technology (no WebAssembly, no browser/Node
target anywhere in `runtime-next`'s scope), but the methodology is exactly
the discipline this project has already been burned by not applying, more
than once.** This project's own real precedents are the same failure
shape: `fused_norm.py`'s docstring records that `aiter.rmsnorm2d_fwd`
silently returns all zeros at this project's actual model dimensions
(correct only at 32768, which is why AMD's own test suite passes it) — a
"fused, accelerated" kernel that looked adopted and was not doing the
claimed thing, discovered only by direct numerical inspection, not by
trusting that the import succeeded. This very session's benchmark output
repeats the pattern every single run: `causal_conv1d`/`flash-linear-
attention` "falling back to reference PyTorch implementation" warnings —
the fast path being silently unavailable is only visible because
`transformers` chooses to log it; nothing here would catch an equivalent
silent substitution that doesn't log.

**The concrete, reusable method, for whenever `runtime-next` enables a
ROCm compiler flag, links a vendor "fused" library, or claims a custom HIP
kernel replaced a reference one**:
1. **Build/run twice with one thing held fixed** (the optimization
   toggled, everything else pinned — compiler version, flags, input) and
   include a **known-unaffected sibling computation as a negative
   control** in the same comparison, so a null result is informative
   ("the method would have shown a difference") rather than ambiguous
   ("nothing to compare against").
2. **Attribute changes to the specific named function/kernel under test**,
   not an aggregate count across the whole binary or process — an
   aggregate "205 SIMD instructions found" or "GPU kernel time dropped
   X%" can be entirely real and still have nothing to do with the code
   path being evaluated.
3. **Only reach for a hand-written kernel/intrinsic as the last step**,
   after this kind of verification proves the generated/dispatched code is
   genuinely missing an available opportunity — not as the first response
   to a slow benchmark number.

**Status**: no action item now — there is nothing to disassemble yet. This
is a standing verification habit to apply the moment `runtime-next` starts
claiming a ROCm flag, vendor library, or custom kernel changed what
actually executes, matching a discipline this project has already
independently arrived at for the Python runtime (`fused_norm.py`'s "NO
SILENT FALLBACKS" policy, already cross-referenced from the fearless_simd
entry above) but has not yet had to apply to compiled/linked artifacts the
way this case study did.

**Minor corroboration, not a new idea**: the article separately measured
that the JS/WASM call-boundary copy cost was 11.5-18.3% of total wrapped-
call time in one case — a small but real illustration that a call
boundary can dilute or hide a kernel-level change, the same lesson already
logged from "GPU Offload in Rust" (the 400x implicit-sync slowdown) and
now independently confirmed a third time. No new entry; reinforces the
existing one.

## Not relevant

### fearless_simd v0.7 (evaluated 2026-09-13)

**What it is**: A safe, zero-dependency Rust abstraction over *CPU* SIMD
(SSE2/AVX on x86, NEON on ARM) — portable vector types, autovectorization/
multiversioning, generic programming over vector width. Heading to v1.0,
no breaking changes planned. Real, mature (year-stable API, 1000+ dependent
repos), used by Linebender projects (vello/kurbo) for CPU-side 2D rendering.

**Verdict: not relevant to `runtime-next`.** It is a CPU vector-math library
with no GPU/HIP component whatsoever. `runtime-next`'s entire compute-heavy
surface — attention, GEMM, GatedDeltaNet, W4A16 dequantization — is GPU
kernels written against HIP, not CPU vector instructions. The only CPU-side
work in an LLM serving engine at all is tokenization, sampling
post-processing, and request/tensor bookkeeping, none of which are typically
hot enough to justify hand-vectorization, and none of which are currently
identified as a bottleneck anywhere in this project's own benchmarking
(the Python runtime's own profiling has never once pointed at CPU-side
numeric work as a cost center — every real finding this session and prior
sessions traces back to GPU kernel dispatch, VRAM residency, or verification
cost, never CPU-bound preprocessing).

**Would only become relevant if**: `runtime-next` ends up doing real,
measurably-hot per-token CPU-side numeric work outside the GPU kernel path
(e.g. a CPU fallback decode path, or heavy CPU-side logit
post-processing/sampling at very high QPS). No current design calls for
that, and nothing in this project's own measurement history has ever
surfaced a CPU-bound hotspot of the shape fearless_simd would address.
Re-evaluate only if such a hotspot is found, not preemptively.

**But see "Techniques worth adapting" above** — the crate itself doesn't
apply, but four of its design decisions (type-state capability gating over
unsafe FFI, compile-time-monomorphized capability dispatch, explicit
hand-verified fallbacks, explicit wrap/saturate choices on narrowing
conversions) are worth carrying into `runtime-next`'s own design regardless.

### "Improving std::simd::swizzle_dyn" (same author/crate family as fearless_simd, evaluated 2026-09-14)

**What it is**: a deep, hardware-port-level optimization of one CPU SIMD
operation (byte shuffle/swizzle) across AVX2/NEON/AVX-512 — instruction-
level-parallelism tuning verified with `llvm-mca`, a real base64 codec
built specifically to confirm the micro-optimization survives contact with
a full algorithm (it does, ~6x). Real, rigorous, well-executed work.

**Verdict: not relevant, same reason as fearless_simd — this is CPU SIMD,
and`runtime-next`'s compute surface is GPU/HIP.** This is a narrower,
deeper follow-up in the exact same non-applicable domain, not new ground.
Two of its findings were worth cross-referencing into already-logged
entries rather than a new one: the register-pressure-can-silently-negate-
a-"faster"-operation mechanism (added to the "GPU Offload in Rust" entry
above, since that paper's own Rust-vs-CUDA register-count gap now has a
concrete reason to take seriously), and the `#[cfg(target_feature)]`-
resolves-too-early failure mode that quietly defeats runtime dispatch
(added to the compile-time-monomorphized-dispatch entry above, as a
concrete pitfall to check for if `runtime-next` ever adds multi-
architecture support). Nothing else here — CPU port-pressure modeling, x86/
ARM-specific shuffle intrinsics — has a meaningful GPU analog worth forcing.

**Status**: no action item; see the two cross-referenced entries above for
what was actually kept.

### flodl v0.8.0 (evaluated 2026-09-13)

**What it is**: a Rust deep learning *training* framework built on libtorch
(PyTorch's C++ backend, via a `tch-rs`-shaped binding) — fluent graph-builder
API, full optimizer/scheduler/loss parity with PyTorch, heterogeneous
multi-GPU/multi-host DDP (its standout result: a mixed RTX 5060 Ti + 2x GTX
1060 cluster beating a published CIFAR-10/ResNet-20 baseline on both
accuracy and wall time), a live training dashboard, and its own `fdl` CLI
for detecting hardware and managing libtorch installations. Real, measured,
honestly reported (single-GPU throughput: 8 wins/2 ties/0 regressions vs.
PyTorch 2.10 across 10 models). As of v0.8.0, AMD/ROCm support is explicitly
build-and-link-tested only — the authors state plainly "no AMD card has ever
run a training step" on this framework.

**Verdict: not relevant to `runtime-next`, on two independent grounds.**
(1) **Wrong layer.** flodl is a *training* framework — graph builder,
optimizers, DDP, training loops. `runtime-next`'s entire scope is
*inference serving* (KV cache management, HIP graph capture/replay for
decode, request batching, quantized-weight loading). Nothing in flodl's
feature set — DDP, schedulers, live training dashboards — has an inference-
serving analog. (2) **It doesn't escape the actual bottleneck this project
has repeatedly hit.** flodl links libtorch and therefore runs on *PyTorch's
own kernels* under the hood. This session's real findings (§76's economics
note in `TODO.md` item 4: no fused GatedDeltaNet kernels installable on this
rig, forcing a slow chunked-scan PyTorch fallback for any multi-token
verification; the conv1d batch-dependence bug itself, which lives in
PyTorch's own ROCm/MIOpen dispatch for a *reference* kernel) are all,
specifically, problems with relying on PyTorch's kernel layer on this
hardware. Depending on flodl would keep `runtime-next` on that same
foundation rather than escaping it — which is very plausibly the entire
point of writing a native Rust/HIP engine in the first place. On top of
both of those: flodl's own AMD path is explicitly unvalidated on real
hardware as of this post ("no AMD card has ever run a training step"), so
even setting aside the layer mismatch, it would be adopting unproven ground
on exactly this project's hardware class.

**Worth flagging, but explicitly out of scope for this file**: flodl's
*domain* (training, not inference) is a much closer match for `apps/factory`
(this project's existing Python LoRA/PEFT training pipeline) than for
`runtime-next`. If a Rust port of the training side is ever scoped as its
own project, flodl would be worth a real look then — including finding out
whether its AMD path has since been validated on real hardware. That's a
separate hypothetical to a separate future project, not an action item
here; not tracking it further in this file, which is scoped to
`runtime-next` specifically.

**(Corroboration, not a new idea)**: flodl's 0.8.0 release notes state, of
its own untested AMD path, "I would rather say that here, plainly, than let
someone discover it after an afternoon of setup... When the numbers arrive
they will arrive as numbers, not as a claim." Independent confirmation,
from a completely different project, of the same discipline already
corroborated from renew-engine and already this project's own standing
practice (the Zero-Mock Invariant, CANON-stamping). No action item — just
further evidence this is recognized good practice broadly, not a house
quirk.

**See "Techniques worth adapting" above** for what *is* worth carrying over
regardless of the framework-level "not relevant" verdict: `Drop`-based
deterministic GPU memory release (replacing this project's own manual
VRAM-cap workarounds), and an explainable, provenance-annotated resolved-
config command.

### Zero-copy wgpu-into-Electron rendering (Murlet, evaluated 2026-09-13)

**What it is**: a blog post on displaying GPU-rendered video frames inside
an Electron app without a CPU copy, comparing two techniques — "hole
punching" (a native NSView stacked beneath Electron's transparent content,
synchronized by simulating the frontend's CSS layout on the backend so the
two rarely need to talk) and Electron's experimental `sharedTexture` API (a
reference-counted GPU texture pool handed to Electron, with an
async release callback). The author chose hole punching; `sharedTexture`
broke down specifically at window-resize, when the whole texture pool has
to be invalidated and reallocated at once.

**Verdict: not relevant, and honestly so — no adjacent technique to salvage
either, unlike every other entry in this file so far.** Checked the two
places in this project that looked like plausible analogs before writing
this off: `apps/runtime/state_handoff_27b.py` (an in-process, GPU-internal
`.clone()` of ~147MB of DeltaNet/attention state between agent turns,
confirmed by reading the file — already zero-copy in the sense that
matters, but never crosses a process or rendering-compositor boundary, so
none of hole-punching's or `sharedTexture`'s problems apply), and
`apps/dashboard` (confirmed via its own README and `package.json`: a plain
Next.js/React web app, not Electron — a browser tab has no equivalent to
punching a hole through a native window's compositor or importing a shared
GPU texture from an external process; that capability is specifically an
Electron/native-app affordance). `runtime-next` is a headless inference
server with no windowing, no video, and no Electron anywhere in its stated
scope. Forcing a connection here would be padding, not adaptation.

**The one honestly-abstracted lesson, kept narrow on purpose**: the
`sharedTexture` postmortem's real content is "a reference-counted resource
pool works fine in steady state and fails specifically at a reconfiguration
event that invalidates the whole pool at once (there: window resize; the
fix was to explicitly suspend delivery during resize rather than make the
fast path also handle it)." That specific *shape* of failure — a fixed-size
pool/buffer design that's correct for steady-state but silently wrong the
moment the usage pattern changes shape — is one this project has hit for
real, twice, this session: `StaticCache`'s fixed-address buffers assume a
stable request shape, and `StateRingBuffer`'s write-pointer bug (`docs/DECISIONS.md`
§75) only appeared once the access pattern changed from single-branch to
multi-branch. The transferable point isn't "handle resize" — it's "when
designing any fixed-size GPU resource pool for `runtime-next`, explicitly
design and test the reconfiguration/shape-change path as its own case
(even if the answer is 'briefly serialize/degrade during it'), rather than
assuming the steady-state-optimized design generalizes." Not a new
technique so much as a second, independent data point for a lesson this
project already learned the hard way.

**Status**: no action item. The reconfiguration-boundary lesson is already
implicitly covered by `TODO.md` item 2's design guidance (explicit
caller-assigned slot IDs, not implicit pointers) — logged here only as
corroboration from a completely unrelated domain, not a new requirement.

### Searching 150 GiB/s with SIMD (Ashwa, evaluated 2026-09-13)

**What it is**: a real, well-measured walkthrough of substring/byte search
getting faster through three stages — scalar byte-by-byte scan (~2.14 GiB/s,
compute-bound), SWAR (8 bytes/register via bitwise zero-byte-detection
tricks, ~9.25 GiB/s), and AVX512BW (`vpcmpeqb` over a full 64-byte cache
line at once, ~150 GiB/s cache-resident). Its sharpest point isn't the
speed-up itself but the ceiling: once the payload is DRAM-bound rather than
cache-resident, throughput collapses to ~11-12 GiB/s *regardless of vector
width*, because a single CPU core has a small, fixed pool of Line Fill
Buffers tracking in-flight cache misses — wider registers can't out-run a
memory-bandwidth wall governed by a completely different hardware resource.

**Verdict: not relevant, same shape as fearless_simd — CPU SIMD, and this
project's compute surface is GPU/HIP.** Checked for a real point of contact
before dismissing it, same as every other entry here: `apps/runtime/long_context_engine.py`'s
`PinnedPrefixCache` is a single fixed system prompt (an equality/reuse
check, not a search over candidates), and `context_shift_if_needed`
estimates token count by `len(content)//4`, not a byte scan. The one place
generation-time substring matching genuinely happens — `stop_strings=[...]`
checks against freshly generated text, used throughout this project's own
benchmark scripts — operates on tens of bytes per check, not gigabytes;
every stage in this article's own numbers only starts to separate at
kilobyte-and-up payloads. No operation in this codebase runs at the scale
where any of these three techniques would show a measurable difference over
a naive scan. If `runtime-next` ever needs real high-throughput text
scanning (JSON request parsing, a tokenizer's own internals), that would be
delegated to an existing crate that already does this internally
(`simd-json`, a real BPE tokenizer crate) — not hand-rolled, the same
reasoning already applied to fearless_simd.

**One abstracted principle worth keeping, independent of CPU SIMD
specifically**: classify a workload as compute-bound or memory-bound
*before* deciding what kind of optimization can possibly help — wider
execution units and more parallelism only help the compute-bound case;
past that, throughput is capped by an entirely different resource (Line
Fill Buffers here; on a GPU, memory-controller/L2 bandwidth and occupancy).
This isn't a new idea, but it's a concrete, numbers-backed illustration of
a check worth applying deliberately to `runtime-next`'s own future kernel
work: `TODO.md` item 4's "flat ~2.7-2.84x verification cost" finding is
plausibly a compute-bound-vs-memory-bound story in exactly this shape
(a missing fused kernel forcing a slower, more ALU/instruction-heavy
reference path) — worth explicitly classifying which regime each real
kernel (GDN's chunked scan, decode-phase attention, W4A16 dequant) falls
into before assuming a "just write a custom HIP kernel" fix will help,
since a memory-bound kernel needs a bandwidth/data-movement fix, not a
wider or more-parallel compute unit.

**Status**: no action item on the crate/article itself; the roofline-style
classification habit is worth applying whenever real kernel-writing starts
in `runtime-next`, tied here to `TODO.md` item 4 rather than logged as a
new requirement.

### "Rewriting tracing_subscriber::reload to panic less" (evaluated 2026-09-13)

**What it is**: a real, deeply-worked-through rewrite of `tracing-subscriber`'s
built-in `reload::Layer` (the mechanism for changing a Rust server's logging/
tracing configuration — log level, output format, OpenTelemetry endpoint —
without restarting the process). The stock version panics in several real
scenarios (downcasting to a wrapped layer, per-layer-filter setup after
reload, layers that track span lifecycle). The rewrite fixes those, but at a
disclosed, deliberate cost: it **leaks memory on every reload**, because the
only sound alternative to leaking is either undefined behavior (a raw
pointer to a layer that might get freed while external code still holds a
copy of it) or an unsafe opt-in escape hatch. The author's own framing:
"Leaking is memory safe" — a genuine, well-reasoned safety-over-thrift
tradeoff, not an accident.

**Verdict: `tracing`/`tracing-subscriber` (the base crate, not this reload
rewrite) is a real, relevant future need for `runtime-next` — but this
specific crate is not, and the underlying feature it provides (reload
*without a restart*) is a weaker fit here than for the general server this
article is written for.** `runtime-next` will want structured logging and
request-lifecycle tracing eventually — `tracing`/`tracing-subscriber` is the
mature, ecosystem-standard choice for that in Rust, and worth naming now as
the default rather than reinventing one later. But this project's own
Python runtime already documents its serving model as single-tenant,
GPU-exclusive, one request at a time (`apps/runtime/server.py`'s own
docstring), and this project routinely restarts the server for model/adapter
changes already (every benchmark this session that swaps adapters reloads
the whole process). The motivating scenario for a *dynamic, no-restart*
reload feature — "a large always-on service where restarting to change log
level is itself risky or expensive" — describes a different kind of system
than `runtime-next` is shaping up to be. Configuring `tracing` once at
startup, with a restart to change it, is very plausibly all `runtime-next`
will ever need — which avoids this crate's whole leak-or-UB dilemma by
never entering it.

**One lesson worth keeping, reinforcing (not duplicating) the `Preload`/
`PreloadMut` pattern already logged from "GPU Offload in Rust" above**:
this crate was forced into "leak or risk UB" specifically because `tracing`'s
own ecosystem-wide API contract (`Layer::downcast_raw`) hands out a raw
`*const T` that external, uncontrolled code is free to cache indefinitely —
once a pointer escapes like that, nothing (not even a well-designed Rust
API) can know when it's safe to free the thing it points to. This is a
concrete cautionary tale for `runtime-next`'s own future GPU buffer wrapper
types: the reason a `Preload`/`PreloadMut`-style design (tying a GPU-resident
buffer's lifetime to a live Rust borrow, `drop` as the sync point) works
without needing to leak is that it *never* hands out an unconstrained raw
pointer that outside code could cache past the borrow's lifetime. The moment
a future `DeviceBuffer<T>` API needs to expose a raw pointer to, say, a
vendor library or an FFI boundary that might hold onto it indefinitely,
this exact dilemma reappears — worth deciding deliberately (leak, unsafe
escape hatch, or refuse the API) rather than discovering it the way this
crate's author did.

**Status**: no action item now — `runtime-next` has no logging/tracing setup
yet, and none is urgent before the engine itself exists. When it is added,
default to plain `tracing`/`tracing-subscriber` configured once at startup;
revisit dynamic reload only if a real operational need for it shows up
(e.g. if `runtime-next` ever stops being single-tenant/restart-tolerant),
and if so, re-evaluate this specific crate's leak tradeoff against
`runtime-next`'s own zero-allocation goals at that time rather than assuming
it's still the best option.

### Two real techniques from a roguelike's threading architecture ("Leaves of Steel" dev blog, evaluated 2026-09-14)

**What it is**: a long, mostly game-specific architecture post (level data
structures, TOML prefab formats, corridor generation, game-balance
mechanics — none of that is relevant here and isn't logged). Buried in it
is a genuinely reusable pair of concurrency techniques from the game's
render/logic thread split, worth extracting on their own.

**Technique 1 — eliminate a whole deadlock class by minimizing distinct
lock instances, not by proving lock-ordering discipline.** The author
uses exactly one `RwLock` in the entire program (`HashMap<LevelIdentifier,
Arc<RwLock<Level>>>`), specifically because a deadlock requires circular
wait between two *different* locks — with only one lock type in the whole
system, that category of bug becomes structurally impossible rather than
something to audit for. **Why it's worth keeping in mind for
`runtime-next`**: if the Rust port ever needs internal locking around
shared mutable state (a request queue, a KV-cache-slot table, anything not
already covered by the GPU-buffer ownership patterns logged above), fewer
distinct lock types is a real, low-cost way to shrink the deadlock surface
by construction, worth deciding as a constraint up front rather than an
audit to perform later.

**Technique 2 — force a lock's guard lifetime through the borrow checker by
requiring `&mut` on its wrapper, even for logical reads.** The concrete
hazard: a thread that locks the same `RwLock` twice (once in an outer
function, again in a function it calls) deadlocks itself, and that bug is
invisible until it actually runs. The fix: never expose the `Arc<RwLock<_>>`
directly — thread it through a wrapper struct (`ActionParams`) that every
function receives as `&mut`, even functions that only read the lock. Since
Rust won't let you hold two live `&mut` borrows of the same wrapper at once,
forgetting to `drop` a read guard before calling into another function that
also wants the lock becomes a **compile-time borrow-checker error** instead
of a runtime deadlock discovered under load. **Why it's worth adapting
here**: this is a genuinely different, complementary technique from the
`Preload`/`PreloadMut` pattern already logged (that one is about GPU buffer
*lifetime*; this one is about *serializing lock-guard scope* through
ordinary borrow-checker rules) — worth remembering as the specific playbook
if `runtime-next` ever has a lock a request-handling call graph might
re-enter, since the failure mode (self-deadlock across a call boundary) is
exactly the kind of bug that's cheap to prevent by construction and
expensive to debug after the fact.

**Minor, real tooling note**: the author praises `rr` (Mozilla's
record-and-replay debugger, wraps `gdb`) for exactly this class of
multithreaded bug — deterministic replay of a hard-to-reproduce lock/timing
issue. Worth remembering as a real, known-good tool to reach for if
`runtime-next` ever hits a concurrency bug that doesn't reproduce reliably,
rather than rediscovering that such tooling exists mid-crisis.

**Status**: no action item now — no internal locking exists in `runtime-next`
yet, since it's still a stub. Both techniques are cheap, load-bearing design
decisions to make *before* the first `Mutex`/`RwLock` is added, not
refactors to do afterward.

### NVIDIA CUDA Rust: cuda-oxide (SIMT) and cutile-rs (Tile) (evaluated 2026-09-14)

**What it is**: NVIDIA's own announcement of native Rust GPU kernel
authoring — kernels written in Rust, compiled to PTX directly, not FFI
wrappers around CUDA C++. Two tracks matching CUDA's own two programming
models: `cuda-oxide` (SIMT — one thread's work, launch thousands; a custom
`rustc` codegen backend, early alpha) and `cutile-rs` (Tile — one tile's
work, the compiler maps it onto real threads; published on crates.io,
already used outside NVIDIA in HuggingFace's Grout inference engine and
in mistral.rs). Both make the same real safety argument the type system
enforces at compile time: shared inputs are ordinary borrows, and the one
mutable output gets a purpose-built type instead of `&mut`, so the classic
GPU data race (two threads touch the same address, one writing) becomes a
borrow-check error instead of a bug that "rarely reproduces on demand and
passes tests before failing in production" (the announcement's own words —
and this project has its own version of that exact experience with §76's
conv1d bug).

**Verdict: not directly usable — both compile to PTX via the CUDA toolkit,
with no AMD/HIP target mentioned anywhere — but this is the most
industrially significant, most concrete evidence yet that safe native-Rust
GPU kernel authoring is a real, live, well-funded direction, not a niche
research idea.** Real production-adjacent usage (Grout, mistral.rs) is a
meaningfully stronger adoption signal than anything else evaluated in this
file. If `runtime-next` ever writes custom HIP kernels by hand — which the
missing-fused-kernel economics already documented (`TODO.md` item 4) make
a real possibility, not a hypothetical — the *patterns* below are worth
copying even though the crates themselves are the wrong vendor.

**Two concrete, adoptable blueprints, not just principles this time**:

1. **`DisjointSlice<T>`** — the answer to "N GPU threads each need to write
   their own element of one output buffer, safely." `&mut [T]` is the wrong
   shape (every thread would need the same `&mut`, which Rust correctly
   refuses); `DisjointSlice` splits one mutable borrow into per-thread
   pieces, indexed by a real index type (`thread::index_1d()`) rather than
   a bare integer, with `.get_mut(idx)` returning `Option` so out-of-bounds
   is a branch, not a memory error found later. This is a complementary
   problem to the already-logged `Preload`/`PreloadMut` pattern (that one
   is host/device data movement; this one is intra-kernel parallel-write
   safety) — if `runtime-next` ever hand-writes a kernel where many threads
   write disjoint outputs (an elementwise dequant kernel is the most
   obvious candidate given the W4A16 path already discussed), this is the
   concrete shape to copy, not a novel design problem to solve from
   scratch.
2. **`#[launch_contract]` + a `prepare_*` step returning a proof token** —
   already cross-referenced into the type-state-gating entry above; the
   short version is that a kernel declares its expected grid/block shape as
   an attribute, and a `prepare_*` call validates a caller's launch config
   against both that contract and live device limits *before* handing back
   a token the actual (safe) launch method requires. Kernels without a
   contract only expose raw `unsafe` launch methods. Directly portable to
   however `runtime-next` ends up exposing HIP kernel launches.

**One real strategic framing, not just a technique**: the article is
explicit that Tile is "safe by construction" specifically *because* there
are no threads or shared memory for the programmer to get wrong — and that
this is also what gets traded away, since "shared memory is the bedrock of
fast SIMT kernels" and today requires dropping to the unsafe SIMT track
even in NVIDIA's own ecosystem. This maps directly onto a real, already-
documented fact about this project's own workload: GatedDeltaNet's chunked-
scan kernel is exactly the kind of thing that needs real shared-memory-
level control to be fast (that's the whole reason `flash-linear-attention`/
`causal_conv1d` exist as hand-tuned kernels rather than something a
tile-level compiler could have generated). **The lesson for `runtime-next`**:
if a higher-level, safer-by-construction Rust GPU abstraction (Tile-shaped
or otherwise) is ever adopted for simple elementwise ops, don't assume the
same abstraction extends to GDN-shaped kernels needing fine-grained shared-
memory control — that class of kernel will very plausibly need the lower-
level, `DisjointSlice`-shaped safety story instead, matching which of
NVIDIA's own two tracks its authors reach for first.

**Minor corroboration**: `cutile-rs`'s host-side API is fully lazy — nothing
touches the GPU until one final `.sync_on(&stream)` call; every op before
that (allocation, the kernel launch, the copy back) is recorded, not
executed. This is independent, real-world confirmation that "record a
pipeline, then execute it as one unit at a single sync point" is a
legitimate, industry-recognized shape for a Rust GPU API — which is close
to what `runtime-next`'s own "monolithic HIP Graph pipeline" framing already
intends. No new idea, just corroboration that the framing is a well-trodden
one.

**Status**: no action item — wrong vendor for direct use. Worth remembering
as a first place to look for real Rust-GPU-kernel prior art (specifically
the `DisjointSlice` and launch-contract patterns) whenever `runtime-next`
first needs to hand-write a custom HIP kernel rather than link one, and
worth periodically checking whether either project (or an AMD-targeting
equivalent) gains a ROCm backend before that point, the same "recheck
later" status already given to "GPU Offload in Rust" above.

### A fast chess move generator ("Watermelon", evaluated 2026-09-14) — one real technique, plus the clearest confirmation yet of a pattern this file keeps re-discovering

**What it is**: a real, benchmarked (475M nodes/sec), fully-legal chess move
generator in Rust — bitboard state representation, PEXT-based sliding-piece
lookup tables, checkmask/pinmask-based legality filtering. Domain is
chess-specific and mostly not transferable (bitboards, chess-specific bit
tricks), but three things underneath it are real, general Rust techniques,
and the article's own honesty section is worth a one-line mention on its
own.

**1. New, concrete: derive a true worst-case bound from domain rules, then
use a fixed-capacity, stack-allocated collection instead of a heap-growable
one.** `MoveList` holds up to 271 moves — not a guessed buffer size, but
chess's own combinatorics *proving* no legal position ever exceeds that —
stored on the stack, zero heap allocation, ever. **Directly reinforces
`runtime-next`'s own zero-allocation goal with a concrete recipe**: for any
variable-but-boundable collection (live speculative branches, candidate
tokens in a beam, active adapters in a stack), derive the actual worst-case
size from the domain's own constraints — the same way `docs/DECISIONS.md`
§75 already did for this project's own KV-fork tree ("0.26 MB for width 2 x
depth 4" was exactly this kind of derived bound) — and size a fixed
stack/arena allocation to it, rather than defaulting to a growable
heap collection "to be safe."

**2. New, minor: `std::hint::cold_path()` for hinting a rarely-taken branch**
(used here for the "pawn on last rank, check for promotions" case). A
small, concrete, real Rust perf tool worth having on hand for
`runtime-next`'s own decode loop, where the common case (plain next-token
decode) vastly outweighs rare cases (EOS, an adapter swap, a gate firing).

**3. Not new, but the clearest, most concrete instance yet of a pattern this
file has now independently rediscovered four separate times.** `SearchZero`
→ `.legal_moves()` → `SearchMany` (a move list *proven* legal for the
frozen position, by the type itself) → `.as_one(i)` → `SearchOne` →
`.test(closure)`, which plays the move via the unsafe fast path *without
re-checking legality*, because the type already proves it was checked. The
article's own framing nails exactly why this matters: "You just generated a
list of legal moves, of course they're all legal!" — redundant re-
validation is itself a real performance and correctness-hazard-surface
problem, not just an efficiency nitpick. This is the same shape as
`Preload`/`PreloadMut` (GPU Offload in Rust), `#[launch_contract]` +
proof-token (CUDA Rust), and the original type-state-gating entry
(fearless_simd) — four independent domains (CPU SIMD, GPU offload, GPU
kernel launch, chess move generation) converging on one idiom: **carry a
validated precondition through the type system so a later expensive or
unsafe step is structurally unable to skip or redundantly repeat the
check.** Worth naming explicitly as a standing design principle for
`runtime-next` generally, not just for GPU buffers specifically — anywhere
`runtime-next` validates something once (a request's shape, a KV-cache
state's validity, a completed speculative-decode exploration's result),
the API around it should make re-validation structurally unnecessary, the
same way `SearchOne::test` does.

**Minor, unrelated but worth filing next to the earlier SIMD-substring-search
rejection**: this article's Zobrist hashing (an incrementally-*updated* hash
of a large state, XORed in/out per move rather than rehashed from scratch)
is a real, efficient technique for cheap "have I seen this exact state
before" checks. If `runtime-next` ever builds real prompt-prefix caching or
KV-cache-state deduplication (the gap already noted when the SIMD substring-
search article was evaluated — no such mechanism exists in this project
today), incremental rolling-hash techniques like this, not "hash the whole
prefix from scratch per request," are the right family of technique to
reach for.

**Minor corroboration**: the article's own honest closing section — "this
project is in fact worthless for real-world chess engines... pseudolegal
move generation is preferred to our legal move generation" — is a good
reminder that a technically fast, impressive implementation is not the
same claim as "the right architecture for the actual target use case."
Same spirit as this file's other performance-claims-need-context
corroborations; no new entry.

**Status**: no action item — items 1 and 2 are cheap, adoptable habits for
whenever `runtime-next` writes its own collections/hot loops; item 3 is a
standing design principle worth stating explicitly given how many times
this file has now converged on it independently.

### Flat 2D arrays vs. nested `Vec<Vec<T>>` (Nazmul Idris / developerlife.com, evaluated 2026-09-14)

**What it is**: a terminal-UI-focused case study on 2D grid layout in Rust —
`Vec<Vec<T>>` (one heap allocation per row, scattered in memory) vs. a flat
`Box<[T]>` with manual `row * cols + col` indexing. Real, measured numbers:
flattening wins big on every scan-heavy operation (2.3x read/render, 1.4x
clear, 1.8x coordinate-aware traversal, 39x for a size calculation that
degenerates into pointer-chasing on the nested version) — because a
contiguous allocation lets the CPU's hardware prefetcher and SIMD
auto-vectorization actually engage, where scattered per-row allocations
defeat both.

**Verdict: not about a technology to adopt — this is CPU cache/memory-
layout discipline, which applies to plain host-side Rust data structures
regardless of domain — but it lands on a real, current gap in this
project's own work, not a hypothetical one.** This session's own
`state_replay` tree-exploration scripts (§75/§78/§79) build their trees as
Python dicts, each node holding a `"parent"` reference to another
separately-allocated dict — exactly the `Vec<Vec<T>>`-shaped anti-pattern
this article measures, just in Python instead of Rust. **If any of that
tree-building logic is ever ported to `runtime-next`**, don't translate the
node-with-a-parent-pointer shape literally into `Vec<Box<Node>>` or an
`Rc`/`Arc`-linked tree — flatten it into one contiguous arena (a `Vec<Node>`
indexed by integer offset, parent/child stored as indices, not pointers),
the same way this article's `Flat2DArray` replaces per-row allocations with
one buffer and integer math.

**A second, real angle beyond throughput**: the article notes `Vec<Vec<T>>`
doesn't just lose on average speed, it loses on *consistency* — cache-miss-
driven variance up to ±98%, vs. flatline timing for the contiguous version.
For a UI that's stutter; for an inference server, the analogous concern is
**tail latency**. A host-side data structure that's fine on average
throughput but has scattered-allocation-driven variance would show up as
inconsistent per-token latency (bad P99, fine mean) — exactly the kind of
thing an average-throughput benchmark hides. Worth keeping in mind as
another reason `runtime-next`'s own host-side bookkeeping (not just its GPU
buffers) deserves the same contiguous-layout discipline, and worth
measuring latency *variance*, not just mean throughput, once real
benchmarks exist there.

**One honestly-reported counter-example, worth taking as seriously as the
wins**: `Vec<Vec<T>>` actually *beats* the flat array for one specific
operation — scrolling — because rotating an array of row-pointers
(`rotate_left(1)`) is cheaper than physically moving the row bytes
(`copy_within`). The lesson isn't "always flatten," it's "pick the layout
that matches the *dominant* operation," and this project already has a
live example of exactly that tension: `StateRingBuffer` is fundamentally a
ring of slots that needs cheap *rotation*, not scanning — the same shape
that favors pointer-indirection here. Worth remembering as calibration if
`StateRingBuffer`'s own real, already-documented write-pointer bug
(`docs/DECISIONS.md` §75, `TODO.md` item 2) is ever redesigned from
scratch: don't reflexively flatten it just because flattening won other
benchmarks in this article — rotation-heavy structures are the case where
indirection is the right call.

**Minor, adoptable technique**: `.chunks_exact(width)` for row-aware
iteration instead of manual `idx / width, idx % width` — division/modulo
are slow and defeat vectorization, `chunks_exact` compiles to pure pointer
addition. Small, real, easy to apply wherever `runtime-next` ends up with
any grid/matrix-shaped host-side buffer (batch × sequence-length token or
position tables are the most obvious candidate).

**Status**: no action item now — no host-side tree/grid structures exist in
`runtime-next` yet. The arena-not-pointers lesson is specifically worth
recalling if `state_replay`'s tree-exploration work (still just Python
experiments per `TODO.md` item 5) is ever ported; the ring-buffer
calibration note is worth recalling if `StateRingBuffer` is ever
redesigned rather than reimplemented as-is.

### multicalc (evaluated 2026-09-14) — one real, narrow connection inside a large robotics/control crate

**What it is**: a large, real, well-executed `no_std` scientific-computing
crate for robotics and control — Kalman filters, PID/LQR control,
kinematics/dynamics, autodiff, fixed-size stack-allocated linear algebra
(LU/Cholesky/QR/SVD/`expm`), ODE integrators, signal processing
(`Biquad`, `MovingAverage`, `RunningMedian`, `Deadband`, `Hysteresis`,
`SlewRateLimiter`). Real discipline behind it: `#![forbid(unsafe_code)]`,
no heap, `unwrap`/`panic` denied on library paths, tested on six real
embedded targets under QEMU every commit, and every module's numbers
checked against `numpy`/`scipy`/`filterpy` fixtures to ~1 ulp.

**Verdict: almost entirely a different domain (robotics/control), correctly
not relevant — except for one real, narrow, worth-checking connection.**
`multicalc`'s signal-processing module (`MovingAverage`, `RunningMedian`,
`Deadband`, `Hysteresis`, `Biquad` filters — all small, fixed-size,
real-time-safe adaptive statistics trackers with EMA/variance-style rolling
state) is structurally the same *kind* of thing `range_statistic_gate.py`'s
`update_volatility` hand-rolls (a Bollinger-band/EMA/ATR volatility tracker
feeding the weibull-hazard gate already discussed extensively this
session). **If `range_statistic_gate.py` is ever ported to `runtime-next`**,
this is worth checking as either a real dependency or, at minimum, an
independently-validated reference implementation to check a hand-rolled
port's numerics against — given this project's own repeated experience this
session with subtle, hard-to-spot numeric bugs in hand-rolled statistics
and kernel code (the conv1d batch-dependence bug, the AITER "correct only
at one specific shape" bug), an externally-verified-to-1-ulp library is a
real, cheap sanity check to reach for rather than trusting a fresh
translation of the Python math by eye.

**Explicitly ruled out, not just skipped**: the autodiff feature does *not*
meaningfully inform `apps/factory-next`'s already-logged "backward-pass is
the hard, unresolved part" gap. `multicalc`'s autodiff differentiates small
hand-written scalar/vector formulas (a handful of inputs, control-theory-
sized), not tensor-level backprop through a multi-billion-parameter network
with custom GatedDeltaNet/attention layers — a different scale and problem
entirely. Worth stating this explicitly since "autodiff" is exactly the
kind of keyword match that invites a false connection to that gap.

**Corroboration only, no new entries**: another large, real, diverse-domain
proof that `no_std`/no-heap/no-panic/no-`unsafe` is achievable together at
real scale (joining renew-engine and the chess move generator already
logged), and another confirmation of "verify against real external
references, not just your own tests" (numpy/scipy/filterpy fixtures to
~1 ulp) — both principles this file has now logged corroboration for
several times over.

**Status**: no action item — nothing in `runtime-next` needs robotics math.
The one real thread (signal-processing primitives as a reference for a
future `range_statistic_gate.py` port) is worth recalling specifically
when that porting work actually starts, not before.

### Rust's algebraic float operators, `algebraic_add`/`algebraic_sub`/`algebraic_mul` (Itamar Turner-Trauring, evaluated 2026-09-14) — directly relevant, and quietly already logged once without the explanation

**What it is**: Rust 1.98 (Aug 2026) stabilizes per-operation "algebraic"
float arithmetic — `algebraic_add` etc. — that gives the compiler
permission to reassociate/reorder that *specific* operation the way
`-ffast-math` does globally in C/C++, without `-ffast-math`'s dangerous
baggage (no blanket "assume no NaN/no infinity" that can trigger UB
elsewhere in the same program). Regular `+` stays exactly order-preserving
because float addition genuinely isn't associative (`1e16 + 1.0 == 1e16`
in float arithmetic — adding the small number does nothing), so the
compiler is correctly forbidden from reordering it by default; algebraic
ops are how you explicitly waive that guarantee, one call site at a time.
Real, measured numbers: a naive sequential sum of 1M f64s got zero
auto-vectorization and ran ~3.5x slower than the integer equivalent;
rewriting just the innermost reduction step of a pairwise-summation
algorithm with `algebraic_add` let the compiler auto-vectorize it,
beating even NumPy's own pairwise sum, at the *same* accuracy — and a
sum-of-squared-differences kernel went 2x faster from three `algebraic_*`
calls.

**Verdict: directly relevant — and this is the same underlying Rust
feature the "GPU Offload in Rust" paper already used without this file
explaining what it actually was.** That paper's own "Fast-Math Impact"
section measured a real 2x speedup on one RAJAPerf kernel from "Rust's
experimental algebraic floating-point operations" — logged in this file's
entry for that paper, but only as a headline number, not as an explained
mechanism. This article is the missing explanation, and it adds a
genuinely more useful technique than "turn on algebraic ops and see what
happens."

**The real, concrete technique — mix strict and algebraic operators
*within one algorithm*, deliberately, not as an all-or-nothing switch**:
the pairwise-summation example uses plain `+` for the top-level combine of
two independently-computed subtree sums (order matters there for
accuracy) and `algebraic_add` only for the base-case loop below a size
threshold (order doesn't matter there, so let the compiler vectorize
freely). This is the real value over blanket `-ffast-math`: **precision-
critical accumulation and speed-critical accumulation can coexist in the
same function**, decided call-site by call-site, not module-wide.

**Why this matters more here than in most projects**: this project's
central, already-documented economic problem all session has been missing
*fused* kernels (`flash-linear-attention`, `causal_conv1d`) forcing slow
reference-path reductions for GatedDeltaNet's chunked scan (`TODO.md`
item 4). Reductions are exactly where this technique bites hardest —
softmax's normalization sum, RMSNorm's mean-of-squares, attention's
weighted KV sum, and this very session's own `cum_logprob` accumulation
(`benchmark_gated_tree_quality.py`/`benchmark_gated_lookahead_policy.py`)
are all sequential float reductions. **If `runtime-next` ever hand-writes
a fused reduction kernel**, this is the tool to reach for — and,
consistent with this project's own repeated scars this session (the
conv1d batch-dependence bug, AITER's shape-dependent silent wrongness),
the discipline should be the same as the pairwise-sum example: use
algebraic ops only where reassociation is provably safe for the
algorithm's own accuracy needs, verified against a reference computation,
never applied blanket "for speed" without checking what specifically
becomes reorderable.

**Status**: no action item now — no custom Rust/HIP kernels exist yet to
apply this to. Worth recalling specifically once `runtime-next` writes its
first hand-rolled reduction-shaped kernel (normalization, softmax, or a
GDN-scan replacement), and worth updating the "GPU Offload in Rust" entry's
own fast-math mention to point here if that paper's finding is revisited.

### A trading bot's Python→Rust migration (Alexander Vinokurov, evaluated 2026-09-14) — a genuine reconsideration of HOW to approach `runtime-next` itself, not just a technique

**What it is**: a real, measured case study of moving a latency-critical
system from Python to Rust — a Polymarket market-making bot leaking money
on one specific event (repairing a position after a partial fill, which
took 0-2+ seconds in pure Python: a multi-second poll plus DB writes plus
reconciliation). The team's explicit framing: "the temptation, when a
Python system is too slow, is to rewrite the whole thing in Rust. That was
the wrong instinct here. The problem was not the language — it was a slow
control loop." They moved *only* order-book ingestion, the decision loop,
and order placement into a Rust sidecar — keeping quant models, signing,
accounting, risk, and the dashboard in Python — connected over a local
Unix socket, with the Rust side explicitly best-effort (if it drops,
Python falls back to native handling, so the sidecar can never take the
whole system down). Result: reaction time on the expensive event dropped
from 0-2+ seconds to the low tens of milliseconds. The core migration
shipped in about two weeks.

**Verdict: directly relevant, and worth surfacing as a real strategic
question rather than filing away as a minor technique — this is a
plausible alternative shape for the whole `runtime-next` effort, not just
one more pattern for its eventual internals.** `runtime-next`'s current
framing (`src/main.rs`'s own docstring, now scoped to phase 1 — see
`TODO.md` "Phased scope": "Zero-allocation inference & monolithic HIP Graph
pipeline for Qwen3.5-4B/9B") reads as a from-scratch, all-in-Rust rewrite
that has to reach feature parity with the existing Python runtime
(`apps/runtime`/`apps/runtime-ipwf` — adapter management,
request routing, tokenization, the HTTP surface, dashboards) before it's
useful for anything. This case study demonstrates a real, lower-risk
alternative: **identify the actual narrow latency-critical hot path (for
`runtime-next`, plausibly just the per-token decode step / HIP graph
replay), move only that into a Rust sidecar, and keep everything else —
model/adapter loading, the HTTP server, tokenization, routing — in the
already-working, already-debugged Python runtime, talking to the sidecar
over a fast local channel.** This would let real GPU-dispatch-overhead and
kernel-fusion wins (the actual motivation for `runtime-next` per `TODO.md`
item 4) get captured and measured incrementally, without first having to
rebuild everything the Python runtime already does correctly. Whether this
is actually the better path is a real decision, not a foregone one — a
monolithic rewrite gives cleaner long-term ownership of the whole stack,
while a sidecar defers (perhaps indefinitely) ever fully retiring the
Python runtime — but it's a genuine, load-bearing architectural choice
worth making deliberately, not by default inertia toward "full rewrite"
just because that's how the stub currently reads.

**Two smaller, real supporting patterns worth keeping regardless of that
larger decision**:
1. **Best-effort fast path with a native fallback** — the sidecar dropping
   never takes the system down; Python's slower native path is always
   there as a safety net. If `runtime-next` starts as a sidecar for even
   one narrow operation, design it the same way from day one: a HIP-side
   failure should degrade to the existing Python path, not crash serving
   entirely.
2. **Investigate the scariest-looking part first.** The team assumed
   cryptographic pre-signing would be the hard part of the migration; on
   investigation it was already ~1ms and not the bottleneck at all,
   "which removed most of the risk." Worth doing the same due-diligence
   pass on whatever part of a `runtime-next` sidecar looks scariest before
   assuming it drives the whole project's timeline or risk.

**Status**: this is a decision for the user, not something to act on
unilaterally — flagged here as a real strategic question to weigh (full
rewrite vs. narrow hot-path sidecar) whenever `runtime-next` moves from
prep/docs into actual scaffolding, not a technique to quietly bake in.

### Tail-call-based "direct dispatch" for VM/interpreter loops (Jimmy Ostler, evaluated 2026-09-14) — real technique, likely wrong bottleneck for this project

**What it is**: a benchmarked comparison of bytecode-VM dispatch strategies
in Rust using the unstable `explicit_tail_calls` (`become`) feature —
switch dispatch, subroutine threading, indirect threading, and "direct
dispatch" (each instruction handler tail-calls the next handler directly,
no central dispatch loop at all). Real measured result: direct dispatch
won clearly in both a stack-machine and a register-machine test (~2.4x and
~1.5x faster than switch dispatch respectively), because `become` compiles
recursive dispatch into a plain jump with no new stack frame and no return
to a central loop.

**Verdict: a real technique, but almost certainly not where `runtime-next`'s
performance will come from, and worth saying so plainly rather than
force-fitting it as valuable.** Two honest caveats: (1) `become` is a
nightly-only, unstable Rust feature — same adoption caveat already logged
for other early-stage tools in this file. (2) More importantly, these are
nanosecond-scale CPU dispatch-loop measurements (24-183 ns/iter), and
`runtime-next`'s actual bottleneck is GPU-side — a single real decode step
already costs tens of *milliseconds* on this hardware (see any of this
session's own real timing numbers, e.g. `docs/DECISIONS.md` §78's ~35ms
per forward pass). A CPU dispatch loop tail-called instead of switch-
dispatched saves single-digit nanoseconds per step — five to six orders of
magnitude below the actual cost per step. This is exactly the kind of
micro-optimization the "measure before optimizing," Amdahl's-Law-style
discipline already logged in this file (from the flat-2D-array article)
warns against chasing without first confirming it's the fraction of time
that matters.

**Where it could still matter, narrowly**: only if `runtime-next` ends up
with a CPU-side request-multiplexing or per-token state-machine dispatch
loop (juggling {prefill, decode, gate-check, adapter-swap} as enum
variants) that is *itself* independently measured as a real bottleneck —
which nothing in this project's architecture suggests today, since the
GPU kernel dispatch and memory bandwidth are the established real costs.

**Status**: no action item. Worth recalling only if a future CPU-side
dispatch loop in `runtime-next` is specifically profiled and found to be a
real bottleneck — not worth designing around preemptively.
