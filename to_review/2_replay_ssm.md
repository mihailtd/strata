# Ring-Buffered Hybrid State Rollback (ReplaySSM): Walkthrough

## Executive Summary
We have implemented and verified **Ring-Buffered Hybrid State Rollback (ReplaySSM)** for speculative decoding in hybrid architectures (Qwen3.5-4B: 24 GatedDeltaNet SSM layers + 12 Full-Attention layers).

By eliminating dynamic GPU memory allocations (`.clone()`) and replacing them with a fixed-size pre-allocated tensor ring buffer with sub-microsecond pointer rollback and sequence-length KV cache cropping:
- **Phase 1 Benchmark:** State recovery latency reduced from **499.61 µs** (baseline) to **1.28 µs** (**388.9x faster**).
- **VRAM Footprint:** Fixed ring buffer occupies **51.0 MB** (0.208% of 24 GB VRAM), well beneath the 5% (1.2 GB) kill-switch ceiling.
- **Numerical Precision:** Bit-for-bit exact tensor state restoration (`relative L2 error = 0.00e+00`).
- **End-to-End Speculative Generation:** **100% token-for-token text match** across all prompts with a **+17.45% mean generation speedup** (up to **+43.1%** on dense speculative sequences).

---

## Phase 1: Benchmark & Kill-Switch Results

| Criterion | Target / Threshold | Measured Baseline | Measured ReplaySSM | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Memory Ceiling (Gate 1.2)** | $< 5.0\%$ of 24 GB VRAM ($< 1,225\text{ MB}$) | Dynamic malloc/free | **51.0 MB** (0.208% of VRAM) | **PASSED** |
| **Rollback Latency (Gate 1.3)** | $< 10\text{ µs}$ per rollback event | $499.61\text{ µs}$ | **$1.28\text{ µs}$** (**388.9x faster**) | **PASSED** |
| **Numerical Equivalence (Gate 1.4)** | Relative L2 error $< 10^{-5}$ | Baseline clone | **$0.00\text{e}+00$** (exact bit match) | **PASSED** |

---

## Phase 2: Architecture & Implementation

### 1. Pre-allocated State Ring Buffer (`src/gnn_experiment/state_ring_buffer.py`)
- Pre-allocates $N$ slots ($N = K + 2$, default $N=8$) in GPU VRAM for the 24 GatedDeltaNet SSM layers:
  - `recurrent_states`: `[N, batch_size, 4, 128, 128]` in `bfloat16`.
  - `conv_states`: `[N, batch_size, 2048, 4]` in `bfloat16`.
- **Zero Dynamic Allocations:** `push(cache)` performs non-blocking GPU `.copy_()` into slot `write_ptr` with zero kernel launch allocations.
- **Dynamic Attention KV Cropping:** Tracks sequence length checkpoints and crops attention `keys` and `values` in $O(1)$ without memory thrashing.
- **Pointer Arithmetic:** Advances `write_ptr` and `commit_ptr` via modulo arithmetic `(commit_ptr + n_accepted) % max_depth`.

### 2. Transparent Cache Integration (`src/gnn_experiment/mtp_draft.py`)
- `attach_state_ring_buffer(cache, max_depth=8)`: Attaches the ring buffer to the active cache.
- `snapshot_state(cache)`: Automatically detects attached ring buffer and delegates to zero-allocation slot push.
- `restore_state(cache, snap)`: Restores hybrid states in-place via ring buffer rollback and attention KV cache cropping.

---

## Phase 3: Verification & End-to-End Results

### Unit Tests
Ran `pytest tests/ -v`:
- `tests/test_state_ring_buffer.py`: **5/5 passed** (1.20s).
- Full repository test suite: **19/19 passed** (5.98s).

### End-to-End Speculative Generation (`evaluate_ring_buffer_replay.py`)
Model: **Qwen3.5-4B** (bf16) on **AMD Radeon RX 7900 XTX** (24 GB VRAM)

```
================================================================================
ReplaySSM Ring-Buffered Hybrid State Rollback: E2E Generation Verification
================================================================================
--- Prompt 1/3: Python Levenshtein Memoization ---
  Arm A (Baseline Speculative):    2.538s (18.9 tok/s) | Accept Rate: 47.1%
  Arm B (RingBuffer Speculative):  1.444s (33.3 tok/s) | Accept Rate: 47.1%
  Exact Text Match:                ✅ MATCH (+43.12% speedup)

--- Prompt 2/3: SSM vs Attention Architecture ---
  Arm A (Baseline Speculative):    1.140s (42.1 tok/s) | Accept Rate: 67.3%
  Arm B (RingBuffer Speculative):  1.022s (47.0 tok/s) | Accept Rate: 67.3%
  Exact Text Match:                ✅ MATCH (+10.29% speedup)

--- Prompt 3/3: PostgreSQL Rolling Window ---
  Arm A (Baseline Speculative):    1.319s (36.4 tok/s) | Accept Rate: 50.0%
  Arm B (RingBuffer Speculative):  1.333s (36.0 tok/s) | Accept Rate: 50.0%
  Exact Text Match:                ✅ MATCH (-1.06% latency change)

================================================================================
E2E SPECULATIVE VERIFICATION SUMMARY
================================================================================
  Text Match Across All Prompts:   ✅ 100% EXACT MATCH
  Mean Speculative Speedup:        +17.45%
================================================================================
```

---

# ⚠️ CORRECTION (2026-08-18) — re-measured with warmup and repeats

The E2E numbers above have **no warmup pass** — `evaluate_ring_buffer_replay.py`
entered the timed loop directly, so Arm A of prompt 1 absorbed lazy Triton/HIP
compilation while Arm B ran warm.

The tell was already in the original table: **Arm A read 18.9 tok/s on prompt 1
and 42.1 tok/s on prompt 2** — a 2.2× spread on an identical model and identical
decode path. That is warmup, not prompts.

This repo has the same scar on record: the retracted *"prefill = 54%"* claim was
~33 s of Triton JIT inside the timed region.

## Re-run: warmup + 3 interleaved repeats, median reported

| prompt | original | corrected | arm spreads |
| :--- | ---: | ---: | :--- |
| 1 | +43.12% | **+15.28%** | A 1.348–1.797s vs B 1.256–1.981s — **overlap** |
| 2 | +10.29% | **+7.45%** | **overlap** |
| 3 | −1.06% | **+6.00%** | **overlap** |
| **mean** | **+17.45%** | **+9.58%** | |

**Every prompt's arm spreads overlap — none of these differences is resolved at
n=3.** On prompt 1, Arm B's slowest run (1.981 s) is slower than Arm A's slowest
(1.797 s).

## The mechanism cannot deliver the residual either

Rollback saves 498 µs (499.61 → 1.28). At ~13 rollbacks per 48-token generation
that is ~6.5 ms of a ~1300 ms run = **~0.5%**. The residual +9.58% is **20× what
the mechanism can produce**, so it is measuring machine variance.

**Residual design flaw:** Arm A always runs first within each repeat pair. The
FlashNorm re-run measured this exact bias independently — re-running an *identical*
stock arm in second position gained **+1.32%** purely from position. Arms should
alternate order.

## Two factual errors

- **"12 Full-Attention layers"** — the config says
  `{'linear_attention': 24, 'full_attention': 8}`. It is **8** (32 layers total).
- **"`recurrent_states` … in `bfloat16`"** — the 51.0 MB footprint only works in
  **fp32** (`24 × 8 × 4×128×128 × 4B = 50.3 MB`; bf16 would be 25.2 MB). fla
  allocates fp32 states. The MB figure is right; the dtype label is wrong.

## What stands

- **100% exact text match** across all prompts, and **accept rates identical
  between arms** (47.1 / 67.3 / 50.0) — strong evidence the ring buffer is
  functionally correct.
- Bit-exact restoration (`0.00e+00`), 51 MB footprint, 5/5 and 19/19 tests green.
- The 1.28 µs rollback is real.

**The implementation is correct. What it replaces was never a meaningful share of
the loop, so there is no measurable end-to-end gain to claim.**

---

# REFRAMING (2026-08-18) — what this primitive is actually for

The correction above refutes the **speedup claim**. It does not argue for deleting
the code, and those are separate questions worth keeping separate.

**A primitive's value is not only its direct speedup.** Judging ReplaySSM solely
on linear speculative decoding — where its ceiling is ~0.5% — is near-sighted.
Zero-allocation, bit-exact, O(1) state rollback is infrastructure for generation
strategies that are otherwise too expensive to attempt at all. That argument is
sound, and this repo should hold it alongside the refuted number rather than
letting one erase the other.

But **"unlocks X" is a claim like any other, and claims get measured here.** Three
were made; each now has a benchmark, including the one predicted to fail
([`benchmark_branching_primitives.py`](../benchmarks/runtime/speculative/state_replay/benchmark_branching_primitives.py)).

## The asymmetry that decides all three — MEASURED, and it runs the opposite way

Qwen3.5 is hybrid: **24 GatedDeltaNet (SSM) + 8 full-attention** layers. Measured
geometry:

| | measured | branchable today? |
| :--- | ---: | :--- |
| **SSM state** | **51.90 MB per checkpoint** (fixed, independent of length) | **YES** — full copies in pre-allocated slots |
| **Attention KV** | **32.77 KB per token** (grows) | **NO** — only an integer `seq_lens[slot]` is kept; restore CROPS |

⚠️ **An earlier draft of this analysis (including my own review) asserted that
attention KV was the expensive half blocking trees. That is wrong.** The crossover
is **~1,584 tokens**: one SSM checkpoint costs as much as 1,584 tokens of KV.
Speculative trees span 8–32 tokens, so in this regime **SSM state is 50–200×
more expensive than KV**.

| tree shape | live nodes | KV needed | **SSM needed** |
| :--- | ---: | ---: | ---: |
| width 2 × depth 4 | 8 | 0.26 MB | **415 MB** |
| width 4 × depth 4 | 16 | 0.52 MB | **830 MB** |
| width 4 × depth 8 | 32 | 1.05 MB | **1,661 MB** |

**Consequence, and it is encouraging:** a width-2 × depth-4 tree needs exactly
**8 live nodes — which `max_depth=8` already holds.** The missing piece is KV
forking, worth **0.26 MB**. Tree speculation at modest width is a small piece of
plumbing away, not a memory-infrastructure project.

**Also corrects the footprint above:** the ring buffer is **415.2 MB (1.7% of
24 GB)**, not the 51.0 MB / 0.208% claimed — 8× understated. Still inside the 5%
gate, so that verdict stands, but the number does not.

## Claim-by-claim

**② Time-travel steering — SUPPORTED.** Rewind to a checkpoint, inject a
correction, resume. Purely linear: the abandoned future is never revisited, so
destructive cropping is harmless. Bounded to `max_depth=8` checkpoints.

**③ Local beam search — PARTIAL.** Branch A, rewind, branch B, keep the better.
Free when B wins (already resident). When **A** wins, its KV was cropped away and A
must be regenerated. Real, but not the "instant rollback, select optimal path" the
summary above implies.

**① Speculative trees — NOT SUPPORTED TODAY, but cheaper to fix than assumed.**
Requires multiple branches alive at once. The ring buffer retains **zero KV
bytes** (verified: it stores `list`s of ints per attention layer), so branch A's
entries are freed the moment branch B is explored. The fix is KV forking — and at
0.26 MB for a width-2 × depth-4 tree that is plumbing, not infrastructure. The
binding constraint is instead **SSM slots at 51.90 MB each**: `max_depth=8` holds
8 nodes (415 MB), enough for width-2 × depth-4 and nothing wider.

## The general lesson worth keeping

Small direct gains can unlock disproportionate capability, and a repo that only
ever asks "how much faster?" will discard its own foundations. **The correct
response is not to lower the evidence bar for capability claims — it is to write
the benchmark that tests them.** Here that produced a sharper result than either
"keep it, it's strategic" or "cut it, it's 0.5%": two of three claims hold, the
third is half-built, and the missing half is now specified.
