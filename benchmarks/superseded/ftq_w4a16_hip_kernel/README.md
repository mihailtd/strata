# FTQ W4A16 HIP Kernel — SUPERSEDED

**Not retired for being wrong or slow — it worked, and its numbers below are real.**
Superseded because the repo moved to a Triton implementation of the same idea instead of
a raw HIP C++ one. Kept for provenance.

---

## What this is

An early ("FTQ" — Fast Token Quantization) exploration of fused 4-bit weight
dequantization on `gfx1100`, done on Qwen3.5-0.8B before this repo's current 27B/9B
engines existed. Three pieces:

- **`w4a16.hip`** — a hand-written HIP C++ kernel (`w4a16_matvec`). One wavefront per
  output row, cross-lane-shuffle reduction (no shared memory, no barriers), 4-bit codes
  dequantized straight from registers into the accumulator — the weight's fp16 form
  never exists in memory. Batch-1 decode only.
- **`quantizers.py`** — the Python reference for turning an fp16/bf16 weight into 4-bit
  codes + per-group scales (simulated compression, for validating the kernel against).
- **`packing.py`** — turns those codes into the dense bit-packed `uint32` layout the
  kernel above expects (`pack_bits` for power-of-two bit widths, `pack_base` for
  mixed-radix alphabets).
- **`hip_bench.txt`** — the raw benchmark capture behind the numbers below.

These three files used to live together as a package, `src/ftq/` (the kernel at
`src/ftq/csrc/w4a16.hip`) — see `../ftq_0.8b_experiments.md` (moved from the original
root `EXPERIMENTS.md`) for the full T-01–T-17 experiment log that package came from, and
`../AUDIT_HISTORY_2026-08-11.md` §"Step 1" for why it was evaluated (and rejected) for
reuse in KronA+DoRA training. That package no longer exists; these were the only three
files still sitting in the tree (`w4a16.hip` loose at the repo root, the two Python files
under a root-level `hip-kernel-experiment/`), disconnected from any build step or import.

## What it achieved

Measured on an AMD Radeon RX 7900 XTX (`gfx1100`, 24 GiB), against Qwen3.5-0.8B's own
shapes, fp16 baseline vs. this kernel:

| shape | K × N | fp16 | HIP | speedup |
| :--- | ---: | ---: | ---: | ---: |
| 0.8B qkv | 1024×4096 | 0.040 ms | 0.016 ms | 2.48× |
| 0.8B mlp up | 1024×3584 | 0.038 ms | 0.021 ms | 1.78× |
| 0.8B mlp down | 3584×1024 | 0.058 ms | 0.019 ms | 3.13× |
| 9B qkv | 4096×4096 | 0.083 ms | 0.032 ms | 2.59× |
| 9B mlp up | 4096×12288 | 0.148 ms | 0.050 ms | 2.94× |
| 9B mlp down | 12288×4096 | 0.159 ms | 0.052 ms | 3.03× |

Summed: **2.75× over fp16**. `../ftq_0.8b_experiments.md` additionally cites 4.6× on the
`lm_head` GEMV specifically (where a generic GEMM library picks a bad tile shape for a
1×N vector-matrix product) and 20–26× over the naive "unpack to fp16 then GEMM" Python
path that motivated writing a kernel at all.

## Known issues — this was already dead code, not just superseded

- **Never wired into a build.** No `hipcc` invocation, `setup.py`, or CMake target
  anywhere in this repo compiles `w4a16.hip`. It's uncompiled source.
- **`quantizers.py` has a broken import** (`from . import grids`) — `grids.py` never
  existed anywhere in this repo's git history. This package was a partial copy from
  wherever the original FTQ work lived, not a self-contained module here.
- **The kernel's own docstring references `tests/test_hip_kernel.py`** ("pinned against
  the Python reference by...") — that test doesn't exist in this repo either.
- **It targets a Windows DLL, not a Linux `.so`.** `w4a16_matvec_launch` is declared
  `extern "C" __declspec(dllexport)`, with a comment noting clang warns the attribute is
  unsupported but keeping it anyway because dropping it made `ctypes` report the symbol
  as not found. `__declspec(dllexport)` is an MSVC/MinGW mechanism for marking a symbol
  exported from a Windows DLL; it's meaningless on a Linux ELF shared object, where
  symbols are exported by default. That's a genuine artifact of this kernel having been
  built and loaded as a Windows DLL at some point — unlike the repo's current native-Linux
  ROCm stack (`SYSTEM.md`'s "WSL-specific DXCore shims... completely eliminated"), this
  file predates that and was never touched during the migration.

## What replaced it

[`apps/runtime-triton/triton_w4a16.py`](../../../apps/runtime-triton/triton_w4a16.py) —
the same core idea (4-bit packed weights, group-wise scaling, register-level fused
dequantization) reimplemented in **Triton** instead of hand-written HIP C++, feeding
16×16×16 WMMA hardware tiles directly. Two concrete improvements over this kernel, beyond
just being written in a higher-level language with no separate build step to wire up:

1. It also fuses the active LoRA adapter branch into the same kernel
   (`Out = X @ dequant(W) + alpha * (X @ L_A) @ L_B`) — this kernel only ever did the base
   weight GEMV.
2. It's actually shipped and live, vendored into `apps/runtime-triton` for
   self-sufficiency (see that file's own docstring).

Separately, `../AUDIT_HISTORY_2026-08-11.md` found this kernel **cannot be reused** for
KronA+DoRA training regardless: it's a forward-only, single-token decode GEMV, while
training needs the Kronecker-product forward pass, the DoRA magnitude norm, and backward
autograd through all of it — a different kernel entirely, never written.

## Reproduce (for provenance only)

There's nothing to run — the kernel was never compiled in this repo, so `hip_bench.txt`'s
numbers can't be regenerated from this tree as-is. They're preserved as historical
evidence that the register-dequant approach works on this hardware, which is what
motivated writing `triton_w4a16.py` the way it's written today.
