# ⚡ Kernel Benchmarks (RDNA3 Triton Engine)

One real sub-benchmark lives here today:

| Directory | Sub-Benchmark | Key Metric |
| :--- | :--- | :--- |
| [`w4a16_gemv_m1/`](./w4a16_gemv_m1/README.md) | 128-Bit Memory Coalescing & GEMV | $620.4\text{ GB/s}$ ($64.6\%$ bus saturation), $66.40\text{ tok/s}$ vs Ollama's $48.68\text{ tok/s}$ |

Backed by a real Triton kernel (`_w4a16_gemv_m1_kernel` in [`apps/runtime-triton/triton_w4a16.py`](../../apps/runtime-triton/triton_w4a16.py)) and a real benchmark script ([`benchmark_rdna3_gemv_coalescing.py`](./w4a16_gemv_m1/benchmark_rdna3_gemv_coalescing.py)) that loads the kernel, times it with real `torch.cuda.Event`s on a real GPU, and writes its own output.

## What used to be here

This directory previously listed five more "sub-benchmarks" — fused SwiGLU, fused QKV+RoPE, outlier-protected W4A16, static tree speculation, and entropy-adaptive tree speculation — each with a polished results table. None of them had a backing kernel or benchmark script anywhere in the repo; two directly contradicted the real, already-migrated tree-speculation result (`benchmarks/README_TREE_SPECULATION.md`, which found tree speculation **not worth it**). They were fabricated, not superseded-by-something-better, so they were removed rather than fixed — there was no real implementation to build a benchmark around. See [`benchmarks/superseded/kernel_fabricated_claims/`](../superseded/kernel_fabricated_claims/) for the retired files and the full rationale, and [`benchmarks/superseded/README.md`](../superseded/README.md) for this repo's retirement policy.

If any of those five ideas gets a real Triton implementation in the future, it earns a new directory here with a benchmark script that actually runs it — not a README describing what it would do.
