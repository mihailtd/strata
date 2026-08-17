# Performance & Hardware Frontiers

Kernel-level optimizations, throughput characterization, and hardware scaling frontiers on AMD ROCm (gfx1100).

| Submodule | Tier | Key Contribution |
| :--- | :---: | :--- |
| **[`prefill_vs_decode/`](prefill_vs_decode/)** | **⭐ Standard** / **🔥 Applied** | **Prefill vs. Decode Latency Share**: Proves prompt prefill is only **5.9% of wall time at 8k context** while decode constitutes **94.1%**, proving that decode acceleration (speculation, folding) controls >94% of user latency. |
| **[`fla_triton_kernels/`](fla_triton_kernels/)** | **⭐ Standard** / **🔥 Applied** | **Fused Triton Attention Kernels**: `flash-linear-attention` Triton kernels bound and verified active on gfx1100 (`fla_active: true`), flattening $K=4$ verification from 2.84x down to 1.19x and unlocking real speculative speedup. |
| **[`batch_scaling/`](batch_scaling/)** | **⭐ Industry Standard** | **High-Batch Scaling Frontier ($B=1 \dots 64$)**: Probes batch scaling up to $B=64$: aggregate decode throughput rises **17.52x** (34.87 → **611.04 tok/s**), which is **27.4% of the perfect 64x**, while per-request throughput drops **73%** to 9.55 tok/s. Speculation's break-even $\tau$ **rises** 1.15 → 1.45 (margin erodes, does not stay flat), and the weight-folding win decays 1.86x → 1.21x. |
