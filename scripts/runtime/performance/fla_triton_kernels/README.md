# 🔥 Fused FLA Triton Kernels on AMD ROCm (gfx1100)

This module documents and benchmarks the successful integration of Flash Linear Attention (`fla`) on an undocumented consumer hardware stack (AMD ROCm / WSL2 / gfx1100). 

## ⚠️ Important Distinction: FLA vs. causal_conv1d
Historically, speculative decoding was blocked on this stack because engineers conflated `fla` with `causal_conv1d`. They are two separate dependencies with independent fallbacks:
- `causal_conv1d`: Remains unbuildable. However, profiling proved this fallback only accounts for ~3.8% of verification wall time, so it is **not worth chasing**.
- `flash-linear-attention`: **Successfully installed and bound!** This accounts for 40.4% of wall time. Getting Triton to actually bind this kernel (rather than silently falling back) is the load-bearing dependency that made speculative decoding viable on this machine.

## Benchmark Verification
With `fla` active, verification costs remain incredibly flat as $K$ draft tokens increase:
- $K=1$: 1.00x base cost
- $K=4$: 1.19x base cost
- $K=8$: 1.23x base cost

This flattened verification curve moved the theoretical break-even point to $\tau \approx 1.39$, allowing the engine to achieve real-world end-to-end speculative speedups of **1.32x - 1.39x**.

## Scripts
- **`profile_mtp_verification_path.py`**: The hardware profiler that explicitly asserts `M.chunk_gated_delta_rule is not None` to prove the Triton kernels are bound and actively bypassing the CPU fallback.
