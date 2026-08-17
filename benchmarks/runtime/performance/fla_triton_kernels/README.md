# 🔥 Fused FLA Triton Kernels on AMD ROCm (`gfx1100`)

> **Tier Classification**: **⭐ Industry Standard** (Triton Kernels) / **🔥 Applied Practice** (Hardware Port & Verification)  
> **Concept Origin**: **Binding specialized linear attention chunked Triton kernels (`chunk_gated_delta_rule`) on consumer AMD RDNA3 architectures (`gfx1100`).**

---

### Classification Breakdown: What is Standard vs. What is Innovative
* **⭐ Industry Standard Baseline**: Flash-Linear-Attention (`fla`) kernels are developed upstream for NVIDIA CUDA architectures to accelerate chunked GatedDeltaNet computation.
* **🔥 Our Applied Practice**: Resolving the silent CPU fallback failure mode on AMD ROCm 7.2 / RDNA3 (`gfx1100`). By isolating `fla 0.5.2` from `causal_conv1d` and enforcing early CUDA initialization, we successfully bound the fused Triton chunked kernels, flattening the $K=4$ verification cost from $2.84\times$ down to **$1.19\times$** and unlocking speculative speedups on consumer AMD GPUs.

---

## 1. ⚠️ Critical Distinction: FLA vs. `causal_conv1d`
Historically, speculative decoding was assumed unviable on this machine because engineers conflated `fla` with `causal_conv1d`. They are two separate dependencies with independent fallbacks:
- `causal_conv1d`: Remains uninstalled. However, profiling proved its PyTorch fallback only accounts for ~3.8% of verification wall time, making it negligible.
- `flash-linear-attention`: **Successfully installed and bound!** This accounts for 40.4% of wall time. Binding this kernel is what dropped the verification penalty from $2.84\times$ to $1.19\times$.

---

## 2. Benchmark Verification
With `fla` active, verification costs remain flat as $K$ draft tokens increase:
- **$K=1$**: 1.00x base cost (27.87 ms)
- **$K=4$**: **1.19x base cost (33.27 ms)**
- **$K=8$**: 1.23x base cost (34.23 ms)

This flattened verification curve moved the theoretical break-even point to $\tau \approx 1.39$, allowing the engine to achieve **up to 2.20x net speedup (55.71 tok/s at $K=6$)**.

---

## 3. Scripts
- **`profile_mtp_verification_path.py`**: The hardware profiler that explicitly asserts `M.chunk_gated_delta_rule is not None` to prove the Triton kernels are bound and actively bypassing CPU fallbacks.
