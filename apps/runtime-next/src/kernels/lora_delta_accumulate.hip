// Real HIP kernel: quantized-path LoRA -- the additive counterpart of
// `lora.rs`'s bf16 in-place weight folding (§100). See
// `apps/runtime-next/src/quantized_lora.rs`'s own header doc for the full
// architectural rationale (ported from the real, already-serving
// `apps/runtime-triton/w4a16_loader.py`'s `fused_w4a16_lora_matmul`
// design, adapted to this engine's simpler decode-only M=1 GEMV shape).
//
// This kernel does NOT touch `qweight`/`scales` at all -- the real W4A16
// base dequant+GEMV (`w4a16_gemv.hip`) stays completely unmodified, run
// as its own, separate, unchanged kernel launch first. This kernel only
// accumulates the small low-rank LoRA delta into that base kernel's
// already-written output, in place:
//
//   y[row] += sum_{r=0}^{rank-1} b[r, row] * mid[r]
//
// where `mid = lora_a @ x` was already computed by a SEPARATE, ordinary
// call to the existing `gemv_bf16_kernel` (`mid` is shared across every
// row this kernel touches, so computing it once outside this kernel
// avoids `out_features` redundant re-computations of the same rank-sized
// reduction). `b`'s rows beyond the real adapter's rank, and `mid`'s
// entries beyond it, are always zero (zeroed by `activate_quantized_adapter`
// via `hipMemsetAsync` on the bank prior to uploading real data -- see
// `quantized_lora.rs` §132) -- so a real rank-8 adapter and an inactive
// (all-zero) slot both flow through the exact same kernel code path, no
// branching needed.
//
// §131/§132 LAYOUT CHANGE (vs. original §128 implementation):
//
// `b` is now stored column-major: `b[rank, out_features]` (equivalently:
// `b[r, row]` = element at linear index `r * out_features + row`).
//
// Previously `b` was row-major `[out_features, rank]`. The change was made
// to enable §132's single-prefix contiguous upload: for a real rank-r
// adapter, the real data occupies rows 0..r of the `[MAX_RANK, out_feat]`
// bank buffer (a contiguous prefix of `r * out_features` u16 elements),
// and rows r..MAX_RANK are zero. A single `hipMemcpyAsync` copies the
// prefix; a preceding `hipMemsetAsync` zeroes the full bank first -- no
// per-slot calls needed. The original row-major layout had real columns at
// stride-`MAX_RANK` offsets (non-contiguous), which would have required
// N*`hipMemcpy2DAsync` calls to skip the zero-column stride.
//
// The math is unchanged: `b[r, row] * mid[r]` is the same value as the
// original `b[row, r] * mid[r]` -- only the memory layout differs.
//
// One thread per output row (not one block per row like the base GEMV
// kernel): `rank` is tiny (<=32 in this engine, see `MAX_LORA_RANK` in
// `model.rs`), so a single thread doing `rank` serial FMAs is cheap and
// avoids the block-launch/shared-memory-reduction overhead the base
// kernel's much larger per-row reduction (thousands of `in_features`
// elements) actually needs.
//
// `y` was already written by the base kernel on the SAME stream just
// before this launch -- sequential kernel execution on one stream
// guarantees the read-modify-write here is race-free, no atomics needed.

#include "bf16_utils.hip"

extern "C" __global__ void lora_delta_accumulate_kernel(
    const unsigned short* __restrict__ mid,   // [rank], bf16 bits
    const unsigned short* __restrict__ b,     // [rank, out_features], bf16 bits, col-major, alpha pre-folded
    unsigned short* __restrict__ y,           // [out_features], bf16 bits, accumulated in place
    int out_features,
    int rank
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_features) {
        return;
    }
    float acc = bf16_bits_to_f32(y[row]);
    // b is [rank, out_features] col-major: b[r, row] = b[r * out_features + row]
    for (int r = 0; r < rank; r++) {
        acc += bf16_bits_to_f32(b[(size_t)r * (size_t)out_features + (size_t)row])
             * bf16_bits_to_f32(mid[r]);
    }
    y[row] = f32_to_bf16_bits(acc);
}

// Plain C entry point. `stream`: NULL reproduces default-stream behavior;
// a real explicitly-created stream is required for capture into a HIP
// Graph (matching every other kernel launcher in this crate).
extern "C" void launch_lora_delta_accumulate_bf16(
    const void* mid,
    const void* b,
    void* y,
    int out_features,
    int rank,
    void* stream
) {
    const int threads = 256;
    const int blocks = (out_features + threads - 1) / threads;
    hipLaunchKernelGGL(
        lora_delta_accumulate_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        (const unsigned short*)mid, (const unsigned short*)b, (unsigned short*)y,
        out_features, rank
    );
}
