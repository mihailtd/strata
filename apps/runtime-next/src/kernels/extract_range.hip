// Real HIP kernel: §96 (batched prefill). Copies a fixed sub-range of each
// row out of a STRIDED [rows, src_stride] buffer into a TIGHTLY PACKED
// [rows, len] destination buffer.
//
// Why this is needed and didn't exist before: every "§92 combined GEMM"
// fusion in this crate (qkv_proj, gdn's in_proj, gate_up_proj) packs
// several logically-separate sub-tensors (q|k|v, or qkv|z|b|a, or
// gate|up) back-to-back inside ONE wide output row. At the single-token
// decode rows=1 case this crate has used until now, "a pointer offset into
// row 0" and "a fresh tightly-packed buffer" are the exact same bytes --
// there is only one row, so no real stride vs. no stride is indistinguishable,
// and `DeviceBuffer::as_device_ptr_at` (a plain pointer offset, no kernel)
// was always enough (see `model.rs`'s `attn_layer_forward`/`gdn_layer_forward`
// comments on this). Batched prefill breaks that equivalence: row t's q|k|v
// (say) lives at `t*src_stride + 0`, but a caller that wants a real
// TIGHTLY-PACKED `[T, q_len]` view (so it can hand it to any of this crate's
// existing per-row kernels -- rmsnorm/rope/split_last_dim/swiglu -- all of
// which assume row stride == row length) needs the bytes physically
// gathered, not just re-pointed. That's exactly what this kernel does, and
// nothing else -- a straight strided-to-packed copy, no math.
//
// One block per row, `len` threads (strided if len > blockDim.x).

#include <hip/hip_runtime.h>

extern "C" __global__ void extract_range_bf16_kernel(
    const unsigned short* __restrict__ src,  // [rows, src_stride]
    unsigned short* __restrict__ dst,         // [rows, len]
    int rows,
    int src_stride,
    int src_offset,
    int len
) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const unsigned short* srow = src + (size_t)row * (size_t)src_stride + (size_t)src_offset;
    unsigned short* drow = dst + (size_t)row * (size_t)len;
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        drow[i] = srow[i];
    }
}

// `stream` (§93 convention, followed here too): NULL reproduces plain
// default-stream behavior; a real stream lets this be queued back-to-back
// with the surrounding batched-prefill kernels with no host sync between
// them.
extern "C" void launch_extract_range_bf16(
    const void* src,
    void* dst,
    int rows,
    int src_stride,
    int src_offset,
    int len,
    void* stream
) {
    int threads = len < 256 ? len : 256;
    hipLaunchKernelGGL(
        extract_range_bf16_kernel,
        dim3(rows), dim3(threads), 0, (hipStream_t)stream,
        (const unsigned short*)src, (unsigned short*)dst,
        rows, src_stride, src_offset, len
    );
}
