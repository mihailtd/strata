// Real HIP kernel: the SwiGLU activation used in this model's MLP block
// (down_proj(silu(gate_proj(x)) * up_proj(x)) -- this kernel is the
// silu(gate) * up part, taking the already-projected gate/up tensors;
// gate_proj/up_proj themselves are GEMMs, a separate feature).
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x)), computed in fp32
// regardless of bf16 input/output, matching this project's own established
// convention (ExactRMSNorm, rmsnorm.hip) of doing elementwise math in fp32
// and only rounding to bf16 at the boundary.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float bf16_bits_to_f32(unsigned short bits) {
    return __uint_as_float(((unsigned int)bits) << 16);
}

__device__ __forceinline__ unsigned short f32_to_bf16_bits(float v) {
    unsigned int bits = __float_as_uint(v);
    unsigned int lsb = (bits >> 16) & 1u;
    unsigned int rounded = bits + 0x7fffu + lsb;
    return (unsigned short)(rounded >> 16);
}

extern "C" __global__ void swiglu_bf16_kernel(
    const unsigned short* __restrict__ gate,
    const unsigned short* __restrict__ up,
    unsigned short* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float g = bf16_bits_to_f32(gate[i]);
    float u = bf16_bits_to_f32(up[i]);
    float silu_g = g / (1.0f + expf(-g));
    out[i] = f32_to_bf16_bits(silu_g * u);
}

// Plain C entry point. `n` total elements (e.g. num_rows * intermediate_size).
// `stream` (§93): see `rmsnorm.hip`'s launcher comment.
extern "C" void launch_swiglu_bf16(const void* gate, const void* up, void* out, int n, void* stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(
        swiglu_bf16_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        (const unsigned short*)gate, (const unsigned short*)up, (unsigned short*)out, n
    );
}

// §114: real launch-count-reduction fusion for batched prefill's MLP
// block. Before this, `mlp_block_prefill` read `gate|up`'s real combined
// GEMM output (one wide `[T, GATE_UP_COMBINED_DIM]` row per token, gate
// and up sub-ranges back-to-back) via TWO real `extract_range_bf16`
// launches (one per sub-range, physically gathering each into its own
// tightly-packed buffer -- see that kernel's own header for why the
// gather was needed at all) before a THIRD launch (`swiglu_bf16` above)
// combined them. Real rocprofv3 profiling on a served prefill request
// found `extract_range_bf16_kernel` at 138 real calls -- a real,
// eliminable launch-count cost, not analyzed compute (this kernel does no
// arithmetic at all, pure strided-to-packed copies) contributing to §111's
// own real finding that prefill's real wall-clock time is dominated by
// per-launch dispatch overhead, not GPU compute.
//
// This kernel reads gate/up DIRECTLY from the real strided combined-GEMM
// output (two offsets into the same wide row) and writes `silu(gate)*up`
// straight to the final tightly-packed destination -- the exact same
// reads and the exact same arithmetic as the extract_range+extract_range+
// swiglu sequence it replaces, just fused into ONE real kernel launch
// instead of three. Not an approximation of any kind -- a pure algebraic
// identity (reading via stride+offset instead of via a pre-copied buffer
// changes none of the real bytes read or computed), so real output is
// expected to be BIT-EXACT identical to the unfused sequence, not merely
// "close."
extern "C" __global__ void swiglu_strided_bf16_kernel(
    const unsigned short* __restrict__ gate_up, // [rows, src_stride]
    unsigned short* __restrict__ out,            // [rows, len], tightly packed
    int rows,
    int src_stride,
    int gate_offset,
    int up_offset,
    int len
) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const unsigned short* grow = gate_up + (size_t)row * (size_t)src_stride + (size_t)gate_offset;
    const unsigned short* urow = gate_up + (size_t)row * (size_t)src_stride + (size_t)up_offset;
    unsigned short* orow = out + (size_t)row * (size_t)len;
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        float g = bf16_bits_to_f32(grow[i]);
        float u = bf16_bits_to_f32(urow[i]);
        float silu_g = g / (1.0f + expf(-g));
        orow[i] = f32_to_bf16_bits(silu_g * u);
    }
}

extern "C" void launch_swiglu_strided_bf16(
    const void* gate_up,
    void* out,
    int rows,
    int src_stride,
    int gate_offset,
    int up_offset,
    int len,
    void* stream
) {
    int threads = len < 256 ? len : 256;
    hipLaunchKernelGGL(
        swiglu_strided_bf16_kernel,
        dim3(rows), dim3(threads), 0, (hipStream_t)stream,
        (const unsigned short*)gate_up, (unsigned short*)out, rows, src_stride, gate_offset, up_offset, len
    );
}
