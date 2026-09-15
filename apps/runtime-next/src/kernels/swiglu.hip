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
