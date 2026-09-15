// Real HIP kernel: elementwise residual add (`hidden_states = residual +
// hidden_states`), used twice per decoder layer (after the token-mixer
// block and after the MLP block) in Qwen3_5DecoderLayer.forward. Trivial,
// but not yet built until the full forward pass needed it.

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

extern "C" __global__ void add_bf16_kernel(
    const unsigned short* __restrict__ a,
    const unsigned short* __restrict__ b,
    unsigned short* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = f32_to_bf16_bits(bf16_bits_to_f32(a[i]) + bf16_bits_to_f32(b[i]));
}

// `stream` (§93): see `rmsnorm.hip`'s launcher comment.
extern "C" void launch_add_bf16(const void* a, const void* b, void* out, int n, void* stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(
        add_bf16_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        (const unsigned short*)a, (const unsigned short*)b, (unsigned short*)out, n
    );
}
