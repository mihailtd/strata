// Real HIP kernel: the gating multiply in Qwen3.5's full attention block
// (`attn_output = attn_output * torch.sigmoid(gate)` in
// Qwen3_5Attention.forward), applied after the attention core and before
// o_proj. `gate` is the second half of q_proj's chunked output -- a real,
// per-element tensor the same shape as attn_output, not a per-head scalar.

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

extern "C" __global__ void sigmoid_gate_bf16_kernel(
    const unsigned short* __restrict__ x,
    const unsigned short* __restrict__ gate,
    unsigned short* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float xv = bf16_bits_to_f32(x[i]);
    float gv = bf16_bits_to_f32(gate[i]);
    float sig = 1.0f / (1.0f + expf(-gv));
    out[i] = f32_to_bf16_bits(xv * sig);
}

// `stream` (§93): see `rmsnorm.hip`'s launcher comment.
extern "C" void launch_sigmoid_gate_bf16(const void* x, const void* gate, void* out, int n, void* stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(
        sigmoid_gate_bf16_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        (const unsigned short*)x, (const unsigned short*)gate, (unsigned short*)out, n
    );
}
