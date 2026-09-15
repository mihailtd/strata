// Real HIP kernel: Qwen3_5RMSNormGated -- GatedDeltaNet's final norm
// (`self.norm(core_attn_out, z)` in Qwen3_5GatedDeltaNet.forward). A
// DIFFERENT formula from rmsnorm.hip's Qwen3_5RMSNorm: that one inits its
// weight to zeros and computes `x_normed * (1+weight)`; this one inits its
// weight to ONES and computes `weight * x_normed`, then multiplies by
// `silu(gate)` -- confirmed by reading Qwen3_5RMSNormGated.forward
// directly, not assumed to be the same op as the other norm.
//
//   x_normed = x.float() * rsqrt(mean(x.float()^2, dim=-1) + eps)
//   out      = (weight.float() * x_normed) * silu(gate.float())
//
// UPDATED (model-assembly pass): the real checkpoint stores this specific
// weight (`linear_attn.norm.weight`) as native f32, unlike every other
// weight in this model (all bf16) -- confirmed by reading the real
// safetensors header directly, not assumed. `weight` is therefore a plain
// `float*` here, not a bf16-bit-pattern buffer like every other kernel in
// this crate.

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

extern "C" __global__ void rmsnorm_gated_bf16_kernel(
    const unsigned short* __restrict__ x,
    const unsigned short* __restrict__ gate,
    const float* __restrict__ weight,
    unsigned short* __restrict__ out,
    int hidden_size,
    float eps
) {
    extern __shared__ float sdata[];
    const int row = blockIdx.x;
    const unsigned short* xrow = x + (size_t)row * (size_t)hidden_size;
    const unsigned short* grow = gate + (size_t)row * (size_t)hidden_size;
    unsigned short* orow = out + (size_t)row * (size_t)hidden_size;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    float sumsq = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = bf16_bits_to_f32(xrow[i]);
        sumsq += v * v;
    }
    sdata[tid] = sumsq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    const float rms = rsqrtf(sdata[0] / (float)hidden_size + eps);

    for (int i = tid; i < hidden_size; i += stride) {
        float xv = bf16_bits_to_f32(xrow[i]);
        float wv = weight[i];
        float gv = bf16_bits_to_f32(grow[i]);
        float silu_g = gv / (1.0f + expf(-gv));
        float outv = (wv * (xv * rms)) * silu_g;
        orow[i] = f32_to_bf16_bits(outv);
    }
}

// `stream` (§93): see `rmsnorm.hip`'s launcher comment.
extern "C" void launch_rmsnorm_gated_bf16(
    const void* x,
    const void* gate,
    const float* weight,
    void* out,
    int num_rows,
    int hidden_size,
    float eps,
    int threads,
    void* stream
) {
    size_t shmem_bytes = (size_t)threads * sizeof(float);
    hipLaunchKernelGGL(
        rmsnorm_gated_bf16_kernel,
        dim3(num_rows), dim3(threads), shmem_bytes, (hipStream_t)stream,
        (const unsigned short*)x, (const unsigned short*)gate, weight,
        (unsigned short*)out, hidden_size, eps
    );
}
