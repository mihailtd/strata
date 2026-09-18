// Real HIP kernel: §100 (chunked GDN parallel prefill). Pure L2
// normalization, matching transformers' own real `l2norm()` function
// EXACTLY (modeling_qwen3_5.py:294-297):
//
//   out = x * rsqrt(sum(x^2, dim=-1) + eps)
//
// Deliberately NOT `rmsnorm.hip` (which computes `mean(x^2)`, not
// `sum(x^2)`, and multiplies by a learned `(1+weight)` factor) -- this is
// a genuinely different real formula, used by GatedDeltaNet's own
// `use_qk_l2norm_in_kernel=True` path for both the single-token decode
// kernel (`gdn_recurrent_decode_bf16_kernel`'s own inline Phase 1) and,
// now, the chunked prefill path. An optional `post_scale` multiplies the
// normalized result once more (real usage: query's own extra
// `1/sqrt(head_dim)` attention-style scaling, applied once here instead
// of via a separate elementwise pass -- `post_scale=1.0` for key, which
// has no such extra scaling in the real reference).
//
// One block per row, `hidden_size` threads (or fewer, strided) -- same
// block-level reduction idiom `rmsnorm.hip` already established.

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

extern "C" __global__ void l2norm_bf16_kernel(
    const unsigned short* __restrict__ x,
    unsigned short* __restrict__ out,
    int num_rows,
    int hidden_size,
    float eps,
    float post_scale
) {
    extern __shared__ float reduce_scratch[];

    const int row = blockIdx.x;
    if (row >= num_rows) return;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const unsigned short* xrow = x + (size_t)row * (size_t)hidden_size;
    unsigned short* orow = out + (size_t)row * (size_t)hidden_size;

    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += stride) {
        float v = bf16_bits_to_f32(xrow[i]);
        local_sum += v * v;
    }
    reduce_scratch[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            reduce_scratch[tid] += reduce_scratch[tid + s];
        }
        __syncthreads();
    }
    const float inv_norm = rsqrtf(reduce_scratch[0] + eps);
    __syncthreads();

    for (int i = tid; i < hidden_size; i += stride) {
        float v = bf16_bits_to_f32(xrow[i]);
        orow[i] = f32_to_bf16_bits(v * inv_norm * post_scale);
    }
}

// Plain C entry point. `threads` must be a power of two; shared memory
// sized for `threads` floats.
extern "C" void launch_l2norm_bf16(
    const void* x,
    void* out,
    int num_rows,
    int hidden_size,
    float eps,
    float post_scale,
    int threads,
    void* stream
) {
    size_t shmem_bytes = (size_t)threads * sizeof(float);
    hipLaunchKernelGGL(
        l2norm_bf16_kernel,
        dim3(num_rows), dim3(threads), shmem_bytes, (hipStream_t)stream,
        (const unsigned short*)x, (unsigned short*)out, num_rows, hidden_size, eps, post_scale
    );
}
