// Real HIP kernel: §103 (real batched-prefill RoPE, eliminating the
// per-token `rope` loop `attn_layer_forward_prefill` used until now).
// Identical real math to `rope.hip`'s own per-token kernel (see that
// file's header for the exact formula, read directly from the real
// transformers source) -- the ONLY real change is that `position` is now
// looked up PER TOKEN from a real `position_buf[num_tokens]` array
// (`token = row / num_heads`) instead of being one shared scalar for the
// whole launch. RoPE has no cross-token dependency at all (each token's
// rotation depends only on its own absolute position, already known
// before this kernel runs), so this is an exact, not approximate,
// parallel restatement -- the per-token loop existed only because the
// OLD kernel's single `position_ptr` couldn't vary per row within one
// launch.

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

extern "C" __global__ void rope_prefill_bf16_kernel(
    const unsigned short* __restrict__ x,   // [num_tokens*num_heads, head_dim], tightly packed
    unsigned short* __restrict__ out,
    int num_tokens,
    int num_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    const int* __restrict__ position_buf     // [num_tokens]
) {
    const int row = blockIdx.x;
    const int i = threadIdx.x;
    const int total_rows = num_tokens * num_heads;
    if (row >= total_rows || i >= head_dim) {
        return;
    }
    const int token = row / num_heads;
    const long long position = (long long)position_buf[token];

    const unsigned short* xrow = x + (size_t)row * (size_t)head_dim;
    unsigned short* orow = out + (size_t)row * (size_t)head_dim;

    if (i >= rotary_dim) {
        orow[i] = xrow[i];
        return;
    }

    const int half = rotary_dim / 2;
    const int freq_idx = i % half;
    const float inv_freq = powf(theta, -((float)(2 * freq_idx)) / (float)rotary_dim);
    const float angle = (float)position * inv_freq;
    const float c = cosf(angle);
    const float s = sinf(angle);

    float out_val;
    if (i < half) {
        float x_i = bf16_bits_to_f32(xrow[i]);
        float x_pair = bf16_bits_to_f32(xrow[i + half]);
        out_val = x_i * c - x_pair * s;
    } else {
        float x_i = bf16_bits_to_f32(xrow[i]);
        float x_pair = bf16_bits_to_f32(xrow[i - half]);
        out_val = x_i * c + x_pair * s;
    }
    orow[i] = f32_to_bf16_bits(out_val);
}

// `position_buf` is a real DEVICE array of `num_tokens` real absolute
// positions (`state.position + i` for `i` in the chunk) -- ONE launch
// covers the whole chunk instead of `num_tokens` separate
// `launch_rope_bf16` calls.
extern "C" void launch_rope_prefill_bf16(
    const void* x,
    void* out,
    int num_tokens,
    int num_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    const int* position_buf,
    void* stream
) {
    hipLaunchKernelGGL(
        rope_prefill_bf16_kernel,
        dim3(num_tokens * num_heads), dim3(head_dim), 0, (hipStream_t)stream,
        (const unsigned short*)x, (unsigned short*)out,
        num_tokens, num_heads, head_dim, rotary_dim, theta, position_buf
    );
}
