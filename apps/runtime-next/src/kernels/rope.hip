// Real HIP kernel: partial RoPE (Rotary Position Embedding), matching
// Qwen3.5's exact formula as read directly from the real transformers
// source (transformers/models/qwen3_5/modeling_qwen3_5.py,
// Qwen3_5RotaryEmbedding.forward + rotate_half + apply_rotary_pos_emb),
// not a textbook/assumed version:
//
//   dim         = head_dim * partial_rotary_factor   (256 * 0.25 = 64 here)
//   inv_freq[j] = 1 / theta^(2j/dim),  j = 0..dim/2-1
//   angle[j]    = position * inv_freq[j]
//   cos/sin     = cat(cos(angle), cos(angle)), cat(sin(angle), sin(angle))  -- length `dim`
//   q_rot, q_pass = q[:dim], q[dim:]     (only the first `dim` of head_dim rotates)
//   q_embed[i<half]  = q_rot[i]*cos[i]         - q_rot[i+half]*sin[i]
//   q_embed[i>=half] = q_rot[i]*cos[i]         + q_rot[i-half]*sin[i]
//   out = cat(q_embed, q_pass)
//
// The model's mRoPE 3-section "recomposition" (for vision T/H/W grid
// positions) is a real part of this model but is a mathematical no-op for
// pure text inference -- all three position-id channels are identical for
// text tokens, so the recomposed frequencies equal the single-section
// result. This kernel implements the text case: one scalar `position` per
// row, not the 3-channel vision path.
//
// `attention_scaling` is 1.0 for this model's rope_type ("default"), so no
// extra multiplier is applied to cos/sin -- confirmed from
// compute_default_rope_parameters's own comment ("Unused in this type of
// RoPE"), not assumed.

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

extern "C" __global__ void rope_bf16_kernel(
    const unsigned short* __restrict__ x,
    unsigned short* __restrict__ out,
    int num_rows,
    int head_dim,
    int rotary_dim,
    float theta,
    const int* __restrict__ position_ptr
) {
    const int row = blockIdx.x;
    const int i = threadIdx.x;
    if (row >= num_rows || i >= head_dim) {
        return;
    }
    const long long position = (long long)(*position_ptr);

    const unsigned short* xrow = x + (size_t)row * (size_t)head_dim;
    unsigned short* orow = out + (size_t)row * (size_t)head_dim;

    if (i >= rotary_dim) {
        // Pass-through: untouched by RoPE, matching q_pass in apply_rotary_pos_emb.
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

// Plain C entry point. `num_rows` independent vectors (e.g. batch*num_heads
// for a single decode step, all sharing the same `position`), `head_dim`
// elements per row, only the first `rotary_dim` of each row are rotated.
//
// `position` is a DEVICE pointer to a single `int`, not a host value --
// so the launch's recorded parameters (pointer, unchanging) stay valid
// across a HIP Graph replay even though the position it points to advances
// every decode step; only the pointee needs to change between replays, not
// the launch itself. Passing `kv_stride == kv_len`-style host values here
// would bake the position into the graph at capture time, wrong for every
// subsequent token (§93).
extern "C" void launch_rope_bf16(
    const void* x,
    void* out,
    int num_rows,
    int head_dim,
    int rotary_dim,
    float theta,
    const int* position,
    void* stream
) {
    hipLaunchKernelGGL(
        rope_bf16_kernel,
        dim3(num_rows), dim3(head_dim), 0, (hipStream_t)stream,
        (const unsigned short*)x, (unsigned short*)out,
        num_rows, head_dim, rotary_dim, theta, position
    );
}
