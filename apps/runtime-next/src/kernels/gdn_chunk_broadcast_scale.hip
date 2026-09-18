// Real HIP kernel: §100 (chunked GDN parallel prefill). One small, general
// "GQA-broadcast-then-scale" step, reused for every real per-row multiply
// this pipeline needs after `l2norm_bf16`/`gdn_chunk_decay_bf16`:
//
//   dst[...] = src[token, h/n_rep, :] * scale[token, h]
//
// `n_rep=1` degenerates to a plain per-row scale with NO broadcast (src
// and dst share the same head count) -- covers every real use in this
// pipeline with one kernel:
//   - `k_beta[h,:]           = key_normalized[h/n_rep,:] * beta[h]`        (n_rep=2, real GQA broadcast)
//   - `q_final[h,:]          = query_normalized[h/n_rep,:] * decay_exp[h]` (n_rep=2)
//   - `k_final[h,:]          = key_normalized[h/n_rep,:] * remaining_decay[h]` (n_rep=2)
//   - `v_beta[h,:]           = value[h,:] * beta[h]`                       (n_rep=1, plain scale)
//   - `decayed_k_beta[h,:]   = k_beta[h,:] * decay_exp[h]`                 (n_rep=1, plain scale)
//
// The SOURCE (`src`, `scale`) is always read TOKEN-major (matching how
// query/key/value/beta/decay_exp are all naturally laid out). The
// DESTINATION layout is chosen by `head_major_output`:
//   - 0: token-major `[num_tokens, dst_heads, head_dim]` -- e.g. `k_beta`,
//     which is read back both as a strided view (for `gemm_qkt_bf16`'s Q
//     operand, no physical extraction needed there) AND as the SOURCE for
//     a SECOND call producing `decayed_k_beta`.
//   - 1: head-major `[dst_heads, num_chunks, chunk_size, head_dim]` -- the
//     layout `gemm_qkt_bf16`/`gemm_atb_bf16`'s per-(head,chunk) contiguous
//     slicing and `gdn_chunk_utsolve_bf16`'s own RHS contract need (e.g.
//     `v_beta`, `decayed_k_beta`, the final rescaled `query`/`key` used by
//     the inter-chunk scan).
// Doing the layout choice HERE (a cheap, already-strided elementwise copy)
// avoids a separate transpose kernel, or -- worse -- loosening
// `gdn_chunk_utsolve_bf16`'s already-proven tightly-packed contract to
// accept arbitrary strides.

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

extern "C" __global__ void gdn_chunk_broadcast_scale_bf16_kernel(
    const unsigned short* __restrict__ src,  // [num_tokens, src_heads, head_dim], token-major
    const float* __restrict__ scale,          // [num_tokens, dst_heads], token-major
    unsigned short* __restrict__ dst,          // see `head_major_output`
    int num_tokens,
    int src_heads,
    int dst_heads,
    int head_dim,
    int n_rep,
    int chunk_size,
    int head_major_output
) {
    const int token = blockIdx.x;
    const int dst_head = blockIdx.y;
    if (token >= num_tokens || dst_head >= dst_heads) return;
    const int src_head = dst_head / n_rep;

    const unsigned short* src_row = src + ((size_t)token * src_heads + src_head) * (size_t)head_dim;
    unsigned short* dst_row;
    if (head_major_output) {
        const int num_chunks = num_tokens / chunk_size;
        const int chunk = token / chunk_size;
        const int pos = token % chunk_size;
        dst_row = dst + (((size_t)dst_head * num_chunks + chunk) * chunk_size + pos) * (size_t)head_dim;
    } else {
        dst_row = dst + ((size_t)token * dst_heads + dst_head) * (size_t)head_dim;
    }
    const float s = scale[(size_t)token * dst_heads + dst_head];

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        dst_row[d] = f32_to_bf16_bits(bf16_bits_to_f32(src_row[d]) * s);
    }
}

extern "C" void launch_gdn_chunk_broadcast_scale_bf16(
    const void* src,
    const float* scale,
    void* dst,
    int num_tokens,
    int src_heads,
    int dst_heads,
    int head_dim,
    int n_rep,
    int chunk_size,
    int head_major_output,
    int threads,
    void* stream
) {
    hipLaunchKernelGGL(
        gdn_chunk_broadcast_scale_bf16_kernel,
        dim3(num_tokens, dst_heads), dim3(threads), 0, (hipStream_t)stream,
        (const unsigned short*)src, scale, (unsigned short*)dst,
        num_tokens, src_heads, dst_heads, head_dim, n_rep, chunk_size, head_major_output
    );
}
