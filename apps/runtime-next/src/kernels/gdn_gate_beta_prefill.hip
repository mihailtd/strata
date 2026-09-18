// Real HIP kernel: §103 (real batched-prefill GDN gate/beta, eliminating
// the per-token `gdn_gate_beta` loop `gdn_layer_forward_prefill` used
// until now). Same real motivation as `causal_conv1d_prefill.hip`'s own
// header doc: `gdn_gate_beta` has NO cross-token dependency at all (every
// token's `g`/`beta` depends only on that token's own `a`/`b` plus this
// layer's real trained `a_log`/`dt_bias` parameters) -- the per-token loop
// existed only because `a_log`/`dt_bias` are `[num_heads]`-wide REAL
// per-layer weights, and calling the old kernel with
// `num_heads = num_tokens*num_heads` in one shot would have read them out
// of bounds past the first token's own range. This kernel fixes that by
// indexing `a_log`/`dt_bias` with `h = idx % num_heads` (shared across
// every token) while `a`/`b`/`g_out`/`beta_out` use the full flat
// `(token, head)` index -- one launch covers the whole chunk.
//
// `a_src`/`b_src` are read as STRIDED views (`*_stride`/`*_offset`), the
// same "zero-copy" technique `causal_conv1d_prefill.hip` uses -- lets the
// caller read `a`/`b` directly out of the wider in-proj GEMM output row,
// no physical `extract_range` pass needed first.

#include "bf16_utils.hip"

__device__ __forceinline__ float softplus_f32(float x) {
    return x > 20.0f ? x : log1pf(expf(x));
}

extern "C" __global__ void gdn_gate_beta_prefill_bf16_kernel(
    const unsigned short* __restrict__ a_src, int a_stride, int a_offset,
    const unsigned short* __restrict__ b_src, int b_stride, int b_offset,
    const float* __restrict__ a_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ g_out,     // [num_tokens, num_heads], tightly packed
    float* __restrict__ beta_out,   // [num_tokens, num_heads], tightly packed
    int num_tokens,
    int num_heads
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_tokens * num_heads;
    if (idx >= total) return;
    const int t = idx / num_heads;
    const int h = idx % num_heads;

    const float av = bf16_bits_to_f32(a_src[(size_t)t * (size_t)a_stride + (size_t)a_offset + (size_t)h]);
    const float bv = bf16_bits_to_f32(b_src[(size_t)t * (size_t)b_stride + (size_t)b_offset + (size_t)h]);
    g_out[idx] = -expf(a_log[h]) * softplus_f32(av + dt_bias[h]);
    beta_out[idx] = 1.0f / (1.0f + expf(-bv));
}

extern "C" void launch_gdn_gate_beta_prefill_bf16(
    const void* a_src, int a_stride, int a_offset,
    const void* b_src, int b_stride, int b_offset,
    const float* a_log,
    const float* dt_bias,
    float* g_out,
    float* beta_out,
    int num_tokens,
    int num_heads,
    int threads,
    void* stream
) {
    int total = num_tokens * num_heads;
    int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(
        gdn_gate_beta_prefill_bf16_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        (const unsigned short*)a_src, a_stride, a_offset,
        (const unsigned short*)b_src, b_stride, b_offset,
        a_log, dt_bias, g_out, beta_out, num_tokens, num_heads
    );
}
