// Real HIP kernel: §100 (chunked GDN parallel prefill). Computes the
// real chunked-DeltaNet decay bookkeeping, matching transformers'
// `torch_chunk_gated_delta_rule` EXACTLY (real function read directly at
// transformers/models/qwen3_5/modeling_qwen3_5.py:300-434, not
// re-derived from scratch):
//
//   cum_decay[t]       = cumsum(g[0..=t])            // g is log-decay, <= 0
//   pairwise_decay[i,j] = exp(cum_decay[i] - cum_decay[j])   for i >= j, else 0
//   decay_exp[t]        = exp(cum_decay[t])
//   remaining_decay[t]  = exp(cum_decay[C-1] - cum_decay[t])
//   chunk_decay         = exp(cum_decay[C-1])           // one scalar per head
//
// `pairwise_decay` is applied IN PLACE to TWO real raw score matrices in
// the same launch (`ut_system_raw = k_beta @ key^T` and
// `intra_chunk_attn_raw = query @ key^T`, both already computed by
// `blas::gemm_qkt_bf16` before this kernel runs) -- both need the exact
// same causal decay mask, so masking them together avoids recomputing
// `pairwise_decay` twice. Entries where `j > i` (causally future within
// the chunk) are masked to exactly 0, matching the real function's own
// `masked_fill(strictly_upper_mask, -inf)` BEFORE `.exp()` (which also
// yields exactly 0).
//
// One block per (head, chunk) pair (2D grid: `blockIdx.x = head`,
// `blockIdx.y = chunk`) -- decay/masking has no cross-chunk dependency
// (only the LATER inter-chunk scan does), so every chunk's decay
// bookkeeping is embarrassingly parallel and computed in ONE launch
// regardless of how many chunks there are.
//
// `decay_exp_out`/`remaining_decay_out`/`chunk_decay_out` are written in
// TOKEN-MAJOR layout (`[num_chunks, chunk_size, num_heads]`, matching `g`
// itself and every Q/K/V/beta buffer's own natural layout), NOT the
// per-block-convenient head-major layout the internal computation uses --
// a deliberate choice so every later elementwise scale-by-decay step
// (`decayed_k_beta = k_beta*decay_exp`, `query *= decay_exp`, `key *=
// remaining_decay`) can share ONE simple, consistent per-row indexing
// scheme with `beta`'s own `[T, num_heads]` layout, instead of juggling
// two different layouts across the pipeline.

#include "bf16_utils.hip"

extern "C" __global__ void gdn_chunk_decay_bf16_kernel(
    const float* __restrict__ g,               // [num_chunks, chunk_size, num_heads], log-decay
    unsigned short* __restrict__ ut_system,      // [num_heads, num_chunks, chunk_size, chunk_size], in-place
    unsigned short* __restrict__ intra_chunk_attn, // same shape, in-place
    float* __restrict__ decay_exp_out,          // [num_chunks, chunk_size, num_heads] -- TOKEN-major
    float* __restrict__ remaining_decay_out,    // [num_chunks, chunk_size, num_heads] -- TOKEN-major
    float* __restrict__ chunk_decay_out,        // [num_chunks, num_heads] -- TOKEN-major
    int num_heads,
    int num_chunks,
    int chunk_size
) {
    extern __shared__ float cum_decay[]; // [chunk_size]

    const int head = blockIdx.x;
    const int chunk = blockIdx.y;
    if (head >= num_heads || chunk >= num_chunks) return;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    // Real cumulative sum over the chunk's C positions for this head --
    // C is small (<=128 in this model's own real usage), so a single
    // sequential pass (thread 0 only) is simple and cheap; every other
    // thread waits at the barrier immediately after.
    if (tid == 0) {
        float running = 0.0f;
        for (int t = 0; t < chunk_size; ++t) {
            running += g[((size_t)chunk * chunk_size + t) * num_heads + head];
            cum_decay[t] = running;
        }
    }
    __syncthreads();

    const float last = cum_decay[chunk_size - 1];
    for (int t = tid; t < chunk_size; t += stride) {
        // Token-major write: row (chunk, t), column head.
        const size_t out_idx = ((size_t)chunk * chunk_size + t) * num_heads + head;
        decay_exp_out[out_idx] = expf(cum_decay[t]);
        remaining_decay_out[out_idx] = expf(last - cum_decay[t]);
    }
    if (tid == 0) {
        chunk_decay_out[(size_t)chunk * num_heads + head] = expf(last);
    }

    // Apply the real causal pairwise decay mask to both score matrices,
    // one (i,j) pair per thread (strided over the full C*C grid of this
    // block).
    unsigned short* ut_row_base = ut_system + ((size_t)head * num_chunks + chunk) * (size_t)chunk_size * (size_t)chunk_size;
    unsigned short* attn_row_base = intra_chunk_attn + ((size_t)head * num_chunks + chunk) * (size_t)chunk_size * (size_t)chunk_size;
    const int total = chunk_size * chunk_size;
    for (int idx = tid; idx < total; idx += stride) {
        const int i = idx / chunk_size;
        const int j = idx % chunk_size;
        float decay = (j <= i) ? expf(cum_decay[i] - cum_decay[j]) : 0.0f;
        ut_row_base[idx] = f32_to_bf16_bits(bf16_bits_to_f32(ut_row_base[idx]) * decay);
        attn_row_base[idx] = f32_to_bf16_bits(bf16_bits_to_f32(attn_row_base[idx]) * decay);
    }
}

// Plain C entry point. `threads` must be >= 1 (a power of two is not
// required here -- no tree reduction, just strided element ownership);
// shared memory sized for `chunk_size` floats (`cum_decay`).
extern "C" void launch_gdn_chunk_decay_bf16(
    const float* g,
    void* ut_system,
    void* intra_chunk_attn,
    float* decay_exp_out,
    float* remaining_decay_out,
    float* chunk_decay_out,
    int num_heads,
    int num_chunks,
    int chunk_size,
    int threads,
    void* stream
) {
    size_t shmem_bytes = (size_t)chunk_size * sizeof(float);
    hipLaunchKernelGGL(
        gdn_chunk_decay_bf16_kernel,
        dim3(num_heads, num_chunks), dim3(threads), shmem_bytes, (hipStream_t)stream,
        g, (unsigned short*)ut_system, (unsigned short*)intra_chunk_attn,
        decay_exp_out, remaining_decay_out, chunk_decay_out,
        num_heads, num_chunks, chunk_size
    );
}
