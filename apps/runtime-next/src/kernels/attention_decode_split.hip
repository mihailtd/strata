// Real HIP kernel: §104 -- a faster single-token decode-step full
// attention, targeting the REAL, measured bottleneck `attention.hip`'s
// existing kernel has: decode throughput genuinely DECLINES as the KV
// cache grows (measured directly, `~81.7 -> ~77.5 tok/s` over positions
// 54->354 in a real 300-token generation, see `docs/DECISIONS.md` §103's
// own investigation) because that kernel's final weighted-V-sum phase
// gives each thread exactly ONE output dimension, then has that ONE
// thread do a fully SERIAL `O(kv_len)` reduction over the whole KV cache
// -- the one part of the whole kernel whose per-thread work keeps
// growing with position, unparallelized.
//
// Real, validated technique this ports (found via a direct repo survey,
// not invented from scratch): llama.cpp's own decode-attention kernel
// (`apps/runtime-llama/llama.cpp/ggml/src/ggml-cuda/fattn-vec.cuh`,
// actively used in this repo's own `runtime-llama` deployment) splits the
// KV sequence across multiple parallel workers instead of one serial
// scan per output dimension -- the real "flash-decoding" idea. This
// kernel ports the SAME idea via a SIMPLER mechanism that avoids a
// separate combine-kernel launch (and the HIP-Graph-capture-safety
// complications a variable-grid-size second launch would need): more
// threads per block (`K` threads cooperate on EACH output dimension
// instead of one), each thread scanning `kv_len/K` positions instead of
// the full `kv_len`, then a cheap shared-memory combine across the `K`
// partial sums per dimension.
//
// Formula: bit-for-bit the SAME real math as `attention_decode_bf16`
// (see that kernel's own header for the derivation from the real
// `eager_attention_forward`+`repeat_kv` functions) -- this kernel is a
// real PARALLELISM restructuring, not a different computation. Phases 1
// (score) and 2 (softmax reduction) are unchanged in spirit but now run
// with `head_dim * K` threads instead of `head_dim`, so both also get
// strictly more parallelism for free. Phase 3 (weighted V sum) is the
// real, deliberate restructuring:
//   d      = tid % head_dim
//   split  = tid / head_dim                         // 0..K-1
//   partial[split, d] = sum_{j=split,step K}^{kv_len} prob[j] * V[j,d]
//   out[d] = sum_{s=0}^{K-1} partial[s, d]            // combined by split==0

#include "bf16_utils.hip"

extern "C" void launch_attention_decode_split_batched_bf16(
    const void* q, const void* k, const void* v, void* out,
    int num_q_heads, int num_kv_heads, const int* position, int kv_stride,
    int head_dim, int kv_split, float scaling, int batch,
    long long cache_stride, long long q_stride, void* stream);

extern "C" __global__ void attention_decode_split_bf16_kernel(
    const unsigned short* q,  // [batch, num_q_heads, head_dim]
    const unsigned short* k,  // [batch, num_kv_heads, kv_stride, head_dim]
    const unsigned short* v,  // [batch, num_kv_heads, kv_stride, head_dim]
    unsigned short* out,      // [batch, num_q_heads, head_dim]
    int num_q_heads,
    int num_kv_heads,
    const int* __restrict__ position_ptr,  // [batch]; kv_len = position_ptr[b] + 1
    int kv_stride,
    int head_dim,
    int kv_split,                           // K: threads cooperating per output dim (blockDim.x == head_dim * kv_split)
    float scaling,
    long long cache_stride,                 // elements between sequences in k/v
    long long q_stride                      // elements between sequences in q/out
) {
    // Shared memory (fixed upper bound at `kv_stride`, not the real
    // `kv_len` -- same real graph-capture-safety reasoning
    // `attention_decode_bf16_kernel` already established, see its own
    // comment): q_shared[head_dim], scores[kv_stride],
    // reduce_scratch[blockDim.x], combine[head_dim*kv_split].
    const int b = blockIdx.y;
    const int kv_len = position_ptr[b] + 1;
    extern __shared__ float shared[];
    float* q_shared = shared;
    float* scores = shared + head_dim;
    float* reduce_scratch = scores + kv_stride;
    float* combine = reduce_scratch + blockDim.x; // [head_dim * kv_split]

    const int h = blockIdx.x;
    // Each sequence owns a contiguous slice of the shared q/out and K/V
    // allocations. batch=1 with zero strides is the original behaviour.
    q = q + (size_t)b * (size_t)q_stride;
    out = out + (size_t)b * (size_t)q_stride;
    k = k + (size_t)b * (size_t)cache_stride;
    v = v + (size_t)b * (size_t)cache_stride;
    const int n_rep = num_q_heads / num_kv_heads;
    const int h_kv = h / n_rep;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const unsigned short* q_row = q + (size_t)h * (size_t)head_dim;
    const unsigned short* k_head = k + (size_t)h_kv * (size_t)kv_stride * (size_t)head_dim;
    const unsigned short* v_head = v + (size_t)h_kv * (size_t)kv_stride * (size_t)head_dim;

    for (int d = tid; d < head_dim; d += stride) {
        q_shared[d] = bf16_bits_to_f32(q_row[d]);
    }
    __syncthreads();

    // Phase 1: scores[j] = dot(Q, K[j]) * scaling -- MORE threads than
    // the original kernel (head_dim*kv_split instead of head_dim), so
    // this phase is strictly more parallel too, for free.
    for (int j = tid; j < kv_len; j += stride) {
        const unsigned short* k_row = k_head + (size_t)j * (size_t)head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d += 4) {
            ushort4 kv = *reinterpret_cast<const ushort4*>(k_row + d);
            dot += q_shared[d] * bf16_bits_to_f32(kv.x);
            dot += q_shared[d + 1] * bf16_bits_to_f32(kv.y);
            dot += q_shared[d + 2] * bf16_bits_to_f32(kv.z);
            dot += q_shared[d + 3] * bf16_bits_to_f32(kv.w);
        }
        scores[j] = dot * scaling;
    }
    __syncthreads();

    // Phase 2: numerically-stable softmax, same block-level max/sum
    // reduction idiom as the original kernel, just over more threads.
    float local_max = -3.402823466e38f;
    for (int j = tid; j < kv_len; j += stride) {
        local_max = fmaxf(local_max, scores[j]);
    }
    reduce_scratch[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            reduce_scratch[tid] = fmaxf(reduce_scratch[tid], reduce_scratch[tid + s]);
        }
        __syncthreads();
    }
    const float max_score = reduce_scratch[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int j = tid; j < kv_len; j += stride) {
        float e = expf(scores[j] - max_score);
        scores[j] = e;
        local_sum += e;
    }
    reduce_scratch[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            reduce_scratch[tid] += reduce_scratch[tid + s];
        }
        __syncthreads();
    }
    const float sum_exp = reduce_scratch[0];
    __syncthreads();

    // Phase 3, the REAL restructuring: `kv_split` threads cooperate on
    // EACH output dimension `d`, each scanning `kv_len/kv_split`
    // positions instead of the full `kv_len` -- the actual fix for the
    // position-dependent slowdown this kernel exists to address.
    const int d = tid % head_dim;
    const int split = tid / head_dim;
    float partial = 0.0f;
    for (int j = split; j < kv_len; j += kv_split) {
        partial += scores[j] * bf16_bits_to_f32(v_head[(size_t)j * (size_t)head_dim + d]);
    }
    combine[split * head_dim + d] = partial;
    __syncthreads();

    if (split == 0) {
        float acc = 0.0f;
        for (int s = 0; s < kv_split; ++s) {
            acc += combine[s * head_dim + d];
        }
        unsigned short* out_row = out + (size_t)h * (size_t)head_dim;
        out_row[d] = f32_to_bf16_bits(acc / sum_exp);
    }
}

// Plain C entry point. `threads` MUST equal `head_dim * kv_split` exactly
// (checked by the Rust safe wrapper, not just assumed) -- shared memory
// sized for `head_dim + kv_stride + threads + head_dim*kv_split` floats,
// the FIXED upper bound (kv_stride, not the varying kv_len), same real
// graph-capture-safety reasoning as `attention.hip`'s own launcher.
extern "C" void launch_attention_decode_split_bf16(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int num_q_heads,
    int num_kv_heads,
    const int* position,
    int kv_stride,
    int head_dim,
    int kv_split,
    float scaling,
    void* stream
) {
    launch_attention_decode_split_batched_bf16(q, k, v, out, num_q_heads, num_kv_heads, position, kv_stride, head_dim, kv_split, scaling, 1, 0, 0, stream);
}

// Batched entry point: `batch` independent sequences, each with its own
// `position[b]` and its own `cache_stride`-element slice of the shared K/V
// allocation. Per-block shared memory is unchanged (one block still serves one
// (head, sequence) pair), so the gfx1100 LDS ceiling on `kv_stride` documented
// in `server::DEFAULT_MAX_SEQ_LEN` is unaffected by batching.
extern "C" void launch_attention_decode_split_batched_bf16(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int num_q_heads,
    int num_kv_heads,
    const int* position,
    int kv_stride,
    int head_dim,
    int kv_split,
    float scaling,
    int batch,
    long long cache_stride,
    long long q_stride,
    void* stream
) {
    int threads = head_dim * kv_split;
    size_t shmem_bytes = (size_t)(head_dim + kv_stride + threads + head_dim * kv_split) * sizeof(float);
    hipLaunchKernelGGL(
        attention_decode_split_bf16_kernel,
        dim3(num_q_heads, batch), dim3(threads), shmem_bytes, (hipStream_t)stream,
        (const unsigned short*)q, (const unsigned short*)k, (const unsigned short*)v,
        (unsigned short*)out, num_q_heads, num_kv_heads, position, kv_stride, head_dim, kv_split, scaling,
        cache_stride, q_stride
    );
}
