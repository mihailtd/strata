// Real HIP kernel: single-token decode-step full attention (the "Hard
// tier" feature -- QK^T -> scale -> softmax -> AV, with GQA broadcasting
// `num_kv_heads` KV heads to `num_q_heads` Q heads), matching the real
// `eager_attention_forward` + `repeat_kv` functions in
// transformers/models/qwen3_5/modeling_qwen3_5.py exactly. Used by 8 of
// this model's 32 layers (the `full_attention` layers, every 4th).
//
// Scope, deliberately: this kernel is exactly `eager_attention_forward`'s
// scope -- Q/K/V in, attended output out -- not the surrounding q_proj/
// k_proj/v_proj/q_norm/k_norm/RoPE/gate/o_proj machinery around it in
// `Qwen3_5Attention.forward`. Those are separate real ops this project
// already has kernels for (RMSNorm from §82, RoPE from §83, a GEMM from
// §85) or hasn't ported yet (gating, o_proj) -- same "one real feature at
// a time" discipline as `causal_conv1d_update.hip`.
//
// No causal mask: this kernel is the single-new-token decode case (Q has
// exactly one token per head), where every position in the KV cache is,
// by construction, causally valid to attend to -- there is nothing future
// in the cache to mask. This matches how real incremental-decode serving
// loops call attention (attention_mask=None once a Cache is in play and
// there is no batch padding), not a shortcut taken to avoid the mask.
//
// Formula (batch=1, one new query token per head):
//   h_kv          = h / (num_q_heads / num_kv_heads)      // repeat_kv's mapping
//   score[h, j]   = dot(Q[h, :], K[h_kv, j, :]) * scaling  // scaling = head_dim^-0.5
//   prob[h, :]    = softmax(score[h, :])                   // computed in fp32
//   out[h, :]     = sum_j prob[h, j] * V[h_kv, j, :]
//
// One block per Q head. Three shared-memory regions: a cached copy of
// this head's Q vector (head_dim floats, read many times, so cached once
// rather than re-read from global memory per kv position), the per-
// position scores/probabilities (kv_len floats), and a small reduction
// scratch (blockDim.x floats) for the block-level max/sum reductions --
// same power-of-two halving reduction idiom `rmsnorm.hip` already
// established, reused here for two reductions instead of one.
//
// UPDATED (§92 performance pass): `kv_stride` reads K/V DIRECTLY out of a
// head-major KV cache whose per-head capacity (`max_seq_len`) differs from
// the number of currently-valid positions (`kv_len` = position+1) --
// avoiding a real, previously-required per-token device-to-device copy
// into a tightly-packed scratch buffer (`model.rs` used to `hipMemcpyAsync`
// the valid prefix of the cache into a [num_kv_heads,kv_len,head_dim]
// buffer before this kernel could read it, since it only ever understood
// tightly-packed input; 8 such copies per attention layer, growing with
// position). `kv_stride` is the real per-head stride in the underlying
// buffer (`max_seq_len`); `kv_len` is unchanged, still "how many of those
// positions are valid to read." Passing `kv_stride == kv_len` reproduces
// the exact old tightly-packed behavior, so this is a strict
// generalization, not a formula change.

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

extern "C" __global__ void attention_decode_bf16_kernel(
    const unsigned short* __restrict__ q,  // [num_q_heads, head_dim]
    const unsigned short* __restrict__ k,  // [num_kv_heads, kv_stride, head_dim], first kv_len valid per head
    const unsigned short* __restrict__ v,  // [num_kv_heads, kv_stride, head_dim], first kv_len valid per head
    unsigned short* __restrict__ out,      // [num_q_heads, head_dim]
    int num_q_heads,
    int num_kv_heads,
    const int* __restrict__ position_ptr,  // kv_len = *position_ptr + 1
    int kv_stride,
    int head_dim,
    float scaling
) {
    // Shared memory is allocated by the HOST for the fixed upper bound
    // `kv_stride` (see `launch_attention_decode_bf16` below), not the
    // actual runtime `kv_len` -- §93: a HIP Graph node's launch
    // configuration (including dynamic shared memory size) is fixed at
    // capture time, so it must not depend on a value that changes every
    // decode step. `scores`/`reduce_scratch` below are indexed only up to
    // the real `kv_len`, which is always <= kv_stride, so this never reads
    // or writes past what's actually been computed -- the extra
    // kv_stride-kv_len slots of allocated shared memory simply go unused.
    const int kv_len = *position_ptr + 1;
    extern __shared__ float shared[];
    float* q_shared = shared;                  // [head_dim]
    float* scores = shared + head_dim;          // [kv_len] (of kv_stride allocated)
    float* reduce_scratch = scores + kv_len;     // [blockDim.x]

    const int h = blockIdx.x;
    const int n_rep = num_q_heads / num_kv_heads;
    const int h_kv = h / n_rep;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const unsigned short* q_row = q + (size_t)h * (size_t)head_dim;
    const unsigned short* k_head = k + (size_t)h_kv * (size_t)kv_stride * (size_t)head_dim;
    const unsigned short* v_head = v + (size_t)h_kv * (size_t)kv_stride * (size_t)head_dim;

    // Cache this head's Q vector in shared memory (read kv_len times below).
    for (int d = tid; d < head_dim; d += stride) {
        q_shared[d] = bf16_bits_to_f32(q_row[d]);
    }
    __syncthreads();

    // scores[j] = dot(Q, K[j]) * scaling, one full head_dim-length dot
    // product per thread per assigned j. Vectorized `ushort4` (4 bf16
    // elements/load) reads of K -- same real memory-bandwidth lever
    // `gemv.hip` established for this project's GEMM-as-GEMV calls,
    // applied here since `head_dim` is a multiple of 4 for every real
    // shape this kernel is called with (asserted by the Rust wrapper, see
    // `kernels.rs::attention_decode_bf16`).
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

    // Block-level max reduction over scores[0..kv_len) (numerically stable softmax).
    float local_max = -3.402823466e38f; // -FLT_MAX
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

    // exp(score - max), plus a block-level sum reduction.
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

    // out[d] = sum_j (scores[j] / sum_exp) * V[j, d], one output dim per thread.
    unsigned short* out_row = out + (size_t)h * (size_t)head_dim;
    for (int d = tid; d < head_dim; d += stride) {
        float acc = 0.0f;
        for (int j = 0; j < kv_len; ++j) {
            acc += scores[j] * bf16_bits_to_f32(v_head[(size_t)j * (size_t)head_dim + d]);
        }
        out_row[d] = f32_to_bf16_bits(acc / sum_exp);
    }
}

// Plain C entry point. `threads` must be a power of two (reduction loops
// above); shared memory sized for `head_dim + kv_stride + threads` floats
// -- the FIXED upper bound, not the actual (varying) `kv_len`, so this
// launch's configuration is identical every call regardless of position
// (§93, graph-capture-safety). `kv_stride` is the real per-head capacity of
// the K/V buffers; `position` is a DEVICE pointer, `kv_len = *position + 1`
// is computed inside the kernel from it (see the kernel's own comment).
extern "C" void launch_attention_decode_bf16(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int num_q_heads,
    int num_kv_heads,
    const int* position,
    int kv_stride,
    int head_dim,
    float scaling,
    int threads,
    void* stream
) {
    size_t shmem_bytes = (size_t)(head_dim + kv_stride + threads) * sizeof(float);
    hipLaunchKernelGGL(
        attention_decode_bf16_kernel,
        dim3(num_q_heads), dim3(threads), shmem_bytes, (hipStream_t)stream,
        (const unsigned short*)q, (const unsigned short*)k, (const unsigned short*)v,
        (unsigned short*)out, num_q_heads, num_kv_heads, position, kv_stride, head_dim, scaling
    );
}
