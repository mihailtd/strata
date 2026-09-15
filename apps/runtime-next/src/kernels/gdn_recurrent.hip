// Real HIP kernel: GatedDeltaNet's single-token recurrent decode update
// (`torch_recurrent_gated_delta_rule` in
// transformers/models/qwen3_5/modeling_qwen3_5.py, the `use_precomputed_states
// and seq_len == 1` branch of `Qwen3_5GatedDeltaNet.forward`) -- the Hard
// tier's hardest, highest-payoff feature: 24 of this model's 32 layers use
// this exact op on EVERY decode step, and it is the actual thing this
// session's Python-runtime economics research (docs/DECISIONS.md §75-§79,
// `mtp_draft.py`'s flat ~2.7-2.84x verification tax) traced every cost
// back to -- no fused kernel exists for it even in the Python runtime:
// running the real reference generator for this kernel's test data printed
// `[transformers] "fused_recurrent_gated_delta_rule" is falling back to its
// reference PyTorch implementation because "flash-linear-attention" is not
// installed`, confirming directly (not just inferred from the TODO note)
// that Python pays the same "no real fused kernel on this rig" tax here
// that causal_conv1d_update.hip (§84) already found and beat.
//
// Scope, deliberately: exactly `torch_recurrent_gated_delta_rule`'s own
// function body for the `sequence_length == 1` case (the L2-norm of Q/K,
// the query/sqrt(head_dim) scaling, and one step of the decay + delta-rule
// state update) -- not the surrounding in_proj/gate/RMSNormGated/out_proj
// machinery in `Qwen3_5GatedDeltaNet.forward`.
//
// UPDATED (model-assembly pass): the real module-level `repeat_interleave`
// that broadcasts Q/K from num_k_heads=16 to num_v_heads=32 BEFORE calling
// this function is now folded into this kernel's own indexing (`h_kv = h /
// (num_heads/num_k_heads)`), the same pattern `attention.hip` already used
// for GQA -- rather than requiring the caller to physically duplicate Q/K
// into a separate buffer first. When `num_k_heads == num_heads` this is a
// no-op (`h_kv == h`), so the original correctness tests (which always
// passed already-broadcast Q/K) remain valid unchanged.
//
// Formula, per head h (real dims: num_v_heads=32, head_dim=128 for both
// k_head_dim and v_head_dim in this model), matching the real function's
// per-token loop body exactly:
//   q_n   = l2norm(Q[h]) / sqrt(head_dim)     // l2norm(x) = x / sqrt(sum(x^2)+eps), eps=1e-6
//   k_n   = l2norm(K[h])
//   decay = exp(g[h])                         // g is the pre-exp log-decay, matching decay_t = decay[...].exp()
//   state[h] *= decay                          // [head_dim, head_dim], decayed in place
//   kv_mem[v]  = sum_k state[h,k,v] * k_n[k]
//   delta[v]   = (V[h,v] - kv_mem[v]) * beta[h]
//   state[h,k,v] += k_n[k] * delta[v]           // rank-1 update, in place
//   out[h,v]   = sum_k state[h,k,v] * q_n[k]     // uses the JUST-updated state
//
// One block per head. Five phases, `__syncthreads()` between each (state
// must be fully decayed before `kv_mem` reads it; `delta` must be fully
// computed and shared before the rank-1 update; the update must complete
// before the final query read) -- same block-shared-memory-staging idiom
// `attention.hip` already established for a multi-phase per-block
// computation, extended here to five phases instead of three.
//
// `state` is real, persistent recurrent state -- read AND written in
// place across calls, exactly like `causal_conv1d_update.hip`'s
// `conv_state` -- not recomputed from scratch each call.

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

extern "C" __global__ void gdn_recurrent_decode_bf16_kernel(
    const unsigned short* __restrict__ q,  // [num_k_heads, head_dim]
    const unsigned short* __restrict__ k,  // [num_k_heads, head_dim]
    const unsigned short* __restrict__ v,  // [num_heads, head_dim]
    const float* __restrict__ g,           // [num_heads], pre-exp log-decay
    const float* __restrict__ beta,        // [num_heads]
    float* __restrict__ state,             // [num_heads, head_dim, head_dim], read+write in place
    unsigned short* __restrict__ out,      // [num_heads, head_dim]
    int num_heads,
    int num_k_heads,
    int head_dim
) {
    extern __shared__ float shared[];
    float* q_shared = shared;                      // [head_dim]
    float* k_shared = q_shared + head_dim;           // [head_dim]
    float* v_shared = k_shared + head_dim;           // [head_dim]
    float* delta_shared = v_shared + head_dim;        // [head_dim]
    float* reduce_scratch = delta_shared + head_dim;   // [blockDim.x]

    const int h = blockIdx.x;
    const int n_rep = num_heads / num_k_heads;
    const int h_kv = h / n_rep; // GQA broadcast: repeat_interleave's exact head-ordering
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const float eps = 1e-6f;

    const unsigned short* q_row = q + (size_t)h_kv * (size_t)head_dim;
    const unsigned short* k_row = k + (size_t)h_kv * (size_t)head_dim;
    const unsigned short* v_row = v + (size_t)h * (size_t)head_dim;
    float* state_head = state + (size_t)h * (size_t)head_dim * (size_t)head_dim;
    unsigned short* out_row = out + (size_t)h * (size_t)head_dim;

    // ---- Phase 1: load Q/K/V, L2-normalize Q and K ----
    float q_local = (tid < head_dim) ? bf16_bits_to_f32(q_row[tid]) : 0.0f;
    float k_local = (tid < head_dim) ? bf16_bits_to_f32(k_row[tid]) : 0.0f;
    if (tid < head_dim) {
        v_shared[tid] = bf16_bits_to_f32(v_row[tid]);
    }

    reduce_scratch[tid] = q_local * q_local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_scratch[tid] += reduce_scratch[tid + s];
        __syncthreads();
    }
    const float q_norm_factor = rsqrtf(reduce_scratch[0] + eps);
    __syncthreads();

    reduce_scratch[tid] = k_local * k_local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) reduce_scratch[tid] += reduce_scratch[tid + s];
        __syncthreads();
    }
    const float k_norm_factor = rsqrtf(reduce_scratch[0] + eps);
    __syncthreads();

    if (tid < head_dim) {
        q_shared[tid] = q_local * q_norm_factor * rsqrtf((float)head_dim);
        k_shared[tid] = k_local * k_norm_factor;
    }
    __syncthreads();

    // ---- Phase 2: decay the recurrent state in place ----
    const float decay = expf(g[h]);
    for (int idx = tid; idx < head_dim * head_dim; idx += stride) {
        state_head[idx] *= decay;
    }
    __syncthreads();

    // ---- Phase 3: kv_mem[v] = sum_k state[k,v]*k_n[k]; delta[v] ----
    const float beta_h = beta[h];
    for (int vv = tid; vv < head_dim; vv += stride) {
        float kv_mem = 0.0f;
        for (int kk = 0; kk < head_dim; ++kk) {
            kv_mem += state_head[(size_t)kk * (size_t)head_dim + vv] * k_shared[kk];
        }
        delta_shared[vv] = (v_shared[vv] - kv_mem) * beta_h;
    }
    __syncthreads();

    // ---- Phase 4: rank-1 update: state[k,v] += k_n[k] * delta[v] ----
    for (int idx = tid; idx < head_dim * head_dim; idx += stride) {
        int kk = idx / head_dim;
        int vv = idx % head_dim;
        state_head[idx] += k_shared[kk] * delta_shared[vv];
    }
    __syncthreads();

    // ---- Phase 5: out[v] = sum_k state[k,v]*q_n[k] (using UPDATED state) ----
    for (int vv = tid; vv < head_dim; vv += stride) {
        float acc = 0.0f;
        for (int kk = 0; kk < head_dim; ++kk) {
            acc += state_head[(size_t)kk * (size_t)head_dim + vv] * q_shared[kk];
        }
        out_row[vv] = f32_to_bf16_bits(acc);
    }
}

// Plain C entry point. `threads` must be a power of two and >= head_dim
// (phases 1/3/5 assume one thread covers each `[0, head_dim)` index
// directly, matching this model's real head_dim=128 with e.g. threads=128).
// `num_heads` must be a multiple of `num_k_heads`. Shared memory sized for
// `4*head_dim + threads` floats.
// `stream` (§93): see `rmsnorm.hip`'s launcher comment.
extern "C" void launch_gdn_recurrent_decode_bf16(
    const void* q,
    const void* k,
    const void* v,
    const float* g,
    const float* beta,
    float* state,
    void* out,
    int num_heads,
    int num_k_heads,
    int head_dim,
    int threads,
    void* stream
) {
    size_t shmem_bytes = (size_t)(4 * head_dim + threads) * sizeof(float);
    hipLaunchKernelGGL(
        gdn_recurrent_decode_bf16_kernel,
        dim3(num_heads), dim3(threads), shmem_bytes, (hipStream_t)stream,
        (const unsigned short*)q, (const unsigned short*)k, (const unsigned short*)v,
        g, beta, state, (unsigned short*)out, num_heads, num_k_heads, head_dim
    );
}
