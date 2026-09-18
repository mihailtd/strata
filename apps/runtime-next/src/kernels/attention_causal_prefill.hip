// Real HIP kernel: §119 -- Batched causal attention for prefill (T < 128),
// replacing the per-token host loop of `attention_decode_split` launches.
//
// Grid: `dim3(num_q_heads, num_tokens)` -- block (h, t) computes query head `h`
// for prompt token `t`.
// Causal mask: token `t` has position `positions[t]`, so valid KV length is
// `kv_len = positions[t] + 1`. Only keys/values up to `kv_len` are scored.
//
// LDS sizing: `head_dim + max_chunk_kv_len + blockDim.x + head_dim * kv_split` floats.
// For short prompts (<128 tokens), this is <4 KB LDS per block, maximizing RDNA3 CU occupancy.

#include "bf16_utils.hip"

extern "C" __global__ void attention_causal_prefill_bf16_kernel(
    const unsigned short* __restrict__ q,          // [num_tokens, num_q_heads * head_dim]
    const unsigned short* __restrict__ k,          // [num_kv_heads, kv_stride, head_dim]
    const unsigned short* __restrict__ v,          // [num_kv_heads, kv_stride, head_dim]
    unsigned short* __restrict__ out,              // [num_tokens, num_q_heads * head_dim]
    int num_q_heads,
    int num_kv_heads,
    const int* __restrict__ positions,             // [num_tokens], position per token
    int kv_stride,
    int head_dim,
    int kv_split,                                  // threads cooperating per output dim (blockDim.x == head_dim * kv_split)
    float scaling,
    int max_chunk_kv_len
) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int kv_len = positions[t] + 1;

    extern __shared__ float shared[];
    float* q_shared = shared;
    float* scores = shared + head_dim;
    float* reduce_scratch = scores + max_chunk_kv_len;
    float* combine = reduce_scratch + blockDim.x;

    const int n_rep = num_q_heads / num_kv_heads;
    const int h_kv = h / n_rep;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const size_t q_row_len = (size_t)num_q_heads * (size_t)head_dim;

    const unsigned short* q_row = q + (size_t)t * q_row_len + (size_t)h * (size_t)head_dim;
    const unsigned short* k_head = k + (size_t)h_kv * (size_t)kv_stride * (size_t)head_dim;
    const unsigned short* v_head = v + (size_t)h_kv * (size_t)kv_stride * (size_t)head_dim;

    for (int d = tid; d < head_dim; d += stride) {
        q_shared[d] = bf16_bits_to_f32(q_row[d]);
    }
    __syncthreads();

    // Phase 1: scores[j] = dot(Q, K[j]) * scaling for j in [0, kv_len)
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

    // Phase 2: numerically-stable softmax
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

    // Phase 3: kv_split threads cooperate per output dimension d
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
        unsigned short* out_row = out + (size_t)t * q_row_len + (size_t)h * (size_t)head_dim;
        out_row[d] = f32_to_bf16_bits(acc / sum_exp);
    }
}

extern "C" void launch_attention_causal_prefill_bf16(
    const void* q,
    const void* k,
    const void* v,
    void* out,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    const int* positions,
    int kv_stride,
    int head_dim,
    int kv_split,
    float scaling,
    int max_chunk_kv_len,
    void* stream
) {
    int threads = head_dim * kv_split;
    dim3 grid(num_q_heads, num_tokens);
    size_t shared_bytes = (head_dim + max_chunk_kv_len + threads + head_dim * kv_split) * sizeof(float);
    hipLaunchKernelGGL(
        attention_causal_prefill_bf16_kernel,
        grid, dim3(threads), shared_bytes, (hipStream_t)stream,
        (const unsigned short*)q,
        (const unsigned short*)k,
        (const unsigned short*)v,
        (unsigned short*)out,
        num_q_heads,
        num_kv_heads,
        positions,
        kv_stride,
        head_dim,
        kv_split,
        scaling,
        max_chunk_kv_len
    );
}
