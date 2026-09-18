// Real HIP kernel: §119 -- Fused attention QKV prep for prefill.
//
// Replaces 6 separate small kernel launches:
//   1. extract_range(q_raw)
//   2. extract_range(k_raw)
//   3. extract_range(v_raw)
//   4. split_last_dim(q_raw -> query, gate)
//   5. rmsnorm(query -> query_normed)
//   6. rmsnorm(k_raw -> key_normed)
//
// Slices query, gate, key, and value directly from `qkv_in` in registers.
// Computes Q RMSNorm and K RMSNorm in LDS registers without intermediate DRAM round trips.
//
// Grid: `dim3(num_q_heads, num_tokens)`
// Block: `dim3(head_dim)` threads (128 threads per block).

#include "bf16_utils.hip"

extern "C" __global__ void fused_attn_qkv_prep_bf16_kernel(
    const unsigned short* __restrict__ qkv_in,      // [num_tokens, combined_dim]
    const unsigned short* __restrict__ q_weight,    // [head_dim]
    const unsigned short* __restrict__ k_weight,    // [head_dim]
    unsigned short* __restrict__ q_normed_out,      // [num_tokens, num_q_heads * head_dim]
    unsigned short* __restrict__ gate_out,          // [num_tokens, num_q_heads * head_dim]
    unsigned short* __restrict__ k_normed_out,      // [num_tokens, num_kv_heads * head_dim]
    unsigned short* __restrict__ v_out,             // [num_tokens, num_kv_heads * head_dim]
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float eps
) {
    extern __shared__ float sdata[];
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    const int tid = threadIdx.x;

    const size_t q_row_len = (size_t)num_q_heads * (size_t)head_dim;
    const size_t kv_row_len = (size_t)num_kv_heads * (size_t)head_dim;
    const size_t combined_dim = q_row_len * 2 + kv_row_len * 2;

    const unsigned short* qkv_row = qkv_in + (size_t)t * combined_dim;

    // 1. Process Query and Gate for head h
    float q_val = 0.0f;
    if (tid < head_dim) {
        size_t q_idx = (size_t)h * (size_t)(2 * head_dim) + (size_t)tid;
        size_t g_idx = q_idx + (size_t)head_dim;
        q_val = bf16_bits_to_f32(qkv_row[q_idx]);
        unsigned short g_bits = qkv_row[g_idx];
        gate_out[(size_t)t * q_row_len + (size_t)h * (size_t)head_dim + (size_t)tid] = g_bits;
        sdata[tid] = q_val * q_val;
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float q_mean_sq = sdata[0] / (float)head_dim;
    float q_rms = rsqrtf(q_mean_sq + eps);
    if (tid < head_dim) {
        float qw = bf16_bits_to_f32(q_weight[tid]);
        float normed = (q_val * q_rms) * (1.0f + qw);
        q_normed_out[(size_t)t * q_row_len + (size_t)h * (size_t)head_dim + (size_t)tid] = f32_to_bf16_bits(normed);
    }
    __syncthreads();

    // 2. Process Key and Value if h < num_kv_heads
    if (h < num_kv_heads) {
        float k_val = 0.0f;
        if (tid < head_dim) {
            size_t k_idx = q_row_len * 2 + (size_t)h * (size_t)head_dim + (size_t)tid;
            size_t v_idx = q_row_len * 2 + kv_row_len + (size_t)h * (size_t)head_dim + (size_t)tid;
            k_val = bf16_bits_to_f32(qkv_row[k_idx]);
            unsigned short v_bits = qkv_row[v_idx];
            v_out[(size_t)t * kv_row_len + (size_t)h * (size_t)head_dim + (size_t)tid] = v_bits;
            sdata[tid] = k_val * k_val;
        } else {
            sdata[tid] = 0.0f;
        }
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }

        float k_mean_sq = sdata[0] / (float)head_dim;
        float k_rms = rsqrtf(k_mean_sq + eps);
        if (tid < head_dim) {
            float kw = bf16_bits_to_f32(k_weight[tid]);
            float normed = (k_val * k_rms) * (1.0f + kw);
            k_normed_out[(size_t)t * kv_row_len + (size_t)h * (size_t)head_dim + (size_t)tid] = f32_to_bf16_bits(normed);
        }
    }
}

extern "C" void launch_fused_attn_qkv_prep_bf16(
    const void* qkv_in,
    const void* q_weight,
    const void* k_weight,
    void* q_normed_out,
    void* gate_out,
    void* k_normed_out,
    void* v_out,
    int num_tokens,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float eps,
    void* stream
) {
    dim3 grid(num_q_heads, num_tokens);
    int threads = head_dim;
    // Power of 2 threads for clean reduction:
    int threads_pow2 = 1;
    while (threads_pow2 < threads) {
        threads_pow2 <<= 1;
    }
    size_t shared_bytes = threads_pow2 * sizeof(float);
    hipLaunchKernelGGL(
        fused_attn_qkv_prep_bf16_kernel,
        grid, dim3(threads_pow2), shared_bytes, (hipStream_t)stream,
        (const unsigned short*)qkv_in,
        (const unsigned short*)q_weight,
        (const unsigned short*)k_weight,
        (unsigned short*)q_normed_out,
        (unsigned short*)gate_out,
        (unsigned short*)k_normed_out,
        (unsigned short*)v_out,
        num_tokens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        eps
    );
}
