// Real HIP kernel: §103 (real batched-prefill KV cache append, eliminating
// the per-token `kv_cache_append` loop `attn_layer_forward_prefill` used
// until now). Identical real per-append math to `kv_cache_append.hip`'s
// own kernel -- writes one new token's already-RoPE'd K/V into the
// head-major `[num_kv_heads, max_seq_len, head_dim]` cache -- just for
// every token in the chunk in ONE launch (`grid = (num_kv_heads,
// num_tokens)`), reading each token's own real absolute position from a
// `position_buf[num_tokens]` array. No cross-token dependency at all
// (each append writes to a distinct `position` slot, never read by
// another token's own append), so this is exact, not approximate.

#include <hip/hip_runtime.h>

extern "C" __global__ void kv_cache_append_prefill_bf16_kernel(
    const unsigned short* __restrict__ new_k,  // [num_tokens, num_kv_heads, head_dim], tightly packed
    const unsigned short* __restrict__ new_v,   // same shape
    unsigned short* __restrict__ k_cache,        // [num_kv_heads, max_seq_len, head_dim]
    unsigned short* __restrict__ v_cache,         // same shape
    int num_tokens,
    int num_kv_heads,
    int max_seq_len,
    int head_dim,
    const int* __restrict__ position_buf          // [num_tokens]
) {
    const int h = blockIdx.x;
    const int t = blockIdx.y;
    if (h >= num_kv_heads || t >= num_tokens) return;
    const int position = position_buf[t];

    const unsigned short* new_k_row = new_k + ((size_t)t * (size_t)num_kv_heads + (size_t)h) * (size_t)head_dim;
    const unsigned short* new_v_row = new_v + ((size_t)t * (size_t)num_kv_heads + (size_t)h) * (size_t)head_dim;
    unsigned short* k_slot = k_cache + ((size_t)h * (size_t)max_seq_len + (size_t)position) * (size_t)head_dim;
    unsigned short* v_slot = v_cache + ((size_t)h * (size_t)max_seq_len + (size_t)position) * (size_t)head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        k_slot[d] = new_k_row[d];
        v_slot[d] = new_v_row[d];
    }
}

// ONE launch appends every token in the chunk (`grid = (num_kv_heads,
// num_tokens)`), replacing `num_tokens` separate `launch_kv_cache_append_bf16`
// calls.
extern "C" void launch_kv_cache_append_prefill_bf16(
    const void* new_k,
    const void* new_v,
    void* k_cache,
    void* v_cache,
    int num_tokens,
    int num_kv_heads,
    int max_seq_len,
    int head_dim,
    const int* position_buf,
    void* stream
) {
    int threads = head_dim < 256 ? head_dim : 256;
    hipLaunchKernelGGL(
        kv_cache_append_prefill_bf16_kernel,
        dim3(num_kv_heads, num_tokens), dim3(threads), 0, (hipStream_t)stream,
        (const unsigned short*)new_k, (const unsigned short*)new_v,
        (unsigned short*)k_cache, (unsigned short*)v_cache,
        num_tokens, num_kv_heads, max_seq_len, head_dim, position_buf
    );
}
