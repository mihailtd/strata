// Real HIP kernel: appends one new token's K/V (already RoPE'd/normed) into
// a head-major KV cache at a given position -- `past_key_values.update(...)`
// in the real Cache API, scoped to exactly the single-token-append case
// this decode-only port needs. Cache layout is `[num_kv_heads, max_seq_len,
// head_dim]` (head-major), matching what `attention.hip` already expects
// to read directly -- chosen specifically so a read needs no further
// reshaping, at the cost of this append needing to write to
// `num_kv_heads` separate, non-contiguous offsets (cheap: this model's
// real num_kv_heads=4, head_dim=256, so 1024 elements total per append).

#include <hip/hip_runtime.h>

extern "C" void launch_kv_cache_append_batched_bf16(
    const void* new_k, const void* new_v, void* k_cache, void* v_cache,
    int num_kv_heads, int max_seq_len, int head_dim, const int* position,
    int batch, long long cache_stride, void* stream);

// Batched form: `blockIdx.y` is the sequence. Each sequence owns a
// contiguous `[num_kv_heads, max_seq_len, head_dim]` slice of one shared
// allocation (stride `cache_stride`) and its OWN absolute position, read from
// `position_ptr[b]` -- B unrelated sequences, not B tokens of one sequence.
// batch=1 with cache_stride ignored reproduces the original single-sequence
// behaviour exactly.
extern "C" __global__ void kv_cache_append_bf16_kernel(
    const unsigned short* __restrict__ new_k,  // [batch, num_kv_heads, head_dim]
    const unsigned short* __restrict__ new_v,  // [batch, num_kv_heads, head_dim]
    unsigned short* __restrict__ k_cache,       // [batch, num_kv_heads, max_seq_len, head_dim]
    unsigned short* __restrict__ v_cache,       // same shape
    int num_kv_heads,
    int max_seq_len,
    int head_dim,
    const int* __restrict__ position_ptr,       // [batch]
    long long cache_stride                       // elements between sequences
) {
    int h = blockIdx.x;
    if (h >= num_kv_heads) return;
    const int b = blockIdx.y;
    const int position = position_ptr[b];
    const size_t row_stride = (size_t)num_kv_heads * (size_t)head_dim;
    const unsigned short* new_k_row = new_k + (size_t)b * row_stride + (size_t)h * (size_t)head_dim;
    const unsigned short* new_v_row = new_v + (size_t)b * row_stride + (size_t)h * (size_t)head_dim;
    unsigned short* k_base = k_cache + (size_t)b * (size_t)cache_stride;
    unsigned short* v_base = v_cache + (size_t)b * (size_t)cache_stride;
    unsigned short* k_slot = k_base + ((size_t)h * (size_t)max_seq_len + (size_t)position) * (size_t)head_dim;
    unsigned short* v_slot = v_base + ((size_t)h * (size_t)max_seq_len + (size_t)position) * (size_t)head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        k_slot[d] = new_k_row[d];
        v_slot[d] = new_v_row[d];
    }
}

// `position` is a DEVICE pointer -- same reasoning as `rope.hip`'s
// `launch_rope_bf16` (§93): keeps the graph-recorded launch parameters
// constant across replays even as the pointee advances every decode step.
extern "C" void launch_kv_cache_append_bf16(
    const void* new_k,
    const void* new_v,
    void* k_cache,
    void* v_cache,
    int num_kv_heads,
    int max_seq_len,
    int head_dim,
    const int* position,
    void* stream
) {
    launch_kv_cache_append_batched_bf16(new_k, new_v, k_cache, v_cache, num_kv_heads, max_seq_len, head_dim, position, 1, 0, stream);
}

// Batched entry point: `batch` sequences, each owning a `cache_stride`-element
// slice of the shared K/V allocations and its own `position[b]`.
extern "C" void launch_kv_cache_append_batched_bf16(
    const void* new_k,
    const void* new_v,
    void* k_cache,
    void* v_cache,
    int num_kv_heads,
    int max_seq_len,
    int head_dim,
    const int* position,
    int batch,
    long long cache_stride,
    void* stream
) {
    int threads = head_dim < 256 ? head_dim : 256;
    hipLaunchKernelGGL(
        kv_cache_append_bf16_kernel,
        dim3(num_kv_heads, batch), dim3(threads), 0, (hipStream_t)stream,
        (const unsigned short*)new_k, (const unsigned short*)new_v,
        (unsigned short*)k_cache, (unsigned short*)v_cache,
        num_kv_heads, max_seq_len, head_dim, position, cache_stride
    );
}
