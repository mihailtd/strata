// Real HIP kernel: embedding lookup. Given token ids, gathers the matching
// rows out of the real embedding table (model.language_model.embed_tokens.weight,
// [vocab_size=248320, hidden_size=2560], loaded for real in model_loader.rs).
// A pure gather -- no arithmetic, one row copy per requested token.
//
// FOUND THE HARD WAY: the first version of this kernel launched with
// `blockDim.x = hidden_size` (2560) directly, one thread per element, no
// stride loop -- unlike rmsnorm.hip, which already used a stride loop for
// exactly this reason. AMD hardware caps threads-per-block at 1024;
// launching 2560 is an invalid configuration. It did not error loudly: the
// launch silently produced wrong output (large stretches of zeros with a
// few real values mixed in from what looks like adjacent memory), caught
// only because the real correctness test in kernels.rs compares against
// the real table's own bytes rather than trusting the kernel. Fixed here
// (bounded thread count + stride loop) and in kernels.rs (every launch now
// checks `hipGetLastError()` immediately, not just the later
// `hipDeviceSynchronize()`, so an invalid launch configuration like this
// one becomes a loud, immediate error instead of silently wrong data next
// time).

#include <hip/hip_runtime.h>

extern "C" __global__ void embedding_lookup_bf16_kernel(
    const unsigned short* __restrict__ table,
    const int* __restrict__ token_ids,
    unsigned short* __restrict__ out,
    int hidden_size
) {
    const int row = blockIdx.x; // which requested token
    const int token_id = token_ids[row];
    const unsigned short* src_row = table + (size_t)token_id * (size_t)hidden_size;
    unsigned short* dst_row = out + (size_t)row * (size_t)hidden_size;

    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        dst_row[i] = src_row[i];
    }
}

// Plain C entry point. `num_tokens` requested lookups, `hidden_size`
// elements copied per lookup. `token_ids` is `num_tokens` real int32 ids,
// already validated to be < vocab_size by the caller (see kernels.rs).
// `stream` (§93): see `rmsnorm.hip`'s launcher comment.
extern "C" void launch_embedding_lookup_bf16(
    const void* table,
    const int* token_ids,
    void* out,
    int num_tokens,
    int hidden_size,
    void* stream
) {
    // Bounded, real thread count (not `hidden_size` directly -- see the
    // header comment above for why that was a real bug) with a stride loop
    // in the kernel to cover the rest, same pattern rmsnorm.hip already
    // used correctly from the start.
    int threads = hidden_size < 256 ? hidden_size : 256;
    hipLaunchKernelGGL(
        embedding_lookup_bf16_kernel,
        dim3(num_tokens), dim3(threads), 0, (hipStream_t)stream,
        (const unsigned short*)table, token_ids, (unsigned short*)out, hidden_size
    );
}
