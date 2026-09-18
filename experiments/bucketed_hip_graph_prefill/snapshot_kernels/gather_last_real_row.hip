// Real HIP kernel: §117 -- copies row `(*real_num_tokens_ptr - 1)` out of
// a `[num_rows, row_width]` bf16 buffer into a fixed, tightly-packed
// `[row_width]` destination, reading `real_num_tokens` off a real DEVICE
// pointer INSIDE the kernel.
//
// Real, found-not-assumed reason this exists: `run_prefill_chunk_body`
// used to compute which row of `final_hidden` feeds `rmsnorm`+`lm_head`
// (the only row prefill ever needs logits for) via a real HOST pointer
// offset, `final_hidden.as_device_ptr_at((real_num_tokens - 1) *
// HIDDEN_SIZE)`. That offset is a HOST value baked directly into
// `rmsnorm`'s own launch arguments -- once this call is captured into a
// HIP Graph (§117 Stage 2), a real replay does NOT re-evaluate it: the
// SAME row index (from whichever real prompt captured first) gets reused
// for every later real request sharing that bucket, silently feeding the
// wrong row's logits. This kernel makes the row SELECTION itself a
// real device-side, per-replay operation: `rmsnorm`'s own input pointer
// stays a FIXED destination (`dst`, offset 0, always) across every
// replay, while THIS kernel decides which real source row to copy there,
// fresh, every single call.

#include <hip/hip_runtime.h>

extern "C" __global__ void gather_last_real_row_bf16_kernel(
    const unsigned short* __restrict__ src,   // [num_rows, row_width]
    const int* __restrict__ real_num_tokens_ptr,
    unsigned short* __restrict__ dst,         // [row_width], tightly packed
    int row_width
) {
    const int row = *real_num_tokens_ptr - 1;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < row_width) {
        dst[i] = src[(size_t)row * (size_t)row_width + (size_t)i];
    }
}

extern "C" void launch_gather_last_real_row_bf16(
    const void* src,
    const void* real_num_tokens_ptr,
    void* dst,
    int row_width,
    void* stream
) {
    int threads = 256;
    int blocks = (row_width + threads - 1) / threads;
    hipLaunchKernelGGL(
        gather_last_real_row_bf16_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        (const unsigned short*)src, (const int*)real_num_tokens_ptr, (unsigned short*)dst, row_width
    );
}
