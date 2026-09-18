// Real HIP kernel: §117 -- real, unconditional (no host-side branch),
// device-boundary-aware zero-fill: `buf[real_num_tokens*row_width ..
// num_rows*row_width) = 0`, reading `real_num_tokens` from a real DEVICE
// pointer INSIDE the kernel instead of as a host-baked launch parameter
// or (worse) a host-side `if` deciding whether to launch the zero-fill
// AT ALL.
//
// Real, found-not-assumed reason this exists: `gdn_chunk_forward_prefill`
// used to zero `gdn_mixed_qkv`/`gdn_g`/`gdn_beta`'s real padding tail via
// `DeviceBuffer::fill_zero_from_async`, gated by a real HOST comparison
// (`if t_pad > real_num_tokens`) and a real HOST-computed byte offset
// (`real_num_tokens * row_width`). Both of those are HOST-SIDE values --
// once this whole real prefill pipeline is captured into a HIP Graph
// (§117 Stage 2), a real graph replay does NOT re-evaluate either: the
// branch (whether the zero-fill node exists in the graph AT ALL) and the
// offset (baked into that node's own memset arguments) are BOTH fixed at
// CAPTURE time, from whichever real prompt happened to capture first.
// This kernel replaces that host-side branch+offset with a real,
// ALWAYS-present kernel launch (so it's ALWAYS in the graph) that reads
// the real, live `real_num_tokens` value off the device EVERY replay,
// zeroing exactly the real rows that call actually needs zeroed -- no
// host-side decision baked in anywhere.
//
// `num_rows`/`row_width` ARE real host-side values too, but they're
// legitimately FIXED per bucket (the whole reason bucketing exists), not
// something that needs to vary call to call the way `real_num_tokens`
// does -- safe to bake in.

#include <hip/hip_runtime.h>

extern "C" __global__ void zero_from_device_offset_bf16_kernel(
    unsigned short* __restrict__ buf,
    const int* __restrict__ real_num_tokens_ptr,
    int num_rows,
    int row_width
) {
    const int real_num_tokens = *real_num_tokens_ptr;
    const long long total = (long long)num_rows * (long long)row_width;
    const long long start = (long long)real_num_tokens * (long long)row_width;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = start + i; idx < total; idx += stride) {
        buf[idx] = 0;
    }
}

extern "C" __global__ void zero_from_device_offset_f32_kernel(
    float* __restrict__ buf,
    const int* __restrict__ real_num_tokens_ptr,
    int num_rows,
    int row_width
) {
    const int real_num_tokens = *real_num_tokens_ptr;
    const long long total = (long long)num_rows * (long long)row_width;
    const long long start = (long long)real_num_tokens * (long long)row_width;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = start + i; idx < total; idx += stride) {
        buf[idx] = 0.0f;
    }
}

extern "C" void launch_zero_from_device_offset_bf16(void* buf, const void* real_num_tokens_ptr, int num_rows, int row_width, void* stream) {
    int threads = 256;
    long long total = (long long)num_rows * (long long)row_width;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    hipLaunchKernelGGL(
        zero_from_device_offset_bf16_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        (unsigned short*)buf, (const int*)real_num_tokens_ptr, num_rows, row_width
    );
}

extern "C" void launch_zero_from_device_offset_f32(void* buf, const void* real_num_tokens_ptr, int num_rows, int row_width, void* stream) {
    int threads = 256;
    long long total = (long long)num_rows * (long long)row_width;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    hipLaunchKernelGGL(
        zero_from_device_offset_f32_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        (float*)buf, (const int*)real_num_tokens_ptr, num_rows, row_width
    );
}
