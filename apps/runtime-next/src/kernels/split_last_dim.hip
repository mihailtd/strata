// Real HIP kernel: splits a [rows, 2*half] buffer into two [rows, half]
// contiguous buffers along the LAST dimension per row -- matching
// `torch.chunk(q_proj(x).view(rows, -1, head_dim*2), 2, dim=-1)` in
// Qwen3_5Attention.forward exactly (query = the first half of each row,
// gate = the second half). NOT a simple contiguous split of the flat
// buffer: row h's two halves live at offsets h*2*half and h*2*half+half,
// interleaved with every other row's halves, so this needs a real kernel,
// not just a pointer-offset view (unlike in_proj_qkv's query/key/value
// split, which IS three contiguous back-to-back ranges and needs no
// kernel at all -- see `DeviceBuffer::as_device_ptr_at`).

#include <hip/hip_runtime.h>

extern "C" __global__ void split_last_dim_bf16_kernel(
    const unsigned short* __restrict__ x,   // [rows, 2*half]
    unsigned short* __restrict__ first,      // [rows, half]
    unsigned short* __restrict__ second,     // [rows, half]
    int rows,
    int half
) {
    int row = blockIdx.x;
    const unsigned short* xrow = x + (size_t)row * (size_t)(2 * half);
    unsigned short* frow = first + (size_t)row * (size_t)half;
    unsigned short* srow = second + (size_t)row * (size_t)half;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        frow[i] = xrow[i];
        srow[i] = xrow[half + i];
    }
}

// `stream` (§93): see `rmsnorm.hip`'s launcher comment.
extern "C" void launch_split_last_dim_bf16(
    const void* x,
    void* first,
    void* second,
    int rows,
    int half,
    void* stream
) {
    int threads = half < 256 ? half : 256;
    hipLaunchKernelGGL(
        split_last_dim_bf16_kernel,
        dim3(rows), dim3(threads), 0, (hipStream_t)stream,
        (const unsigned short*)x, (unsigned short*)first, (unsigned short*)second, rows, half
    );
}
