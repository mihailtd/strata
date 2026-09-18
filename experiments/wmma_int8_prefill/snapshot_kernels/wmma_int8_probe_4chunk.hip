// §126 DEBUG: isolates the multi-chunk (4x K=32) accumulation loop from
// `w4a16_gemm_prefill_wmma_int8.hip`, using the SAME simple, direct
// per-row packed-int8 input format as the already-verified
// `wmma_int8_probe.hip` (K=32, bit-exact), extended to K=128 via 4
// sequential chunks accumulated into ONE persistent `acc`. Exists to
// determine whether accumulating across multiple WMMA chunk calls in a
// loop is where a real bug in the production kernel lives, or whether
// it's elsewhere (activation quantization, weight dequant, output write).

#include <hip/hip_runtime.h>

typedef int __attribute__((ext_vector_type(4))) v4i_t;
typedef int __attribute__((ext_vector_type(8))) v8i_t;

extern "C" __global__ void wmma_int8_probe_4chunk_kernel(
    const int* __restrict__ a_packed,  // [16 rows, 32 int32] = [16, 128] real int8, row-major
    const int* __restrict__ b_packed,  // same real layout
    int* __restrict__ c_out            // [16, 16] real int32 output, row-major
) {
    const int tid = threadIdx.x;
    const int row = tid % 16;

    v8i_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int c = 0; c < 4; c++) {
        v4i_t a_vec[2];
        v4i_t b_vec[2];
        int* a_s = reinterpret_cast<int*>(a_vec);
        int* b_s = reinterpret_cast<int*>(b_vec);
#pragma unroll
        for (int e = 0; e < 8; e++) {
            a_s[e] = a_packed[row * 32 + c * 8 + e];
            b_s[e] = b_packed[row * 32 + c * 8 + e];
        }
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[0], true, b_vec[0], acc, true);
        acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[1], true, b_vec[1], acc, true);
    }

#pragma unroll
    for (int l = 0; l < 8; l++) {
        const int out_row = 2 * l + (tid / 16);
        const int out_col = tid % 16;
        c_out[out_row * 16 + out_col] = acc[l];
    }
}

extern "C" void launch_wmma_int8_probe_4chunk(
    const void* a_packed,
    const void* b_packed,
    void* c_out,
    void* stream
) {
    hipLaunchKernelGGL(
        wmma_int8_probe_4chunk_kernel,
        dim3(1), dim3(32), 0, (hipStream_t)stream,
        (const int*)a_packed, (const int*)b_packed, (int*)c_out
    );
}
