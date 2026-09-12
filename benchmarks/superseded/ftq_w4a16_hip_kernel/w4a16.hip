// Fused 4-bit dequantize + matvec for batch-1 decoding on gfx1100.
//
// This is the point of the packing work. PackedLinear in Python unpacks the
// weight into a full fp16 tensor and then calls a normal GEMM, so per token it
// reads the packed data, writes an fp16 copy and reads that back -- more
// traffic than storing fp16 in the first place, which is why it measured 3.1x
// slower than fp16 on 9B. Here the fp16 weight never exists.
//
// Layout must match ftq.packing.pack_bits exactly:
//   * 4-bit codes, LSB-first, 8 per uint32, row-major over (out, in)
//   * scale is fp16, one per (row, group), group = in_features / group_size
//   * levels is a 16-entry fp32 table shared by the tensor; w = level * scale
//
// Pinned against the Python reference by tests/test_hip_kernel.py.
//
// Structure notes (v1 was one block per row with a shared-memory reduction and
// the level table in per-thread registers; that cost 64 bytes of registers per
// thread and wasted most of each wavefront, and lost to fp16 on large shapes):
//   * one *wavefront* per output row, several rows per block
//   * reduction by cross-lane shuffle, so no shared memory and no barriers
//   * level table in shared memory, read once per block
//   * codes read as uint4 -- 4 words, 32 codes, one 16-byte load

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#ifndef WAVE
#define WAVE 32  // RDNA3 runs wave32 for compute
#endif

extern "C" __global__ __launch_bounds__(256) void w4a16_matvec(
    const __half* __restrict__ x,
    const unsigned int* __restrict__ q,
    const __half* __restrict__ scale,
    const float* __restrict__ levels,
    __half* __restrict__ y,
    int in_features,
    int out_features,
    int group_size) {
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = levels[threadIdx.x];
    __syncthreads();

    const int lane = threadIdx.x % WAVE;
    const int wave = threadIdx.x / WAVE;
    const int waves_per_block = blockDim.x / WAVE;
    const int row = blockIdx.x * waves_per_block + wave;
    if (row >= out_features) return;

    const int groups = in_features / group_size;
    const long row_word0 = ((long)row * in_features) >> 3;  // 8 codes per word
    const int words = in_features >> 3;

    float acc = 0.0f;

    // Each lane takes 4 words (32 codes) per step, read as one 16-byte load.
    const int words4 = words >> 2;
    const uint4* __restrict__ q4 = (const uint4*)(q + row_word0);

    for (int i = lane; i < words4; i += WAVE) {
        const uint4 packed = q4[i];
        const unsigned int w[4] = {packed.x, packed.y, packed.z, packed.w};
        const int base = i * 32;

#pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int k0 = base + t * 8;
            const float s = __half2float(scale[(long)row * groups + (k0 / group_size)]);
            const unsigned int word = w[t];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc += __half2float(x[k0 + j]) * lut[(word >> (4 * j)) & 0xFu] * s;
            }
        }
    }

    // Tail: words not covered by the uint4 loop (in_features not a multiple of 32).
    for (int i = (words4 << 2) + lane; i < words; i += WAVE) {
        const unsigned int word = q[row_word0 + i];
        const int k0 = i * 8;
        const float s = __half2float(scale[(long)row * groups + (k0 / group_size)]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc += __half2float(x[k0 + j]) * lut[(word >> (4 * j)) & 0xFu] * s;
        }
    }

#pragma unroll
    for (int off = WAVE / 2; off > 0; off >>= 1) acc += __shfl_down(acc, off, WAVE);

    if (lane == 0) y[row] = __float2half(acc);
}

// clang warns that it "does not support" __declspec(dllexport) here, but the
// symbol is not exported without it -- dropping it is what made ctypes report
// "function 'w4a16_matvec_launch' not found". Keep it and ignore the warning.
extern "C" __declspec(dllexport) int w4a16_matvec_launch(
    const void* x, const void* q, const void* scale, const void* levels, void* y,
    int in_features, int out_features, int group_size, int threads, void* stream) {
    if (in_features % 8 != 0) return -1;
    if (group_size % 8 != 0) return -2;
    if (in_features % group_size != 0) return -3;

    const int block = (threads > 0 ? threads : 256);
    if (block % WAVE != 0 || block > 1024) return -4;
    const int waves_per_block = block / WAVE;
    const int grid = (out_features + waves_per_block - 1) / waves_per_block;

    hipLaunchKernelGGL(w4a16_matvec, dim3(grid), dim3(block), 0, (hipStream_t)stream,
                       (const __half*)x, (const unsigned int*)q, (const __half*)scale,
                       (const float*)levels, (__half*)y,
                       in_features, out_features, group_size);
    return (int)hipGetLastError();
}
