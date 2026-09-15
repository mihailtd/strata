// Real HIP kernel: argmax over real vocab logits, entirely on-device.
//
// Motivation: `model.rs::argmax_sample`'s original implementation copied
// the FULL logits buffer (`VOCAB_SIZE=248320` bf16 elements, ~496KB) back
// to the host via `hipMemcpy`, then did a serial host-side scan over all
// 248320 elements to find the max -- real, measured cost: ~0.209ms/token
// (see the (now-removed) ad-hoc timing split in `bench_real_graphed_
// decode_tokens_per_second`), on top of the real GPU forward-pass time.
// This kernel computes the SAME result on-device and copies back only the
// resulting 4-byte index, eliminating essentially all of that overhead
// without changing `argmax_sample`'s external signature or behavior.
//
// TIE-BREAKING: matches the original host implementation's semantics
// exactly -- the FIRST (lowest-index) occurrence of the maximum value
// wins on an exact tie, not an arbitrary one. The per-thread scan
// naturally does this (strict `>`, threads scan increasing indices); the
// block-wide reduction below does it explicitly (see its own comment) --
// astronomically unlikely to matter for real model logits, but this is a
// correctness-critical kernel (feeds directly into what token gets
// generated next), so it's handled exactly, not assumed away.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float bf16_bits_to_f32(unsigned short bits) {
    return __uint_as_float(((unsigned int)bits) << 16);
}

// One block, `blockDim.x` threads (power of two). Each thread scans a
// strided subset of `logits[n]`, tracking its own best (value, index);
// a block-wide tree reduction over (value, index) pairs produces the
// global argmax.
extern "C" __global__ void argmax_bf16_kernel(
    const unsigned short* __restrict__ logits,
    int* __restrict__ out_idx,
    int n
) {
    extern __shared__ unsigned char smem[];
    float* best_val = reinterpret_cast<float*>(smem);
    int* best_idx = reinterpret_cast<int*>(smem + blockDim.x * sizeof(float));

    const int tid = threadIdx.x;
    float local_best_val = -3.402823466e38f; // -FLT_MAX
    int local_best_idx = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = bf16_bits_to_f32(logits[i]);
        if (v > local_best_val) {
            local_best_val = v;
            local_best_idx = i;
        }
    }
    best_val[tid] = local_best_val;
    best_idx[tid] = local_best_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            // Explicit lowest-index tie-break (see header comment) --
            // NOT just `>`, since the strided per-thread assignment means
            // a lower `tid` does not always already hold the lower index.
            if (best_val[tid + s] > best_val[tid] ||
                (best_val[tid + s] == best_val[tid] && best_idx[tid + s] < best_idx[tid])) {
                best_val[tid] = best_val[tid + s];
                best_idx[tid] = best_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out_idx = best_idx[0];
    }
}

// Plain C entry point. `threads` must be a power of two (reduction loop
// above). `stream` (§93): NULL reproduces default-stream behavior; a real
// explicitly-created stream is required for capture into a HIP Graph.
extern "C" void launch_argmax_bf16(
    const void* logits,
    int* out_idx,
    int n,
    int threads,
    void* stream
) {
    size_t shmem_bytes = (size_t)threads * (sizeof(float) + sizeof(int));
    hipLaunchKernelGGL(
        argmax_bf16_kernel,
        dim3(1), dim3(threads), shmem_bytes, (hipStream_t)stream,
        (const unsigned short*)logits, out_idx, n
    );
}
