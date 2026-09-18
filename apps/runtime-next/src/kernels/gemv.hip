// Real HIP kernel: bf16 matrix-vector product `y = w @ x`, `w: [out_features,
// in_features]` row-major (nn.Linear layout, matches `blas.rs`'s own GEMM),
// `x: [in_features]`, `y: [out_features]`. Exactly the operation this
// decode-only engine's `hipblasGemmEx`/`hipblasLtMatmul` calls actually
// compute every time (this crate always calls with `rows=1` -- a single
// decode-step token, never a batch) -- a real matrix-VECTOR product, not a
// general matrix-matrix product, even though it was being dispatched
// through general-purpose GEMM libraries.
//
// §93's own real, measured diagnostic (`diagnose_real_per_layer_type_cost`)
// found GDN layers -- whose per-token cost is dominated by exactly these
// GEMV calls -- achieving only ~37-39% of this GPU's theoretical memory
// bandwidth. Both `hipblasGemmEx` (plain hipBLAS, HIPBLAS_GEMM_DEFAULT) and
// `hipblasLtMatmul` (hipBLASLt, heuristically-searched algorithm, see
// `blaslt.rs`) were tried as fixes for that; hipBLASLt was a reproducible
// REGRESSION (see `docs/DECISIONS.md` §92). This kernel is the standard
// alternative production serving engines use for exactly this shape class
// (batch=1 decode): a hand-written, vectorized GEMV, rather than routing a
// degenerate M x 1 x K "GEMM" through a general M x N x K BLAS library.
//
// One block per OUTPUT row. Threads within a block stride across
// `in_features`, each iteration reading 4 bf16 elements at once via a
// `ushort4` vectorized load (8 bytes/thread/iteration instead of 2) from
// BOTH `w`'s row and `x` -- the real lever here: wider loads improve
// achieved memory-transaction efficiency on this hardware, the standard
// GPU memory-bandwidth optimization for a bandwidth-bound reduction like
// this. `x` is re-read by every block, but it's small (this model's real
// shapes: <=18432 elements = 36KB) and stays resident in L2 across the
// `out_features`-many blocks reading it.
//
// PRECONDITION (asserted by the Rust launcher, not handled by a tail loop
// here): `in_features % 4 == 0`. Every real shape in this model satisfies
// this (2560, 4096, 8192, 9216, 10240, 12352, 18432, 248320 -- all
// divisible by 4), so a general remainder-handling tail loop would be
// real, unexercised code with no test coverage ever exercising it -- this
// crate's own "no untested code paths" discipline says assert the real
// invariant instead of writing speculative generality for it.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float bf16_bits_to_f32(unsigned short bits) {
    return __uint_as_float(((unsigned int)bits) << 16);
}

__device__ __forceinline__ unsigned short f32_to_bf16_bits(float v) {
    unsigned int bits = __float_as_uint(v);
    unsigned int lsb = (bits >> 16) & 1u;
    unsigned int rounded = bits + 0x7fffu + lsb;
    return (unsigned short)(rounded >> 16);
}

extern "C" __global__ void gemv_bf16_kernel(
    const unsigned short* __restrict__ x,  // [in_features]
    const unsigned short* __restrict__ w,  // [out_features, in_features]
    unsigned short* __restrict__ y,        // [out_features]
    int in_features
) {
    // Warp-level partial sums first (via `__shfl_down`, no `__syncthreads`
    // needed within a warp/wavefront), THEN a small shared-memory
    // reduction across just the per-warp partials -- replaces a full
    // `blockDim.x`-wide tree reduction's `log2(blockDim.x)` round trips of
    // `__syncthreads()` with `log2(warpSize)` sync-free shuffle steps plus
    // ONE final small reduction. `warpSize` is HIP's own builtin (not
    // hardcoded 32/64) so this is correct regardless of this GPU's actual
    // compiled wavefront width.
    extern __shared__ float warp_partials[];
    const int o = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const unsigned short* w_row = w + (size_t)o * (size_t)in_features;

    // Two `ushort4` loads/iteration (8 elements/thread/iteration) instead
    // of one -- more independent memory requests in flight per iteration,
    // and two independent accumulators to break the serial dependency
    // chain a single running `acc` would otherwise force. A real,
    // rocprofv3-measured experiment (not guessed): this dropped this
    // kernel's own average duration further on top of the `ushort4`
    // single-load version. Falls back to a single-load tail loop for
    // whatever doesn't fill a full 8-element pair (real for this model's
    // shapes when `in_features / 4` is odd, e.g. down_proj's
    // `9216/4=2304` iterations split across `stride` threads).
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    int i = tid * 4;
    const int step = stride * 4;
    for (; i + step < in_features; i += step * 2) {
        ushort4 wv0 = *reinterpret_cast<const ushort4*>(w_row + i);
        ushort4 xv0 = *reinterpret_cast<const ushort4*>(x + i);
        ushort4 wv1 = *reinterpret_cast<const ushort4*>(w_row + i + step);
        ushort4 xv1 = *reinterpret_cast<const ushort4*>(x + i + step);
        acc0 += bf16_bits_to_f32(wv0.x) * bf16_bits_to_f32(xv0.x);
        acc0 += bf16_bits_to_f32(wv0.y) * bf16_bits_to_f32(xv0.y);
        acc0 += bf16_bits_to_f32(wv0.z) * bf16_bits_to_f32(xv0.z);
        acc0 += bf16_bits_to_f32(wv0.w) * bf16_bits_to_f32(xv0.w);
        acc1 += bf16_bits_to_f32(wv1.x) * bf16_bits_to_f32(xv1.x);
        acc1 += bf16_bits_to_f32(wv1.y) * bf16_bits_to_f32(xv1.y);
        acc1 += bf16_bits_to_f32(wv1.z) * bf16_bits_to_f32(xv1.z);
        acc1 += bf16_bits_to_f32(wv1.w) * bf16_bits_to_f32(xv1.w);
    }
    for (; i < in_features; i += step) {
        ushort4 wv = *reinterpret_cast<const ushort4*>(w_row + i);
        ushort4 xv = *reinterpret_cast<const ushort4*>(x + i);
        acc0 += bf16_bits_to_f32(wv.x) * bf16_bits_to_f32(xv.x);
        acc0 += bf16_bits_to_f32(wv.y) * bf16_bits_to_f32(xv.y);
        acc0 += bf16_bits_to_f32(wv.z) * bf16_bits_to_f32(xv.z);
        acc0 += bf16_bits_to_f32(wv.w) * bf16_bits_to_f32(xv.w);
    }
    float acc = acc0 + acc1;

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down(acc, offset);
    }

    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    const int num_warps = blockDim.x / warpSize;
    if (lane_id == 0) {
        warp_partials[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? warp_partials[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            v += __shfl_down(v, offset);
        }
        if (lane_id == 0) {
            y[o] = f32_to_bf16_bits(v);
        }
    }
}

// Plain C entry point. `threads` must be a power of two (reduction loop
// above); `in_features` must be a multiple of 4 (see header comment).
// `stream` (§93): NULL reproduces default-stream behavior; a real
// explicitly-created stream is required for capture into a HIP Graph.
extern "C" void launch_gemv_bf16(
    const void* x,
    const void* w,
    void* y,
    int out_features,
    int in_features,
    int threads,
    void* stream
) {
    size_t shmem_bytes = (size_t)threads * sizeof(float);
    hipLaunchKernelGGL(
        gemv_bf16_kernel,
        dim3(out_features), dim3(threads), shmem_bytes, (hipStream_t)stream,
        (const unsigned short*)x, (const unsigned short*)w, (unsigned short*)y, in_features
    );
}
