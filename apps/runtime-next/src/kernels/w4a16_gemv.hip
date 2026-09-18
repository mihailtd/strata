// Real HIP kernel: §106 Phase A3 -- W4A16 (INT4 weight, BF16 activation)
// fused dequant + matrix-vector product `y = dequant(qweight, scales) @ x`.
// Same real shape class as `gemv.hip`'s own `gemv_bf16_kernel` (this
// engine always calls with `rows=1` -- one decode-step token, never a
// batch), same "one block per output row, vectorized loads" design --
// the new, real work here is the dequant math itself.
//
// Quantization scheme (ported from `apps/runtime-triton/triton_w4a16.py`,
// same real math -- see `quantize_w4a16.py`'s own header for the full
// derivation and the deliberate LAYOUT adaptation from that source):
// symmetric INT4, one scale per 128-element group along `in_features`,
// bf16 scales, fixed zero-point=8 (`q in [0,15]`, dequant
// `w ~= (q - 8) * scale`). Packed 8 nibbles per `int32`, LSB-first, ROW-
// MAJOR (`qweight[out_features, in_features/8]`, `scales[out_features,
// in_features/group_size]`) -- matching this crate's own existing bf16
// weight layout (`gemv.hip`'s `w[out_features, in_features]`) exactly,
// just with `in_features` compressed 8x, so each block's row read stays
// fully contiguous.
//
// PRECONDITION (asserted by the Rust launcher, not a tail loop here):
// `in_features % 8 == 0` and `in_features % group_size == 0` -- real for
// every quantized shape in this model (quantize_w4a16.py itself asserts
// this at conversion time, so a checkpoint that violates it never gets
// this far).
//
// §106 real profiling (rocprofv3 against the real served 27B decode
// loop) found this kernel at 90.6% of all decode kernel time -- by far
// the dominant real cost, not diluted by anything else. This version
// adds one real, bit-exact-up-to-summation-order algebraic optimization:
// since all 8 nibbles packed into one `int32` always share the SAME
// scale (8 always divides `group_size`), `sum_{i=0..7}((n_i-8)*scale*x_i)
// == scale * (sum(n_i*x_i) - 8*sum(x_i))` -- computed by accumulating
// `sum_qx`/`sum_x` across the 8 elements FIRST, applying `scale` and the
// `-8` bias ONCE per int32 instead of once per element. Real reduction:
// ~32 FLOPs/int32 -> ~26 FLOPs/int32 (measured via rocprofv3 before/
// after, not assumed).
//
// A further, bigger version of this same idea -- factoring scale/bias
// across the FULL 128-element group (16 int32s) instead of just one --
// was deliberately NOT done here: with only `in_features/group_size`
// real "groups" (136 for this model's widest shape, 40 for the rest)
// as work items, and this kernel launching 256+ threads/block, most
// threads would sit idle waiting for group-sized work chunks -- a real
// occupancy loss that could easily cost more than the FLOP reduction
// buys back. Doing that properly needs a real thread-to-group
// cooperative-reduction redesign (multiple threads per group, a small
// intra-group reduction before the one scale-apply), a bigger, separate
// step, not bundled into this same-occupancy quick win.
//
// §110 (lossless, bandwidth-targeting pass): each thread's stride-loop
// now reads FOUR consecutive `qweight` int32s per iteration via one
// vectorized 128-bit `uint4` load (`x`'s own `ushort4` loads already did
// this same trick from the start -- `qweight` didn't). Same real bytes
// moved, same real dequant math, same real per-int32 scale application
// as before -- purely fewer, larger memory transactions instead of four
// separate 32-bit ones, plus one scale global-memory read amortized
// across 4 int32s instead of 1 (safe: `groups_per_int32` is always 16 for
// this codebase's only real `group_size`, 128, and 4 always divides that
// evenly -- see `W4A16_GROUP_SIZE` in `model.rs` -- so a 4-int32 chunk
// never crosses a scale-group boundary). Real, measured effect: see
// `docs/DECISIONS.md` §110.
//
// §113 (tried, real, REVERTED -- see the loop body below for the full
// real before/after numbers): after real rocprofv3 profiling on the
// fully §110/§111-optimized kernel found only 61.5%/49.0% of the
// 960GB/s peak achieved on the real 27B down_proj/up_proj shapes (vs.
// the simpler, dequant-free `gemv_bf16` kernel's own established
// 79-91%), doubled the same vectorization trick again (8 int32s/iter
// instead of 4). Passed both correctness gates (CPU cross-validation,
// byte-exact token ids), and looked promising in isolated per-shape
// kernel benchmarks -- but the real, full end-to-end 27B decode
// throughput came back a 4% REGRESSION (35.9 vs. 37.4 tok/s), because
// the extra register pressure cost more across the model's smaller
// shapes than it saved on the two biggest ones. A real negative result,
// reverted rather than kept for a benchmark number that didn't hold up
// at the model level.

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

extern "C" __global__ void w4a16_gemv_kernel(
    const unsigned short* __restrict__ x,        // [in_features], bf16 bits
    const unsigned int* __restrict__ qweight,    // [out_features, in_features/8], packed nibbles
    const unsigned short* __restrict__ scales,   // [out_features, in_features/group_size], bf16 bits
    unsigned short* __restrict__ y,              // [out_features], bf16 bits
    int in_features,
    int group_size
) {
    extern __shared__ float warp_partials[];
    const int o = blockIdx.x;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;

    const int k8 = in_features / 8;
    const int groups_per_int32 = group_size / 8; // real for group_size=128: 16
    const unsigned int* qw_row = qweight + (size_t)o * (size_t)k8;
    const unsigned short* scale_row = scales + (size_t)o * (size_t)(in_features / group_size);

    // §110: stride over 4-int32 chunks (32 nibbles) instead of 1 -- one
    // vectorized 128-bit `uint4` load for `qweight` and one `scale` read
    // per chunk instead of per int32. `k8` is always a multiple of 4
    // here (see the header comment).
    //
    // §113 REVERTED: tried doubling this to 8-int32 chunks. Real,
    // isolated kernel benchmarks on the 27B `down_proj`/`up_proj` shapes
    // looked mixed-to-promising (down_proj -1.9%, up_proj +13.4%), but
    // the real, full end-to-end 27B decode throughput -- the number that
    // actually matters, summed across EVERY shape in the model, not just
    // the two biggest MLP ones -- came back at 35.9 tok/s vs. this
    // version's own 37.4 tok/s: a real 4% REGRESSION. The extra register
    // pressure from unrolling 8 sub-computations instead of 4 cost more
    // across the model's smaller shapes (qkv/o/in_proj/out_proj) than it
    // saved on the two biggest ones. Reverted on the real net number, not
    // kept for an isolated-shape win that didn't hold up.
    const int k8_vec = k8 / 4;
    const unsigned int* qw_row_vec = reinterpret_cast<const unsigned int*>(qw_row);

    float acc = 0.0f;
    for (int j4 = tid; j4 < k8_vec; j4 += stride) {
        const int j = j4 * 4;
        uint4 packed4 = *reinterpret_cast<const uint4*>(qw_row_vec + j);
        float scale = bf16_bits_to_f32(scale_row[j / groups_per_int32]);

        const unsigned int packed_arr[4] = {packed4.x, packed4.y, packed4.z, packed4.w};
#pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            unsigned int packed = packed_arr[sub];
            const int jj = j + sub;

            ushort4 xv0 = *reinterpret_cast<const ushort4*>(x + jj * 8);
            ushort4 xv1 = *reinterpret_cast<const ushort4*>(x + jj * 8 + 4);

            float n0 = (float)((packed >> 0) & 0xF);
            float n1 = (float)((packed >> 4) & 0xF);
            float n2 = (float)((packed >> 8) & 0xF);
            float n3 = (float)((packed >> 12) & 0xF);
            float n4 = (float)((packed >> 16) & 0xF);
            float n5 = (float)((packed >> 20) & 0xF);
            float n6 = (float)((packed >> 24) & 0xF);
            float n7 = (float)((packed >> 28) & 0xF);

            float x0 = bf16_bits_to_f32(xv0.x);
            float x1 = bf16_bits_to_f32(xv0.y);
            float x2 = bf16_bits_to_f32(xv0.z);
            float x3 = bf16_bits_to_f32(xv0.w);
            float x4 = bf16_bits_to_f32(xv1.x);
            float x5 = bf16_bits_to_f32(xv1.y);
            float x6 = bf16_bits_to_f32(xv1.z);
            float x7 = bf16_bits_to_f32(xv1.w);

            // All 8 nibbles in this int32 share ONE scale (8 always
            // divides group_size) -- accumulate the two real sums first,
            // apply scale and the -8 zero-point bias ONCE for all 8
            // elements instead of once each.
            float sum_qx = n0 * x0 + n1 * x1 + n2 * x2 + n3 * x3 + n4 * x4 + n5 * x5 + n6 * x6 + n7 * x7;
            float sum_x = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
            acc += scale * (sum_qx - 8.0f * sum_x);
        }
    }

    // Same two-stage warp-shuffle + small shared-memory reduction as
    // `gemv_bf16_kernel` -- proven pattern, reused exactly.
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
// above). `stream`: NULL reproduces default-stream behavior; a real
// explicitly-created stream is required for capture into a HIP Graph.
extern "C" void launch_w4a16_gemv_bf16(
    const void* x,
    const void* qweight,
    const void* scales,
    void* y,
    int out_features,
    int in_features,
    int group_size,
    int threads,
    void* stream
) {
    size_t shmem_bytes = (size_t)threads * sizeof(float);
    hipLaunchKernelGGL(
        w4a16_gemv_kernel,
        dim3(out_features), dim3(threads), shmem_bytes, (hipStream_t)stream,
        (const unsigned short*)x, (const unsigned int*)qweight, (const unsigned short*)scales,
        (unsigned short*)y, in_features, group_size
    );
}
