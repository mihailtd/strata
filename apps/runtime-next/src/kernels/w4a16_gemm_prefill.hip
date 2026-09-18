// Real HIP kernel: §108 batched W4A16 prefill -- fixes the real, disclosed
// TTFT gap from §106 (quantized layers were prefilling via a per-token
// `forward_one_token` loop, `w4a16_gemv_kernel` launched once PER PROMPT
// TOKEN, re-streaming the full weight matrix from VRAM every single time
// even though every token needs the exact same weights).
//
// This kernel batches `W4A16_PREFILL_TILE_T` tokens per block, so each
// loaded/dequantized weight nibble is reused across all of them instead of
// being re-fetched and re-dequantized once per token -- the same real
// amortization `raw::gemm`'s own `rows > 1` hipBLAS branch already gets
// for bf16 prefill (see that function's doc comment in `model.rs`), just
// hand-written here since hipBLAS can't consume a packed-int4 operand.
//
// Grid: `dim3(out_features, ceil(num_tokens / TILE_T))` -- same
// one-block-per-output-row design as `w4a16_gemv.hip`'s decode kernel,
// with a second grid dimension tiling the token axis. Each block still
// does ONE real pass over this row's `in_features/8` packed int32s
// (exactly like the decode kernel), but accumulates `TILE_T` independent
// dot products per thread instead of one -- decoding each weight nibble
// once and reusing it `TILE_T` times, not just re-reading it once.
//
// `TILE_T=8`: real, bounded register footprint (8 float
// accumulators/thread, 32B). §111 tried raising this to 16 and measured
// NO real improvement (45.4/71.4/181.5ms vs 45.5/68.9/180.8ms across
// 0.8B/2B/4B -- flat within noise) -- a real negative result: at this
// benchmark's real prompt lengths (~20-30 tokens), weight-read
// amortization is already close to saturated by TILE_T=8 (ceil(25/8)=4
// tiles vs ceil(25/16)=2 -- a small relative difference for a short
// prompt), so doubling register pressure bought nothing measurable.
// Reverted rather than kept for a win that didn't materialize.
//
// Same quantization scheme as `w4a16_gemv.hip` (symmetric INT4, one bf16
// scale per 128-element group, zero-point=8, row-major packing) -- see
// that file's header for the full derivation. `x`/`y` are real row-major
// `[num_tokens, features]` bf16 buffers, matching `raw::gemm`'s own
// `rows>1` convention exactly (so this kernel is a drop-in replacement at
// any of this engine's existing batched-prefill linear call sites).
//
// §111: same real vectorized-load pass `w4a16_gemv.hip` got in §110 --
// each thread's stride-loop reads FOUR consecutive `qweight` int32s per
// iteration via one 128-bit `uint4` load (and one `scale` read amortized
// across all 4, safe for the same reason: `group_size=128` always makes
// 4 divide `groups_per_int32=16` evenly) instead of four separate 32-bit
// ones. `threads` is now chosen by the caller via the same
// `w4a16_decode_threads`-style sizing (`kernels.rs`), not hardcoded --
// §110 found a fixed 256 leaves most of a block idle at this model
// family's smaller real `in_features` once reads are vectorized by 4x.

#include <hip/hip_runtime.h>

#define W4A16_PREFILL_TILE_T 8

__device__ __forceinline__ float bf16_bits_to_f32(unsigned short bits) {
    return __uint_as_float(((unsigned int)bits) << 16);
}

__device__ __forceinline__ unsigned short f32_to_bf16_bits(float v) {
    unsigned int bits = __float_as_uint(v);
    unsigned int lsb = (bits >> 16) & 1u;
    unsigned int rounded = bits + 0x7fffu + lsb;
    return (unsigned short)(rounded >> 16);
}

extern "C" __global__ void w4a16_gemm_prefill_kernel(
    const unsigned short* __restrict__ x,        // [num_tokens, in_features], bf16 bits, row-major
    const unsigned int* __restrict__ qweight,    // [out_features, in_features/8], packed nibbles
    const unsigned short* __restrict__ scales,   // [out_features, in_features/group_size], bf16 bits
    unsigned short* __restrict__ y,              // [num_tokens, out_features], bf16 bits, row-major
    int in_features,
    int out_features,
    int num_tokens,
    int group_size
) {
    extern __shared__ float smem[]; // [TILE_T, num_warps]

    const int o = blockIdx.x;
    const int token_base = blockIdx.y * W4A16_PREFILL_TILE_T;
    const int tile_tokens = min(W4A16_PREFILL_TILE_T, num_tokens - token_base);
    if (tile_tokens <= 0) {
        return;
    }

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const int k8 = in_features / 8;
    const int groups_per_int32 = group_size / 8;

    const unsigned int* qw_row = qweight + (size_t)o * (size_t)k8;
    const unsigned short* scale_row = scales + (size_t)o * (size_t)(in_features / group_size);

    float acc[W4A16_PREFILL_TILE_T];
#pragma unroll
    for (int lt = 0; lt < W4A16_PREFILL_TILE_T; lt++) {
        acc[lt] = 0.0f;
    }

    // §111: stride over 4-int32 chunks (32 nibbles), same real technique
    // as `w4a16_gemv.hip`'s decode kernel -- see that file for why this
    // is safe (never crosses a scale-group boundary).
    const int k8_vec = k8 / 4;
    const unsigned int* qw_row_vec = reinterpret_cast<const unsigned int*>(qw_row);

    for (int j4 = tid; j4 < k8_vec; j4 += stride) {
        const int j = j4 * 4;
        uint4 packed4 = *reinterpret_cast<const uint4*>(qw_row_vec + j);
        float scale = bf16_bits_to_f32(scale_row[j / groups_per_int32]);

        const unsigned int packed_arr[4] = {packed4.x, packed4.y, packed4.z, packed4.w};
#pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            unsigned int packed = packed_arr[sub];
            const int jj = j + sub;

            float n0 = (float)((packed >> 0) & 0xF);
            float n1 = (float)((packed >> 4) & 0xF);
            float n2 = (float)((packed >> 8) & 0xF);
            float n3 = (float)((packed >> 12) & 0xF);
            float n4 = (float)((packed >> 16) & 0xF);
            float n5 = (float)((packed >> 20) & 0xF);
            float n6 = (float)((packed >> 24) & 0xF);
            float n7 = (float)((packed >> 28) & 0xF);

            for (int lt = 0; lt < tile_tokens; lt++) {
                const unsigned short* xr = x + (size_t)(token_base + lt) * (size_t)in_features + (size_t)jj * 8;
                ushort4 xv0 = *reinterpret_cast<const ushort4*>(xr);
                ushort4 xv1 = *reinterpret_cast<const ushort4*>(xr + 4);

                float x0 = bf16_bits_to_f32(xv0.x);
                float x1 = bf16_bits_to_f32(xv0.y);
                float x2 = bf16_bits_to_f32(xv0.z);
                float x3 = bf16_bits_to_f32(xv0.w);
                float x4 = bf16_bits_to_f32(xv1.x);
                float x5 = bf16_bits_to_f32(xv1.y);
                float x6 = bf16_bits_to_f32(xv1.z);
                float x7 = bf16_bits_to_f32(xv1.w);

                float sum_qx = n0 * x0 + n1 * x1 + n2 * x2 + n3 * x3 + n4 * x4 + n5 * x5 + n6 * x6 + n7 * x7;
                float sum_x = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
                acc[lt] += scale * (sum_qx - 8.0f * sum_x);
            }
        }
    }

    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    const int num_warps = blockDim.x / warpSize;

#pragma unroll
    for (int lt = 0; lt < W4A16_PREFILL_TILE_T; lt++) {
        float v = acc[lt];
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            v += __shfl_down(v, offset);
        }
        if (lane_id == 0) {
            smem[lt * num_warps + warp_id] = v;
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        for (int lt = 0; lt < tile_tokens; lt++) {
            float v = (lane_id < num_warps) ? smem[lt * num_warps + lane_id] : 0.0f;
            for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
                v += __shfl_down(v, offset);
            }
            if (lane_id == 0) {
                y[(size_t)(token_base + lt) * (size_t)out_features + o] = f32_to_bf16_bits(v);
            }
        }
    }
}

// Plain C entry point. `threads` must be a power of two (reduction loop
// above), same convention as `w4a16_gemv.hip`'s decode kernel.
extern "C" void launch_w4a16_gemm_prefill_bf16(
    const void* x,
    const void* qweight,
    const void* scales,
    void* y,
    int out_features,
    int in_features,
    int group_size,
    int num_tokens,
    int threads,
    void* stream
) {
    const int tile_t = W4A16_PREFILL_TILE_T;
    const int num_tiles = (num_tokens + tile_t - 1) / tile_t;
    const int num_warps = threads / 32;
    size_t shmem_bytes = (size_t)tile_t * (size_t)num_warps * sizeof(float);
    hipLaunchKernelGGL(
        w4a16_gemm_prefill_kernel,
        dim3(out_features, num_tiles), dim3(threads), shmem_bytes, (hipStream_t)stream,
        (const unsigned short*)x, (const unsigned int*)qweight, (const unsigned short*)scales,
        (unsigned short*)y, in_features, out_features, num_tokens, group_size
    );
}
