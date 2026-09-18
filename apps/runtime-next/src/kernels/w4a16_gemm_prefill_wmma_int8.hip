// §126 EXPERIMENT: real, production-shaped W4A16 prefill GEMM using
// RDNA3 hardware WMMA INT8 tensor cores instead of scalar VALU FMA --
// the real mechanism this session confirmed (via direct llama.cpp source
// reading, `ggml_cuda_should_use_mmq`'s dispatch logic) explains why
// llama.cpp's Q4_K quantized prefill beats this engine's scalar kernel at
// larger models. NOT wired into `raw::linear_quantized_prefill` yet --
// isolated, correctness-gated first, per this crate's established
// discipline (§125's own hard-won lesson: a first hand-derived fragment
// layout can be plausible and wrong).
//
// Real fragment-layout formulas used here are the SAME, now-CONFIRMED
// (bit-exact vs a real independent CPU reference, see
// `real_wmma_int8_probe_16x16x32_matches_cpu_reference` in `kernels.rs`)
// ones from `wmma_int8_probe.hip` -- ported verbatim from llama.cpp's own
// real, production `mma.cuh` (`tile<16,8,int,DATA_LAYOUT_I_MAJOR_MIRRORED>`,
// lines 529-569) and `mmq-vec-dot.cuh`'s real INT8 MMQ dot-product
// function. NOT re-derived a second time.
//
// Real design, differing from the shipped scalar `TILE_N=16` kernel only
// in HOW the dot product is computed, not the real quantization format
// on disk (same INT4 nibbles, same symmetric zero-point=8, same
// group_size=128 bf16 scales -- no checkpoint change):
//
// - Grid: dim3(ceil(out_features/128), ceil(num_tokens/16)).
// - Block: 256 threads (8 warps/waves). Each warp owns a full, real
//   16x16 WMMA output tile: 16 output rows (real WMMA "A" operand,
//   matching this engine's own existing weight-row-per-lane convention)
//   x the block's 16 tokens (real WMMA "B" operand).
// - K-step = 128 (ONE full quantization group -- real, exact for every
//   real model shape in this family: 0.8B/2B/4B/9B/27B all have
//   `in_features` an exact multiple of 128, confirmed directly against
//   `model.rs`'s real per-size constants, so no K-tail branch is needed).
//   Each K=128 step decomposes into 4 real K=32 WMMA chunks (2 real
//   hardware WMMA calls per chunk, per the confirmed `mma()` RDNA3
//   branch), accumulating a RAW (unscaled) int32 dot product across all
//   4 chunks before applying this step's real weight/activation scales
//   ONCE -- the same "accumulate raw, scale once per group" structure
//   already used by the shipped kernel's own §105 "safe algebra" pass,
//   just now driven by hardware WMMA instead of scalar FMA.
// - Real, NEW step this kernel needs that the scalar kernel does not:
//   on-the-fly, per-token, per-K=128-group INT8 activation quantization
//   (symmetric, `scale = max_abs/127`, real round-to-nearest, clamped to
//   [-127,127] -- llama.cpp's own real "Q8_1-style" scheme, a real new
//   rounding source disclosed plainly, not hidden). Computed
//   cooperatively by all 256 threads (16 threads/token, each handling 8
//   of that token's 128 real bf16 values, reduced via `__shfl_xor`
//   within the token's own 16-lane sub-group) directly into a small LDS
//   buffer (2KB: 16 tokens x 32 int32 = 128 real int8 values/token, plus
//   16 real f32 per-token scales) -- real, deliberately NOT
//   double-buffered this first pass (LDS budget is tiny compared to the
//   shipped kernel's 16KB double-buffered activation tile, real
//   double-buffering headroom exists but is not spent yet -- correctness
//   first, matching this crate's own established discipline).
// - Real weight dequant: INT4 nibble -> signed INT8 is a PLAIN
//   `nibble - 8` (fits whole in `[-8,7]`, no clamping needed) -- real,
//   SIMPLER than the shipped kernel's own zero-point handling, which
//   needs a separate `sum_x` correction term because it accumulates in
//   float; here the zero-point is baked directly into each stored int8
//   value before the integer dot product, so no correction term is
//   needed at all.
// - Real output write: `y[token*out_features + row]`, matching the
//   shipped kernel's own token-major layout exactly (no format change).

#include "bf16_utils.hip"

#define WMMA_I8_TILE_M 16   // tokens per block (WMMA "B"/N dimension)
#define WMMA_I8_TILE_N 128  // output rows per block (8 warps x 16 rows, WMMA "A"/M dimension)
#define WMMA_I8_GROUP 128   // K-step = one real quantization group (exact for every real model size)

typedef int __attribute__((ext_vector_type(4))) v4i_t;
typedef int __attribute__((ext_vector_type(8))) v8i_t;

extern "C" __global__ void w4a16_gemm_prefill_wmma_int8_kernel(
    const unsigned short* __restrict__ x,       // [num_tokens, in_features] real bf16, row-major
    const unsigned int* __restrict__ qweight,    // [out_features, in_features/8] real packed INT4, row-major
    const unsigned short* __restrict__ scales,   // [out_features, in_features/group_size] real bf16
    unsigned short* __restrict__ y,              // [num_tokens, out_features] real bf16, row-major
    int in_features,
    int out_features,
    int num_tokens,
    int group_size
) {
    // Real, activation-only LDS: 16 tokens x 32 int32 (128 real int8
    // values/token) + 16 real per-token f32 scales for the CURRENT
    // K=128 group. Weight data is never staged through LDS (each lane
    // reads its own output row directly from global memory, exactly
    // like the shipped kernel).
    __shared__ int   smem_x_i8[WMMA_I8_TILE_M * (WMMA_I8_GROUP / 4)];
    __shared__ float smem_x_scale[WMMA_I8_TILE_M];

    const int token_base = blockIdx.y * WMMA_I8_TILE_M;
    const int tile_tokens = min(WMMA_I8_TILE_M, num_tokens - token_base);
    if (tile_tokens <= 0) {
        return;
    }

    const int tid = threadIdx.x;         // 0..255
    const int warp_id = tid / 32;        // 0..7
    const int lane_id = tid % 32;        // 0..31
    const int frag_row = lane_id % 16;   // real, verbatim: get_i(l) = lane_id % 16 for the A/B mirrored tile

    const int row_base = blockIdx.x * WMMA_I8_TILE_N + warp_id * 16;
    // Real, IMPORTANT distinction (a real bug this exact comment exists
    // to prevent regressing): `my_row` is ONLY which row's WEIGHT DATA
    // this lane feeds into the WMMA "A" operand -- the hardware combines
    // ALL 32 lanes' A-data into the real 16x16 cross product, so THIS
    // lane's own `facc_f[l]` does NOT correspond to `my_row` at all; it
    // corresponds to the real output row `2*l + lane_id/16` (see the
    // scale-application loop below, which fetches the scale for THAT
    // row, never for `my_row`). Real, ALSO important: the WMMA calls
    // below run UNCONDITIONALLY for every lane in the warp (never
    // gated by a per-lane `if`), since a WMMA instruction is a
    // warp-collective op -- divergent execution across lanes of the
    // same warp would corrupt every lane's result, not just the
    // out-of-bounds one. Out-of-bounds rows get a safe all-8-nibbles
    // dummy weight word (dequantizes to all-zero, contributing nothing),
    // matching the shipped scalar kernel's own established convention.
    const int my_row = row_base + frag_row;
    const bool my_row_valid = my_row < out_features;

    const unsigned int* qw_row = qweight + (size_t)(my_row_valid ? my_row : 0) * (size_t)(in_features / 8);

    float facc_f[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};

    for (int k_base = 0; k_base < in_features; k_base += WMMA_I8_GROUP) {
        // --- Real, cooperative, on-the-fly INT8 activation quantization ---
        // 16 threads/token (256 threads / 16 tokens), each owns 8 of the
        // real 128 bf16 values in this K=128 group for its token.
        {
            const int q_token = tid / 16;      // 0..15
            const int q_sub = tid % 16;         // 0..15, each covers 8 real elements
            const int q_k0 = q_sub * 8;
            const bool q_valid_token = (token_base + q_token) < num_tokens;

            float v[8];
#pragma unroll
            for (int e = 0; e < 8; e++) {
                v[e] = 0.0f;
            }
            if (q_valid_token) {
                const unsigned short* xr = x + (size_t)(token_base + q_token) * (size_t)in_features + (size_t)(k_base + q_k0);
                ushort4 xv0 = *reinterpret_cast<const ushort4*>(xr);
                ushort4 xv1 = *reinterpret_cast<const ushort4*>(xr + 4);
                v[0] = bf16_bits_to_f32(xv0.x);
                v[1] = bf16_bits_to_f32(xv0.y);
                v[2] = bf16_bits_to_f32(xv0.z);
                v[3] = bf16_bits_to_f32(xv0.w);
                v[4] = bf16_bits_to_f32(xv1.x);
                v[5] = bf16_bits_to_f32(xv1.y);
                v[6] = bf16_bits_to_f32(xv1.z);
                v[7] = bf16_bits_to_f32(xv1.w);
            }

            float local_max = 0.0f;
#pragma unroll
            for (int e = 0; e < 8; e++) {
                local_max = fmaxf(local_max, fabsf(v[e]));
            }
            // Real 16-wide reduction within this token's own 16 lanes
            // (guaranteed to lie in one warp: 2 tokens/warp, 16 lanes
            // each). Masks 1,2,4,8 only -- stays within the 16-lane half.
#pragma unroll
            for (int offset = 8; offset > 0; offset >>= 1) {
                local_max = fmaxf(local_max, __shfl_xor(local_max, offset, 32));
            }
            const float scale = local_max > 0.0f ? (local_max / 127.0f) : 1.0f;
            const float inv_scale = 1.0f / scale;

            int packed0 = 0, packed1 = 0;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int q = __float2int_rn(v[e] * inv_scale);
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                packed0 |= (q & 0xFF) << (e * 8);
            }
#pragma unroll
            for (int e = 0; e < 4; e++) {
                int q = __float2int_rn(v[e + 4] * inv_scale);
                q = q < -127 ? -127 : (q > 127 ? 127 : q);
                packed1 |= (q & 0xFF) << (e * 8);
            }
            smem_x_i8[q_token * 32 + q_sub * 2 + 0] = packed0;
            smem_x_i8[q_token * 32 + q_sub * 2 + 1] = packed1;
            if (q_sub == 0) {
                smem_x_scale[q_token] = scale;
            }
        }
        __syncthreads();

        // --- Real 4x K=32 WMMA chunks, raw int32 accumulation ---
        // Real, unconditional (no per-lane `if`): every lane in the warp
        // executes every WMMA call, regardless of `my_row_valid`.
        v8i_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int c = 0; c < 4; c++) {
            // Real weight dequant: 4 original packed uint32 (32 real
            // nibbles) -> 8 new int8-packed int32 (32 real int8
            // values), `nibble - 8` applied directly, matching the
            // real K=32 chunk this WMMA pass consumes. Out-of-bounds
            // rows read a safe all-8-nibbles dummy (dequantizes to
            // all-zero) instead of skipping the call.
            v4i_t a_vec[2];
            int* a_s = reinterpret_cast<int*>(a_vec);
#pragma unroll
            for (int kw = 0; kw < 4; kw++) {
                unsigned int packed = my_row_valid ? qw_row[k_base / 8 + c * 4 + kw] : 0x88888888u;
                int n0 = (int)((packed >> 0) & 0xF) - 8;
                int n1 = (int)((packed >> 4) & 0xF) - 8;
                int n2 = (int)((packed >> 8) & 0xF) - 8;
                int n3 = (int)((packed >> 12) & 0xF) - 8;
                int n4 = (int)((packed >> 16) & 0xF) - 8;
                int n5 = (int)((packed >> 20) & 0xF) - 8;
                int n6 = (int)((packed >> 24) & 0xF) - 8;
                int n7 = (int)((packed >> 28) & 0xF) - 8;
                a_s[kw * 2 + 0] = (n0 & 0xFF) | ((n1 & 0xFF) << 8) | ((n2 & 0xFF) << 16) | ((n3 & 0xFF) << 24);
                a_s[kw * 2 + 1] = (n4 & 0xFF) | ((n5 & 0xFF) << 8) | ((n6 & 0xFF) << 16) | ((n7 & 0xFF) << 24);
            }

            // Real activation fragment load from LDS: this lane's
            // "row" (frag_row = token index within the tile) reads
            // its own 8 int32 for this chunk -- identical for lanes
            // frag_row and frag_row+16 (real duplication, matches
            // the confirmed probe exactly).
            v4i_t b_vec[2];
            int* b_s = reinterpret_cast<int*>(b_vec);
#pragma unroll
            for (int e = 0; e < 8; e++) {
                b_s[e] = smem_x_i8[frag_row * 32 + c * 8 + e];
            }

            acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[0], true, b_vec[0], acc, true);
            acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[1], true, b_vec[1], acc, true);
        }

        // Real scale application, once per K=128 group: each l in 0..7
        // corresponds to a DIFFERENT real output row `2*l + lane_id/16`
        // (within this warp's 16-row slice) and the SAME token
        // `lane_id % 16` -- real, verbatim J_MAJOR accumulator formula.
        // The weight scale MUST be looked up for THAT row (not `my_row`
        // -- see the real bug this fixed, documented above `my_row`'s
        // own declaration).
        {
            const int scale_idx = k_base / group_size;
            const int tok_local = lane_id % 16;
            const float a_scale = smem_x_scale[tok_local];
#pragma unroll
            for (int l = 0; l < 8; l++) {
                const int out_row_l = row_base + 2 * l + (lane_id / 16);
                if (out_row_l < out_features && tok_local < tile_tokens) {
                    const float w_scale_l = bf16_bits_to_f32(scales[(size_t)out_row_l * (size_t)(in_features / group_size) + scale_idx]);
                    facc_f[l] += w_scale_l * a_scale * (float)acc[l];
                }
            }
        }
        __syncthreads();
    }

    // Real C-tile extraction: row_base already encodes this warp's own
    // 16-row slice (`row_base = blockIdx.x*128 + warp_id*16`); the real
    // output row within that slice is `2*l + lane_id/16`. Real, NOT
    // gated by `my_row_valid` (that only ever described this lane's OWN
    // weight-loading row, irrelevant to which output rows this lane's
    // `facc_f` actually holds) -- gated per-`l` by the real `out_row`
    // bounds check only, matching the scale-application loop above.
#pragma unroll
    for (int l = 0; l < 8; l++) {
        const int out_row = row_base + 2 * l + (lane_id / 16);
        const int out_col = lane_id % 16;
        if (out_row < out_features && out_col < tile_tokens) {
            y[(size_t)(token_base + out_col) * (size_t)out_features + out_row] = f32_to_bf16_bits(facc_f[l]);
        }
    }
}

extern "C" void launch_w4a16_gemm_prefill_wmma_int8_bf16(
    const void* x,
    const void* qweight,
    const void* scales,
    void* y,
    int out_features,
    int in_features,
    int group_size,
    int num_tokens,
    void* stream
) {
    const int grid_x = (out_features + WMMA_I8_TILE_N - 1) / WMMA_I8_TILE_N;
    const int grid_y = (num_tokens + WMMA_I8_TILE_M - 1) / WMMA_I8_TILE_M;
    dim3 grid(grid_x, grid_y);
    dim3 block(256);

    hipLaunchKernelGGL(
        w4a16_gemm_prefill_wmma_int8_kernel,
        grid, block, 0, (hipStream_t)stream,
        (const unsigned short*)x, (const unsigned int*)qweight, (const unsigned short*)scales,
        (unsigned short*)y, in_features, out_features, num_tokens, group_size
    );
}
