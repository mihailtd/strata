// Real HIP kernel: §120 2D LDS-Tiled W4A16 prefill GEMM -- replaces §108's
// 1D batched GEMV with a true 2D LDS-tiled matrix multiplication kernel.
//
// Root-cause bottleneck of §108:
// 1. In §108, each block computed 1 output row, streaming the activation
//    matrix X directly from global DRAM for each output row independently.
//    Across thousands of output rows, this generated gigabytes of redundant
//    activation DRAM reads (e.g. 5.7 GB for 27B down_proj on 32 tokens).
// 2. TILE_T was 8, forcing ceil(T/8) full weight sweeps from VRAM.
// 3. Every block performed an inter-warp shared memory tree reduction.
//
// §120 2D LDS-Tiled Architecture:
// - Grid: dim3(ceil(out_features / 8), ceil(num_tokens / 16))
// - Block: 256 threads (8 warps of 32 lanes each)
// - Tile: 16 tokens (TILE_M) x 8 output rows (TILE_N)
// - Inner step: 256 input elements (K_TILE = 256)
//
// Execution Flow:
// 1. Cooperative LDS Load:
//    At each K step (k_base = 0; k_base < in_features; k_base += 256):
//    The 256 threads in the block cooperatively load the 16 x 256 activation
//    tile X[token_base..token_base+16, k_base..k_base+256] into LDS (`smem_x`).
//    Each thread executes two 128-bit vector loads (uint4).
//    __syncthreads().
//    Result: Global DRAM activation reads are reduced by 8x!
//
// 2. Warp-to-Row Specialization:
//    Warp w = tid / 32 (w in 0..7) is dedicated exclusively to row:
//    row_o = blockIdx.x * 8 + w.
//    The 32 lanes in warp w load 32 consecutive packed uint32s from qweight:
//    qw_row[k_base / 8 + lane_id] (100% coalesced 128-byte warp read).
//    Each lane unpacks its 8 nibbles once into registers, and computes dot
//    products against 16 tokens read from LDS (`smem_x`) using 128-bit LDS reads.
//
// 3. Pure Intra-Warp Reduction:
//    At the end of the K loop, each warp independently reduces its accumulators
//    across its 32 lanes using __shfl_down.
//    No cross-warp reduction, no shared-memory reduction buffer, no atomic adds!
//    Lane 0 of each warp writes its 16 token outputs directly to global Y.

#include "bf16_utils.hip"

#define W4A16_TILE_M 16
#define W4A16_TILE_N 8
#define W4A16_TILE_K 256

extern "C" __global__ void w4a16_gemm_prefill_kernel(
    const unsigned short* __restrict__ x,        // [num_tokens, in_features], bf16 bits, row-major
    const unsigned int* __restrict__ qweight,    // [out_features, in_features/8], packed nibbles
    const unsigned short* __restrict__ scales,   // [out_features, in_features/group_size], bf16 bits
    unsigned short* __restrict__ y,              // [num_tokens, out_features], bf16 bits, row-major
    int in_features,     // INVARIANT: must be a multiple of W4A16_TILE_K (256).
                         // The cooperative LDS load uses uint4 vector loads that
                         // read 16 consecutive elements per thread. If in_features
                         // is not 256-aligned, the tail tile's vector loads could
                         // read past the allocation boundary. All real model dims
                         // (2560, 5120, 9216, 17408) satisfy this.
    int out_features,
    int num_tokens,
    int group_size       // INVARIANT: must be 128. The scale lookup (below) assumes
                         // exactly 2 scale groups per K_TILE=256: lanes 0..15 use
                         // group 0, lanes 16..31 use group 1. For other group sizes,
                         // generalize to: (k_base + lane_id * 8) / group_size.
) {
    // Double-buffered LDS: 2 x (16 tokens x 256 elements) = 16 KB.
    // Overlaps global activation loads with LDS-to-register GEMM compute,
    // and halves __syncthreads() barrier overhead across the K loop.
    __shared__ unsigned short smem_x[2][W4A16_TILE_M * W4A16_TILE_K];

    const int token_base = blockIdx.y * W4A16_TILE_M;
    const int tile_tokens = min(W4A16_TILE_M, num_tokens - token_base);
    if (tile_tokens <= 0) {
        return;
    }

    const int tid = threadIdx.x; // 0..255
    const int warp_id = tid / 32; // 0..7
    const int lane_id = tid % 32; // 0..31
    const int row_o = blockIdx.x * W4A16_TILE_N + warp_id;

    // Each thread in warp w maintains 16 float accumulators for row_o
    float acc[W4A16_TILE_M];
#pragma unroll
    for (int lt = 0; lt < W4A16_TILE_M; lt++) {
        acc[lt] = 0.0f;
    }

    const unsigned int* qw_row = (row_o < out_features) ? (qweight + (size_t)row_o * (size_t)(in_features / 8)) : nullptr;
    const unsigned short* scale_row = (row_o < out_features) ? (scales + (size_t)row_o * (size_t)(in_features / group_size)) : nullptr;

    // Cooperative load mapping: 256 threads load 4096 ushorts.
    // Each thread loads 16 ushorts (32 bytes) = two uint4s.
    const int load_elem = tid * 16;
    const int t_idx = load_elem / W4A16_TILE_K;      // token 0..15
    const int k_local = load_elem % W4A16_TILE_K;    // offset 0..255 (multiple of 16)
    const bool valid_token = (token_base + t_idx < num_tokens);
    const unsigned short* gx_ptr = valid_token ? (x + (size_t)(token_base + t_idx) * (size_t)in_features + (size_t)k_local) : nullptr;

    // --- Prologue: Pre-load Tile 0 into smem_x[0] ---
    if (valid_token && (k_local + 15 < in_features)) {
        const unsigned short* gx = gx_ptr;
        uint4 v0 = *reinterpret_cast<const uint4*>(gx);
        uint4 v1 = *reinterpret_cast<const uint4*>(gx + 8);
        *reinterpret_cast<uint4*>(&smem_x[0][load_elem]) = v0;
        *reinterpret_cast<uint4*>(&smem_x[0][load_elem + 8]) = v1;
    } else {
        uint4 zero = {0, 0, 0, 0};
        *reinterpret_cast<uint4*>(&smem_x[0][load_elem]) = zero;
        *reinterpret_cast<uint4*>(&smem_x[0][load_elem + 8]) = zero;
    }
    __syncthreads();

    // --- Pipelined Loop: Compute Tile N while Loading Tile N+1 ---
    int buf_curr = 0;
    for (int k_base = 0; k_base < in_features; k_base += W4A16_TILE_K) {
        const int next_k_base = k_base + W4A16_TILE_K;
        const int buf_next = buf_curr ^ 1;

        // 1. Asynchronously issue load of next tile into smem_x[buf_next]
        if (next_k_base < in_features) {
            if (valid_token && (next_k_base + k_local + 15 < in_features)) {
                const unsigned short* gx = gx_ptr + next_k_base;
                uint4 v0 = *reinterpret_cast<const uint4*>(gx);
                uint4 v1 = *reinterpret_cast<const uint4*>(gx + 8);
                *reinterpret_cast<uint4*>(&smem_x[buf_next][load_elem]) = v0;
                *reinterpret_cast<uint4*>(&smem_x[buf_next][load_elem + 8]) = v1;
            } else {
                uint4 zero = {0, 0, 0, 0};
                *reinterpret_cast<uint4*>(&smem_x[buf_next][load_elem]) = zero;
                *reinterpret_cast<uint4*>(&smem_x[buf_next][load_elem + 8]) = zero;
            }
        }

        // 2. Compute: Warp w processes row_o against LDS smem_x[buf_curr]
        if (row_o < out_features) {
            unsigned int packed = 0x88888888u; // zero-point 8: (8 - 8) = 0
            if (k_base + lane_id * 8 < in_features) {
                packed = qw_row[k_base / 8 + lane_id];
            }

            // General scale lookup: lane_id covers 8 consecutive inputs along K.
            // Works for any group_size (e.g. 64, 128, 256).
            float scale = 0.0f;
            int scale_idx = (k_base + lane_id * 8) / group_size;
            if (scale_idx < in_features / group_size) {
                scale = bf16_bits_to_f32(scale_row[scale_idx]);
            }

            float n0 = (float)((packed >> 0) & 0xF);
            float n1 = (float)((packed >> 4) & 0xF);
            float n2 = (float)((packed >> 8) & 0xF);
            float n3 = (float)((packed >> 12) & 0xF);
            float n4 = (float)((packed >> 16) & 0xF);
            float n5 = (float)((packed >> 20) & 0xF);
            float n6 = (float)((packed >> 24) & 0xF);
            float n7 = (float)((packed >> 28) & 0xF);

            for (int lt = 0; lt < tile_tokens; lt++) {
                const unsigned short* xr = &smem_x[buf_curr][lt * W4A16_TILE_K + lane_id * 8];
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

        // 3. Sync: guarantees smem_x[buf_next] write is complete and smem_x[buf_curr] read is complete
        __syncthreads();
        buf_curr = buf_next;
    }

    // --- 3. Intra-Warp Reduction & Output Write ---
    if (row_o < out_features) {
        for (int lt = 0; lt < tile_tokens; lt++) {
            float v = acc[lt];
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_down(v, offset);
            }
            if (lane_id == 0) {
                y[(size_t)(token_base + lt) * (size_t)out_features + row_o] = f32_to_bf16_bits(v);
            }
        }
    }
}

// §123 EXPERIMENT (real, isolated, NOT wired into the model yet -- see
// `docs/DECISIONS.md` §123 for the real measured verdict): identical to
// `w4a16_gemm_prefill_kernel` above in every respect EXCEPT
// `TILE_M=32` instead of 16 -- tests §121 Part 2's own "Dynamic Batch
// Tile Sizing" roadmap item directly: does halving the number of
// weight-VRAM sweeps (`ceil(num_tokens/32)` instead of `ceil(num_tokens/16)`)
// for a real ~35-54-token prompt actually win, or does the real doubled
// register (`acc[32]` vs `acc[16]`) and LDS (`32KB` vs `16KB` for the
// double-buffered tile) pressure cost more occupancy than the halved
// weight-sweep count buys back? A real, measured question, not assumed
// either way -- occupancy costs are exactly the kind of real hardware
// tradeoff that doesn't resolve from reasoning about FLOPs/bytes alone.
#define W4A16_TILE_M32 32

extern "C" __global__ void w4a16_gemm_prefill_tile32_kernel(
    const unsigned short* __restrict__ x,
    const unsigned int* __restrict__ qweight,
    const unsigned short* __restrict__ scales,
    unsigned short* __restrict__ y,
    int in_features,
    int out_features,
    int num_tokens,
    int group_size
) {
    __shared__ unsigned short smem_x[2][W4A16_TILE_M32 * W4A16_TILE_K];

    const int token_base = blockIdx.y * W4A16_TILE_M32;
    const int tile_tokens = min(W4A16_TILE_M32, num_tokens - token_base);
    if (tile_tokens <= 0) {
        return;
    }

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int row_o = blockIdx.x * W4A16_TILE_N + warp_id;

    float acc[W4A16_TILE_M32];
#pragma unroll
    for (int lt = 0; lt < W4A16_TILE_M32; lt++) {
        acc[lt] = 0.0f;
    }

    const unsigned int* qw_row = (row_o < out_features) ? (qweight + (size_t)row_o * (size_t)(in_features / 8)) : nullptr;
    const unsigned short* scale_row = (row_o < out_features) ? (scales + (size_t)row_o * (size_t)(in_features / group_size)) : nullptr;

    // Cooperative load: 256 threads load W4A16_TILE_M32*W4A16_TILE_K =
    // 8192 ushorts -- each thread loads 32 (4x uint4, was 2x for TILE_M=16).
    const int load_elem = tid * 32;
    const int t_idx = load_elem / W4A16_TILE_K;
    const int k_local = load_elem % W4A16_TILE_K;
    const bool valid_token = (token_base + t_idx < num_tokens);
    const unsigned short* gx_ptr = valid_token ? (x + (size_t)(token_base + t_idx) * (size_t)in_features + (size_t)k_local) : nullptr;

    auto load_tile = [&](int buf, int k_off) {
        if (valid_token && (k_off + k_local + 31 < in_features)) {
            const unsigned short* gx = gx_ptr + k_off;
#pragma unroll
            for (int v = 0; v < 4; v++) {
                *reinterpret_cast<uint4*>(&smem_x[buf][load_elem + v * 8]) = *reinterpret_cast<const uint4*>(gx + v * 8);
            }
        } else {
            uint4 zero = {0, 0, 0, 0};
#pragma unroll
            for (int v = 0; v < 4; v++) {
                *reinterpret_cast<uint4*>(&smem_x[buf][load_elem + v * 8]) = zero;
            }
        }
    };

    load_tile(0, 0);
    __syncthreads();

    int buf_curr = 0;
    for (int k_base = 0; k_base < in_features; k_base += W4A16_TILE_K) {
        const int next_k_base = k_base + W4A16_TILE_K;
        const int buf_next = buf_curr ^ 1;

        if (next_k_base < in_features) {
            load_tile(buf_next, next_k_base);
        }

        if (row_o < out_features) {
            unsigned int packed = 0x88888888u;
            if (k_base + lane_id * 8 < in_features) {
                packed = qw_row[k_base / 8 + lane_id];
            }

            float scale = 0.0f;
            int scale_idx = (k_base + lane_id * 8) / group_size;
            if (scale_idx < in_features / group_size) {
                scale = bf16_bits_to_f32(scale_row[scale_idx]);
            }

            float n0 = (float)((packed >> 0) & 0xF);
            float n1 = (float)((packed >> 4) & 0xF);
            float n2 = (float)((packed >> 8) & 0xF);
            float n3 = (float)((packed >> 12) & 0xF);
            float n4 = (float)((packed >> 16) & 0xF);
            float n5 = (float)((packed >> 20) & 0xF);
            float n6 = (float)((packed >> 24) & 0xF);
            float n7 = (float)((packed >> 28) & 0xF);

            for (int lt = 0; lt < tile_tokens; lt++) {
                const unsigned short* xr = &smem_x[buf_curr][lt * W4A16_TILE_K + lane_id * 8];
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

        __syncthreads();
        buf_curr = buf_next;
    }

    if (row_o < out_features) {
        for (int lt = 0; lt < tile_tokens; lt++) {
            float v = acc[lt];
            for (int offset = 16; offset > 0; offset >>= 1) {
                v += __shfl_down(v, offset);
            }
            if (lane_id == 0) {
                y[(size_t)(token_base + lt) * (size_t)out_features + row_o] = f32_to_bf16_bits(v);
            }
        }
    }
}

extern "C" void launch_w4a16_gemm_prefill_tile32_bf16(
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
    const int grid_x = (out_features + W4A16_TILE_N - 1) / W4A16_TILE_N;
    const int grid_y = (num_tokens + W4A16_TILE_M32 - 1) / W4A16_TILE_M32;
    dim3 grid(grid_x, grid_y);
    dim3 block(256);

    hipLaunchKernelGGL(
        w4a16_gemm_prefill_tile32_kernel,
        grid, block, 0, (hipStream_t)stream,
        (const unsigned short*)x, (const unsigned int*)qweight, (const unsigned short*)scales,
        (unsigned short*)y, in_features, out_features, num_tokens, group_size
    );
}

// §124 EXPERIMENT (real, isolated, NOT wired into the model yet -- see
// `docs/DECISIONS.md` §124): a DIFFERENT tradeoff from the rejected
// `TILE_M=32` experiment above -- doubles `TILE_N` (output rows per
// block, 8->16) instead of `TILE_M`. Real, found-not-assumed motivation:
// the shipped kernel's grid is `dim3(ceil(out_features/8),
// ceil(num_tokens/16))` -- for 27B `down_proj` (5120 rows), that's 640
// INDEPENDENT blocks along the X dimension, each of which redundantly
// re-loads the SAME `[16 tokens, 256 K]` activation tile from global
// memory into its OWN block-local LDS (LDS is never shared ACROSS
// blocks) -- a real, structural 640x activation re-read factor per
// K-step, only partially masked by L2 cache reuse (unmeasured directly
// -- rocprofv3's PMC counter collection crashed against this crate's
// real HIP-Graph-captured server process in this environment, a real
// tooling limitation, not silently assumed away).
//
// Doubling TILE_N halves the block count along X (320 instead of 640
// for 27B `down_proj`), directly halving that redundant-read factor,
// WITHOUT touching LDS size or TILE_M (both stay exactly as already
// proven best) -- each warp now owns 2 output rows instead of 1,
// doubling per-thread accumulator registers (`acc0[16]`+`acc1[16]` = 32
// floats/thread, the SAME real register count the REJECTED `TILE_M=32`
// experiment had) but WITHOUT doubling LDS (still 16KB, not 32KB) --
// a real, different point in the occupancy-vs-redundant-reads tradeoff
// space, not assumed to succeed OR fail just because `TILE_M=32` lost.
#define W4A16_TILE_N16 16

extern "C" __global__ void w4a16_gemm_prefill_tile_n16_kernel(
    const unsigned short* __restrict__ x,
    const unsigned int* __restrict__ qweight,
    const unsigned short* __restrict__ scales,
    unsigned short* __restrict__ y,
    int in_features,
    int out_features,
    int num_tokens,
    int group_size
) {
    __shared__ unsigned short smem_x[2][W4A16_TILE_M * W4A16_TILE_K];

    const int token_base = blockIdx.y * W4A16_TILE_M;
    const int tile_tokens = min(W4A16_TILE_M, num_tokens - token_base);
    if (tile_tokens <= 0) {
        return;
    }

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    // Real, only structural change vs the shipped kernel: each warp owns
    // TWO output rows (sub=0,1) instead of one.
    const int row_o0 = blockIdx.x * W4A16_TILE_N16 + warp_id * 2;
    const int row_o1 = row_o0 + 1;

    float acc0[W4A16_TILE_M];
    float acc1[W4A16_TILE_M];
#pragma unroll
    for (int lt = 0; lt < W4A16_TILE_M; lt++) {
        acc0[lt] = 0.0f;
        acc1[lt] = 0.0f;
    }

    const unsigned int* qw_row0 = (row_o0 < out_features) ? (qweight + (size_t)row_o0 * (size_t)(in_features / 8)) : nullptr;
    const unsigned int* qw_row1 = (row_o1 < out_features) ? (qweight + (size_t)row_o1 * (size_t)(in_features / 8)) : nullptr;
    const unsigned short* scale_row0 = (row_o0 < out_features) ? (scales + (size_t)row_o0 * (size_t)(in_features / group_size)) : nullptr;
    const unsigned short* scale_row1 = (row_o1 < out_features) ? (scales + (size_t)row_o1 * (size_t)(in_features / group_size)) : nullptr;

    const int load_elem = tid * 16;
    const int t_idx = load_elem / W4A16_TILE_K;
    const int k_local = load_elem % W4A16_TILE_K;
    const bool valid_token = (token_base + t_idx < num_tokens);
    const unsigned short* gx_ptr = valid_token ? (x + (size_t)(token_base + t_idx) * (size_t)in_features + (size_t)k_local) : nullptr;

    if (valid_token && (k_local + 15 < in_features)) {
        const unsigned short* gx = gx_ptr;
        uint4 v0 = *reinterpret_cast<const uint4*>(gx);
        uint4 v1 = *reinterpret_cast<const uint4*>(gx + 8);
        *reinterpret_cast<uint4*>(&smem_x[0][load_elem]) = v0;
        *reinterpret_cast<uint4*>(&smem_x[0][load_elem + 8]) = v1;
    } else {
        uint4 zero = {0, 0, 0, 0};
        *reinterpret_cast<uint4*>(&smem_x[0][load_elem]) = zero;
        *reinterpret_cast<uint4*>(&smem_x[0][load_elem + 8]) = zero;
    }
    __syncthreads();

    int buf_curr = 0;
    for (int k_base = 0; k_base < in_features; k_base += W4A16_TILE_K) {
        const int next_k_base = k_base + W4A16_TILE_K;
        const int buf_next = buf_curr ^ 1;

        if (next_k_base < in_features) {
            if (valid_token && (next_k_base + k_local + 15 < in_features)) {
                const unsigned short* gx = gx_ptr + next_k_base;
                uint4 v0 = *reinterpret_cast<const uint4*>(gx);
                uint4 v1 = *reinterpret_cast<const uint4*>(gx + 8);
                *reinterpret_cast<uint4*>(&smem_x[buf_next][load_elem]) = v0;
                *reinterpret_cast<uint4*>(&smem_x[buf_next][load_elem + 8]) = v1;
            } else {
                uint4 zero = {0, 0, 0, 0};
                *reinterpret_cast<uint4*>(&smem_x[buf_next][load_elem]) = zero;
                *reinterpret_cast<uint4*>(&smem_x[buf_next][load_elem + 8]) = zero;
            }
        }

        // Real, only compute-side change: unpack+accumulate for BOTH
        // row_o0 and row_o1 against the SAME LDS-resident activation
        // tile -- the real point of this experiment, sharing one real
        // activation load across twice as many output rows.
#pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            const unsigned int* qw_row = sub == 0 ? qw_row0 : qw_row1;
            const unsigned short* scale_row = sub == 0 ? scale_row0 : scale_row1;
            const int row_o = sub == 0 ? row_o0 : row_o1;
            float* acc = sub == 0 ? acc0 : acc1;
            if (row_o < out_features) {
                unsigned int packed = 0x88888888u;
                if (k_base + lane_id * 8 < in_features) {
                    packed = qw_row[k_base / 8 + lane_id];
                }
                float scale = 0.0f;
                int scale_idx = (k_base + lane_id * 8) / group_size;
                if (scale_idx < in_features / group_size) {
                    scale = bf16_bits_to_f32(scale_row[scale_idx]);
                }

                float n0 = (float)((packed >> 0) & 0xF);
                float n1 = (float)((packed >> 4) & 0xF);
                float n2 = (float)((packed >> 8) & 0xF);
                float n3 = (float)((packed >> 12) & 0xF);
                float n4 = (float)((packed >> 16) & 0xF);
                float n5 = (float)((packed >> 20) & 0xF);
                float n6 = (float)((packed >> 24) & 0xF);
                float n7 = (float)((packed >> 28) & 0xF);

                for (int lt = 0; lt < tile_tokens; lt++) {
                    const unsigned short* xr = &smem_x[buf_curr][lt * W4A16_TILE_K + lane_id * 8];
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

        __syncthreads();
        buf_curr = buf_next;
    }

#pragma unroll
    for (int sub = 0; sub < 2; sub++) {
        const int row_o = sub == 0 ? row_o0 : row_o1;
        float* acc = sub == 0 ? acc0 : acc1;
        if (row_o < out_features) {
            for (int lt = 0; lt < tile_tokens; lt++) {
                float v = acc[lt];
                for (int offset = 16; offset > 0; offset >>= 1) {
                    v += __shfl_down(v, offset);
                }
                if (lane_id == 0) {
                    y[(size_t)(token_base + lt) * (size_t)out_features + row_o] = f32_to_bf16_bits(v);
                }
            }
        }
    }
}

extern "C" void launch_w4a16_gemm_prefill_tile_n16_bf16(
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
    const int grid_x = (out_features + W4A16_TILE_N16 - 1) / W4A16_TILE_N16;
    const int grid_y = (num_tokens + W4A16_TILE_M - 1) / W4A16_TILE_M;
    dim3 grid(grid_x, grid_y);
    dim3 block(256);

    hipLaunchKernelGGL(
        w4a16_gemm_prefill_tile_n16_kernel,
        grid, block, 0, (hipStream_t)stream,
        (const unsigned short*)x, (const unsigned int*)qweight, (const unsigned short*)scales,
        (unsigned short*)y, in_features, out_features, num_tokens, group_size
    );
}

extern "C" void launch_w4a16_gemm_prefill_bf16(
    const void* x,
    const void* qweight,
    const void* scales,
    void* y,
    int out_features,
    int in_features,
    int group_size,
    int num_tokens,
    int /* threads */,
    void* stream
) {
    const int tile_m = W4A16_TILE_M;
    const int tile_n = W4A16_TILE_N;
    const int grid_x = (out_features + tile_n - 1) / tile_n;
    const int grid_y = (num_tokens + tile_m - 1) / tile_m;
    dim3 grid(grid_x, grid_y);
    dim3 block(256);

    hipLaunchKernelGGL(
        w4a16_gemm_prefill_kernel,
        grid, block, 0, (hipStream_t)stream,
        (const unsigned short*)x, (const unsigned int*)qweight, (const unsigned short*)scales,
        (unsigned short*)y, in_features, out_features, num_tokens, group_size
    );
}
