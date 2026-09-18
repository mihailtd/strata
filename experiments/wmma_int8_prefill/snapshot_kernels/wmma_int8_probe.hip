// §126 EXPERIMENT, ISOLATED PROBE (real, NOT wired into the model or any
// other kernel -- see `docs/DECISIONS.md` §126): a minimal, single-tile
// INT8xINT8->INT32 WMMA GEMM, built to verify the RDNA3 fragment layout
// this crate would need to actually replicate llama.cpp's own real,
// production, confirmed-in-use-for-Q4_K quantized WMMA path (`mma.cuh`'s
// `mma(tile<16,16,int>&, tile<16,8,int>&, tile<16,8,int>&)` overload,
// calling `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` twice per K=32
// chunk) -- BEFORE attempting to integrate INT4 dequantization into it.
//
// Real, disclosed context this probe exists to resolve: an earlier probe
// this session (`wmma_probe.hip`, dense bf16xbf16->f32) hand-derived its
// fragment-layout formulas from `mma.cuh`'s generic `get_i`/`get_j` and
// got them WRONG (falsified against a real CPU reference, see §125). This
// probe instead uses formulas read DIRECTLY, verbatim, from the one
// `tile<>` specialization that actually matches what RDNA3 uses here --
// `struct tile<I_, J_, T, DATA_LAYOUT_I_MAJOR_MIRRORED>`
// (`mma.cuh` lines 529-569) -- confirmed (not assumed) to be the real
// layout for this exact path via `load_ldmatrix`'s own RDNA3 branch for
// `tile<16, 8, T, dl>` (`mma.cuh` lines 841-851), which carries a real
// `static_assert(sizeof(t.x) == 32, "bad ne")` -- i.e. 8 real int32
// elements per lane, NOT 4 (the earlier, WRONG reading came from a
// different, non-RDNA3 tile specialization's `ne = I*J/32` formula).
//
// Real, verbatim formulas used below, for `tile<16, 8, int,
// DATA_LAYOUT_I_MAJOR_MIRRORED>` (A and B operands):
//   ne = I*J/32*2 = 8
//   get_i(l) = threadIdx.x % 16                          (constant in l)
//   get_j(l) = l
// `load_ldmatrix` for this tile does a raw, non-permuted row read:
//   t.x[0..3] = xs0[get_i(0)*stride + 0..3]
//   t.x[4..7] = xs0[get_i(0)*stride + 4..7]
// i.e. every lane with the same `tid % 16` loads the IDENTICAL 8 int32s
// (real duplication across the two 16-lane wave halves, not a
// column-parity split -- a real, different scheme than the dense bf16
// path's `DATA_LAYOUT_I_MAJOR_MIRRORED` half2 tile, which does split by
// parity. Two different WMMA operand shapes, two different real layouts;
// conflating them was not attempted here).
//
// Real, verbatim `mma()` RDNA3 branch (`mma.cuh` lines 1305-1337):
//   int32x4_t* a_vec = (int32x4_t*)A.x;   // A.x has 8 real ints -> 2 vec4s
//   int32x4_t* b_vec = (int32x4_t*)B.x;
//   acc = wmma_i32_16x16x16_iu8_w32(true, a_vec[0], true, b_vec[0], acc, true);
//   acc = wmma_i32_16x16x16_iu8_w32(true, a_vec[1], true, b_vec[1], acc, true);
// i.e. two real K=16 WMMA passes, covering K=32 total per tile.
//
// Real, verbatim C/accumulator formula, `tile<16,16,int,
// DATA_LAYOUT_J_MAJOR>` (`mmq-vec-dot.cuh`'s own real `tile_C` for this
// exact call site) -- J_MAJOR swaps I_MAJOR's i/j (`mma.cuh` lines
// 508-525), and I_MAJOR's own T=int,I=16,J=16 RDNA3 "matrix C" branch
// (`mma.cuh` lines 193-211) gives:
//   I_MAJOR::get_i(l) = threadIdx.x % 16
//   I_MAJOR::get_j(l) = 2*l + threadIdx.x/16
// so J_MAJOR (swapped):
//   get_i(l) = 2*l + threadIdx.x/16      (real output ROW, i.e. A's row)
//   get_j(l) = threadIdx.x % 16          (real output COL, i.e. B's row)
//
// Real intrinsic signature confirmed directly (not assumed) against this
// machine's own `/opt/rocm/lib/llvm/include/clang/Basic/BuiltinsAMDGPU.def`:
//   TARGET_BUILTIN(__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32,
//                  "V8iIbV4iIbV4iV8iIb", "nc", "gfx11-insts,wavefrontsize32")
// i.e. V8i(bool a_signed, V4i a, bool b_signed, V4i b, V8i acc, bool clamp).

#include <hip/hip_runtime.h>

typedef int __attribute__((ext_vector_type(4))) v4i_t;
typedef int __attribute__((ext_vector_type(8))) v8i_t;

extern "C" __global__ void wmma_int8_probe_16x16x32_kernel(
    const int* __restrict__ a_packed,  // [16 rows, 8 int32] = [16, 32] real int8, row-major, stride=8 int32/row
    const int* __restrict__ b_packed,  // same real layout as a_packed
    int* __restrict__ c_out            // [16, 16] real int32 output, row-major (row = A's row i, col = B's row j)
) {
    const int tid = threadIdx.x; // 0..31, one real wave (wave32)
    const int row = tid % 16;    // real, verbatim: get_i(l) = tid % 16 for the A/B mirrored tile

    // Real A/B fragment load: get_j(l) = l, i.e. 8 consecutive int32s from
    // `row`'s own data -- identical for lanes `row` and `row+16` (real
    // duplication, not a permutation).
    v4i_t a_vec[2];
    v4i_t b_vec[2];
    int* a_frag_s = reinterpret_cast<int*>(a_vec);
    int* b_frag_s = reinterpret_cast<int*>(b_vec);
#pragma unroll
    for (int l = 0; l < 8; l++) {
        a_frag_s[l] = a_packed[row * 8 + l];
        b_frag_s[l] = b_packed[row * 8 + l];
    }

    // Real mma() RDNA3 branch, verbatim: two K=16 passes accumulating
    // into the same acc, covering K=32 total.
    v8i_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[0], true, b_vec[0], acc, true);
    acc = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[1], true, b_vec[1], acc, true);

    // Real C (int, J_MAJOR, RDNA3) fragment store, verbatim:
    //   get_i(l) = 2*l + tid/16   (output row, A's row index)
    //   get_j(l) = tid % 16       (output col, B's row index)
#pragma unroll
    for (int l = 0; l < 8; l++) {
        const int out_row = 2 * l + (tid / 16);
        const int out_col = tid % 16;
        c_out[out_row * 16 + out_col] = acc[l];
    }
}

extern "C" void launch_wmma_int8_probe_16x16x32(
    const void* a_packed,
    const void* b_packed,
    void* c_out,
    void* stream
) {
    hipLaunchKernelGGL(
        wmma_int8_probe_16x16x32_kernel,
        dim3(1), dim3(32), 0, (hipStream_t)stream,
        (const int*)a_packed, (const int*)b_packed, (int*)c_out
    );
}
