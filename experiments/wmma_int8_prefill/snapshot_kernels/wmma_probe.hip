// §125 EXPERIMENT, ISOLATED PROBE (real, NOT wired into the model or
// into ANY other kernel -- see `docs/DECISIONS.md` §125): a minimal,
// single-16x16x16-tile, plain dense bf16xbf16->f32 WMMA GEMM, built to
// answer ONE real question before committing to the much larger
// INT4-dequant-into-WMMA integration effort: can this crate correctly
// drive RDNA3's `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32` intrinsic
// at all?
//
// Real fragment-layout formulas below are PORTED VERBATIM (not
// re-derived) from `apps/runtime-llama/llama.cpp/ggml/src/ggml-cuda/
// mma.cuh`'s own real, production, RDNA3-verified
// `tile<16,16,T,DATA_LAYOUT_I_MAJOR_MIRRORED>::get_i/get_j` (the
// `#elif defined(AMD_WMMA_AVAILABLE)` branches) and its own
// `load_generic` (`t.x[l] = xs0[get_i(l)*stride + get_j(l)]`), NOT
// hand-derived from scratch -- this crate's own established discipline
// for anything touching real hardware intrinsics it doesn't already
// have a proven, tested implementation of (matches how the real
// `__builtin_amdgcn_sudot4` signature was confirmed against
// `BuiltinsAMDGPU.def` directly, not assumed, earlier this session).
//
// Real, disclosed, UNRESOLVED ambiguity this probe exists to test: for
// the A/B (half2-packed) operand tiles specifically, `get_j(l)` returns
// values in `[0,16)` -- too large to be a half2-slot index into a
// naturally-packed 16-real-column row (which would only need `[0,8)`).
// The real hypothesis this probe tests: RDNA3's WMMA expects each real
// bf16 value physically DUPLICATED into BOTH lanes of its own `half2`
// (i.e. stored as `half2(v, v)`, not two DIFFERENT adjacent real
// values packed together) -- making `stride=16` (not 8) the real
// per-row half2 count, consistent with `get_j`'s own `[0,16)` range.
// This is a real, testable hypothesis, not assumed correct -- see
// `docs/DECISIONS.md` §125 for the real, measured verdict (did the
// output match a real, independently-computed CPU reference or not).

#include "bf16_utils.hip"

typedef short __attribute__((ext_vector_type(16))) v16s_t;
typedef float __attribute__((ext_vector_type(8))) v8f_t;

extern "C" __global__ void wmma_probe_16x16x16_bf16_kernel(
    const unsigned short* __restrict__ a_dup,  // [16, 16] real bf16 values, EACH DUPLICATED into a real half2(v,v) pair -> physically [16 rows, 32 shorts/row] = [16,16] half2 elements, row-major, stride=16 half2/row
    const unsigned short* __restrict__ b_dup,  // same real layout as a_dup
    float* __restrict__ c_out,                 // [16, 16] real f32 output, row-major
    int stride_half2                            // real per-row half2 count for a_dup/b_dup (the hypothesis under test: 16)
) {
    const int tid = threadIdx.x; // 0..31, one real wave (wave32)

    // Real A/B fragment load, VERBATIM per mma.cuh's own
    // `load_generic`/`get_i`/`get_j` for `tile<16,16,half2,
    // DATA_LAYOUT_I_MAJOR_MIRRORED>` on RDNA3:
    //   get_i(l) = tid % 16
    //   get_j(l) = l*2 + (tid / 16)      (matrix A & B, RDNA3)
    const int row = tid % 16;
    const int half_of_wave = tid / 16;

    v16s_t a_frag, b_frag;
    unsigned short* a_frag_s = reinterpret_cast<unsigned short*>(&a_frag);
    unsigned short* b_frag_s = reinterpret_cast<unsigned short*>(&b_frag);
    const unsigned short* a_half2 = reinterpret_cast<const unsigned short*>(a_dup);
    const unsigned short* b_half2 = reinterpret_cast<const unsigned short*>(b_dup);

#pragma unroll
    for (int l = 0; l < 8; l++) {
        const int j = l * 2 + half_of_wave; // real half2-slot index, [0,16)
        const int idx = row * stride_half2 + j;
        // Each half2 slot holds 2 real shorts (the real, hypothesized
        // duplicated (v,v) pair) -- read BOTH into consecutive V16s
        // lanes, matching how `half2 x[8]` reinterprets as `V16s`.
        a_frag_s[2 * l + 0] = a_half2[2 * idx + 0];
        a_frag_s[2 * l + 1] = a_half2[2 * idx + 1];
        b_frag_s[2 * l + 0] = b_half2[2 * idx + 0];
        b_frag_s[2 * l + 1] = b_half2[2 * idx + 1];
    }

    v8f_t c_frag = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    c_frag = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a_frag, b_frag, c_frag);

    // Real C (float, J_MAJOR, RDNA3) fragment store, VERBATIM per
    // mma.cuh's own generic (non-half2) `tile<16,16,T>::get_i/get_j`
    // for T=float on RDNA3: get_i(l)=tid%16, get_j(l)=2*l+(tid/16) --
    // unambiguous (no packing), a REAL direct column index this time.
#pragma unroll
    for (int l = 0; l < 8; l++) {
        const int out_row = tid % 16;
        const int out_col = 2 * l + (tid / 16);
        c_out[out_row * 16 + out_col] = c_frag[l];
    }
}

extern "C" void launch_wmma_probe_16x16x16_bf16(
    const void* a_dup,
    const void* b_dup,
    void* c_out,
    int stride_half2,
    void* stream
) {
    hipLaunchKernelGGL(
        wmma_probe_16x16x16_bf16_kernel,
        dim3(1), dim3(32), 0, (hipStream_t)stream,
        (const unsigned short*)a_dup, (const unsigned short*)b_dup, (float*)c_out, stride_half2
    );
}
