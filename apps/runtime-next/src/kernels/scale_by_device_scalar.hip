// Real HIP kernel: §116 -- in-place `buf *= *scalar_ptr` (bf16 buffer,
// real fp32 scalar read directly off the device, no host involvement).
//
// Real fix for `gdn_chunk_forward_prefill`'s remaining, documented,
// NOT-yet-fixed host sync (see that function's own §115 doc comment):
// Phase 2's real per-(chunk,head) state update, `state = state*chunk_decay
// + key_final^T@v_new`, used `gemm_atb`'s own `beta` parameter to apply
// `chunk_decay` -- but hipBLAS's `alpha`/`beta` are host `f32` values
// under this crate's `HIPBLAS_POINTER_MODE_HOST` convention (used
// throughout this whole file, everywhere else with real, fixed
// compile-time-constant scalars like `1.0`/`-1.0`/`0.0` -- `chunk_decay`
// was the one real, DYNAMIC, runtime-computed scalar that needed a real
// host read to satisfy that convention), forcing a real
// `copy_to_host`+blocking-sync per GDN layer, every real prefill call.
//
// This kernel splits that ONE fused hipBLAS `beta`-scale-and-accumulate
// into two real steps: (1) this kernel scales `state` by `chunk_decay`
// IN PLACE, reading the real scalar directly out of the ALREADY-ON-DEVICE
// `gdnc_chunk_decay` buffer (`gdn_chunk_decay.hip`'s own real output,
// Phase 1c -- no re-copy, just a pointer into an existing buffer); (2) the
// real caller then calls `gemm_atb` with a FIXED host literal `beta=1.0`
// (safe -- the same convention every other real call in this file
// already uses, no hipBLAS handle pointer-mode change needed anywhere).
//
// Real, disclosed precision note (checked, not assumed): hipBLAS's own
// fused `beta*Y + alpha*(A@B)` accumulates BOTH terms in ONE real fp32
// accumulator before a SINGLE round-to-bf16 write. This two-step version
// rounds `state*chunk_decay` to bf16 ONCE here, then rounds AGAIN after
// the GEMM's own accumulate -- a real, tiny, extra double-rounding step
// this change does NOT claim is bit-exact-by-construction (unlike
// §115's own on-device casts, which are). Verified instead via a real
// end-to-end 27B generation A/B (see `docs/DECISIONS.md` §116) -- the
// real, measured answer to whether that extra rounding ever changes a
// real generated token.

#include "bf16_utils.hip"

extern "C" __global__ void scale_bf16_by_device_scalar_kernel(
    unsigned short* __restrict__ buf,
    const float* __restrict__ scalar_ptr,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float scale = *scalar_ptr;
    float v = bf16_bits_to_f32(buf[i]);
    buf[i] = f32_to_bf16_bits(v * scale);
}

extern "C" void launch_scale_bf16_by_device_scalar(void* buf, const void* scalar_ptr, int n, void* stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(
        scale_bf16_by_device_scalar_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        (unsigned short*)buf, (const float*)scalar_ptr, n
    );
}

extern "C" __global__ void scale_bf16_by_device_scalars_batched_kernel(
    unsigned short* __restrict__ buf,
    const float* __restrict__ scalars,
    int n_per_head,
    int h
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = h * n_per_head;
    if (idx >= total) {
        return;
    }
    int head = idx / n_per_head;
    float scale = scalars[head];
    float v = bf16_bits_to_f32(buf[idx]);
    buf[idx] = f32_to_bf16_bits(v * scale);
}

extern "C" void launch_scale_bf16_by_device_scalars_batched(void* buf, const void* scalars, int n_per_head, int h, void* stream) {
    int threads = 256;
    int total = h * n_per_head;
    int blocks = (total + threads - 1) / threads;
    hipLaunchKernelGGL(
        scale_bf16_by_device_scalars_batched_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        (unsigned short*)buf, (const float*)scalars, n_per_head, h
    );
}
