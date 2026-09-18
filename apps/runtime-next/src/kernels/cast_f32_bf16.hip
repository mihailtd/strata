// Real HIP kernel: §115 -- plain elementwise fp32<->bf16 casts, real fix
// for `gdn_chunk_forward_prefill`'s own documented "identified follow-up"
// (see that function's doc comment in `model.rs`): its real recurrent-
// state bf16 shadow (needed because every GEMM primitive in this file's
// chunked-prefill pipeline is bf16-only I/O, while the real persistent
// `recurrent_state` stays fp32 for numerical accuracy across many chunks)
// was being cast via a REAL HOST ROUND TRIP -- `copy_to_host`, a CPU-side
// `Vec` conversion, `copy_from_host` -- TWICE per real GDN layer, every
// real prefill call, bracketed by an explicit blocking
// `hipDeviceSynchronize()` that existed only to make the first
// `copy_to_host` safe. Real, isolated wall-clock profiling (`model::
// tests::diagnose_real_quantized_prefill_wall_clock_vs_kernel_time`)
// found ~58% of real prefill wall-clock time was NOT GPU-kernel time --
// this real host round trip, happening once per GDN layer (24-64 times
// depending on model size) and forcing a full pipeline drain each time,
// is a real, direct, concrete contributor this kernel pair eliminates.
//
// Pure elementwise casts, same real round-to-nearest-even bf16 rounding
// formula every other kernel/CPU conversion in this codebase already
// uses (`model.rs`'s own `f32_to_bf16`/`bf16_to_f32`) -- not an
// approximation of the old host round trip, a BIT-EXACT on-device
// equivalent of it. Queued on the real caller's own stream like every
// other kernel in the chunked-prefill pipeline, so HIP's in-order-per-
// stream guarantee is what keeps this correctly sequenced against the
// GEMMs that read/write the buffers on either side of it -- no explicit
// synchronize needed (the real one this replaces existed ONLY because
// the old code needed the value on the HOST, which no longer applies).

#include <hip/hip_runtime.h>

extern "C" __global__ void f32_to_bf16_cast_kernel(
    const float* __restrict__ src,
    unsigned short* __restrict__ dst,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    unsigned int bits = __float_as_uint(src[i]);
    unsigned int lsb = (bits >> 16) & 1u;
    unsigned int rounded = bits + 0x7fffu + lsb;
    dst[i] = (unsigned short)(rounded >> 16);
}

extern "C" __global__ void bf16_to_f32_cast_kernel(
    const unsigned short* __restrict__ src,
    float* __restrict__ dst,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    dst[i] = __uint_as_float(((unsigned int)src[i]) << 16);
}

extern "C" void launch_f32_to_bf16_cast(const void* src, void* dst, int n, void* stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(
        f32_to_bf16_cast_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        (const float*)src, (unsigned short*)dst, n
    );
}

extern "C" void launch_bf16_to_f32_cast(const void* src, void* dst, int n, void* stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    hipLaunchKernelGGL(
        bf16_to_f32_cast_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        (const unsigned short*)src, (float*)dst, n
    );
}
