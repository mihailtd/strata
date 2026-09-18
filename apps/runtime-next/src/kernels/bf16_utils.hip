// Shared bf16 ↔ f32 device conversion utilities.
//
// Extracted from duplicated inline functions across every .hip kernel file
// in this directory. Single source of truth: if a rounding bug is found,
// fix it here once instead of in 15+ files.
//
// Usage: #include "bf16_utils.hip" at the top of any kernel file that
// needs bf16 ↔ f32 conversion.

#pragma once

#include <hip/hip_runtime.h>

/// Reinterpret a bf16 bit pattern (stored as unsigned short) as float32.
/// bf16 is the upper 16 bits of an IEEE 754 float32, so left-shifting
/// by 16 yields the exact float32 value (with the lower mantissa bits
/// zeroed, which is the definition of bf16 truncation).
__device__ __forceinline__ float bf16_bits_to_f32(unsigned short bits) {
    return __uint_as_float(((unsigned int)bits) << 16);
}

/// Round a float32 value to bf16 using round-to-nearest-even (RNE).
/// This matches PyTorch's own `torch.bfloat16` cast and this codebase's
/// CPU-side `f32_to_bf16` in `kernels.rs` tests (verified bit-exact in
/// `real_f32_bf16_casts_are_bit_exact_both_directions`).
__device__ __forceinline__ unsigned short f32_to_bf16_bits(float v) {
    unsigned int bits = __float_as_uint(v);
    unsigned int lsb = (bits >> 16) & 1u;
    unsigned int rounded = bits + 0x7fffu + lsb;
    return (unsigned short)(rounded >> 16);
}
