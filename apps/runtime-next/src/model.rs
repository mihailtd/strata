//! Stage: assembling every real, individually-benchmarked feature (§80-88)
//! into one real, correct forward pass -- the "Eventually" milestone from
//! `TODO.md`. Real weights for all 32 layers, real per-layer forward
//! functions for both layer types, a real KV cache / recurrent-state
//! manager, and a single-token decode step -- wired together for the
//! first time, not another isolated kernel benchmark.
//!
//! Real architecture facts, confirmed against this machine's actual
//! `config.json` (`text_config`), not assumed:
//! - 32 layers; `full_attention` at indices `[3,7,11,15,19,23,27,31]`
//!   (every 4th), `linear_attention` (GatedDeltaNet) everywhere else.
//! - hidden_size=2560, intermediate_size=9216, vocab_size=248320.
//! - Full attention: num_attention_heads=16, num_key_value_heads=4,
//!   head_dim=256, rotary_dim=64 (partial_rotary_factor=0.25),
//!   rope_theta=1e7, attention_bias=False.
//! - GatedDeltaNet: num_k_heads=16, num_v_heads=32, head_dim=128 (both K
//!   and V), conv_kernel_size=4, conv_dim=8192.
//! - rms_norm_eps=1e-6, hidden_act=silu, tie_word_embeddings=true (no
//!   separate `lm_head.weight` tensor in the checkpoint -- confirmed via
//!   the real index; `embed_tokens.weight` is reused directly).
//!
//! Scope: single sequence, batch=1, decode-only (prefill is just the same
//! single-token step called once per prompt token, sequentially -- no
//! separate parallel/batched prefill path, matching every kernel this
//! port has built so far).
//!
//! PERFORMANCE PASS (§91): §89's first working version allocated ~20 fresh
//! `DeviceBuffer`s (real `hipMalloc`/`hipFree` calls) per layer per token,
//! and every kernel/GEMM call went through the safe wrappers'
//! `hip::device_synchronize()`. §90's real A/B against llama.cpp/Ollama
//! (both ~2.4x faster, using byte-identical bf16 weights) made clear that
//! floor needed fixing, not just estimating. This module now:
//! - Pre-allocates every intermediate tensor ONCE in `Scratch` (built
//!   alongside `DecodeState`), reused every token -- zero `hipMalloc`/
//!   `hipFree` in the hot per-token path.
//! - Calls the RAW kernel/GEMM launchers directly (`mod raw` below, thin
//!   wrappers around `kernels::ffi`/`blas::ffi` -- the exact same audited
//!   unsafe surface, just called without the safe wrappers' per-call
//!   `device_synchronize()`), relying on HIP's own in-order stream
//!   execution guarantee: kernels queued on the same (default) stream run
//!   in issue order without needing the host to wait between them.
//! - Uses `DeviceBuffer::copy_from_device_range_async` (not the blocking
//!   `copy_from_device_range`) for the small device-to-device splits/views
//!   this forward pass needs (GDN's `in_proj_qkv` split, the KV-cache
//!   read view) -- a plain `hipMemcpy` would have silently reintroduced
//!   the exact per-call host stall removing `device_synchronize()`
//!   elsewhere was meant to eliminate.
//! - Synchronizes with the GPU exactly ONCE per token: `forward_one_token`
//!   calls `hip::check_last_error()` + `hip::device_synchronize()` right
//!   after the 32-layer loop, before the final GEMM's result would
//!   otherwise be read back for sampling anyway.

use crate::blas::{BlasHandle, ffi as blas_ffi};
use crate::hip::{self, DeviceBuffer, HipError};
use crate::model_loader::{load_bf16_weight, load_concat_bf16_weights, load_f32_param};
use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;

// §105: real dims confirmed by reading each size's own `config.json`
// directly (not inferred/interpolated). Only 6 values actually vary
// across the real Qwen3.5 family (HIDDEN_SIZE, NUM_LAYERS,
// INTERMEDIATE_SIZE, ATTN_NUM_HEADS, ATTN_NUM_KV_HEADS, GDN_NUM_V_HEADS)
// -- every other dim below (head_dim=256, GDN's own head_dim=128 and
// K-head count=16, full_attention_interval=4, rope_theta=1e7,
// partial_rotary_factor=0.25, vocab_size=248320) is IDENTICAL across
// 0.8B/2B/4B/9B, confirmed directly, not assumed to generalize.
const _: () = assert!(
    cfg!(feature = "qwen35_0_8b") as u8
        + cfg!(feature = "qwen35_2b") as u8
        + cfg!(feature = "qwen35_4b") as u8
        + cfg!(feature = "qwen35_9b") as u8
        + cfg!(feature = "qwen35_27b") as u8
        == 1,
    "exactly one qwen35_* size feature must be active"
);

#[cfg(feature = "qwen35_0_8b")]
pub const HIDDEN_SIZE: usize = 1024;
#[cfg(feature = "qwen35_0_8b")]
pub const NUM_LAYERS: usize = 24;
#[cfg(feature = "qwen35_0_8b")]
pub const INTERMEDIATE_SIZE: usize = 3584;
#[cfg(feature = "qwen35_0_8b")]
pub const ATTN_NUM_HEADS: usize = 8;
#[cfg(feature = "qwen35_0_8b")]
pub const ATTN_NUM_KV_HEADS: usize = 2;
#[cfg(feature = "qwen35_0_8b")]
pub const GDN_NUM_V_HEADS: usize = 16;

#[cfg(feature = "qwen35_2b")]
pub const HIDDEN_SIZE: usize = 2048;
#[cfg(feature = "qwen35_2b")]
pub const NUM_LAYERS: usize = 24;
#[cfg(feature = "qwen35_2b")]
pub const INTERMEDIATE_SIZE: usize = 6144;
#[cfg(feature = "qwen35_2b")]
pub const ATTN_NUM_HEADS: usize = 8;
#[cfg(feature = "qwen35_2b")]
pub const ATTN_NUM_KV_HEADS: usize = 2;
#[cfg(feature = "qwen35_2b")]
pub const GDN_NUM_V_HEADS: usize = 16;

#[cfg(feature = "qwen35_4b")]
pub const HIDDEN_SIZE: usize = 2560;
#[cfg(feature = "qwen35_4b")]
pub const NUM_LAYERS: usize = 32;
#[cfg(feature = "qwen35_4b")]
pub const INTERMEDIATE_SIZE: usize = 9216;
#[cfg(feature = "qwen35_4b")]
pub const ATTN_NUM_HEADS: usize = 16;
#[cfg(feature = "qwen35_4b")]
pub const ATTN_NUM_KV_HEADS: usize = 4;
#[cfg(feature = "qwen35_4b")]
pub const GDN_NUM_V_HEADS: usize = 32;

#[cfg(feature = "qwen35_9b")]
pub const HIDDEN_SIZE: usize = 4096;
#[cfg(feature = "qwen35_9b")]
pub const NUM_LAYERS: usize = 32;
#[cfg(feature = "qwen35_9b")]
pub const INTERMEDIATE_SIZE: usize = 12288;
#[cfg(feature = "qwen35_9b")]
pub const ATTN_NUM_HEADS: usize = 16;
#[cfg(feature = "qwen35_9b")]
pub const ATTN_NUM_KV_HEADS: usize = 4;
#[cfg(feature = "qwen35_9b")]
pub const GDN_NUM_V_HEADS: usize = 32;

// §106: Qwen3.8-27B -- real dims confirmed by reading its own config.json
// directly (identical structurally to Qwen3.5-27B, both real, same
// architecture class). Same "only 6 values vary" pattern as the rest of
// the family.
#[cfg(feature = "qwen35_27b")]
pub const HIDDEN_SIZE: usize = 5120;
#[cfg(feature = "qwen35_27b")]
pub const NUM_LAYERS: usize = 64;
#[cfg(feature = "qwen35_27b")]
pub const INTERMEDIATE_SIZE: usize = 17408;
#[cfg(feature = "qwen35_27b")]
pub const ATTN_NUM_HEADS: usize = 24;
#[cfg(feature = "qwen35_27b")]
pub const ATTN_NUM_KV_HEADS: usize = 4;
#[cfg(feature = "qwen35_27b")]
pub const GDN_NUM_V_HEADS: usize = 48;

pub const VOCAB_SIZE: usize = 248320;
pub const RMS_EPS: f32 = 1e-6;

pub const ATTN_HEAD_DIM: usize = 256;
pub const ATTN_ROTARY_DIM: usize = 64;
pub const ATTN_ROPE_THETA: f32 = 10_000_000.0;

pub const GDN_NUM_K_HEADS: usize = 16;
pub const GDN_HEAD_DIM: usize = 128;
pub const GDN_KEY_DIM: usize = GDN_NUM_K_HEADS * GDN_HEAD_DIM; // 2048
pub const GDN_VALUE_DIM: usize = GDN_NUM_V_HEADS * GDN_HEAD_DIM; // 4096
pub const GDN_CONV_DIM: usize = GDN_KEY_DIM * 2 + GDN_VALUE_DIM; // 8192
pub const GDN_CONV_KERNEL_SIZE: usize = 4;

/// §100 (chunked GDN parallel prefill): the real UT-transform chunk size
/// used by `gdn_chunk_forward_prefill` -- matches the real profiling case
/// (`seq_len=401, chunk_size=64`) that motivated this work and the
/// scoping doc's own hardware-mapping analysis (LDS budget for
/// `gdn_chunk_utsolve_bf16`'s `width=128` threads). `MAX_PREFILL_CHUNK`
/// (256) is already an exact multiple, so no scratch buffer here needs
/// extra padding room beyond `MAX_PREFILL_CHUNK` itself.
pub const GDN_CHUNK_SIZE: usize = 64;
const _: () = assert!(MAX_PREFILL_CHUNK % GDN_CHUNK_SIZE == 0, "MAX_PREFILL_CHUNK must be an exact multiple of GDN_CHUNK_SIZE (chunked-GDN scratch buffers are sized to MAX_PREFILL_CHUNK with no extra padding slack)");
pub const MAX_GDN_CHUNKS: usize = MAX_PREFILL_CHUNK / GDN_CHUNK_SIZE;

/// Real, confirmed-from-`config.json` rule: every 4th layer is full
/// attention (`full_attention_interval=4`), the rest are GDN
/// (`linear_attention`) -- confirmed IDENTICAL across every real Qwen3.5
/// size checked directly (0.8B/2B/4B/9B all have `full_attention_interval:
/// 4` and the same `layer_types` interleave pattern, only `NUM_LAYERS`
/// itself differs: 24 for 0.8B/2B, 32 for 4B/9B). Expressed as a modulo
/// rule rather than a hardcoded per-size index list so it stays correct
/// for any `NUM_LAYERS`, not just 32.
pub(crate) fn is_full_attention_layer(i: usize) -> bool {
    (i + 1) % 4 == 0
}

/// §100 (chunked GDN parallel prefill): the ONE real production host-side
/// bf16<->f32 cast this crate needs -- every other bf16 buffer stays on
/// the GPU end-to-end (even final logits are argmax-reduced on-device via
/// `kernels::argmax_bf16`, never read back as bf16 floats). Used only for
/// `gdn_chunk_forward_prefill`'s real fp32-state<->bf16-shadow roundtrip
/// (see that function's own doc comment for why). Same round-to-nearest-
/// even bit manipulation already proven correct by every decisive test in
/// this crate that computes an independent bf16 reference by hand
/// (`blas.rs`/`kernels.rs`'s own test-only copies of this exact code).
pub(crate) fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7fff + lsb);
    (rounded >> 16) as u16
}

pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Thin, non-syncing wrappers around the exact same audited FFI
/// declarations `kernels.rs`/`blas.rs` already expose -- no new unsafe
/// surface, just called without the trailing `check_last_error()`/
/// `device_synchronize()` every safe wrapper pays per call. Every launch
/// here goes on the default stream (`0`), the same stream every `.hip`
/// launcher and `hipblasGemmEx` call already uses -- so these stay
/// correctly ordered relative to each other without any explicit stream
/// management, purely from HIP's own in-order-per-stream guarantee.
pub(crate) mod raw {
    use super::c_void;
    use crate::blas::ffi as blas_ffi;
    use crate::kernels::ffi as kernels_ffi;

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rmsnorm(x: *const c_void, weight: *const c_void, out: *mut c_void, num_rows: i32, hidden_size: i32, eps: f32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_rmsnorm_bf16(x, weight, out, num_rows, hidden_size, eps, 1024, stream) };
    }

    /// §93: replaced hipBLAS (`hipblasGemmEx`) with a hand-written,
    /// vectorized GEMV kernel (`src/kernels/gemv.hip`) -- this engine only
    /// ever calls with `rows=1` (a single decode-step token, never a real
    /// batch), i.e. every one of these calls is really a matrix-VECTOR
    /// product, not a matrix-matrix product, and general BLAS libraries
    /// (both plain hipBLAS and hipBLASLt, see `docs/DECISIONS.md` §92)
    /// measurably under-utilized this GPU's memory bandwidth for that
    /// degenerate shape. Real, reproduced measurement for this model's
    /// `down_proj` shape: 37.7 us/call (gemv) vs 76.4 us/call
    /// (hipblasGemmEx) -- see `kernels::tests::
    /// bench_real_gemv_vs_gemm_down_proj_shape`. `handle`/`rows` are now
    /// unused (kept in the signature to avoid touching every call site
    /// for a parameter this function no longer needs) -- hipBLAS itself
    /// (`blas.rs`, `blaslt.rs`) remains linked and independently
    /// correctness-tested, just no longer called from this hot path.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm(
        handle: blas_ffi::HipblasHandle,
        x: *const c_void,
        w: *const c_void,
        y: *mut c_void,
        rows: i32,
        in_features: i32,
        out_features: i32,
        stream: *mut c_void,
    ) {
        if rows == 1 {
            unsafe {
                kernels_ffi::launch_gemv_bf16(x, w, y, out_features, in_features, 256, stream);
            }
        } else {
            // §96 (batched prefill): `rows > 1` is a genuine matrix-matrix
            // product (a chunk of real new prompt tokens processed
            // together, never true for the rows=1 decode-step case above),
            // where real hipBLAS amortizes reading this weight matrix's
            // bytes ONCE across all `rows` outputs instead of paying that
            // same memory-bandwidth cost once per row sequentially --
            // exactly the shape §93's comment above documents this GEMV
            // kernel as deliberately NOT covering. Same row-major-via-
            // column-major derivation as `blas::gemm_bf16_linear` (see that
            // function's doc comment), called directly via the raw FFI
            // (skipping its per-call `device_synchronize()`) so this can be
            // queued back-to-back with the surrounding per-token kernels on
            // one stream and synced once per prefill chunk -- same
            // reasoning as every other `raw::` call in this module.
            let alpha: f32 = 1.0;
            let beta: f32 = 0.0;
            unsafe {
                blas_ffi::hipblasGemmEx(
                    handle,
                    blas_ffi::HIPBLAS_OP_T,
                    blas_ffi::HIPBLAS_OP_N,
                    out_features,
                    rows,
                    in_features,
                    &alpha as *const f32 as *const c_void,
                    w,
                    blas_ffi::HIP_R_16BF,
                    in_features,
                    x,
                    blas_ffi::HIP_R_16BF,
                    in_features,
                    &beta as *const f32 as *const c_void,
                    y,
                    blas_ffi::HIP_R_16BF,
                    out_features,
                    blas_ffi::HIPBLAS_COMPUTE_32F,
                    blas_ffi::HIPBLAS_GEMM_DEFAULT,
                );
            }
        }
    }

    /// §106: W4A16 fused dequant+GEMV -- the quantized-weight analogue of
    /// `gemm`'s `rows==1` branch above (no `rows>1` case: this engine's
    /// quantized layers use the per-token decode path for prefill too,
    /// not the batched-GEMM one -- see `docs/DECISIONS.md` §106 for the
    /// real, disclosed scope decision behind that).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn linear_quantized(
        x: *const c_void,
        qweight: *const c_void,
        scales: *const c_void,
        y: *mut c_void,
        in_features: i32,
        out_features: i32,
        group_size: i32,
        stream: *mut c_void,
    ) {
        let threads = crate::kernels::w4a16_decode_threads(in_features as usize);
        unsafe {
            kernels_ffi::launch_w4a16_gemv_bf16(x, qweight, scales, y, out_features, in_features, group_size, threads, stream);
        }
    }

    /// Quantized-path LoRA, decode-only (see `quantized_lora.rs`'s header
    /// doc for the full rationale): computes `mid = lora_a @ x`, the
    /// small rank-sized GEMV every real LoRA slot's delta accumulation
    /// needs first. Literally the same kernel as `gemm`'s `rows==1`
    /// branch (`gemv_bf16_kernel`), called directly here (skipping the
    /// unused hipBLAS-handle branch) since `lora_a` is always a real
    /// `[rank, in_features]` bf16 buffer, never quantized.
    pub unsafe fn lora_mid(x: *const c_void, lora_a: *const c_void, mid: *mut c_void, in_features: i32, rank: i32, stream: *mut c_void) {
        unsafe {
            kernels_ffi::launch_gemv_bf16(x, lora_a, mid, rank, in_features, 256, stream);
        }
    }

    /// Quantized-path LoRA: accumulates `y[row] += sum_r lora_b[row,r] *
    /// mid[r]` in place, for one real LoRA slot's row range within a
    /// (possibly fused) base output buffer -- `y` must point at the
    /// START of that slot's own row range (`row_offset` already applied
    /// by the caller via `as_device_ptr_at_mut`, same idiom `lora.rs`'s
    /// `fold_into` uses), and must already hold the base W4A16 kernel's
    /// real output for those rows.
    pub unsafe fn lora_delta_accumulate(mid: *const c_void, lora_b: *const c_void, y: *mut c_void, out_features: i32, rank: i32, stream: *mut c_void) {
        unsafe {
            kernels_ffi::launch_lora_delta_accumulate_bf16(mid, lora_b, y, out_features, rank, stream);
        }
    }

    /// §108/§124/§126: batched W4A16 prefill -- the `rows>1` analogue of
    /// `linear_quantized` above, real fix for the disclosed §106 TTFT
    /// gap (quantized prefill no longer loops `linear_quantized` once
    /// per token). §126: real, kept, now dispatches to RDNA3 hardware
    /// WMMA INT8 tensor cores (`w4a16_gemm_prefill_wmma_int8.hip`) --
    /// real, measured 1.9x-3.3x faster than the scalar `TILE_N=16`
    /// kernel it replaces, at every real prompt-length-representative
    /// token count tested (16-128 tokens), on real 27B weights. Real,
    /// disclosed new rounding source (on-the-fly INT8 activation
    /// quantization): 0.37% relative L1 vs. the scalar bf16 kernel's own
    /// output, measured on real weights -- smaller than the already-
    /// shipped W4A16 scheme's own original quantization cost. `group_size`
    /// is asserted to be exactly 128 (the WMMA kernel's fixed K-step,
    /// real and exact for every real in_features in this model family --
    /// `ATTN_HEAD_DIM=256` and `GDN_HEAD_DIM=128` are both multiples of
    /// 128, confirmed directly, not assumed). See `docs/DECISIONS.md`
    /// §126 and `src/kernels/w4a16_gemm_prefill_wmma_int8.hip`'s own doc
    /// comment for the real design, the real bug this uncovered (a
    /// per-lane weight-loading row was mistakenly used to scale a
    /// DIFFERENT lane-local output row -- the WMMA hardware combines all
    /// 32 lanes' operand data into one real 16x16 cross product, so a
    /// lane's own accumulator slot does not correspond to its own
    /// operand row), and the real fix.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn linear_quantized_prefill(
        x: *const c_void,
        qweight: *const c_void,
        scales: *const c_void,
        y: *mut c_void,
        in_features: i32,
        out_features: i32,
        group_size: i32,
        num_tokens: i32,
        stream: *mut c_void,
    ) {
        debug_assert_eq!(group_size, 128, "the WMMA INT8 prefill kernel's K-step is fixed at 128");
        debug_assert_eq!(in_features % 128, 0, "the WMMA INT8 prefill kernel requires in_features to be a multiple of 128");
        unsafe {
            kernels_ffi::launch_w4a16_gemm_prefill_wmma_int8_bf16(x, qweight, scales, y, out_features, in_features, group_size, num_tokens, stream);
        }
    }

    /// See `src/kernels/extract_range.hip` (§96: batched prefill).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn extract_range(
        src: *const c_void,
        dst: *mut c_void,
        rows: i32,
        src_stride: i32,
        src_offset: i32,
        len: i32,
        stream: *mut c_void,
    ) {
        unsafe { kernels_ffi::launch_extract_range_bf16(src, dst, rows, src_stride, src_offset, len, stream) };
    }

    /// §99 (batched-prefill causal attention): `S = scale * (Q @ K^T)`,
    /// the raw (unsynced) hot-path counterpart of `blas::gemm_qkt_bf16` --
    /// same math and stride semantics (see that function's own doc comment
    /// for the full derivation), called directly via `hipblasGemmEx`,
    /// skipping the safe wrapper's per-call `device_synchronize()`, same
    /// reasoning as `raw::gemm`'s `rows > 1` branch. Operates on raw
    /// pointers already offset by the caller (`q`/`k`/`s` are the exact
    /// addresses to read/write, not buffer-relative offsets) since this
    /// module works in raw pointers throughout, unlike `blas.rs`'s
    /// `&DeviceBuffer` API. `scale` (real attention's `1/sqrt(head_dim)`)
    /// is folded into `hipblasGemmEx`'s own `alpha` scalar -- no separate
    /// elementwise scaling pass needed.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_qkt(handle: blas_ffi::HipblasHandle, q: *const c_void, q_ld: i32, k: *const c_void, s: *mut c_void, s_ld: i32, t: i32, head_dim: i32, kv_len: i32, scale: f32) {
        let alpha: f32 = scale;
        let beta: f32 = 0.0;
        unsafe {
            blas_ffi::hipblasGemmEx(
                handle,
                blas_ffi::HIPBLAS_OP_T,
                blas_ffi::HIPBLAS_OP_N,
                kv_len,
                t,
                head_dim,
                &alpha as *const f32 as *const c_void,
                k,
                blas_ffi::HIP_R_16BF,
                head_dim,
                q,
                blas_ffi::HIP_R_16BF,
                q_ld,
                &beta as *const f32 as *const c_void,
                s,
                blas_ffi::HIP_R_16BF,
                s_ld,
                blas_ffi::HIPBLAS_COMPUTE_32F,
                blas_ffi::HIPBLAS_GEMM_DEFAULT,
            );
        }
    }

    /// §99: `O = beta*O + alpha*(P @ V)`, the raw (unsynced) hot-path
    /// counterpart of `blas::gemm_pv_bf16` -- see that function's own doc
    /// comment for the full column-major derivation and `alpha`/`beta`'s
    /// own real scale/accumulate semantics (§100: chunked GDN uses
    /// `beta=1.0` to accumulate `intra_chunk_attn@v_new` onto an
    /// already-written `inter_chunk_attn`, and `alpha=-1.0` to subtract
    /// the `k_cumdecay@last_recurrent_state` correction from `new_values`
    /// in place).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_pv(handle: blas_ffi::HipblasHandle, p: *const c_void, p_ld: i32, v: *const c_void, o: *mut c_void, o_ld: i32, t: i32, kv_len: i32, head_dim: i32, alpha: f32, beta: f32) {
        unsafe {
            blas_ffi::hipblasGemmEx(
                handle,
                blas_ffi::HIPBLAS_OP_N,
                blas_ffi::HIPBLAS_OP_N,
                head_dim,
                t,
                kv_len,
                &alpha as *const f32 as *const c_void,
                v,
                blas_ffi::HIP_R_16BF,
                head_dim,
                p,
                blas_ffi::HIP_R_16BF,
                p_ld,
                &beta as *const f32 as *const c_void,
                o,
                blas_ffi::HIP_R_16BF,
                o_ld,
                blas_ffi::HIPBLAS_COMPUTE_32F,
                blas_ffi::HIPBLAS_GEMM_DEFAULT,
            );
        }
    }

    /// See `src/kernels/causal_softmax.hip` (§99: batched-prefill causal
    /// attention).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn causal_softmax(scores: *mut c_void, kv_len: i32, start_position_ptr: *const i32, num_rows: i32, stream: *mut c_void) {
        let threads = 256;
        unsafe { kernels_ffi::launch_causal_softmax_bf16(scores, kv_len, start_position_ptr, num_rows, threads, stream) };
    }

    /// §100 (chunked GDN prefill): `Y = beta*Y + A^T @ B`, the raw
    /// (unsynced) hot-path counterpart of `blas::gemm_atb_bf16` -- see
    /// that function's own doc comment for the full derivation. Used for
    /// the delta-rule's real state update (`state = state*chunk_decay +
    /// key^T@v_new`, `beta = chunk_decay`).
    #[allow(clippy::too_many_arguments, dead_code)]
    pub unsafe fn gemm_atb(handle: blas_ffi::HipblasHandle, a: *const c_void, a_ld: i32, b: *const c_void, b_ld: i32, y: *mut c_void, y_ld: i32, m: i32, k: i32, n: i32, beta: f32) {
        let alpha: f32 = 1.0;
        unsafe {
            blas_ffi::hipblasGemmEx(
                handle,
                blas_ffi::HIPBLAS_OP_N,
                blas_ffi::HIPBLAS_OP_T,
                n,
                k,
                m,
                &alpha as *const f32 as *const c_void,
                b,
                blas_ffi::HIP_R_16BF,
                b_ld,
                a,
                blas_ffi::HIP_R_16BF,
                a_ld,
                &beta as *const f32 as *const c_void,
                y,
                blas_ffi::HIP_R_16BF,
                y_ld,
                blas_ffi::HIPBLAS_COMPUTE_32F,
                blas_ffi::HIPBLAS_GEMM_DEFAULT,
            );
        }
    }

    /// §122: strided batched `S = scale*(Q@K^T)` for GDN Phase 1b --
    /// collapses `h` (32) per-head `gemm_qkt` calls into 1 launch,
    /// batched over the HEAD dimension for one FIXED chunk (see Phase
    /// 1b's own call site doc comment for why chunk, not head, stays the
    /// host-side loop variable: `gdnc_k_beta_token` is token-major, not
    /// head-chunk-major like `gdnc_key_bcast_hm`/`gdnc_query_intra_hm`,
    /// so only a fixed-chunk/batched-head grouping gives every one of
    /// Q/K/output a real, uniform per-batch-element stride).  Same
    /// `HIPBLAS_OP_T`/`HIPBLAS_OP_N` math as `gemm_qkt`, just batched.
    /// Hot path: no sync, but debug-asserts the return status.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_strided_batched_qkt(
        handle: blas_ffi::HipblasHandle,
        q: *const c_void,
        q_ld: i32,
        q_stride: i64,
        k: *const c_void,
        k_stride: i64,
        s: *mut c_void,
        s_ld: i32,
        s_stride: i64,
        t: i32,
        head_dim: i32,
        kv_len: i32,
        batch_count: i32,
        scale: f32,
    ) {
        let alpha: f32 = scale;
        let beta: f32 = 0.0;
        let status = unsafe {
            blas_ffi::hipblasGemmStridedBatchedEx(
                handle,
                blas_ffi::HIPBLAS_OP_T,
                blas_ffi::HIPBLAS_OP_N,
                kv_len,
                t,
                head_dim,
                &alpha as *const f32 as *const c_void,
                k,
                blas_ffi::HIP_R_16BF,
                head_dim,
                k_stride,
                q,
                blas_ffi::HIP_R_16BF,
                q_ld,
                q_stride,
                &beta as *const f32 as *const c_void,
                s,
                blas_ffi::HIP_R_16BF,
                s_ld,
                s_stride,
                batch_count,
                blas_ffi::HIPBLAS_COMPUTE_32F,
                blas_ffi::HIPBLAS_GEMM_DEFAULT,
            )
        };
        debug_assert_eq!(status, blas_ffi::HIPBLAS_STATUS_SUCCESS, "gemm_strided_batched_qkt: hipBLAS returned error {status}");
    }

    /// §119: strided batched P@V for GDN Phase 2 -- collapses 32 per-head
    /// `gemm_pv` calls into 1 launch. Hot path: no sync, but debug-asserts
    /// the return status to catch silent hipBLAS failures.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_strided_batched_pv(
        handle: blas_ffi::HipblasHandle,
        p: *const c_void,
        p_ld: i32,
        p_stride: i64,
        v: *const c_void,
        v_stride: i64,
        o: *mut c_void,
        o_ld: i32,
        o_stride: i64,
        t: i32,
        kv_len: i32,
        head_dim: i32,
        batch_count: i32,
        alpha: f32,
        beta: f32,
    ) {
        let status = unsafe {
            blas_ffi::hipblasGemmStridedBatchedEx(
                handle,
                blas_ffi::HIPBLAS_OP_N,
                blas_ffi::HIPBLAS_OP_N,
                head_dim,
                t,
                kv_len,
                &alpha as *const f32 as *const c_void,
                v,
                blas_ffi::HIP_R_16BF,
                head_dim,
                v_stride,
                p,
                blas_ffi::HIP_R_16BF,
                p_ld,
                p_stride,
                &beta as *const f32 as *const c_void,
                o,
                blas_ffi::HIP_R_16BF,
                o_ld,
                o_stride,
                batch_count,
                blas_ffi::HIPBLAS_COMPUTE_32F,
                blas_ffi::HIPBLAS_GEMM_DEFAULT,
            )
        };
        debug_assert_eq!(status, blas_ffi::HIPBLAS_STATUS_SUCCESS, "gemm_strided_batched_pv: hipBLAS returned error {status}");
    }

    /// §119: strided batched A^T@B for GDN Phase 2 -- collapses 32 per-head
    /// `gemm_atb` calls into 1 launch. Hot path: no sync, but debug-asserts
    /// the return status.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_strided_batched_atb(
        handle: blas_ffi::HipblasHandle,
        a: *const c_void,
        a_ld: i32,
        a_stride: i64,
        b: *const c_void,
        b_ld: i32,
        b_stride: i64,
        y: *mut c_void,
        y_ld: i32,
        y_stride: i64,
        m: i32,
        k: i32,
        n: i32,
        batch_count: i32,
        beta: f32,
    ) {
        let alpha: f32 = 1.0;
        let status = unsafe {
            blas_ffi::hipblasGemmStridedBatchedEx(
                handle,
                blas_ffi::HIPBLAS_OP_N,
                blas_ffi::HIPBLAS_OP_T,
                n,
                k,
                m,
                &alpha as *const f32 as *const c_void,
                b,
                blas_ffi::HIP_R_16BF,
                b_ld,
                b_stride,
                a,
                blas_ffi::HIP_R_16BF,
                a_ld,
                a_stride,
                &beta as *const f32 as *const c_void,
                y,
                blas_ffi::HIP_R_16BF,
                y_ld,
                y_stride,
                batch_count,
                blas_ffi::HIPBLAS_COMPUTE_32F,
                blas_ffi::HIPBLAS_GEMM_DEFAULT,
            )
        };
        debug_assert_eq!(status, blas_ffi::HIPBLAS_STATUS_SUCCESS, "gemm_strided_batched_atb: hipBLAS returned error {status}");
    }

    /// §100: raw (unsynced) hot-path counterpart of `kernels::gdn_chunk_decay_bf16`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gdn_chunk_decay(
        g: *const f32,
        ut_system: *mut c_void,
        intra_chunk_attn: *mut c_void,
        decay_exp_out: *mut f32,
        remaining_decay_out: *mut f32,
        chunk_decay_out: *mut f32,
        num_heads: i32,
        num_chunks: i32,
        chunk_size: i32,
        stream: *mut c_void,
    ) {
        let threads = 256;
        unsafe {
            kernels_ffi::launch_gdn_chunk_decay_bf16(
                g,
                ut_system,
                intra_chunk_attn,
                decay_exp_out,
                remaining_decay_out,
                chunk_decay_out,
                num_heads,
                num_chunks,
                chunk_size,
                threads,
                stream,
            )
        };
    }

    /// §100: raw (unsynced) hot-path counterpart of `kernels::gdn_chunk_utsolve_bf16`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gdn_chunk_utsolve(ut_system: *const c_void, rhs: *const c_void, x_out: *mut c_void, num_heads: i32, num_chunks: i32, chunk_size: i32, width: i32, stream: *mut c_void) {
        let threads = width.min(1024);
        unsafe { kernels_ffi::launch_gdn_chunk_utsolve_bf16(ut_system, rhs, x_out, num_heads, num_chunks, chunk_size, width, threads, stream) };
    }

    /// §100: raw (unsynced) hot-path counterpart of `kernels::l2norm_bf16`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn l2norm(x: *const c_void, out: *mut c_void, num_rows: i32, hidden_size: i32, eps: f32, post_scale: f32, stream: *mut c_void) {
        let threads = 256;
        unsafe { kernels_ffi::launch_l2norm_bf16(x, out, num_rows, hidden_size, eps, post_scale, threads, stream) };
    }

    /// §100: raw (unsynced) hot-path counterpart of
    /// `kernels::gdn_chunk_broadcast_scale_bf16`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gdn_chunk_broadcast_scale(src: *const c_void, scale: *const f32, dst: *mut c_void, num_tokens: i32, src_heads: i32, dst_heads: i32, head_dim: i32, n_rep: i32, chunk_size: i32, head_major_output: bool, stream: *mut c_void) {
        let threads = 128;
        unsafe { kernels_ffi::launch_gdn_chunk_broadcast_scale_bf16(src, scale, dst, num_tokens, src_heads, dst_heads, head_dim, n_rep, chunk_size, head_major_output as i32, threads, stream) };
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn causal_conv1d_update(
        hidden_states: *const c_void,
        conv_state: *mut c_void,
        weight: *const c_void,
        out: *mut c_void,
        batch: i32,
        conv_dim: i32,
        kernel_size: i32,
        stream: *mut c_void,
    ) {
        unsafe { kernels_ffi::launch_causal_conv1d_update_bf16(hidden_states, conv_state, weight, out, batch, conv_dim, kernel_size, stream) };
    }

    /// §103: raw (unsynced) hot-path counterpart of
    /// `kernels::causal_conv1d_prefill_bf16` -- ONE launch for a whole
    /// `num_tokens`-token chunk, replacing `num_tokens` separate
    /// `raw::causal_conv1d_update` calls.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn causal_conv1d_prefill(
        src: *const c_void,
        src_stride: i32,
        src_offset: i32,
        conv_state: *mut c_void,
        weight: *const c_void,
        out: *mut c_void,
        num_tokens: i32,
        conv_dim: i32,
        kernel_size: i32,
        stream: *mut c_void,
    ) {
        let threads = 256;
        unsafe { kernels_ffi::launch_causal_conv1d_prefill_bf16(src, src_stride, src_offset, conv_state, weight, out, num_tokens, conv_dim, kernel_size, threads, stream) };
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gdn_gate_beta(
        a: *const c_void,
        b: *const c_void,
        a_log: *const f32,
        dt_bias: *const f32,
        g_out: *mut f32,
        beta_out: *mut f32,
        num_heads: i32,
        stream: *mut c_void,
    ) {
        unsafe { kernels_ffi::launch_gdn_gate_beta_bf16(a, b, a_log, dt_bias, g_out, beta_out, num_heads, stream) };
    }

    /// §103: raw (unsynced) hot-path counterpart of
    /// `kernels::gdn_gate_beta_prefill_bf16` -- ONE launch for a whole
    /// `num_tokens`-token chunk, replacing `num_tokens` separate
    /// `raw::gdn_gate_beta` calls.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gdn_gate_beta_prefill(
        a_src: *const c_void,
        a_stride: i32,
        a_offset: i32,
        b_src: *const c_void,
        b_stride: i32,
        b_offset: i32,
        a_log: *const f32,
        dt_bias: *const f32,
        g_out: *mut f32,
        beta_out: *mut f32,
        num_tokens: i32,
        num_heads: i32,
        stream: *mut c_void,
    ) {
        let threads = 256;
        unsafe { kernels_ffi::launch_gdn_gate_beta_prefill_bf16(a_src, a_stride, a_offset, b_src, b_stride, b_offset, a_log, dt_bias, g_out, beta_out, num_tokens, num_heads, threads, stream) };
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gdn_recurrent_decode(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const f32,
        beta: *const f32,
        state: *mut f32,
        out: *mut c_void,
        num_heads: i32,
        num_k_heads: i32,
        head_dim: i32,
        stream: *mut c_void,
    ) {
        // §93 (rocprofv3-guided): this kernel launches only `num_heads`=32
        // blocks -- real cross-thread dependencies via `__syncthreads()`
        // within a head's 5-phase computation mean its work can't be split
        // across MORE blocks (no cross-block sync within one launch), but
        // widening each block (more threads covering the same head_dim*
        // head_dim state matrix, fewer stride-loop iterations per thread)
        // is a real, measured, reproducible win: a real per-kernel trace
        // (`rocprofv3 --kernel-trace --stats`, the first real ROCm
        // profiler available this session -- previously absent, guessed
        // around via per-layer-synced diagnostics instead) showed this
        // exact kernel's own average duration dropping monotonically as
        // threads increased: 60.5us (128, the old `next_power_of_two(head_dim)`
        // value) -> 39.4us (256) -> 28.7us (512) -> 24.9us (1024, this
        // GPU's real per-block thread ceiling -- `head_dim(128) * 8`).
        // Correctness re-verified (`real_greedy_generation_matches_real_qwen3_5_4b`)
        // at every step; the kernel body itself already handles any thread
        // count correctly via stride loops and `tid < head_dim` gating, so
        // this needed no kernel-logic change, only this launch-config one.
        let threads = ((head_dim as u32).next_power_of_two() * 8).min(1024) as i32;
        unsafe { kernels_ffi::launch_gdn_recurrent_decode_bf16(q, k, v, g, beta, state, out, num_heads, num_k_heads, head_dim, threads, stream) };
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rmsnorm_gated(x: *const c_void, gate: *const c_void, weight: *const f32, out: *mut c_void, num_rows: i32, hidden_size: i32, eps: f32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_rmsnorm_gated_bf16(x, gate, weight, out, num_rows, hidden_size, eps, 256, stream) };
    }

    pub unsafe fn add(a: *const c_void, b: *const c_void, out: *mut c_void, n: i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_add_bf16(a, b, out, n, stream) };
    }

    /// §115: real on-device fp32<->bf16 casts -- see
    /// `kernels::f32_to_bf16_cast`/`bf16_to_f32_cast`'s own doc comments.
    pub unsafe fn f32_to_bf16_cast(src: *const c_void, dst: *mut c_void, n: i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_f32_to_bf16_cast(src, dst, n, stream) };
    }

    pub unsafe fn bf16_to_f32_cast(src: *const c_void, dst: *mut c_void, n: i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_bf16_to_f32_cast(src, dst, n, stream) };
    }

    /// §116: real, unsynced in-place `buf *= *scalar_ptr` -- see
    /// `kernels::scale_bf16_by_device_scalar`'s own doc comment.
    #[allow(dead_code)]
    pub unsafe fn scale_bf16_by_device_scalar(buf: *mut c_void, scalar_ptr: *const c_void, n: i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_scale_bf16_by_device_scalar(buf, scalar_ptr, n, stream) };
    }

    pub unsafe fn scale_bf16_by_device_scalars_batched(buf: *mut c_void, scalars: *const c_void, n_per_head: i32, h: i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_scale_bf16_by_device_scalars_batched(buf, scalars, n_per_head, h, stream) };
    }

    pub unsafe fn split_last_dim(x: *const c_void, first: *mut c_void, second: *mut c_void, rows: i32, half: i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_split_last_dim_bf16(x, first, second, rows, half, stream) };
    }

    /// §93: `position` is a DEVICE pointer, not a host value -- see
    /// `kv_cache_append.hip`'s own comment for why.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv_cache_append(new_k: *const c_void, new_v: *const c_void, k_cache: *mut c_void, v_cache: *mut c_void, num_kv_heads: i32, max_seq_len: i32, head_dim: i32, position: *const i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_kv_cache_append_bf16(new_k, new_v, k_cache, v_cache, num_kv_heads, max_seq_len, head_dim, position, stream) };
    }

    /// §103: raw (unsynced) hot-path counterpart of
    /// `kernels::kv_cache_append_prefill_bf16` -- ONE launch for a whole
    /// `num_tokens`-token chunk, replacing `num_tokens` separate
    /// `raw::kv_cache_append` calls.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv_cache_append_prefill(new_k: *const c_void, new_v: *const c_void, k_cache: *mut c_void, v_cache: *mut c_void, num_tokens: i32, num_kv_heads: i32, max_seq_len: i32, head_dim: i32, position_buf: *const i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_kv_cache_append_prefill_bf16(new_k, new_v, k_cache, v_cache, num_tokens, num_kv_heads, max_seq_len, head_dim, position_buf, stream) };
    }

    /// §93: `position` is a DEVICE pointer -- see `rope.hip`'s own comment.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rope(x: *const c_void, out: *mut c_void, num_rows: i32, head_dim: i32, rotary_dim: i32, theta: f32, position: *const i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_rope_bf16(x, out, num_rows, head_dim, rotary_dim, theta, position, stream) };
    }

    /// §103: raw (unsynced) hot-path counterpart of
    /// `kernels::rope_prefill_bf16` -- ONE launch for a whole
    /// `num_tokens`-token chunk, replacing `num_tokens` separate
    /// `raw::rope` calls.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rope_prefill(x: *const c_void, out: *mut c_void, num_tokens: i32, num_heads: i32, head_dim: i32, rotary_dim: i32, theta: f32, position_buf: *const i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_rope_prefill_bf16(x, out, num_tokens, num_heads, head_dim, rotary_dim, theta, position_buf, stream) };
    }

    /// §92: `kv_stride` lets this read directly out of the real head-major
    /// KV cache (stride = `max_seq_len`) -- no more per-token
    /// tightly-packed copy first (see `attention.hip`'s header). §93:
    /// `position` is a DEVICE pointer; `kv_len = *position + 1` is derived
    /// inside the kernel, and shared memory is sized for the fixed
    /// `kv_stride` upper bound so the launch configuration itself never
    /// changes across tokens. §104: real, faster counterpart of the
    /// original scalar kernel this used to call -- see
    /// `kernels::attention_decode_split_bf16`'s own doc comment for the
    /// real split-KV technique and the real, measured speedup (1.65x-2.67x
    /// at this model's real decode `kv_len` range, growing with `kv_len`,
    /// see `docs/DECISIONS.md` §104). `ATTENTION_DECODE_KV_SPLIT` is the
    /// real per-output-dimension worker count. The original scalar
    /// kernel (`kernels::attention_decode_bf16`/
    /// `launch_attention_decode_bf16`) is kept as the real cross-
    /// validation oracle this kernel's own decisive test checks against
    /// (see `kernels.rs`'s `real_attention_decode_split_matches_attention_decode_bf16`),
    /// not used anywhere in the real forward pass anymore.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_decode_split(q: *const c_void, k: *const c_void, v: *const c_void, out: *mut c_void, num_q_heads: i32, num_kv_heads: i32, position: *const i32, kv_stride: i32, head_dim: i32, scaling: f32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_attention_decode_split_bf16(q, k, v, out, num_q_heads, num_kv_heads, position, kv_stride, head_dim, super::ATTENTION_DECODE_KV_SPLIT as i32, scaling, stream) };
    }

    /// §119: 2D batched causal attention for prefill (T < 128)
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_causal_prefill(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        num_tokens: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        positions: *const i32,
        kv_stride: i32,
        head_dim: i32,
        scaling: f32,
        max_chunk_kv_len: i32,
        stream: *mut c_void,
    ) {
        debug_assert!(
            max_chunk_kv_len <= 256,
            "attention_causal_prefill max_chunk_kv_len ({max_chunk_kv_len}) exceeds LDS sizing regime (<= 256); use batched GEMM attention instead"
        );
        unsafe {
            kernels_ffi::launch_attention_causal_prefill_bf16(
                q,
                k,
                v,
                out,
                num_tokens,
                num_q_heads,
                num_kv_heads,
                positions,
                kv_stride,
                head_dim,
                super::ATTENTION_DECODE_KV_SPLIT as i32,
                scaling,
                max_chunk_kv_len,
                stream,
            )
        };
    }

    /// §119: Fused attention QKV prep for prefill
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn fused_attn_qkv_prep(
        qkv_in: *const c_void,
        q_weight: *const c_void,
        k_weight: *const c_void,
        q_normed_out: *mut c_void,
        gate_out: *mut c_void,
        k_normed_out: *mut c_void,
        v_out: *mut c_void,
        num_tokens: i32,
        num_q_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        eps: f32,
        stream: *mut c_void,
    ) {
        unsafe {
            kernels_ffi::launch_fused_attn_qkv_prep_bf16(
                qkv_in,
                q_weight,
                k_weight,
                q_normed_out,
                gate_out,
                k_normed_out,
                v_out,
                num_tokens,
                num_q_heads,
                num_kv_heads,
                head_dim,
                eps,
                stream,
            )
        };
    }

    /// §93: advances the on-device position by 1 -- see
    /// `position_state.hip`'s own comment.
    pub unsafe fn increment_position(position: *mut i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_increment_position(position, stream) };
    }

    pub unsafe fn sigmoid_gate(x: *const c_void, gate: *const c_void, out: *mut c_void, n: i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_sigmoid_gate_bf16(x, gate, out, n, stream) };
    }

    pub unsafe fn swiglu(gate: *const c_void, up: *const c_void, out: *mut c_void, n: i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_swiglu_bf16(gate, up, out, n, stream) };
    }

    /// §114: real, launch-count-reducing fusion -- see
    /// `kernels::swiglu_strided_bf16`'s own doc comment.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn swiglu_strided(gate_up: *const c_void, out: *mut c_void, rows: i32, src_stride: i32, gate_offset: i32, up_offset: i32, len: i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_swiglu_strided_bf16(gate_up, out, rows, src_stride, gate_offset, up_offset, len, stream) };
    }

    pub unsafe fn embedding_lookup(table: *const c_void, token_ids: *const i32, out: *mut c_void, num_tokens: i32, hidden_size: i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_embedding_lookup_bf16(table, token_ids, out, num_tokens, hidden_size, stream) };
    }
}

/// `in_proj_qkv`(8192) + `in_proj_z`(4096) + `in_proj_b`(32) + `in_proj_a`(32)
/// concatenated along `out_features` -- one real GEMM instead of four (see
/// `model_loader::load_concat_bf16_weights`'s own doc comment for why this
/// is a real, exact concatenation of each tensor's own trained weights,
/// not an approximation).
pub const GDN_IN_PROJ_COMBINED_DIM: usize = GDN_CONV_DIM + GDN_VALUE_DIM + GDN_NUM_V_HEADS + GDN_NUM_V_HEADS; // 12352
/// `q_proj`(8192) + `k_proj`(1024) + `v_proj`(1024) concatenated.
pub const ATTN_QKV_COMBINED_DIM: usize = ATTN_NUM_HEADS * ATTN_HEAD_DIM * 2 + 2 * (ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM); // 10240
/// `gate_proj`(9216) + `up_proj`(9216) concatenated -- shared by both layer
/// types' MLP block.
pub const GATE_UP_COMBINED_DIM: usize = INTERMEDIATE_SIZE * 2; // 18432

/// §96: the maximum number of new prompt tokens `PrefillScratch` (and
/// `forward_prefill_chunk`) process in ONE real batched forward pass.
/// Prompts longer than this are processed in multiple chunks (still far
/// fewer than one `forward_one_token` call per token -- see
/// `forward_prefill`). Chosen generously above every real prompt this
/// crate's own benchmark (`benchmarks/harness_sdk/
/// run_4b_engine_comparison_benchmark.py`) sends (a few hundred tokens at
/// most): `PrefillScratch`'s buffers scale linearly with this constant and
/// are negligible either way (tens of MB at 256, next to ~9GB of real
/// weights), so there is no real pressure to tune it down.
pub const MAX_PREFILL_CHUNK: usize = 256;

/// §99: the real crossover point between the naive per-token scalar
/// attention kernel and real matrix-core GEMM attention, dispatched on
/// `kv_len` -- the SAME threshold-dispatch pattern `raw::gemm` already
/// uses for `rows==1` (GEVM) vs `rows>1` (real GEMM). Not guessed: this
/// repo's own prior real measurement (`TODO.md` §87, the "hard tier, full
/// attention" A/B) found the scalar kernel winning ~1.7x at `kv_len=128`
/// and losing ~2.6-2.8x to a GEMM-backed path at `kv_len=2048`. A real,
/// direct measurement THIS pass added (`docs/DECISIONS.md` §99): at the
/// actual representative prompt length this crate's own benchmark sends
/// (54 tokens), the GEMM path's `hipblasGemmEx` calls pay a real one-time
/// Tensile solution-selection cost per NEW shape (measured: a cold-call
/// wall-clock time 2.16x the warm steady-state) that the scalar kernel
/// never pays, making it a net LOSS at that scale despite GEMM attention
/// clearly winning at longer real prefill (401 tokens: a real, measured
/// 21% total-kernel-time reduction, `attention_decode`'s own bucket
/// dropping from 22.7% to under 1%). `128` is chosen to match §87's own
/// scalar-wins data point exactly, not split the difference arbitrarily.
pub const ATTENTION_GEMM_KV_LEN_THRESHOLD: usize = 128;

/// §104: the real number of threads cooperating on EACH output dimension
/// in `attention_decode_split_bf16` -- 4 gives a real, measured 1.65x-2.67x
/// decode-attention speedup over the original scalar kernel across this
/// model's real decode `kv_len` range (`docs/DECISIONS.md` §104's own
/// head-to-head kernel benchmark), and `head_dim(256) * 4 = 1024` is
/// exactly this hardware's real max threads-per-block -- the largest
/// `kv_split` this kernel can use without a second launch-configuration
/// dimension.
pub const ATTENTION_DECODE_KV_SPLIT: usize = 4;

/// §106: the real group size `quantize_w4a16.py` used -- must match
/// exactly, or `LinearWeight::Quantized`'s scale lookup silently reads
/// the wrong scale for a given weight index.
pub const W4A16_GROUP_SIZE: usize = 128;

/// Quantized-path LoRA (`quantized_lora.rs`): the fixed rank every real
/// `QuantLoraSlot` buffer is permanently sized to, real adapters' own
/// rank zero-padded up to this if smaller (every real adapter in this
/// repo uses r=8 -- see `results/adapters/*`'s own naming -- so this
/// gives real headroom, matching the real, already-serving Python
/// precedent's own `init_static_lora_buffers(max_rank=32)` in
/// `apps/runtime-triton/native_27b_engine.py`). Fixed, not dynamic, for
/// the same real reason `W4A16_GROUP_SIZE` is fixed: a HIP-graph-captured
/// kernel launch's integer parameters are baked in at capture time, so
/// this can never change per-adapter without invalidating the graph --
/// only the BUFFER VALUES change on a real swap, never this shape.
///
/// §131/§132: `activate_quantized_adapter` stores the active adapter's
/// real rank in `QuantLoraSlot::rank` (kept as real, useful metadata) and
/// uploads only the first `rank` rows of `lora_a`/`lora_b` via PCIe,
/// zeroing the `[rank, MAX_LORA_RANK)` tail on-device via
/// `hipMemsetAsync` (device-local, no PCIe cost).
///
/// §134: `apply()` does NOT pass `slot.rank` to the kernel -- it always
/// passes this fixed `MAX_LORA_RANK` constant, unconditionally, for
/// every slot, every call, regardless of whether any adapter is active.
/// §132 originally made this dynamic (real rank, skip entirely when
/// `rank==0`) as a real optimization, but that's a genuine HIP-graph-
/// safety bug: `GraphedDecodeState` freezes whichever kernel launches
/// actually got ISSUED on the real captured call, and a Rust-level `if
/// slot.rank == 0 { continue; }` means those launches may never be
/// issued at all if capture happens before the first real adapter
/// activation (the common server-startup case) -- confirmed directly via
/// `diagnose_graphed_decode_sees_lora_activated_after_first_capture`
/// (graphed and eager decode diverged, 228,486/248,320 logits, once an
/// adapter was activated after a rank=0 capture; 0 differ after the fix).
/// The tail rows beyond the real rank are always zero, so summing over
/// the full fixed `MAX_LORA_RANK` instead of the real rank changes
/// nothing mathematically -- only the wasted compute on zero terms.
///
/// §134 shrunk this from 32 to 16; §135 shrunk it again, 16 to 8, after
/// confirming (real evidence, not a guess) that every real adapter ever
/// trained in this repo -- dozens of directories under
/// `results/adapters/`, every domain -- uses r=8 uniformly. This is now
/// exactly the real rank, zero slack: a future adapter trained at a
/// higher rank fails LOUDLY at `QuantizedLoraAdapter::load_from_dir`
/// (`r > MAX_LORA_RANK` check), never silently truncated or corrupted --
/// bumping this constant back up and rebuilding is a one-line, reversible
/// fix if that's ever needed, not a redesign. Real effect: eliminates the
/// LAST of the wasted zero-valued FMAs `lora_delta_accumulate_kernel`'s
/// per-thread loop was doing, and halves `QuantLoraSlot`'s VRAM footprint
/// again.
pub const MAX_LORA_RANK: usize = 8;

/// Quantized-path LoRA: one real target sub-projection's row range within
/// a (possibly fused) `LinearWeight::Quantized` buffer, plus its own
/// permanently-allocated, HIP-graph-safe static LoRA buffers. See
/// `quantized_lora.rs`'s header doc for the full rationale -- this is the
/// additive-at-inference-time analogue of `lora.rs`'s bf16 in-place
/// weight folding, architecturally different because folding a rank-r
/// delta into a quantized buffer would mean re-quantizing on every swap
/// (lossy AND slow), exactly the real tradeoff the already-shipped Python
/// engine (`apps/runtime-triton/w4a16_loader.py`) made the same call on.
pub struct QuantLoraSlot {
    /// Element offset (in ROWS, i.e. output features) into the parent
    /// `LinearWeight::Quantized`'s fused output -- 0 for a standalone
    /// weight (`down_proj`/`o_proj`), nonzero for the 2nd+ real
    /// sub-projection concatenated into `gate_up_proj`/`qkv_proj` (same
    /// convention `lora.rs`'s `fold_into`'s own `row_offset` uses).
    pub row_offset: usize,
    pub out_features: usize,
    /// §132: the real rank of the currently-active adapter (0 when no
    /// adapter is active / slot is zeroed). Real, useful host-side
    /// metadata -- §134 stopped `apply()` from reading it for kernel
    /// dispatch (a genuine HIP-graph-safety bug, see `MAX_LORA_RANK`'s
    /// own doc comment), but it's kept here.
    pub rank: usize,
    /// `[MAX_LORA_RANK, out_features]`, bf16, zeroed at alloc and after
    /// `clear_quantized_adapter`. §131/§132: col-major layout (one
    /// transposition vs. the original §128 `[out_features, MAX_LORA_RANK]`
    /// row-major). Real adapter data occupies rows `[0, r)` -- a
    /// contiguous prefix of `r * out_features` elements -- enabling a
    /// single `hipMemcpyAsync` of exactly the real data, with no zero-
    /// column stride interleaving. `lora_alpha/r` pre-folded in.
    ///
    /// §135: `lora_a`/`lora_mid` moved OUT of this per-slot struct into
    /// `QuantLoraMidFused` (one shared instance per `LinearWeight`, not
    /// per slot) -- see that struct's own doc comment for why. `lora_b`
    /// stays here because `lora_delta_accumulate` genuinely needs a
    /// separate call per slot (different `row_offset`/`out_features`
    /// into `y` each), so there's nothing to fuse on this side.
    /// `in_features` moved out along with them (it belongs to the shared
    /// `QuantLoraMidFused.in_features` now -- every slot of a given
    /// `LinearWeight` shares the same `in_features` by construction, so
    /// keeping a redundant per-slot copy here would've been dead data).
    pub lora_b: DeviceBuffer<u16>,
}

impl QuantLoraSlot {
    fn alloc(row_offset: usize, out_features: usize) -> Result<Self, HipError> {
        // §131/§132: col-major [MAX_LORA_RANK, out_features] (was [out_features, MAX_LORA_RANK]).
        let mut lora_b: DeviceBuffer<u16> = DeviceBuffer::alloc(MAX_LORA_RANK * out_features)?;
        lora_b.fill_zero()?;
        Ok(QuantLoraSlot { row_offset, out_features, rank: 0, lora_b })
    }
}

/// §135: the real fix for `gemv_bf16_kernel`'s call-count share of the
/// quantized-LoRA fixed cost (real profiling: `rocprofv3` found this
/// kernel's per-call cost roughly FLAT regardless of rank -- 32 vs. 8:
/// 3.5us vs. 3.2us -- consistent with it being block-count/launch-bound,
/// not per-thread-work-bound, unlike `lora_delta_accumulate_kernel`;
/// §134's `MAX_LORA_RANK` shrink already addressed that one). Since
/// per-call cost doesn't shrink much with rank, the real lever left is
/// CALL COUNT: `gate_up_proj` (2 real sub-projections) and `qkv_proj` (3)
/// each called `raw::lora_mid` once PER SLOT before this -- this struct
/// holds ONE shared, fused `lora_a`/`lora_mid` pair per `LinearWeight`
/// covering ALL its real slots at once, cut down to a SINGLE
/// `raw::lora_mid` call regardless of slot count (`down_proj`/`o_proj`,
/// with only 1 real slot, are structurally unaffected -- 1 slot fused is
/// still 1 call).
///
/// Real layout: `lora_a` is `[num_slots * MAX_LORA_RANK, in_features]`
/// row-major -- slot `i`'s real rank rows occupy
/// `[i*MAX_LORA_RANK, i*MAX_LORA_RANK + r)`, always-zero elsewhere,
/// exactly the same real "zero-padding costs nothing mathematically"
/// property `QuantLoraSlot`'s own per-slot buffers already relied on.
/// `lora_mid` is the matching `[num_slots * MAX_LORA_RANK]` scratch,
/// recomputed fresh by ONE `raw::lora_mid` call every real `apply()`.
/// Slot `i`'s own contribution is the sub-slice
/// `lora_mid[i*MAX_LORA_RANK .. (i+1)*MAX_LORA_RANK]`, read via
/// `DeviceBuffer::as_device_ptr_at` (a real, already-existing offset
/// accessor, not a new primitive) when calling `lora_delta_accumulate`
/// for that slot -- `lora_delta_accumulate` itself is UNCHANGED, still
/// one real call per slot (see `QuantLoraSlot::lora_b`'s own doc comment
/// for why that side isn't fused).
pub struct QuantLoraMidFused {
    pub lora_a: DeviceBuffer<u16>,
    pub lora_mid: DeviceBuffer<u16>,
    pub in_features: usize,
    pub num_slots: usize,
}

impl QuantLoraMidFused {
    fn alloc(in_features: usize, num_slots: usize) -> Result<Self, HipError> {
        let mut lora_a: DeviceBuffer<u16> = DeviceBuffer::alloc(num_slots * MAX_LORA_RANK * in_features)?;
        lora_a.fill_zero()?;
        let lora_mid: DeviceBuffer<u16> = DeviceBuffer::alloc(num_slots * MAX_LORA_RANK)?;
        Ok(QuantLoraMidFused { lora_a, lora_mid, in_features, num_slots })
    }
}

/// §106: one real `nn.Linear`-shaped weight, either still real unquantized
/// bf16 or real W4A16-quantized -- every one of this engine's 7 GEMV hot-
/// path weights (per-layer qkv/o/in_proj/out/gate_up/down, plus lm_head)
/// is one of these. Which variant a given real checkpoint produces is
/// decided once, per tensor, in `ModelWeights::load` (checking the real
/// safetensors index for a `.qweight` sibling -- same discipline as the
/// tied/untied `lm_head` detection above), never assumed from the model
/// size or a config flag.
pub enum LinearWeight {
    Bf16(DeviceBuffer<u16>),
    Quantized {
        qweight: DeviceBuffer<u32>,
        scales: DeviceBuffer<u16>,
        /// Empty for a weight LoRA never targets (GDN's own
        /// `in_proj_combined`/`out_proj`, `lm_head` -- see
        /// `quantized_lora.rs`'s target-module list, mirroring
        /// `lora.rs`'s own real `LoraLayer` shape exactly), populated by
        /// `ModelWeights::load` for the real LoRA-targetable fields
        /// (`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj`). Enum
        /// variant fields can't take their own `pub` qualifier in Rust --
        /// this is reachable from `quantized_lora.rs` the same way
        /// `qweight`/`scales` are reachable from `model.rs` itself
        /// (same-crate visibility of a `pub enum`'s variant fields).
        lora_slots: Vec<QuantLoraSlot>,
        /// §135: `Some` exactly when `lora_slots` is non-empty (one
        /// shared fused mid-computation buffer covering every real slot
        /// in `lora_slots`); `None` for a weight LoRA never targets, same
        /// real condition `lora_slots.is_empty()` already tracks. See
        /// `QuantLoraMidFused`'s own doc comment.
        lora_mid_fused: Option<QuantLoraMidFused>,
    },
}

impl LinearWeight {
    /// Dispatches to the real bf16 GEMV or the real W4A16 fused
    /// dequant+GEMV kernel, whichever this weight actually is -- the
    /// single real call site every one of this engine's hot-path linear
    /// layers goes through now, decode-only (`rows` is always 1 for
    /// every real caller; prefill for a quantized layer uses this same
    /// per-token path in a loop, not a batched GEMM -- see
    /// `docs/DECISIONS.md` §106). For a quantized weight with real LoRA
    /// slots, ALSO runs the real additive LoRA accumulation after the
    /// unmodified base kernel (see `QuantLoraSlot`'s own doc) -- a
    /// quantized weight with zero slots (the common case: GDN's own
    /// in_proj/out_proj, `lm_head`) pays zero extra cost, identical to
    /// before this existed.
    pub unsafe fn apply(&self, x: *const c_void, y: *mut c_void, in_features: i32, out_features: i32, stream: *mut c_void) {
        match self {
            LinearWeight::Bf16(buf) => unsafe {
                raw::gemm(std::ptr::null_mut(), x, buf.as_device_ptr(), y, 1, in_features, out_features, stream);
            },
            LinearWeight::Quantized { qweight, scales, lora_slots, lora_mid_fused } => unsafe {
                raw::linear_quantized(x, qweight.as_device_ptr(), scales.as_device_ptr(), y, in_features, out_features, W4A16_GROUP_SIZE as i32, stream);
                // §134: ALWAYS launch these LoRA kernels for every real
                // slot, unconditionally, with the FIXED `MAX_LORA_RANK`
                // (never a per-adapter-varying rank read from a host-side
                // field). §132's original `if slot.rank == 0 { continue;
                // }` skip -- reading a plain host-side `usize` to decide
                // whether to issue a kernel launch -- was a real,
                // confirmed HIP-graph-safety bug, not just a missed
                // optimization: `GraphedDecodeState` captures whatever
                // kernel launches actually get ISSUED on the FIRST real
                // decode call, and REPLAYS that exact fixed sequence on
                // every later call, never re-running this Rust branch. If
                // that first call happens before any adapter is ever
                // activated (the real, common server-startup case), the
                // LoRA kernels are never recorded, and no later `POST
                // /lora-adapters` activation can ever take effect for
                // decode, silently, forever, on that server. Confirmed
                // directly via `diagnose_graphed_decode_sees_lora_
                // activated_after_first_capture`: graphed and eager
                // decode agreed exactly while unadapted (0 logits
                // differ), then diverged (228,486/248,320 differ) once an
                // adapter was activated post-capture, under the buggy
                // code. Correctness here: the tail rows beyond any real
                // adapter's rank are always zero (`activate_quantized_
                // adapter`'s `fill_zero_from_async`, or the whole buffer
                // when inactive), so summing over the fixed
                // `MAX_LORA_RANK` instead of a real rank changes NOTHING
                // mathematically -- the extra terms are exact zeros.
                //
                // §135: ONE fused `raw::lora_mid` call covers every real
                // slot in `lora_slots` at once (see `QuantLoraMidFused`'s
                // own doc comment for why -- real profiling found this
                // kernel's cost is call-count-, not rank-, bound).
                // `lora_delta_accumulate` stays one real call per slot
                // (different `row_offset`/`out_features` into `y` each --
                // nothing to fuse there), reading its slice of the fused
                // `lora_mid` buffer via the already-existing
                // `as_device_ptr_at` offset accessor.
                if let Some(fused) = lora_mid_fused {
                    let total_rank = (fused.num_slots * MAX_LORA_RANK) as i32;
                    raw::lora_mid(x, fused.lora_a.as_device_ptr(), fused.lora_mid.as_device_ptr() as *mut c_void, fused.in_features as i32, total_rank, stream);
                    for (i, slot) in lora_slots.iter().enumerate() {
                        let mid_slice = fused.lora_mid.as_device_ptr_at(i * MAX_LORA_RANK);
                        let y_slot = (y as *mut u16).add(slot.row_offset) as *mut c_void;
                        raw::lora_delta_accumulate(mid_slice, slot.lora_b.as_device_ptr(), y_slot, slot.out_features as i32, MAX_LORA_RANK as i32, stream);
                    }
                }
            },
        }
    }

    /// §108: the real batched-prefill (`num_tokens > 1`) counterpart of
    /// `apply` above -- dispatches to hipBLAS (bf16) or the new tiled
    /// `w4a16_gemm_prefill` kernel (quantized), whichever this weight
    /// actually is. `x`/`y` are row-major `[num_tokens, features]`, same
    /// convention `raw::gemm`'s own `rows>1` branch already uses.
    ///
    /// Real, disclosed scope boundary: quantized-path LoRA (`lora_slots`)
    /// is NOT applied here -- only in the decode-only `apply` above. A
    /// prompt's own prefill pass always runs unadapted even with a real
    /// quantized LoRA adapter active; only tokens generated after that
    /// (the decode loop) see the adapter's effect. Closing this gap needs
    /// a real second fused kernel for the batched WMMA INT8 prefill path
    /// (`w4a16_gemm_prefill_wmma_int8.hip`) -- a substantially bigger,
    /// separate undertaking, not done this round.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn apply_prefill(&self, handle: blas_ffi::HipblasHandle, x: *const c_void, y: *mut c_void, num_tokens: i32, in_features: i32, out_features: i32, stream: *mut c_void) {
        match self {
            LinearWeight::Bf16(buf) => unsafe {
                raw::gemm(handle, x, buf.as_device_ptr(), y, num_tokens, in_features, out_features, stream);
            },
            LinearWeight::Quantized { qweight, scales, .. } => unsafe {
                raw::linear_quantized_prefill(x, qweight.as_device_ptr(), scales.as_device_ptr(), y, in_features, out_features, W4A16_GROUP_SIZE as i32, num_tokens, stream);
            },
        }
    }

    /// Real, loud failure (not a silent wrong computation) for the one
    /// real operation that still doesn't support quantized weights:
    /// in-place weight-fold LoRA (`lora.rs`, bf16-only, §101). Quantized
    /// weights get real LoRA a different way -- additive, via
    /// `QuantLoraSlot`/`quantized_lora.rs`, not by folding into these
    /// buffers -- so this restriction is real and permanent, not a scope
    /// gap. Batched prefill no longer needs this -- see `apply_prefill`
    /// above (§108).
    pub fn as_bf16(&self) -> &DeviceBuffer<u16> {
        match self {
            LinearWeight::Bf16(buf) => buf,
            LinearWeight::Quantized { .. } => panic!(
                "LinearWeight::as_bf16 called on a quantized weight -- in-place weight-fold LoRA (lora.rs) doesn't support W4A16-quantized layers; see quantized_lora.rs for the real additive equivalent (docs/DECISIONS.md §106/§108)"
            ),
        }
    }

    /// Mutable counterpart of `as_bf16`, same real restriction.
    pub fn as_bf16_mut(&mut self) -> &mut DeviceBuffer<u16> {
        match self {
            LinearWeight::Bf16(buf) => buf,
            LinearWeight::Quantized { .. } => panic!(
                "LinearWeight::as_bf16_mut called on a quantized weight -- in-place weight-fold LoRA (lora.rs) doesn't support W4A16-quantized layers; see quantized_lora.rs for the real additive equivalent (docs/DECISIONS.md §106/§108)"
            ),
        }
    }
}

pub struct GdnLayerWeights {
    pub input_layernorm: DeviceBuffer<u16>,
    pub post_attention_layernorm: DeviceBuffer<u16>,
    /// `[GDN_IN_PROJ_COMBINED_DIM, hidden]` -- see the constant's own doc.
    pub in_proj_combined: LinearWeight,
    pub conv1d_weight: DeviceBuffer<u16>,
    pub a_log: DeviceBuffer<f32>,
    pub dt_bias: DeviceBuffer<f32>,
    /// Native f32 in the real checkpoint (see `rmsnorm_gated.hip`'s header).
    pub norm_weight: DeviceBuffer<f32>,
    pub out_proj: LinearWeight,
    /// `[GATE_UP_COMBINED_DIM, hidden]`.
    pub gate_up_proj: LinearWeight,
    pub down_proj: LinearWeight,
}

pub struct AttnLayerWeights {
    pub input_layernorm: DeviceBuffer<u16>,
    pub post_attention_layernorm: DeviceBuffer<u16>,
    /// `[ATTN_QKV_COMBINED_DIM, hidden]`.
    pub qkv_proj: LinearWeight,
    pub q_norm: DeviceBuffer<u16>,
    pub k_norm: DeviceBuffer<u16>,
    pub o_proj: LinearWeight,
    /// `[GATE_UP_COMBINED_DIM, hidden]`.
    pub gate_up_proj: LinearWeight,
    pub down_proj: LinearWeight,
}

pub enum LayerWeights {
    Gdn(GdnLayerWeights),
    Attn(AttnLayerWeights),
}

pub struct ModelWeights {
    pub embed_tokens: DeviceBuffer<u16>,
    /// Real, checked-not-assumed per-checkpoint fact: most Qwen3.5 sizes
    /// (0.8B/2B/4B) tie `lm_head` to `embed_tokens` (`tie_word_embeddings:
    /// true`, no separate `lm_head.weight` tensor in the checkpoint), but
    /// 9B does NOT (`lm_head.weight` present in its real safetensors
    /// index, distinct learned weights) -- confirmed by reading each
    /// model's own `model.safetensors.index.json` directly rather than
    /// assuming every size ties embeddings the way 4B does. `None` means
    /// tied (reuse `embed_tokens` for the output projection); `Some` means
    /// this checkpoint has its own real, separate output weights.
    /// `None` means tied (see `embed_tokens`'s own doc); `Some` means this
    /// checkpoint has its own real, separate output weights -- either
    /// still bf16 or, per §106, real W4A16-quantized (both real
    /// possibilities, checked from the real checkpoint, never assumed).
    pub lm_head: Option<LinearWeight>,
    pub final_norm: DeviceBuffer<u16>,
    pub layers: Vec<LayerWeights>,
    /// §132: total number of u16 elements in one full adapter's real data
    /// (rank=actual_r rows of lora_a + rank rows of lora_b, summed over
    /// all slots). Zero for a bf16-only checkpoint. Written once by
    /// `attach_lora_slots` / `ModelWeights::load`; used by
    /// `activate_quantized_adapter` to allocate the staging `PinnedBuffer`.
    #[allow(dead_code)]
    pub lora_real_data_elems: usize,
}

/// Attaches real, permanently-allocated (zeroed) quantized-LoRA slots to
/// `w` for each real sub-projection in `sub_projections`
/// (`(row_offset, out_features)` pairs, in the SAME row order the base
/// weight itself was concatenated in -- see each call site below), all
/// sharing `in_features`. A no-op for a `LinearWeight::Bf16` (that path's
/// real LoRA is `lora.rs`'s in-place fold instead, needing no slots).
///
/// Returns the number of u16 elements consumed by the slots' REAL data
/// (lora_a: `actual_rank * in_features`, lora_b: `actual_rank * out_features`
/// per slot) -- accumulated into `ModelWeights::lora_real_data_elems` by
/// the caller so `activate_quantized_adapter` can size its staging buffer.
/// `actual_rank` is set to 0 at alloc time (no active adapter yet); it is
/// written per-slot by `activate_quantized_adapter`.
fn attach_lora_slots(mut w: LinearWeight, in_features: usize, sub_projections: &[(usize, usize)]) -> Result<LinearWeight, String> {
    if let LinearWeight::Quantized { lora_slots, lora_mid_fused, .. } = &mut w {
        let mut slots = Vec::with_capacity(sub_projections.len());
        for &(row_offset, out_features) in sub_projections {
            let slot = QuantLoraSlot::alloc(row_offset, out_features)
                .map_err(|e| format!("allocating quantized LoRA slot (row_offset={row_offset}, out_features={out_features}, in_features={in_features}): {e}"))?;
            slots.push(slot);
        }
        // §135: ONE shared fused mid-buffer covering every real slot above.
        let fused = QuantLoraMidFused::alloc(in_features, sub_projections.len())
            .map_err(|e| format!("allocating quantized LoRA fused mid buffer (in_features={in_features}, num_slots={}): {e}", sub_projections.len()))?;
        *lora_slots = slots;
        *lora_mid_fused = Some(fused);
    }
    Ok(w)
}

impl ModelWeights {
    /// §106: real, checked (not assumed) fact about THIS loaded
    /// checkpoint -- does it have any real W4A16-quantized layers?
    /// Checked against the first layer's own `gate_up_proj` variant
    /// (quantization is applied uniformly across a real checkpoint by
    /// `quantize_w4a16.py`'s own construction, so one representative
    /// field is a real, documented sample, not a guess). Callers use
    /// this to pick a real prefill strategy: quantized layers have no
    /// batched-GEMM kernel yet (§106's disclosed scope decision), so a
    /// quantized checkpoint must prefill via the per-token path instead
    /// of the faster batched one bf16 checkpoints use.
    pub fn is_quantized(&self) -> bool {
        self.layers.iter().any(|l| {
            let gate_up = match l {
                LayerWeights::Gdn(g) => &g.gate_up_proj,
                LayerWeights::Attn(a) => &a.gate_up_proj,
            };
            matches!(gate_up, LinearWeight::Quantized { .. })
        })
    }

    /// Runs the final vocab projection (`hidden -> VOCAB_SIZE` logits)
    /// through whichever real weight this checkpoint actually has: its
    /// own separate `lm_head` (bf16 or quantized) if untied, else the
    /// tied `embed_tokens` (bf16 -- `embed_tokens` itself is never
    /// quantized, see `quantize_w4a16.py`'s own scope decision) --
    /// matching HF `transformers`' `tie_word_embeddings` semantics
    /// exactly, not a guess.
    pub unsafe fn lm_head_apply(&self, x: *const c_void, y: *mut c_void, stream: *mut c_void) {
        match &self.lm_head {
            Some(w) => unsafe { w.apply(x, y, HIDDEN_SIZE as i32, VOCAB_SIZE as i32, stream) },
            None => unsafe {
                raw::gemm(std::ptr::null_mut(), x, self.embed_tokens.as_device_ptr(), y, 1, HIDDEN_SIZE as i32, VOCAB_SIZE as i32, stream);
            },
        }
    }

    /// Loads every real weight this forward pass needs, for all
    /// `NUM_LAYERS` layers, from the real Qwen3.5 checkpoint at
    /// `snapshot_dir` (any real size in the family -- dims come from this
    /// crate's own `HIDDEN_SIZE`/`NUM_LAYERS`/etc. consts, selected at
    /// build time to match whichever snapshot is actually being loaded).
    /// Real network I/O to disk (via mmap), real bf16/f32/quantized
    /// decoding -- no placeholder tensors anywhere. §106: each of the 7
    /// real GEMV hot-path weights is loaded bf16 or W4A16-quantized
    /// depending on what's REALLY in this checkpoint's own safetensors
    /// index (`load_linear`/`load_concat_linear` below check for a real
    /// `.qweight` sibling tensor) -- a bf16 snapshot and a
    /// `quantize_w4a16.py`-produced one both load correctly through this
    /// SAME function, no separate code path needed.
    pub fn load(snapshot_dir: &Path) -> Result<Self, String> {
        let weight_map = crate::model_loader::load_weight_map(snapshot_dir)?;
        let is_quantized = |name: &str| weight_map.contains_key(&format!("{name}.qweight"));
        let load_linear = |name: &str| -> Result<LinearWeight, String> {
            if is_quantized(name) {
                let (qweight, scales) = crate::model_loader::load_w4a16_weight(snapshot_dir, name)?;
                Ok(LinearWeight::Quantized { qweight, scales, lora_slots: Vec::new(), lora_mid_fused: None })
            } else {
                Ok(LinearWeight::Bf16(load_bf16_weight(snapshot_dir, name)?))
            }
        };
        let load_concat_linear = |names: &[&str]| -> Result<LinearWeight, String> {
            if is_quantized(names[0]) {
                let (qweight, scales) = crate::model_loader::load_concat_w4a16_weights(snapshot_dir, names)?;
                Ok(LinearWeight::Quantized { qweight, scales, lora_slots: Vec::new(), lora_mid_fused: None })
            } else {
                Ok(LinearWeight::Bf16(load_concat_bf16_weights(snapshot_dir, names)?))
            }
        };
        // Real LoRA target sub-projection row layout within each fused
        // buffer -- MUST match the concatenation order `load_concat_linear`
        // is called with just below exactly (same real discipline as
        // `ATTN_QKV_COMBINED_DIM`/`GATE_UP_COMBINED_DIM`'s own doc
        // comments: `q_proj`(2x -- real query+gate fused, see
        // `ATTN_QKV_COMBINED_DIM`'s doc) + `k_proj` + `v_proj`;
        // `gate_proj` + `up_proj`). GDN's own `in_proj_combined`/
        // `out_proj` and `lm_head` are never real LoRA targets (see
        // `lora.rs`'s own `LoraLayer`/`adapter_config.json` `target_modules`
        // -- no GDN module names there either), so they're never passed
        // through `attach_lora_slots` below.
        let q_out = 2 * ATTN_NUM_HEADS * ATTN_HEAD_DIM;
        let k_out = ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM;
        let v_out = ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM;
        let qkv_slot_spec = [(0usize, q_out), (q_out, k_out), (q_out + k_out, v_out)];
        let gate_up_slot_spec = [(0usize, INTERMEDIATE_SIZE), (INTERMEDIATE_SIZE, INTERMEDIATE_SIZE)];

        let embed_tokens =
            load_bf16_weight(snapshot_dir, "model.language_model.embed_tokens.weight")?;
        let lm_head = if weight_map.contains_key("lm_head.weight") || weight_map.contains_key("lm_head.weight.qweight") {
            Some(load_linear("lm_head.weight")?)
        } else {
            None
        };
        let final_norm = load_bf16_weight(snapshot_dir, "model.language_model.norm.weight")?;

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            let p = format!("model.language_model.layers.{i}");
            let gate_name = format!("{p}.mlp.gate_proj.weight");
            let up_name = format!("{p}.mlp.up_proj.weight");
            let gate_up_proj = attach_lora_slots(load_concat_linear(&[&gate_name, &up_name])?, HIDDEN_SIZE, &gate_up_slot_spec)?;

            if is_full_attention_layer(i) {
                let q_name = format!("{p}.self_attn.q_proj.weight");
                let k_name = format!("{p}.self_attn.k_proj.weight");
                let v_name = format!("{p}.self_attn.v_proj.weight");
                let qkv_proj = attach_lora_slots(load_concat_linear(&[&q_name, &k_name, &v_name])?, HIDDEN_SIZE, &qkv_slot_spec)?;
                let o_proj = attach_lora_slots(load_linear(&format!("{p}.self_attn.o_proj.weight"))?, ATTN_NUM_HEADS * ATTN_HEAD_DIM, &[(0, HIDDEN_SIZE)])?;
                let down_proj = attach_lora_slots(load_linear(&format!("{p}.mlp.down_proj.weight"))?, INTERMEDIATE_SIZE, &[(0, HIDDEN_SIZE)])?;
                layers.push(LayerWeights::Attn(AttnLayerWeights {
                    input_layernorm: load_bf16_weight(
                        snapshot_dir,
                        &format!("{p}.input_layernorm.weight"),
                    )?,
                    post_attention_layernorm: load_bf16_weight(
                        snapshot_dir,
                        &format!("{p}.post_attention_layernorm.weight"),
                    )?,
                    qkv_proj,
                    q_norm: load_bf16_weight(snapshot_dir, &format!("{p}.self_attn.q_norm.weight"))?,
                    k_norm: load_bf16_weight(snapshot_dir, &format!("{p}.self_attn.k_norm.weight"))?,
                    o_proj,
                    gate_up_proj,
                    down_proj,
                }));
            } else {
                let qkv_name = format!("{p}.linear_attn.in_proj_qkv.weight");
                let z_name = format!("{p}.linear_attn.in_proj_z.weight");
                let b_name = format!("{p}.linear_attn.in_proj_b.weight");
                let a_name = format!("{p}.linear_attn.in_proj_a.weight");
                let in_proj_combined =
                    load_concat_linear(&[&qkv_name, &z_name, &b_name, &a_name])?;
                let down_proj = attach_lora_slots(load_linear(&format!("{p}.mlp.down_proj.weight"))?, INTERMEDIATE_SIZE, &[(0, HIDDEN_SIZE)])?;
                layers.push(LayerWeights::Gdn(GdnLayerWeights {
                    input_layernorm: load_bf16_weight(
                        snapshot_dir,
                        &format!("{p}.input_layernorm.weight"),
                    )?,
                    post_attention_layernorm: load_bf16_weight(
                        snapshot_dir,
                        &format!("{p}.post_attention_layernorm.weight"),
                    )?,
                    in_proj_combined,
                    conv1d_weight: load_bf16_weight(
                        snapshot_dir,
                        &format!("{p}.linear_attn.conv1d.weight"),
                    )?,
                    a_log: load_f32_param(snapshot_dir, &format!("{p}.linear_attn.A_log"))?,
                    dt_bias: load_f32_param(snapshot_dir, &format!("{p}.linear_attn.dt_bias"))?,
                    norm_weight: load_f32_param(snapshot_dir, &format!("{p}.linear_attn.norm.weight"))?,
                    out_proj: load_linear(&format!("{p}.linear_attn.out_proj.weight"))?,
                    gate_up_proj,
                    down_proj,
                }));
            }
        }

        Ok(ModelWeights {
            embed_tokens,
            lm_head,
            final_norm,
            layers,
            lora_real_data_elems: 0,  // §132: populated by quantized_lora.rs at first activate
        })
    }
}

pub struct GdnLayerState {
    /// `[conv_dim, kernel_size-1]`, read+written by `causal_conv1d_update`.
    pub conv_state: DeviceBuffer<u16>,
    /// `[num_v_heads, head_dim, head_dim]` f32, read+written by
    /// `gdn_recurrent_decode`.
    pub recurrent_state: DeviceBuffer<f32>,
}

pub struct AttnLayerState {
    /// `[num_kv_heads, max_seq_len, head_dim]`, head-major (matches
    /// `attention_decode_bf16`'s expected read layout).
    pub k_cache: DeviceBuffer<u16>,
    pub v_cache: DeviceBuffer<u16>,
}

pub enum LayerState {
    Gdn(GdnLayerState),
    Attn(AttnLayerState),
}

/// Every intermediate tensor a single decode step needs, pre-allocated
/// ONCE and reused every token -- see this module's own doc comment for
/// why (§91: this used to be ~20 fresh `hipMalloc`/`hipFree` pairs per
/// layer per token). Sized for the UNION of what a GDN layer and an
/// attention layer each need; a few hundred KB total, negligible next to
/// the ~9GB of real weights.
pub struct Scratch {
    token_ids_dev: DeviceBuffer<i32>,
    /// §93: the current decode position, read by `rope`/`kv_cache_append`/
    /// `attention_decode` through a DEVICE pointer rather than a host
    /// value -- required so a future HIP-Graph-captured decode step's
    /// recorded launch parameters (a fixed pointer) stay valid across a
    /// replay even though the value they point to changes every token.
    /// Written once per token in `forward_one_token`, before the layer
    /// loop (today's eager path); a graphed replay loop would instead
    /// advance it on-device via `kernels::launch_increment_position`.
    position_buf: DeviceBuffer<i32>,
    normed: DeviceBuffer<u16>,
    after_mixer: DeviceBuffer<u16>,
    normed2: DeviceBuffer<u16>,
    /// §92: `gate_proj`+`up_proj` fused into one GEMM call; `gate`/`up`
    /// are contiguous offset VIEWS into this buffer at SwiGLU time, not
    /// separate allocations (see `mlp_block`).
    gate_up_out: DeviceBuffer<u16>,
    swiglu_out: DeviceBuffer<u16>,
    mlp_out: DeviceBuffer<u16>,

    /// §92: `in_proj_qkv`+`in_proj_z`+`in_proj_b`+`in_proj_a` fused into
    /// one GEMM call; the qkv/z/b/a portions are contiguous offset VIEWS
    /// into this buffer (see `gdn_layer_forward`), not separate buffers.
    gdn_in_proj_out: DeviceBuffer<u16>,
    /// `causal_conv1d_update`'s own output -- a real, distinct buffer
    /// (not a view) since it's a kernel WRITE, not a read-only split;
    /// query/key/value are then contiguous offset views into THIS buffer.
    gdn_mixed_qkv: DeviceBuffer<u16>,
    gdn_g: DeviceBuffer<f32>,
    gdn_beta: DeviceBuffer<f32>,
    gdn_out: DeviceBuffer<u16>,
    gdn_normed_gated: DeviceBuffer<u16>,
    gdn_mixer_out: DeviceBuffer<u16>,

    /// §92: `q_proj`+`k_proj`+`v_proj` fused into one GEMM call; q/k/v
    /// portions are contiguous offset views into this buffer.
    attn_qkv_out: DeviceBuffer<u16>,
    attn_query: DeviceBuffer<u16>,
    attn_gate: DeviceBuffer<u16>,
    attn_query_normed: DeviceBuffer<u16>,
    attn_key_normed: DeviceBuffer<u16>,
    attn_query_roped: DeviceBuffer<u16>,
    attn_key_roped: DeviceBuffer<u16>,
    attn_out: DeviceBuffer<u16>,
    attn_gated: DeviceBuffer<u16>,
    attn_mixer_out: DeviceBuffer<u16>,

    final_normed: DeviceBuffer<u16>,
}

impl Scratch {
    /// §92: no longer needs `max_seq_len` -- attention now reads the KV
    /// cache directly via `kv_stride` instead of needing a max_seq_len-
    /// sized scratch view copied out of it first.
    pub fn new() -> Result<Self, HipError> {
        Ok(Scratch {
            token_ids_dev: DeviceBuffer::alloc(1)?,
            position_buf: DeviceBuffer::alloc(1)?,
            normed: DeviceBuffer::alloc(HIDDEN_SIZE)?,
            after_mixer: DeviceBuffer::alloc(HIDDEN_SIZE)?,
            normed2: DeviceBuffer::alloc(HIDDEN_SIZE)?,
            gate_up_out: DeviceBuffer::alloc(GATE_UP_COMBINED_DIM)?,
            swiglu_out: DeviceBuffer::alloc(INTERMEDIATE_SIZE)?,
            mlp_out: DeviceBuffer::alloc(HIDDEN_SIZE)?,

            gdn_in_proj_out: DeviceBuffer::alloc(GDN_IN_PROJ_COMBINED_DIM)?,
            gdn_mixed_qkv: DeviceBuffer::alloc(GDN_CONV_DIM)?,
            gdn_g: DeviceBuffer::alloc(GDN_NUM_V_HEADS)?,
            gdn_beta: DeviceBuffer::alloc(GDN_NUM_V_HEADS)?,
            gdn_out: DeviceBuffer::alloc(GDN_VALUE_DIM)?,
            gdn_normed_gated: DeviceBuffer::alloc(GDN_VALUE_DIM)?,
            gdn_mixer_out: DeviceBuffer::alloc(HIDDEN_SIZE)?,

            attn_qkv_out: DeviceBuffer::alloc(ATTN_QKV_COMBINED_DIM)?,
            attn_query: DeviceBuffer::alloc(ATTN_NUM_HEADS * ATTN_HEAD_DIM)?,
            attn_gate: DeviceBuffer::alloc(ATTN_NUM_HEADS * ATTN_HEAD_DIM)?,
            attn_query_normed: DeviceBuffer::alloc(ATTN_NUM_HEADS * ATTN_HEAD_DIM)?,
            attn_key_normed: DeviceBuffer::alloc(ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM)?,
            attn_query_roped: DeviceBuffer::alloc(ATTN_NUM_HEADS * ATTN_HEAD_DIM)?,
            attn_key_roped: DeviceBuffer::alloc(ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM)?,
            attn_out: DeviceBuffer::alloc(ATTN_NUM_HEADS * ATTN_HEAD_DIM)?,
            attn_gated: DeviceBuffer::alloc(ATTN_NUM_HEADS * ATTN_HEAD_DIM)?,
            attn_mixer_out: DeviceBuffer::alloc(HIDDEN_SIZE)?,

            final_normed: DeviceBuffer::alloc(HIDDEN_SIZE)?,
        })
    }
}

/// §96: every intermediate tensor a real BATCHED prefill chunk needs (up to
/// `MAX_PREFILL_CHUNK` new tokens processed together), pre-allocated ONCE --
/// same "no per-call hipMalloc" discipline `Scratch` already established
/// for the rows=1 decode path, sized up for `rows = T` instead of `rows =
/// 1`.
///
/// Every buffer here is TIGHTLY PACKED as `[T, dim]` (row stride == row
/// length) -- required because every existing per-row kernel this project
/// already has (`rmsnorm`, `rope`, `split_last_dim`, `swiglu`, ...) assumes
/// exactly that layout, and was proven correct at that layout for the
/// rows=1 decode case. Where a §92 "combined GEMM" fusion (`qkv_proj`,
/// GDN's `in_proj_combined`, `gate_up_proj`) would otherwise produce a
/// row whose sub-ranges are NOT independently tightly-packed once `T > 1`
/// (see `extract_range.hip`'s own doc comment for why the old rows=1
/// pointer-offset trick stops being equivalent), the `*_raw`/split fields
/// below hold the real, physically-gathered tightly-packed result instead.
pub struct PrefillScratch {
    /// `T` token ids for this chunk, written once per chunk.
    token_ids_dev: DeviceBuffer<i32>,
    /// `T` positions (`start_position + 0..T`), written once per chunk;
    /// every per-token kernel call within a layer reads a single-element
    /// OFFSET into this buffer (`position_buf_at(t)`) rather than a fresh
    /// host->device write per token -- see `forward_prefill_chunk`.
    position_buf: DeviceBuffer<i32>,

    hidden_a: DeviceBuffer<u16>,
    hidden_b: DeviceBuffer<u16>,

    normed: DeviceBuffer<u16>,
    after_mixer: DeviceBuffer<u16>,
    normed2: DeviceBuffer<u16>,
    gate_up_out: DeviceBuffer<u16>,
    swiglu_out: DeviceBuffer<u16>,
    mlp_out: DeviceBuffer<u16>,

    gdn_in_proj_out: DeviceBuffer<u16>,
    gdn_z: DeviceBuffer<u16>,
    gdn_mixed_qkv: DeviceBuffer<u16>,
    gdn_g: DeviceBuffer<f32>,
    gdn_beta: DeviceBuffer<f32>,
    gdn_out: DeviceBuffer<u16>,
    gdn_normed_gated: DeviceBuffer<u16>,
    gdn_mixer_out: DeviceBuffer<u16>,

    /// §100 (chunked GDN parallel prefill): real intermediate buffers for
    /// `gdn_chunk_forward_prefill`, replacing the per-token
    /// `gdn_recurrent_decode` loop. Naming: `gdnc_*`, `_hm` suffix =
    /// head-major `[GDN_NUM_V_HEADS, MAX_GDN_CHUNKS, GDN_CHUNK_SIZE,
    /// GDN_HEAD_DIM]` (the layout `gdn_chunk_utsolve_bf16`'s RHS/output
    /// contract and both per-(head,chunk) GEMMs need); no suffix =
    /// token-major `[MAX_PREFILL_CHUNK, heads, GDN_HEAD_DIM]` (matching
    /// `gdn_g`/`gdn_beta`'s own natural layout). See
    /// `gdn_chunk_forward_prefill`'s own doc comment for the full
    /// pipeline this feeds.
    gdnc_ones: DeviceBuffer<f32>,
    gdnc_query_normed: DeviceBuffer<u16>,
    gdnc_key_normed: DeviceBuffer<u16>,
    gdnc_value_raw: DeviceBuffer<u16>,
    gdnc_k_beta_token: DeviceBuffer<u16>,
    gdnc_query_intra_hm: DeviceBuffer<u16>,
    gdnc_key_bcast_hm: DeviceBuffer<u16>,
    gdnc_v_beta_hm: DeviceBuffer<u16>,
    gdnc_decayed_k_beta_hm: DeviceBuffer<u16>,
    gdnc_query_final_hm: DeviceBuffer<u16>,
    gdnc_key_final_hm: DeviceBuffer<u16>,
    gdnc_ut_system: DeviceBuffer<u16>,
    gdnc_intra_attn: DeviceBuffer<u16>,
    gdnc_decay_exp: DeviceBuffer<f32>,
    gdnc_remaining_decay: DeviceBuffer<f32>,
    gdnc_chunk_decay: DeviceBuffer<f32>,
    gdnc_new_values: DeviceBuffer<u16>,
    gdnc_k_cumdecay: DeviceBuffer<u16>,
    /// bf16 shadow of whichever GDN layer's `recurrent_state` is
    /// currently being processed -- see `gdn_chunk_forward_prefill`'s own
    /// doc comment for why the real fp32 state is cast to bf16 for the
    /// duration of one chunked-prefill call's sequential scan (matching
    /// every GEMM primitive's own bf16-only I/O) and cast back afterward.
    gdnc_state_bf16: DeviceBuffer<u16>,

    attn_qkv_out: DeviceBuffer<u16>,
    attn_v_raw: DeviceBuffer<u16>,
    attn_gate: DeviceBuffer<u16>,
    attn_query_normed: DeviceBuffer<u16>,
    attn_key_normed: DeviceBuffer<u16>,
    attn_query_roped: DeviceBuffer<u16>,
    attn_key_roped: DeviceBuffer<u16>,
    attn_out: DeviceBuffer<u16>,
    attn_gated: DeviceBuffer<u16>,
    attn_mixer_out: DeviceBuffer<u16>,

    /// §99 (batched-prefill causal attention): real `Q@K^T` scores for one
    /// head at a time, `[MAX_PREFILL_CHUNK, max_seq_len]` -- sized for the
    /// worst case (a chunk starting at `position=0` with `max_seq_len`
    /// already cached), reused across every head and every attention
    /// layer within a chunk (one head's attention fully completes --
    /// GEMM, causal softmax, GEMM -- before the next head reuses this same
    /// buffer, so no larger allocation is needed).
    attn_scores: DeviceBuffer<u16>,

    /// Only the LAST row of the chunk's final hidden state is ever needed
    /// (only the next token's logits matter -- see `forward_prefill_chunk`),
    /// so this stays rows=1, reusing the exact same fast GEMV lm_head path
    /// `run_decode_body` already uses.
    final_normed: DeviceBuffer<u16>,
}

impl PrefillScratch {
    pub fn new(max_seq_len: usize) -> Result<Self, HipError> {
        let t = MAX_PREFILL_CHUNK;
        Ok(PrefillScratch {
            token_ids_dev: DeviceBuffer::alloc(t)?,
            position_buf: DeviceBuffer::alloc(t)?,

            hidden_a: DeviceBuffer::alloc(t * HIDDEN_SIZE)?,
            hidden_b: DeviceBuffer::alloc(t * HIDDEN_SIZE)?,

            normed: DeviceBuffer::alloc(t * HIDDEN_SIZE)?,
            after_mixer: DeviceBuffer::alloc(t * HIDDEN_SIZE)?,
            normed2: DeviceBuffer::alloc(t * HIDDEN_SIZE)?,
            gate_up_out: DeviceBuffer::alloc(t * GATE_UP_COMBINED_DIM)?,
            swiglu_out: DeviceBuffer::alloc(t * INTERMEDIATE_SIZE)?,
            mlp_out: DeviceBuffer::alloc(t * HIDDEN_SIZE)?,

            gdn_in_proj_out: DeviceBuffer::alloc(t * GDN_IN_PROJ_COMBINED_DIM)?,
            gdn_z: DeviceBuffer::alloc(t * GDN_VALUE_DIM)?,
            gdn_mixed_qkv: DeviceBuffer::alloc(t * GDN_CONV_DIM)?,
            gdn_g: DeviceBuffer::alloc(t * GDN_NUM_V_HEADS)?,
            gdn_beta: DeviceBuffer::alloc(t * GDN_NUM_V_HEADS)?,
            gdn_out: DeviceBuffer::alloc(t * GDN_VALUE_DIM)?,
            gdn_normed_gated: DeviceBuffer::alloc(t * GDN_VALUE_DIM)?,
            gdn_mixer_out: DeviceBuffer::alloc(t * HIDDEN_SIZE)?,

            gdnc_ones: {
                let mut buf: DeviceBuffer<f32> = DeviceBuffer::alloc(t * GDN_NUM_V_HEADS)?;
                buf.copy_from_host(&vec![1.0f32; t * GDN_NUM_V_HEADS])?;
                buf
            },
            gdnc_query_normed: DeviceBuffer::alloc(t * GDN_KEY_DIM)?,
            gdnc_key_normed: DeviceBuffer::alloc(t * GDN_KEY_DIM)?,
            gdnc_value_raw: DeviceBuffer::alloc(t * GDN_VALUE_DIM)?,
            gdnc_k_beta_token: DeviceBuffer::alloc(t * GDN_VALUE_DIM)?,
            gdnc_query_intra_hm: DeviceBuffer::alloc(GDN_NUM_V_HEADS * MAX_GDN_CHUNKS * GDN_CHUNK_SIZE * GDN_HEAD_DIM)?,
            gdnc_key_bcast_hm: DeviceBuffer::alloc(GDN_NUM_V_HEADS * MAX_GDN_CHUNKS * GDN_CHUNK_SIZE * GDN_HEAD_DIM)?,
            gdnc_v_beta_hm: DeviceBuffer::alloc(GDN_NUM_V_HEADS * MAX_GDN_CHUNKS * GDN_CHUNK_SIZE * GDN_HEAD_DIM)?,
            gdnc_decayed_k_beta_hm: DeviceBuffer::alloc(GDN_NUM_V_HEADS * MAX_GDN_CHUNKS * GDN_CHUNK_SIZE * GDN_HEAD_DIM)?,
            gdnc_query_final_hm: DeviceBuffer::alloc(GDN_NUM_V_HEADS * MAX_GDN_CHUNKS * GDN_CHUNK_SIZE * GDN_HEAD_DIM)?,
            gdnc_key_final_hm: DeviceBuffer::alloc(GDN_NUM_V_HEADS * MAX_GDN_CHUNKS * GDN_CHUNK_SIZE * GDN_HEAD_DIM)?,
            gdnc_ut_system: DeviceBuffer::alloc(GDN_NUM_V_HEADS * MAX_GDN_CHUNKS * GDN_CHUNK_SIZE * GDN_CHUNK_SIZE)?,
            gdnc_intra_attn: DeviceBuffer::alloc(GDN_NUM_V_HEADS * MAX_GDN_CHUNKS * GDN_CHUNK_SIZE * GDN_CHUNK_SIZE)?,
            gdnc_decay_exp: DeviceBuffer::alloc(t * GDN_NUM_V_HEADS)?,
            gdnc_remaining_decay: DeviceBuffer::alloc(t * GDN_NUM_V_HEADS)?,
            gdnc_chunk_decay: DeviceBuffer::alloc(MAX_GDN_CHUNKS * GDN_NUM_V_HEADS)?,
            gdnc_new_values: DeviceBuffer::alloc(GDN_NUM_V_HEADS * MAX_GDN_CHUNKS * GDN_CHUNK_SIZE * GDN_HEAD_DIM)?,
            gdnc_k_cumdecay: DeviceBuffer::alloc(GDN_NUM_V_HEADS * MAX_GDN_CHUNKS * GDN_CHUNK_SIZE * GDN_HEAD_DIM)?,
            gdnc_state_bf16: DeviceBuffer::alloc(GDN_NUM_V_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM)?,

            attn_qkv_out: DeviceBuffer::alloc(t * ATTN_QKV_COMBINED_DIM)?,
            attn_v_raw: DeviceBuffer::alloc(t * ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM)?,
            attn_gate: DeviceBuffer::alloc(t * ATTN_NUM_HEADS * ATTN_HEAD_DIM)?,
            attn_query_normed: DeviceBuffer::alloc(t * ATTN_NUM_HEADS * ATTN_HEAD_DIM)?,
            attn_key_normed: DeviceBuffer::alloc(t * ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM)?,
            attn_query_roped: DeviceBuffer::alloc(t * ATTN_NUM_HEADS * ATTN_HEAD_DIM)?,
            attn_key_roped: DeviceBuffer::alloc(t * ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM)?,
            attn_out: DeviceBuffer::alloc(t * ATTN_NUM_HEADS * ATTN_HEAD_DIM)?,
            attn_gated: DeviceBuffer::alloc(t * ATTN_NUM_HEADS * ATTN_HEAD_DIM)?,
            attn_mixer_out: DeviceBuffer::alloc(t * HIDDEN_SIZE)?,

            attn_scores: DeviceBuffer::alloc(t * max_seq_len)?,

            final_normed: DeviceBuffer::alloc(HIDDEN_SIZE)?,
        })
    }
}

/// All per-layer mutable state for one real decoding sequence (batch=1).
/// `position` is the number of tokens already processed (0 before the
/// first token; the position the NEXT token will be written/read at).
pub struct DecodeState {
    pub layers: Vec<LayerState>,
    pub max_seq_len: usize,
    pub position: usize,
    pub scratch: Scratch,
    /// Ping-pong hidden-state buffers threaded through all 32 layers,
    /// pre-allocated once. Kept as DIRECT siblings of `scratch` (not
    /// fields inside it) deliberately: `forward_one_token` needs to borrow
    /// `&hidden_a`/`&mut hidden_b` (or vice versa) and `&mut scratch` in
    /// the same call -- disjoint top-level struct fields let the borrow
    /// checker see that's safe directly; nesting them inside `Scratch`
    /// would make that same split illegal (a `&mut Scratch` parameter
    /// would alias a `&Scratch`-derived field borrow of the same struct).
    pub hidden_a: DeviceBuffer<u16>,
    pub hidden_b: DeviceBuffer<u16>,

    /// §96: scratch for `forward_prefill_chunk`'s real batched prefill --
    /// a separate allocation from `scratch` (never touched by the rows=1
    /// decode path above), reused across every request the same way
    /// `scratch` already is.
    pub prefill_scratch: PrefillScratch,
}

impl DecodeState {
    pub fn new(max_seq_len: usize) -> Result<Self, HipError> {
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            if is_full_attention_layer(i) {
                let cache_len = ATTN_NUM_KV_HEADS * max_seq_len * ATTN_HEAD_DIM;
                let mut k_cache: DeviceBuffer<u16> = DeviceBuffer::alloc(cache_len)?;
                let mut v_cache: DeviceBuffer<u16> = DeviceBuffer::alloc(cache_len)?;
                let zeros = vec![0u16; cache_len];
                k_cache.copy_from_host(&zeros)?;
                v_cache.copy_from_host(&zeros)?;
                layers.push(LayerState::Attn(AttnLayerState { k_cache, v_cache }));
            } else {
                let state_len = GDN_CONV_KERNEL_SIZE - 1;
                let conv_len = GDN_CONV_DIM * state_len;
                let mut conv_state: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_len)?;
                let zeros_conv = vec![0u16; conv_len];
                conv_state.copy_from_host(&zeros_conv)?;

                let rs_len = GDN_NUM_V_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM;
                let mut recurrent_state: DeviceBuffer<f32> = DeviceBuffer::alloc(rs_len)?;
                let zeros_rs = vec![0f32; rs_len];
                recurrent_state.copy_from_host(&zeros_rs)?;

                layers.push(LayerState::Gdn(GdnLayerState {
                    conv_state,
                    recurrent_state,
                }));
            }
        }
        let scratch = Scratch::new()?;
        let prefill_scratch = PrefillScratch::new(max_seq_len)?;
        Ok(DecodeState {
            layers,
            max_seq_len,
            position: 0,
            scratch,
            hidden_a: DeviceBuffer::alloc(HIDDEN_SIZE)?,
            hidden_b: DeviceBuffer::alloc(HIDDEN_SIZE)?,
            prefill_scratch,
        })
    }

    /// §95: resets this `DecodeState` for a NEW, independent request/
    /// conversation, reusing every buffer's existing address (required --
    /// `GraphedDecodeState`'s captured HIP graph is tied to these exact
    /// addresses; reallocating any of them would invalidate it).
    ///
    /// Deliberately does NOT touch `k_cache`/`v_cache`: `attention_decode`
    /// only ever reads `[0, position]` (derived from `Scratch::position_buf`,
    /// see that field's own doc comment), and a fresh request rewrites
    /// every position it uses via `kv_cache_append` before ever reading it
    /// -- so stale KV data from a prior, unrelated conversation is
    /// structurally unobservable and zeroing it would be real but wasted
    /// device traffic.
    ///
    /// DOES zero GDN's `conv_state`/`recurrent_state`: unlike the KV
    /// caches, these are read-MODIFY-write every call regardless of
    /// `position` (`state *= decay; state += k*delta`, `causal_conv1d_update`'s
    /// own state carry) -- without this, a new conversation would silently
    /// start from the PREVIOUS conversation's leftover recurrent state.
    /// Correctness is proven by `real_reset_prevents_cross_request_state_leakage`,
    /// not just argued here.
    pub fn reset(&mut self) -> Result<(), HipError> {
        for layer in &mut self.layers {
            if let LayerState::Gdn(gdn) = layer {
                gdn.conv_state.fill_zero()?;
                gdn.recurrent_state.fill_zero()?;
            }
        }
        self.position = 0;
        Ok(())
    }
}

/// One GDN (linear_attention) decoder layer's forward pass for a single
/// token, matching `Qwen3_5DecoderLayer.forward`'s `linear_attention`
/// branch + `Qwen3_5GatedDeltaNet.forward`'s single-token cached-decode
/// path exactly: `input_layernorm` -> mixer -> residual add ->
/// `post_attention_layernorm` -> MLP -> residual add. Writes the final
/// result into `hidden_out` (a scratch-owned ping-pong buffer, not a
/// fresh allocation -- see this module's doc comment).
#[allow(clippy::too_many_arguments)]
fn gdn_layer_forward(
    hidden_states: &DeviceBuffer<u16>,
    hidden_out: &mut DeviceBuffer<u16>,
    w: &GdnLayerWeights,
    state: &mut GdnLayerState,
    s: &mut Scratch,
    stream: *mut c_void,
) {
    unsafe {
        raw::rmsnorm(hidden_states.as_device_ptr(), w.input_layernorm.as_device_ptr(), s.normed.as_device_ptr_mut(), 1, HIDDEN_SIZE as i32, RMS_EPS, stream);

        // §92: ONE combined GEMM for in_proj_qkv + in_proj_z + in_proj_b +
        // in_proj_a (was 4 separate GEMM launches). qkv/z/b/a are
        // contiguous offset VIEWS into the single wide output -- no copy,
        // since this model computes at rows=1 (a single decode-step
        // token), so "concatenated along out_features" and "concatenated
        // along the only row" are the same thing. (hipBLASLt was tried
        // here too -- real, correct, but a reproducible REGRESSION for
        // this shape class; see docs/DECISIONS.md §92. Reverted to plain
        // hipblasGemmEx.)
        w.in_proj_combined.apply(s.normed.as_device_ptr(), s.gdn_in_proj_out.as_device_ptr_mut(), HIDDEN_SIZE as i32, GDN_IN_PROJ_COMBINED_DIM as i32, stream);
        let qkv_raw_ptr = s.gdn_in_proj_out.as_device_ptr(); // offset 0, len GDN_CONV_DIM
        let z_ptr = s.gdn_in_proj_out.as_device_ptr_at(GDN_CONV_DIM); // len GDN_VALUE_DIM
        let b_ptr = s.gdn_in_proj_out.as_device_ptr_at(GDN_CONV_DIM + GDN_VALUE_DIM); // len GDN_NUM_V_HEADS
        let a_ptr = s.gdn_in_proj_out.as_device_ptr_at(GDN_CONV_DIM + GDN_VALUE_DIM + GDN_NUM_V_HEADS); // len GDN_NUM_V_HEADS

        // Causal conv1d update (includes SiLU activation), state updated in place.
        raw::causal_conv1d_update(
            qkv_raw_ptr,
            state.conv_state.as_device_ptr_mut(),
            w.conv1d_weight.as_device_ptr(),
            s.gdn_mixed_qkv.as_device_ptr_mut(),
            1,
            GDN_CONV_DIM as i32,
            GDN_CONV_KERNEL_SIZE as i32,
            stream,
        );

        // §92: query/key/value are contiguous offset VIEWS into
        // gdn_mixed_qkv (its own real kernel output, already
        // query|key|value back-to-back) -- was 3 async device-to-device
        // copies into separate buffers, now zero copies.
        let query_ptr = s.gdn_mixed_qkv.as_device_ptr(); // offset 0, len GDN_KEY_DIM
        let key_ptr = s.gdn_mixed_qkv.as_device_ptr_at(GDN_KEY_DIM); // len GDN_KEY_DIM
        let value_ptr = s.gdn_mixed_qkv.as_device_ptr_at(2 * GDN_KEY_DIM); // len GDN_VALUE_DIM

        // b/a -> beta/g (decay).
        raw::gdn_gate_beta(
            a_ptr,
            b_ptr,
            w.a_log.as_device_ptr() as *const f32,
            w.dt_bias.as_device_ptr() as *const f32,
            s.gdn_g.as_device_ptr_mut() as *mut f32,
            s.gdn_beta.as_device_ptr_mut() as *mut f32,
            GDN_NUM_V_HEADS as i32,
            stream,
        );

        // Recurrent delta-rule update (GQA broadcast handled internally:
        // Q/K are num_k_heads=16, V/g/beta/state are num_v_heads=32).
        raw::gdn_recurrent_decode(
            query_ptr,
            key_ptr,
            value_ptr,
            s.gdn_g.as_device_ptr() as *const f32,
            s.gdn_beta.as_device_ptr() as *const f32,
            state.recurrent_state.as_device_ptr_mut() as *mut f32,
            s.gdn_out.as_device_ptr_mut(),
            GDN_NUM_V_HEADS as i32,
            GDN_NUM_K_HEADS as i32,
            GDN_HEAD_DIM as i32,
            stream,
        );

        // z gate + RMSNormGated, applied per-head (num_rows=num_v_heads,
        // hidden_size=head_dim=128 -- NOT the flattened value_dim=4096 as
        // one row, matching Qwen3_5GatedDeltaNet.forward's own
        // `.reshape(-1, head_v_dim)` before calling `self.norm`).
        raw::rmsnorm_gated(
            s.gdn_out.as_device_ptr(),
            z_ptr,
            w.norm_weight.as_device_ptr() as *const f32,
            s.gdn_normed_gated.as_device_ptr_mut(),
            GDN_NUM_V_HEADS as i32,
            GDN_HEAD_DIM as i32,
            RMS_EPS,
            stream,
        );

        // out_proj: [value_dim] -> [hidden]
        w.out_proj.apply(s.gdn_normed_gated.as_device_ptr(), s.gdn_mixer_out.as_device_ptr_mut(), GDN_VALUE_DIM as i32, HIDDEN_SIZE as i32, stream);

        raw::add(hidden_states.as_device_ptr(), s.gdn_mixer_out.as_device_ptr(), s.after_mixer.as_device_ptr_mut(), HIDDEN_SIZE as i32, stream);
    }

    mlp_block(&s.after_mixer.as_device_ptr(), hidden_out, &w.post_attention_layernorm, &w.gate_up_proj, &w.down_proj, s, stream);
}

/// One full-attention decoder layer's forward pass for a single token,
/// matching `Qwen3_5DecoderLayer.forward`'s `full_attention` branch +
/// `Qwen3_5Attention.forward` exactly: `input_layernorm` -> q/k/v proj ->
/// q/k-norm -> RoPE -> KV-cache append -> attention -> sigmoid gate ->
/// o_proj -> residual add -> `post_attention_layernorm` -> MLP -> residual
/// add. Writes the final result into `hidden_out`.
#[allow(clippy::too_many_arguments)]
/// §137: `pub(crate)`, not private, specifically so `mtp_draft.rs` can
/// reuse this real, already-decisively-verified single-token attention
/// forward pass for the MTP draft head's own one real decoder layer
/// (confirmed, against the real checkpoint's own tensor shapes, to be
/// architecturally identical to a real backbone `Attn` layer at this
/// model size -- same `ATTN_NUM_HEADS`/`ATTN_NUM_KV_HEADS`/
/// `ATTN_HEAD_DIM`/`INTERMEDIATE_SIZE`) -- rather than re-deriving the
/// same attention/MLP kernel-call sequence a second time.
pub(crate) fn attn_layer_forward(
    hidden_states: &DeviceBuffer<u16>,
    hidden_out: &mut DeviceBuffer<u16>,
    w: &AttnLayerWeights,
    state: &mut AttnLayerState,
    position_ptr: *const i32,
    max_seq_len: usize,
    s: &mut Scratch,
    stream: *mut c_void,
) {
    unsafe {
        raw::rmsnorm(hidden_states.as_device_ptr(), w.input_layernorm.as_device_ptr(), s.normed.as_device_ptr_mut(), 1, HIDDEN_SIZE as i32, RMS_EPS, stream);

        // §92: ONE combined GEMM for q_proj + k_proj + v_proj (was 3
        // separate GEMM launches). q/k/v are contiguous offset VIEWS into
        // the single wide output.
        w.qkv_proj.apply(s.normed.as_device_ptr(), s.attn_qkv_out.as_device_ptr_mut(), HIDDEN_SIZE as i32, ATTN_QKV_COMBINED_DIM as i32, stream);
        let q_raw_ptr = s.attn_qkv_out.as_device_ptr(); // offset 0, len num_heads*head_dim*2
        let k_raw_ptr = s.attn_qkv_out.as_device_ptr_at(ATTN_NUM_HEADS * ATTN_HEAD_DIM * 2); // len num_kv_heads*head_dim
        let v_ptr = s.attn_qkv_out.as_device_ptr_at(ATTN_NUM_HEADS * ATTN_HEAD_DIM * 2 + ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM); // len num_kv_heads*head_dim

        // q_proj's own output is [num_heads*head_dim*2], chunked per-row
        // into query[num_heads,head_dim] and gate[num_heads,head_dim].
        raw::split_last_dim(q_raw_ptr, s.attn_query.as_device_ptr_mut(), s.attn_gate.as_device_ptr_mut(), ATTN_NUM_HEADS as i32, ATTN_HEAD_DIM as i32, stream);

        raw::rmsnorm(s.attn_query.as_device_ptr(), w.q_norm.as_device_ptr(), s.attn_query_normed.as_device_ptr_mut(), ATTN_NUM_HEADS as i32, ATTN_HEAD_DIM as i32, RMS_EPS, stream);
        raw::rmsnorm(k_raw_ptr, w.k_norm.as_device_ptr(), s.attn_key_normed.as_device_ptr_mut(), ATTN_NUM_KV_HEADS as i32, ATTN_HEAD_DIM as i32, RMS_EPS, stream);

        raw::rope(s.attn_query_normed.as_device_ptr(), s.attn_query_roped.as_device_ptr_mut(), ATTN_NUM_HEADS as i32, ATTN_HEAD_DIM as i32, ATTN_ROTARY_DIM as i32, ATTN_ROPE_THETA, position_ptr, stream);
        raw::rope(s.attn_key_normed.as_device_ptr(), s.attn_key_roped.as_device_ptr_mut(), ATTN_NUM_KV_HEADS as i32, ATTN_HEAD_DIM as i32, ATTN_ROTARY_DIM as i32, ATTN_ROPE_THETA, position_ptr, stream);

        raw::kv_cache_append(
            s.attn_key_roped.as_device_ptr(),
            v_ptr,
            state.k_cache.as_device_ptr_mut(),
            state.v_cache.as_device_ptr_mut(),
            ATTN_NUM_KV_HEADS as i32,
            max_seq_len as i32,
            ATTN_HEAD_DIM as i32,
            position_ptr,
            stream,
        );

        // §92: attention_decode now reads directly out of the real
        // head-major KV cache (kv_stride = max_seq_len) -- no more
        // per-token tightly-packed copy first (used to be 8 async
        // device-to-device copies here, growing with position). §93:
        // `position_ptr` replaces a host-passed `kv_len`. §104: the real,
        // faster split-KV kernel -- see `raw::attention_decode_split`'s
        // own doc comment for the real, measured speedup.
        let scaling = (ATTN_HEAD_DIM as f32).powf(-0.5);
        raw::attention_decode_split(
            s.attn_query_roped.as_device_ptr(),
            state.k_cache.as_device_ptr(),
            state.v_cache.as_device_ptr(),
            s.attn_out.as_device_ptr_mut(),
            ATTN_NUM_HEADS as i32,
            ATTN_NUM_KV_HEADS as i32,
            position_ptr,
            max_seq_len as i32,
            ATTN_HEAD_DIM as i32,
            scaling,
            stream,
        );

        raw::sigmoid_gate(s.attn_out.as_device_ptr(), s.attn_gate.as_device_ptr(), s.attn_gated.as_device_ptr_mut(), (ATTN_NUM_HEADS * ATTN_HEAD_DIM) as i32, stream);

        w.o_proj.apply(s.attn_gated.as_device_ptr(), s.attn_mixer_out.as_device_ptr_mut(), (ATTN_NUM_HEADS * ATTN_HEAD_DIM) as i32, HIDDEN_SIZE as i32, stream);

        raw::add(hidden_states.as_device_ptr(), s.attn_mixer_out.as_device_ptr(), s.after_mixer.as_device_ptr_mut(), HIDDEN_SIZE as i32, stream);
    }

    mlp_block(&s.after_mixer.as_device_ptr(), hidden_out, &w.post_attention_layernorm, &w.gate_up_proj, &w.down_proj, s, stream);
}

/// The MLP block shared identically by both layer types:
/// `post_attention_layernorm -> gate_proj/up_proj -> SwiGLU -> down_proj
/// -> residual add`. `hidden_in_ptr` is a raw pointer (not a borrow) so
/// the caller can pass `&s.after_mixer`'s pointer while still holding
/// `s` mutably for this call -- both `gdn_layer_forward` and
/// `attn_layer_forward` need exactly that (residual = `after_mixer`,
/// itself a field of `s`). §92: `gate_up_proj` is `gate_proj`+`up_proj`
/// fused into one GEMM call -- was two.
#[allow(clippy::too_many_arguments)]
fn mlp_block(
    hidden_in_ptr: &*const c_void,
    hidden_out: &mut DeviceBuffer<u16>,
    post_attention_layernorm: &DeviceBuffer<u16>,
    gate_up_proj: &LinearWeight,
    down_proj: &LinearWeight,
    s: &mut Scratch,
    stream: *mut c_void,
) {
    let hidden_in_ptr = *hidden_in_ptr;
    unsafe {
        raw::rmsnorm(hidden_in_ptr, post_attention_layernorm.as_device_ptr(), s.normed2.as_device_ptr_mut(), 1, HIDDEN_SIZE as i32, RMS_EPS, stream);

        // §92: ONE combined GEMM for gate_proj + up_proj (was 2). gate/up
        // are contiguous offset VIEWS into the single wide output.
        gate_up_proj.apply(s.normed2.as_device_ptr(), s.gate_up_out.as_device_ptr_mut(), HIDDEN_SIZE as i32, GATE_UP_COMBINED_DIM as i32, stream);
        let gate_ptr = s.gate_up_out.as_device_ptr(); // offset 0, len INTERMEDIATE_SIZE
        let up_ptr = s.gate_up_out.as_device_ptr_at(INTERMEDIATE_SIZE); // len INTERMEDIATE_SIZE
        raw::swiglu(gate_ptr, up_ptr, s.swiglu_out.as_device_ptr_mut(), INTERMEDIATE_SIZE as i32, stream);

        down_proj.apply(s.swiglu_out.as_device_ptr(), s.mlp_out.as_device_ptr_mut(), INTERMEDIATE_SIZE as i32, HIDDEN_SIZE as i32, stream);

        raw::add(hidden_in_ptr, s.mlp_out.as_device_ptr(), hidden_out.as_device_ptr_mut(), HIDDEN_SIZE as i32, stream);
    }
}

/// §96: `mlp_block`'s real batched-prefill counterpart -- identical math,
/// `rows = num_tokens` instead of `rows = 1` throughout. No sequential
/// dependency anywhere in the MLP block (unlike attention/GDN's recurrent
/// core), so every op here is a single call over the whole chunk, exactly
/// like `mlp_block` itself just with a bigger `rows`/`n`. The one real
/// difference: `gate_up_proj`'s combined GEMM output can no longer be split
/// via a plain pointer-offset view once `num_tokens > 1` (see
/// `extract_range.hip`), so `extract_range` physically gathers the gate/up
/// sub-ranges into their own tightly-packed `[T, INTERMEDIATE_SIZE]`
/// buffers first.
#[allow(clippy::too_many_arguments)]
fn mlp_block_prefill(
    handle_raw: blas_ffi::HipblasHandle,
    hidden_in_ptr: *const c_void,
    hidden_out: &mut DeviceBuffer<u16>,
    post_attention_layernorm: &DeviceBuffer<u16>,
    gate_up_proj: &LinearWeight,
    down_proj: &LinearWeight,
    num_tokens: usize,
    s: &mut PrefillScratch,
    stream: *mut c_void,
) {
    let t = num_tokens as i32;
    unsafe {
        raw::rmsnorm(hidden_in_ptr, post_attention_layernorm.as_device_ptr(), s.normed2.as_device_ptr_mut(), t, HIDDEN_SIZE as i32, RMS_EPS, stream);

        gate_up_proj.apply_prefill(handle_raw, s.normed2.as_device_ptr(), s.gate_up_out.as_device_ptr_mut(), t, HIDDEN_SIZE as i32, GATE_UP_COMBINED_DIM as i32, stream);
        // §114: one fused launch replaces the real extract_range(gate) +
        // extract_range(up) + swiglu three-launch sequence -- same real
        // reads, same real math, bit-exact identical output (see
        // `kernels::swiglu_strided_bf16`'s own doc comment).
        raw::swiglu_strided(s.gate_up_out.as_device_ptr(), s.swiglu_out.as_device_ptr_mut(), t, GATE_UP_COMBINED_DIM as i32, 0, INTERMEDIATE_SIZE as i32, INTERMEDIATE_SIZE as i32, stream);

        down_proj.apply_prefill(handle_raw, s.swiglu_out.as_device_ptr(), s.mlp_out.as_device_ptr_mut(), t, INTERMEDIATE_SIZE as i32, HIDDEN_SIZE as i32, stream);

        raw::add(hidden_in_ptr, s.mlp_out.as_device_ptr(), hidden_out.as_device_ptr_mut(), (num_tokens * HIDDEN_SIZE) as i32, stream);
    }
}

/// §96/§99: `attn_layer_forward`'s real batched-prefill counterpart -- a
/// chunk of `num_tokens` new prompt tokens processed together. Two real
/// levers, not one: (1) every GEMM (`qkv_proj`, `o_proj`) is one real
/// batched hipBLAS call (`raw::gemm`'s `rows > 1` branch) that reads this
/// layer's weight matrices ONCE for the whole chunk instead of once per
/// token (§96); (2) attention ITSELF is dispatched between the naive
/// per-token scalar kernel and real matrix-core GEMM, on `kv_len`, exactly
/// like `raw::gemm`'s own `rows==1`/`rows>1` split -- NOT an unconditional
/// replacement (§99's first attempt was, and a real end-to-end HTTP
/// benchmark caught the regression it caused at short, realistic prompt
/// lengths before it shipped -- see `ATTENTION_GEMM_KV_LEN_THRESHOLD`'s own
/// doc comment and `docs/DECISIONS.md` §99 for the full real numbers on
/// both sides of the threshold). `rope`/`kv_cache_append` stay real
/// per-token calls regardless of which attention path runs (RoPE's angle
/// depends on each token's absolute position; a token can only attend to
/// KV-cache slots already written, so token t's append must precede any
/// read of it) -- both touch only this layer's small KV-cache slice, cheap
/// regardless of the loop. Positions `start_position..start_position+num_tokens`
/// must already be written into `position_buf` (`s.position_buf`) by the
/// caller -- see `forward_prefill_chunk`.
#[allow(clippy::too_many_arguments)]
fn attn_layer_forward_prefill(
    handle_raw: blas_ffi::HipblasHandle,
    hidden_states: &DeviceBuffer<u16>,
    hidden_out: &mut DeviceBuffer<u16>,
    w: &AttnLayerWeights,
    layer_state: &mut AttnLayerState,
    num_tokens: usize,
    // §136: the real causal key/value range width this layer's attention
    // computes over -- an INPUT, not derived internally, specifically so
    // a graph-capturing caller can pass a FIXED, bucket-determined value
    // instead of the real `start_position + num_tokens` (which varies
    // per real call and cannot be baked into a captured graph's kernel
    // launch shapes -- see `KV_LEN_BUCKETS`'s own doc comment for the
    // full story). The ordinary eager prefill caller
    // (`run_layers_over_chunk`, via `kv_len_override: None`) still
    // computes and passes exactly `start_position + num_tokens` -- this
    // change is a pure refactor for that path, zero behavior change,
    // already covered by every one of this crate's existing real-prefill
    // decisive tests. Any real content beyond the true causal boundary
    // that a LARGER-than-real `kv_len` causes this function to read
    // (stale/uninitialized K/V-cache slots) is always memory-safe (still
    // within the cache's own `max_seq_len`-sized allocation) and always
    // numerically inert: `causal_softmax`'s masking is derived entirely
    // from the separately device-read `start_position_ptr`, never from
    // `kv_len`, so those extra columns are unconditionally zeroed before
    // they can influence anything (verified directly in
    // `causal_softmax.hip`'s own kernel body).
    kv_len: usize,
    max_seq_len: usize,
    s: &mut PrefillScratch,
    stream: *mut c_void,
) {
    let t = num_tokens as i32;
    let q_row_len = ATTN_NUM_HEADS * ATTN_HEAD_DIM; // per-token query length (pre-split from q_raw's 2x-wide query|gate)
    let n_rep = ATTN_NUM_HEADS / ATTN_NUM_KV_HEADS; // GQA: n_rep query heads share each KV head
    unsafe {
        raw::rmsnorm(hidden_states.as_device_ptr(), w.input_layernorm.as_device_ptr(), s.normed.as_device_ptr_mut(), t, HIDDEN_SIZE as i32, RMS_EPS, stream);

        w.qkv_proj.apply_prefill(handle_raw, s.normed.as_device_ptr(), s.attn_qkv_out.as_device_ptr_mut(), t, HIDDEN_SIZE as i32, ATTN_QKV_COMBINED_DIM as i32, stream);

        // §119: Fused attention QKV prep -- replaces 6 separate small kernel launches
        // (3 extract_range, split_last_dim, 2 rmsnorm) with 1 unified launch,
        // eliminating intermediate DRAM round-trips for q_raw, k_raw, query.
        raw::fused_attn_qkv_prep(
            s.attn_qkv_out.as_device_ptr(),
            w.q_norm.as_device_ptr(),
            w.k_norm.as_device_ptr(),
            s.attn_query_normed.as_device_ptr_mut(),
            s.attn_gate.as_device_ptr_mut(),
            s.attn_key_normed.as_device_ptr_mut(),
            s.attn_v_raw.as_device_ptr_mut(),
            t,
            ATTN_NUM_HEADS as i32,
            ATTN_NUM_KV_HEADS as i32,
            ATTN_HEAD_DIM as i32,
            RMS_EPS,
            stream,
        );

        // §103: RoPE and KV-cache append have NO cross-token dependency
        // at all (RoPE's angle depends only on that token's own real
        // absolute position, already known before this runs; a KV-cache
        // append only ever writes to its own token's `position` slot,
        // never read by another token's own append) -- both are now ONE
        // real batched launch each for the whole chunk (`raw::rope_prefill`/
        // `raw::kv_cache_append_prefill`), reading each token's position
        // from `s.position_buf` directly, replacing the OLD per-token
        // loop (`num_tokens` separate launches of each). Real, direct
        // motivation: a comparison against `apps/runtime-llama/llama.cpp`'s
        // own `qwen35.cpp` GDN prefill path found this exact "many small
        // per-token launches" pattern -- GDN's own two per-token loops
        // were fixed first (see `gdn_layer_forward_prefill`'s own doc
        // comment); this is the same fix applied to attention's own
        // remaining per-token ops.
        //
        // §99/§119: attention itself is dispatched on `kv_len`:
        // Below threshold: §119 2D batched causal attention kernel (`raw::attention_causal_prefill`),
        // executing all heads and prompt tokens in ONE GPU launch, replacing the OLD
        // per-token host loop of `attention_decode_split` launches.
        // Above threshold: batched per-head GEMM block below.
        let use_gemm_attention = kv_len > ATTENTION_GEMM_KV_LEN_THRESHOLD;
        let scaling = (ATTN_HEAD_DIM as f32).powf(-0.5);
        let position_buf_ptr = s.position_buf.as_device_ptr() as *const i32;

        raw::rope_prefill(
            s.attn_query_normed.as_device_ptr(),
            s.attn_query_roped.as_device_ptr_mut(),
            t,
            ATTN_NUM_HEADS as i32,
            ATTN_HEAD_DIM as i32,
            ATTN_ROTARY_DIM as i32,
            ATTN_ROPE_THETA,
            position_buf_ptr,
            stream,
        );
        raw::rope_prefill(
            s.attn_key_normed.as_device_ptr(),
            s.attn_key_roped.as_device_ptr_mut(),
            t,
            ATTN_NUM_KV_HEADS as i32,
            ATTN_HEAD_DIM as i32,
            ATTN_ROTARY_DIM as i32,
            ATTN_ROPE_THETA,
            position_buf_ptr,
            stream,
        );
        raw::kv_cache_append_prefill(
            s.attn_key_roped.as_device_ptr(),
            s.attn_v_raw.as_device_ptr(),
            layer_state.k_cache.as_device_ptr_mut(),
            layer_state.v_cache.as_device_ptr_mut(),
            t,
            ATTN_NUM_KV_HEADS as i32,
            max_seq_len as i32,
            ATTN_HEAD_DIM as i32,
            position_buf_ptr,
            stream,
        );

        if !use_gemm_attention {
            // kv_len ≤ ATTENTION_GEMM_KV_LEN_THRESHOLD (128) here.
            // This bounds the causal prefill kernel's dynamic shared memory to
            // ~4 KB per block (head_dim + kv_len + threads + head_dim*kv_split
            // = 128 + 128 + 512 + 256 = 1024 floats = 4 KB), well within
            // RDNA3's 64 KB LDS limit and allowing high occupancy.
            // WARNING: Do NOT raise ATTENTION_GEMM_KV_LEN_THRESHOLD above ~256
            // without verifying LDS pressure and occupancy impact.
            let max_chunk_kv_len = kv_len as i32;
            raw::attention_causal_prefill(
                s.attn_query_roped.as_device_ptr(),
                layer_state.k_cache.as_device_ptr(),
                layer_state.v_cache.as_device_ptr(),
                s.attn_out.as_device_ptr_mut(),
                t,
                ATTN_NUM_HEADS as i32,
                ATTN_NUM_KV_HEADS as i32,
                position_buf_ptr,
                max_seq_len as i32,
                ATTN_HEAD_DIM as i32,
                scaling,
                max_chunk_kv_len,
                stream,
            );
        }

        if use_gemm_attention {
            // §99: real matrix-core causal attention, one head at a time
            // (all `num_tokens` query rows for that head processed in ONE
            // GEMM, not `num_tokens` scalar-loop kernel launches). All T
            // tokens' K/V are now written (the loop above), so the FULL
            // causal window `[0, kv_len)` is available for every query
            // row in this chunk.
            //
            // `S = Q_h @ K_h^T` reads `Q_h` as a STRIDED VIEW directly out
            // of `attn_query_roped` (`q_ld = q_row_len`, no physical
            // per-head extraction needed -- see `blas::gemm_qkt_bf16`'s
            // own doc comment) and `K_h` as the tightly-packed
            // `[kv_len, head_dim]` prefix of this KV head's cache slot
            // (head-major layout, `kv_stride = max_seq_len`, but only the
            // first `kv_len` rows are real/written). `causal_softmax` then
            // masks each query row `i` to its own real valid range
            // `[0, start_position+i]` -- computed on the FULL `S` in one
            // call, not per-row. `O = P @ V_h` writes directly into this
            // head's strided slice of `attn_out` (`o_ld = q_row_len`), no
            // separate per-head output buffer.
            //
            // Scores are scaled by `1/sqrt(head_dim)` via `hipblasGemmEx`'s
            // own `alpha` scalar inside `raw::gemm_qkt` -- no separate
            // elementwise pass over Q needed.
            for h in 0..ATTN_NUM_HEADS {
                let h_kv = h / n_rep;
                let q_ptr = s.attn_query_roped.as_device_ptr_at(h * ATTN_HEAD_DIM);
                let k_ptr = layer_state.k_cache.as_device_ptr_at(h_kv * max_seq_len * ATTN_HEAD_DIM);
                let v_ptr = layer_state.v_cache.as_device_ptr_at(h_kv * max_seq_len * ATTN_HEAD_DIM);
                let scores_ptr = s.attn_scores.as_device_ptr_mut();
                let out_ptr = s.attn_out.as_device_ptr_at_mut(h * ATTN_HEAD_DIM);
                let start_position_ptr = s.position_buf.as_device_ptr_at(0) as *const i32;

                raw::gemm_qkt(handle_raw, q_ptr, q_row_len as i32, k_ptr, scores_ptr, kv_len as i32, t, ATTN_HEAD_DIM as i32, kv_len as i32, scaling);
                raw::causal_softmax(scores_ptr, kv_len as i32, start_position_ptr, t, stream);
                raw::gemm_pv(handle_raw, s.attn_scores.as_device_ptr(), kv_len as i32, v_ptr, out_ptr, q_row_len as i32, t, kv_len as i32, ATTN_HEAD_DIM as i32, 1.0, 0.0);
            }
        }

        raw::sigmoid_gate(s.attn_out.as_device_ptr(), s.attn_gate.as_device_ptr(), s.attn_gated.as_device_ptr_mut(), (num_tokens * q_row_len) as i32, stream);

        w.o_proj.apply_prefill(handle_raw, s.attn_gated.as_device_ptr(), s.attn_mixer_out.as_device_ptr_mut(), t, q_row_len as i32, HIDDEN_SIZE as i32, stream);

        raw::add(hidden_states.as_device_ptr(), s.attn_mixer_out.as_device_ptr(), s.after_mixer.as_device_ptr_mut(), (num_tokens * HIDDEN_SIZE) as i32, stream);
    }

    mlp_block_prefill(handle_raw, s.after_mixer.as_device_ptr(), hidden_out, &w.post_attention_layernorm, &w.gate_up_proj, &w.down_proj, num_tokens, s, stream);
}

/// §100 (chunked GDN parallel prefill): `gdn_recurrent_decode`'s real
/// chunked/parallel replacement for `num_tokens > 1`, implementing the
/// REAL `torch_chunk_gated_delta_rule`
/// (transformers/models/qwen3_5/modeling_qwen3_5.py:300-434) exactly, not
/// a re-derivation -- see `docs/DECISIONS.md` §100 for the full math and
/// the empirical cross-check (`scratchpad/gen_chunked_gdn_reference.py`)
/// this pipeline was built against. Splits the real delta-rule recurrence
/// into an embarrassingly-parallel intra-chunk phase (phase 1: every
/// (head, chunk) block computed independently, real matrix-core GEMMs)
/// and a genuinely sequential inter-chunk phase (phase 2: only
/// `num_chunks` steps, not `num_tokens`).
///
/// Reads `s.gdn_mixed_qkv`/`s.gdn_g`/`s.gdn_beta` (already populated by
/// `gdn_layer_forward_prefill`'s own per-token `causal_conv1d_update`/
/// `gdn_gate_beta` loops -- those two stay genuinely per-token sequential,
/// out of THIS pass's scope, see that function's own doc comment) and
/// writes the real chunked-delta-rule output into `s.gdn_out`
/// (token-major `[num_tokens, GDN_NUM_V_HEADS, GDN_HEAD_DIM]`, exactly
/// what the old per-token loop used to write there), plus the real final
/// `layer_state.recurrent_state` (so decode continues correctly
/// afterward).
///
/// Real GQA broadcast note: `torch_chunk_gated_delta_rule` itself always
/// runs on already-32-head-wide Q/K (the real module-level
/// `repeat_interleave` from 16 k-heads happens BEFORE it's ever called --
/// confirmed directly against `modeling_qwen3_5.py:620-622`). This
/// pipeline folds that broadcast into `gdn_chunk_broadcast_scale_bf16`'s
/// own `n_rep` support instead of physically materializing broadcast
/// Q/K first -- see that kernel's own doc comment for the 5 real
/// broadcast+scale operations this reuses one kernel for.
///
/// KNOWN, DELIBERATE perf/precision tradeoff, now FULLY fixed
/// (§115+§116) on the host-sync side -- real profiling (`model::tests::
/// diagnose_real_quantized_prefill_wall_clock_vs_kernel_time`) found ~58%
/// of real prefill wall-clock time was NOT GPU-kernel time, traced to
/// TWO real host round trips this function used to make per GDN layer,
/// every real prefill call: the whole sequential scan runs in bf16
/// (`layer_state.recurrent_state`'s real fp32 contents cast to a bf16
/// shadow buffer for the scan's duration, then cast back -- §115, now a
/// real bit-exact ON-DEVICE kernel pair, `raw::
/// f32_to_bf16_cast`/`bf16_to_f32_cast`), and each chunk's real
/// `chunk_decay` scalar used to need a HOST `f32` (hipBLAS's `alpha`/
/// `beta` are host pointers under this crate's default
/// `HIPBLAS_POINTER_MODE_HOST`) to feed `gemm_atb`'s own `beta` parameter
/// directly (§116, now a real on-device pre-scale kernel --
/// `scale_by_device_scalar.hip` -- reading `chunk_decay` straight off the
/// already-computed device buffer, followed by `gemm_atb` with a fixed
/// host literal `beta=1.0`, the same convention every other real call in
/// this file already uses). Neither the explicit `hipDeviceSynchronize()`
/// nor either host readback that used to live here remain -- see
/// `docs/DECISIONS.md` §116 for the real before/after TTFT numbers.
///
/// §116's one real, disclosed precision note (checked, not assumed):
/// splitting hipBLAS's fused `beta*Y+alpha*(A@B)` into a separate
/// pre-scale-then-accumulate introduces one extra bf16 rounding step
/// versus the original single-fp32-accumulator-then-round. Verified via
/// a real end-to-end 27B generation A/B, not assumed equivalent --
/// `docs/DECISIONS.md` §116 has the real result.
///
/// STILL NOT addressed: this function's `HIPBLAS_POINTER_MODE_HOST`
/// convention itself is unchanged (every OTHER real scalar here is
/// already a fixed host literal, so this was never a global blocker --
/// §116 only needed to stop `chunk_decay` specifically from being one of
/// the DYNAMIC ones).
#[allow(clippy::too_many_arguments)]
fn gdn_chunk_forward_prefill(handle_raw: blas_ffi::HipblasHandle, layer_state: &mut GdnLayerState, num_tokens: usize, s: &mut PrefillScratch, stream: *mut c_void) {
    let h = GDN_NUM_V_HEADS;
    let kh = GDN_NUM_K_HEADS;
    let d = GDN_HEAD_DIM;
    let c = GDN_CHUNK_SIZE;
    let n_rep = (GDN_NUM_V_HEADS / GDN_NUM_K_HEADS) as i32;
    let num_chunks = num_tokens.div_ceil(GDN_CHUNK_SIZE);
    let t_pad = num_chunks * GDN_CHUNK_SIZE;

    unsafe {
        // Real chunk padding: `t_pad` rounds `num_tokens` UP to a whole
        // number of `GDN_CHUNK_SIZE`-wide chunks -- the trailing
        // `[num_tokens, t_pad)` tail (only present when `num_tokens` isn't
        // already a multiple of `GDN_CHUNK_SIZE`) must be zeroed so those
        // fake positions become pure no-op identity steps (zero
        // key/value/beta, zero log-decay), matching the real reference's
        // own `F.pad(..., 0)` exactly.
        if t_pad > num_tokens {
            s.gdn_mixed_qkv.fill_zero_from_async(num_tokens * GDN_CONV_DIM, stream).expect("fill_zero_from_async(gdn_mixed_qkv) failed");
            s.gdn_g.fill_zero_from_async(num_tokens * GDN_NUM_V_HEADS, stream).expect("fill_zero_from_async(gdn_g) failed");
            s.gdn_beta.fill_zero_from_async(num_tokens * GDN_NUM_V_HEADS, stream).expect("fill_zero_from_async(gdn_beta) failed");
        }

        // Phase 0: un-fuse query/key/value out of `gdn_mixed_qkv`'s wider
        // per-token stride into tightly-packed buffers (`l2norm_bf16`/
        // `gdn_chunk_broadcast_scale_bf16` both assume a tightly-packed
        // token-major source, no row-stride parameter), then L2-normalize
        // query/key IN PLACE (safe: each thread's second-pass write only
        // touches the same index its own first-pass read used). Query
        // also gets the real `1/sqrt(head_dim)` attention-style scaling
        // folded in via `post_scale`.
        raw::extract_range(s.gdn_mixed_qkv.as_device_ptr(), s.gdnc_query_normed.as_device_ptr_mut(), t_pad as i32, GDN_CONV_DIM as i32, 0, GDN_KEY_DIM as i32, stream);
        raw::extract_range(s.gdn_mixed_qkv.as_device_ptr(), s.gdnc_key_normed.as_device_ptr_mut(), t_pad as i32, GDN_CONV_DIM as i32, GDN_KEY_DIM as i32, GDN_KEY_DIM as i32, stream);
        raw::extract_range(s.gdn_mixed_qkv.as_device_ptr(), s.gdnc_value_raw.as_device_ptr_mut(), t_pad as i32, GDN_CONV_DIM as i32, (2 * GDN_KEY_DIM) as i32, GDN_VALUE_DIM as i32, stream);

        let scaling = (d as f32).powf(-0.5);
        raw::l2norm(s.gdnc_query_normed.as_device_ptr(), s.gdnc_query_normed.as_device_ptr_mut(), (t_pad * kh) as i32, d as i32, 1e-6, scaling, stream);
        raw::l2norm(s.gdnc_key_normed.as_device_ptr(), s.gdnc_key_normed.as_device_ptr_mut(), (t_pad * kh) as i32, d as i32, 1e-6, 1.0, stream);

        // Phase 1a: the three real broadcast+scale quantities needed
        // BEFORE decay is known -- undecayed key (head-major, the K
        // operand for both intra-chunk GEMMs below), undecayed
        // query*scaling (head-major, the Q operand for
        // `intra_chunk_attn`), and `k_beta` (kept TOKEN-major -- reused
        // below both as a strided Q operand for `ut_system` and as the
        // broadcast SOURCE for `decayed_k_beta`, which needs a
        // token-major source per `gdn_chunk_broadcast_scale_bf16`'s own
        // contract).
        let ones_ptr = s.gdnc_ones.as_device_ptr() as *const f32;
        raw::gdn_chunk_broadcast_scale(s.gdnc_key_normed.as_device_ptr(), ones_ptr, s.gdnc_key_bcast_hm.as_device_ptr_mut(), t_pad as i32, kh as i32, h as i32, d as i32, n_rep, c as i32, true, stream);
        raw::gdn_chunk_broadcast_scale(s.gdnc_query_normed.as_device_ptr(), ones_ptr, s.gdnc_query_intra_hm.as_device_ptr_mut(), t_pad as i32, kh as i32, h as i32, d as i32, n_rep, c as i32, true, stream);
        let beta_ptr = s.gdn_beta.as_device_ptr() as *const f32;
        raw::gdn_chunk_broadcast_scale(s.gdnc_key_normed.as_device_ptr(), beta_ptr, s.gdnc_k_beta_token.as_device_ptr_mut(), t_pad as i32, kh as i32, h as i32, d as i32, n_rep, c as i32, false, stream);

        // §122: Phase 1b: `ut_system = k_beta @ key^T`, `intra_chunk_attn
        // = query @ key^T` -- real matrix-core GEMMs, now STRIDED
        // BATCHED over all `h` (32) heads at once per chunk (was: one
        // `raw::gemm_qkt` launch per (head, chunk) pair -- 64 launches
        // for a real single-chunk prefill, 1,536 across 24 GDN layers at
        // 27B -- see `docs/DECISIONS.md` §122).
        //
        // Real, found-not-assumed reason `chunk` stays the host loop
        // variable (not `head`, which P1's own Phase 2 batching uses):
        // `gdnc_key_bcast_hm`/`gdnc_query_intra_hm`/`gdnc_ut_system`/
        // `gdnc_intra_attn` are all real head-major-then-chunk `[H,
        // num_chunks, C, *]` buffers, so for a FIXED chunk, striding
        // over `head` gives each a real, uniform per-batch-element
        // stride (`num_chunks*c*d` or `num_chunks*c*c`). But
        // `gdnc_k_beta_token` is real TOKEN-major `[T_pad, h, d]` (kept
        // that way because it's also reused elsewhere as a token-major
        // broadcast source) -- for a FIXED chunk, striding over `head`
        // still gives it a real, uniform stride (`d`), but striding over
        // `chunk` for a fixed head would NOT (its per-chunk stride is
        // `c*(h*d)`, unrelated to `head`'s own `d` stride) -- so batching
        // over head-for-fixed-chunk is the only grouping under which
        // EVERY real operand (Q, K, and the token-major k_beta) gets one
        // consistent stride, which `hipblasGemmStridedBatchedEx` requires.
        for chunk in 0..num_chunks {
            let hm_d_off = chunk * c * d;
            let hm_d_stride = (num_chunks * c * d) as i64;
            let hm_c_off = chunk * c * c;
            let hm_c_stride = (num_chunks * c * c) as i64;
            let k_ptr = s.gdnc_key_bcast_hm.as_device_ptr_at(hm_d_off);

            let kbeta_off = chunk * c * (h * d);
            let kbeta_ptr = s.gdnc_k_beta_token.as_device_ptr_at(kbeta_off);
            let ut_ptr = s.gdnc_ut_system.as_device_ptr_at_mut(hm_c_off);
            raw::gemm_strided_batched_qkt(handle_raw, kbeta_ptr, (h * d) as i32, d as i64, k_ptr, hm_d_stride, ut_ptr, c as i32, hm_c_stride, c as i32, d as i32, c as i32, h as i32, 1.0);

            let q_ptr = s.gdnc_query_intra_hm.as_device_ptr_at(hm_d_off);
            let intra_ptr = s.gdnc_intra_attn.as_device_ptr_at_mut(hm_c_off);
            raw::gemm_strided_batched_qkt(handle_raw, q_ptr, d as i32, hm_d_stride, k_ptr, hm_d_stride, intra_ptr, c as i32, hm_c_stride, c as i32, d as i32, c as i32, h as i32, 1.0);
        }

        // Phase 1c: real per-chunk decay bookkeeping + causal masking of
        // both score matrices above, in ONE launch (covers every (head,
        // chunk) pair internally via its own 2D grid).
        raw::gdn_chunk_decay(
            s.gdn_g.as_device_ptr() as *const f32,
            s.gdnc_ut_system.as_device_ptr_mut(),
            s.gdnc_intra_attn.as_device_ptr_mut(),
            s.gdnc_decay_exp.as_device_ptr_mut() as *mut f32,
            s.gdnc_remaining_decay.as_device_ptr_mut() as *mut f32,
            s.gdnc_chunk_decay.as_device_ptr_mut() as *mut f32,
            h as i32,
            num_chunks as i32,
            c as i32,
            stream,
        );

        // Phase 1d: the remaining decay-scaled, head-major quantities the
        // UT-solve and sequential scan need -- each a single
        // `gdn_chunk_broadcast_scale_bf16` launch (`n_rep=1` for
        // value/k_beta -- already 32-head-wide, a plain per-row scale;
        // `n_rep=2` for query/key -- real GQA broadcast, combined with
        // the decay scale in one pass).
        let decay_exp_ptr = s.gdnc_decay_exp.as_device_ptr() as *const f32;
        let remaining_decay_ptr = s.gdnc_remaining_decay.as_device_ptr() as *const f32;
        raw::gdn_chunk_broadcast_scale(s.gdnc_value_raw.as_device_ptr(), beta_ptr, s.gdnc_v_beta_hm.as_device_ptr_mut(), t_pad as i32, h as i32, h as i32, d as i32, 1, c as i32, true, stream);
        raw::gdn_chunk_broadcast_scale(s.gdnc_k_beta_token.as_device_ptr(), decay_exp_ptr, s.gdnc_decayed_k_beta_hm.as_device_ptr_mut(), t_pad as i32, h as i32, h as i32, d as i32, 1, c as i32, true, stream);
        raw::gdn_chunk_broadcast_scale(s.gdnc_query_normed.as_device_ptr(), decay_exp_ptr, s.gdnc_query_final_hm.as_device_ptr_mut(), t_pad as i32, kh as i32, h as i32, d as i32, n_rep, c as i32, true, stream);
        raw::gdn_chunk_broadcast_scale(s.gdnc_key_normed.as_device_ptr(), remaining_decay_ptr, s.gdnc_key_final_hm.as_device_ptr_mut(), t_pad as i32, kh as i32, h as i32, d as i32, n_rep, c as i32, true, stream);

        // Phase 1e: the real UT-transform solves (`new_values =
        // solve(ut_system, v_beta)`, `k_cumdecay = solve(ut_system,
        // decayed_k_beta)`) -- one launch each, covering every (head,
        // chunk) block internally.
        raw::gdn_chunk_utsolve(s.gdnc_ut_system.as_device_ptr(), s.gdnc_v_beta_hm.as_device_ptr(), s.gdnc_new_values.as_device_ptr_mut(), h as i32, num_chunks as i32, c as i32, d as i32, stream);
        raw::gdn_chunk_utsolve(s.gdnc_ut_system.as_device_ptr(), s.gdnc_decayed_k_beta_hm.as_device_ptr(), s.gdnc_k_cumdecay.as_device_ptr_mut(), h as i32, num_chunks as i32, c as i32, d as i32, stream);
    }

    // §116: real, cheap, non-blocking error check (does NOT sync) -- the
    // real explicit `hipDeviceSynchronize()` plus `chunk_decay` host
    // readback that used to live here are GONE: §116 replaced the one
    // real remaining reason this function needed the host at all (see
    // `scale_by_device_scalar.hip`'s own header) with an on-device
    // pre-scale kernel reading `gdnc_chunk_decay` directly by device
    // pointer. Nothing downstream of Phase 1 reads anything on the HOST
    // anymore.
    hip::check_last_error().expect("gdn_chunk_forward_prefill: phase 1 kernel launch failed");

    // §115: bf16 shadow of the real fp32 recurrent state, for the
    // duration of this call's sequential scan (see this function's own
    // doc comment) -- a real ON-DEVICE cast (same stream, no host round
    // trip). Correctly ordered relative to every Phase 1 kernel purely by
    // HIP's own in-order-per-stream guarantee (same `stream` throughout
    // this whole function) -- no explicit sync needed.
    unsafe {
        raw::f32_to_bf16_cast(layer_state.recurrent_state.as_device_ptr() as *const c_void, s.gdnc_state_bf16.as_device_ptr_mut() as *mut c_void, (h * d * d) as i32, stream);
    }

    // Phase 2: the real sequential scan over `num_chunks` chunks (NOT
    // `num_tokens` tokens -- the actual payoff of this whole pipeline).
    // Per (chunk, head): `v_new = new_values - k_cumdecay @ state`
    // (in-place correction via `alpha=-1.0, beta=1.0`), `out =
    // query_final @ state` (fresh write, the inter-chunk read of the OLD
    // state), `out += intra_chunk_attn @ v_new` (accumulate, `beta=1.0`),
    // then `state = state*chunk_decay + key_final^T @ v_new` (real
    // hipBLAS accumulate via `gemm_atb`'s own `beta`). All four steps for
    // a given (chunk, head) read the state BEFORE this chunk's own update
    // (steps 1-3) and only step 4 writes it -- queued in exactly that
    // order on the same stream, matching hipBLAS's own in-order-per-
    // stream execution guarantee (no explicit sync needed between them).
    unsafe {
        let hm_d_stride = (num_chunks * c * d) as i64;
        let hm_c_stride = (num_chunks * c * c) as i64;
        let state_stride = (d * d) as i64;
        let out_head_stride = d as i64;

        for chunk in 0..num_chunks {
            let hm_d_chunk_off = chunk * c * d;
            let hm_c_chunk_off = chunk * c * c;
            let out_chunk_off = chunk * c * GDN_VALUE_DIM;

            // Step 1: v_new = new_values - k_cumdecay @ state (across all h heads)
            let k_cumdecay_ptr = s.gdnc_k_cumdecay.as_device_ptr_at(hm_d_chunk_off);
            let state_ptr = s.gdnc_state_bf16.as_device_ptr();
            let v_new_ptr = s.gdnc_new_values.as_device_ptr_at_mut(hm_d_chunk_off);
            raw::gemm_strided_batched_pv(
                handle_raw,
                k_cumdecay_ptr, d as i32, hm_d_stride,
                state_ptr, state_stride,
                v_new_ptr, d as i32, hm_d_stride,
                c as i32, d as i32, d as i32,
                h as i32,
                -1.0, 1.0,
            );

            // Step 2: out = query_final @ state (across all h heads)
            let query_final_ptr = s.gdnc_query_final_hm.as_device_ptr_at(hm_d_chunk_off);
            let out_ptr = s.gdn_out.as_device_ptr_at_mut(out_chunk_off);
            raw::gemm_strided_batched_pv(
                handle_raw,
                query_final_ptr, d as i32, hm_d_stride,
                state_ptr, state_stride,
                out_ptr, GDN_VALUE_DIM as i32, out_head_stride,
                c as i32, d as i32, d as i32,
                h as i32,
                1.0, 0.0,
            );

            // Step 3: out += intra_chunk_attn @ v_new (across all h heads)
            let intra_ptr = s.gdnc_intra_attn.as_device_ptr_at(hm_c_chunk_off);
            let v_new_ro_ptr = s.gdnc_new_values.as_device_ptr_at(hm_d_chunk_off);
            let out_ptr2 = s.gdn_out.as_device_ptr_at_mut(out_chunk_off);
            raw::gemm_strided_batched_pv(
                handle_raw,
                intra_ptr, c as i32, hm_c_stride,
                v_new_ro_ptr, hm_d_stride,
                out_ptr2, GDN_VALUE_DIM as i32, out_head_stride,
                c as i32, c as i32, d as i32,
                h as i32,
                1.0, 1.0,
            );

            // Step 4: state = state * chunk_decay (all h heads batched in 1 launch)
            let chunk_decay_ptr = s.gdnc_chunk_decay.as_device_ptr_at(chunk * h);
            raw::scale_bf16_by_device_scalars_batched(
                s.gdnc_state_bf16.as_device_ptr_mut(),
                chunk_decay_ptr as *const c_void,
                (d * d) as i32,
                h as i32,
                stream,
            );

            // Step 5: state += key_final^T @ v_new (across all h heads)
            let key_final_ptr = s.gdnc_key_final_hm.as_device_ptr_at(hm_d_chunk_off);
            let v_new_ro2_ptr = s.gdnc_new_values.as_device_ptr_at(hm_d_chunk_off);
            let state_ptr_mut = s.gdnc_state_bf16.as_device_ptr_mut();
            raw::gemm_strided_batched_atb(
                handle_raw,
                key_final_ptr, d as i32, hm_d_stride,
                v_new_ro2_ptr, d as i32, hm_d_stride,
                state_ptr_mut, d as i32, state_stride,
                c as i32, d as i32, d as i32,
                h as i32,
                1.0,
            );
        }
    }

    // §115: real on-device cast back, replacing the old host round trip
    // (`copy_to_host`+CPU-convert+`copy_from_host`) -- same-stream
    // ordering (HIP's own in-order-per-stream guarantee) is what makes
    // this correctly see Phase 2's real final `gemm_atb` write into
    // `s.gdnc_state_bf16`, no explicit sync needed. The real blocking
    // `hip::device_synchronize()` this used to need is gone with it --
    // nothing downstream of this call reads `layer_state.recurrent_state`
    // from the HOST anymore.
    unsafe {
        raw::bf16_to_f32_cast(s.gdnc_state_bf16.as_device_ptr() as *const c_void, layer_state.recurrent_state.as_device_ptr_mut() as *mut c_void, (h * d * d) as i32, stream);
    }
    hip::check_last_error().expect("gdn_chunk_forward_prefill: phase 2 kernel launch failed");
}

/// §96/§100/§103: `gdn_layer_forward`'s real batched-prefill counterpart.
/// The in_proj/out_proj GEMMs and the gated-RMSNorm batch over the whole
/// chunk in one call each (same reasoning as `attn_layer_forward_prefill`).
///
/// §103 update: `causal_conv1d_update` and `gdn_gate_beta`'s OLD
/// per-token loops (`num_tokens` separate kernel launches each, `~2,592`
/// total for a real 54-token prompt across 24 GDN layers -- confirmed
/// against a real, direct comparison with `apps/runtime-llama/llama.cpp`'s
/// own `qwen35.cpp` GDN prefill path, which issues ONE batched
/// `ggml_ssm_conv` launch instead) are now `raw::causal_conv1d_prefill`/
/// `raw::gdn_gate_beta_prefill`, ONE real launch each for the whole
/// chunk. Neither was ever a genuine unbounded-lookback recurrence like
/// GDN's own delta-rule state (§100's real reason for a chunked
/// UT-transform algorithm): causal conv1d has a BOUNDED lookback
/// (`kernel_size=4`), and `gdn_gate_beta` has no cross-token dependency
/// at all -- both are exact, not approximate, parallel restatements of
/// the same per-token formula (see each new kernel's own doc comment).
/// Both new kernels also read `a`/`b`/conv input directly (strided) out
/// of `gdn_in_proj_out`'s own wide row -- no more `extract_range` calls
/// for `gdn_qkv_raw`/`gdn_a`/`gdn_b` (the real "zero-copy view" technique
/// `qwen35.cpp`'s own `ggml_view_4d` uses). `gdn_z`'s own `extract_range`
/// stays (used once per layer by `rmsnorm_gated` below, not per-token,
/// so not part of the launch-count problem this pass targets).
///
/// The delta-rule recurrence itself, previously a per-token
/// `gdn_recurrent_decode` loop here (see `docs/DECISIONS.md` §96's own
/// note on this), is the real chunked/parallel `gdn_chunk_forward_prefill`
/// (§100) -- see that function's own doc comment for the full pipeline.
#[allow(clippy::too_many_arguments)]
fn gdn_layer_forward_prefill(
    handle_raw: blas_ffi::HipblasHandle,
    hidden_states: &DeviceBuffer<u16>,
    hidden_out: &mut DeviceBuffer<u16>,
    w: &GdnLayerWeights,
    layer_state: &mut GdnLayerState,
    num_tokens: usize,
    s: &mut PrefillScratch,
    stream: *mut c_void,
) {
    let t = num_tokens as i32;
    unsafe {
        raw::rmsnorm(hidden_states.as_device_ptr(), w.input_layernorm.as_device_ptr(), s.normed.as_device_ptr_mut(), t, HIDDEN_SIZE as i32, RMS_EPS, stream);

        w.in_proj_combined.apply_prefill(handle_raw, s.normed.as_device_ptr(), s.gdn_in_proj_out.as_device_ptr_mut(), t, HIDDEN_SIZE as i32, GDN_IN_PROJ_COMBINED_DIM as i32, stream);

        raw::extract_range(s.gdn_in_proj_out.as_device_ptr(), s.gdn_z.as_device_ptr_mut(), t, GDN_IN_PROJ_COMBINED_DIM as i32, GDN_CONV_DIM as i32, GDN_VALUE_DIM as i32, stream);

        // §103: real batched causal conv1d, ONE launch for the whole
        // chunk, reading the real conv input directly (strided, offset 0)
        // out of `gdn_in_proj_out`'s own wide `[T, GDN_IN_PROJ_COMBINED_DIM]`
        // row -- no `gdn_qkv_raw` extraction needed.
        raw::causal_conv1d_prefill(
            s.gdn_in_proj_out.as_device_ptr(),
            GDN_IN_PROJ_COMBINED_DIM as i32,
            0,
            layer_state.conv_state.as_device_ptr_mut(),
            w.conv1d_weight.as_device_ptr(),
            s.gdn_mixed_qkv.as_device_ptr_mut(),
            t,
            GDN_CONV_DIM as i32,
            GDN_CONV_KERNEL_SIZE as i32,
            stream,
        );

        // §103: real batched gate/beta, ONE launch for the whole chunk,
        // reading `a`/`b` directly (strided) out of `gdn_in_proj_out` --
        // no `gdn_a`/`gdn_b` extraction needed.
        raw::gdn_gate_beta_prefill(
            s.gdn_in_proj_out.as_device_ptr(),
            GDN_IN_PROJ_COMBINED_DIM as i32,
            (GDN_CONV_DIM + GDN_VALUE_DIM + GDN_NUM_V_HEADS) as i32,
            s.gdn_in_proj_out.as_device_ptr(),
            GDN_IN_PROJ_COMBINED_DIM as i32,
            (GDN_CONV_DIM + GDN_VALUE_DIM) as i32,
            w.a_log.as_device_ptr() as *const f32,
            w.dt_bias.as_device_ptr() as *const f32,
            s.gdn_g.as_device_ptr_mut() as *mut f32,
            s.gdn_beta.as_device_ptr_mut() as *mut f32,
            t,
            GDN_NUM_V_HEADS as i32,
            stream,
        );
    }

    // §100: the real chunked/parallel delta-rule recurrence, replacing
    // the old per-token `gdn_recurrent_decode` loop -- reads
    // `s.gdn_mixed_qkv`/`s.gdn_g`/`s.gdn_beta` (just populated above) and
    // writes `s.gdn_out` + the real final `layer_state.recurrent_state`.
    // Called OUTSIDE the surrounding `unsafe` block: it manages its own
    // (documented, deliberate) sync points internally.
    gdn_chunk_forward_prefill(handle_raw, layer_state, num_tokens, s, stream);

    unsafe {
        // Per-head gated RMSNorm over ALL (token, head) pairs in ONE call
        // -- same "[T, value_dim] == [T*num_v_heads, head_dim]" trick
        // `attn_layer_forward_prefill` uses for q/k-norm.
        raw::rmsnorm_gated(
            s.gdn_out.as_device_ptr(),
            s.gdn_z.as_device_ptr(),
            w.norm_weight.as_device_ptr() as *const f32,
            s.gdn_normed_gated.as_device_ptr_mut(),
            (num_tokens * GDN_NUM_V_HEADS) as i32,
            GDN_HEAD_DIM as i32,
            RMS_EPS,
            stream,
        );

        w.out_proj.apply_prefill(handle_raw, s.gdn_normed_gated.as_device_ptr(), s.gdn_mixer_out.as_device_ptr_mut(), t, GDN_VALUE_DIM as i32, HIDDEN_SIZE as i32, stream);

        raw::add(hidden_states.as_device_ptr(), s.gdn_mixer_out.as_device_ptr(), s.after_mixer.as_device_ptr_mut(), (num_tokens * HIDDEN_SIZE) as i32, stream);
    }

    mlp_block_prefill(handle_raw, s.after_mixer.as_device_ptr(), hidden_out, &w.post_attention_layernorm, &w.gate_up_proj, &w.down_proj, num_tokens, s, stream);
}

/// §96: embedding lookup for all `num_tokens` new tokens -> all 32 real
/// layers (each layer's real batched-prefill variant above), used by
/// `run_prefill_chunk_body` below. Mirrors `run_decode_body`'s own
/// structure and ping-pong buffer pattern exactly, just against
/// `state.prefill_scratch`'s `[T, HIDDEN_SIZE]` buffers instead of
/// `state.scratch`'s `[1, HIDDEN_SIZE]` ones. Does NOT write
/// `token_ids_dev`/`position_buf` or advance `state.position` -- the
/// caller's job, same division of responsibility `run_decode_body` already
/// documents. Returns a pointer to whichever of `state.prefill_scratch`'s
/// `hidden_a`/`hidden_b` holds the final `[num_tokens, HIDDEN_SIZE]` hidden
/// state.
fn run_layers_over_chunk(
    handle_raw: blas_ffi::HipblasHandle,
    weights: &ModelWeights,
    state: &mut DecodeState,
    num_tokens: usize,
    // §136: `None` for every existing real caller (ordinary eager
    // prefill) -- reproduces today's exact `start_position + num_tokens`
    // derivation, zero behavior change. `Some(bucket_kv_len)` lets a
    // graph-capturing caller force attention's real key/value range
    // width to a FIXED, bucket-determined constant instead of the real
    // (per-call-varying, graph-unsafe) value -- see `KV_LEN_BUCKETS`'s
    // own doc comment and `attn_layer_forward_prefill`'s `kv_len`
    // parameter doc for the full story.
    kv_len_override: Option<usize>,
    stream: *mut c_void,
) -> *const DeviceBuffer<u16> {
    // Stable for the whole layer stack -- `state.position` is only
    // advanced by the caller AFTER this whole function returns (see
    // `forward_prefill_chunk`), so capturing it once here is exactly the
    // "start position of this chunk" every layer needs for real causal
    // attention (§99).
    let start_position = state.position;
    let attn_kv_len = kv_len_override.unwrap_or(start_position + num_tokens);
    let s = &mut state.prefill_scratch;
    unsafe {
        raw::embedding_lookup(
            weights.embed_tokens.as_device_ptr(),
            s.token_ids_dev.as_device_ptr() as *const i32,
            s.hidden_a.as_device_ptr_mut(),
            num_tokens as i32,
            HIDDEN_SIZE as i32,
            stream,
        );
    }

    let mut use_a_as_input = true;
    for (i, layer_weights) in weights.layers.iter().enumerate() {
        let layer_state = &mut state.layers[i];
        // Re-borrow `state.prefill_scratch` fresh each iteration (the
        // `let s = ...` above was consumed by the embedding-lookup call);
        // `hidden_a`/`hidden_b` are fields of `state.prefill_scratch`
        // itself here, unlike `DecodeState`'s decode-path pair which are
        // direct top-level `DecodeState` fields -- so the disjoint-borrow
        // split needed is `&mut state.prefill_scratch.hidden_a` /
        // `&mut state.prefill_scratch.hidden_b` vs. `&mut
        // state.prefill_scratch` (everything else) at once, which the
        // borrow checker cannot see through one shared `&mut
        // PrefillScratch` reference the way `run_decode_body` can for
        // `DecodeState`'s top-level fields. Resolved by taking the two
        // ping-pong buffers out by raw pointer swap before the call instead
        // (safe: both are real, live, distinct allocations for the whole
        // of `state`'s lifetime).
        let (hidden_in_ptr, hidden_out_ptr): (*const DeviceBuffer<u16>, *mut DeviceBuffer<u16>) = if use_a_as_input {
            (&state.prefill_scratch.hidden_a, &mut state.prefill_scratch.hidden_b)
        } else {
            (&state.prefill_scratch.hidden_b, &mut state.prefill_scratch.hidden_a)
        };
        // SAFETY: `hidden_in_ptr`/`hidden_out_ptr` point at
        // `state.prefill_scratch`'s two distinct `hidden_a`/`hidden_b`
        // fields (never the same one on either side of the branch above),
        // both live for all of `state`'s lifetime; `layer_state` borrows a
        // disjoint field of `state` (`state.layers[i]`), and
        // `state.prefill_scratch` itself (borrowed `&mut` inside
        // `attn_layer_forward_prefill`/`gdn_layer_forward_prefill` below)
        // is likewise disjoint from `state.layers`. No aliasing occurs.
        let hidden_in: &DeviceBuffer<u16> = unsafe { &*hidden_in_ptr };
        let hidden_out: &mut DeviceBuffer<u16> = unsafe { &mut *hidden_out_ptr };
        match (layer_weights, layer_state) {
            (LayerWeights::Gdn(w), LayerState::Gdn(gs)) => {
                gdn_layer_forward_prefill(handle_raw, hidden_in, hidden_out, w, gs, num_tokens, &mut state.prefill_scratch, stream)
            }
            (LayerWeights::Attn(w), LayerState::Attn(as_)) => attn_layer_forward_prefill(
                handle_raw,
                hidden_in,
                hidden_out,
                w,
                as_,
                num_tokens,
                attn_kv_len,
                state.max_seq_len,
                &mut state.prefill_scratch,
                stream,
            ),
            _ => unreachable!("layer weights/state type mismatch at index {i}"),
        }
        use_a_as_input = !use_a_as_input;
    }
    if use_a_as_input { &state.prefill_scratch.hidden_a } else { &state.prefill_scratch.hidden_b }
}

/// §96: prefill's own final projection -- only the LAST token's hidden
/// state feeds final_norm + lm_head, a real, deliberate optimization (not
/// a correctness shortcut): earlier positions' logits are never used by
/// anything during prefill (no echo/logprobs support in this server), and
/// `VOCAB_SIZE` (248320) makes lm_head by far the single most expensive
/// GEMM in the whole stack, so computing it `num_tokens` times here
/// instead of once would be real, wasted GPU time -- exactly the kind of
/// per-token weight-read repetition this whole feature exists to
/// eliminate elsewhere.
fn run_prefill_chunk_body(
    handle_raw: blas_ffi::HipblasHandle,
    weights: &ModelWeights,
    state: &mut DecodeState,
    num_tokens: usize,
    logits_out: &mut DeviceBuffer<u16>,
    stream: *mut c_void,
) {
    let final_hidden_ptr = run_layers_over_chunk(handle_raw, weights, state, num_tokens, None, stream);
    // SAFETY: points at one of `state.prefill_scratch`'s two live
    // `hidden_a`/`hidden_b` fields (`run_layers_over_chunk`'s own
    // guarantee).
    let final_hidden: &DeviceBuffer<u16> = unsafe { &*final_hidden_ptr };

    let s = &mut state.prefill_scratch;
    let last_row_ptr = final_hidden.as_device_ptr_at((num_tokens - 1) * HIDDEN_SIZE);
    unsafe {
        raw::rmsnorm(last_row_ptr, weights.final_norm.as_device_ptr(), s.final_normed.as_device_ptr_mut(), 1, HIDDEN_SIZE as i32, RMS_EPS, stream);
        weights.lm_head_apply(s.final_normed.as_device_ptr(), logits_out.as_device_ptr_mut(), stream);
    }
}

/// §96: runs one real batched prefill chunk of up to `MAX_PREFILL_CHUNK`
/// new prompt tokens, advancing `state.position` by `token_ids.len()` on
/// success. Writes real vocab logits for ONLY the last token in
/// `logits_out` -- the caller argmax-samples that to get the first
/// generated token, exactly as if every one of these tokens had gone
/// through `forward_one_token` sequentially (this function does not change
/// what gets computed, only how much of it is batched into fewer, larger
/// GPU calls -- see `real_batched_prefill_matches_sequential_forward_one_token`
/// for the decisive proof this produces bit-for-bit the same KV-cache/
/// GDN-state/logits as the original sequential path).
pub fn forward_prefill_chunk(
    handle: &BlasHandle,
    weights: &ModelWeights,
    state: &mut DecodeState,
    token_ids: &[i32],
    logits_out: &mut DeviceBuffer<u16>,
) -> Result<(), HipError> {
    let num_tokens = token_ids.len();
    assert!(num_tokens > 0, "forward_prefill_chunk: token_ids must be non-empty");
    assert!(
        num_tokens <= MAX_PREFILL_CHUNK,
        "forward_prefill_chunk: {num_tokens} tokens exceeds MAX_PREFILL_CHUNK ({MAX_PREFILL_CHUNK}); caller must chunk (see forward_prefill)"
    );
    assert!(
        state.position + num_tokens <= state.max_seq_len,
        "forward_prefill_chunk: position ({}) + num_tokens ({}) exceeds max_seq_len ({})",
        state.position,
        num_tokens,
        state.max_seq_len
    );
    let handle_raw = handle.raw();

    state.prefill_scratch.token_ids_dev.copy_from_host_prefix(token_ids)?;
    let positions: Vec<i32> = (0..num_tokens as i32).map(|i| state.position as i32 + i).collect();
    state.prefill_scratch.position_buf.copy_from_host_prefix(&positions)?;

    run_prefill_chunk_body(handle_raw, weights, state, num_tokens, logits_out, std::ptr::null_mut());

    hip::check_last_error()?;
    hip::device_synchronize()?;

    state.position += num_tokens;
    Ok(())
}

/// §96: processes a full real prompt (arbitrarily long, up to
/// `state.max_seq_len - state.position`), chunking it into
/// `MAX_PREFILL_CHUNK`-sized real batched prefill calls. Real vocab logits
/// for the prompt's LAST token end up in `logits_out` after the final
/// chunk -- the caller argmax-samples that exactly as `forward_prefill_chunk`
/// documents. For any real prompt this crate's own benchmark sends (well
/// under `MAX_PREFILL_CHUNK`), this is exactly ONE real batched call instead
/// of `token_ids.len()` sequential `forward_one_token` calls.
pub fn forward_prefill(
    handle: &BlasHandle,
    weights: &ModelWeights,
    state: &mut DecodeState,
    token_ids: &[i32],
    logits_out: &mut DeviceBuffer<u16>,
) -> Result<(), HipError> {
    assert!(!token_ids.is_empty(), "forward_prefill: token_ids must be non-empty");
    assert!(
        state.position + token_ids.len() <= state.max_seq_len,
        "forward_prefill: total tokens ({}) exceeds max_seq_len ({})",
        state.position + token_ids.len(),
        state.max_seq_len
    );
    for chunk in token_ids.chunks(MAX_PREFILL_CHUNK) {
        forward_prefill_chunk(handle, weights, state, chunk, logits_out)?;
    }
    Ok(())
}

/// The actual decode-step body: embedding lookup -> all 32 real layers ->
/// final norm -> lm_head (tied to `embed_tokens`) -> real vocab logits.
/// Extracted from `forward_one_token` (§93) so it can be reused UNCHANGED
/// by a future HIP-Graph-captured driver -- capture just records whatever
/// GPU operations this function issues onto `stream`, so the eager path
/// (`stream = null`) and a graphed path (`stream` = a real captured
/// stream) share exactly one implementation, never two that could drift
/// apart. Does NOT write `token_ids_dev`/`position_buf` (the caller must
/// have already done that via `copy_from_host`, which is a synchronous,
/// non-capturable host operation -- see `Scratch::position_buf`'s own doc
/// comment) and does NOT synchronize or advance `state.position` (also the
/// caller's job, since a graphed replay's position advance happens
/// on-device instead, inside the captured region itself).
fn run_decode_body(
    weights: &ModelWeights,
    state: &mut DecodeState,
    position_ptr: *const i32,
    logits_out: &mut DeviceBuffer<u16>,
    stream: *mut c_void,
) {
    // Ping-pong hidden-state buffers (`state.hidden_a`/`state.hidden_b`,
    // pre-allocated in `DecodeState::new` -- zero allocation in this
    // function). `hidden_a` holds the embedding lookup's output first;
    // each layer alternates which of the two it reads from and writes to.
    unsafe {
        raw::embedding_lookup(
            weights.embed_tokens.as_device_ptr(),
            state.scratch.token_ids_dev.as_device_ptr() as *const i32,
            state.hidden_a.as_device_ptr_mut(),
            1,
            HIDDEN_SIZE as i32,
            stream,
        );
    }

    let mut use_a_as_input = true;
    for (i, layer_weights) in weights.layers.iter().enumerate() {
        let layer_state = &mut state.layers[i];
        let (hidden_in, hidden_out): (&DeviceBuffer<u16>, &mut DeviceBuffer<u16>) = if use_a_as_input {
            (&state.hidden_a, &mut state.hidden_b)
        } else {
            // Disjoint field borrow, not aliasing: `hidden_a`/`hidden_b`
            // are two distinct top-level fields of `state`.
            (&state.hidden_b, &mut state.hidden_a)
        };
        match (layer_weights, layer_state) {
            (LayerWeights::Gdn(w), LayerState::Gdn(gs)) => {
                gdn_layer_forward(hidden_in, hidden_out, w, gs, &mut state.scratch, stream)
            }
            (LayerWeights::Attn(w), LayerState::Attn(as_)) => attn_layer_forward(
                hidden_in,
                hidden_out,
                w,
                as_,
                position_ptr,
                state.max_seq_len,
                &mut state.scratch,
                stream,
            ),
            _ => unreachable!("layer weights/state type mismatch at index {i}"),
        }
        use_a_as_input = !use_a_as_input;
    }
    let final_hidden = if use_a_as_input { &state.hidden_a } else { &state.hidden_b };

    unsafe {
        raw::rmsnorm(
            final_hidden.as_device_ptr(),
            weights.final_norm.as_device_ptr(),
            state.scratch.final_normed.as_device_ptr_mut(),
            1,
            HIDDEN_SIZE as i32,
            RMS_EPS,
            stream,
        );
        weights.lm_head_apply(state.scratch.final_normed.as_device_ptr(), logits_out.as_device_ptr_mut(), stream);
    }
}

/// Runs one full, real decode step on the default/null stream (the
/// original, decisively-verified eager path -- unchanged behavior from
/// before §93's graph-capture work, since `stream = null` reproduces
/// exactly what every kernel launch already did). Advances `state.position`
/// by one on success. Synchronizes with the GPU exactly once (after the
/// layer loop) -- see this module's doc comment.
/// §137: whether `state.hidden_a` (vs `hidden_b`) holds the real,
/// PRE-final-norm final hidden state after a real `forward_one_token`
/// call -- the exact real input the MTP draft head needs (`out.
/// hidden_states[-1]` in HF terms, confirmed against the real production
/// Python integration, `apps/runtime-ipwf/bucketed_speculative.py`'s own
/// `out.hidden_states[-1]`, NOT the post-final-norm value `lm_head`
/// consumes). A real, DETERMINISTIC, compile-time fact, not runtime
/// state: `run_decode_body`'s own `use_a_as_input` ping-pong starts
/// `true` and flips exactly once per layer, so after a FIXED
/// `NUM_LAYERS` it always lands on the SAME buffer, every call, for a
/// given model-size build -- never alternates between calls the way a
/// naive reader might assume.
const FINAL_HIDDEN_IS_A: bool = NUM_LAYERS % 2 == 0;

impl DecodeState {
    /// The real, current pre-final-norm final hidden state -- see
    /// `FINAL_HIDDEN_IS_A`'s own doc comment for why this is a plain,
    /// deterministic buffer pick, not tracked runtime state. Valid
    /// immediately after any real `forward_one_token`/`GraphedDecodeState::
    /// forward_one_token` call.
    pub fn final_hidden(&self) -> &DeviceBuffer<u16> {
        if FINAL_HIDDEN_IS_A { &self.hidden_a } else { &self.hidden_b }
    }
}

pub fn forward_one_token(
    _handle: &BlasHandle,
    weights: &ModelWeights,
    state: &mut DecodeState,
    token_id: i32,
    logits_out: &mut DeviceBuffer<u16>,
) -> Result<(), HipError> {
    let token_ids_host = [token_id];
    state.scratch.token_ids_dev.copy_from_host(&token_ids_host)?;
    // §93: written once per token, read by rope/kv_cache_append/
    // attention_decode through a device pointer (see `Scratch::position_buf`'s
    // own doc comment for why).
    state
        .scratch
        .position_buf
        .copy_from_host(&[state.position as i32])?;
    let position_ptr = state.scratch.position_buf.as_device_ptr() as *const i32;

    run_decode_body(weights, state, position_ptr, logits_out, std::ptr::null_mut());

    // The ONE sync point per token: everything above was queued on the
    // default stream without the host waiting between calls.
    hip::check_last_error()?;
    hip::device_synchronize()?;

    state.position += 1;
    Ok(())
}

/// §93: a HIP-Graph-captured decode step. Same real per-layer math as
/// `forward_one_token` (both call `run_decode_body`), replayed via
/// `hipGraphLaunch` after the first call instead of re-issuing ~480
/// individual kernel/GEMM launches every token -- see `docs/DECISIONS.md`
/// for the full real-benchmark comparison against the eager path.
pub struct GraphedDecodeState {
    stream: hip::Stream,
    _handle: BlasHandle,
    graph_exec: Option<hip::GraphExec>,
}

impl GraphedDecodeState {
    pub fn new() -> Result<Self, HipError> {
        let stream = hip::Stream::create()?;
        let handle = BlasHandle::create().map_err(|_| HipError { code: -1 })?;
        handle.set_stream(&stream).map_err(|_| HipError { code: -1 })?;
        Ok(GraphedDecodeState {
            stream,
            _handle: handle,
            graph_exec: None,
        })
    }

    /// Runs one decode step. The FIRST call captures `run_decode_body`
    /// (plus the on-device position increment) into a graph and launches
    /// it; every call after that replays the SAME graph against whatever
    /// `token_ids_dev`/`position_buf` hold NOW (written fresh, below,
    /// before every launch -- these two small host->device writes are the
    /// only host-side work left in the per-token hot path besides the
    /// final synchronize and the caller's own argmax).
    pub fn forward_one_token(
        &mut self,
        weights: &ModelWeights,
        state: &mut DecodeState,
        token_id: i32,
        logits_out: &mut DeviceBuffer<u16>,
    ) -> Result<(), HipError> {
        state.scratch.token_ids_dev.copy_from_host(&[token_id])?;
        state
            .scratch
            .position_buf
            .copy_from_host(&[state.position as i32])?;
        let position_ptr = state.scratch.position_buf.as_device_ptr() as *const i32;

        if self.graph_exec.is_none() {
            hip::begin_capture(&self.stream)?;
            run_decode_body(weights, state, position_ptr, logits_out, self.stream.raw());
            // On-device position advance -- the last captured node, so a
            // replay needs no host-side position write to stay correct
            // (see `Scratch::position_buf`'s doc comment).
            unsafe {
                raw::increment_position(state.scratch.position_buf.as_device_ptr_mut() as *mut i32, self.stream.raw());
            }
            self.graph_exec = Some(hip::end_capture(&self.stream)?);
        }
        self.graph_exec.as_ref().unwrap().launch(&self.stream)?;
        self.stream.synchronize()?;

        state.position += 1;
        Ok(())
    }
}

/// §136: real, FIXED key/value range width buckets for a graph-captured
/// speculative-decode verify chunk. The real problem this exists to
/// solve: `attn_layer_forward_prefill`'s attention math needs a `kv_len`
/// (the causal key/value range width) that naturally grows with
/// `state.position` over a real generation's lifetime -- but a captured
/// HIP graph's kernel launches have their shapes fixed FOREVER at capture
/// time, and (confirmed directly, not assumed) `hipblasGemmEx`'s own C
/// API has no device-pointer variant for its shape arguments at all, so
/// `kv_len` genuinely cannot be made "just read from a device pointer"
/// the way `position_buf` already is for RoPE/KV-cache-append. The fix:
/// pad every real call up to a small, discrete set of FIXED widths (this
/// array), and rely on `causal_softmax`'s masking -- verified directly in
/// `causal_softmax.hip`'s own kernel body to be derived ENTIRELY from a
/// separately device-read real position, never from `kv_len` -- to zero
/// out whatever stale/uninitialized K/V-cache content a bucket's padding
/// reads beyond the real causal boundary. This is the same real strategy
/// Python's own proven `bucketed_speculative.py` design uses
/// (`docs/DECISIONS.md`'s §136 research: "the resolution is that
/// speculative chunk widths are discrete and tightly bounded... capture
/// one graph per width"), applied here to `kv_len` instead of chunk
/// width, and the same "fixed shape + real masking" idiom this crate's
/// own `GDN_CHUNK_SIZE` padding already uses elsewhere.
///
/// Doubling schedule, capped at the real production `MAX_SEQ_LEN`
/// (`server.rs`, 12288) -- keeps real wasted attention compute bounded to
/// at most ~2x the real content at any position, the same real tradeoff
/// this crate's own `docs/DECISIONS.md` §117 postmortem found could go
/// badly wrong if bucket granularity doesn't match the real workload (that
/// failure was prefill bucketed by PROMPT length against a mostly-short
/// real distribution; this is decode-side, bucketed by the real, always-
/// growing `state.position`, a structurally different distribution -- but
/// the SAME lesson applies: never assume a bucket schedule is safe
/// without a real measurement, see this section's own future benchmark
/// work before this ships).
pub const KV_LEN_BUCKETS: [usize; 8] = [128, 256, 512, 1024, 2048, 4096, 8192, 12288];

/// Smallest real bucket that can hold `real_kv_len` real positions.
/// Panics if `real_kv_len` exceeds the largest bucket (the real
/// `MAX_SEQ_LEN` itself) -- a real caller has already lost by then (the
/// server's own `forward_prefill`/`forward_prefill_chunk` already refuse
/// a request whose real length would exceed `MAX_SEQ_LEN`), not a
/// condition this function should paper over.
pub fn kv_len_bucket_for(real_kv_len: usize) -> usize {
    KV_LEN_BUCKETS
        .iter()
        .copied()
        .find(|&b| b >= real_kv_len)
        .unwrap_or_else(|| panic!("real_kv_len={real_kv_len} exceeds the largest real KV_LEN_BUCKETS entry ({}) -- caller should have already refused this request", KV_LEN_BUCKETS[KV_LEN_BUCKETS.len() - 1]))
}

/// §136: a `GraphedDecodeState`-style wrapper around a FIXED-K
/// speculative-decode verify chunk, bucketed by real `kv_len` (see
/// `KV_LEN_BUCKETS`) so it stays graph-safe across a real generation's
/// entire lifetime, not just at whatever position the graph happened to
/// first capture at (the real, confirmed bug this replaces -- see
/// `docs/DECISIONS.md` §136 for the decisive test that found a captured
/// verify-chunk graph diverged 100% on a replay after a real position
/// change, root-caused to `kv_len` being baked into kernel launch
/// arguments and a host-side kernel-selection branch, neither re-read on
/// replay).
///
/// One real captured `GraphExec` per bucket actually seen so far (lazy:
/// only the buckets a real generation actually reaches get captured,
/// never all `KV_LEN_BUCKETS::len()` up front). Does NOT advance
/// `state.position` -- same real, deliberate division of responsibility
/// `GraphedVerifyState`'s original (unbucketed) spike already established:
/// a verify round's real advance amount depends on how many drafts the
/// caller decides to accept AFTER seeing this call's logits, which has to
/// stay entirely host-side.
pub struct GraphedVerifyState {
    stream: hip::Stream,
    handle: BlasHandle,
    graph_execs: HashMap<usize, hip::GraphExec>,
    k: usize,
}

impl GraphedVerifyState {
    pub fn new(k: usize) -> Result<Self, HipError> {
        let stream = hip::Stream::create()?;
        let handle = BlasHandle::create().map_err(|_| HipError { code: -1 })?;
        handle.set_stream(&stream).map_err(|_| HipError { code: -1 })?;
        Ok(GraphedVerifyState { stream, handle, graph_execs: HashMap::new(), k })
    }

    /// Runs one K-token verify chunk from `state.position`, real
    /// per-position logits written into `verify_logits`
    /// (`[k * VOCAB_SIZE]`). Real attention computes over
    /// `kv_len_bucket_for(state.position + k)` positions (padded, masked
    /// safe per this struct's own doc comment), NOT the real
    /// `state.position + k` directly -- the one real difference from the
    /// original unbucketed spike, and the actual fix. Captures a NEW
    /// graph the first time a given bucket is seen; every later call
    /// landing in an ALREADY-seen bucket replays that bucket's existing
    /// graph. Does NOT advance `state.position` -- see this struct's own
    /// doc comment.
    pub fn verify_chunk(
        &mut self,
        weights: &ModelWeights,
        state: &mut DecodeState,
        draft_ids: &[i32],
        normed_out: &mut DeviceBuffer<u16>,
        verify_logits: &mut DeviceBuffer<u16>,
        raw_hidden_out: &mut DeviceBuffer<u16>,
    ) -> Result<(), HipError> {
        assert_eq!(draft_ids.len(), self.k, "GraphedVerifyState: draft_ids length must match the fixed K this graph was built for");
        state.prefill_scratch.token_ids_dev.copy_from_host_prefix(draft_ids)?;
        let positions: Vec<i32> = (0..self.k as i32).map(|i| state.position as i32 + i).collect();
        state.prefill_scratch.position_buf.copy_from_host_prefix(&positions)?;

        let real_kv_len = state.position + self.k;
        let bucket = kv_len_bucket_for(real_kv_len);

        let handle_raw = self.handle.raw();
        if !self.graph_execs.contains_key(&bucket) {
            hip::begin_capture(&self.stream)?;
            let final_hidden_ptr = run_layers_over_chunk(handle_raw, weights, state, self.k, Some(bucket), self.stream.raw());
            let final_hidden: &DeviceBuffer<u16> = unsafe { &*final_hidden_ptr };
            // §137: preserve the REAL, PRE-final-norm per-position hidden
            // state (the exact real input the MTP draft head needs --
            // `out.hidden_states[-1]` in HF terms, confirmed against the
            // real production Python integration) into a caller-owned
            // buffer, as part of THIS SAME captured, unconditional
            // sequence -- `final_hidden_ptr` points into this struct's
            // own internal ping-pong scratch, which the NEXT real call
            // (a different bucket, or even this same one on a later
            // replay) will overwrite, so it cannot be read AFTER
            // `launch()` returns without a real copy captured here.
            raw_hidden_out.copy_from_device_async_prefix(final_hidden, self.k * HIDDEN_SIZE, self.stream.raw())?;
            unsafe {
                raw::rmsnorm(
                    final_hidden.as_device_ptr(),
                    weights.final_norm.as_device_ptr(),
                    normed_out.as_device_ptr_mut(),
                    self.k as i32,
                    HIDDEN_SIZE as i32,
                    RMS_EPS,
                    self.stream.raw(),
                );
                match &weights.lm_head {
                    Some(LinearWeight::Bf16(buf)) => {
                        raw::gemm(handle_raw, normed_out.as_device_ptr(), buf.as_device_ptr(), verify_logits.as_device_ptr_mut(), self.k as i32, HIDDEN_SIZE as i32, VOCAB_SIZE as i32, self.stream.raw());
                    }
                    None => {
                        raw::gemm(handle_raw, normed_out.as_device_ptr(), weights.embed_tokens.as_device_ptr(), verify_logits.as_device_ptr_mut(), self.k as i32, HIDDEN_SIZE as i32, VOCAB_SIZE as i32, self.stream.raw());
                    }
                    Some(LinearWeight::Quantized { .. }) => panic!("scoped to bf16 dense sizes only"),
                }
            }
            self.graph_execs.insert(bucket, hip::end_capture(&self.stream)?);
        }
        self.graph_execs.get(&bucket).unwrap().launch(&self.stream)?;
        self.stream.synchronize()?;
        Ok(())
    }
}

/// Greedy sampling: argmax over real vocab logits. §93: computed on-device
/// (`kernels::argmax_bf16`) instead of copying the full `VOCAB_SIZE`-
/// element logits buffer to host and scanning it there -- real, measured
/// cost of the old approach was ~0.209ms/token, on top of the actual
/// forward-pass time. Same external signature and exact tie-breaking
/// semantics (first/lowest-index occurrence wins) as before.
pub fn argmax_sample(logits: &DeviceBuffer<u16>) -> Result<i32, HipError> {
    crate::kernels::argmax_bf16(logits)
}

/// §137: greedy sampling for one row of a `[num_rows, VOCAB_SIZE]` logits
/// buffer (a real speculative-decode verify chunk's per-position output).
pub fn argmax_sample_row(logits: &DeviceBuffer<u16>, row: usize) -> Result<i32, HipError> {
    crate::kernels::argmax_bf16_row(logits, row, VOCAB_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hip;
    use crate::model_loader::locate_model_snapshot;

    /// Shared setup for the §100 chunked-GDN decisive tests below: builds
    /// the exact same deterministic `synth()` data
    /// `gen_chunked_gdn_reference.py` uses, writes it into a fresh
    /// `PrefillScratch`/`GdnLayerState` in `gdn_mixed_qkv`'s own real
    /// per-token `[query(2048)|key(2048)|value(4096)]` layout (exactly
    /// what `gdn_layer_forward_prefill`'s `causal_conv1d_update` loop
    /// would have produced), runs `gdn_chunk_forward_prefill`, and
    /// returns `(out, final_recurrent_state)` as host `f32` for the
    /// caller to check against independently-captured reference values.
    /// The real GQA broadcast (16 k-heads -> 32 v-heads) happens INSIDE
    /// the pipeline under test, not pre-applied here.
    fn run_chunked_gdn_case(seq_len: usize) -> (Vec<f32>, Vec<f32>) {
        let kh = GDN_NUM_K_HEADS;
        let h = GDN_NUM_V_HEADS;
        let d = GDN_HEAD_DIM;

        let synth = |n: usize, offset: i32, period: i32, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let m = ((i as i32 + offset) % period) as f32;
                    (m - (period as f32) / 2.0) * scale
                })
                .collect()
        };
        let q = synth(seq_len * kh * d, 0, 13, 0.05);
        let k = synth(seq_len * kh * d, 3, 11, 0.04);
        let v = synth(seq_len * h * d, 7, 19, 0.03);
        let g: Vec<f32> = (0..seq_len).flat_map(|t| (0..h).map(move |hh| -(0.01 + 0.1 * (((hh + t) % 5) as f32)))).collect();
        let beta: Vec<f32> = (0..seq_len).flat_map(|t| (0..h).map(move |hh| 0.2 + 0.6 * (((hh * 7 + t * 3) % 11) as f32 / 11.0))).collect();

        let mut mixed_qkv_host = vec![0f32; seq_len * GDN_CONV_DIM];
        for t in 0..seq_len {
            mixed_qkv_host[t * GDN_CONV_DIM..t * GDN_CONV_DIM + GDN_KEY_DIM].copy_from_slice(&q[t * GDN_KEY_DIM..(t + 1) * GDN_KEY_DIM]);
            mixed_qkv_host[t * GDN_CONV_DIM + GDN_KEY_DIM..t * GDN_CONV_DIM + 2 * GDN_KEY_DIM].copy_from_slice(&k[t * GDN_KEY_DIM..(t + 1) * GDN_KEY_DIM]);
            mixed_qkv_host[t * GDN_CONV_DIM + 2 * GDN_KEY_DIM..(t + 1) * GDN_CONV_DIM].copy_from_slice(&v[t * GDN_VALUE_DIM..(t + 1) * GDN_VALUE_DIM]);
        }
        let mixed_qkv_bf16: Vec<u16> = mixed_qkv_host.iter().map(|&x| f32_to_bf16(x)).collect();

        let mut s = PrefillScratch::new(4096).expect("PrefillScratch::new failed");
        s.gdn_mixed_qkv.copy_from_host_prefix(&mixed_qkv_bf16).expect("copy_from_host_prefix(gdn_mixed_qkv) failed");
        s.gdn_g.copy_from_host_prefix(&g).expect("copy_from_host_prefix(gdn_g) failed");
        s.gdn_beta.copy_from_host_prefix(&beta).expect("copy_from_host_prefix(gdn_beta) failed");

        let conv_len = GDN_CONV_DIM * (GDN_CONV_KERNEL_SIZE - 1);
        let mut conv_state: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_len).unwrap();
        conv_state.copy_from_host(&vec![0u16; conv_len]).unwrap();
        let mut recurrent_state: DeviceBuffer<f32> = DeviceBuffer::alloc(h * d * d).unwrap();
        recurrent_state.copy_from_host(&vec![0f32; h * d * d]).unwrap();
        let mut layer_state = GdnLayerState { conv_state, recurrent_state };

        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        gdn_chunk_forward_prefill(handle.raw(), &mut layer_state, seq_len, &mut s, std::ptr::null_mut());

        let mut out_full = vec![0u16; s.gdn_out.len()];
        s.gdn_out.copy_to_host(&mut out_full).expect("copy_to_host(gdn_out) failed");
        let out: Vec<f32> = out_full.iter().map(|&b| bf16_to_f32(b)).collect();

        let mut state_after = vec![0f32; h * d * d];
        layer_state.recurrent_state.copy_to_host(&mut state_after).expect("copy_to_host(recurrent_state) failed");

        (out, state_after)
    }

    /// §100 decisive test: `gdn_chunk_forward_prefill` (the real
    /// chunked/parallel GDN prefill pipeline) matches the REAL
    /// `torch_chunk_gated_delta_rule` transformers function exactly
    /// (`scratchpad/gen_chunked_gdn_reference.py` -- an independent
    /// process, not this code's own path). `seq_len=8 < GDN_CHUNK_SIZE=64`
    /// deliberately exercises the real chunk-padding path (one chunk, 56
    /// fake zero-padded positions) -- the smallest real case that still
    /// touches every phase of the pipeline.
    ///
    /// Real, deliberate scope: the reference values below (including
    /// `head=31`) were independently computed for `GDN_NUM_V_HEADS=32`
    /// specifically -- real and shared by `qwen35_4b`/`qwen35_9b` (both
    /// real 32-head configs), but NOT `qwen35_0_8b`/`qwen35_2b` (16 heads)
    /// or `qwen35_27b` (48 heads), where the real synthetic input this
    /// test generates is a genuinely different shape/formula and these
    /// specific numbers do not apply. Gated accordingly, rather than
    /// failing for a real but misleading-looking reason under those
    /// features.
    #[test]
    #[cfg(any(feature = "qwen35_4b", feature = "qwen35_9b"))]
    fn real_gdn_chunk_forward_prefill_matches_real_transformers_function() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let d = GDN_HEAD_DIM;
        let (out, state_after) = run_chunked_gdn_case(8);

        // Independently computed by the real transformers
        // torch_chunk_gated_delta_rule on this exact synthetic data --
        // see scratchpad/gen_chunked_gdn_reference.py's stdout
        // ("seq=8 chunk=64" case).
        let out_t0_h0_expected: [f32; 8] = [-0.0000192810, -0.0000115686, -0.0000038562, 0.0000038562, 0.0000115686, 0.0000192810, 0.0000269934, 0.0000347058];
        let out_t0_h31_expected: [f32; 8] = [0.0006562563, 0.0005369370, 0.0004176176, 0.0002982983, 0.0001789790, 0.0000596597, -0.0000596597, -0.0001789790];
        let out_t7_h0_expected: [f32; 8] = [0.0002737664, 0.0003991439, 0.0006950248, -0.0008943378, -0.0005984569, -0.0003025760, 0.0000398988, 0.0003357797];
        let out_t7_h31_expected: [f32; 8] = [-0.0004500012, -0.0005173064, -0.0003445547, -0.0001718030, -0.0000780824, 0.0000946693, -0.0006016739, -0.0004289221];
        let state_h0_row0_expected: [f32; 8] = [-0.0094241491, -0.0100941947, -0.0113990232, -0.0011821315, -0.0024869598, -0.0037917888, 0.0105796494, 0.0092748199];
        let state_h31_row0_expected: [f32; 8] = [0.0095543936, 0.0121570667, 0.0109002497, 0.0096434327, 0.0125330016, 0.0112761846, -0.0131629938, -0.0144198108];

        let check = |got: &[f32], expected: &[f32; 8], label: &str| {
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert!((g - e).abs() < 0.002, "{label}[{i}]: GPU chunked GDN got {g}, independent Python reference {e}");
            }
        };
        check(&out[0 * GDN_VALUE_DIM..0 * GDN_VALUE_DIM + 8], &out_t0_h0_expected, "out[token=0,head=0]");
        check(&out[0 * GDN_VALUE_DIM + 31 * d..0 * GDN_VALUE_DIM + 31 * d + 8], &out_t0_h31_expected, "out[token=0,head=31]");
        check(&out[7 * GDN_VALUE_DIM..7 * GDN_VALUE_DIM + 8], &out_t7_h0_expected, "out[token=7,head=0]");
        check(&out[7 * GDN_VALUE_DIM + 31 * d..7 * GDN_VALUE_DIM + 31 * d + 8], &out_t7_h31_expected, "out[token=7,head=31]");
        check(&state_after[0..8], &state_h0_row0_expected, "state[head=0,row=0]");
        check(&state_after[31 * d * d..31 * d * d + 8], &state_h31_row0_expected, "state[head=31,row=0]");
    }

    /// §100 decisive test, MULTI-CHUNK case: `seq_len=70` spans TWO real
    /// chunks (`64 + 6`, the second chunk padded), specifically exercising
    /// the phase-2 sequential inter-chunk scan (state carried from chunk
    /// 0 into chunk 1's `inter_chunk_attn`/state update) -- the ONE part
    /// of this pipeline the single-chunk test above cannot touch at all.
    ///
    /// Real, deliberate scope: same `GDN_NUM_V_HEADS=32`-specific reference
    /// values as the single-chunk test above -- gated the same way, for
    /// the same reason.
    #[test]
    #[cfg(any(feature = "qwen35_4b", feature = "qwen35_9b"))]
    fn real_gdn_chunk_forward_prefill_multi_chunk_matches_real_transformers_function() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let d = GDN_HEAD_DIM;
        let seq_len = 70usize;
        let (out, state_after) = run_chunked_gdn_case(seq_len);

        // Independently computed by the real transformers
        // torch_chunk_gated_delta_rule -- see
        // scratchpad/gen_chunked_gdn_reference.py's stdout ("seq=70
        // chunk=64" case).
        let out_t0_h0_expected: [f32; 8] = [-0.0000192810, -0.0000115686, -0.0000038562, 0.0000038562, 0.0000115686, 0.0000192810, 0.0000269934, 0.0000347058];
        let out_t0_h31_expected: [f32; 8] = [0.0006562563, 0.0005369370, 0.0004176176, 0.0002982983, 0.0001789790, 0.0000596597, -0.0000596597, -0.0001789790];
        let out_t69_h0_expected: [f32; 8] = [-0.0005009993, -0.0002872147, -0.0000737846, 0.0000456523, 0.0002540201, -0.0001974405, 0.0000486784, 0.0002392080];
        let out_t69_h31_expected: [f32; 8] = [-0.0003250900, -0.0001639636, 0.0000633923, -0.0002338555, -0.0000596682, 0.0001661924, -0.0002314948, 0.0000220814];
        let state_h0_row0_expected: [f32; 8] = [-0.0000409468, 0.0020315913, 0.0039389962, 0.0060902084, 0.0081145335, 0.0034287462, 0.0050197304, 0.0073352465];
        let state_h31_row0_expected: [f32; 8] = [0.0096132690, 0.0079817381, 0.0060865935, 0.0037434942, 0.0011068368, -0.0006364647, 0.0006271894, -0.0005059629];

        let check = |got: &[f32], expected: &[f32; 8], label: &str| {
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert!((g - e).abs() < 0.002, "{label}[{i}]: GPU chunked GDN got {g}, independent Python reference {e}");
            }
        };
        check(&out[0 * GDN_VALUE_DIM..0 * GDN_VALUE_DIM + 8], &out_t0_h0_expected, "out[token=0,head=0]");
        check(&out[0 * GDN_VALUE_DIM + 31 * d..0 * GDN_VALUE_DIM + 31 * d + 8], &out_t0_h31_expected, "out[token=0,head=31]");
        let last = seq_len - 1;
        check(&out[last * GDN_VALUE_DIM..last * GDN_VALUE_DIM + 8], &out_t69_h0_expected, "out[token=69,head=0]");
        check(&out[last * GDN_VALUE_DIM + 31 * d..last * GDN_VALUE_DIM + 31 * d + 8], &out_t69_h31_expected, "out[token=69,head=31]");
        check(&state_after[0..8], &state_h0_row0_expected, "state[head=0,row=0]");
        check(&state_after[31 * d * d..31 * d * d + 8], &state_h31_row0_expected, "state[head=31,row=0]");
    }

    /// THE decisive test for this entire port: real weights, all 32 real
    /// layers, a real prompt, greedy-decoded through this from-scratch
    /// Rust/HIP forward pass -- and checked, token id for token id,
    /// against the REAL Qwen3.5-4B model's own greedy generation for the
    /// exact same prompt (`scratchpad/gen_reference_generation.py`, an
    /// independent process using real `transformers`/`AutoModelForCausalLM`,
    /// not this code's own path). Every kernel this session built has its
    /// own isolated correctness test; this is the one that asks the actual
    /// question this whole port exists to answer: does the ASSEMBLED
    /// system reproduce the real model's real behavior, not just each
    /// piece in isolation. Re-validated after §91's performance rewrite
    /// (raw non-syncing launches, scratch reuse) -- correctness must
    /// survive the optimization, not just the benchmark number.
    /// `#[ignore]`d like every other real-GPU, real-weights test in this
    /// crate (real weight loading takes real time); run explicitly.
    #[test]
    #[ignore]
    #[cfg(feature = "qwen35_4b")]
    fn real_greedy_generation_matches_real_qwen3_5_4b() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        eprintln!("loading real weights for all {NUM_LAYERS} layers...");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        eprintln!("weights loaded.");

        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        // Real prompt: "The capital of France is" -> real token ids from
        // this model's own tokenizer (see gen_reference_generation.py).
        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        // Real greedy continuation from the actual model
        // (scratchpad/gen_reference_generation.py's stdout, 2026-09-15).
        let expected_new_ids: [i32; 6] = [11751, 13, 198, 32, 13, 2912];

        let max_seq_len = 32usize;
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();

        // Prefill: feed every prompt token through the same single-token
        // decode step. Only the LAST prompt token's logits matter -- they
        // predict the first new token, matching real greedy generation's
        // own semantics.
        for &token_id in prompt_ids.iter() {
            forward_one_token(&handle, &weights, &mut state, token_id, &mut logits)
                .expect("forward_one_token failed during prefill");
        }

        let mut generated = Vec::with_capacity(expected_new_ids.len());
        for _ in 0..expected_new_ids.len() {
            let next_id = argmax_sample(&logits).expect("argmax_sample failed");
            generated.push(next_id);
            forward_one_token(&handle, &weights, &mut state, next_id, &mut logits)
                .expect("forward_one_token failed during generation");
        }

        eprintln!("generated: {generated:?}");
        eprintln!("expected:  {expected_new_ids:?}");
        assert_eq!(
            generated, expected_new_ids,
            "Rust decode loop's greedy generation did not match the real model's own greedy generation for the same prompt"
        );
    }

    /// §105: the SAME decisive correctness check as
    /// `real_greedy_generation_matches_real_qwen3_5_4b`, generalized across
    /// the real Qwen3.5 family -- one sibling test per size, each gated to
    /// only run when this binary was actually built for that size (`cargo
    /// test --features qwen35_0_8b`, etc.), against a real reference
    /// generated the same way (`scratchpad/gen_reference_multi.py`, same
    /// prompt, same real HF `transformers` forward pass, 2026-09-17). Same
    /// tokenizer/vocab confirmed shared across the whole family (identical
    /// `prompt_ids` for all four sizes), so only `expected_new_ids` and
    /// the model-name string differ per size.
    #[test]
    #[ignore]
    #[cfg(feature = "qwen35_0_8b")]
    fn real_greedy_generation_matches_real_qwen3_5_0_8b() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-0.8B snapshot found on this machine");
        eprintln!("loading real weights for all {NUM_LAYERS} layers...");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        eprintln!("weights loaded.");

        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        // Real greedy continuation from the actual model
        // (scratchpad/gen_reference_multi.py's stdout, 2026-09-17).
        let expected_new_ids: [i32; 6] = [11751, 13, 198, 760, 6511, 314];

        let max_seq_len = 32usize;
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();

        for &token_id in prompt_ids.iter() {
            forward_one_token(&handle, &weights, &mut state, token_id, &mut logits)
                .expect("forward_one_token failed during prefill");
        }

        let mut generated = Vec::with_capacity(expected_new_ids.len());
        for _ in 0..expected_new_ids.len() {
            let next_id = argmax_sample(&logits).expect("argmax_sample failed");
            generated.push(next_id);
            forward_one_token(&handle, &weights, &mut state, next_id, &mut logits)
                .expect("forward_one_token failed during generation");
        }

        eprintln!("generated: {generated:?}");
        eprintln!("expected:  {expected_new_ids:?}");
        assert_eq!(
            generated, expected_new_ids,
            "Rust decode loop's greedy generation did not match the real model's own greedy generation for the same prompt"
        );
    }

    #[test]
    #[ignore]
    #[cfg(feature = "qwen35_2b")]
    fn real_greedy_generation_matches_real_qwen3_5_2b() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-2B snapshot found on this machine");
        eprintln!("loading real weights for all {NUM_LAYERS} layers...");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        eprintln!("weights loaded.");

        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        // Real greedy continuation from the actual model
        // (scratchpad/gen_reference_multi.py's stdout, 2026-09-17).
        let expected_new_ids: [i32; 6] = [11751, 13, 198, 32, 13, 2912];

        let max_seq_len = 32usize;
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();

        for &token_id in prompt_ids.iter() {
            forward_one_token(&handle, &weights, &mut state, token_id, &mut logits)
                .expect("forward_one_token failed during prefill");
        }

        let mut generated = Vec::with_capacity(expected_new_ids.len());
        for _ in 0..expected_new_ids.len() {
            let next_id = argmax_sample(&logits).expect("argmax_sample failed");
            generated.push(next_id);
            forward_one_token(&handle, &weights, &mut state, next_id, &mut logits)
                .expect("forward_one_token failed during generation");
        }

        eprintln!("generated: {generated:?}");
        eprintln!("expected:  {expected_new_ids:?}");
        assert_eq!(
            generated, expected_new_ids,
            "Rust decode loop's greedy generation did not match the real model's own greedy generation for the same prompt"
        );
    }

    #[test]
    #[ignore]
    #[cfg(feature = "qwen35_9b")]
    fn real_greedy_generation_matches_real_qwen3_5_9b() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-9B snapshot found on this machine");
        eprintln!("loading real weights for all {NUM_LAYERS} layers (including the real, separate, untied lm_head -- §105)...");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        assert!(weights.lm_head.is_some(), "Qwen3.5-9B is known (checked via its real safetensors index) to have an untied lm_head.weight -- if this is None, weight loading silently fell back to the wrong (tied) path");
        eprintln!("weights loaded.");

        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        // Real greedy continuation from the actual model
        // (scratchpad/gen_reference_multi.py's stdout, 2026-09-17).
        let expected_new_ids: [i32; 6] = [11751, 13, 198, 760, 6511, 314];

        let max_seq_len = 32usize;
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();

        for &token_id in prompt_ids.iter() {
            forward_one_token(&handle, &weights, &mut state, token_id, &mut logits)
                .expect("forward_one_token failed during prefill");
        }

        let mut generated = Vec::with_capacity(expected_new_ids.len());
        for _ in 0..expected_new_ids.len() {
            let next_id = argmax_sample(&logits).expect("argmax_sample failed");
            generated.push(next_id);
            forward_one_token(&handle, &weights, &mut state, next_id, &mut logits)
                .expect("forward_one_token failed during generation");
        }

        eprintln!("generated: {generated:?}");
        eprintln!("expected:  {expected_new_ids:?}");
        assert_eq!(
            generated, expected_new_ids,
            "Rust decode loop's greedy generation did not match the real model's own greedy generation for the same prompt"
        );
    }

    /// §106: the first real, end-to-end exercise of the quantized 27B
    /// path -- real W4A16 weights (from `quantize_w4a16.py`'s real
    /// output), real tokenizer, real chat template, real greedy
    /// generation. No hardcoded expected token ids here (unlike the
    /// bf16 decisive tests above) -- this is the coherence sanity gate
    /// that must pass BEFORE a real top-1-agreement comparison against
    /// llama.cpp/Ollama's own real Q4_K_M generation means anything (a
    /// broken quantized model could still coincidentally "agree" on a
    /// few tokens; readable, on-topic real text is the first real bar).
    #[test]
    #[ignore]
    #[cfg(feature = "qwen35_27b")]
    fn real_quantized_27b_produces_coherent_generation() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let quantized_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/qwen38_27b_w4a16");
        if !quantized_dir.is_dir() {
            eprintln!("skipping: no real quantized 27B checkpoint at {}", quantized_dir.display());
            return;
        }

        eprintln!("loading real quantized weights for all {NUM_LAYERS} layers...");
        let weights = ModelWeights::load(&quantized_dir).expect("real quantized weight loading failed");
        eprintln!("weights loaded.");
        let tok = crate::tokenizer::ChatTokenizer::load(&quantized_dir).expect("real tokenizer.json failed to load");

        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        let prompt = tok.apply_chat_template(&[
            ("system", "You are a helpful assistant."),
            ("user", "What is the capital of France? Answer in one short sentence."),
        ]);
        let prompt_ids = tok.encode(&prompt).expect("real tokenizer encode failed");
        eprintln!("prompt ({} real tokens): {prompt:?}", prompt_ids.len());

        let max_seq_len = 256usize;
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();

        // §108: real batched prefill, the same path server.rs's
        // `start_request` now uses for quantized checkpoints (the
        // per-token loop this test used before §108 is gone from
        // production -- this test now exercises what actually runs).
        forward_prefill(&handle, &weights, &mut state, &prompt_ids, &mut logits).expect("forward_prefill failed during prefill");

        let eos = tok.eos_token_id();
        let mut generated = Vec::new();
        for _ in 0..60 {
            let next_id = argmax_sample(&logits).expect("argmax_sample failed");
            if next_id == eos {
                break;
            }
            generated.push(next_id);
            forward_one_token(&handle, &weights, &mut state, next_id, &mut logits)
                .expect("forward_one_token failed during generation");
        }

        let text = tok.decode(&generated).expect("real tokenizer decode failed");
        eprintln!("=== real quantized 27B generation ({} tokens) ===\n{text}\n===", generated.len());
        eprintln!("raw token ids: {generated:?}");
        assert!(!generated.is_empty(), "generated zero tokens -- immediate EOS is not a real answer");
        assert!(text.to_lowercase().contains("paris"), "expected a real, on-topic answer mentioning Paris, got: {text:?}");
    }

    /// §111 diagnostic: real WALL-CLOCK time for `forward_prefill` on the
    /// real quantized checkpoint, isolated from HTTP/server overhead --
    /// compared against rocprofv3's real GPU-kernel-busy-time finding for
    /// the same real prompt (~42 tokens, matching the HTTP benchmark's
    /// own real prompt length) to see how much of the real observed TTFT
    /// gap is GPU kernel time vs. something else (per-launch host
    /// dispatch overhead -- prefill isn't HIP-Graph-captured, unlike
    /// decode -- HTTP/server layers, etc).
    #[test]
    #[ignore]
    #[cfg(feature = "qwen35_4b")]
    fn diagnose_real_quantized_prefill_wall_clock_vs_kernel_time() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let quantized_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen35_4b_w4a16");
        if !quantized_dir.is_dir() {
            eprintln!("skipping: no real quantized 4B checkpoint at {}", quantized_dir.display());
            return;
        }

        let weights = ModelWeights::load(&quantized_dir).expect("real quantized weight loading failed");
        let tok = crate::tokenizer::ChatTokenizer::load(&quantized_dir).expect("real tokenizer.json failed to load");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        // Real prompt text matching run_4b_engine_comparison_benchmark.py's
        // BENCHMARK_PROMPTS[0] -- same 42-real-token scale as the rocprofv3
        // trace this compares against.
        let system = "You are a senior PostgreSQL and Python backend architect.";
        let user = "Write a complete production FastAPI application with an asyncpg connection pool, pgvector HNSW search endpoint, and Pydantic response models.";
        let prompt = tok.apply_chat_template(&[("system", system), ("user", user)]);
        let prompt_ids = tok.encode(&prompt).expect("real encode failed");
        eprintln!("real prompt: {} tokens", prompt_ids.len());

        let max_seq_len = 256usize;
        let iters = 10;
        let mut total = std::time::Duration::ZERO;
        for i in 0..iters {
            let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
            let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
            hip::device_synchronize().unwrap();
            let t0 = std::time::Instant::now();
            forward_prefill(&handle, &weights, &mut state, &prompt_ids, &mut logits).expect("forward_prefill failed");
            hip::device_synchronize().unwrap();
            let elapsed = t0.elapsed();
            if i > 0 {
                // Skip the first iter (real one-time warmup/JIT-ish cost).
                total += elapsed;
            }
            eprintln!("iter {i}: {:.3} ms", elapsed.as_secs_f64() * 1000.0);
        }
        let avg_ms = total.as_secs_f64() * 1000.0 / (iters - 1) as f64;
        eprintln!("\nreal avg wall-clock forward_prefill time ({} tokens, {} iters after warmup): {avg_ms:.3} ms", prompt_ids.len(), iters - 1);
    }

    /// The §93 graph-capture equivalent of the decisive test above: same
    /// real weights, real prompt, real independently-verified expected
    /// token ids -- but decoded through `GraphedDecodeState` instead of
    /// the eager `forward_one_token`. The graph is captured on the FIRST
    /// call (the first prefill token, at position 0) and every remaining
    /// prefill/generation call REPLAYS that same graph against advancing
    /// positions -- the exact property `hip::tests::
    /// real_hip_graph_capture_replay_reflects_new_data_without_recapture`
    /// and `blas::tests::real_gemm_is_capturable_and_replays_new_input_correctly`
    /// already proved in isolation, now exercised end-to-end against the
    /// real model.
    #[test]
    #[ignore]
    #[cfg(feature = "qwen35_4b")]
    fn real_graphed_greedy_generation_matches_real_qwen3_5_4b() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        eprintln!("loading real weights for all {NUM_LAYERS} layers...");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        eprintln!("weights loaded.");

        let mut graphed = GraphedDecodeState::new().expect("GraphedDecodeState::new failed");

        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let expected_new_ids: [i32; 6] = [11751, 13, 198, 32, 13, 2912];

        let max_seq_len = 32usize;
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();

        for &token_id in prompt_ids.iter() {
            graphed
                .forward_one_token(&weights, &mut state, token_id, &mut logits)
                .expect("GraphedDecodeState::forward_one_token failed during prefill");
        }

        let mut generated = Vec::with_capacity(expected_new_ids.len());
        for _ in 0..expected_new_ids.len() {
            let next_id = argmax_sample(&logits).expect("argmax_sample failed");
            generated.push(next_id);
            graphed
                .forward_one_token(&weights, &mut state, next_id, &mut logits)
                .expect("GraphedDecodeState::forward_one_token failed during generation");
        }

        eprintln!("generated (graphed): {generated:?}");
        eprintln!("expected:            {expected_new_ids:?}");
        assert_eq!(
            generated, expected_new_ids,
            "GraphedDecodeState's greedy generation did not match the real model's own greedy generation for the same prompt"
        );
    }

    /// §96: the correctness gate for real batched prefill. Same real
    /// prompt and same independently-verified real greedy continuation as
    /// `real_greedy_generation_matches_real_qwen3_5_4b` above (real HF
    /// transformers' own reference output, not this crate's own oracle) --
    /// but the 5 real prompt tokens go through ONE real `forward_prefill`
    /// call (a genuine `T=5` batch: real batched GEMMs for every layer's
    /// projections, real per-token inner loops for RoPE/KV-cache-append/
    /// attention/conv1d/gdn-recurrent) instead of 5 sequential
    /// `forward_one_token` calls. Generation after the prefill continues
    /// through the ordinary `forward_one_token` decode path unchanged --
    /// so this test also proves the KV cache and GDN conv/recurrent state
    /// the batched path writes are correctly READABLE by the existing,
    /// already-proven decode path afterward (a real correctness boundary:
    /// wrong per-head/per-token indexing anywhere in the new batched code
    /// would show up as wrong logits here, either for the first generated
    /// token -- proving `forward_prefill` itself is wrong -- or for a LATER
    /// one -- proving the state it left behind for `forward_one_token` to
    /// continue from is wrong).
    #[test]
    #[ignore]
    #[cfg(feature = "qwen35_4b")]
    fn real_batched_prefill_matches_real_qwen3_5_4b_greedy_generation() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        eprintln!("loading real weights for all {NUM_LAYERS} layers...");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        eprintln!("weights loaded.");

        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let expected_new_ids: [i32; 6] = [11751, 13, 198, 32, 13, 2912];

        let max_seq_len = 32usize;
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();

        // ONE real batched prefill call for all 5 real prompt tokens,
        // instead of 5 sequential forward_one_token calls.
        forward_prefill(&handle, &weights, &mut state, &prompt_ids, &mut logits)
            .expect("forward_prefill failed");
        assert_eq!(
            state.position,
            prompt_ids.len(),
            "forward_prefill must advance state.position by the full prompt length"
        );

        let mut generated = Vec::with_capacity(expected_new_ids.len());
        for _ in 0..expected_new_ids.len() {
            let next_id = argmax_sample(&logits).expect("argmax_sample failed");
            generated.push(next_id);
            forward_one_token(&handle, &weights, &mut state, next_id, &mut logits)
                .expect("forward_one_token failed during generation");
        }

        eprintln!("generated (batched prefill): {generated:?}");
        eprintln!("expected:                    {expected_new_ids:?}");
        assert_eq!(
            generated, expected_new_ids,
            "batched-prefill-then-decode did not match the real model's own greedy generation for the same prompt"
        );
    }

    /// §96: a second, independent decisive test targeting the case the
    /// test above cannot: a prompt LONGER than `state.position == 0`'s
    /// trivial single-chunk case is already covered, so this one instead
    /// proves batched prefill and the ordinary sequential decode path
    /// produce NUMERICALLY EQUIVALENT logits for the exact same real
    /// prompt -- not just the same final greedy argmax id (which the test
    /// above already proves, but could in principle coincide even with a
    /// real bug elsewhere in the vocab that never happens to touch the
    /// top-1 logit).
    ///
    /// Deliberately NOT bit-exact: the batched path's projections go
    /// through real hipBLAS (`raw::gemm`'s `rows > 1` branch,
    /// `hipblasGemmEx`'s own tuned reduction order), while the sequential
    /// path's rows=1 projections go through this crate's hand-written GEMV
    /// kernel (`gemv.hip`, a different reduction order) -- both are real,
    /// independently-verified, CORRECT fp32-accumulate implementations of
    /// the exact same GEMM (see `blas::tests::
    /// real_gemm_matches_real_down_proj_weight` for hipBLAS's own oracle
    /// check, and `kernels::tests::real_gemv_matches_real_down_proj_weight`
    /// for the GEMV kernel's), but floating-point addition is not
    /// associative, so two different reduction orders over the same real
    /// values can legitimately differ in the last few mantissa bits -- the
    /// same reason `blas.rs`'s own tests use a numeric tolerance
    /// (`(g-e).abs() < 0.01/0.02`), not `assert_eq!`, when comparing GEMM
    /// output to an independent reference. This test uses that same
    /// convention: real prompt, real weights, compare in f32 space with a
    /// real tolerance, and separately assert the (much stronger) property
    /// that already held: the top-1 (argmax) token agrees exactly.
    #[test]
    #[ignore]
    fn real_batched_prefill_logits_numerically_match_sequential_forward_one_token() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        eprintln!("loading real weights for all {NUM_LAYERS} layers...");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        eprintln!("weights loaded.");

        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let max_seq_len = 32usize;

        let mut state_batched = DecodeState::new(max_seq_len).expect("DecodeState allocation failed (batched)");
        let mut logits_batched: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        forward_prefill(&handle, &weights, &mut state_batched, &prompt_ids, &mut logits_batched)
            .expect("forward_prefill failed");

        let mut state_sequential = DecodeState::new(max_seq_len).expect("DecodeState allocation failed (sequential)");
        let mut logits_sequential: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        for &token_id in prompt_ids.iter() {
            forward_one_token(&handle, &weights, &mut state_sequential, token_id, &mut logits_sequential)
                .expect("forward_one_token failed during sequential prefill");
        }

        let mut batched_host = vec![0u16; VOCAB_SIZE];
        let mut sequential_host = vec![0u16; VOCAB_SIZE];
        logits_batched.copy_to_host(&mut batched_host).unwrap();
        logits_sequential.copy_to_host(&mut sequential_host).unwrap();

        fn bf16_to_f32(bits: u16) -> f32 {
            f32::from_bits((bits as u32) << 16)
        }

        // The strong property: the two paths' top-1 token must agree
        // exactly (this is what `argmax_sample`/greedy decoding actually
        // relies on downstream -- see the sibling
        // `real_batched_prefill_matches_real_qwen3_5_4b_greedy_generation`
        // test, which already proves this end-to-end across 6 real
        // generation steps against the real model's own reference output).
        let argmax = |host: &[u16]| -> (usize, f32) {
            let mut best_i = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &b) in host.iter().enumerate() {
                let v = bf16_to_f32(b);
                if v > best_v {
                    best_v = v;
                    best_i = i;
                }
            }
            (best_i, best_v)
        };
        let (batched_argmax, _) = argmax(&batched_host);
        let (sequential_argmax, _) = argmax(&sequential_host);
        assert_eq!(
            batched_argmax, sequential_argmax,
            "batched and sequential prefill must agree on the top-1 (argmax) logit"
        );

        // The numeric property: every logit should be CLOSE (real
        // fp32-accumulate GEMM vs. real fp32-accumulate GEMV, different
        // reduction order -- see this test's own doc comment), not
        // identical. Report both the max deviation and how many of the
        // 248320 real vocab logits exceed a real, generous tolerance --
        // honest diagnostics rather than a single silent pass/fail.
        let tolerance = 0.5f32;
        let mut max_diff = 0f32;
        let mut num_exceeding = 0usize;
        for (&b, &s) in batched_host.iter().zip(sequential_host.iter()) {
            let diff = (bf16_to_f32(b) - bf16_to_f32(s)).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > tolerance {
                num_exceeding += 1;
            }
        }
        eprintln!(
            "logit comparison: max_diff={max_diff}, {num_exceeding}/{VOCAB_SIZE} elements exceed tolerance={tolerance}"
        );
        assert!(
            num_exceeding == 0,
            "{num_exceeding} of {VOCAB_SIZE} logits differed by more than {tolerance} between batched and sequential prefill (max_diff={max_diff})"
        );
    }

    /// §95: the correctness gate for `apps/runtime-next`'s new HTTP server
    /// reusing ONE `DecodeState`+`GraphedDecodeState` (and its ALREADY-
    /// CAPTURED HIP graph, tied to fixed buffer addresses) across many
    /// independent requests via `DecodeState::reset()`, instead of
    /// allocating a fresh `DecodeState` per request (which would force a
    /// fresh graph capture every request -- defeating the whole point of
    /// §93's graph-capture work).
    ///
    /// Real method: run a real prompt through `graphed`+`state` to
    /// completion (this "dirties" GDN's real recurrent state with real,
    /// prompt-1-specific values), call `state.reset()`, then run a
    /// DIFFERENT real prompt through the SAME `graphed`+`state`. Compare
    /// its generated ids against what a GENUINELY FRESH `DecodeState` (a
    /// second, independent instance, never touched by prompt 1 at all)
    /// produces for the identical second prompt. If `reset()` left any
    /// real state leaking from prompt 1 into prompt 2's computation (most
    /// plausibly GDN's `recurrent_state`, since it's read-modify-write
    /// regardless of position -- see `reset()`'s own doc comment), the two
    /// would diverge; if `reset()` is correct, they must match EXACTLY
    /// (this is still fully deterministic greedy decoding).
    #[test]
    #[ignore]
    fn real_reset_prevents_cross_request_state_leakage() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        let tok = crate::tokenizer::ChatTokenizer::load(&snapshot).expect("real tokenizer.json failed to load");

        // Two real, different prompts -- real BPE-encoded, not hardcoded.
        let prompt1_ids = tok.encode("The capital of France is").expect("real encode failed");
        let prompt2_ids = tok.encode("The largest planet in our solar system is").expect("real encode failed");

        let max_seq_len = 32usize;
        let new_tokens = 6usize;

        // Run prompt 1 to completion on `graphed`/`state`, dirtying real
        // GDN recurrent state with real prompt-1-specific values.
        let mut graphed = GraphedDecodeState::new().expect("GraphedDecodeState::new failed");
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        for &token_id in prompt1_ids.iter() {
            graphed
                .forward_one_token(&weights, &mut state, token_id, &mut logits)
                .expect("forward_one_token failed during prompt 1 prefill");
        }
        for _ in 0..new_tokens {
            let next_id = argmax_sample(&logits).expect("argmax_sample failed");
            graphed
                .forward_one_token(&weights, &mut state, next_id, &mut logits)
                .expect("forward_one_token failed during prompt 1 generation");
        }

        // Reset, then run prompt 2 on the SAME (reused) graphed/state.
        state.reset().expect("DecodeState::reset failed");
        for &token_id in prompt2_ids.iter() {
            graphed
                .forward_one_token(&weights, &mut state, token_id, &mut logits)
                .expect("forward_one_token failed during prompt 2 prefill (after reset)");
        }
        let mut generated_after_reset = Vec::with_capacity(new_tokens);
        for _ in 0..new_tokens {
            let next_id = argmax_sample(&logits).expect("argmax_sample failed");
            generated_after_reset.push(next_id);
            graphed
                .forward_one_token(&weights, &mut state, next_id, &mut logits)
                .expect("forward_one_token failed during prompt 2 generation (after reset)");
        }

        // Run prompt 2 on a GENUINELY FRESH graphed/state, never touched by prompt 1.
        let mut graphed_fresh = GraphedDecodeState::new().expect("GraphedDecodeState::new failed (fresh)");
        let mut state_fresh = DecodeState::new(max_seq_len).expect("DecodeState allocation failed (fresh)");
        let mut logits_fresh: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        for &token_id in prompt2_ids.iter() {
            graphed_fresh
                .forward_one_token(&weights, &mut state_fresh, token_id, &mut logits_fresh)
                .expect("forward_one_token failed during prompt 2 prefill (fresh)");
        }
        let mut generated_fresh = Vec::with_capacity(new_tokens);
        for _ in 0..new_tokens {
            let next_id = argmax_sample(&logits_fresh).expect("argmax_sample failed");
            generated_fresh.push(next_id);
            graphed_fresh
                .forward_one_token(&weights, &mut state_fresh, next_id, &mut logits_fresh)
                .expect("forward_one_token failed during prompt 2 generation (fresh)");
        }

        eprintln!("generated after reset: {generated_after_reset:?}");
        eprintln!("generated fresh:       {generated_fresh:?}");
        assert_eq!(
            generated_after_reset, generated_fresh,
            "DecodeState::reset() left real state leaking from prompt 1 into prompt 2's generation"
        );
    }

    /// §98: a dedicated, PREFILL-ONLY real workload for `rocprofv3` to
    /// trace -- exists purely so a kernel-trace run captures a clean
    /// picture of `forward_prefill`'s own real kernel mix, undiluted by
    /// any decode-loop kernels (`GraphedDecodeState::forward_one_token`)
    /// that would otherwise dominate a longer-running test. Real ~500-token
    /// prompt (matching the scale this repo's own TTFT discussion has used
    /// throughout §95-§98), one real `forward_prefill` call, nothing else.
    /// Run via:
    /// `rocprofv3 --kernel-trace --stats -S -- <compiled test binary>
    /// model::tests::bench_real_prefill_only_for_profiling --ignored --exact
    /// --nocapture --test-threads=1`
    /// -- same real methodology §93/§94 already established for this crate.
    #[test]
    #[ignore]
    fn bench_real_prefill_only_for_profiling() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        let tok = crate::tokenizer::ChatTokenizer::load(&snapshot).expect("real tokenizer.json failed to load");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        // A real, ~500-token instruction -- repeats real prose (not padding
        // tokens) so BPE produces a realistic token distribution, matching
        // the scale of a real multi-paragraph coding-task prompt.
        let paragraph = "Design a complete production-grade PostgreSQL architecture for a multi-tenant \
            semantic document retrieval system. Include schemas for tenants, documents, chunks, pgvector \
            HNSW indexes with optimal m and ef_construction parameters, partition tables by tenant_id, and \
            provide realistic DDL with foreign keys, check constraints, and composite indexes. ";
        let long_prompt = paragraph.repeat(6);
        let prompt = tok.apply_chat_template(&[("system", "You are a senior database architect."), ("user", &long_prompt)]);
        let prompt_ids = tok.encode(&prompt).expect("real encode failed");
        eprintln!("profiling prefill for a real {}-token prompt", prompt_ids.len());

        let max_seq_len = 1024usize;
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();

        forward_prefill(&handle, &weights, &mut state, &prompt_ids, &mut logits).expect("forward_prefill failed");
    }

    /// §99: the SAME profiling harness as `bench_real_prefill_only_for_profiling`
    /// above, but at the REAL, representative prompt length this crate's own
    /// HTTP benchmark (`benchmarks/harness_sdk/run_4b_engine_comparison_benchmark.py`)
    /// actually sends -- confirmed via a real `curl` against the running
    /// server's `usage.prompt_tokens`: 54 tokens, not the ~400-token scale
    /// the other profiling test used. Exists because the §99 attention
    /// rewrite's real end-to-end TTFT benchmark came back UNCHANGED (even
    /// slightly worse) despite a real, measured 21% total-kernel-time
    /// reduction at 401 tokens -- this test isolates whether the new
    /// per-head GEMM attention path has a real crossover point where its
    /// fixed per-call (Tensile dispatch) overhead stops being amortized by
    /// the actual compute at small `T`/`kv_len`.
    #[test]
    #[ignore]
    fn bench_real_prefill_only_for_profiling_short_prompt() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        let tok = crate::tokenizer::ChatTokenizer::load(&snapshot).expect("real tokenizer.json failed to load");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        // The EXACT real prompt text from BENCHMARK_PROMPTS[0] in
        // run_4b_engine_comparison_benchmark.py -- confirmed 54 real
        // prompt tokens via a live curl against the server.
        let system = "You are a senior PostgreSQL and Python backend architect.";
        let user = "Write a complete production FastAPI application with an asyncpg connection pool, pgvector HNSW search endpoint, and Pydantic response models.";
        let prompt = tok.apply_chat_template(&[("system", system), ("user", user)]);
        let prompt_ids = tok.encode(&prompt).expect("real encode failed");
        eprintln!("profiling prefill for a real {}-token prompt (representative scale)", prompt_ids.len());

        let max_seq_len = 256usize;
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();

        // The VERY FIRST real call on a freshly-loaded engine -- matches
        // exactly what a real server's first real request after startup
        // experiences (this is the number the real HTTP benchmark's TTFT
        // actually measures, not a warmed-up steady-state number).
        let t_cold0 = std::time::Instant::now();
        forward_prefill(&handle, &weights, &mut state, &prompt_ids, &mut logits).expect("forward_prefill failed (cold)");
        let cold_ms = t_cold0.elapsed().as_secs_f64() * 1000.0;

        let repeats = 10usize;
        let mut wall_times_ms = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            state.position = 0; // real prefill from a clean position each time, same real call
            let t0 = std::time::Instant::now();
            forward_prefill(&handle, &weights, &mut state, &prompt_ids, &mut logits).expect("forward_prefill failed");
            wall_times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let avg_ms = wall_times_ms.iter().sum::<f64>() / repeats as f64;
        eprintln!("COLD (first-ever) forward_prefill call: {cold_ms:.2}ms");
        eprintln!("WARM forward_prefill time over {repeats} repeats: {wall_times_ms:.2?} ms, avg {avg_ms:.2}ms");
        eprintln!("cold/warm ratio: {:.2}x", cold_ms / avg_ms);

        // §103: the REAL server calls `state.reset()` before EVERY
        // request (not just `position=0`) -- timed separately here to
        // rule it out as a real TTFT contributor. RESULT (kept as a real
        // regression guard, not just a one-off investigation):
        // `reset()`'s 48 `fill_zero()` calls (24 GDN layers x 2 buffers)
        // cost ~0.05ms total, negligible -- the REAL gap between this
        // microbenchmark's own ~70ms and the HTTP server's own
        // previously-measured ~600-700ms TTFT was `server.rs`'s SSE
        // streaming path, NOT anything in this forward pass (see
        // `docs/DECISIONS.md` §103's own writeup: `tiny_http`'s
        // `chunked_transfer::Encoder` buffers 8192 bytes with no
        // per-write flush, fixed by bypassing it via `Request::into_writer`).
        let mut reset_times_ms = Vec::with_capacity(repeats);
        let mut full_request_times_ms = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let tr = std::time::Instant::now();
            state.reset().expect("state.reset() failed");
            reset_times_ms.push(tr.elapsed().as_secs_f64() * 1000.0);
            let t0 = std::time::Instant::now();
            forward_prefill(&handle, &weights, &mut state, &prompt_ids, &mut logits).expect("forward_prefill failed");
            full_request_times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let avg_reset_ms = reset_times_ms.iter().sum::<f64>() / repeats as f64;
        let avg_full_ms = full_request_times_ms.iter().sum::<f64>() / repeats as f64;
        eprintln!("state.reset() alone: {reset_times_ms:.2?} ms, avg {avg_reset_ms:.2}ms");
        eprintln!("forward_prefill AFTER a real reset(): {full_request_times_ms:.2?} ms, avg {avg_full_ms:.2}ms");
    }

    /// The real, measured decode throughput after §91's performance
    /// rewrite (pre-allocated scratch buffers, raw non-syncing kernel/GEMM
    /// launches, one sync per token) -- directly comparable to §89's
    /// "naive assembly" 31.6-31.8 tok/s and §90's real llama.cpp
    /// (75.10 ± 0.18 tok/s) / Ollama (76.0-76.2 tok/s) numbers, same
    /// prompt, same shapes, same machine.
    #[test]
    #[ignore]
    fn bench_real_decode_tokens_per_second() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let max_seq_len = 64usize;
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();

        for &token_id in prompt_ids.iter() {
            forward_one_token(&handle, &weights, &mut state, token_id, &mut logits).unwrap();
        }

        let warmup = 3usize;
        for _ in 0..warmup {
            let next_id = argmax_sample(&logits).unwrap();
            forward_one_token(&handle, &weights, &mut state, next_id, &mut logits).unwrap();
        }

        let timed_tokens = 20usize;
        let t0 = std::time::Instant::now();
        for _ in 0..timed_tokens {
            let next_id = argmax_sample(&logits).unwrap();
            forward_one_token(&handle, &weights, &mut state, next_id, &mut logits).unwrap();
        }
        let elapsed = t0.elapsed();
        let per_token_ms = elapsed.as_secs_f64() * 1000.0 / timed_tokens as f64;
        let tokens_per_sec = timed_tokens as f64 / elapsed.as_secs_f64();
        println!(
            "REAL measured decode throughput (§91: scratch reuse + non-syncing raw launches, KV cache at position ~{}-{}): {per_token_ms:.3} ms/token, {tokens_per_sec:.2} tokens/sec ({timed_tokens} tokens timed, {warmup} warmup)",
            prompt_ids.len(),
            prompt_ids.len() + warmup + timed_tokens
        );
    }

    /// §93: same shapes/prompt/warmup/timed-token counts as
    /// `bench_real_decode_tokens_per_second` above, decoded through
    /// `GraphedDecodeState` instead -- the real, directly-comparable
    /// before/after number for the whole graph-capture effort.
    #[test]
    #[ignore]
    fn bench_real_graphed_decode_tokens_per_second() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        let mut graphed = GraphedDecodeState::new().expect("GraphedDecodeState::new failed");

        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let max_seq_len = 64usize;
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();

        for &token_id in prompt_ids.iter() {
            graphed.forward_one_token(&weights, &mut state, token_id, &mut logits).unwrap();
        }

        let warmup = 3usize;
        for _ in 0..warmup {
            let next_id = argmax_sample(&logits).unwrap();
            graphed.forward_one_token(&weights, &mut state, next_id, &mut logits).unwrap();
        }

        let timed_tokens = 20usize;
        let t0 = std::time::Instant::now();
        for _ in 0..timed_tokens {
            let next_id = argmax_sample(&logits).unwrap();
            graphed.forward_one_token(&weights, &mut state, next_id, &mut logits).unwrap();
        }
        let elapsed = t0.elapsed();
        let per_token_ms = elapsed.as_secs_f64() * 1000.0 / timed_tokens as f64;
        let tokens_per_sec = timed_tokens as f64 / elapsed.as_secs_f64();
        println!(
            "REAL measured GRAPHED decode throughput (§93: HIP Graph capture/replay, KV cache at position ~{}-{}): {per_token_ms:.3} ms/token, {tokens_per_sec:.2} tokens/sec ({timed_tokens} tokens timed, {warmup} warmup)",
            prompt_ids.len(),
            prompt_ids.len() + warmup + timed_tokens
        );
    }

    /// DIAGNOSTIC (speculative-decoding feasibility prep, not a shipped
    /// feature -- this crate has no drafting mechanism): measures the real
    /// cost of a speculative-decode "verify" step using ONLY kernels this
    /// crate already has, no new kernels written. A verify step is a
    /// K-token batched chunk forward continuing from live decode state --
    /// exactly the real, already-decisively-verified prefill machinery
    /// (`run_layers_over_chunk`, proven bit-for-bit-tolerance-equivalent to
    /// sequential `forward_one_token` by
    /// `real_batched_prefill_logits_numerically_match_sequential_forward_one_token`
    /// and `real_incremental_prefill_matches_one_shot_prefill_numerically`)
    /// -- PLUS real per-position logits at EVERY one of the K positions,
    /// which a genuine verify step needs (to argmax-check each drafted
    /// token) but `forward_prefill_chunk` deliberately skips (its own doc
    /// comment: only the last token's logits are ever computed, since nothing
    /// else needs them there). Real batched multi-row `rmsnorm`+lm_head GEMM
    /// (`raw::gemm` already supports `rows>1` as a genuine hipBLAS
    /// matrix-matrix product, see its own doc comment) stands in for that
    /// missing piece -- no new kernel, just calling existing ones with
    /// `rows=K` instead of `rows=1`.
    ///
    /// Reports, for K in {2,4,6,8} (matching the legacy Python engine's own
    /// real K sweep, `docs/DECISIONS.md` §61-65): the real verify-chunk
    /// cost `C_verify(K)`, the real steady-state single-token decode cost
    /// `C_1`, and the real break-even accepted-token count `tau_breakeven(K)
    /// = C_verify(K) / C_1` -- how many drafted tokens a round would need to
    /// accept, ON AVERAGE, before speculation nets a real win on THIS
    /// crate's actual kernels. Scoped to bf16 dense sizes (4B/9B) --
    /// panics if `lm_head`/`embed_tokens` turns out quantized.
    #[test]
    #[ignore]
    fn diagnose_real_speculative_verify_chunk_breakeven() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        assert!(!weights.is_quantized(), "this diagnostic assumes a bf16 dense checkpoint (4B/9B) -- lm_head batched GEMM below assumes bf16");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        let handle_raw = handle.raw();

        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let max_seq_len = 256usize;
        let warmup = 3usize;

        // --- C_1: real steady-state single-token decode cost ---
        let mut c1_state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        for &token_id in prompt_ids.iter() {
            forward_one_token(&handle, &weights, &mut c1_state, token_id, &mut logits).unwrap();
        }
        for _ in 0..warmup {
            let next_id = argmax_sample(&logits).unwrap();
            forward_one_token(&handle, &weights, &mut c1_state, next_id, &mut logits).unwrap();
        }
        let timed_tokens = 30usize;
        let t0 = std::time::Instant::now();
        for _ in 0..timed_tokens {
            let next_id = argmax_sample(&logits).unwrap();
            forward_one_token(&handle, &weights, &mut c1_state, next_id, &mut logits).unwrap();
        }
        let c1_ms = t0.elapsed().as_secs_f64() * 1000.0 / timed_tokens as f64;
        eprintln!("C_1 (real steady-state single-token decode): {c1_ms:.4} ms/token");

        // --- C_verify(K): real batched K-token verify-chunk cost, fresh
        // DecodeState per K (warmed up to the same real position as C_1's
        // measurement) to avoid any cross-trial state drift ---
        let repeats = 15usize;
        for &k in &[2usize, 4, 6, 8] {
            let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
            let mut warm_logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
            for &token_id in prompt_ids.iter() {
                forward_one_token(&handle, &weights, &mut state, token_id, &mut warm_logits).unwrap();
            }
            for _ in 0..warmup {
                let next_id = argmax_sample(&warm_logits).unwrap();
                forward_one_token(&handle, &weights, &mut state, next_id, &mut warm_logits).unwrap();
            }

            // Real in-vocab draft token ids -- content doesn't affect real
            // kernel cost (attention/GDN/GEMM cost is shape-bound, not
            // value-bound), only shape does.
            let draft_ids: Vec<i32> = (0..k as i32).map(|i| (100 + i) % VOCAB_SIZE as i32).collect();
            let mut normed_out: DeviceBuffer<u16> = DeviceBuffer::alloc(k * HIDDEN_SIZE).unwrap();
            let mut verify_logits: DeviceBuffer<u16> = DeviceBuffer::alloc(k * VOCAB_SIZE).unwrap();

            let mut times_ms = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let t0 = std::time::Instant::now();

                state.prefill_scratch.token_ids_dev.copy_from_host_prefix(&draft_ids).unwrap();
                let positions: Vec<i32> = (0..k as i32).map(|i| state.position as i32 + i).collect();
                state.prefill_scratch.position_buf.copy_from_host_prefix(&positions).unwrap();

                let final_hidden_ptr = run_layers_over_chunk(handle_raw, &weights, &mut state, k, None, std::ptr::null_mut());
                // SAFETY: points at one of `state.prefill_scratch`'s two
                // live `hidden_a`/`hidden_b` fields, same guarantee
                // `run_prefill_chunk_body` already relies on.
                let final_hidden: &DeviceBuffer<u16> = unsafe { &*final_hidden_ptr };
                unsafe {
                    raw::rmsnorm(
                        final_hidden.as_device_ptr(),
                        weights.final_norm.as_device_ptr(),
                        normed_out.as_device_ptr_mut(),
                        k as i32,
                        HIDDEN_SIZE as i32,
                        RMS_EPS,
                        std::ptr::null_mut(),
                    );
                    match &weights.lm_head {
                        Some(LinearWeight::Bf16(buf)) => {
                            raw::gemm(handle_raw, normed_out.as_device_ptr(), buf.as_device_ptr(), verify_logits.as_device_ptr_mut(), k as i32, HIDDEN_SIZE as i32, VOCAB_SIZE as i32, std::ptr::null_mut());
                        }
                        None => {
                            raw::gemm(handle_raw, normed_out.as_device_ptr(), weights.embed_tokens.as_device_ptr(), verify_logits.as_device_ptr_mut(), k as i32, HIDDEN_SIZE as i32, VOCAB_SIZE as i32, std::ptr::null_mut());
                        }
                        Some(LinearWeight::Quantized { .. }) => unreachable!("checked is_quantized() above"),
                    }
                }
                hip::check_last_error().unwrap();
                hip::device_synchronize().unwrap();
                times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            }
            let avg_ms = times_ms.iter().sum::<f64>() / repeats as f64;
            let tau_breakeven = avg_ms / c1_ms;
            eprintln!(
                "K={k}: C_verify(K)={avg_ms:.4} ms (avg of {repeats} real runs), tau_breakeven = C_verify/C_1 = {tau_breakeven:.3} -- need >{tau_breakeven:.2} real accepted drafted tokens/round just to break even on THIS crate's kernels"
            );
        }
    }

    /// Runs the SAME real verify-chunk math the production
    /// `GraphedVerifyState::verify_chunk` uses, eagerly (null stream, no
    /// capture/replay) -- the independent reference every test below
    /// checks the graphed/bucketed path against. Deliberately duplicated
    /// rather than shared: keeping the eager reference textually separate
    /// from the thing under test is the whole point (see
    /// `diagnose_graphed_decode_sees_lora_activated_after_first_capture`'s
    /// own reasoning in `quantized_lora.rs` for why this crate always
    /// cross-checks a graphed path against a genuinely independent eager
    /// computation, not a refactor of it). `kv_len_override`: `None` for
    /// the TRUE, real, non-padded ground truth (the real bar a bucketed
    /// graph's output must match exactly); `Some(bucket)` to compute the
    /// SAME padded/masked math a real bucket's graph does, eagerly --
    /// useful for isolating a padding-specific bug from a graph-specific
    /// one if the two ever disagree.
    fn eager_verify_chunk(
        weights: &ModelWeights,
        state: &mut DecodeState,
        handle_raw: blas_ffi::HipblasHandle,
        draft_ids: &[i32],
        k: usize,
        kv_len_override: Option<usize>,
        normed_out: &mut DeviceBuffer<u16>,
        verify_logits: &mut DeviceBuffer<u16>,
        raw_hidden_out: Option<&mut DeviceBuffer<u16>>,
    ) {
        state.prefill_scratch.token_ids_dev.copy_from_host_prefix(draft_ids).unwrap();
        let positions: Vec<i32> = (0..k as i32).map(|i| state.position as i32 + i).collect();
        state.prefill_scratch.position_buf.copy_from_host_prefix(&positions).unwrap();
        let final_hidden_ptr = run_layers_over_chunk(handle_raw, weights, state, k, kv_len_override, std::ptr::null_mut());
        let final_hidden: &DeviceBuffer<u16> = unsafe { &*final_hidden_ptr };
        if let Some(raw_out) = raw_hidden_out {
            raw_out.copy_from_device_offset(final_hidden, 0).unwrap();
        }
        unsafe {
            raw::rmsnorm(final_hidden.as_device_ptr(), weights.final_norm.as_device_ptr(), normed_out.as_device_ptr_mut(), k as i32, HIDDEN_SIZE as i32, RMS_EPS, std::ptr::null_mut());
            match &weights.lm_head {
                Some(LinearWeight::Bf16(buf)) => {
                    raw::gemm(handle_raw, normed_out.as_device_ptr(), buf.as_device_ptr(), verify_logits.as_device_ptr_mut(), k as i32, HIDDEN_SIZE as i32, VOCAB_SIZE as i32, std::ptr::null_mut());
                }
                None => {
                    raw::gemm(handle_raw, normed_out.as_device_ptr(), weights.embed_tokens.as_device_ptr(), verify_logits.as_device_ptr_mut(), k as i32, HIDDEN_SIZE as i32, VOCAB_SIZE as i32, std::ptr::null_mut());
                }
                Some(LinearWeight::Quantized { .. }) => panic!("scoped to bf16 dense sizes only"),
            }
        }
        hip::check_last_error().unwrap();
        hip::device_synchronize().unwrap();
    }

    /// Copies `verify_logits` (`[k * VOCAB_SIZE]`) to host and returns the
    /// count of bf16-bit-differing elements against `other`, for the
    /// repeated graphed-vs-eager comparisons below.
    fn count_logit_diffs(a: &DeviceBuffer<u16>, b: &DeviceBuffer<u16>, len: usize) -> usize {
        let mut ah = vec![0u16; len];
        a.copy_to_host(&mut ah).unwrap();
        let mut bh = vec![0u16; len];
        b.copy_to_host(&mut bh).unwrap();
        ah.iter().zip(bh.iter()).filter(|(x, y)| x != y).count()
    }

    /// DECISIVE, §136: does the real, FIXED-bucket `GraphedVerifyState`
    /// (production code, `model.rs`) survive the real risk its unbucketed
    /// predecessor was built to expose and then found broken by
    /// (`docs/DECISIONS.md` §136's own research) -- a real, variable-
    /// amount partial-accept rollback landing the SAME captured bucket's
    /// graph at a DIFFERENT real position than it was captured at? This
    /// is the exact mechanic that broke the original, unbucketed spike
    /// (100% logit divergence, root-caused to `kv_len` being baked into
    /// kernel launch arguments and a host-side kernel-selection branch,
    /// neither re-read on replay).
    ///
    /// Deliberately scoped to `kv_len <= ATTENTION_GEMM_KV_LEN_THRESHOLD`
    /// (128) -- i.e. only the `attention_causal_prefill` custom-kernel
    /// attention path, never the `hipblasGemmEx`-based GEMM-attention
    /// path. A SEPARATE, real, currently-OPEN bug was found in that GEMM
    /// path specifically when its `kv_len` is padded past ~184-192 (see
    /// `diagnose_gemm_attention_path_breaks_when_kv_len_padded_past_a_
    /// real_threshold` below) -- independent of this fix, present even in
    /// plain eager execution with no graph involved at all, so it does
    /// NOT bear on the real risk this test exists to answer. Splitting
    /// these apart keeps this test a clean, trustworthy PASS proving what
    /// is actually proven, rather than bundling a proven-correct result
    /// with a real, separate, unresolved one.
    ///
    /// Every real graphed result is checked against `eager_verify_chunk`
    /// called with `kv_len_override: None` -- the TRUE, non-padded real
    /// math, not a padded reference -- so a passing test proves bucket
    /// padding is not just "consistent with itself" but numerically
    /// IDENTICAL to the real, unpadded ground truth, exactly as
    /// `causal_softmax.hip`'s own position-derived (not kv_len-derived)
    /// masking predicts.
    ///
    /// Real position schedule (K=6 throughout): prefill+decode to P=100
    /// (round 1's real kv_len = 106, bucket 128) -> rollback to P, commit
    /// all 6 drafts real-accepted -> position 106 (round 2's real kv_len =
    /// 112, STILL bucket 128 -- same-bucket replay at a real, DIFFERENT
    /// position than the graph was captured at).
    #[test]
    #[ignore]
    fn real_bucketed_verify_chunk_survives_variable_position_replay_within_a_bucket() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot_path = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot_path).expect("real weight loading failed");
        assert!(!weights.is_quantized(), "this diagnostic assumes a bf16 dense checkpoint (4B/9B)");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        let handle_raw = handle.raw();

        let max_seq_len = 512usize;
        let k = 6usize;

        // Real prefill + real decode to a real position P=100.
        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        for &token_id in prompt_ids.iter() {
            forward_one_token(&handle, &weights, &mut state, token_id, &mut logits).unwrap();
        }
        while state.position < 100 {
            let next_id = argmax_sample(&logits).unwrap();
            forward_one_token(&handle, &weights, &mut state, next_id, &mut logits).unwrap();
        }
        let p = state.position;
        assert_eq!(p, 100, "real position after prefill + decode should land exactly at P=100");
        eprintln!("real position P = {p}");

        let snapshot_p = crate::state_handoff::TensorStateSnapshot::capture(&state).expect("snapshot capture failed");

        let mut graphed = GraphedVerifyState::new(k).expect("GraphedVerifyState::new failed");
        let mut normed_out: DeviceBuffer<u16> = DeviceBuffer::alloc(k * HIDDEN_SIZE).unwrap();
        // §136 real lesson (found by this very test, on its first run):
        // a captured graph's GEMM output pointer is baked in at CAPTURE
        // time, exactly like every other captured address in this crate
        // -- a caller MUST reuse the SAME `verify_logits`/`normed_out`
        // buffers across every real call (matching how
        // `bench_real_graphed_decode_tokens_per_second` reuses ONE
        // `logits` buffer across all its real decode calls), never
        // allocate a fresh one per round expecting a "replay" to notice.
        // Read out via `count_logit_diffs` immediately after each round,
        // before the NEXT round's call overwrites this same buffer.
        let mut graphed_logits: DeviceBuffer<u16> = DeviceBuffer::alloc(k * VOCAB_SIZE).unwrap();
        let mut graphed_raw_hidden: DeviceBuffer<u16> = DeviceBuffer::alloc(k * HIDDEN_SIZE).unwrap();
        let mut eager_normed: DeviceBuffer<u16> = DeviceBuffer::alloc(k * HIDDEN_SIZE).unwrap();
        let mut eager_logits: DeviceBuffer<u16> = DeviceBuffer::alloc(k * VOCAB_SIZE).unwrap();

        // --- ROUND 1: real kv_len = 106 -> bucket 128 (captures it) ---
        let draft_ids_r1: [i32; 6] = [1000, 2000, 3000, 4000, 5000, 6000];
        let real_kv_len_r1 = state.position + k;
        assert_eq!(kv_len_bucket_for(real_kv_len_r1), 128, "round 1 real kv_len ({real_kv_len_r1}) should land in the 128 bucket");
        graphed.verify_chunk(&weights, &mut state, &draft_ids_r1, &mut normed_out, &mut graphed_logits, &mut graphed_raw_hidden).expect("round 1 verify_chunk (capture bucket 128) failed");

        let mut eager_state_r1 = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        snapshot_p.restore(&mut eager_state_r1).expect("restore failed");
        let mut eager_raw_hidden: DeviceBuffer<u16> = DeviceBuffer::alloc(k * HIDDEN_SIZE).unwrap();
        eager_verify_chunk(&weights, &mut eager_state_r1, handle_raw, &draft_ids_r1, k, None, &mut eager_normed, &mut eager_logits, Some(&mut eager_raw_hidden));
        let diff1 = count_logit_diffs(&graphed_logits, &eager_logits, k * VOCAB_SIZE);
        let diff1_raw = count_logit_diffs(&graphed_raw_hidden, &eager_raw_hidden, k * HIDDEN_SIZE);
        eprintln!("Round 1: raw (pre-final-norm) hidden graphed-vs-eager: {diff1_raw}/{} differ (must be 0)", k * HIDDEN_SIZE);
        assert_eq!(diff1_raw, 0, "round 1: the new raw_hidden_out capture inside GraphedVerifyState diverged from the TRUE, unpadded eager raw hidden state -- the new copy_from_device_async addition has a real bug");
        eprintln!("Round 1 (capture bucket 128, real kv_len={real_kv_len_r1}) vs TRUE (unpadded) eager reference: {diff1}/{} logits differ (must be 0)", k * VOCAB_SIZE);
        assert_eq!(diff1, 0, "round 1: bucket-128-padded graph output diverged from the TRUE, unpadded real math");

        // --- Real rollback to P, commit ALL 6 drafts as real accepted
        // tokens -> real position 106, STILL inside bucket 128 ---
        snapshot_p.restore(&mut state).expect("rollback restore failed");
        for &tok in draft_ids_r1.iter() {
            forward_one_token(&handle, &weights, &mut state, tok, &mut logits).unwrap();
        }
        assert_eq!(state.position, p + 6, "real position after committing all 6 accepted drafts should be P+6");

        // --- ROUND 2: SAME bucket (128), DIFFERENT real position (106,
        // not the P=100 the graph was captured at) -- the exact mechanic
        // that broke the original unbucketed spike ---
        let draft_ids_r2: [i32; 6] = [7000, 8000, 9000, 10000, 11000, 12000];
        let real_kv_len_r2 = state.position + k;
        assert_eq!(kv_len_bucket_for(real_kv_len_r2), 128, "round 2 real kv_len ({real_kv_len_r2}) should STILL land in the 128 bucket (same-bucket replay case)");
        graphed.verify_chunk(&weights, &mut state, &draft_ids_r2, &mut normed_out, &mut graphed_logits, &mut graphed_raw_hidden).expect("round 2 verify_chunk (replay bucket 128) failed");

        let mut eager_state_r2 = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        snapshot_p.restore(&mut eager_state_r2).expect("restore failed");
        for &tok in draft_ids_r1.iter() {
            forward_one_token(&handle, &weights, &mut eager_state_r2, tok, &mut logits).unwrap();
        }
        eager_verify_chunk(&weights, &mut eager_state_r2, handle_raw, &draft_ids_r2, k, None, &mut eager_normed, &mut eager_logits, None);
        let diff2 = count_logit_diffs(&graphed_logits, &eager_logits, k * VOCAB_SIZE);
        eprintln!("Round 2 (REPLAY bucket 128 at a real, different position={}) vs TRUE eager reference: {diff2}/{} logits differ (must be 0)", state.position, k * VOCAB_SIZE);
        assert_eq!(diff2, 0, "round 2: bucket-128 graph REPLAY at a real, different position diverged from TRUE eager math -- the original bug is back");

        eprintln!("VERDICT: the bucketed verify-chunk graph survived a real, variable-position replay within a bucket, bit-exact against the true, unpadded real math.");
    }

    /// DIAGNOSTIC, §138 -- root-causing a real divergence
    /// `bench_real_speculative_vs_graphed_decode_on_real_aider_bench_tasks`
    /// (`speculative.rs`) found on the real `bank_account` aider-bench
    /// prompt (298 real tokens): speculative decoding and this engine's
    /// own graphed decode agreed on the first 15 real generated tokens,
    /// then produced DIFFERENT tokens at output position 15 (real
    /// backbone position 313). Unlike `real_speculative_decode_matches_
    /// plain_greedy_decode`'s own toy 5-token prompt (which never left
    /// `state.position` anywhere near 128), a real 298-token prompt is
    /// already past `ATTENTION_GEMM_KV_LEN_THRESHOLD` before generation
    /// even starts -- squarely in the territory §136 already found a
    /// REAL, root-caused, but explicitly UNRESOLVED 1-ULP floating-point
    /// non-associativity in `gemm_pv_bf16`, deliberately left open there
    /// ("whether this actually matters for real speculative decoding...
    /// has NOT been separately measured"). This test measures it
    /// directly: reproduces the exact real context at the divergence
    /// point (prompt + the first 14 real, independently-confirmed-
    /// correct generated tokens, via `GraphedDecodeState` -- the same
    /// trusted reference the serving-gate benchmark's own correctness
    /// check uses), then calls `eager_verify_chunk` TWICE on two
    /// snapshot-restored copies of that IDENTICAL real state: once with
    /// `kv_len_override=None` (the TRUE, unpadded math) and once with
    /// `kv_len_override=Some(512)` (the SAME padded/masked math a real
    /// verify_chunk graph at this kv_len uses -- `kv_len_bucket_for(312 +
    /// 7) = 512`). If the two disagree specifically at row 0 (predicting
    /// what follows the real, already-committed context), and the
    /// PADDED row 0 argmax matches what the real benchmark run actually
    /// produced (wrong) while the UNPADDED row 0 argmax matches
    /// `GraphedDecodeState`'s own real output (right), that is decisive:
    /// the open §136 risk is real, not hypothetical, and it is what
    /// broke the serving-gate benchmark.
    #[test]
    #[ignore]
    fn diagnose_speculative_aider_bench_divergence_root_cause() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        let tokenizer = crate::tokenizer::ChatTokenizer::load(&snapshot).expect("real tokenizer loading failed");

        let prompts_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../benchmarks/runtime/speculative/runtime_next_serving_gate/aider_bench_prompts.json");
        let prompts_json = std::fs::read_to_string(&prompts_path).unwrap_or_else(|e| panic!("failed to read {prompts_path:?}: {e}"));
        let parsed: serde_json::Value = serde_json::from_str(&prompts_json).unwrap();
        let task = parsed["tasks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "bank_account")
            .expect("bank_account task missing from aider_bench_prompts.json");
        let templated = tokenizer.apply_chat_template(&[
            ("system", task["system_prompt"].as_str().unwrap()),
            ("user", task["user_prompt"].as_str().unwrap()),
        ]);
        let prompt_ids = tokenizer.encode(&templated).expect("real tokenizer encode failed");
        eprintln!("real bank_account prompt: {} tokens", prompt_ids.len());

        // Sized well past the largest `KV_LEN_BUCKETS` entry the padded
        // eager call below will address into (512 here) -- the padded
        // path reads/writes KV-cache positions up to the BUCKET size, not
        // just the real content length, so an allocation only sized for
        // the real prompt+few-tokens (a real, first attempt at this test
        // used `prompt_ids.len() + 64` = 362 here and crashed with a real
        // GPU page fault at kv_len_override=Some(512) -- exactly this).
        let max_seq_len = prompt_ids.len() + 1024;

        // Real, trusted reference: 15 real tokens via GraphedDecodeState,
        // matching what the serving-gate benchmark's own baseline arm
        // produced (and independently, this engine's own graphed decode
        // is validated elsewhere against the real Python reference).
        let mut graphed = GraphedDecodeState::new().expect("GraphedDecodeState::new failed");
        let mut ref_state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut ref_logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        for &tok in &prompt_ids {
            graphed.forward_one_token(&weights, &mut ref_state, tok, &mut ref_logits).unwrap();
        }
        let mut ref_tokens: Vec<i32> = Vec::with_capacity(20);
        let mut next_id = argmax_sample(&ref_logits).unwrap();
        for _ in 0..20 {
            ref_tokens.push(next_id);
            graphed.forward_one_token(&weights, &mut ref_state, next_id, &mut ref_logits).unwrap();
            next_id = argmax_sample(&ref_logits).unwrap();
        }
        eprintln!("real reference tokens (positions {}..{}): {ref_tokens:?}", prompt_ids.len(), prompt_ids.len() + 20);

        // §138 UPDATE: after the §138 fix (always rebuild, no more
        // full-accept "no rebuild" branch), a NEW divergence appeared two
        // rounds later than the original one (round 3, pre_pos=313,
        // real position 314 -- the original was round 2, real position
        // 312). Round 3's own INPUT state is clean (round 2 was NOT a
        // full accept, so it already went through the unconditional
        // rebuild) -- so this isolates directly to `verify_chunk`'s own
        // real_argmax computation at THIS position, testing the
        // already-documented, already-open §136 GEMM-attention risk at a
        // SECOND real position, not the GDN-continuation bug (already
        // fixed and separately confirmed absent here).
        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        let handle_raw = handle.raw();
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        for &tok in &prompt_ids {
            forward_one_token(&handle, &weights, &mut state, tok, &mut logits).unwrap();
        }
        for &tok in &ref_tokens[..15] {
            forward_one_token(&handle, &weights, &mut state, tok, &mut logits).unwrap();
        }
        eprintln!("real replay state.position = {} (expect {}, matching round 3's own real pre_pos)", state.position, prompt_ids.len() + 15);

        let snapshot_p = crate::state_handoff::TensorStateSnapshot::capture(&state).expect("snapshot capture failed");

        // chunk = [next_token=ref_tokens[15], ref_tokens[16], + 5 filler
        // real-vocabulary tokens] -- k=7 total width, EXACTLY matching
        // round 3's own real chunk (pre_pos=313=298+15). Row 1 (not row
        // 0) is the one that matters here: real_argmax[1] is round 3's
        // own real bonus-token computation (predicting real position
        // 315, given context through chunk[1]=drafted[0]@314) -- the
        // value the real benchmark run got wrong.
        let chunk_k = 7usize;
        let mut chunk: Vec<i32> = vec![ref_tokens[15], ref_tokens[16]];
        chunk.extend_from_slice(&[100, 200, 300, 400, 500]);
        assert_eq!(chunk.len(), chunk_k);
        let real_kv_len = state.position + chunk_k;
        let bucket = kv_len_bucket_for(real_kv_len);
        eprintln!("real_kv_len = {real_kv_len}, bucket = {bucket}");

        let mut normed_out: DeviceBuffer<u16> = DeviceBuffer::alloc(chunk_k * HIDDEN_SIZE).unwrap();
        let mut unpadded_logits: DeviceBuffer<u16> = DeviceBuffer::alloc(chunk_k * VOCAB_SIZE).unwrap();
        let mut padded_logits: DeviceBuffer<u16> = DeviceBuffer::alloc(chunk_k * VOCAB_SIZE).unwrap();

        let mut unpadded_state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        snapshot_p.restore(&mut unpadded_state).expect("restore failed");
        eager_verify_chunk(&weights, &mut unpadded_state, handle_raw, &chunk, chunk_k, None, &mut normed_out, &mut unpadded_logits, None);
        let unpadded_argmax0 = crate::kernels::argmax_bf16_row(&unpadded_logits, 1, VOCAB_SIZE).unwrap();

        let mut padded_state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        snapshot_p.restore(&mut padded_state).expect("restore failed");
        eager_verify_chunk(&weights, &mut padded_state, handle_raw, &chunk, chunk_k, Some(bucket), &mut normed_out, &mut padded_logits, None);
        let padded_argmax0 = crate::kernels::argmax_bf16_row(&padded_logits, 1, VOCAB_SIZE).unwrap();

        let mut padded_row0: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        padded_row0.copy_from_device_offset(&padded_logits, VOCAB_SIZE).unwrap();
        let mut unpadded_row0: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        unpadded_row0.copy_from_device_offset(&unpadded_logits, VOCAB_SIZE).unwrap();
        let diff_row0 = count_logit_diffs(&padded_row0, &unpadded_row0, VOCAB_SIZE);
        eprintln!("row 1 (predicting real position 315): {diff_row0}/{VOCAB_SIZE} logits differ between padded (bucket {bucket}) and unpadded (TRUE) math");
        eprintln!("unpadded (TRUE) argmax = {unpadded_argmax0}, padded-eager (bucket {bucket}) argmax = {padded_argmax0}");

        // The ACTUAL code path `speculative_decode_round` calls -- a real
        // captured/replayed HIP graph, not the eager-with-override stand-
        // in above. `real_bucketed_verify_chunk_survives_variable_
        // position_replay_within_a_bucket` only ever proved this bit-
        // exact at bucket 128; bucket 512 has never been exercised.
        let mut graphed_verify = GraphedVerifyState::new(chunk_k).expect("GraphedVerifyState::new failed");
        let mut graphed_state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        snapshot_p.restore(&mut graphed_state).expect("restore failed");
        let mut graphed_normed: DeviceBuffer<u16> = DeviceBuffer::alloc(chunk_k * HIDDEN_SIZE).unwrap();
        let mut graphed_logits: DeviceBuffer<u16> = DeviceBuffer::alloc(chunk_k * VOCAB_SIZE).unwrap();
        let mut graphed_raw_hidden: DeviceBuffer<u16> = DeviceBuffer::alloc(chunk_k * HIDDEN_SIZE).unwrap();
        graphed_verify
            .verify_chunk(&weights, &mut graphed_state, &chunk, &mut graphed_normed, &mut graphed_logits, &mut graphed_raw_hidden)
            .expect("real GraphedVerifyState::verify_chunk failed");
        let graphed_argmax0 = crate::kernels::argmax_bf16_row(&graphed_logits, 1, VOCAB_SIZE).unwrap();
        let mut graphed_row0: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        graphed_row0.copy_from_device_offset(&graphed_logits, VOCAB_SIZE).unwrap();
        let diff_graphed_vs_unpadded = count_logit_diffs(&graphed_row0, &unpadded_row0, VOCAB_SIZE);
        eprintln!("row 1: {diff_graphed_vs_unpadded}/{VOCAB_SIZE} logits differ between the REAL captured graph (bucket {bucket}) and unpadded (TRUE) math");
        eprintln!("REAL graphed-verify argmax = {graphed_argmax0}");
        eprintln!("real graphed-decode reference said the true next token is {}", ref_tokens[17]);

        if unpadded_argmax0 == ref_tokens[17] && graphed_argmax0 != ref_tokens[17] {
            eprintln!("VERDICT: CONFIRMED. The unpadded, TRUE math matches the real graphed-decode reference; the REAL captured verify-chunk graph produces a DIFFERENT argmax at bucket {bucket}. This is a real graph-capture bug at this (previously untested) bucket size, not proven-safe by the existing bucket-128-only decisive test.");
        } else if unpadded_argmax0 == graphed_argmax0 && padded_argmax0 == graphed_argmax0 {
            eprintln!("VERDICT: NOT this. The real captured graph, padded-eager math, and unpadded TRUE math all agree at this exact position -- the real benchmark divergence has a different root cause entirely.");
        } else {
            eprintln!("VERDICT: INCONCLUSIVE -- results disagree in a pattern not yet explained. Needs further isolation.");
        }
    }

    /// DIAGNOSTIC, §136 -- a real, currently OPEN, NOT YET UNDERSTOOD bug,
    /// deliberately kept as a non-asserting record (this crate's own
    /// "PARKED" precedent, e.g. §29's own doc comment: "do not retry until
    /// the mechanism is understood") rather than a decisive test, since it
    /// is not yet fixed and a hard-failing test would misrepresent an open
    /// question as a known regression.
    ///
    /// Found while extending `real_bucketed_verify_chunk_survives_
    /// variable_position_replay_within_a_bucket` (above) with a genuine
    /// cross-bucket transition (bucket 128 -> 256, crossing
    /// `ATTENTION_GEMM_KV_LEN_THRESHOLD` and switching from the custom
    /// `attention_causal_prefill` kernel to the `hipblasGemmEx`-based
    /// GEMM-attention path): the GEMM-attention path produces REAL,
    /// substantial numerical divergence when its `kv_len` shape argument
    /// is padded past a real threshold -- bisected directly against an
    /// independent eager (non-graphed) reference at the EXACT same real
    /// position/content, isolating this cleanly from anything graph- or
    /// refactor-related:
    ///
    /// - `eager(None)` (real kv_len=132) vs `eager(Some(140..184))`
    ///   (8-52 padded columns): bit-exact, 0 differ, at every value tried.
    /// - `eager(None)` vs `eager(Some(192))` (60 padded columns): **1.19M
    ///   / 1.49M logits differ** -- and every larger value tried (200,
    ///   232, 256) differs similarly.
    ///
    /// ROOT CAUSE, FULLY CONFIRMED (see `diagnose_gemm_attention_bug_
    /// per_layer_divergence_point` and `diagnose_gemm_attention_bug_with_
    /// real_layer7_data_isolated` below for the full trace): this is NOT
    /// a logic bug, NOT NaN/Inf from uninitialized padding (K/V-cache
    /// padding confirmed zero-initialized), and NOT in `causal_softmax`
    /// (its masking is derived entirely from a device-read real position,
    /// confirmed both by reading the kernel and by direct isolated
    /// testing at this exact shape). It is a REAL, genuine floating-point
    /// non-associativity in `gemm_pv_bf16`'s (`O = P @ V`) `hipblasGemmEx`
    /// reduction over the `kv_len` (K) dimension: `hipblasGemmEx`'s
    /// internal tiling/blocking strategy is chosen based on the TOTAL
    /// `kv_len`, so summing the SAME real, non-zero terms (`P` bit-
    /// identical, confirmed by isolated testing, for the shared valid
    /// range) in a DIFFERENT internal accumulation order at kv_len=184 vs
    /// 192 -- even though every EXTRA padded term is an exact `0.0`
    /// contribution -- rounds to an ADJACENT bf16 value (a 1-ULP
    /// difference) for a small number of real (row, head, dim)
    /// combinations where the true sum happens to sit near a bf16
    /// rounding boundary. Directly confirmed: extracting REAL layer-7 Q/K/
    /// V data and re-running ONLY `gemm_qkt`+`causal_softmax`+`gemm_pv` in
    /// isolation (no model layers) reproduces exactly 3/24,576 elements
    /// differing by 1 ULP each (e.g. bf16 bits `0x38ef` vs `0x38f0`); for
    /// at least one of those three (row=1, head=0), `gemm_qkt`'s AND
    /// `causal_softmax`'s own outputs were independently confirmed
    /// bit-identical between the two runs, leaving `gemm_pv` as the only
    /// possible source for that case. This is normal, expected,
    /// individually-correct BLAS behavior -- bit-exact reproducibility
    /// across DIFFERENT problem sizes is not a guarantee `hipblasGemmEx`
    /// (or BLAS libraries generally) makes, even when the size difference
    /// is mathematically inert. The reason it shows up as "99% of logits
    /// differ" rather than "a few ULPs" is amplification: this tiny,
    /// individually-harmless perturbation at layer 7 propagates through
    /// 24 more real, nonlinear layers and (very likely, not separately
    /// confirmed) flips at least one greedy argmax decision, after which
    /// autoregressive generation is a completely different sequence --
    /// the same well-known sensitivity any two numerically-close-but-not-
    /// identical forward passes of a deep greedy-decoded model have,
    /// unrelated to this specific bug.
    ///
    /// Real, honest consequence: a captured bucket's graph replays
    /// bit-exact WITHIN that bucket forever (kv_len is fixed per bucket,
    /// confirmed by `real_bucketed_verify_chunk_survives_variable_
    /// position_replay_within_a_bucket`'s own passing result) -- the
    /// non-determinism here is specifically about comparing a PADDED
    /// bucket's output against a TRUTH reference computed at the real,
    /// UNPADDED kv_len, which is a stricter bar than ordinary BLAS usage
    /// naturally satisfies. Whether this actually matters for real
    /// speculative decoding depends on whether it ever flips a real
    /// accept/reject decision (real logit gaps between top-1 and runner-
    /// up are typically far larger than 1 bf16 ULP) -- NOT separately
    /// measured here, a real, disclosed open question, not papered over.
    /// `KV_LEN_BUCKETS`' entries above `ATTENTION_GEMM_KV_LEN_THRESHOLD`
    /// (128) remain unvalidated against a bit-exact bar; whether that bar
    /// is even the right one to hold this to is the real next decision.
    #[test]
    #[ignore]
    fn diagnose_gemm_attention_path_breaks_when_kv_len_padded_past_a_real_threshold() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot_path = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot_path).expect("real weight loading failed");
        assert!(!weights.is_quantized(), "this diagnostic assumes a bf16 dense checkpoint (4B/9B)");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        let handle_raw = handle.raw();

        let max_seq_len = 512usize;
        let k = 6usize;

        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        for &token_id in prompt_ids.iter() {
            forward_one_token(&handle, &weights, &mut state, token_id, &mut logits).unwrap();
        }
        while state.position < 100 {
            let next_id = argmax_sample(&logits).unwrap();
            forward_one_token(&handle, &weights, &mut state, next_id, &mut logits).unwrap();
        }
        let p = state.position;
        let snapshot_p = crate::state_handoff::TensorStateSnapshot::capture(&state).expect("snapshot capture failed");

        let draft_ids_r1: [i32; 6] = [1000, 2000, 3000, 4000, 5000, 6000];
        snapshot_p.restore(&mut state).expect("rollback restore failed");
        for &tok in draft_ids_r1.iter() {
            forward_one_token(&handle, &weights, &mut state, tok, &mut logits).unwrap();
        }
        for _ in 0..20 {
            let next_id = argmax_sample(&logits).unwrap();
            forward_one_token(&handle, &weights, &mut state, next_id, &mut logits).unwrap();
        }
        assert_eq!(state.position, p + 26, "real position after 20 more real decode steps should be P+26");

        // Real kv_len = 132 here (would cross into bucket 256 if this
        // were driven through `GraphedVerifyState` -- deliberately NOT
        // done here, see this test's own doc comment for why: the graph
        // is not the variable under test, the eager GEMM-attention math
        // itself already diverges, isolated below with zero graph
        // involvement).
        let draft_ids_r3: [i32; 6] = [13000, 14000, 15000, 16000, 17000, 18000];
        let real_kv_len_r3 = state.position + k;
        assert_eq!(kv_len_bucket_for(real_kv_len_r3), 256, "round 3 real kv_len ({real_kv_len_r3}) should have crossed into the 256 bucket");

        let mut eager_normed: DeviceBuffer<u16> = DeviceBuffer::alloc(k * HIDDEN_SIZE).unwrap();
        let mut eager_logits: DeviceBuffer<u16> = DeviceBuffer::alloc(k * VOCAB_SIZE).unwrap();
        eager_verify_chunk(&weights, &mut state, handle_raw, &draft_ids_r3, k, None, &mut eager_normed, &mut eager_logits, None);

        // DEBUG: minimal padding (real 132 -> forced 140, only 8 extra
        // columns) vs the exact-value reference -- isolates whether ANY
        // padding at all breaks the GEMM-attention path, or only large
        // amounts.
        let mut eager_state_r3c = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        snapshot_p.restore(&mut eager_state_r3c).expect("restore failed");
        for &tok in draft_ids_r1.iter() {
            forward_one_token(&handle, &weights, &mut eager_state_r3c, tok, &mut logits).unwrap();
        }
        for _ in 0..20 {
            let next_id = argmax_sample(&logits).unwrap();
            forward_one_token(&handle, &weights, &mut eager_state_r3c, next_id, &mut logits).unwrap();
        }
        let mut eager_normed_minipad: DeviceBuffer<u16> = DeviceBuffer::alloc(k * HIDDEN_SIZE).unwrap();
        let mut eager_logits_minipad: DeviceBuffer<u16> = DeviceBuffer::alloc(k * VOCAB_SIZE).unwrap();
        eager_verify_chunk(&weights, &mut eager_state_r3c, handle_raw, &draft_ids_r3, k, Some(140), &mut eager_normed_minipad, &mut eager_logits_minipad, None);
        let diff3_minipad = count_logit_diffs(&eager_logits, &eager_logits_minipad, k * VOCAB_SIZE);
        eprintln!("DEBUG round 3: eager(None, real=132) vs eager(Some(140), 8 extra padded columns): {diff3_minipad}/{} differ", k * VOCAB_SIZE);

        // DEBUG: bisect the padding amount that breaks it.
        for &bucket_probe in &[160usize, 168, 176, 184, 192, 200, 232, 256] {
            let mut eager_state_probe = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
            snapshot_p.restore(&mut eager_state_probe).expect("restore failed");
            for &tok in draft_ids_r1.iter() {
                forward_one_token(&handle, &weights, &mut eager_state_probe, tok, &mut logits).unwrap();
            }
            for _ in 0..20 {
                let next_id = argmax_sample(&logits).unwrap();
                forward_one_token(&handle, &weights, &mut eager_state_probe, next_id, &mut logits).unwrap();
            }
            let mut eager_normed_probe: DeviceBuffer<u16> = DeviceBuffer::alloc(k * HIDDEN_SIZE).unwrap();
            let mut eager_logits_probe: DeviceBuffer<u16> = DeviceBuffer::alloc(k * VOCAB_SIZE).unwrap();
            eager_verify_chunk(&weights, &mut eager_state_probe, handle_raw, &draft_ids_r3, k, Some(bucket_probe), &mut eager_normed_probe, &mut eager_logits_probe, None);
            let diff_probe = count_logit_diffs(&eager_logits, &eager_logits_probe, k * VOCAB_SIZE);
            eprintln!("DEBUG round 3 bisect: eager(None, real=132) vs eager(Some({bucket_probe})): {diff_probe}/{} differ", k * VOCAB_SIZE);
        }

        eprintln!(
            "VERDICT: real kv_len={real_kv_len_r3} (bucket 256) -- ROOT CAUSE CONFIRMED (see `diagnose_gemm_attention_bug_with_real_layer7_data_isolated`): gemm_pv_bf16's hipBLAS reduction rounds a small number of real elements to an adjacent bf16 value (1 ULP) when kv_len is padded past ~184-192, due to a real, size-dependent internal accumulation-order change -- normal BLAS non-associativity, not a logic bug -- amplified by 24 downstream layers into a fully different generation. KV_LEN_BUCKETS entries above ATTENTION_GEMM_KV_LEN_THRESHOLD (128) are not bit-exact against an unpadded reference; whether that's the right bar to hold them to is the real open question."
        );
    }

    /// §136 root-cause, continued: `gemm_qkt_bf16`/`causal_softmax_bf16`/
    /// `gemm_pv_bf16` were each independently swept across the exact real
    /// kv_len boundary (176-256) AND reproduced as a full, realistic
    /// 16-head GQA loop with real offsets into a shared cache buffer
    /// (`blas::tests::real_multi_head_gqa_attention_loop_at_real_shape_
    /// matches_reference_across_kv_len_sweep`) -- all bit-clean. So the
    /// bug is NOT in the attention primitives themselves. This test goes
    /// back to the REAL model (real weights, real position, real Q/K/V
    /// magnitudes -- the one variable the synthetic isolation tests above
    /// couldn't cover) and checks EVERY one of the 32 real layers' hidden-
    /// state output individually, `Some(184)` vs `Some(192)`, to find the
    /// FIRST layer where they diverge -- turning "the whole stack ends up
    /// 99.7% different" into "layer N is where it starts."
    #[test]
    #[ignore]
    fn diagnose_gemm_attention_bug_per_layer_divergence_point() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot_path = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot_path).expect("real weight loading failed");
        assert!(!weights.is_quantized(), "this diagnostic assumes a bf16 dense checkpoint (4B/9B)");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        let handle_raw = handle.raw();

        let max_seq_len = 512usize;
        let k = 6usize;

        // Real setup, matching §136's own round-3 scenario exactly:
        // prefill + decode to P=100, commit 6 real accepted drafts, 20
        // more real decode steps -> real position 126.
        let build_state = || -> DecodeState {
            let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
            let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
            let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
            for &token_id in prompt_ids.iter() {
                forward_one_token(&handle, &weights, &mut state, token_id, &mut logits).unwrap();
            }
            while state.position < 100 {
                let next_id = argmax_sample(&logits).unwrap();
                forward_one_token(&handle, &weights, &mut state, next_id, &mut logits).unwrap();
            }
            let draft_ids_r1: [i32; 6] = [1000, 2000, 3000, 4000, 5000, 6000];
            for &tok in draft_ids_r1.iter() {
                forward_one_token(&handle, &weights, &mut state, tok, &mut logits).unwrap();
            }
            for _ in 0..20 {
                let next_id = argmax_sample(&logits).unwrap();
                forward_one_token(&handle, &weights, &mut state, next_id, &mut logits).unwrap();
            }
            assert_eq!(state.position, 126, "real position after the real §136 setup should be 126");
            state
        };

        let mut state_184 = build_state();
        let mut state_192 = build_state();

        let draft_ids_r3: [i32; 6] = [13000, 14000, 15000, 16000, 17000, 18000];
        for state in [&mut state_184, &mut state_192] {
            state.prefill_scratch.token_ids_dev.copy_from_host_prefix(&draft_ids_r3).unwrap();
            let positions: Vec<i32> = (0..k as i32).map(|i| state.position as i32 + i).collect();
            state.prefill_scratch.position_buf.copy_from_host_prefix(&positions).unwrap();
        }

        // Manual per-layer loop, mirroring `run_layers_over_chunk`'s own
        // structure exactly, but comparing hidden state after EVERY layer
        // between the two real, otherwise-identical runs.
        let mut use_a_184 = true;
        let mut use_a_192 = true;
        unsafe {
            raw::embedding_lookup(weights.embed_tokens.as_device_ptr(), state_184.prefill_scratch.token_ids_dev.as_device_ptr() as *const i32, state_184.prefill_scratch.hidden_a.as_device_ptr_mut(), k as i32, HIDDEN_SIZE as i32, std::ptr::null_mut());
            raw::embedding_lookup(weights.embed_tokens.as_device_ptr(), state_192.prefill_scratch.token_ids_dev.as_device_ptr() as *const i32, state_192.prefill_scratch.hidden_a.as_device_ptr_mut(), k as i32, HIDDEN_SIZE as i32, std::ptr::null_mut());
        }

        let mut first_diverging_layer: Option<usize> = None;
        for (i, layer_weights) in weights.layers.iter().enumerate() {
            let (hidden_in_184, hidden_out_184): (*const DeviceBuffer<u16>, *mut DeviceBuffer<u16>) =
                if use_a_184 { (&state_184.prefill_scratch.hidden_a, &mut state_184.prefill_scratch.hidden_b) } else { (&state_184.prefill_scratch.hidden_b, &mut state_184.prefill_scratch.hidden_a) };
            let (hidden_in_192, hidden_out_192): (*const DeviceBuffer<u16>, *mut DeviceBuffer<u16>) =
                if use_a_192 { (&state_192.prefill_scratch.hidden_a, &mut state_192.prefill_scratch.hidden_b) } else { (&state_192.prefill_scratch.hidden_b, &mut state_192.prefill_scratch.hidden_a) };

            match (layer_weights, &mut state_184.layers[i], &mut state_192.layers[i]) {
                (LayerWeights::Gdn(w), LayerState::Gdn(gs184), LayerState::Gdn(gs192)) => unsafe {
                    gdn_layer_forward_prefill(handle_raw, &*hidden_in_184, &mut *hidden_out_184, w, gs184, k, &mut state_184.prefill_scratch, std::ptr::null_mut());
                    gdn_layer_forward_prefill(handle_raw, &*hidden_in_192, &mut *hidden_out_192, w, gs192, k, &mut state_192.prefill_scratch, std::ptr::null_mut());
                },
                (LayerWeights::Attn(w), LayerState::Attn(as184), LayerState::Attn(as192)) => unsafe {
                    attn_layer_forward_prefill(handle_raw, &*hidden_in_184, &mut *hidden_out_184, w, as184, k, 184, max_seq_len, &mut state_184.prefill_scratch, std::ptr::null_mut());
                    attn_layer_forward_prefill(handle_raw, &*hidden_in_192, &mut *hidden_out_192, w, as192, k, 192, max_seq_len, &mut state_192.prefill_scratch, std::ptr::null_mut());
                },
                _ => unreachable!("layer weights/state type mismatch at index {i}"),
            }
            use_a_184 = !use_a_184;
            use_a_192 = !use_a_192;

            hip::device_synchronize().unwrap();
            let out_184: &DeviceBuffer<u16> = unsafe { &*hidden_out_184 };
            let out_192: &DeviceBuffer<u16> = unsafe { &*hidden_out_192 };
            let mut h184_full = vec![0u16; out_184.len()];
            out_184.copy_to_host(&mut h184_full).unwrap();
            let mut h192_full = vec![0u16; out_192.len()];
            out_192.copy_to_host(&mut h192_full).unwrap();
            let diff = h184_full[..k * HIDDEN_SIZE].iter().zip(h192_full[..k * HIDDEN_SIZE].iter()).filter(|(a, b)| a != b).count();
            let layer_kind = if is_full_attention_layer(i) { "Attn" } else { "Gdn" };
            eprintln!("layer {i:2} ({layer_kind}): {diff}/{} elements differ (184 vs 192)", k * HIDDEN_SIZE);
            if diff > 0 && first_diverging_layer.is_none() {
                first_diverging_layer = Some(i);
            }
        }

        match first_diverging_layer {
            Some(i) => eprintln!("VERDICT: first diverging layer is {i} ({})", if is_full_attention_layer(i) { "Attn" } else { "Gdn" }),
            None => eprintln!("VERDICT: no layer diverged?! (184 vs 192 matched at every layer -- contradicts the earlier full-model finding, worth re-checking)"),
        }
    }

    /// §136 root-cause, FINAL: `diagnose_gemm_attention_bug_per_layer_
    /// divergence_point` found layer 7 (the SECOND real attention layer)
    /// is where 184-vs-192 first diverges, while layer 3 (the FIRST) is
    /// bit-exact -- with IDENTICAL upstream input to both (layers 4-6 are
    /// themselves bit-exact). Since every synthetic isolation test above
    /// used the SAME shape and STILL came back clean, the remaining real
    /// difference is layer 7's own REAL, LEARNED weight values -- this
    /// test extracts them directly (real `attn_query_roped`, real
    /// `k_cache`/`v_cache`, all real bf16 bytes, straight off a REAL,
    /// CORRECT kv_len=184 call to layer 7) and re-runs ONLY the isolated
    /// GEMM/softmax math against them at both 184 and 192, with no model
    /// layers, no o_proj, no sigmoid_gate, no residual add involved.
    ///
    /// If this diverges: the bug is real-data-triggered inside
    /// `gemm_qkt`/`causal_softmax`/`gemm_pv` themselves (a genuine
    /// hipBLAS/kernel numerics bug specific to this data, not just this
    /// shape) -- the earlier synthetic sweeps simply didn't hit the
    /// triggering value pattern.
    /// If this does NOT diverge: the bug is somewhere else entirely in
    /// `attn_layer_forward_prefill` (sigmoid_gate, o_proj, the residual
    /// add) or in how layer 7's REAL kv_len=192 call diverges from its
    /// own kv_len=184 call in some way this extraction (taken from the
    /// 184 call) can't capture -- pointing the remaining search
    /// elsewhere.
    #[test]
    #[ignore]
    fn diagnose_gemm_attention_bug_with_real_layer7_data_isolated() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot_path = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot_path).expect("real weight loading failed");
        assert!(!weights.is_quantized(), "this diagnostic assumes a bf16 dense checkpoint (4B/9B)");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        let handle_raw = handle.raw();

        let max_seq_len = 512usize;
        let k = 6usize;

        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        for &token_id in prompt_ids.iter() {
            forward_one_token(&handle, &weights, &mut state, token_id, &mut logits).unwrap();
        }
        while state.position < 100 {
            let next_id = argmax_sample(&logits).unwrap();
            forward_one_token(&handle, &weights, &mut state, next_id, &mut logits).unwrap();
        }
        let draft_ids_r1: [i32; 6] = [1000, 2000, 3000, 4000, 5000, 6000];
        for &tok in draft_ids_r1.iter() {
            forward_one_token(&handle, &weights, &mut state, tok, &mut logits).unwrap();
        }
        for _ in 0..20 {
            let next_id = argmax_sample(&logits).unwrap();
            forward_one_token(&handle, &weights, &mut state, next_id, &mut logits).unwrap();
        }
        assert_eq!(state.position, 126);

        let draft_ids_r3: [i32; 6] = [13000, 14000, 15000, 16000, 17000, 18000];
        state.prefill_scratch.token_ids_dev.copy_from_host_prefix(&draft_ids_r3).unwrap();
        let positions: Vec<i32> = (0..k as i32).map(|i| state.position as i32 + i).collect();
        state.prefill_scratch.position_buf.copy_from_host_prefix(&positions).unwrap();

        // Run the REAL, correct layer stack (kv_len=184, a call already
        // proven correct) up through and including layer 7, leaving
        // `state.prefill_scratch.attn_query_roped` and
        // `state.layers[7]`'s real k_cache/v_cache populated with real,
        // correct, layer-7-specific data.
        unsafe {
            raw::embedding_lookup(weights.embed_tokens.as_device_ptr(), state.prefill_scratch.token_ids_dev.as_device_ptr() as *const i32, state.prefill_scratch.hidden_a.as_device_ptr_mut(), k as i32, HIDDEN_SIZE as i32, std::ptr::null_mut());
        }
        let mut use_a = true;
        for (i, layer_weights) in weights.layers.iter().enumerate().take(8) {
            let (hidden_in, hidden_out): (*const DeviceBuffer<u16>, *mut DeviceBuffer<u16>) =
                if use_a { (&state.prefill_scratch.hidden_a, &mut state.prefill_scratch.hidden_b) } else { (&state.prefill_scratch.hidden_b, &mut state.prefill_scratch.hidden_a) };
            match (layer_weights, &mut state.layers[i]) {
                (LayerWeights::Gdn(w), LayerState::Gdn(gs)) => unsafe {
                    gdn_layer_forward_prefill(handle_raw, &*hidden_in, &mut *hidden_out, w, gs, k, &mut state.prefill_scratch, std::ptr::null_mut());
                },
                (LayerWeights::Attn(w), LayerState::Attn(as_)) => unsafe {
                    attn_layer_forward_prefill(handle_raw, &*hidden_in, &mut *hidden_out, w, as_, k, 184, max_seq_len, &mut state.prefill_scratch, std::ptr::null_mut());
                },
                _ => unreachable!(),
            }
            use_a = !use_a;
            hip::device_synchronize().unwrap();
        }

        let num_q_heads = ATTN_NUM_HEADS;
        let num_kv_heads = ATTN_NUM_KV_HEADS;
        let head_dim = ATTN_HEAD_DIM;
        let q_row_len = num_q_heads * head_dim;
        let real_len = 132usize;
        let start_position = 126i32;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // Extract REAL, layer-7-derived bf16 bytes directly off the
        // device -- no synthetic data anywhere in this test. Only the
        // real, written `[k, q_row_len]` prefix of `attn_query_roped`'s
        // full `MAX_PREFILL_CHUNK`-sized scratch capacity is meaningful.
        let mut q_roped_full = vec![0u16; state.prefill_scratch.attn_query_roped.len()];
        state.prefill_scratch.attn_query_roped.copy_to_host(&mut q_roped_full).unwrap();
        let q_roped_real = q_roped_full[..k * ATTN_NUM_HEADS * ATTN_HEAD_DIM].to_vec();

        let layer7_state = match &state.layers[7] {
            LayerState::Attn(a) => a,
            _ => unreachable!("layer 7 must be an attention layer"),
        };
        let mut k_cache_real = vec![0u16; layer7_state.k_cache.len()];
        layer7_state.k_cache.copy_to_host(&mut k_cache_real).unwrap();
        let mut v_cache_real = vec![0u16; layer7_state.v_cache.len()];
        layer7_state.v_cache.copy_to_host(&mut v_cache_real).unwrap();

        // Re-upload the SAME real bytes into fresh, isolated buffers, and
        // re-run ONLY the GEMM/softmax attention math -- no model layers,
        // no o_proj, no sigmoid_gate.
        let mut q_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(q_roped_real.len()).unwrap();
        q_buf.copy_from_host(&q_roped_real).unwrap();
        let mut k_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(k_cache_real.len()).unwrap();
        k_buf.copy_from_host(&k_cache_real).unwrap();
        let mut v_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(v_cache_real.len()).unwrap();
        v_buf.copy_from_host(&v_cache_real).unwrap();

        let n_rep = num_q_heads / num_kv_heads;
        let mut outputs: Vec<Vec<u16>> = Vec::new();
        // Per-stage capture for h=0 only (cheap, sufficient to localize
        // which of the two GEMM calls / the softmax first disagrees):
        // raw scores right after gemm_qkt (pre-softmax) and right after
        // causal_softmax (pre-gemm_pv), for both kv_len values.
        let mut qkt_raw_h0: Vec<Vec<u16>> = Vec::new();
        let mut softmax_h0: Vec<Vec<u16>> = Vec::new();
        for &kv_len in &[184usize, 192] {
            let mut attn_scores: DeviceBuffer<u16> = DeviceBuffer::alloc(k * kv_len).unwrap();
            let mut attn_out: DeviceBuffer<u16> = DeviceBuffer::alloc(k * q_row_len).unwrap();
            attn_out.copy_from_host(&vec![0u16; k * q_row_len]).unwrap();
            for h in 0..num_q_heads {
                let h_kv = h / n_rep;
                unsafe {
                    raw::gemm_qkt(handle_raw, q_buf.as_device_ptr_at(h * head_dim), q_row_len as i32, k_buf.as_device_ptr_at(h_kv * max_seq_len * head_dim), attn_scores.as_device_ptr_mut(), kv_len as i32, k as i32, head_dim as i32, kv_len as i32, scale);
                }
                if h == 0 {
                    hip::device_synchronize().unwrap();
                    let mut raw_host = vec![0u16; k * kv_len];
                    attn_scores.copy_to_host(&mut raw_host).unwrap();
                    qkt_raw_h0.push(raw_host);
                }
                unsafe {
                    let start_position_ptr = state.prefill_scratch.position_buf.as_device_ptr_at(0) as *const i32;
                    raw::causal_softmax(attn_scores.as_device_ptr_mut() as *mut c_void, kv_len as i32, start_position_ptr, k as i32, std::ptr::null_mut());
                }
                if h == 0 {
                    hip::device_synchronize().unwrap();
                    let mut sm_host = vec![0u16; k * kv_len];
                    attn_scores.copy_to_host(&mut sm_host).unwrap();
                    softmax_h0.push(sm_host);
                }
                unsafe {
                    raw::gemm_pv(handle_raw, attn_scores.as_device_ptr(), kv_len as i32, v_buf.as_device_ptr_at(h_kv * max_seq_len * head_dim), attn_out.as_device_ptr_at_mut(h * head_dim), q_row_len as i32, k as i32, kv_len as i32, head_dim as i32, 1.0, 0.0);
                }
            }
            hip::device_synchronize().unwrap();
            let mut out_host = vec![0u16; k * q_row_len];
            attn_out.copy_to_host(&mut out_host).unwrap();
            outputs.push(out_host);
        }

        // Compare h=0's valid-range columns only (kv_len differs between
        // the two runs, so only the shared, real [0, valid_len) prefix
        // per row is a meaningful comparison).
        let (kv184, kv192) = (184usize, 192usize);
        let mut qkt_diff = 0usize;
        let mut sm_diff = 0usize;
        for r in 0..k {
            let valid_len = (start_position as usize) + r + 1;
            for j in 0..valid_len {
                if qkt_raw_h0[0][r * kv184 + j] != qkt_raw_h0[1][r * kv192 + j] {
                    qkt_diff += 1;
                }
                if softmax_h0[0][r * kv184 + j] != softmax_h0[1][r * kv192 + j] {
                    sm_diff += 1;
                }
            }
        }
        eprintln!("h=0 stage-by-stage (valid-range columns only): post-gemm_qkt diff={qkt_diff}, post-causal_softmax diff={sm_diff}");

        for r in 0..k {
            for h in 0..num_q_heads {
                for d in 0..head_dim {
                    let idx = r * q_row_len + h * head_dim + d;
                    if outputs[0][idx] != outputs[1][idx] {
                        let a = bf16_to_f32(outputs[0][idx]);
                        let b = bf16_to_f32(outputs[1][idx]);
                        eprintln!("  differing element: row={r} head={h} dim={d}: kv_len=184 -> {a} (bits {:#06x}), kv_len=192 -> {b} (bits {:#06x}), diff={}", outputs[0][idx], outputs[1][idx], (a - b).abs());
                    }
                }
            }
        }

        let diff = outputs[0].iter().zip(outputs[1].iter()).filter(|(a, b)| a != b).count();
        eprintln!("isolated GEMM attention math with REAL layer-7 data (real_len={real_len}, start_position={start_position}): 184 vs 192 -> {diff}/{} elements differ", k * q_row_len);
        if diff > 0 {
            eprintln!("VERDICT: REPRODUCED with real layer-7 data + isolated GEMM math alone -- the bug IS a real-data-triggered hipBLAS/kernel numerics issue at this shape, not something in the surrounding model code.");
        } else {
            eprintln!("VERDICT: NOT reproduced with real layer-7 data + isolated GEMM math alone -- the bug must be elsewhere (o_proj, sigmoid_gate, the residual add, or something about the REAL kv_len=192 call's own k_cache/v_cache content differing from what a kv_len=184 call produces).");
        }
    }

    /// DIAGNOSTIC, not a reported number: syncs after every layer (and
    /// after the final norm+lm_head) to measure where the real per-token
    /// time actually goes, broken down by GDN-layer-average vs.
    /// attn-layer-average vs. the final projection -- inserting a sync
    /// per layer necessarily inflates the absolute total well above the
    /// real pipelined number, but the RELATIVE proportions between
    /// buckets stay informative, and no ROCm profiler (rocprof/rocprofv2)
    /// is installed on this machine to get a real per-kernel trace instead.
    #[test]
    #[ignore]
    fn diagnose_real_per_layer_type_cost() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        let max_seq_len = 64usize;
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");

        // Prime the KV cache/state with a short prefill (position matters
        // for attention layers' kv_len).
        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        for &token_id in prompt_ids.iter() {
            forward_one_token(&handle, &weights, &mut state, token_id, &mut logits).unwrap();
        }

        let iters = 15usize;
        let mut gdn_total = std::time::Duration::ZERO;
        let mut gdn_count = 0u32;
        let mut attn_total = std::time::Duration::ZERO;
        let mut attn_count = 0u32;
        let mut final_total = std::time::Duration::ZERO;
        let mut embed_total = std::time::Duration::ZERO;

        for _ in 0..iters {
            let token_id = argmax_sample(&logits).unwrap();
            let token_ids_host = [token_id];
            state.scratch.token_ids_dev.copy_from_host(&token_ids_host).unwrap();
            state
                .scratch
                .position_buf
                .copy_from_host(&[state.position as i32])
                .unwrap();
            let position_ptr = state.scratch.position_buf.as_device_ptr() as *const i32;

            let t_embed = std::time::Instant::now();
            unsafe {
                raw::embedding_lookup(
                    weights.embed_tokens.as_device_ptr(),
                    state.scratch.token_ids_dev.as_device_ptr() as *const i32,
                    state.hidden_a.as_device_ptr_mut(),
                    1,
                    HIDDEN_SIZE as i32,
                    std::ptr::null_mut(),
                );
            }
            hip::device_synchronize().unwrap();
            embed_total += t_embed.elapsed();

            let mut use_a_as_input = true;
            for (i, layer_weights) in weights.layers.iter().enumerate() {
                let layer_state = &mut state.layers[i];
                let (hidden_in, hidden_out): (&DeviceBuffer<u16>, &mut DeviceBuffer<u16>) = if use_a_as_input {
                    (&state.hidden_a, &mut state.hidden_b)
                } else {
                    (&state.hidden_b, &mut state.hidden_a)
                };
                let t_layer = std::time::Instant::now();
                match (layer_weights, layer_state) {
                    (LayerWeights::Gdn(w), LayerState::Gdn(gs)) => {
                        gdn_layer_forward(hidden_in, hidden_out, w, gs, &mut state.scratch, std::ptr::null_mut());
                        hip::device_synchronize().unwrap();
                        gdn_total += t_layer.elapsed();
                        gdn_count += 1;
                    }
                    (LayerWeights::Attn(w), LayerState::Attn(as_)) => {
                        attn_layer_forward(hidden_in, hidden_out, w, as_, position_ptr, state.max_seq_len, &mut state.scratch, std::ptr::null_mut());
                        hip::device_synchronize().unwrap();
                        attn_total += t_layer.elapsed();
                        attn_count += 1;
                    }
                    _ => unreachable!(),
                }
                use_a_as_input = !use_a_as_input;
            }
            let final_hidden = if use_a_as_input { &state.hidden_a } else { &state.hidden_b };

            let t_final = std::time::Instant::now();
            unsafe {
                raw::rmsnorm(final_hidden.as_device_ptr(), weights.final_norm.as_device_ptr(), state.scratch.final_normed.as_device_ptr_mut(), 1, HIDDEN_SIZE as i32, RMS_EPS, std::ptr::null_mut());
                weights.lm_head_apply(state.scratch.final_normed.as_device_ptr(), logits.as_device_ptr_mut(), std::ptr::null_mut());
            }
            hip::device_synchronize().unwrap();
            final_total += t_final.elapsed();
            state.position += 1;
        }

        let gdn_avg_us = gdn_total.as_secs_f64() * 1e6 / gdn_count as f64;
        let attn_avg_us = attn_total.as_secs_f64() * 1e6 / attn_count as f64;
        let final_avg_us = final_total.as_secs_f64() * 1e6 / iters as f64;
        let embed_avg_us = embed_total.as_secs_f64() * 1e6 / iters as f64;
        let reconstructed_total_ms = (gdn_avg_us * 24.0 + attn_avg_us * 8.0 + final_avg_us + embed_avg_us) / 1000.0;
        println!(
            "DIAGNOSTIC (per-layer synced, inflated absolute numbers, informative RATIOS only):\n\
             GDN layer avg:  {gdn_avg_us:.2} us/layer  (x24 = {:.3} ms)\n\
             Attn layer avg: {attn_avg_us:.2} us/layer  (x8 = {:.3} ms)\n\
             Final norm+lm_head avg: {final_avg_us:.2} us\n\
             Embedding lookup avg: {embed_avg_us:.2} us\n\
             Reconstructed per-token total (fully synced): {reconstructed_total_ms:.3} ms",
            gdn_avg_us * 24.0 / 1000.0,
            attn_avg_us * 8.0 / 1000.0,
        );
    }

    #[test]
    #[ignore]
    fn test_multi_chunk_prefill_repro() {
        let snapshot = crate::model_loader::locate_model_snapshot().expect("snapshot");
        let weights = ModelWeights::load(&snapshot).expect("weights");
        let max_seq_len = 4096;
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let handle = BlasHandle::create().expect("handle");
        let mut logits = DeviceBuffer::alloc(VOCAB_SIZE).expect("logits");
        let prompt_ids: Vec<i32> = (0..2500).map(|i| (i % 1000) + 10).collect();
        forward_prefill(&handle, &weights, &mut state, &prompt_ids, &mut logits).expect("forward_prefill failed");
        let mut graphed = GraphedDecodeState::new().expect("graphed");
        for _ in 0..10 {
            let next_token = argmax_sample(&logits).expect("argmax");
            graphed.forward_one_token(&weights, &mut state, next_token, &mut logits).expect("forward_one_token failed");
        }
    }
}
