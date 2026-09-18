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

    /// §108: batched W4A16 prefill -- the `rows>1` analogue of
    /// `linear_quantized` above, real fix for the disclosed §106 TTFT gap
    /// (quantized prefill no longer loops `linear_quantized` once per
    /// token). See `src/kernels/w4a16_gemm_prefill.hip`.
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
        let threads = crate::kernels::w4a16_prefill_threads(in_features as usize);
        unsafe {
            kernels_ffi::launch_w4a16_gemm_prefill_bf16(x, qweight, scales, y, out_features, in_features, group_size, num_tokens, threads, stream);
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
    #[allow(clippy::too_many_arguments)]
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
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn causal_conv1d_prefill(
        src: *const c_void,
        src_stride: i32,
        src_offset: i32,
        conv_state: *mut c_void,
        weight: *const c_void,
        out: *mut c_void,
        num_tokens: i32,
        real_num_tokens_ptr: *const c_void,
        conv_dim: i32,
        kernel_size: i32,
        stream: *mut c_void,
    ) {
        let threads = 256;
        unsafe { kernels_ffi::launch_causal_conv1d_prefill_bf16(src, src_stride, src_offset, conv_state, weight, out, num_tokens, real_num_tokens_ptr, conv_dim, kernel_size, threads, stream) };
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
    pub unsafe fn scale_bf16_by_device_scalar(buf: *mut c_void, scalar_ptr: *const c_void, n: i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_scale_bf16_by_device_scalar(buf, scalar_ptr, n, stream) };
    }

    /// §117: real, unsynced, unconditional (no host-side branch) zero-fill
    /// of `buf[real_num_tokens*row_width .. num_rows*row_width)` -- see
    /// `kernels::zero_from_device_offset_bf16`/`_f32`'s own doc comments
    /// for why this replaces a host-side `if`+offset.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn zero_from_device_offset_bf16(buf: *mut c_void, real_num_tokens_ptr: *const c_void, num_rows: i32, row_width: i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_zero_from_device_offset_bf16(buf, real_num_tokens_ptr, num_rows, row_width, stream) };
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn zero_from_device_offset_f32(buf: *mut c_void, real_num_tokens_ptr: *const c_void, num_rows: i32, row_width: i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_zero_from_device_offset_f32(buf, real_num_tokens_ptr, num_rows, row_width, stream) };
    }

    /// §117: real, unsynced device-side row-select -- see
    /// `kernels::gather_last_real_row_bf16`'s own doc comment.
    pub unsafe fn gather_last_real_row_bf16(src: *const c_void, real_num_tokens_ptr: *const c_void, dst: *mut c_void, row_width: i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_gather_last_real_row_bf16(src, real_num_tokens_ptr, dst, row_width, stream) };
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
    },
}

impl LinearWeight {
    /// Dispatches to the real bf16 GEMV or the real W4A16 fused
    /// dequant+GEMV kernel, whichever this weight actually is -- the
    /// single real call site every one of this engine's hot-path linear
    /// layers goes through now, decode-only (`rows` is always 1 for
    /// every real caller; prefill for a quantized layer uses this same
    /// per-token path in a loop, not a batched GEMM -- see
    /// `docs/DECISIONS.md` §106).
    pub unsafe fn apply(&self, x: *const c_void, y: *mut c_void, in_features: i32, out_features: i32, stream: *mut c_void) {
        match self {
            LinearWeight::Bf16(buf) => unsafe {
                raw::gemm(std::ptr::null_mut(), x, buf.as_device_ptr(), y, 1, in_features, out_features, stream);
            },
            LinearWeight::Quantized { qweight, scales } => unsafe {
                raw::linear_quantized(x, qweight.as_device_ptr(), scales.as_device_ptr(), y, in_features, out_features, W4A16_GROUP_SIZE as i32, stream);
            },
        }
    }

    /// §108: the real batched-prefill (`num_tokens > 1`) counterpart of
    /// `apply` above -- dispatches to hipBLAS (bf16) or the new tiled
    /// `w4a16_gemm_prefill` kernel (quantized), whichever this weight
    /// actually is. `x`/`y` are row-major `[num_tokens, features]`, same
    /// convention `raw::gemm`'s own `rows>1` branch already uses.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn apply_prefill(&self, handle: blas_ffi::HipblasHandle, x: *const c_void, y: *mut c_void, num_tokens: i32, in_features: i32, out_features: i32, stream: *mut c_void) {
        match self {
            LinearWeight::Bf16(buf) => unsafe {
                raw::gemm(handle, x, buf.as_device_ptr(), y, num_tokens, in_features, out_features, stream);
            },
            LinearWeight::Quantized { qweight, scales } => unsafe {
                raw::linear_quantized_prefill(x, qweight.as_device_ptr(), scales.as_device_ptr(), y, in_features, out_features, W4A16_GROUP_SIZE as i32, num_tokens, stream);
            },
        }
    }

    /// Real, loud failure (not a silent wrong computation) for the one
    /// real operation that still doesn't support quantized weights: LoRA
    /// fold/snapshot-restore (`lora.rs`, bf16-only, §101). Batched prefill
    /// no longer needs this -- see `apply_prefill` above (§108).
    pub fn as_bf16(&self) -> &DeviceBuffer<u16> {
        match self {
            LinearWeight::Bf16(buf) => buf,
            LinearWeight::Quantized { .. } => panic!(
                "LinearWeight::as_bf16 called on a quantized weight -- LoRA folding doesn't support W4A16-quantized layers yet (see docs/DECISIONS.md §106/§108)"
            ),
        }
    }

    /// Mutable counterpart of `as_bf16`, same real restriction.
    pub fn as_bf16_mut(&mut self) -> &mut DeviceBuffer<u16> {
        match self {
            LinearWeight::Bf16(buf) => buf,
            LinearWeight::Quantized { .. } => panic!(
                "LinearWeight::as_bf16_mut called on a quantized weight -- LoRA folding doesn't support W4A16-quantized layers yet (see docs/DECISIONS.md §106/§108)"
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
                Ok(LinearWeight::Quantized { qweight, scales })
            } else {
                Ok(LinearWeight::Bf16(load_bf16_weight(snapshot_dir, name)?))
            }
        };
        let load_concat_linear = |names: &[&str]| -> Result<LinearWeight, String> {
            if is_quantized(names[0]) {
                let (qweight, scales) = crate::model_loader::load_concat_w4a16_weights(snapshot_dir, names)?;
                Ok(LinearWeight::Quantized { qweight, scales })
            } else {
                Ok(LinearWeight::Bf16(load_concat_bf16_weights(snapshot_dir, names)?))
            }
        };

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
            let gate_up_proj = load_concat_linear(&[&gate_name, &up_name])?;

            if is_full_attention_layer(i) {
                let q_name = format!("{p}.self_attn.q_proj.weight");
                let k_name = format!("{p}.self_attn.k_proj.weight");
                let v_name = format!("{p}.self_attn.v_proj.weight");
                let qkv_proj = load_concat_linear(&[&q_name, &k_name, &v_name])?;
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
                    o_proj: load_linear(&format!("{p}.self_attn.o_proj.weight"))?,
                    gate_up_proj,
                    down_proj: load_linear(&format!("{p}.mlp.down_proj.weight"))?,
                }));
            } else {
                let qkv_name = format!("{p}.linear_attn.in_proj_qkv.weight");
                let z_name = format!("{p}.linear_attn.in_proj_z.weight");
                let b_name = format!("{p}.linear_attn.in_proj_b.weight");
                let a_name = format!("{p}.linear_attn.in_proj_a.weight");
                let in_proj_combined =
                    load_concat_linear(&[&qkv_name, &z_name, &b_name, &a_name])?;
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
                    down_proj: load_linear(&format!("{p}.mlp.down_proj.weight"))?,
                }));
            }
        }

        Ok(ModelWeights {
            embed_tokens,
            lm_head,
            final_norm,
            layers,
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
    /// §117: the real, current `real_num_tokens` (how many of this
    /// chunk's rows are real prompt tokens vs. real bucket padding),
    /// written FRESH before every real call -- unlike `num_tokens`
    /// itself (a real, HOST-side value baked into a captured graph's own
    /// kernel launch CONFIGURATION, safe because it's fixed per bucket),
    /// `real_num_tokens` genuinely VARIES call to call for the SAME
    /// bucket (different real prompts, same bucket, different real
    /// lengths) -- a real HIP Graph replay does NOT re-evaluate host-side
    /// integer kernel ARGUMENTS baked in at capture time, only re-reads
    /// live BUFFER CONTENTS. Every real kernel whose correctness depends
    /// on `real_num_tokens` (the logits row-select, GDN's zero-pad
    /// boundary, causal_conv1d's real trailing-window snapshot point)
    /// reads it from THIS buffer internally now, not from a host-passed
    /// parameter -- the real, root fix for a real bug this session found
    /// directly (not by inspection): baking `real_num_tokens` in as a
    /// host parameter meant a graph captured for one real prompt length
    /// silently reused that SAME length for every later real request
    /// sharing its bucket, corrupting both the immediate response and
    /// all subsequent decode state.
    real_num_tokens_buf: DeviceBuffer<i32>,

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
    attn_q_raw: DeviceBuffer<u16>,
    attn_k_raw: DeviceBuffer<u16>,
    attn_v_raw: DeviceBuffer<u16>,
    attn_query: DeviceBuffer<u16>,
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

    /// §117: real, fixed destination for `raw::gather_last_real_row_bf16`
    /// -- the REAL last token's row, physically copied here (device-side,
    /// reading `real_num_tokens_buf`) before `rmsnorm` reads it, so
    /// `rmsnorm`'s own captured-graph input pointer is always this SAME
    /// fixed address across every replay, never a host-computed offset
    /// into `final_hidden` that would otherwise bake in whichever real
    /// prompt's row index captured first. See `run_prefill_chunk_body`.
    last_row_gathered: DeviceBuffer<u16>,
}

impl PrefillScratch {
    pub fn new(max_seq_len: usize) -> Result<Self, HipError> {
        let t = MAX_PREFILL_CHUNK;
        Ok(PrefillScratch {
            token_ids_dev: DeviceBuffer::alloc(t)?,
            position_buf: DeviceBuffer::alloc(t)?,
            real_num_tokens_buf: DeviceBuffer::alloc(1)?,

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
            attn_q_raw: DeviceBuffer::alloc(t * ATTN_NUM_HEADS * ATTN_HEAD_DIM * 2)?,
            attn_k_raw: DeviceBuffer::alloc(t * ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM)?,
            attn_v_raw: DeviceBuffer::alloc(t * ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM)?,
            attn_query: DeviceBuffer::alloc(t * ATTN_NUM_HEADS * ATTN_HEAD_DIM)?,
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
            last_row_gathered: DeviceBuffer::alloc(HIDDEN_SIZE)?,
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
    handle_raw: blas_ffi::HipblasHandle,
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
fn attn_layer_forward(
    handle_raw: blas_ffi::HipblasHandle,
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
    start_position: usize,
    max_seq_len: usize,
    s: &mut PrefillScratch,
    stream: *mut c_void,
) {
    let t = num_tokens as i32;
    let q_row_len = ATTN_NUM_HEADS * ATTN_HEAD_DIM; // per-token query length (pre-split from q_raw's 2x-wide query|gate)
    let kv_row_len = ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM;
    let kv_len = start_position + num_tokens;
    let n_rep = ATTN_NUM_HEADS / ATTN_NUM_KV_HEADS; // GQA: n_rep query heads share each KV head
    unsafe {
        raw::rmsnorm(hidden_states.as_device_ptr(), w.input_layernorm.as_device_ptr(), s.normed.as_device_ptr_mut(), t, HIDDEN_SIZE as i32, RMS_EPS, stream);

        w.qkv_proj.apply_prefill(handle_raw, s.normed.as_device_ptr(), s.attn_qkv_out.as_device_ptr_mut(), t, HIDDEN_SIZE as i32, ATTN_QKV_COMBINED_DIM as i32, stream);

        // Un-fuse q_raw|k_raw|v_raw out of each row's real (wider) stride
        // into their own tightly-packed [T, len] buffers -- see
        // `extract_range.hip`.
        raw::extract_range(s.attn_qkv_out.as_device_ptr(), s.attn_q_raw.as_device_ptr_mut(), t, ATTN_QKV_COMBINED_DIM as i32, 0, (q_row_len * 2) as i32, stream);
        raw::extract_range(s.attn_qkv_out.as_device_ptr(), s.attn_k_raw.as_device_ptr_mut(), t, ATTN_QKV_COMBINED_DIM as i32, (q_row_len * 2) as i32, kv_row_len as i32, stream);
        raw::extract_range(s.attn_qkv_out.as_device_ptr(), s.attn_v_raw.as_device_ptr_mut(), t, ATTN_QKV_COMBINED_DIM as i32, (q_row_len * 2 + kv_row_len) as i32, kv_row_len as i32, stream);

        // q_raw's real per-token layout is [num_heads, 2*head_dim] (query
        // and gate interleaved PER HEAD, matching `split_last_dim`'s own
        // decode-path call: `rows=ATTN_NUM_HEADS, half=ATTN_HEAD_DIM`, NOT
        // `rows=1, half=num_heads*head_dim`). Now that q_raw is tightly
        // packed as [T, num_heads*2*head_dim], reinterpreting it as
        // [T*num_heads, 2*head_dim] is the same "no stride mismatch"
        // trick used for q/k-norm below -- exactly matches the per-head
        // split every real token already goes through in the decode path.
        raw::split_last_dim(s.attn_q_raw.as_device_ptr(), s.attn_query.as_device_ptr_mut(), s.attn_gate.as_device_ptr_mut(), (num_tokens * ATTN_NUM_HEADS) as i32, ATTN_HEAD_DIM as i32, stream);

        // Per-head RMSNorm over ALL (token, head) pairs in ONE call --
        // [T, num_heads*head_dim] reinterpreted as [T*num_heads, head_dim]
        // is exactly the same bytes, and rmsnorm has no cross-row
        // dependency (see `mlp_block_prefill`'s own comment on this trick).
        raw::rmsnorm(s.attn_query.as_device_ptr(), w.q_norm.as_device_ptr(), s.attn_query_normed.as_device_ptr_mut(), (num_tokens * ATTN_NUM_HEADS) as i32, ATTN_HEAD_DIM as i32, RMS_EPS, stream);
        raw::rmsnorm(s.attn_k_raw.as_device_ptr(), w.k_norm.as_device_ptr(), s.attn_key_normed.as_device_ptr_mut(), (num_tokens * ATTN_NUM_KV_HEADS) as i32, ATTN_HEAD_DIM as i32, RMS_EPS, stream);

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
        // §99: attention itself is STILL dispatched on `kv_len`, the same
        // threshold-based pattern `raw::gemm` already uses for `rows==1`
        // vs `rows>1` -- see `ATTENTION_GEMM_KV_LEN_THRESHOLD`'s own doc
        // comment for the real measurements behind this (the naive scalar
        // kernel genuinely wins below the threshold; real matrix-core GEMM
        // genuinely wins above it -- neither is a strict improvement on
        // its own). Below threshold, `attention_decode` stays a real
        // per-token loop (unchanged from §96) -- its own real recurrence
        // (each row must read the FULL KV cache up to and including its
        // own just-appended slot) isn't part of the batching this pass
        // targets. Above threshold, it's skipped here and replaced by the
        // batched per-head GEMM block below, once the whole chunk's K/V
        // is written.
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
            for i in 0..num_tokens {
                let position_ptr = s.position_buf.as_device_ptr_at(i) as *const i32;
                raw::attention_decode_split(
                    s.attn_query_roped.as_device_ptr_at(i * q_row_len),
                    layer_state.k_cache.as_device_ptr(),
                    layer_state.v_cache.as_device_ptr(),
                    s.attn_out.as_device_ptr_at_mut(i * q_row_len),
                    ATTN_NUM_HEADS as i32,
                    ATTN_NUM_KV_HEADS as i32,
                    position_ptr,
                    max_seq_len as i32,
                    ATTN_HEAD_DIM as i32,
                    scaling,
                    stream,
                );
            }
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
fn gdn_chunk_forward_prefill(handle_raw: blas_ffi::HipblasHandle, layer_state: &mut GdnLayerState, num_tokens: usize, real_num_tokens: usize, s: &mut PrefillScratch, stream: *mut c_void) {
    debug_assert!(real_num_tokens <= num_tokens, "real_num_tokens ({real_num_tokens}) must not exceed num_tokens ({num_tokens})");
    let h = GDN_NUM_V_HEADS;
    let kh = GDN_NUM_K_HEADS;
    let d = GDN_HEAD_DIM;
    let c = GDN_CHUNK_SIZE;
    let n_rep = (GDN_NUM_V_HEADS / GDN_NUM_K_HEADS) as i32;
    let num_chunks = num_tokens.div_ceil(GDN_CHUNK_SIZE);
    let t_pad = num_chunks * GDN_CHUNK_SIZE;

    unsafe {
        // §117: real chunk padding, generalized from `num_tokens` to
        // `real_num_tokens` -- zero the `[real_num_tokens, t_pad)` tail so
        // fake positions become pure no-op identity steps (zero
        // key/value/beta, zero log-decay), matching the real reference's
        // own `F.pad(..., 0)` exactly. For every EXISTING (eager,
        // non-bucketed) caller, `real_num_tokens == num_tokens`,
        // reproducing the exact original `t_pad > num_tokens` behavior
        // unchanged. The new real case this generalization is FOR: a
        // real bucketed prefill call where `num_tokens` is a FIXED bucket
        // size (for real HIP-Graph-capture shape stability) strictly
        // larger than the real token count -- in that case `t_pad` may
        // already equal `num_tokens` (no rounding left to do), but there
        // can still be a real, potentially multi-chunk-wide gap between
        // `real_num_tokens` and `t_pad` that MUST be zeroed (an entire
        // trailing chunk can be pure padding -- `real_gdn_chunk_forward_prefill_bucketed_padding_matches_real_transformers_function`
        // is the real, direct proof this generalization is correct, not
        // just the original partial-tail case).
        //
        // §117 Stage 2 fix: this USED to be a host-side `if t_pad >
        // real_num_tokens { fill_zero_from_async(real_num_tokens * ...) }`
        // -- both the BRANCH (whether the zero-fill nodes exist in the
        // graph at all) and the OFFSET (baked into each node's own memset
        // arguments) are host-side values that a captured HIP Graph
        // replay never re-evaluates; they get frozen at whichever real
        // prompt happens to capture a given bucket first, silently
        // corrupting every later real request with a DIFFERENT
        // `real_num_tokens` sharing that bucket. Found via a real,
        // reproduced HTTP bug (warmup capturing with `real_num_tokens=11`,
        // a later real 54-token request replaying that same stale
        // branch/offset and generating only 2 tokens before an incorrect
        // early EOS). Fixed by making the zero-fill ALWAYS launch (so
        // it's always a node in the graph) and read `real_num_tokens`
        // itself off the device every call, inside the kernel -- see
        // `zero_from_device_offset.hip`'s own doc comment.
        let real_num_tokens_ptr = s.real_num_tokens_buf.as_device_ptr() as *const c_void;
        raw::zero_from_device_offset_bf16(s.gdn_mixed_qkv.as_device_ptr_mut() as *mut c_void, real_num_tokens_ptr, t_pad as i32, GDN_CONV_DIM as i32, stream);
        raw::zero_from_device_offset_f32(s.gdn_g.as_device_ptr_mut() as *mut c_void, real_num_tokens_ptr, t_pad as i32, GDN_NUM_V_HEADS as i32, stream);
        raw::zero_from_device_offset_f32(s.gdn_beta.as_device_ptr_mut() as *mut c_void, real_num_tokens_ptr, t_pad as i32, GDN_NUM_V_HEADS as i32, stream);

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

        // Phase 1b: `ut_system = k_beta @ key^T`, `intra_chunk_attn =
        // query @ key^T` -- one real matrix-core GEMM per (head, chunk)
        // block (see `blas::gemm_qkt_bf16`'s own doc comment for the
        // derivation; identical shape, just applied per-chunk-block
        // instead of per-whole-kv-cache). Both write into head-major
        // `[H, num_chunks, C, C]` score buffers, decay-masked IN PLACE by
        // `gdn_chunk_decay_bf16` right after.
        for head in 0..h {
            for chunk in 0..num_chunks {
                let hm_d_off = (head * num_chunks + chunk) * c * d;
                let hm_c_off = (head * num_chunks + chunk) * c * c;
                let k_ptr = s.gdnc_key_bcast_hm.as_device_ptr_at(hm_d_off);

                let kbeta_off = chunk * c * (h * d) + head * d;
                let kbeta_ptr = s.gdnc_k_beta_token.as_device_ptr_at(kbeta_off);
                let ut_ptr = s.gdnc_ut_system.as_device_ptr_at_mut(hm_c_off);
                raw::gemm_qkt(handle_raw, kbeta_ptr, (h * d) as i32, k_ptr, ut_ptr, c as i32, c as i32, d as i32, c as i32, 1.0);

                let q_ptr = s.gdnc_query_intra_hm.as_device_ptr_at(hm_d_off);
                let intra_ptr = s.gdnc_intra_attn.as_device_ptr_at_mut(hm_c_off);
                raw::gemm_qkt(handle_raw, q_ptr, d as i32, k_ptr, intra_ptr, c as i32, c as i32, d as i32, c as i32, 1.0);
            }
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
        for chunk in 0..num_chunks {
            for head in 0..h {
                let hm_d_off = (head * num_chunks + chunk) * c * d;
                let hm_c_off = (head * num_chunks + chunk) * c * c;
                let state_off = head * d * d;

                let k_cumdecay_ptr = s.gdnc_k_cumdecay.as_device_ptr_at(hm_d_off);
                let state_ptr = s.gdnc_state_bf16.as_device_ptr_at(state_off);
                let v_new_ptr = s.gdnc_new_values.as_device_ptr_at_mut(hm_d_off);
                raw::gemm_pv(handle_raw, k_cumdecay_ptr, d as i32, state_ptr, v_new_ptr, d as i32, c as i32, d as i32, d as i32, -1.0, 1.0);

                let out_off = chunk * c * GDN_VALUE_DIM + head * d;
                let query_final_ptr = s.gdnc_query_final_hm.as_device_ptr_at(hm_d_off);
                let out_ptr = s.gdn_out.as_device_ptr_at_mut(out_off);
                raw::gemm_pv(handle_raw, query_final_ptr, d as i32, state_ptr, out_ptr, GDN_VALUE_DIM as i32, c as i32, d as i32, d as i32, 1.0, 0.0);

                let intra_ptr = s.gdnc_intra_attn.as_device_ptr_at(hm_c_off);
                let v_new_ptr_ro = s.gdnc_new_values.as_device_ptr_at(hm_d_off);
                let out_ptr2 = s.gdn_out.as_device_ptr_at_mut(out_off);
                raw::gemm_pv(handle_raw, intra_ptr, c as i32, v_new_ptr_ro, out_ptr2, GDN_VALUE_DIM as i32, c as i32, c as i32, d as i32, 1.0, 1.0);

                let key_final_ptr = s.gdnc_key_final_hm.as_device_ptr_at(hm_d_off);
                let v_new_ptr_ro2 = s.gdnc_new_values.as_device_ptr_at(hm_d_off);
                // §116: `state = state*chunk_decay + key_final^T@v_new`
                // used to be ONE fused hipBLAS call (`gemm_atb`'s own
                // `beta=chunk_decay_scalar`, a real HOST f32 -- the one
                // remaining reason this function needed a host sync at
                // all, per its own §115 doc comment). Now two real
                // on-device steps: pre-scale `state` in place by
                // `chunk_decay`, read directly off the ALREADY-COMPUTED
                // device buffer (no host copy), then accumulate via
                // `gemm_atb` with a FIXED host literal `beta=1.0` -- safe,
                // the same convention every other real call in this file
                // already uses. See `scale_by_device_scalar.hip`'s own
                // header for the one real, disclosed precision note this
                // introduces (an extra bf16 rounding step) and how it was
                // checked, not assumed, to be safe.
                let chunk_decay_idx = chunk * h + head;
                raw::scale_bf16_by_device_scalar(s.gdnc_state_bf16.as_device_ptr_at_mut(state_off), s.gdnc_chunk_decay.as_device_ptr_at(chunk_decay_idx), (d * d) as i32, stream);
                let state_ptr_mut = s.gdnc_state_bf16.as_device_ptr_at_mut(state_off);
                raw::gemm_atb(handle_raw, key_final_ptr, d as i32, v_new_ptr_ro2, d as i32, state_ptr_mut, d as i32, c as i32, d as i32, d as i32, 1.0);
            }
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
    real_num_tokens: usize,
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
            s.real_num_tokens_buf.as_device_ptr() as *const c_void,
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
    gdn_chunk_forward_prefill(handle_raw, layer_state, num_tokens, real_num_tokens, s, stream);

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
    real_num_tokens: usize,
    stream: *mut c_void,
) -> *const DeviceBuffer<u16> {
    // Stable for the whole layer stack -- `state.position` is only
    // advanced by the caller AFTER this whole function returns (see
    // `forward_prefill_chunk`), so capturing it once here is exactly the
    // "start position of this chunk" every layer needs for real causal
    // attention (§99).
    let start_position = state.position;
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
                gdn_layer_forward_prefill(handle_raw, hidden_in, hidden_out, w, gs, num_tokens, real_num_tokens, &mut state.prefill_scratch, stream)
            }
            (LayerWeights::Attn(w), LayerState::Attn(as_)) => attn_layer_forward_prefill(
                handle_raw,
                hidden_in,
                hidden_out,
                w,
                as_,
                num_tokens,
                start_position,
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
    real_num_tokens: usize,
    logits_out: &mut DeviceBuffer<u16>,
    stream: *mut c_void,
) {
    let final_hidden_ptr = run_layers_over_chunk(handle_raw, weights, state, num_tokens, real_num_tokens, stream);
    // SAFETY: points at one of `state.prefill_scratch`'s two live
    // `hidden_a`/`hidden_b` fields (`run_layers_over_chunk`'s own
    // guarantee).
    let final_hidden: &DeviceBuffer<u16> = unsafe { &*final_hidden_ptr };

    let s = &mut state.prefill_scratch;
    // §117: the REAL last token's row -- `real_num_tokens - 1`, not
    // `num_tokens - 1` (a real bucketed call's `num_tokens` may be a
    // fixed bucket size strictly larger than the real prompt; the extra
    // rows are padding, never meant to feed the real logits).
    //
    // §117 Stage 2 fix: this USED to be a host-computed pointer offset
    // (`final_hidden.as_device_ptr_at((real_num_tokens - 1) *
    // HIDDEN_SIZE)`) fed straight into `rmsnorm`'s own launch arguments --
    // exactly the same class of graph-capture-unsafe host value as the
    // GDN zero-fill above (a captured replay never re-evaluates it, so a
    // later real request sharing a bucket would silently get the FIRST
    // captor's row index instead of its own). Fixed by a real device-side
    // gather: `raw::gather_last_real_row_bf16` reads `real_num_tokens`
    // off the device itself and copies the correct row into a FIXED
    // destination (`s.last_row_gathered`), so `rmsnorm`'s own input
    // pointer never changes across replays -- only the gather kernel's
    // internal row selection does.
    unsafe {
        raw::gather_last_real_row_bf16(final_hidden.as_device_ptr(), s.real_num_tokens_buf.as_device_ptr() as *const c_void, s.last_row_gathered.as_device_ptr_mut() as *mut c_void, HIDDEN_SIZE as i32, stream);
        raw::rmsnorm(s.last_row_gathered.as_device_ptr(), weights.final_norm.as_device_ptr(), s.final_normed.as_device_ptr_mut(), 1, HIDDEN_SIZE as i32, RMS_EPS, stream);
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
    let handle_raw = handle.raw();

    state.prefill_scratch.token_ids_dev.copy_from_host_prefix(token_ids)?;
    let positions: Vec<i32> = (0..num_tokens as i32).map(|i| state.position as i32 + i).collect();
    state.prefill_scratch.position_buf.copy_from_host_prefix(&positions)?;
    state.prefill_scratch.real_num_tokens_buf.copy_from_host(&[num_tokens as i32])?;

    run_prefill_chunk_body(handle_raw, weights, state, num_tokens, num_tokens, logits_out, std::ptr::null_mut());

    hip::check_last_error()?;
    hip::device_synchronize()?;

    state.position += num_tokens;
    Ok(())
}

/// §117: real, fixed prompt-length buckets for HIP-Graph-capturable
/// prefill -- every one a multiple of `GDN_CHUNK_SIZE` (64), so
/// `gdn_chunk_forward_prefill`'s own `t_pad` never needs to round UP past
/// the chosen bucket (`t_pad == bucket` exactly, see that function's own
/// §117 doc comment), capped at `MAX_PREFILL_CHUNK`.
const PREFILL_BUCKETS: [usize; 4] = [64, 128, 192, 256];

/// Real, safe placeholder token id for padding rows -- never read back for
/// anything semantic (no logits are ever computed for a padding row, and
/// `gdn_chunk_forward_prefill`'s own real zero-fill overwrites whatever
/// this embeds into before GDN's recurrence ever sees it), just needs to
/// be a real, valid, in-vocab id so `embedding_lookup`'s real gather never
/// reads out of bounds.
const PREFILL_PAD_TOKEN_ID: i32 = 0;

/// §117: the smallest real bucket that fits `real_num_tokens`, or `None`
/// if it exceeds every real bucket (the caller falls back to the
/// existing eager `forward_prefill_chunk` path for oversized chunks).
fn prefill_bucket_for(real_num_tokens: usize) -> Option<usize> {
    PREFILL_BUCKETS.into_iter().find(|&b| b >= real_num_tokens)
}

/// §117 Stage 1 (no HIP Graph capture yet -- real padding/bucketing logic
/// only, validated on its own before Stage 2 wraps it in capture/replay).
/// Real, fixed-bucket-size counterpart of `forward_prefill_chunk`: pads
/// the real `real_num_tokens` up to the smallest real bucket
/// (`prefill_bucket_for`), runs the SAME real `run_prefill_chunk_body`
/// pipeline over the full bucket (every kernel's row count becomes the
/// FIXED bucket size, not the real token count -- see this module's own
/// §117 notes on `gdn_chunk_forward_prefill`/`run_prefill_chunk_body` for
/// why this is safe), and advances `state.position` by the REAL token
/// count only. Real correctness gate: `real_bucketed_prefill_matches_eager_prefill_chunk`
/// checks this produces the IDENTICAL real logits and the IDENTICAL real
/// resulting `DecodeState` (KV cache, GDN state) as the existing eager
/// path, for the same real prompt.
fn forward_prefill_chunk_bucketed(
    handle_raw: blas_ffi::HipblasHandle,
    weights: &ModelWeights,
    state: &mut DecodeState,
    token_ids: &[i32],
    bucket: usize,
    logits_out: &mut DeviceBuffer<u16>,
    stream: *mut c_void,
) -> Result<(), HipError> {
    let real_num_tokens = token_ids.len();
    debug_assert!(real_num_tokens <= bucket, "real_num_tokens ({real_num_tokens}) must not exceed the chosen bucket ({bucket})");
    // Real, loud, checked-not-assumed safety net: every bucket row (real
    // AND padding) writes a real KV-cache entry at a real position up to
    // `state.position + bucket - 1`, and `attn_scores`/the KV caches
    // themselves are only ever allocated for `state.max_seq_len` real
    // positions -- silently exceeding that is a real, genuine
    // out-of-bounds GPU write, not a subtle numerical issue (caught
    // directly during this feature's own development: an early test
    // using too small a `max_seq_len` produced completely unrelated
    // garbage output, not a near-miss).
    assert!(
        state.position + bucket <= state.max_seq_len,
        "forward_prefill_chunk_bucketed: bucket {bucket} at position {} would exceed max_seq_len {} -- caller must ensure the chosen bucket fits real KV-cache capacity",
        state.position,
        state.max_seq_len
    );

    // Real token ids for the real prefix, a real safe placeholder for the
    // padding tail -- `copy_from_host_prefix` only writes `[0,
    // real_num_tokens)`, so the padding tail must be written explicitly
    // (unlike the real data, this genuinely differs call to call only in
    // LENGTH, not in the placeholder VALUE, so writing the same constant
    // every time is real, correct, and cheap).
    let mut padded_ids = vec![PREFILL_PAD_TOKEN_ID; bucket];
    padded_ids[..real_num_tokens].copy_from_slice(token_ids);
    state.prefill_scratch.token_ids_dev.copy_from_host_prefix(&padded_ids)?;

    // Real, continuing positions for EVERY bucket row (both real and
    // padding) -- see this module's own §117 notes on why padding rows
    // safely reusing the real forward position sequence (rather than any
    // special/reserved value) is what makes their real, transient
    // KV-cache writes harmless (provably overwritten by the real decode
    // steps that later reach those same positions, never read before
    // then).
    let positions: Vec<i32> = (0..bucket as i32).map(|i| state.position as i32 + i).collect();
    state.prefill_scratch.position_buf.copy_from_host_prefix(&positions)?;
    state.prefill_scratch.real_num_tokens_buf.copy_from_host(&[real_num_tokens as i32])?;

    run_prefill_chunk_body(handle_raw, weights, state, bucket, real_num_tokens, logits_out, stream);

    hip::check_last_error()?;
    hip::device_synchronize()?;

    // Real: advance by the REAL token count, not the padded bucket size.
    state.position += real_num_tokens;
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
    handle_raw: blas_ffi::HipblasHandle,
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
                gdn_layer_forward(handle_raw, hidden_in, hidden_out, w, gs, &mut state.scratch, stream)
            }
            (LayerWeights::Attn(w), LayerState::Attn(as_)) => attn_layer_forward(
                handle_raw,
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
pub fn forward_one_token(
    handle: &BlasHandle,
    weights: &ModelWeights,
    state: &mut DecodeState,
    token_id: i32,
    logits_out: &mut DeviceBuffer<u16>,
) -> Result<(), HipError> {
    let handle_raw = handle.raw();
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

    run_decode_body(handle_raw, weights, state, position_ptr, logits_out, std::ptr::null_mut());

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
    handle: BlasHandle,
    graph_exec: Option<hip::GraphExec>,
}

impl GraphedDecodeState {
    pub fn new() -> Result<Self, HipError> {
        let stream = hip::Stream::create()?;
        let handle = BlasHandle::create().map_err(|_| HipError { code: -1 })?;
        handle.set_stream(&stream).map_err(|_| HipError { code: -1 })?;
        Ok(GraphedDecodeState {
            stream,
            handle,
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
            run_decode_body(self.handle.raw(), weights, state, position_ptr, logits_out, self.stream.raw());
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

/// §117 Stage 2: real HIP-Graph-captured, bucketed prefill -- the same
/// real capture/replay pattern `GraphedDecodeState` already established
/// for decode, extended to prefill now that Stage 1
/// (`forward_prefill_chunk_bucketed`) is real, tested, and proven correct
/// on its own (kernel-level, GDN-chunk-level, and full-model byte-exact
/// generation, all against real references -- see this module's own
/// §117 notes). One real `GraphExec` PER real bucket (`PREFILL_BUCKETS`),
/// captured lazily on first use, cached thereafter -- unlike decode
/// (always exactly one real shape), prefill's real launch configs
/// genuinely depend on the chosen bucket size, so one graph cannot serve
/// every real prompt length.
///
/// Real, deliberate restriction: only ever used for a REAL prefill
/// starting at `state.position == 0` (a fresh request -- this server's
/// own real, actual usage pattern, confirmed by `server.rs`'s
/// `start_request` calling `state.reset()` immediately before every real
/// prefill call). `attn_layer_forward_prefill`'s own `use_gemm_attention`
/// threshold decision is baked into the captured graph at CAPTURE time
/// (a real, static Rust-level branch, not something graph replay
/// re-evaluates) -- restricting to `start_position==0` keeps `kv_len`
/// (and therefore that decision) identical, and therefore VALID, for
/// every real replay of a given bucket's graph.
pub struct GraphedPrefillState {
    stream: hip::Stream,
    handle: BlasHandle,
    graphs: std::collections::HashMap<usize, hip::GraphExec>,
}

impl GraphedPrefillState {
    pub fn new() -> Result<Self, HipError> {
        let stream = hip::Stream::create()?;
        let handle = BlasHandle::create().map_err(|_| HipError { code: -1 })?;
        handle.set_stream(&stream).map_err(|_| HipError { code: -1 })?;
        Ok(GraphedPrefillState {
            stream,
            handle,
            graphs: std::collections::HashMap::new(),
        })
    }

    /// Runs one real bucketed prefill call. Returns `Ok(None)` (real,
    /// explicit, NOT an error) if `token_ids` doesn't fit any real bucket
    /// (longer than `PREFILL_BUCKETS`'s own max) or `state.position != 0`
    /// -- the caller falls back to the existing real eager `forward_prefill`
    /// path for either case, matching this module's own established
    /// "loud, explicit fallback, never a silent wrong computation" idiom.
    /// Returns `Ok(Some(real_num_tokens))` on a real, successful graphed
    /// prefill (also the number of real tokens `state.position` advanced
    /// by).
    pub fn forward_prefill_bucketed(
        &mut self,
        weights: &ModelWeights,
        state: &mut DecodeState,
        token_ids: &[i32],
        logits_out: &mut DeviceBuffer<u16>,
    ) -> Result<Option<usize>, HipError> {
        if state.position != 0 {
            return Ok(None);
        }
        let Some(bucket) = prefill_bucket_for(token_ids.len()) else {
            return Ok(None);
        };
        if state.position + bucket > state.max_seq_len {
            return Ok(None);
        }
        let real_num_tokens = token_ids.len();

        let mut padded_ids = vec![PREFILL_PAD_TOKEN_ID; bucket];
        padded_ids[..real_num_tokens].copy_from_slice(token_ids);
        state.prefill_scratch.token_ids_dev.copy_from_host_prefix(&padded_ids)?;
        let positions: Vec<i32> = (0..bucket as i32).map(|i| state.position as i32 + i).collect();
        state.prefill_scratch.position_buf.copy_from_host_prefix(&positions)?;
        // §117, the real fix for a real, found bug: written fresh on
        // EVERY call (capture AND every later replay), same as
        // `token_ids_dev`/`position_buf` above -- see
        // `real_num_tokens_buf`'s own doc comment for why this can no
        // longer be a host-side kernel-launch parameter once graph
        // replay is involved.
        state.prefill_scratch.real_num_tokens_buf.copy_from_host(&[real_num_tokens as i32])?;

        if !self.graphs.contains_key(&bucket) {
            // §117 Stage 2, real and found-not-assumed: capturing this
            // real graph (the WHOLE 32-layer prefill pipeline, thousands
            // of real kernel-launch nodes -- decode's own graph, by
            // contrast, is one real decode step, far smaller) genuinely
            // overflows a default 8MB thread stack -- confirmed directly
            // (a real `RUST_MIN_STACK` experiment fixed it, not a
            // guess). Rather than require every real caller (this
            // engine's own HTTP server included) to remember to run on
            // an oversized-stack thread, the capture step runs on its
            // own real, dedicated, generously-sized scoped thread here
            // -- self-contained, and the borrow-checker-verified
            // `std::thread::scope` API means `weights`/`state`/
            // `logits_out` can be safely borrowed across the real OS
            // thread boundary without `unsafe`. `hipSetDevice` is
            // real, per-OS-thread context in HIP -- must be set again on
            // this NEW thread, exactly like every other test/entry point
            // in this crate already does on ITS own thread.
            // `HipblasHandle` (a real `*mut c_void`) is not `Send` on its
            // own (raw pointers never are) -- pass the SAFE `&BlasHandle`
            // wrapper across the thread boundary instead (real, already
            // `unsafe impl Send + Sync`, see `blas.rs`) and call `.raw()`
            // on the new thread, same real handle either way.
            let handle_ref = &self.handle;
            let stream_ref = &self.stream;
            let state_reborrow: &mut DecodeState = &mut *state;
            let logits_reborrow: &mut DeviceBuffer<u16> = &mut *logits_out;
            let graph_exec = std::thread::scope(|scope| -> Result<hip::GraphExec, HipError> {
                std::thread::Builder::new()
                    .stack_size(64 * 1024 * 1024)
                    .spawn_scoped(scope, move || -> Result<hip::GraphExec, HipError> {
                        hip::set_device(0)?;
                        hip::begin_capture(stream_ref)?;
                        run_prefill_chunk_body(handle_ref.raw(), weights, state_reborrow, bucket, real_num_tokens, logits_reborrow, stream_ref.raw());
                        hip::end_capture(stream_ref)
                    })
                    .expect("failed to spawn the real prefill-graph-capture thread")
                    .join()
                    .expect("the real prefill-graph-capture thread panicked")
            })?;
            self.graphs.insert(bucket, graph_exec);
        }
        self.graphs.get(&bucket).unwrap().launch(&self.stream)?;
        self.stream.synchronize()?;

        state.position += real_num_tokens;
        Ok(Some(real_num_tokens))
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
        // §117: `gdn_chunk_forward_prefill`'s zero-fill now reads
        // `real_num_tokens` off THIS device buffer, not the host
        // parameter -- real call sites (`forward_prefill_chunk` etc.)
        // always write it first; this direct, below-`PrefillScratch`-level
        // test must do the same.
        s.real_num_tokens_buf.copy_from_host(&[seq_len as i32]).expect("copy_from_host(real_num_tokens_buf) failed");

        let conv_len = GDN_CONV_DIM * (GDN_CONV_KERNEL_SIZE - 1);
        let mut conv_state: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_len).unwrap();
        conv_state.copy_from_host(&vec![0u16; conv_len]).unwrap();
        let mut recurrent_state: DeviceBuffer<f32> = DeviceBuffer::alloc(h * d * d).unwrap();
        recurrent_state.copy_from_host(&vec![0f32; h * d * d]).unwrap();
        let mut layer_state = GdnLayerState { conv_state, recurrent_state };

        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        gdn_chunk_forward_prefill(handle.raw(), &mut layer_state, seq_len, seq_len, &mut s, std::ptr::null_mut());

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
    #[test]
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
    #[test]
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

    /// §117 decisive test: the real, generalized zero-pad boundary
    /// (`real_num_tokens` instead of `num_tokens`) that a real bucketed
    /// HIP-Graph-capturable prefill call needs. Calls
    /// `gdn_chunk_forward_prefill` with `num_tokens=128` (a real FIXED
    /// bucket -- TWO full chunks) but `real_num_tokens=8` (matching the
    /// EXACT same real 8-token synthetic case and EXACT same real
    /// `torch_chunk_gated_delta_rule` reference values as
    /// `real_gdn_chunk_forward_prefill_matches_real_transformers_function`
    /// above) -- if the real chunk-padding output/state for those first 8
    /// real tokens comes out identical to that already-proven reference
    /// EVEN THOUGH there's now a real SECOND, ENTIRELY-padding chunk
    /// (positions 64-127) processed by the SAME real sequential scan,
    /// that's real, direct proof a fully-padding trailing chunk is a pure
    /// no-op -- not the already-tested partial-tail-of-one-chunk case,
    /// the genuinely new scenario bucketed prefill introduces.
    ///
    /// Critically, the real padding region (`gdn_mixed_qkv`/`gdn_g`/
    /// `gdn_beta` positions `[8, 128)`) is deliberately POISONED with
    /// real, large, non-zero, decidedly-not-accidentally-zero synthetic
    /// garbage BEFORE the call -- proving the real zero-fill mechanism is
    /// what produces the correct result, not a lucky zero-initialized
    /// buffer.
    #[test]
    fn real_gdn_chunk_forward_prefill_bucketed_padding_matches_real_transformers_function() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let kh = GDN_NUM_K_HEADS;
        let h = GDN_NUM_V_HEADS;
        let d = GDN_HEAD_DIM;
        let seq_len = 8usize;
        let bucket_len = 128usize; // two real GDN_CHUNK_SIZE=64 chunks, second entirely padding

        let synth = |n: usize, offset: i32, period: i32, scale: f32| -> Vec<f32> {
            (0..n).map(|i| { let m = ((i as i32 + offset) % period) as f32; (m - (period as f32) / 2.0) * scale }).collect()
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
        // Real, deliberate poison: fill the WHOLE buffers with large,
        // obviously-not-zero synthetic garbage FIRST, then write the real
        // seq_len=8 data over the real prefix -- so [8, 128) stays
        // poisoned unless the real zero-fill mechanism cleans it up.
        let poison_qkv: Vec<u16> = (0..s.gdn_mixed_qkv.len()).map(|i| f32_to_bf16(999.0 + (i % 13) as f32)).collect();
        let poison_g: Vec<f32> = (0..s.gdn_g.len()).map(|i| 777.0 + (i % 7) as f32).collect();
        let poison_beta: Vec<f32> = (0..s.gdn_beta.len()).map(|i| 555.0 + (i % 5) as f32).collect();
        s.gdn_mixed_qkv.copy_from_host(&poison_qkv).expect("poison copy_from_host(gdn_mixed_qkv) failed");
        s.gdn_g.copy_from_host(&poison_g).expect("poison copy_from_host(gdn_g) failed");
        s.gdn_beta.copy_from_host(&poison_beta).expect("poison copy_from_host(gdn_beta) failed");

        s.gdn_mixed_qkv.copy_from_host_prefix(&mixed_qkv_bf16).expect("copy_from_host_prefix(gdn_mixed_qkv) failed");
        s.gdn_g.copy_from_host_prefix(&g).expect("copy_from_host_prefix(gdn_g) failed");
        s.gdn_beta.copy_from_host_prefix(&beta).expect("copy_from_host_prefix(gdn_beta) failed");
        // §117: see the same write in `run_chunked_gdn_case` above -- the
        // zero-fill now reads `real_num_tokens` off this device buffer.
        s.real_num_tokens_buf.copy_from_host(&[seq_len as i32]).expect("copy_from_host(real_num_tokens_buf) failed");

        let conv_len = GDN_CONV_DIM * (GDN_CONV_KERNEL_SIZE - 1);
        let mut conv_state: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_len).unwrap();
        conv_state.copy_from_host(&vec![0u16; conv_len]).unwrap();
        let mut recurrent_state: DeviceBuffer<f32> = DeviceBuffer::alloc(h * d * d).unwrap();
        recurrent_state.copy_from_host(&vec![0f32; h * d * d]).unwrap();
        let mut layer_state = GdnLayerState { conv_state, recurrent_state };

        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        gdn_chunk_forward_prefill(handle.raw(), &mut layer_state, bucket_len, seq_len, &mut s, std::ptr::null_mut());
        hip::device_synchronize().expect("device_synchronize after bucketed gdn_chunk_forward_prefill failed");

        let mut out_full = vec![0u16; s.gdn_out.len()];
        s.gdn_out.copy_to_host(&mut out_full).expect("copy_to_host(gdn_out) failed");
        let out: Vec<f32> = out_full.iter().map(|&b| bf16_to_f32(b)).collect();
        let mut state_after = vec![0f32; h * d * d];
        layer_state.recurrent_state.copy_to_host(&mut state_after).expect("copy_to_host(recurrent_state) failed");

        // Same real reference values as the seq_len=8 single-chunk test
        // above (`scratchpad/gen_chunked_gdn_reference.py`'s "seq=8
        // chunk=64" case) -- must be IDENTICAL despite the real second,
        // entirely-padding chunk now in play.
        let out_t0_h0_expected: [f32; 8] = [-0.0000192810, -0.0000115686, -0.0000038562, 0.0000038562, 0.0000115686, 0.0000192810, 0.0000269934, 0.0000347058];
        let out_t7_h31_expected: [f32; 8] = [-0.0004500012, -0.0005173064, -0.0003445547, -0.0001718030, -0.0000780824, 0.0000946693, -0.0006016739, -0.0004289221];
        let state_h0_row0_expected: [f32; 8] = [-0.0094241491, -0.0100941947, -0.0113990232, -0.0011821315, -0.0024869598, -0.0037917888, 0.0105796494, 0.0092748199];
        let state_h31_row0_expected: [f32; 8] = [0.0095543936, 0.0121570667, 0.0109002497, 0.0096434327, 0.0125330016, 0.0112761846, -0.0131629938, -0.0144198108];

        let check = |got: &[f32], expected: &[f32; 8], label: &str| {
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert!((g - e).abs() < 0.002, "{label}[{i}]: bucketed GPU chunked GDN got {g}, real reference {e} -- real padding chunk was NOT a clean no-op");
            }
        };
        check(&out[0 * GDN_VALUE_DIM..0 * GDN_VALUE_DIM + 8], &out_t0_h0_expected, "out[token=0,head=0]");
        check(&out[7 * GDN_VALUE_DIM + 31 * d..7 * GDN_VALUE_DIM + 31 * d + 8], &out_t7_h31_expected, "out[token=7,head=31]");
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

    /// §117 decisive test: the real, full-model, end-to-end proof that
    /// bucketed prefill (Stage 1: real padding/bucketing, no HIP Graph
    /// capture yet) produces the IDENTICAL real generation as the
    /// already-proven eager path above -- same real 5-token prompt, same
    /// real independently-verified `expected_new_ids`, but routed through
    /// `forward_prefill_chunk_bucketed` with `bucket=128` (deliberately
    /// NOT the smallest bucket that fits 5 tokens -- forces a real SECOND,
    /// entirely-padding GDN chunk, the same real scenario
    /// `real_gdn_chunk_forward_prefill_bucketed_padding_matches_real_transformers_function`
    /// already proved in isolation, now exercised through the FULL real
    /// model: every layer, both attention paths, the real logits-row
    /// selection, and the real `state.position` advance-by-real-count-only
    /// rule all have to be correct simultaneously for this to pass.
    #[test]
    #[ignore]
    fn real_bucketed_prefill_matches_real_qwen3_5_4b_greedy_generation() {
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
        let bucket = 128usize; // deliberately > prefill_bucket_for(5)'s own minimum (64), forces a real fully-padding second chunk

        // Real KV-cache capacity: every bucket row (real AND padding)
        // writes a real KV entry, so this must cover the real bucket
        // itself plus room for the real decode steps afterward -- NOT
        // just the real 5-token prompt (caught directly: an earlier
        // version of this test used max_seq_len=32, a real silent
        // out-of-bounds GPU write once `bucket=128` exceeded it).
        let max_seq_len = 160usize;
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();

        forward_prefill_chunk_bucketed(handle.raw(), &weights, &mut state, &prompt_ids, bucket, &mut logits, std::ptr::null_mut())
            .expect("forward_prefill_chunk_bucketed failed");
        assert_eq!(
            state.position,
            prompt_ids.len(),
            "forward_prefill_chunk_bucketed must advance state.position by the REAL token count (5), not the bucket size (128)"
        );

        let mut generated = Vec::with_capacity(expected_new_ids.len());
        for _ in 0..expected_new_ids.len() {
            let next_id = argmax_sample(&logits).expect("argmax_sample failed");
            generated.push(next_id);
            forward_one_token(&handle, &weights, &mut state, next_id, &mut logits).expect("forward_one_token failed during generation");
        }

        eprintln!("generated (bucketed prefill, bucket=128): {generated:?}");
        eprintln!("expected:                                 {expected_new_ids:?}");
        assert_eq!(
            generated, expected_new_ids,
            "bucketed-prefill-then-decode did not match the real model's own greedy generation for the same prompt -- real padding leaked into real state or logits"
        );
    }

    /// §117 Stage 2 decisive test: the real HIP-Graph-captured path
    /// (`GraphedPrefillState`) produces the IDENTICAL real generation as
    /// the already-proven Stage 1 eager-bucketed path above -- run
    /// TWICE, on two SEPARATE fresh `DecodeState`s but the SAME
    /// `GraphedPrefillState` instance, so the real FIRST call captures a
    /// new graph for bucket=128 and the real SECOND call REPLAYS the
    /// cached one -- proving both the initial capture AND a real replay
    /// against fresh (different-address... no, same-address-different-
    /// CONTENT, since both `DecodeState`s independently allocate their
    /// own `prefill_scratch`) buffers are correct, not just the capture
    /// pass alone.
    #[test]
    #[ignore]
    fn real_graphed_bucketed_prefill_matches_real_qwen3_5_4b_greedy_generation() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        eprintln!("loading real weights for all {NUM_LAYERS} layers...");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        eprintln!("weights loaded.");

        let decode_handle = BlasHandle::create().expect("real hipblasCreate failed");
        let mut graphed = GraphedPrefillState::new().expect("GraphedPrefillState::new failed");

        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let expected_new_ids: [i32; 6] = [11751, 13, 198, 32, 13, 2912];
        let max_seq_len = 160usize;

        // §117 Stage 2, real and important: a captured graph bakes in the
        // real GPU buffer ADDRESSES live at capture time (`state`'s own
        // `prefill_scratch`, `logits`) -- it is NOT valid to replay
        // against a DIFFERENT `DecodeState`/logits buffer, only the SAME
        // one, reset in place between real requests (`DecodeState::reset`
        // zeros GDN state and `position` WITHOUT reallocating anything --
        // exactly matching how a real server reuses one `DecodeState`
        // across a connection's real requests). Caught directly: an
        // earlier version of this test allocated a FRESH `DecodeState`
        // per pass, and pass 2's replay produced a real GPU memory access
        // fault reading pass 1's already-freed buffers -- a real test
        // bug, not a `GraphedPrefillState` bug.
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();

        for pass in 0..2 {
            if pass > 0 {
                state.reset().expect("DecodeState::reset failed");
            }
            let advanced = graphed
                .forward_prefill_bucketed(&weights, &mut state, &prompt_ids, &mut logits)
                .expect("forward_prefill_bucketed failed")
                .expect("forward_prefill_bucketed must accept this real 5-token prompt at position 0");
            assert_eq!(advanced, prompt_ids.len(), "pass {pass}: must advance by the REAL token count");
            assert_eq!(state.position, prompt_ids.len(), "pass {pass}: state.position must be the REAL token count");

            let mut generated = Vec::with_capacity(expected_new_ids.len());
            for _ in 0..expected_new_ids.len() {
                let next_id = argmax_sample(&logits).expect("argmax_sample failed");
                generated.push(next_id);
                forward_one_token(&decode_handle, &weights, &mut state, next_id, &mut logits).expect("forward_one_token failed during generation");
            }

            eprintln!("pass {pass} ({}): generated = {generated:?}", if pass == 0 { "fresh capture" } else { "cached replay" });
            assert_eq!(
                generated, expected_new_ids,
                "pass {pass}: graphed-bucketed-prefill-then-decode did not match the real model's own greedy generation"
            );
        }
    }

    /// §117 Stage 2 decisive test, THE real regression this whole Stage 2
    /// fix exists for: capture a real bucket's graph with ONE real
    /// prompt's `real_num_tokens`, then REPLAY that same cached graph
    /// with a DIFFERENT real prompt (different real length, same bucket)
    /// -- exactly the real, reproduced HTTP bug that motivated this fix
    /// (a warmup request, "Hi" at `real_num_tokens=11`, captured
    /// bucket=64; a later real streaming request for a genuinely
    /// different, longer prompt at `real_num_tokens=54` then replayed
    /// that SAME stale graph and generated only 2 tokens before an
    /// incorrect early EOS). Every OTHER real graphed-prefill test in
    /// this file only ever replays with the IDENTICAL prompt used to
    /// capture (see the test immediately above) -- that exercises replay
    /// at all, but can never catch a stale-host-value-baked-into-the-
    /// graph bug, since the "stale" value and the "real" value are
    /// always the same number. This test is the one that closes that
    /// real gap.
    ///
    /// Reference: prompt B run through the ALREADY-PROVEN-CORRECT eager
    /// `forward_prefill` path (used throughout this file, always matched
    /// against real transformers references) on its own independent
    /// `DecodeState` -- if graphed replay's `real_num_tokens_buf` fix is
    /// correct, prompt B's graphed-replay generation must be BYTE-FOR-
    /// BYTE identical to its own eager generation, despite having been
    /// replayed against a graph captured for a DIFFERENT prompt entirely.
    #[test]
    #[ignore]
    fn real_graphed_bucketed_prefill_replay_with_different_real_num_tokens_matches_eager_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        eprintln!("loading real weights for all {NUM_LAYERS} layers...");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        let tok = crate::tokenizer::ChatTokenizer::load(&snapshot).expect("real tokenizer.json failed to load");
        eprintln!("weights loaded.");

        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        let max_seq_len = 160usize;
        let num_generate = 6;

        // Prompt A: the short, already-well-established 5-token warmup
        // prompt used throughout this file -- stands in for the real
        // HTTP bug's own short warmup ("Hi").
        let prompt_a: [i32; 5] = [760, 6511, 314, 9338, 369];

        // Prompt B: a real, DIFFERENT, LONGER prompt (real tokenizer,
        // real chat template) -- still small enough to share bucket=64
        // with prompt A (`GDN_CHUNK_SIZE=64` is the smallest real
        // bucket), the exact real scenario that corrupted the real HTTP
        // request in the original bug.
        let prompt_b_text = tok.apply_chat_template(&[
            ("system", "You are a senior PostgreSQL and Python backend architect."),
            ("user", "Write a short production FastAPI health check endpoint."),
        ]);
        let prompt_b = tok.encode(&prompt_b_text).expect("real tokenizer encode failed");
        eprintln!("prompt A: {} real tokens, prompt B: {} real tokens", prompt_a.len(), prompt_b.len());
        assert_ne!(prompt_a.len(), prompt_b.len(), "prompt A and B must have genuinely different real_num_tokens to exercise the real bug");
        assert_eq!(prefill_bucket_for(prompt_a.len()), Some(64), "prompt A must land in bucket=64 for this test to be decisive");
        assert_eq!(prefill_bucket_for(prompt_b.len()), Some(64), "prompt B must land in the SAME bucket=64 as prompt A for this test to be decisive");

        // Real, deliberate, matching-the-real-server design: ONE shared
        // `DecodeState`/`logits` buffer for BOTH prompt A and prompt B,
        // `state.reset()` between them -- NOT two independently allocated
        // `DecodeState`s. `server.rs`'s own `Engine` holds exactly ONE
        // `DecodeState` for the whole engine lifetime (never reallocated
        // per request), and a captured HIP Graph is permanently bound to
        // the specific buffer ADDRESSES live at capture time (confirmed
        // directly by an earlier real test-design bug in the test
        // immediately above this one: a fresh `DecodeState` per pass
        // caused a real GPU memory access fault on replay). Two
        // independent `DecodeState`s here would test something this
        // engine never actually does, and would fault or silently read
        // stale unrelated memory rather than exercise the real bug this
        // test targets.
        let mut graphed = GraphedPrefillState::new().expect("GraphedPrefillState::new failed");
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();

        // Pass 1 (warmup): prompt A captures bucket=64's graph fresh
        // (real_num_tokens=5 baked into the capture, per this fix, ONLY
        // as a fixed launch shape -- never as a value any kernel logic
        // depends on).
        graphed
            .forward_prefill_bucketed(&weights, &mut state, &prompt_a, &mut logits)
            .expect("forward_prefill_bucketed failed for prompt A")
            .expect("prompt A must be accepted by the graphed bucketed path");
        assert_eq!(graphed.graphs.len(), 1, "prompt A must have captured exactly one real graph (bucket=64)");

        // Pass 2 (the real request): reset the SAME `state` (matching
        // `server.rs`'s own `start_request`, which always resets before a
        // new real prefill), then prompt B replays the SAME cached
        // bucket=64 graph captured above for prompt A -- exactly the real
        // warmup-then-real-request HTTP sequence that exposed the
        // original bug.
        state.reset().expect("DecodeState::reset failed");
        let advanced = graphed
            .forward_prefill_bucketed(&weights, &mut state, &prompt_b, &mut logits)
            .expect("forward_prefill_bucketed failed for prompt B")
            .expect("prompt B must be accepted by the graphed bucketed path");
        assert_eq!(graphed.graphs.len(), 1, "prompt B must REPLAY the cached bucket=64 graph, not capture a second one");
        assert_eq!(advanced, prompt_b.len(), "must advance by prompt B's own REAL token count, not prompt A's stale one");
        assert_eq!(state.position, prompt_b.len(), "state.position must be prompt B's own REAL token count");

        let mut generated_b_graphed = Vec::with_capacity(num_generate);
        for _ in 0..num_generate {
            let next_id = argmax_sample(&logits).expect("argmax_sample failed");
            generated_b_graphed.push(next_id);
            forward_one_token(&handle, &weights, &mut state, next_id, &mut logits).expect("forward_one_token failed during generation (graphed)");
        }

        // Independent reference: prompt B through the already-proven-correct eager path, on its own independent DecodeState, completely untouched by prompt A or the graphed path.
        let mut state_ref = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits_ref: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        forward_prefill(&handle, &weights, &mut state_ref, &prompt_b, &mut logits_ref).expect("forward_prefill failed for the reference pass");
        let mut generated_b_eager = Vec::with_capacity(num_generate);
        for _ in 0..num_generate {
            let next_id = argmax_sample(&logits_ref).expect("argmax_sample failed");
            generated_b_eager.push(next_id);
            forward_one_token(&handle, &weights, &mut state_ref, next_id, &mut logits_ref).expect("forward_one_token failed during generation (eager reference)");
        }

        eprintln!("prompt B via graphed replay (contaminated by prompt A's capture): {generated_b_graphed:?}");
        eprintln!("prompt B via eager reference (independent):                       {generated_b_eager:?}");
        assert_eq!(
            generated_b_graphed, generated_b_eager,
            "graphed replay with a DIFFERENT real_num_tokens than the one that captured the graph diverged from the independent eager reference -- real_num_tokens_buf's device-side fix did not fully close the stale-host-value bug"
        );
    }

    /// §117 DIAGNOSTIC (not a permanent correctness gate): runs the SAME
    /// real prompt through eager `forward_prefill_chunk` and bucketed
    /// `forward_prefill_chunk_bucketed` on two independent `DecodeState`s,
    /// then compares every layer's real resulting state (K/V cache at the
    /// REAL positions only, GDN recurrent_state) plus the real logits
    /// themselves, printing exactly where the first real divergence shows
    /// up -- built to root-cause the real, measured failure of
    /// `real_bucketed_prefill_matches_real_qwen3_5_4b_greedy_generation`
    /// rather than guess.
    #[test]
    #[ignore]
    fn diagnose_real_bucketed_vs_eager_prefill_state_divergence() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let real_num_tokens = prompt_ids.len();
        // §117: real, confirmed finding -- `bucket=64` (matching eager's
        // own internal `t_pad` exactly, same GEMM/reduction shape) gives
        // BIT-EXACT-ZERO divergence everywhere (logits AND every layer's
        // state). `bucket=128` (the real production case: TWO chunks, the
        // second entirely padding) shows small, non-zero, but real
        // floating-point REDUCTION-ORDER noise (GEMM/softmax summing a
        // different real number of zero-valued terms rounds differently
        // at the bf16 boundary, a well-known associativity artifact, not
        // a logic bug) -- confirmed, not assumed, by directly comparing
        // both bucket sizes against the same real eager baseline. Real,
        // decisive proof this is benign: `real_bucketed_prefill_matches_real_qwen3_5_4b_greedy_generation`
        // (bucket=128) still produces the exact real reference token
        // sequence -- this noise is real but never large enough to flip
        // a real argmax decision for that prompt, consistent with this
        // codebase's own pre-existing GDN correctness bar (`0.002`
        // tolerance against `transformers`, not bit-exact, even in the
        // ORIGINAL eager-only code).
        let bucket = 128usize;
        let max_seq_len = 160usize;

        let mut state_eager = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits_eager: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        forward_prefill_chunk(&handle, &weights, &mut state_eager, &prompt_ids, &mut logits_eager).expect("eager forward_prefill_chunk failed");

        let mut state_bucketed = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits_bucketed: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        forward_prefill_chunk_bucketed(handle.raw(), &weights, &mut state_bucketed, &prompt_ids, bucket, &mut logits_bucketed, std::ptr::null_mut())
            .expect("bucketed forward_prefill_chunk_bucketed failed");

        // Real logits comparison first (row 4, the real last token).
        let mut le = vec![0u16; VOCAB_SIZE];
        let mut lb = vec![0u16; VOCAB_SIZE];
        logits_eager.copy_to_host(&mut le).unwrap();
        logits_bucketed.copy_to_host(&mut lb).unwrap();
        let logits_diff = le.iter().zip(lb.iter()).filter(|(a, b)| a != b).count();
        eprintln!("logits: {logits_diff}/{VOCAB_SIZE} bf16 values differ between eager and bucketed");

        // Real per-layer state comparison.
        for (i, (ls_e, ls_b)) in state_eager.layers.iter().zip(state_bucketed.layers.iter()).enumerate() {
            match (ls_e, ls_b) {
                (LayerState::Attn(ae), LayerState::Attn(ab)) => {
                    let cache_len = ATTN_NUM_KV_HEADS * max_seq_len * ATTN_HEAD_DIM;
                    let mut ke = vec![0u16; cache_len];
                    let mut kb = vec![0u16; cache_len];
                    ae.k_cache.copy_to_host(&mut ke).unwrap();
                    ab.k_cache.copy_to_host(&mut kb).unwrap();
                    let mut real_diff = 0usize;
                    for h in 0..ATTN_NUM_KV_HEADS {
                        let base = h * max_seq_len * ATTN_HEAD_DIM;
                        let real_range = base..(base + real_num_tokens * ATTN_HEAD_DIM);
                        real_diff += ke[real_range.clone()].iter().zip(kb[real_range].iter()).filter(|(a, b)| a != b).count();
                    }
                    eprintln!("layer {i} (Attn): k_cache real-position diff = {real_diff} elements");
                }
                (LayerState::Gdn(ge), LayerState::Gdn(gb)) => {
                    let mut se = vec![0f32; ge.recurrent_state.len()];
                    let mut sb = vec![0f32; gb.recurrent_state.len()];
                    ge.recurrent_state.copy_to_host(&mut se).unwrap();
                    gb.recurrent_state.copy_to_host(&mut sb).unwrap();
                    let max_abs_diff = se.iter().zip(sb.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
                    eprintln!("layer {i} (Gdn): recurrent_state max abs diff = {max_abs_diff}");
                }
                _ => unreachable!(),
            }
        }
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
        let handle_raw = handle.raw();

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
                        gdn_layer_forward(handle_raw, hidden_in, hidden_out, w, gs, &mut state.scratch, std::ptr::null_mut());
                        hip::device_synchronize().unwrap();
                        gdn_total += t_layer.elapsed();
                        gdn_count += 1;
                    }
                    (LayerWeights::Attn(w), LayerState::Attn(as_)) => {
                        attn_layer_forward(handle_raw, hidden_in, hidden_out, w, as_, position_ptr, state.max_seq_len, &mut state.scratch, std::ptr::null_mut());
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
}
