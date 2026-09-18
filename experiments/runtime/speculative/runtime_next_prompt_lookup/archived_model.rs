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

pub const HIDDEN_SIZE: usize = 2560;
pub const NUM_LAYERS: usize = 32;
pub const INTERMEDIATE_SIZE: usize = 9216;
pub const VOCAB_SIZE: usize = 248320;
pub const RMS_EPS: f32 = 1e-6;

pub const ATTN_NUM_HEADS: usize = 16;
pub const ATTN_NUM_KV_HEADS: usize = 4;
pub const ATTN_HEAD_DIM: usize = 256;
pub const ATTN_ROTARY_DIM: usize = 64;
pub const ATTN_ROPE_THETA: f32 = 10_000_000.0;

pub const GDN_NUM_K_HEADS: usize = 16;
pub const GDN_NUM_V_HEADS: usize = 32;
pub const GDN_HEAD_DIM: usize = 128;
pub const GDN_KEY_DIM: usize = GDN_NUM_K_HEADS * GDN_HEAD_DIM; // 2048
pub const GDN_VALUE_DIM: usize = GDN_NUM_V_HEADS * GDN_HEAD_DIM; // 4096
pub const GDN_CONV_DIM: usize = GDN_KEY_DIM * 2 + GDN_VALUE_DIM; // 8192
pub const GDN_CONV_KERNEL_SIZE: usize = 4;

/// Real, confirmed-from-`config.json` full-attention layer indices (every
/// 4th layer, 1-indexed from `full_attention_interval=4`).
const FULL_ATTENTION_LAYERS: [usize; 8] = [3, 7, 11, 15, 19, 23, 27, 31];

fn is_full_attention_layer(i: usize) -> bool {
    FULL_ATTENTION_LAYERS.contains(&i)
}

/// Thin, non-syncing wrappers around the exact same audited FFI
/// declarations `kernels.rs`/`blas.rs` already expose -- no new unsafe
/// surface, just called without the trailing `check_last_error()`/
/// `device_synchronize()` every safe wrapper pays per call. Every launch
/// here goes on the default stream (`0`), the same stream every `.hip`
/// launcher and `hipblasGemmEx` call already uses -- so these stay
/// correctly ordered relative to each other without any explicit stream
/// management, purely from HIP's own in-order-per-stream guarantee.
mod raw {
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

    pub unsafe fn split_last_dim(x: *const c_void, first: *mut c_void, second: *mut c_void, rows: i32, half: i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_split_last_dim_bf16(x, first, second, rows, half, stream) };
    }

    /// §93: `position` is a DEVICE pointer, not a host value -- see
    /// `kv_cache_append.hip`'s own comment for why.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn kv_cache_append(new_k: *const c_void, new_v: *const c_void, k_cache: *mut c_void, v_cache: *mut c_void, num_kv_heads: i32, max_seq_len: i32, head_dim: i32, position: *const i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_kv_cache_append_bf16(new_k, new_v, k_cache, v_cache, num_kv_heads, max_seq_len, head_dim, position, stream) };
    }

    /// §93: `position` is a DEVICE pointer -- see `rope.hip`'s own comment.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn rope(x: *const c_void, out: *mut c_void, num_rows: i32, head_dim: i32, rotary_dim: i32, theta: f32, position: *const i32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_rope_bf16(x, out, num_rows, head_dim, rotary_dim, theta, position, stream) };
    }

    /// §92: `kv_stride` lets this read directly out of the real head-major
    /// KV cache (stride = `max_seq_len`) -- no more per-token
    /// tightly-packed copy first (see `attention.hip`'s header). §93:
    /// `position` is a DEVICE pointer; `kv_len = *position + 1` is derived
    /// inside the kernel, and shared memory is sized for the fixed
    /// `kv_stride` upper bound so the launch configuration itself never
    /// changes across tokens.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn attention_decode(q: *const c_void, k: *const c_void, v: *const c_void, out: *mut c_void, num_q_heads: i32, num_kv_heads: i32, position: *const i32, kv_stride: i32, head_dim: i32, scaling: f32, stream: *mut c_void) {
        unsafe { kernels_ffi::launch_attention_decode_bf16(q, k, v, out, num_q_heads, num_kv_heads, position, kv_stride, head_dim, scaling, 256, stream) };
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

/// §96 (speculative decoding): the maximum number of draft tokens verified
/// in ONE speculative round (`forward_verify_chunk` processes
/// `MAX_DRAFT_TOKENS + 1` positions: the one real last-accepted token plus
/// up to `MAX_DRAFT_TOKENS` speculative candidates). Kept well under
/// `MAX_PREFILL_CHUNK` -- `PrefillScratch`'s buffers are shared by both
/// prefill and verify, sized for the larger `MAX_PREFILL_CHUNK`, so this
/// only bounds `verify_logits`' own size (`(MAX_DRAFT_TOKENS+1) *
/// VOCAB_SIZE` bf16 elements -- real but small: ~4.5MB at 8).
pub const MAX_DRAFT_TOKENS: usize = 8;

pub struct GdnLayerWeights {
    pub input_layernorm: DeviceBuffer<u16>,
    pub post_attention_layernorm: DeviceBuffer<u16>,
    /// `[GDN_IN_PROJ_COMBINED_DIM, hidden]` -- see the constant's own doc.
    pub in_proj_combined: DeviceBuffer<u16>,
    pub conv1d_weight: DeviceBuffer<u16>,
    pub a_log: DeviceBuffer<f32>,
    pub dt_bias: DeviceBuffer<f32>,
    /// Native f32 in the real checkpoint (see `rmsnorm_gated.hip`'s header).
    pub norm_weight: DeviceBuffer<f32>,
    pub out_proj: DeviceBuffer<u16>,
    /// `[GATE_UP_COMBINED_DIM, hidden]`.
    pub gate_up_proj: DeviceBuffer<u16>,
    pub down_proj: DeviceBuffer<u16>,
}

pub struct AttnLayerWeights {
    pub input_layernorm: DeviceBuffer<u16>,
    pub post_attention_layernorm: DeviceBuffer<u16>,
    /// `[ATTN_QKV_COMBINED_DIM, hidden]`.
    pub qkv_proj: DeviceBuffer<u16>,
    pub q_norm: DeviceBuffer<u16>,
    pub k_norm: DeviceBuffer<u16>,
    pub o_proj: DeviceBuffer<u16>,
    /// `[GATE_UP_COMBINED_DIM, hidden]`.
    pub gate_up_proj: DeviceBuffer<u16>,
    pub down_proj: DeviceBuffer<u16>,
}

pub enum LayerWeights {
    Gdn(GdnLayerWeights),
    Attn(AttnLayerWeights),
}

pub struct ModelWeights {
    pub embed_tokens: DeviceBuffer<u16>,
    pub final_norm: DeviceBuffer<u16>,
    pub layers: Vec<LayerWeights>,
}

impl ModelWeights {
    /// Loads every real weight this forward pass needs, for all 32 layers,
    /// from the real Qwen3.5-4B checkpoint. Real network I/O to disk (via
    /// mmap), real bf16/f32 decoding -- no placeholder tensors anywhere.
    pub fn load(snapshot_dir: &Path) -> Result<Self, String> {
        let embed_tokens =
            load_bf16_weight(snapshot_dir, "model.language_model.embed_tokens.weight")?;
        let final_norm = load_bf16_weight(snapshot_dir, "model.language_model.norm.weight")?;

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            let p = format!("model.language_model.layers.{i}");
            let gate_name = format!("{p}.mlp.gate_proj.weight");
            let up_name = format!("{p}.mlp.up_proj.weight");
            let gate_up_proj = load_concat_bf16_weights(snapshot_dir, &[&gate_name, &up_name])?;

            if is_full_attention_layer(i) {
                let q_name = format!("{p}.self_attn.q_proj.weight");
                let k_name = format!("{p}.self_attn.k_proj.weight");
                let v_name = format!("{p}.self_attn.v_proj.weight");
                let qkv_proj = load_concat_bf16_weights(snapshot_dir, &[&q_name, &k_name, &v_name])?;
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
                    o_proj: load_bf16_weight(snapshot_dir, &format!("{p}.self_attn.o_proj.weight"))?,
                    gate_up_proj,
                    down_proj: load_bf16_weight(snapshot_dir, &format!("{p}.mlp.down_proj.weight"))?,
                }));
            } else {
                let qkv_name = format!("{p}.linear_attn.in_proj_qkv.weight");
                let z_name = format!("{p}.linear_attn.in_proj_z.weight");
                let b_name = format!("{p}.linear_attn.in_proj_b.weight");
                let a_name = format!("{p}.linear_attn.in_proj_a.weight");
                let in_proj_combined =
                    load_concat_bf16_weights(snapshot_dir, &[&qkv_name, &z_name, &b_name, &a_name])?;
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
                    out_proj: load_bf16_weight(snapshot_dir, &format!("{p}.linear_attn.out_proj.weight"))?,
                    gate_up_proj,
                    down_proj: load_bf16_weight(snapshot_dir, &format!("{p}.mlp.down_proj.weight"))?,
                }));
            }
        }

        Ok(ModelWeights {
            embed_tokens,
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

    hidden_a: DeviceBuffer<u16>,
    hidden_b: DeviceBuffer<u16>,

    normed: DeviceBuffer<u16>,
    after_mixer: DeviceBuffer<u16>,
    normed2: DeviceBuffer<u16>,
    gate_up_out: DeviceBuffer<u16>,
    mlp_gate: DeviceBuffer<u16>,
    mlp_up: DeviceBuffer<u16>,
    swiglu_out: DeviceBuffer<u16>,
    mlp_out: DeviceBuffer<u16>,

    gdn_in_proj_out: DeviceBuffer<u16>,
    gdn_qkv_raw: DeviceBuffer<u16>,
    gdn_z: DeviceBuffer<u16>,
    gdn_b: DeviceBuffer<u16>,
    gdn_a: DeviceBuffer<u16>,
    gdn_mixed_qkv: DeviceBuffer<u16>,
    gdn_g: DeviceBuffer<f32>,
    gdn_beta: DeviceBuffer<f32>,
    gdn_out: DeviceBuffer<u16>,
    gdn_normed_gated: DeviceBuffer<u16>,
    gdn_mixer_out: DeviceBuffer<u16>,

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

    /// Only the LAST row of the chunk's final hidden state is ever needed
    /// (only the next token's logits matter -- see `forward_prefill_chunk`),
    /// so this stays rows=1, reusing the exact same fast GEMV lm_head path
    /// `run_decode_body` already uses.
    final_normed: DeviceBuffer<u16>,

    /// §96 (speculative decoding): `forward_verify_chunk`'s counterpart of
    /// `final_normed`/lm_head above -- UNLIKE prefill, verification needs
    /// logits at EVERY position in the chunk (each drafted token's
    /// correctness is checked against the base model's own prediction at
    /// its OWN position), not just the last. Sized for
    /// `MAX_DRAFT_TOKENS + 1` rows (the one real last-accepted token plus
    /// up to `MAX_DRAFT_TOKENS` speculative candidates) -- far smaller than
    /// `MAX_PREFILL_CHUNK`, so these are separate, smaller allocations
    /// rather than reusing the prefill-sized buffers above.
    verify_final_normed: DeviceBuffer<u16>,
    verify_logits: DeviceBuffer<u16>,
}

impl PrefillScratch {
    pub fn new() -> Result<Self, HipError> {
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
            mlp_gate: DeviceBuffer::alloc(t * INTERMEDIATE_SIZE)?,
            mlp_up: DeviceBuffer::alloc(t * INTERMEDIATE_SIZE)?,
            swiglu_out: DeviceBuffer::alloc(t * INTERMEDIATE_SIZE)?,
            mlp_out: DeviceBuffer::alloc(t * HIDDEN_SIZE)?,

            gdn_in_proj_out: DeviceBuffer::alloc(t * GDN_IN_PROJ_COMBINED_DIM)?,
            gdn_qkv_raw: DeviceBuffer::alloc(t * GDN_CONV_DIM)?,
            gdn_z: DeviceBuffer::alloc(t * GDN_VALUE_DIM)?,
            gdn_b: DeviceBuffer::alloc(t * GDN_NUM_V_HEADS)?,
            gdn_a: DeviceBuffer::alloc(t * GDN_NUM_V_HEADS)?,
            gdn_mixed_qkv: DeviceBuffer::alloc(t * GDN_CONV_DIM)?,
            gdn_g: DeviceBuffer::alloc(t * GDN_NUM_V_HEADS)?,
            gdn_beta: DeviceBuffer::alloc(t * GDN_NUM_V_HEADS)?,
            gdn_out: DeviceBuffer::alloc(t * GDN_VALUE_DIM)?,
            gdn_normed_gated: DeviceBuffer::alloc(t * GDN_VALUE_DIM)?,
            gdn_mixer_out: DeviceBuffer::alloc(t * HIDDEN_SIZE)?,

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

            final_normed: DeviceBuffer::alloc(HIDDEN_SIZE)?,

            verify_final_normed: DeviceBuffer::alloc((MAX_DRAFT_TOKENS + 1) * HIDDEN_SIZE)?,
            verify_logits: DeviceBuffer::alloc((MAX_DRAFT_TOKENS + 1) * VOCAB_SIZE)?,
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

    /// §96 (speculative decoding): one real, persistent snapshot slot per
    /// GDN layer (`conv_state`, `recurrent_state`), allocated once and
    /// reused across every speculative round -- see `snapshot_gdn_state`/
    /// `restore_gdn_state`. Indexed in the SAME order as `self.layers`
    /// (only GDN entries are ever populated/used; the vec has one slot per
    /// GDN layer, not one per `self.layers` index).
    gdn_snapshot: Vec<(DeviceBuffer<u16>, DeviceBuffer<f32>)>,
}

impl DecodeState {
    pub fn new(max_seq_len: usize) -> Result<Self, HipError> {
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        let mut gdn_snapshot = Vec::new();
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

                // §96: one real snapshot slot for this GDN layer, same
                // sizes as its own conv_state/recurrent_state -- see
                // `gdn_snapshot`'s own doc comment.
                let snapshot_conv: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_len)?;
                let snapshot_rs: DeviceBuffer<f32> = DeviceBuffer::alloc(rs_len)?;
                gdn_snapshot.push((snapshot_conv, snapshot_rs));

                layers.push(LayerState::Gdn(GdnLayerState {
                    conv_state,
                    recurrent_state,
                }));
            }
        }
        let scratch = Scratch::new()?;
        let prefill_scratch = PrefillScratch::new()?;
        Ok(DecodeState {
            layers,
            max_seq_len,
            position: 0,
            scratch,
            hidden_a: DeviceBuffer::alloc(HIDDEN_SIZE)?,
            hidden_b: DeviceBuffer::alloc(HIDDEN_SIZE)?,
            prefill_scratch,
            gdn_snapshot,
        })
    }

    /// §96 (speculative decoding): copies every GDN layer's real
    /// `conv_state`/`recurrent_state` into this `DecodeState`'s own
    /// persistent snapshot slots -- call BEFORE a speculative verify round
    /// that might need to be undone. Real device-to-device copies (see
    /// `DeviceBuffer::copy_from_device`'s own doc comment for why GDN's
    /// state, unlike the KV caches, cannot just be left unread).
    pub fn snapshot_gdn_state(&mut self) -> Result<(), HipError> {
        let mut slot = 0;
        for i in 0..self.layers.len() {
            if let LayerState::Gdn(gdn) = &self.layers[i] {
                let conv_src: *const DeviceBuffer<u16> = &gdn.conv_state;
                let rs_src: *const DeviceBuffer<f32> = &gdn.recurrent_state;
                // Queued (not blocking): all 24 layers' 48 copies go onto
                // the null stream back-to-back, ONE `device_synchronize()`
                // at the end -- see `DeviceBuffer::copy_from_device_async`'s
                // own doc comment for why the original per-call-blocking
                // `copy_from_device` was a real, measured cost.
                // SAFETY: `conv_src`/`rs_src` point at `self.layers[i]`'s
                // own fields, disjoint from `self.gdn_snapshot` (a
                // different top-level field) -- no aliasing.
                unsafe {
                    self.gdn_snapshot[slot].0.copy_from_device_async(&*conv_src, std::ptr::null_mut())?;
                    self.gdn_snapshot[slot].1.copy_from_device_async(&*rs_src, std::ptr::null_mut())?;
                }
                slot += 1;
            }
        }
        hip::check_last_error()?;
        hip::device_synchronize()
    }

    /// §96 (speculative decoding): the inverse of `snapshot_gdn_state` --
    /// restores every GDN layer's real `conv_state`/`recurrent_state` from
    /// this `DecodeState`'s snapshot slots. Call when a speculative round's
    /// draft was only PARTIALLY (or not at all) accepted, before replaying
    /// the accepted prefix via sequential `forward_one_token` calls.
    pub fn restore_gdn_state(&mut self) -> Result<(), HipError> {
        let mut slot = 0;
        for i in 0..self.layers.len() {
            if let LayerState::Gdn(gdn) = &mut self.layers[i] {
                gdn.conv_state.copy_from_device_async(&self.gdn_snapshot[slot].0, std::ptr::null_mut())?;
                gdn.recurrent_state.copy_from_device_async(&self.gdn_snapshot[slot].1, std::ptr::null_mut())?;
                slot += 1;
            }
        }
        hip::check_last_error()?;
        hip::device_synchronize()
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
        raw::gemm(handle_raw, s.normed.as_device_ptr(), w.in_proj_combined.as_device_ptr(), s.gdn_in_proj_out.as_device_ptr_mut(), 1, HIDDEN_SIZE as i32, GDN_IN_PROJ_COMBINED_DIM as i32, stream);
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
        raw::gemm(handle_raw, s.gdn_normed_gated.as_device_ptr(), w.out_proj.as_device_ptr(), s.gdn_mixer_out.as_device_ptr_mut(), 1, GDN_VALUE_DIM as i32, HIDDEN_SIZE as i32, stream);

        raw::add(hidden_states.as_device_ptr(), s.gdn_mixer_out.as_device_ptr(), s.after_mixer.as_device_ptr_mut(), HIDDEN_SIZE as i32, stream);
    }

    mlp_block(handle_raw, &s.after_mixer.as_device_ptr(), hidden_out, &w.post_attention_layernorm, &w.gate_up_proj, &w.down_proj, s, stream);
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
        raw::gemm(handle_raw, s.normed.as_device_ptr(), w.qkv_proj.as_device_ptr(), s.attn_qkv_out.as_device_ptr_mut(), 1, HIDDEN_SIZE as i32, ATTN_QKV_COMBINED_DIM as i32, stream);
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
        // `position_ptr` replaces a host-passed `kv_len`.
        let scaling = (ATTN_HEAD_DIM as f32).powf(-0.5);
        raw::attention_decode(
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

        raw::gemm(handle_raw, s.attn_gated.as_device_ptr(), w.o_proj.as_device_ptr(), s.attn_mixer_out.as_device_ptr_mut(), 1, (ATTN_NUM_HEADS * ATTN_HEAD_DIM) as i32, HIDDEN_SIZE as i32, stream);

        raw::add(hidden_states.as_device_ptr(), s.attn_mixer_out.as_device_ptr(), s.after_mixer.as_device_ptr_mut(), HIDDEN_SIZE as i32, stream);
    }

    mlp_block(handle_raw, &s.after_mixer.as_device_ptr(), hidden_out, &w.post_attention_layernorm, &w.gate_up_proj, &w.down_proj, s, stream);
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
    handle_raw: blas_ffi::HipblasHandle,
    hidden_in_ptr: &*const c_void,
    hidden_out: &mut DeviceBuffer<u16>,
    post_attention_layernorm: &DeviceBuffer<u16>,
    gate_up_proj: &DeviceBuffer<u16>,
    down_proj: &DeviceBuffer<u16>,
    s: &mut Scratch,
    stream: *mut c_void,
) {
    let hidden_in_ptr = *hidden_in_ptr;
    unsafe {
        raw::rmsnorm(hidden_in_ptr, post_attention_layernorm.as_device_ptr(), s.normed2.as_device_ptr_mut(), 1, HIDDEN_SIZE as i32, RMS_EPS, stream);

        // §92: ONE combined GEMM for gate_proj + up_proj (was 2). gate/up
        // are contiguous offset VIEWS into the single wide output.
        raw::gemm(handle_raw, s.normed2.as_device_ptr(), gate_up_proj.as_device_ptr(), s.gate_up_out.as_device_ptr_mut(), 1, HIDDEN_SIZE as i32, GATE_UP_COMBINED_DIM as i32, stream);
        let gate_ptr = s.gate_up_out.as_device_ptr(); // offset 0, len INTERMEDIATE_SIZE
        let up_ptr = s.gate_up_out.as_device_ptr_at(INTERMEDIATE_SIZE); // len INTERMEDIATE_SIZE
        raw::swiglu(gate_ptr, up_ptr, s.swiglu_out.as_device_ptr_mut(), INTERMEDIATE_SIZE as i32, stream);

        raw::gemm(handle_raw, s.swiglu_out.as_device_ptr(), down_proj.as_device_ptr(), s.mlp_out.as_device_ptr_mut(), 1, INTERMEDIATE_SIZE as i32, HIDDEN_SIZE as i32, stream);

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
    gate_up_proj: &DeviceBuffer<u16>,
    down_proj: &DeviceBuffer<u16>,
    num_tokens: usize,
    s: &mut PrefillScratch,
    stream: *mut c_void,
) {
    let t = num_tokens as i32;
    unsafe {
        raw::rmsnorm(hidden_in_ptr, post_attention_layernorm.as_device_ptr(), s.normed2.as_device_ptr_mut(), t, HIDDEN_SIZE as i32, RMS_EPS, stream);

        raw::gemm(handle_raw, s.normed2.as_device_ptr(), gate_up_proj.as_device_ptr(), s.gate_up_out.as_device_ptr_mut(), t, HIDDEN_SIZE as i32, GATE_UP_COMBINED_DIM as i32, stream);
        raw::extract_range(s.gate_up_out.as_device_ptr(), s.mlp_gate.as_device_ptr_mut(), t, GATE_UP_COMBINED_DIM as i32, 0, INTERMEDIATE_SIZE as i32, stream);
        raw::extract_range(s.gate_up_out.as_device_ptr(), s.mlp_up.as_device_ptr_mut(), t, GATE_UP_COMBINED_DIM as i32, INTERMEDIATE_SIZE as i32, INTERMEDIATE_SIZE as i32, stream);
        raw::swiglu(s.mlp_gate.as_device_ptr(), s.mlp_up.as_device_ptr(), s.swiglu_out.as_device_ptr_mut(), (num_tokens * INTERMEDIATE_SIZE) as i32, stream);

        raw::gemm(handle_raw, s.swiglu_out.as_device_ptr(), down_proj.as_device_ptr(), s.mlp_out.as_device_ptr_mut(), t, INTERMEDIATE_SIZE as i32, HIDDEN_SIZE as i32, stream);

        raw::add(hidden_in_ptr, s.mlp_out.as_device_ptr(), hidden_out.as_device_ptr_mut(), (num_tokens * HIDDEN_SIZE) as i32, stream);
    }
}

/// §96: `attn_layer_forward`'s real batched-prefill counterpart -- a chunk
/// of `num_tokens` new prompt tokens processed together. The real lever:
/// every GEMM (`qkv_proj`, `o_proj`) becomes one real batched hipBLAS call
/// (`raw::gemm`'s `rows > 1` branch) that reads this layer's weight
/// matrices ONCE for the whole chunk, instead of once per token -- see
/// that function's own comment. `rope`/`kv_cache_append`/`attention_decode`
/// stay real per-token calls in a tight inner loop (deliberately NOT
/// rewritten into a new batched-causal-attention kernel this pass -- see
/// `docs/DECISIONS.md` §96): each touches only this layer's small KV-cache
/// slice, not the multi-megabyte weight matrices the GEMMs above dominate
/// the cost of, so looping here costs comparatively little next to the real
/// GEMM win. Positions `start_position..start_position+num_tokens` must
/// already be written into `position_buf` (`s.position_buf`) by the caller
/// -- see `forward_prefill_chunk`.
#[allow(clippy::too_many_arguments)]
fn attn_layer_forward_prefill(
    handle_raw: blas_ffi::HipblasHandle,
    hidden_states: &DeviceBuffer<u16>,
    hidden_out: &mut DeviceBuffer<u16>,
    w: &AttnLayerWeights,
    layer_state: &mut AttnLayerState,
    num_tokens: usize,
    max_seq_len: usize,
    s: &mut PrefillScratch,
    stream: *mut c_void,
) {
    let t = num_tokens as i32;
    let q_row_len = ATTN_NUM_HEADS * ATTN_HEAD_DIM; // per-token query length (pre-split from q_raw's 2x-wide query|gate)
    let kv_row_len = ATTN_NUM_KV_HEADS * ATTN_HEAD_DIM;
    unsafe {
        raw::rmsnorm(hidden_states.as_device_ptr(), w.input_layernorm.as_device_ptr(), s.normed.as_device_ptr_mut(), t, HIDDEN_SIZE as i32, RMS_EPS, stream);

        raw::gemm(handle_raw, s.normed.as_device_ptr(), w.qkv_proj.as_device_ptr(), s.attn_qkv_out.as_device_ptr_mut(), t, HIDDEN_SIZE as i32, ATTN_QKV_COMBINED_DIM as i32, stream);

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

        // RoPE, KV-cache append, and attention itself are real per-position
        // ops (RoPE's angle depends on this token's absolute position; a
        // token can only attend to KV-cache slots already written, so
        // token t's append must precede token t+1's attention read) -- a
        // tight loop over the chunk, reusing the exact same decisively-
        // tested per-token kernels the decode path uses, each call offset
        // into this chunk's tightly-packed buffers.
        let scaling = (ATTN_HEAD_DIM as f32).powf(-0.5);
        for i in 0..num_tokens {
            let position_ptr = s.position_buf.as_device_ptr_at(i) as *const i32;
            raw::rope(
                s.attn_query_normed.as_device_ptr_at(i * q_row_len),
                s.attn_query_roped.as_device_ptr_at_mut(i * q_row_len),
                ATTN_NUM_HEADS as i32,
                ATTN_HEAD_DIM as i32,
                ATTN_ROTARY_DIM as i32,
                ATTN_ROPE_THETA,
                position_ptr,
                stream,
            );
            raw::rope(
                s.attn_key_normed.as_device_ptr_at(i * kv_row_len),
                s.attn_key_roped.as_device_ptr_at_mut(i * kv_row_len),
                ATTN_NUM_KV_HEADS as i32,
                ATTN_HEAD_DIM as i32,
                ATTN_ROTARY_DIM as i32,
                ATTN_ROPE_THETA,
                position_ptr,
                stream,
            );
            raw::kv_cache_append(
                s.attn_key_roped.as_device_ptr_at(i * kv_row_len),
                s.attn_v_raw.as_device_ptr_at(i * kv_row_len),
                layer_state.k_cache.as_device_ptr_mut(),
                layer_state.v_cache.as_device_ptr_mut(),
                ATTN_NUM_KV_HEADS as i32,
                max_seq_len as i32,
                ATTN_HEAD_DIM as i32,
                position_ptr,
                stream,
            );
            raw::attention_decode(
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

        raw::sigmoid_gate(s.attn_out.as_device_ptr(), s.attn_gate.as_device_ptr(), s.attn_gated.as_device_ptr_mut(), (num_tokens * q_row_len) as i32, stream);

        raw::gemm(handle_raw, s.attn_gated.as_device_ptr(), w.o_proj.as_device_ptr(), s.attn_mixer_out.as_device_ptr_mut(), t, q_row_len as i32, HIDDEN_SIZE as i32, stream);

        raw::add(hidden_states.as_device_ptr(), s.attn_mixer_out.as_device_ptr(), s.after_mixer.as_device_ptr_mut(), (num_tokens * HIDDEN_SIZE) as i32, stream);
    }

    mlp_block_prefill(handle_raw, s.after_mixer.as_device_ptr(), hidden_out, &w.post_attention_layernorm, &w.gate_up_proj, &w.down_proj, num_tokens, s, stream);
}

/// §96: `gdn_layer_forward`'s real batched-prefill counterpart. The
/// in_proj/out_proj GEMMs and the gated-RMSNorm batch over the whole
/// chunk in one call each (same reasoning as `attn_layer_forward_prefill`).
/// `causal_conv1d_update` and `gdn_recurrent_decode` stay genuinely
/// per-token sequential loops -- NOT a batching gap left unaddressed, but a
/// real, structural property of this op: both are true read-modify-write
/// recurrences (conv1d's shift-register state, the delta-rule's
/// `state[k,v]`), where token t's output depends on token t-1's update
/// having already happened. A real parallel "chunked" form exists in the
/// literature for both (chunked causal conv, and the delta-rule's UT-
/// transform chunked form) but porting either correctly is real, separate,
/// higher-risk work -- deliberately out of scope for this pass (see
/// `docs/DECISIONS.md` §96). Both loops here are cheap regardless: they
/// touch only this layer's tiny conv/recurrent state, not the multi-
/// megabyte weight matrices the batched GEMMs above dominate the cost of.
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

        raw::gemm(handle_raw, s.normed.as_device_ptr(), w.in_proj_combined.as_device_ptr(), s.gdn_in_proj_out.as_device_ptr_mut(), t, HIDDEN_SIZE as i32, GDN_IN_PROJ_COMBINED_DIM as i32, stream);

        raw::extract_range(s.gdn_in_proj_out.as_device_ptr(), s.gdn_qkv_raw.as_device_ptr_mut(), t, GDN_IN_PROJ_COMBINED_DIM as i32, 0, GDN_CONV_DIM as i32, stream);
        raw::extract_range(s.gdn_in_proj_out.as_device_ptr(), s.gdn_z.as_device_ptr_mut(), t, GDN_IN_PROJ_COMBINED_DIM as i32, GDN_CONV_DIM as i32, GDN_VALUE_DIM as i32, stream);
        raw::extract_range(s.gdn_in_proj_out.as_device_ptr(), s.gdn_b.as_device_ptr_mut(), t, GDN_IN_PROJ_COMBINED_DIM as i32, (GDN_CONV_DIM + GDN_VALUE_DIM) as i32, GDN_NUM_V_HEADS as i32, stream);
        raw::extract_range(s.gdn_in_proj_out.as_device_ptr(), s.gdn_a.as_device_ptr_mut(), t, GDN_IN_PROJ_COMBINED_DIM as i32, (GDN_CONV_DIM + GDN_VALUE_DIM + GDN_NUM_V_HEADS) as i32, GDN_NUM_V_HEADS as i32, stream);

        // Real sequential recurrence #1: causal conv1d's shift-register
        // state. `layer_state.conv_state` is the SAME buffer threaded
        // through all `num_tokens` iterations, exactly like the decode
        // path's single call -- just `num_tokens` of them in position
        // order.
        for i in 0..num_tokens {
            raw::causal_conv1d_update(
                s.gdn_qkv_raw.as_device_ptr_at(i * GDN_CONV_DIM),
                layer_state.conv_state.as_device_ptr_mut(),
                w.conv1d_weight.as_device_ptr(),
                s.gdn_mixed_qkv.as_device_ptr_at_mut(i * GDN_CONV_DIM),
                1,
                GDN_CONV_DIM as i32,
                GDN_CONV_KERNEL_SIZE as i32,
                stream,
            );
        }

        // gdn_gate_beta has no cross-token state, but `a_log`/`dt_bias`
        // (real per-layer trained weights, length num_v_heads) are indexed
        // directly by `h` inside the kernel -- calling it with
        // `num_heads = T*num_v_heads` in one shot would read `a_log`/
        // `dt_bias` out of bounds past the first token's `num_v_heads`
        // entries. Loop instead (tiny: num_v_heads=32 elements/call,
        // negligible next to the GEMMs above).
        for i in 0..num_tokens {
            raw::gdn_gate_beta(
                s.gdn_a.as_device_ptr_at(i * GDN_NUM_V_HEADS),
                s.gdn_b.as_device_ptr_at(i * GDN_NUM_V_HEADS),
                w.a_log.as_device_ptr() as *const f32,
                w.dt_bias.as_device_ptr() as *const f32,
                s.gdn_g.as_device_ptr_at_mut(i * GDN_NUM_V_HEADS) as *mut f32,
                s.gdn_beta.as_device_ptr_at_mut(i * GDN_NUM_V_HEADS) as *mut f32,
                GDN_NUM_V_HEADS as i32,
                stream,
            );
        }

        // Real sequential recurrence #2: the delta-rule's own
        // `state[k,v]`, the actual reason this op can't be one-shot
        // batched -- token t's output reads the state AFTER token t's own
        // update, which depends on token t-1's update having already
        // landed. `layer_state.recurrent_state` is the SAME buffer for
        // every iteration, by construction (this is what makes it a real
        // recurrence, not just a loop).
        for i in 0..num_tokens {
            let query_ptr = s.gdn_mixed_qkv.as_device_ptr_at(i * GDN_CONV_DIM); // offset 0, len GDN_KEY_DIM
            let key_ptr = s.gdn_mixed_qkv.as_device_ptr_at(i * GDN_CONV_DIM + GDN_KEY_DIM); // len GDN_KEY_DIM
            let value_ptr = s.gdn_mixed_qkv.as_device_ptr_at(i * GDN_CONV_DIM + 2 * GDN_KEY_DIM); // len GDN_VALUE_DIM
            raw::gdn_recurrent_decode(
                query_ptr,
                key_ptr,
                value_ptr,
                s.gdn_g.as_device_ptr_at(i * GDN_NUM_V_HEADS) as *const f32,
                s.gdn_beta.as_device_ptr_at(i * GDN_NUM_V_HEADS) as *const f32,
                layer_state.recurrent_state.as_device_ptr_mut() as *mut f32,
                s.gdn_out.as_device_ptr_at_mut(i * GDN_VALUE_DIM),
                GDN_NUM_V_HEADS as i32,
                GDN_NUM_K_HEADS as i32,
                GDN_HEAD_DIM as i32,
                stream,
            );
        }

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

        raw::gemm(handle_raw, s.gdn_normed_gated.as_device_ptr(), w.out_proj.as_device_ptr(), s.gdn_mixer_out.as_device_ptr_mut(), t, GDN_VALUE_DIM as i32, HIDDEN_SIZE as i32, stream);

        raw::add(hidden_states.as_device_ptr(), s.gdn_mixer_out.as_device_ptr(), s.after_mixer.as_device_ptr_mut(), (num_tokens * HIDDEN_SIZE) as i32, stream);
    }

    mlp_block_prefill(handle_raw, s.after_mixer.as_device_ptr(), hidden_out, &w.post_attention_layernorm, &w.gate_up_proj, &w.down_proj, num_tokens, s, stream);
}

/// §96: embedding lookup for all `num_tokens` new tokens -> all 32 real
/// layers (each layer's real batched-prefill variant above), shared by
/// both `run_prefill_chunk_body` (prefill: only the LAST row's logits are
/// ever needed) and `run_verify_chunk_body` (speculative-decode
/// verification: EVERY row's logits are needed). Mirrors `run_decode_body`'s
/// own structure and ping-pong buffer pattern exactly, just against
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
    stream: *mut c_void,
) -> *const DeviceBuffer<u16> {
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
/// eliminate elsewhere. Contrast `run_verify_chunk_body`, which genuinely
/// needs every row's logits.
fn run_prefill_chunk_body(
    handle_raw: blas_ffi::HipblasHandle,
    weights: &ModelWeights,
    state: &mut DecodeState,
    num_tokens: usize,
    logits_out: &mut DeviceBuffer<u16>,
    stream: *mut c_void,
) {
    let final_hidden_ptr = run_layers_over_chunk(handle_raw, weights, state, num_tokens, stream);
    // SAFETY: points at one of `state.prefill_scratch`'s two live
    // `hidden_a`/`hidden_b` fields (`run_layers_over_chunk`'s own
    // guarantee).
    let final_hidden: &DeviceBuffer<u16> = unsafe { &*final_hidden_ptr };

    let s = &mut state.prefill_scratch;
    let last_row_ptr = final_hidden.as_device_ptr_at((num_tokens - 1) * HIDDEN_SIZE);
    unsafe {
        raw::rmsnorm(last_row_ptr, weights.final_norm.as_device_ptr(), s.final_normed.as_device_ptr_mut(), 1, HIDDEN_SIZE as i32, RMS_EPS, stream);
        raw::gemm(
            handle_raw,
            s.final_normed.as_device_ptr(),
            weights.embed_tokens.as_device_ptr(),
            logits_out.as_device_ptr_mut(),
            1,
            HIDDEN_SIZE as i32,
            VOCAB_SIZE as i32,
            stream,
        );
    }
}

/// §96 (speculative decoding): verification's own final projection -- final
/// norm + lm_head for EVERY row in the chunk (batched: one real GEMM over
/// `num_tokens` rows, not `num_tokens` separate calls), since a
/// speculative round needs to check EACH drafted token's correctness
/// against the base model's own prediction at its OWN position, not just
/// the last. Writes `[num_tokens, VOCAB_SIZE]` into `logits_out`
/// (`state.prefill_scratch.verify_logits`, sized for `MAX_DRAFT_TOKENS + 1`
/// rows -- see that field's own doc comment).
fn run_verify_chunk_body(
    handle_raw: blas_ffi::HipblasHandle,
    weights: &ModelWeights,
    state: &mut DecodeState,
    num_tokens: usize,
    stream: *mut c_void,
) {
    let final_hidden_ptr = run_layers_over_chunk(handle_raw, weights, state, num_tokens, stream);
    // SAFETY: points at one of `state.prefill_scratch`'s two live
    // `hidden_a`/`hidden_b` fields (`run_layers_over_chunk`'s own
    // guarantee).
    let final_hidden: &DeviceBuffer<u16> = unsafe { &*final_hidden_ptr };

    let s = &mut state.prefill_scratch;
    let t = num_tokens as i32;
    unsafe {
        raw::rmsnorm(final_hidden.as_device_ptr(), weights.final_norm.as_device_ptr(), s.verify_final_normed.as_device_ptr_mut(), t, HIDDEN_SIZE as i32, RMS_EPS, stream);
        raw::gemm(
            handle_raw,
            s.verify_final_normed.as_device_ptr(),
            weights.embed_tokens.as_device_ptr(),
            s.verify_logits.as_device_ptr_mut(),
            t,
            HIDDEN_SIZE as i32,
            VOCAB_SIZE as i32,
            stream,
        );
    }
}

/// §96 (speculative decoding): verifies up to `MAX_DRAFT_TOKENS` candidate
/// tokens in ONE real batched forward pass. `token_ids` must be
/// `[last_accepted_token, draft_1, ..., draft_K]` -- the one real token
/// already accepted (whose OWN prediction is what draft_1 is checked
/// against), followed by up to `MAX_DRAFT_TOKENS` speculative candidates.
/// After this call, `state.prefill_scratch.verify_logits` holds
/// `[token_ids.len(), VOCAB_SIZE]` real logits: row `i` is the base
/// model's own real prediction for the token that comes after `token_ids[i]`
/// -- exactly what `forward_one_token` would have produced had it been fed
/// `token_ids[i]` at this same position. The caller checks `argmax(row i)
/// == token_ids[i+1]` for each `i` to find how many draft tokens were
/// actually correct.
///
/// Advances `state.position` by `token_ids.len()` UNCONDITIONALLY, and
/// writes real KV-cache entries and GDN conv/recurrent-state updates for
/// EVERY position, including any that a rejected draft later turns out to
/// be wrong. This is safe for the KV cache (rolling `state.position` back
/// down afterward makes the extra entries structurally unread, same
/// property `DecodeState::reset()` relies on) but NOT for GDN's state
/// (real read-modify-write regardless of position) -- callers MUST call
/// `state.snapshot_gdn_state()` before this and `state.restore_gdn_state()`
/// afterward if any tokens might be rejected. See
/// `real_speculative_decode_matches_real_sequential_greedy_generation` for
/// the decisive proof this whole protocol reproduces exactly what plain
/// sequential greedy decoding would have produced.
pub fn forward_verify_chunk(
    handle: &BlasHandle,
    weights: &ModelWeights,
    state: &mut DecodeState,
    token_ids: &[i32],
) -> Result<(), HipError> {
    let num_tokens = token_ids.len();
    assert!(num_tokens > 0, "forward_verify_chunk: token_ids must be non-empty");
    assert!(
        num_tokens <= MAX_DRAFT_TOKENS + 1,
        "forward_verify_chunk: {num_tokens} tokens exceeds MAX_DRAFT_TOKENS+1 ({})",
        MAX_DRAFT_TOKENS + 1
    );
    let handle_raw = handle.raw();

    state.prefill_scratch.token_ids_dev.copy_from_host_prefix(token_ids)?;
    let positions: Vec<i32> = (0..num_tokens as i32).map(|i| state.position as i32 + i).collect();
    state.prefill_scratch.position_buf.copy_from_host_prefix(&positions)?;

    run_verify_chunk_body(handle_raw, weights, state, num_tokens, std::ptr::null_mut());

    hip::check_last_error()?;
    hip::device_synchronize()?;

    state.position += num_tokens;
    Ok(())
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
        raw::gemm(
            handle_raw,
            state.scratch.final_normed.as_device_ptr(),
            weights.embed_tokens.as_device_ptr(),
            logits_out.as_device_ptr_mut(),
            1,
            HIDDEN_SIZE as i32,
            VOCAB_SIZE as i32,
            stream,
        );
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

/// Greedy sampling: argmax over real vocab logits. §93: computed on-device
/// (`kernels::argmax_bf16`) instead of copying the full `VOCAB_SIZE`-
/// element logits buffer to host and scanning it there -- real, measured
/// cost of the old approach was ~0.209ms/token, on top of the actual
/// forward-pass time. Same external signature and exact tie-breaking
/// semantics (first/lowest-index occurrence wins) as before.
pub fn argmax_sample(logits: &DeviceBuffer<u16>) -> Result<i32, HipError> {
    crate::kernels::argmax_bf16(logits)
}

/// §96: prompt-lookup speculative drafting -- a real, well-established,
/// model-free technique (matching HF transformers' own
/// `PromptLookupCandidateGenerator`): search `context` for the most recent
/// EARLIER occurrence of its own last `ngram_size` tokens, and if found,
/// draft the up-to-`num_draft` tokens that followed that earlier
/// occurrence as speculative candidates for what comes next. No separate
/// draft model needed -- correct by construction to return an EMPTY draft
/// (never a wrong non-empty guess) when no matching n-gram exists;
/// `speculative_generate` degrades to plain sequential decoding whenever
/// this returns empty. Deliberately the SIMPLEST real drafter, not an MTP
/// head or trained model -- this experiment is about whether verifying
/// drafts against real batched-prefill machinery helps AT ALL for this
/// engine, not about maximizing acceptance rate.
pub fn prompt_lookup_draft(context: &[i32], ngram_size: usize, num_draft: usize) -> Vec<i32> {
    if context.len() < ngram_size || num_draft == 0 {
        return Vec::new();
    }
    let needle = &context[context.len() - ngram_size..];
    let search_end = context.len() - ngram_size; // exclusive: skip the needle's own (most recent) position
    for start in (0..search_end).rev() {
        if &context[start..start + ngram_size] == needle {
            let draft_start = start + ngram_size;
            let draft_end = (draft_start + num_draft).min(context.len());
            if draft_start < draft_end {
                return context[draft_start..draft_end].to_vec();
            }
        }
    }
    Vec::new()
}

/// §96: one real speculative-decoding round's outcome, returned by
/// `speculative_round` for the caller (`speculative_generate`'s loop, or a
/// benchmark/test) to inspect -- real, honest accounting of what actually
/// happened, not just the generated tokens.
pub struct SpeculativeRoundResult {
    /// Every real new token accepted/emitted this round (always at least
    /// 1 -- the "bonus"/corrected token -- even on a full reject).
    pub generated: Vec<i32>,
    /// How many DRAFT tokens this round proposed (0 if no n-gram match).
    pub drafted: usize,
    /// How many of `drafted` were actually correct (verified against the
    /// base model's own real prediction).
    pub accepted: usize,
}

/// §96: runs ONE real speculative-decoding round. `context` must already
/// end with the last REAL accepted token (used only for drafting -- never
/// re-fed through the model here); `next_logits` must already hold that
/// same last accepted token's real resulting logits (from a prior
/// `forward_one_token`/`forward_prefill`/`speculative_round` call) --
/// UPDATED in place to hold the new last-token's logits before returning,
/// so the next round can call this again unchanged.
///
/// Real protocol (see `docs/DECISIONS.md` §96 for the full derivation):
/// draft up to `num_draft` tokens via `prompt_lookup_draft`; if empty, fall
/// back to one plain `forward_one_token` step (this IS what plain
/// sequential decoding does, not a degraded imitation of it). Otherwise,
/// snapshot GDN state, verify the WHOLE draft in one real batched
/// `forward_verify_chunk` call, then check each draft token against the
/// base model's own real prediction at its own position (the first
/// mismatch, if any, is where the batched verify's real, honest GDN-state
/// corruption starts -- see `DecodeState::restore_gdn_state`'s own doc
/// comment) -- roll `state.position` back to the real accepted count,
/// restore GDN state, replay the accepted prefix via `forward_one_token`
/// (cheap: at most `num_draft` calls), then feed the corrected/bonus real
/// token through ONE more `forward_one_token` call to seed the next
/// round's `next_logits`.
pub fn speculative_round(
    handle: &BlasHandle,
    weights: &ModelWeights,
    state: &mut DecodeState,
    context: &mut Vec<i32>,
    next_logits: &mut DeviceBuffer<u16>,
    ngram_size: usize,
    num_draft: usize,
) -> Result<SpeculativeRoundResult, HipError> {
    let draft = prompt_lookup_draft(context, ngram_size, num_draft);
    if draft.is_empty() {
        let next_token = argmax_sample(next_logits)?;
        forward_one_token(handle, weights, state, next_token, next_logits)?;
        context.push(next_token);
        return Ok(SpeculativeRoundResult {
            generated: vec![next_token],
            drafted: 0,
            accepted: 0,
        });
    }
    let d = draft.len();

    state.snapshot_gdn_state()?;
    forward_verify_chunk(handle, weights, state, &draft)?;

    // predicted[i] is what the base model actually predicts comes after
    // draft[i-1] (or, for i=0, what it predicted BEFORE this round even
    // started -- `next_logits`, already real and already available).
    // Accept while draft[i] == predicted[i].
    let mut accepted = 0usize;
    let mut predicted = Vec::with_capacity(d + 1);
    predicted.push(argmax_sample(next_logits)?);
    for i in 0..d {
        if i > 0 {
            predicted.push(crate::kernels::argmax_bf16_at(&state.prefill_scratch.verify_logits, i - 1, VOCAB_SIZE)?);
        }
        if predicted[i] == draft[i] {
            accepted += 1;
        } else {
            break;
        }
    }

    let corrected_token = if accepted == d {
        // Full accept: every draft token was correct, GDN state and KV
        // cache are already fully correct (verify wrote real, accepted
        // data for every position) -- the last verify row's own
        // prediction is the real "bonus" token, still needing ONE
        // `forward_one_token` call below to actually be processed.
        crate::kernels::argmax_bf16_at(&state.prefill_scratch.verify_logits, d - 1, VOCAB_SIZE)?
    } else {
        // Partial/no accept: `predicted[accepted]` is the real corrected
        // token at the first wrong position. Roll position back to BEFORE
        // this whole round (undoing all `d` of verify's speculative
        // advances, not just the rejected tail -- the replay loop below
        // re-advances it, one real token at a time, for exactly the
        // accepted prefix), restore GDN state to the same pre-round
        // snapshot, then replay the accepted prefix (cheap: `accepted <=
        // d <= num_draft` calls) to bring GDN state back up to date for
        // exactly those tokens (KV cache entries for the accepted prefix
        // are already correct and get harmlessly, identically overwritten
        // again by the replay).
        state.position -= d;
        state.restore_gdn_state()?;
        for &tok in &draft[..accepted] {
            forward_one_token(handle, weights, state, tok, next_logits)?;
        }
        predicted[accepted]
    };

    forward_one_token(handle, weights, state, corrected_token, next_logits)?;

    let mut generated = draft[..accepted].to_vec();
    generated.push(corrected_token);
    context.extend_from_slice(&generated);

    Ok(SpeculativeRoundResult {
        generated,
        drafted: d,
        accepted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hip;
    use crate::model_loader::locate_model_snapshot;

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

    /// §96: THE decisive correctness gate for speculative decoding --
    /// real prompt, real weights, a REPETITIVE prompt deliberately chosen
    /// to make `prompt_lookup_draft` actually find matches (so this test
    /// exercises real accept AND real reject paths, not just the trivial
    /// "draft always empty, degrades to plain decoding" case). Runs the
    /// same real generation twice: once through `speculative_round` in a
    /// loop, once through plain sequential `forward_one_token` -- greedy
    /// speculative decoding is REQUIRED to reproduce the exact same
    /// output as plain greedy decoding, token for token (this is the
    /// whole point of verifying against the base model's own real
    /// predictions rather than trusting the draft) -- any real bug in the
    /// accept/reject/rollback/replay protocol would show up as a
    /// real divergence here, not just a slowdown.
    #[test]
    #[ignore]
    fn real_speculative_decode_matches_real_sequential_greedy_generation() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        eprintln!("loading real weights for all {NUM_LAYERS} layers...");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        eprintln!("weights loaded.");
        let tok = crate::tokenizer::ChatTokenizer::load(&snapshot).expect("real tokenizer.json failed to load");

        let prompt = "Repeat the following sentence exactly four times, with a space between each repetition, and nothing else: The quick brown fox jumps over the lazy dog. Repetition:";
        let prompt_ids = tok.encode(prompt).expect("real encode failed");
        let num_new_tokens = 40usize;
        let ngram_size = 3usize;
        let num_draft = 6usize;
        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        let max_seq_len = 256usize;

        // Path 1: real speculative decoding.
        let mut state_spec = DecodeState::new(max_seq_len).expect("DecodeState allocation failed (speculative)");
        let mut logits_spec: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        forward_prefill(&handle, &weights, &mut state_spec, &prompt_ids, &mut logits_spec).expect("forward_prefill failed");
        let mut context = prompt_ids.clone();
        let mut generated_spec = Vec::with_capacity(num_new_tokens);
        let mut total_drafted = 0usize;
        let mut total_accepted = 0usize;
        let mut rounds = 0usize;
        while generated_spec.len() < num_new_tokens {
            let result = speculative_round(&handle, &weights, &mut state_spec, &mut context, &mut logits_spec, ngram_size, num_draft)
                .expect("speculative_round failed");
            total_drafted += result.drafted;
            total_accepted += result.accepted;
            rounds += 1;
            generated_spec.extend(result.generated);
        }
        generated_spec.truncate(num_new_tokens);
        eprintln!(
            "speculative decoding: {rounds} rounds, {total_drafted} tokens drafted, {total_accepted} accepted ({:.1}% acceptance)",
            if total_drafted > 0 { 100.0 * total_accepted as f64 / total_drafted as f64 } else { 0.0 }
        );
        assert!(total_drafted > 0, "test is meaningless if prompt_lookup_draft never found a match -- prompt needs more repetition");
        assert!(total_accepted > 0, "test should exercise at least one real accept");
        assert!(total_accepted < total_drafted, "test should exercise at least one real reject (drafted > accepted)");

        // Path 2: plain sequential greedy decoding, same real prompt.
        let mut state_seq = DecodeState::new(max_seq_len).expect("DecodeState allocation failed (sequential)");
        let mut logits_seq: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        forward_prefill(&handle, &weights, &mut state_seq, &prompt_ids, &mut logits_seq).expect("forward_prefill failed");
        let mut generated_seq = Vec::with_capacity(num_new_tokens);
        for _ in 0..num_new_tokens {
            let next_id = argmax_sample(&logits_seq).expect("argmax_sample failed");
            generated_seq.push(next_id);
            forward_one_token(&handle, &weights, &mut state_seq, next_id, &mut logits_seq).expect("forward_one_token failed");
        }

        eprintln!("generated (speculative): {generated_spec:?}");
        eprintln!("generated (sequential):  {generated_seq:?}");
        assert_eq!(
            generated_spec, generated_seq,
            "speculative decoding must reproduce EXACTLY the same output as plain sequential greedy decoding"
        );
    }

    /// §96: the real A/B this whole feature exists to answer -- does
    /// speculative decoding (prompt-lookup drafting + real batched
    /// verification) actually make wall-clock generation faster on this
    /// engine, and does that answer depend on the prompt's own content?
    /// Both arms use the SAME plain eager `forward_one_token` as their
    /// baseline decode primitive (speculative decoding's own fallback/
    /// replay path already calls it) -- NOT the faster `GraphedDecodeState`
    /// path the HTTP server uses for steady-state decode, since
    /// speculative decoding isn't wired into the graphed path in this
    /// pass (a real, separate integration question, not attempted here).
    /// Two real prompts, deliberately chosen for contrast: one with heavy
    /// literal repetition (where prompt-lookup should find real matches
    /// often) and one open-ended/creative (where it structurally can't).
    /// `#[ignore]`d like every other `bench_real_*` test -- real GPU time,
    /// not part of the default suite.
    #[test]
    #[ignore]
    fn bench_real_speculative_vs_sequential_decode() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        eprintln!("loading real weights for all {NUM_LAYERS} layers...");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        eprintln!("weights loaded.");
        let tok = crate::tokenizer::ChatTokenizer::load(&snapshot).expect("real tokenizer.json failed to load");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        let prompts: [(&str, &str); 2] = [
            (
                "repetitive_boilerplate",
                "Write a Python dictionary literal named config with keys \"field_1\" through \"field_24\", each mapped to the empty string \"\", one key-value pair per line, formatted exactly like:\nconfig = {\n    \"field_1\": \"\",\n    \"field_2\": \"\",\nContinue this pattern for all 24 fields and then close the dictionary. Output only the code.",
            ),
            (
                "creative_low_repetition",
                "Write a short, original paragraph describing an imaginary alien marketplace, using vivid and varied sensory imagery -- avoid repeating words or sentence structures.",
            ),
        ];

        let ngram_size = 3usize;
        let num_draft = 6usize;
        let num_tokens = 120usize;
        let max_seq_len = 512usize;

        for (label, prompt_text) in prompts {
            let prompt_ids = tok.encode(prompt_text).expect("real encode failed");

            // Arm 1: plain sequential eager decode.
            let mut state_seq = DecodeState::new(max_seq_len).expect("DecodeState allocation failed (sequential)");
            let mut logits_seq: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
            forward_prefill(&handle, &weights, &mut state_seq, &prompt_ids, &mut logits_seq).expect("forward_prefill failed");
            let t0 = std::time::Instant::now();
            for _ in 0..num_tokens {
                let next_id = argmax_sample(&logits_seq).expect("argmax_sample failed");
                forward_one_token(&handle, &weights, &mut state_seq, next_id, &mut logits_seq).expect("forward_one_token failed");
            }
            let seq_elapsed = t0.elapsed();
            let seq_tok_s = num_tokens as f64 / seq_elapsed.as_secs_f64();

            // Arm 2: speculative decoding.
            let mut state_spec = DecodeState::new(max_seq_len).expect("DecodeState allocation failed (speculative)");
            let mut logits_spec: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
            forward_prefill(&handle, &weights, &mut state_spec, &prompt_ids, &mut logits_spec).expect("forward_prefill failed");
            let mut context = prompt_ids.clone();
            let mut generated = 0usize;
            let mut total_drafted = 0usize;
            let mut total_accepted = 0usize;
            let mut rounds = 0usize;
            let t0 = std::time::Instant::now();
            while generated < num_tokens {
                let result = speculative_round(&handle, &weights, &mut state_spec, &mut context, &mut logits_spec, ngram_size, num_draft)
                    .expect("speculative_round failed");
                generated += result.generated.len();
                total_drafted += result.drafted;
                total_accepted += result.accepted;
                rounds += 1;
            }
            let spec_elapsed = t0.elapsed();
            let spec_tok_s = generated as f64 / spec_elapsed.as_secs_f64();

            let acceptance_pct = if total_drafted > 0 { 100.0 * total_accepted as f64 / total_drafted as f64 } else { 0.0 };
            println!(
                "[{label}] sequential: {seq_tok_s:.2} tok/s | speculative: {spec_tok_s:.2} tok/s ({generated} tokens, {rounds} rounds, {total_drafted} drafted, {total_accepted} accepted, {acceptance_pct:.1}% acceptance) | speedup: {:.3}x",
                spec_tok_s / seq_tok_s
            );
        }
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
                raw::gemm(handle_raw, state.scratch.final_normed.as_device_ptr(), weights.embed_tokens.as_device_ptr(), logits.as_device_ptr_mut(), 1, HIDDEN_SIZE as i32, VOCAB_SIZE as i32, std::ptr::null_mut());
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
