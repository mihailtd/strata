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
        unsafe { kernels_ffi::launch_rmsnorm_bf16(x, weight, out, num_rows, hidden_size, eps, 256, stream) };
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
    #[allow(clippy::too_many_arguments, unused_variables)]
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
        unsafe {
            kernels_ffi::launch_gemv_bf16(x, w, y, out_features, in_features, 256, stream);
        }
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
        Ok(DecodeState {
            layers,
            max_seq_len,
            position: 0,
            scratch,
            hidden_a: DeviceBuffer::alloc(HIDDEN_SIZE)?,
            hidden_b: DeviceBuffer::alloc(HIDDEN_SIZE)?,
        })
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
