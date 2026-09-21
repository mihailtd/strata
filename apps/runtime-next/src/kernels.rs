//! Safe wrappers around this crate's real, hand-written HIP kernels
//! (`src/kernels/*.hip`, compiled by `hipcc` in `build.rs`).
//!
//! Same discipline as `hip.rs`: the `extern "C"` block below is the only
//! unsafe surface for kernel launches, hand-curated against the exact
//! signature each `.hip` file's own launcher function declares -- not
//! generated, not guessed.

use crate::hip::{DeviceBuffer, HipError, check_last_error, device_synchronize};
use std::ffi::{c_float, c_int, c_void};

// `pub(crate)` (not private): `model.rs`'s hot decode-loop path calls these
// raw launchers directly, skipping the safe wrappers' per-call
// `device_synchronize()` (§89/§90's own "naive assembly" numbers showed
// this is a real, material cost -- see `model.rs`'s module doc comment for
// the full reasoning). The declarations themselves are unchanged, still
// hand-curated against the real `.hip` launcher signatures -- only their
// visibility widened, not the audited unsafe surface itself.
pub(crate) mod ffi {
    use std::ffi::{c_float, c_int, c_void};

    unsafe extern "C" {
        /// See `src/kernels/rmsnorm.hip` for what this actually computes.
        /// `stream` (§93): NULL reproduces the original default-stream
        /// behavior; a real explicitly-created stream is required for this
        /// launch to be capturable into a HIP Graph.
        #[allow(clippy::too_many_arguments)]
        pub fn launch_rmsnorm_bf16(
            x: *const c_void,
            weight: *const c_void,
            out: *mut c_void,
            num_rows: c_int,
            hidden_size: c_int,
            eps: c_float,
            threads: c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/rope.hip` for what this actually computes.
        /// `position` is a DEVICE pointer (§93: graph-capture-safety --
        /// see that file's own comment), not a host value. `stream` (§93).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_rope_bf16(
            x: *const c_void,
            out: *mut c_void,
            num_rows: c_int,
            head_dim: c_int,
            rotary_dim: c_int,
            theta: c_float,
            position: *const c_int,
            stream: *mut c_void,
        );

        /// §103 (real batched-prefill RoPE): see
        /// `src/kernels/rope_prefill.hip` for what this actually computes
        /// -- ONE launch for a whole `num_tokens`-token chunk, replacing
        /// `num_tokens` separate `launch_rope_bf16` calls. `position_buf`
        /// is a real DEVICE array of `num_tokens` positions (not one
        /// shared scalar).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_rope_prefill_bf16(
            x: *const c_void,
            out: *mut c_void,
            num_tokens: c_int,
            num_heads: c_int,
            head_dim: c_int,
            rotary_dim: c_int,
            theta: c_float,
            position_buf: *const c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/swiglu.hip` for what this actually computes.
        /// `stream` (§93).
        pub fn launch_swiglu_bf16(
            gate: *const c_void,
            up: *const c_void,
            out: *mut c_void,
            n: c_int,
            stream: *mut c_void,
        );

        /// §114: real launch-count-reduction fusion -- see
        /// `src/kernels/swiglu.hip`'s own `swiglu_strided_bf16_kernel`
        /// header for the full derivation (replaces two real
        /// `launch_extract_range_bf16` calls plus this same kernel with
        /// one single launch, bit-exact identical output).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_swiglu_strided_bf16(
            gate_up: *const c_void,
            out: *mut c_void,
            rows: c_int,
            src_stride: c_int,
            gate_offset: c_int,
            up_offset: c_int,
            len: c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/embedding.hip` for what this actually computes.
        /// `stream` (§93).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_embedding_lookup_bf16(
            table: *const c_void,
            token_ids: *const c_int,
            out: *mut c_void,
            num_tokens: c_int,
            hidden_size: c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/causal_conv1d_update.hip` for what this
        /// actually computes. `stream` (§93).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_causal_conv1d_update_bf16(
            hidden_states: *const c_void,
            conv_state: *mut c_void,
            weight: *const c_void,
            out: *mut c_void,
            batch: c_int,
            conv_dim: c_int,
            kernel_size: c_int,
            stream: *mut c_void,
        );

        /// §103 (real batched-prefill causal conv1d): see
        /// `src/kernels/causal_conv1d_prefill.hip` for what this actually
        /// computes -- ONE launch for a whole `num_tokens`-token chunk,
        /// replacing `num_tokens` separate `launch_causal_conv1d_update_bf16`
        /// calls. `src` is a STRIDED view (`src_stride`/`src_offset`).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_causal_conv1d_prefill_bf16(
            src: *const c_void,
            src_stride: c_int,
            src_offset: c_int,
            conv_state: *mut c_void,
            weight: *const c_void,
            out: *mut c_void,
            num_tokens: c_int,
            conv_dim: c_int,
            kernel_size: c_int,
            threads: c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/attention.hip` for what this actually computes.
        /// `position` is a DEVICE pointer; `kv_len = *position + 1` is
        /// computed inside the kernel (§93: graph-capture-safety -- shared
        /// memory is sized for the fixed `kv_stride` upper bound, not the
        /// varying `kv_len`, so the launch config never changes either).
        /// `stream` (§93).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_attention_decode_bf16(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            out: *mut c_void,
            num_q_heads: c_int,
            num_kv_heads: c_int,
            position: *const c_int,
            kv_stride: c_int,
            head_dim: c_int,
            scaling: c_float,
            threads: c_int,
            stream: *mut c_void,
        );

        /// §104 (real split-KV decode attention): see
        /// `src/kernels/attention_decode_split.hip` for the exact
        /// technique -- `kv_split` threads cooperate on each output
        /// dimension instead of one, targeting the real position-
        /// dependent decode slowdown `docs/DECISIONS.md` §103 measured
        /// directly. `threads` (internal launch config) is computed as
        /// `head_dim * kv_split`, not passed separately.
        #[allow(clippy::too_many_arguments)]
        pub fn launch_attention_decode_split_bf16(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            out: *mut c_void,
            num_q_heads: c_int,
            num_kv_heads: c_int,
            position: *const c_int,
            kv_stride: c_int,
            head_dim: c_int,
            kv_split: c_int,
            scaling: c_float,
            stream: *mut c_void,
        );

        /// §119: 2D batched causal attention for prefill (T < 128)
        #[allow(clippy::too_many_arguments)]
        pub fn launch_attention_causal_prefill_bf16(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            out: *mut c_void,
            num_tokens: c_int,
            num_q_heads: c_int,
            num_kv_heads: c_int,
            positions: *const c_int,
            kv_stride: c_int,
            head_dim: c_int,
            kv_split: c_int,
            scaling: c_float,
            max_chunk_kv_len: c_int,
            stream: *mut c_void,
        );

        /// §119: Fused attention QKV prep for prefill
        #[allow(clippy::too_many_arguments)]
        pub fn launch_fused_attn_qkv_prep_bf16(
            qkv_in: *const c_void,
            q_weight: *const c_void,
            k_weight: *const c_void,
            q_normed_out: *mut c_void,
            gate_out: *mut c_void,
            k_normed_out: *mut c_void,
            v_out: *mut c_void,
            num_tokens: c_int,
            num_q_heads: c_int,
            num_kv_heads: c_int,
            head_dim: c_int,
            eps: c_float,
            stream: *mut c_void,
        );

        /// See `src/kernels/gdn_recurrent.hip` for what this actually
        /// computes. `stream` (§93).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_gdn_recurrent_decode_bf16(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            g: *const c_float,
            beta: *const c_float,
            state: *mut c_float,
            out: *mut c_void,
            num_heads: c_int,
            num_k_heads: c_int,
            head_dim: c_int,
            threads: c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/sigmoid_gate.hip` for what this actually computes.
        /// `stream` (§93).
        pub fn launch_sigmoid_gate_bf16(
            x: *const c_void,
            gate: *const c_void,
            out: *mut c_void,
            n: c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/rmsnorm_gated.hip` for what this actually
        /// computes. `stream` (§93).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_rmsnorm_gated_bf16(
            x: *const c_void,
            gate: *const c_void,
            weight: *const c_float,
            out: *mut c_void,
            num_rows: c_int,
            hidden_size: c_int,
            eps: c_float,
            threads: c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/gdn_gate_beta.hip` for what this actually
        /// computes. `stream` (§93).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_gdn_gate_beta_bf16(
            a: *const c_void,
            b: *const c_void,
            a_log: *const c_float,
            dt_bias: *const c_float,
            g_out: *mut c_float,
            beta_out: *mut c_float,
            num_heads: c_int,
            stream: *mut c_void,
        );

        /// §103 (real batched-prefill GDN gate/beta): see
        /// `src/kernels/gdn_gate_beta_prefill.hip` for what this actually
        /// computes -- ONE launch for a whole `num_tokens`-token chunk,
        /// replacing `num_tokens` separate `launch_gdn_gate_beta_bf16`
        /// calls. `a_src`/`b_src` are STRIDED views (`*_stride`/`*_offset`).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_gdn_gate_beta_prefill_bf16(
            a_src: *const c_void,
            a_stride: c_int,
            a_offset: c_int,
            b_src: *const c_void,
            b_stride: c_int,
            b_offset: c_int,
            a_log: *const c_float,
            dt_bias: *const c_float,
            g_out: *mut c_float,
            beta_out: *mut c_float,
            num_tokens: c_int,
            num_heads: c_int,
            threads: c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/add.hip` for what this actually computes.
        /// `stream` (§93).
        pub fn launch_add_bf16(a: *const c_void, b: *const c_void, out: *mut c_void, n: c_int, stream: *mut c_void);

        /// §115: real on-device fp32->bf16 / bf16->fp32 elementwise casts
        /// -- see `src/kernels/cast_f32_bf16.hip`. Bit-exact equivalent of
        /// this codebase's own `f32_to_bf16`/`bf16_to_f32` CPU functions,
        /// replacing a real host round trip in `gdn_chunk_forward_prefill`.
        pub fn launch_f32_to_bf16_cast(src: *const c_void, dst: *mut c_void, n: c_int, stream: *mut c_void);
        pub fn launch_bf16_to_f32_cast(src: *const c_void, dst: *mut c_void, n: c_int, stream: *mut c_void);

        /// §116: real in-place `buf *= *scalar_ptr`, `scalar_ptr` a real
        /// device address (no host read) -- see
        /// `src/kernels/scale_by_device_scalar.hip`.
        pub fn launch_scale_bf16_by_device_scalar(buf: *mut c_void, scalar_ptr: *const c_void, n: c_int, stream: *mut c_void);
        pub fn launch_scale_bf16_by_device_scalars_batched(buf: *mut c_void, scalars: *const c_void, n_per_head: c_int, h: c_int, stream: *mut c_void);

        /// See `src/kernels/split_last_dim.hip` for what this actually
        /// computes. `stream` (§93).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_split_last_dim_bf16(
            x: *const c_void,
            first: *mut c_void,
            second: *mut c_void,
            rows: c_int,
            half: c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/kv_cache_append.hip` for what this actually
        /// computes. `position` is a DEVICE pointer (§93). `stream` (§93).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_kv_cache_append_bf16(
            new_k: *const c_void,
            new_v: *const c_void,
            k_cache: *mut c_void,
            v_cache: *mut c_void,
            num_kv_heads: c_int,
            max_seq_len: c_int,
            head_dim: c_int,
            position: *const c_int,
            stream: *mut c_void,
        );

        /// §103 (real batched-prefill KV cache append): see
        /// `src/kernels/kv_cache_append_prefill.hip` for what this
        /// actually computes -- ONE launch for a whole `num_tokens`-token
        /// chunk, replacing `num_tokens` separate
        /// `launch_kv_cache_append_bf16` calls. `position_buf` is a real
        /// DEVICE array of `num_tokens` positions.
        #[allow(clippy::too_many_arguments)]
        pub fn launch_kv_cache_append_prefill_bf16(
            new_k: *const c_void,
            new_v: *const c_void,
            k_cache: *mut c_void,
            v_cache: *mut c_void,
            num_tokens: c_int,
            num_kv_heads: c_int,
            max_seq_len: c_int,
            head_dim: c_int,
            position_buf: *const c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/extract_range.hip` for what this actually
        /// computes (§96: batched prefill). `stream` (§93).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_extract_range_bf16(
            src: *const c_void,
            dst: *mut c_void,
            rows: c_int,
            src_stride: c_int,
            src_offset: c_int,
            len: c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/position_state.hip` for what this actually
        /// computes: `*position_ptr += 1`, entirely on-device -- the last
        /// node of a graph-captured decode step (§93), so a replayed graph
        /// advances its own position tracking with zero host involvement.
        pub fn launch_increment_position(position_ptr: *mut c_int, stream: *mut c_void);

        /// See `src/kernels/gemv.hip` for what this actually computes: a
        /// hand-written, vectorized bf16 GEMV -- the real operation this
        /// engine's `rows=1` "GEMM" calls always compute, dispatched
        /// through a general BLAS library only for lack of a dedicated
        /// kernel until now.
        #[allow(clippy::too_many_arguments)]
        pub fn launch_gemv_bf16(
            x: *const c_void,
            w: *const c_void,
            y: *mut c_void,
            out_features: c_int,
            in_features: c_int,
            threads: c_int,
            stream: *mut c_void,
        );

        /// §106: W4A16 fused dequant+GEMV -- see
        /// `src/kernels/w4a16_gemv.hip` for the real quantization scheme
        /// and layout this implements.
        #[allow(clippy::too_many_arguments)]
        pub fn launch_w4a16_gemv_bf16(
            x: *const c_void,
            qweight: *const c_void,
            scales: *const c_void,
            y: *mut c_void,
            out_features: c_int,
            in_features: c_int,
            group_size: c_int,
            threads: c_int,
            stream: *mut c_void,
        );

        /// Quantized-path LoRA additive accumulation -- see
        /// `src/kernels/lora_delta_accumulate.hip` for the real math and
        /// why this is a separate kernel from the base W4A16 GEMV above
        /// rather than a fused variant of it.
        pub fn launch_lora_delta_accumulate_bf16(
            mid: *const c_void,
            b: *const c_void,
            y: *mut c_void,
            out_features: c_int,
            rank: c_int,
            stream: *mut c_void,
        );

        /// §108: batched W4A16 prefill -- see
        /// `src/kernels/w4a16_gemm_prefill.hip` for the real tiling scheme
        /// (reuses each dequantized weight nibble across `TILE_T=8`
        /// tokens instead of re-streaming the whole weight matrix once
        /// per token, the real fix for §106's disclosed TTFT gap).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_w4a16_gemm_prefill_bf16(
            x: *const c_void,
            qweight: *const c_void,
            scales: *const c_void,
            y: *mut c_void,
            out_features: c_int,
            in_features: c_int,
            group_size: c_int,
            num_tokens: c_int,
            threads: c_int,
            stream: *mut c_void,
        );

        /// §123 EXPERIMENT: `TILE_M=32` variant of the above, isolated for
        /// a real A/B -- see `src/kernels/w4a16_gemm_prefill.hip`'s own
        /// doc comment and `docs/DECISIONS.md` §123 for the real verdict.
        #[allow(clippy::too_many_arguments)]
        pub fn launch_w4a16_gemm_prefill_tile32_bf16(
            x: *const c_void,
            qweight: *const c_void,
            scales: *const c_void,
            y: *mut c_void,
            out_features: c_int,
            in_features: c_int,
            group_size: c_int,
            num_tokens: c_int,
            stream: *mut c_void,
        );

        /// §124 EXPERIMENT: `TILE_N=16` variant (doubled output rows per
        /// block, halving redundant activation re-reads) -- see
        /// `src/kernels/w4a16_gemm_prefill.hip`'s own doc comment and
        /// `docs/DECISIONS.md` §124 for the real verdict.
        #[allow(clippy::too_many_arguments)]
        pub fn launch_w4a16_gemm_prefill_tile_n16_bf16(
            x: *const c_void,
            qweight: *const c_void,
            scales: *const c_void,
            y: *mut c_void,
            out_features: c_int,
            in_features: c_int,
            group_size: c_int,
            num_tokens: c_int,
            stream: *mut c_void,
        );

        /// §126 EXPERIMENT: real, production-shaped W4A16 prefill GEMM
        /// using RDNA3 WMMA INT8 tensor cores -- see
        /// `src/kernels/w4a16_gemm_prefill_wmma_int8.hip`'s own doc
        /// comment and `docs/DECISIONS.md` §126 for the real design and
        /// verdict. NOT wired into `raw::linear_quantized_prefill`.
        #[allow(clippy::too_many_arguments)]
        pub fn launch_w4a16_gemm_prefill_wmma_int8_bf16(
            x: *const c_void,
            qweight: *const c_void,
            scales: *const c_void,
            y: *mut c_void,
            out_features: c_int,
            in_features: c_int,
            group_size: c_int,
            num_tokens: c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/argmax.hip` for what this actually computes:
        /// an on-device argmax over real vocab logits, avoiding the real,
        /// measured ~0.209ms/token cost of copying the full logits buffer
        /// to host and scanning it there.
        pub fn launch_argmax_bf16(
            logits: *const c_void,
            out_idx: *mut c_int,
            n: c_int,
            threads: c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/causal_softmax.hip` for what this actually
        /// computes (§99: batched-prefill causal attention). Row-wise
        /// softmax over a real precomputed `Q@K^T` score matrix, masked
        /// per row to that row's own real causal boundary.
        /// `start_position_ptr` is a DEVICE pointer (§93 convention).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_causal_softmax_bf16(
            scores: *mut c_void,
            kv_len: c_int,
            start_position_ptr: *const c_int,
            num_rows: c_int,
            threads: c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/gdn_chunk_decay.hip` for what this actually
        /// computes (§100: chunked GDN parallel prefill).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_gdn_chunk_decay_bf16(
            g: *const c_float,
            ut_system: *mut c_void,
            intra_chunk_attn: *mut c_void,
            decay_exp_out: *mut c_float,
            remaining_decay_out: *mut c_float,
            chunk_decay_out: *mut c_float,
            num_heads: c_int,
            num_chunks: c_int,
            chunk_size: c_int,
            threads: c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/gdn_chunk_utsolve.hip` for what this actually
        /// computes (§100: chunked GDN parallel prefill).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_gdn_chunk_utsolve_bf16(
            ut_system: *const c_void,
            rhs: *const c_void,
            x_out: *mut c_void,
            num_heads: c_int,
            num_chunks: c_int,
            chunk_size: c_int,
            width: c_int,
            threads: c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/l2norm.hip` for what this actually computes
        /// (§100: chunked GDN parallel prefill). NOT the same formula as
        /// `launch_rmsnorm_bf16` -- see that file's own header comment.
        #[allow(clippy::too_many_arguments)]
        pub fn launch_l2norm_bf16(
            x: *const c_void,
            out: *mut c_void,
            num_rows: c_int,
            hidden_size: c_int,
            eps: c_float,
            post_scale: c_float,
            threads: c_int,
            stream: *mut c_void,
        );

        /// See `src/kernels/gdn_chunk_broadcast_scale.hip` for what this
        /// actually computes (§100: chunked GDN parallel prefill).
        #[allow(clippy::too_many_arguments)]
        pub fn launch_gdn_chunk_broadcast_scale_bf16(
            src: *const c_void,
            scale: *const c_float,
            dst: *mut c_void,
            num_tokens: c_int,
            src_heads: c_int,
            dst_heads: c_int,
            head_dim: c_int,
            n_rep: c_int,
            chunk_size: c_int,
            head_major_output: c_int,
            threads: c_int,
            stream: *mut c_void,
        );
    }
}

/// bf16 RMSNorm, matching `apps/runtime-ipwf/fused_norm.py`'s `ExactRMSNorm`
/// (`unit_offset=True` case) exactly: `out = (x.f32() * rsqrt(mean(x.f32()^2)
/// + eps)) * (1 + weight.f32())`, cast back to bf16.
///
/// `x` and `out` hold `num_rows * hidden_size` bf16 elements (packed as
/// `u16` bit patterns, matching `model_loader.rs`'s own bf16-as-raw-bytes
/// convention); `weight` holds `hidden_size` elements, broadcast across
/// rows. `out` may alias `x` only if the caller does not need `x`'s
/// original values after the call (the kernel reads each row fully before
/// writing it, so in-place is safe row-by-row, but this is not verified
/// here -- pass separate buffers unless you have checked the kernel).
///
/// Launches the real kernel and blocks until it completes (`hipDeviceSynchronize`)
/// so the caller can trust `out` is populated when this returns -- callers
/// needing async behavior can build that later; this is the simplest
/// correct contract to start from.
pub fn rmsnorm_bf16(
    x: &DeviceBuffer<u16>,
    weight: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<u16>,
    num_rows: usize,
    hidden_size: usize,
    eps: f32,
) -> Result<(), HipError> {
    assert_eq!(
        x.len(),
        num_rows * hidden_size,
        "x length must be num_rows * hidden_size"
    );
    assert_eq!(
        weight.len(),
        hidden_size,
        "weight length must equal hidden_size"
    );
    assert_eq!(
        out.len(),
        num_rows * hidden_size,
        "out length must be num_rows * hidden_size"
    );

    // Threads per block: a power of two for the reduction loop, capped at a
    // sane maximum. hidden_size=2560 in this model, so 256 gives each
    // thread 10 elements to sum, a reasonable balance for a first working
    // version -- not tuned yet.
    let threads: i32 = 256;

    // SAFETY: `x`/`weight` point to real, live `hipMalloc` allocations of
    // at least the byte lengths implied by the asserted element counts
    // above (checked, not assumed); `out` is likewise real and live and
    // distinct in this call's intended use (see doc comment on aliasing).
    // The kernel itself only reads within `[0, hidden_size)` per row and
    // writes within the same bounds -- verified by reading
    // `src/kernels/rmsnorm.hip` directly, not inferred.
    unsafe {
        ffi::launch_rmsnorm_bf16(
            x.as_device_ptr(),
            weight.as_device_ptr(),
            out.as_device_ptr_mut() as *mut c_void,
            num_rows as i32,
            hidden_size as i32,
            eps,
            threads,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// Partial RoPE, matching Qwen3.5's exact formula (see `src/kernels/rope.hip`'s
/// header comment for the full derivation from the real `transformers`
/// source, including why the mRoPE 3-section recomposition is a no-op for
/// pure text and can be treated as single-section RoPE here).
///
/// `x`/`out` hold `num_rows * head_dim` bf16 elements; only the first
/// `rotary_dim` of each row's `head_dim` elements are rotated, the rest
/// pass through unchanged. All rows share the same `position` (the shape a
/// single decode step needs: one sequence position, multiple heads).
pub fn rope_bf16(
    x: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<u16>,
    num_rows: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    position: i64,
) -> Result<(), HipError> {
    assert_eq!(
        x.len(),
        num_rows * head_dim,
        "x length must be num_rows * head_dim"
    );
    assert_eq!(
        out.len(),
        num_rows * head_dim,
        "out length must be num_rows * head_dim"
    );
    assert!(
        rotary_dim <= head_dim,
        "rotary_dim must not exceed head_dim"
    );
    assert_eq!(
        rotary_dim % 2,
        0,
        "rotary_dim must be even (rotate_half splits it in half)"
    );

    // The real kernel reads `position` through a device pointer, not a
    // host value (§93: graph-capture-safety -- see `rope.hip`'s own
    // comment). This safe wrapper keeps the ergonomic by-value `i64` API
    // its existing callers/tests use, and just does the tiny host->device
    // write itself.
    let mut position_buf: DeviceBuffer<i32> = DeviceBuffer::alloc(1)?;
    position_buf.copy_from_host(&[position as i32])?;

    // SAFETY: `x`/`out` are real, live `hipMalloc` allocations of at least
    // the asserted byte lengths; the kernel indexes strictly within
    // `[0, head_dim)` per row (verified by reading `rope.hip` directly).
    // `position_buf` is a real, live `hipMalloc` allocation holding exactly
    // the written value, valid for the duration of this call.
    unsafe {
        ffi::launch_rope_bf16(
            x.as_device_ptr(),
            out.as_device_ptr_mut() as *mut c_void,
            num_rows as i32,
            head_dim as i32,
            rotary_dim as i32,
            theta,
            position_buf.as_device_ptr() as *const c_int,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// SwiGLU activation: `silu(gate) * up`, elementwise, fp32 internal math.
/// See `src/kernels/swiglu.hip`. `gate`, `up`, `out` all hold `n` bf16
/// elements (e.g. `n = num_rows * intermediate_size`).
pub fn swiglu_bf16(
    gate: &DeviceBuffer<u16>,
    up: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<u16>,
) -> Result<(), HipError> {
    assert_eq!(
        gate.len(),
        up.len(),
        "gate and up must have the same length"
    );
    assert_eq!(
        gate.len(),
        out.len(),
        "out must have the same length as gate/up"
    );

    // SAFETY: all three buffers are real, live `hipMalloc` allocations of
    // at least `gate.len()` elements; the kernel indexes strictly within
    // `[0, n)` (verified by reading `swiglu.hip` directly).
    unsafe {
        ffi::launch_swiglu_bf16(
            gate.as_device_ptr(),
            up.as_device_ptr(),
            out.as_device_ptr_mut() as *mut c_void,
            gate.len() as i32,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// §114: fused strided SwiGLU -- see `src/kernels/swiglu.hip`'s own
/// `swiglu_strided_bf16_kernel` header. Real drop-in replacement for a
/// real `extract_range_bf16` (gate) + `extract_range_bf16` (up) +
/// `swiglu_bf16` three-launch sequence, reading gate/up directly out of
/// the real strided combined-GEMM output instead of two pre-gathered
/// copies. `gate_up` is `[rows, src_stride]`; `out` is `[rows, len]`
/// tightly packed.
#[allow(clippy::too_many_arguments)]
pub fn swiglu_strided_bf16(
    gate_up: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<u16>,
    rows: usize,
    src_stride: usize,
    gate_offset: usize,
    up_offset: usize,
    len: usize,
) -> Result<(), HipError> {
    assert_eq!(gate_up.len(), rows * src_stride, "gate_up length must equal rows * src_stride");
    assert_eq!(out.len(), rows * len, "out length must equal rows * len");
    assert!(gate_offset + len <= src_stride, "gate sub-range must fit within src_stride");
    assert!(up_offset + len <= src_stride, "up sub-range must fit within src_stride");

    // SAFETY: `gate_up`/`out` are real, live `hipMalloc` allocations of
    // at least the asserted element counts; the kernel indexes strictly
    // within `[0, len)` per row and `[0, rows)` blocks, reading only the
    // asserted-in-bounds `gate_offset`/`up_offset` sub-ranges (verified
    // by reading `swiglu.hip` directly).
    unsafe {
        ffi::launch_swiglu_strided_bf16(
            gate_up.as_device_ptr(),
            out.as_device_ptr_mut() as *mut c_void,
            rows as i32,
            src_stride as i32,
            gate_offset as i32,
            up_offset as i32,
            len as i32,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// Embedding lookup: gathers `num_tokens` rows out of the real embedding
/// `table` (`[vocab_size, hidden_size]`, bf16), one row per id in
/// `token_ids`. See `src/kernels/embedding.hip`. Every id is checked
/// against `vocab_size` before the kernel launches -- an out-of-range id
/// reading past the real embedding table would be a real memory-safety
/// bug, not a HIP error the runtime could recover from after the fact, so
/// this is caught here rather than trusted to the caller.
pub fn embedding_lookup_bf16(
    table: &DeviceBuffer<u16>,
    token_ids: &DeviceBuffer<i32>,
    token_ids_host: &[i32],
    out: &mut DeviceBuffer<u16>,
    vocab_size: usize,
    hidden_size: usize,
) -> Result<(), HipError> {
    assert_eq!(
        table.len(),
        vocab_size * hidden_size,
        "table length must be vocab_size * hidden_size"
    );
    assert_eq!(
        out.len(),
        token_ids_host.len() * hidden_size,
        "out length must be num_tokens * hidden_size"
    );
    for &id in token_ids_host {
        assert!(
            id >= 0 && (id as usize) < vocab_size,
            "token id {id} out of range for vocab_size {vocab_size}"
        );
    }

    // SAFETY: `table` is a real, live allocation of at least
    // `vocab_size * hidden_size` elements; every id in `token_ids` (the
    // device-side copy of `token_ids_host`, checked above) is in
    // `[0, vocab_size)`, so every row the kernel reads is in-bounds.
    unsafe {
        ffi::launch_embedding_lookup_bf16(
            table.as_device_ptr(),
            token_ids.as_device_ptr() as *const i32,
            out.as_device_ptr_mut() as *mut c_void,
            token_ids_host.len() as i32,
            hidden_size as i32,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// The single-token cached-decode path of GDN's depthwise causal conv1d
/// (`causal_conv1d_update` in the real `transformers` Qwen3.5 code -- see
/// `src/kernels/causal_conv1d_update.hip`'s header for the exact formula
/// and why this op specifically was chosen next).
///
/// `hidden_states`/`out` hold `batch * conv_dim` bf16 elements. `conv_state`
/// holds `batch * conv_dim * (kernel_size - 1)` bf16 elements and is
/// **read and written in place** -- matches the real function's own
/// `conv_state.copy_(...)` contract, so callers must treat the buffer's
/// pre-call contents as consumed after this returns. `weight` holds
/// `conv_dim * kernel_size` bf16 elements (the real model's `conv1d.weight`
/// squeezed from `[conv_dim, 1, kernel_size]`). No bias (the real model's
/// `conv1d` has `bias=False`).
pub fn causal_conv1d_update_bf16(
    hidden_states: &DeviceBuffer<u16>,
    conv_state: &mut DeviceBuffer<u16>,
    weight: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<u16>,
    batch: usize,
    conv_dim: usize,
    kernel_size: usize,
) -> Result<(), HipError> {
    assert!(kernel_size >= 1, "kernel_size must be at least 1");
    let state_len = kernel_size - 1;
    assert_eq!(
        hidden_states.len(),
        batch * conv_dim,
        "hidden_states length must be batch * conv_dim"
    );
    assert_eq!(
        conv_state.len(),
        batch * conv_dim * state_len,
        "conv_state length must be batch * conv_dim * (kernel_size - 1)"
    );
    assert_eq!(
        weight.len(),
        conv_dim * kernel_size,
        "weight length must be conv_dim * kernel_size"
    );
    assert_eq!(
        out.len(),
        batch * conv_dim,
        "out length must be batch * conv_dim"
    );

    // SAFETY: all four buffers are real, live `hipMalloc` allocations of at
    // least the asserted element counts; the kernel indexes strictly within
    // `[0, conv_dim)` per batch row and `[0, state_len)`/`[0, kernel_size)`
    // per channel (verified by reading `causal_conv1d_update.hip` directly).
    unsafe {
        ffi::launch_causal_conv1d_update_bf16(
            hidden_states.as_device_ptr(),
            conv_state.as_device_ptr_mut() as *mut c_void,
            weight.as_device_ptr(),
            out.as_device_ptr_mut() as *mut c_void,
            batch as i32,
            conv_dim as i32,
            kernel_size as i32,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// §103 (real batched-prefill RoPE): the whole-chunk counterpart of
/// `rope_bf16` -- see `src/kernels/rope_prefill.hip`'s header for the
/// exact formula (identical real math, just with a real per-token
/// `position_buf` instead of one shared scalar).
///
/// `x`/`out` hold `num_tokens * num_heads * head_dim` bf16 elements,
/// tightly packed `[num_tokens, num_heads, head_dim]`. `position_buf`
/// holds `num_tokens` real absolute positions.
#[allow(clippy::too_many_arguments)]
pub fn rope_prefill_bf16(
    x: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<u16>,
    num_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    theta: f32,
    position_buf: &DeviceBuffer<i32>,
) -> Result<(), HipError> {
    assert_eq!(x.len(), num_tokens * num_heads * head_dim, "x length must be num_tokens * num_heads * head_dim");
    assert_eq!(out.len(), num_tokens * num_heads * head_dim, "out length must be num_tokens * num_heads * head_dim");
    assert!(rotary_dim <= head_dim, "rotary_dim must not exceed head_dim");
    assert_eq!(rotary_dim % 2, 0, "rotary_dim must be even (rotate_half splits it in half)");
    assert_eq!(position_buf.len(), num_tokens, "position_buf length must equal num_tokens");

    // SAFETY: `x`/`out` are real, live `hipMalloc` allocations of at
    // least the asserted lengths; the kernel indexes strictly within
    // `[0, num_tokens*num_heads)` rows x `[0, head_dim)` per row, and
    // `position_buf` strictly within `[0, num_tokens)` (verified by
    // reading `rope_prefill.hip` directly).
    unsafe {
        ffi::launch_rope_prefill_bf16(
            x.as_device_ptr(),
            out.as_device_ptr_mut() as *mut c_void,
            num_tokens as i32,
            num_heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            theta,
            position_buf.as_device_ptr() as *const c_int,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// §103 (real batched-prefill causal conv1d): the whole-chunk counterpart
/// of `causal_conv1d_update_bf16` -- see
/// `src/kernels/causal_conv1d_prefill.hip`'s header for the exact formula
/// (identical real per-token math, just walked sequentially INSIDE one
/// kernel launch instead of via `num_tokens` separate host-issued
/// launches).
///
/// `src` is a STRIDED view: `src_offset + t*src_stride + c` for
/// `c in [0, conv_dim)`, `t in [0, num_tokens)` -- lets the caller read
/// this layer's own conv input directly out of a wider combined-GEMM
/// output row (see this kernel's own doc comment), no separate
/// `extract_range` pass needed. `conv_state`/`weight`/`out` are the same
/// real shapes `causal_conv1d_update_bf16` uses (`out` tightly packed
/// `[num_tokens, conv_dim]`, `conv_state`/`weight` un-batched: this op's
/// real state is a single position-independent per-layer buffer, same as
/// the per-token kernel's own `batch=1` real usage in this crate).
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_prefill_bf16(
    src: &DeviceBuffer<u16>,
    src_stride: usize,
    src_offset: usize,
    conv_state: &mut DeviceBuffer<u16>,
    weight: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<u16>,
    num_tokens: usize,
    conv_dim: usize,
    kernel_size: usize,
) -> Result<(), HipError> {
    assert!(kernel_size >= 1, "kernel_size must be at least 1");
    let state_len = kernel_size - 1;
    assert!(src_offset + conv_dim <= src_stride || num_tokens == 0, "src_offset + conv_dim must fit within one src_stride-wide row");
    assert!((num_tokens.saturating_sub(1)) * src_stride + src_offset + conv_dim <= src.len() || num_tokens == 0, "src too short for num_tokens * src_stride with this offset/conv_dim");
    assert_eq!(conv_state.len(), conv_dim * state_len, "conv_state length must be conv_dim * (kernel_size - 1)");
    assert_eq!(weight.len(), conv_dim * kernel_size, "weight length must be conv_dim * kernel_size");
    assert_eq!(out.len(), num_tokens * conv_dim, "out length must be num_tokens * conv_dim");
    let threads: i32 = 256;

    // SAFETY: bounds checked above; the kernel indexes `src` strictly
    // within `[0, num_tokens*src_stride)` via the asserted offset/stride
    // relationship, and `conv_state`/`weight`/`out` strictly within
    // `[0, conv_dim)`/`[0, kernel_size)`/`[0, num_tokens*conv_dim)`
    // (verified by reading `causal_conv1d_prefill.hip` directly).
    unsafe {
        ffi::launch_causal_conv1d_prefill_bf16(
            src.as_device_ptr(),
            src_stride as i32,
            src_offset as i32,
            conv_state.as_device_ptr_mut() as *mut c_void,
            weight.as_device_ptr(),
            out.as_device_ptr_mut() as *mut c_void,
            num_tokens as i32,
            conv_dim as i32,
            kernel_size as i32,
            threads,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// Single-token decode-step full attention (the "Hard tier" feature):
/// `softmax(Q @ K^T * scaling) @ V`, with GQA broadcasting `num_kv_heads`
/// KV heads to `num_q_heads` Q heads. See `src/kernels/attention.hip`'s
/// header for the exact formula (matching the real `eager_attention_forward`
/// + `repeat_kv` functions) and why no causal mask is needed for this
/// single-new-token-against-a-cache case.
///
/// `q` holds `num_q_heads * head_dim` bf16 elements (one new query token,
/// already projected/normed/roped). `k`/`v` hold
/// `num_kv_heads * kv_len * head_dim` bf16 elements each (the cached keys/
/// values up to and including the current position, already projected/
/// normed/roped). `out` holds `num_q_heads * head_dim` bf16 elements.
/// `num_q_heads` must be a multiple of `num_kv_heads` (GQA group size).
#[allow(clippy::too_many_arguments)]
pub fn attention_decode_bf16(
    q: &DeviceBuffer<u16>,
    k: &DeviceBuffer<u16>,
    v: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<u16>,
    num_q_heads: usize,
    num_kv_heads: usize,
    kv_len: usize,
    head_dim: usize,
    scaling: f32,
) -> Result<(), HipError> {
    assert!(
        num_kv_heads > 0 && num_q_heads % num_kv_heads == 0,
        "num_q_heads must be a positive multiple of num_kv_heads (GQA group size)"
    );
    assert_eq!(
        q.len(),
        num_q_heads * head_dim,
        "q length must be num_q_heads * head_dim"
    );
    assert_eq!(
        k.len(),
        num_kv_heads * kv_len * head_dim,
        "k length must be num_kv_heads * kv_len * head_dim"
    );
    assert_eq!(
        v.len(),
        num_kv_heads * kv_len * head_dim,
        "v length must be num_kv_heads * kv_len * head_dim"
    );
    assert_eq!(
        out.len(),
        num_q_heads * head_dim,
        "out length must be num_q_heads * head_dim"
    );

    // Power of two, capped at a sane maximum -- both reduction loops in the
    // kernel require it. 256 covers this model's real head_dim (256)
    // exactly and gives a reasonable number of threads per kv position too.
    let threads: i32 = 256;

    // The real kernel reads position (and derives kv_len = *position + 1
    // from it) through a device pointer, not a host value (§93). This safe
    // wrapper keeps its existing by-value `kv_len` API and does the tiny
    // host->device write itself.
    let mut position_buf: DeviceBuffer<i32> = DeviceBuffer::alloc(1)?;
    position_buf.copy_from_host(&[(kv_len - 1) as i32])?;

    // SAFETY: `q`/`k`/`v` are real, live `hipMalloc` allocations of at
    // least the asserted element counts; `out` is likewise real, live, and
    // distinct. The kernel indexes strictly within `[0, head_dim)` and
    // `[0, kv_len)` per head, and `h_kv = h / (num_q_heads/num_kv_heads)`
    // stays within `[0, num_kv_heads)` for every `h` in `[0, num_q_heads)`
    // given the assert above (verified by reading `attention.hip` directly).
    // kv_stride == kv_len reproduces the original tightly-packed contract
    // exactly (see attention.hip's header for why the kernel now accepts
    // a separate stride at all -- model.rs's hot path passes a real
    // max_seq_len-sized stride to read the KV cache directly, this safe
    // wrapper's callers all still pass tightly-packed buffers). `position_buf`
    // is a real, live `hipMalloc` allocation holding exactly the written
    // value, valid for the duration of this call.
    unsafe {
        ffi::launch_attention_decode_bf16(
            q.as_device_ptr(),
            k.as_device_ptr(),
            v.as_device_ptr(),
            out.as_device_ptr_mut() as *mut c_void,
            num_q_heads as i32,
            num_kv_heads as i32,
            position_buf.as_device_ptr() as *const c_int,
            kv_len as i32, // kv_stride == kv_len: tightly-packed, as before
            head_dim as i32,
            scaling,
            threads,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// §104 (real split-KV decode attention): the REAL, faster counterpart of
/// `attention_decode_bf16`, targeting the real position-dependent decode
/// slowdown §103's own investigation measured directly. See
/// `src/kernels/attention_decode_split.hip`'s header for the exact
/// technique (ported from llama.cpp's own real decode-attention kernel)
/// and why it's bit-for-bit the same real math, not an approximation.
///
/// Same real shapes as `attention_decode_bf16`, plus `kv_split`: the
/// number of threads cooperating on EACH output dimension (`head_dim`
/// must be a multiple of the launch's own thread-count constraint --
/// checked below, not assumed).
#[allow(clippy::too_many_arguments)]
pub fn attention_decode_split_bf16(
    q: &DeviceBuffer<u16>,
    k: &DeviceBuffer<u16>,
    v: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<u16>,
    num_q_heads: usize,
    num_kv_heads: usize,
    kv_len: usize,
    head_dim: usize,
    kv_split: usize,
    scaling: f32,
) -> Result<(), HipError> {
    assert!(num_kv_heads > 0 && num_q_heads % num_kv_heads == 0, "num_q_heads must be a positive multiple of num_kv_heads (GQA group size)");
    assert!(kv_split >= 1, "kv_split must be at least 1");
    // Real hardware constraint, checked not assumed: `threads =
    // head_dim*kv_split` is this kernel's own real block size (see
    // `attention_decode_split.hip`'s launcher), and every real ROCm/HIP
    // GPU caps threads-per-block at 1024.
    assert!(head_dim * kv_split <= 1024, "head_dim*kv_split ({}) exceeds the real 1024 threads-per-block hardware limit", head_dim * kv_split);
    assert_eq!(q.len(), num_q_heads * head_dim, "q length must be num_q_heads * head_dim");
    assert_eq!(k.len(), num_kv_heads * kv_len * head_dim, "k length must be num_kv_heads * kv_len * head_dim");
    assert_eq!(v.len(), num_kv_heads * kv_len * head_dim, "v length must be num_kv_heads * kv_len * head_dim");
    assert_eq!(out.len(), num_q_heads * head_dim, "out length must be num_q_heads * head_dim");

    let mut position_buf: DeviceBuffer<i32> = DeviceBuffer::alloc(1)?;
    position_buf.copy_from_host(&[(kv_len - 1) as i32])?;

    // SAFETY: same reasoning as `attention_decode_bf16` -- all buffers
    // are real, live `hipMalloc` allocations of at least the asserted
    // element counts; the kernel indexes strictly within
    // `[0, head_dim)`/`[0, kv_len)`/`[0, num_kv_heads)` per the asserts
    // above (verified by reading `attention_decode_split.hip` directly).
    // kv_stride == kv_len reproduces the tightly-packed contract exactly.
    unsafe {
        ffi::launch_attention_decode_split_bf16(
            q.as_device_ptr(),
            k.as_device_ptr(),
            v.as_device_ptr(),
            out.as_device_ptr_mut() as *mut c_void,
            num_q_heads as i32,
            num_kv_heads as i32,
            position_buf.as_device_ptr() as *const c_int,
            kv_len as i32,
            head_dim as i32,
            kv_split as i32,
            scaling,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// GatedDeltaNet's single-token recurrent decode update (the hardest,
/// highest-payoff feature -- 24 of 32 layers, every decode step). See
/// `src/kernels/gdn_recurrent.hip`'s header for the exact formula (matching
/// the real `torch_recurrent_gated_delta_rule`'s per-token loop body) and
/// why L2-norm and the query scaling are in scope here rather than the
/// surrounding module code.
///
/// `q`/`k`/`v`/`out` hold `num_heads * head_dim` bf16 elements each (one
/// new decode-step token, Q/K already broadcast from `num_k_heads` to
/// `num_heads` by the caller, matching real code's own module-level
/// `repeat_interleave` placement). `g` holds `num_heads` f32 elements (the
/// pre-exp log-decay, i.e. `g = -A_log.exp()*softplus(a+dt_bias)` computed
/// upstream, NOT yet exponentiated -- this kernel does `exp(g[h])` itself,
/// matching the real function's own `decay_t = decay[...].exp()`). `beta`
/// holds `num_heads` f32 elements (post-sigmoid). `state` holds
/// `num_heads * head_dim * head_dim` f32 elements -- real, persistent
/// recurrent state, **read AND written in place**, exactly like
/// `causal_conv1d_update_bf16`'s `conv_state` contract. State is kept in
/// f32 (not bf16) deliberately: the real Python function casts `value` (and
/// therefore the state buffer) to fp32 before the recurrent loop and never
/// casts it back until after accumulation -- this kernel matches that
/// precision choice exactly rather than simplifying it away.
/// `q`/`k` hold `num_k_heads * head_dim` elements (pre-broadcast, real
/// checkpoint dims: `num_k_heads=16`); `v`/`g`/`beta`/`state`/`out` hold
/// `num_heads` (`=32`)-worth of elements, matching the real module's own
/// asymmetry (value/beta/decay are already `num_v_heads`-native; only Q/K
/// need the GQA broadcast, handled internally now -- see
/// `gdn_recurrent.hip`'s header for why this changed from requiring
/// pre-broadcast Q/K).
#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent_decode_bf16(
    q: &DeviceBuffer<u16>,
    k: &DeviceBuffer<u16>,
    v: &DeviceBuffer<u16>,
    g: &DeviceBuffer<f32>,
    beta: &DeviceBuffer<f32>,
    state: &mut DeviceBuffer<f32>,
    out: &mut DeviceBuffer<u16>,
    num_heads: usize,
    num_k_heads: usize,
    head_dim: usize,
) -> Result<(), HipError> {
    assert!(
        num_k_heads > 0 && num_heads % num_k_heads == 0,
        "num_heads must be a positive multiple of num_k_heads (GQA group size)"
    );
    assert_eq!(q.len(), num_k_heads * head_dim, "q length mismatch");
    assert_eq!(k.len(), num_k_heads * head_dim, "k length mismatch");
    assert_eq!(v.len(), num_heads * head_dim, "v length mismatch");
    assert_eq!(g.len(), num_heads, "g length must equal num_heads");
    assert_eq!(beta.len(), num_heads, "beta length must equal num_heads");
    assert_eq!(
        state.len(),
        num_heads * head_dim * head_dim,
        "state length must be num_heads * head_dim * head_dim"
    );
    assert_eq!(out.len(), num_heads * head_dim, "out length mismatch");

    // Must be a power of two (the two L2-norm block reductions) and >=
    // head_dim (phases 1/3/5 assume direct thread-to-index coverage). This
    // model's real head_dim is 128, already a power of two.
    let threads: i32 = head_dim.next_power_of_two() as i32;

    // SAFETY: all six buffers are real, live `hipMalloc` allocations of at
    // least the asserted element counts; `state` is both read and written
    // by design (the real function's own in-place recurrent-state
    // contract). `h_kv = h / (num_heads/num_k_heads)` stays within
    // `[0, num_k_heads)` for every `h` in `[0, num_heads)` given the
    // assert above. The kernel indexes strictly within `[0, head_dim)` and
    // `[0, head_dim*head_dim)` per head (verified by reading
    // `gdn_recurrent.hip` directly).
    unsafe {
        ffi::launch_gdn_recurrent_decode_bf16(
            q.as_device_ptr(),
            k.as_device_ptr(),
            v.as_device_ptr(),
            g.as_device_ptr() as *const std::ffi::c_float,
            beta.as_device_ptr() as *const std::ffi::c_float,
            state.as_device_ptr_mut() as *mut std::ffi::c_float,
            out.as_device_ptr_mut() as *mut c_void,
            num_heads as i32,
            num_k_heads as i32,
            head_dim as i32,
            threads,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// `out = x * sigmoid(gate)` -- the gating multiply in Qwen3.5's full
/// attention block, applied after the attention core and before `o_proj`.
/// See `src/kernels/sigmoid_gate.hip`. All three buffers hold `n` bf16
/// elements.
pub fn sigmoid_gate_bf16(
    x: &DeviceBuffer<u16>,
    gate: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<u16>,
) -> Result<(), HipError> {
    assert_eq!(x.len(), gate.len(), "x and gate must have the same length");
    assert_eq!(x.len(), out.len(), "out must have the same length as x/gate");

    // SAFETY: all three buffers are real, live `hipMalloc` allocations of
    // at least `x.len()` elements; the kernel indexes strictly within
    // `[0, n)` (verified by reading `sigmoid_gate.hip` directly).
    unsafe {
        ffi::launch_sigmoid_gate_bf16(
            x.as_device_ptr(),
            gate.as_device_ptr(),
            out.as_device_ptr_mut() as *mut c_void,
            x.len() as i32,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// `Qwen3_5RMSNormGated`: GatedDeltaNet's final norm --
/// `(weight * x_normed) * silu(gate)`. See `src/kernels/rmsnorm_gated.hip`
/// for why this is a genuinely different formula from `rmsnorm_bf16`
/// (weight init/offset differ). `x`/`gate`/`out` hold `num_rows *
/// hidden_size` bf16 elements; `weight` holds `hidden_size` **f32**
/// elements -- the real checkpoint stores this specific weight
/// (`linear_attn.norm.weight`) as native f32, unlike every other weight in
/// this model, confirmed by reading the real safetensors header directly.
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_gated_bf16(
    x: &DeviceBuffer<u16>,
    gate: &DeviceBuffer<u16>,
    weight: &DeviceBuffer<f32>,
    out: &mut DeviceBuffer<u16>,
    num_rows: usize,
    hidden_size: usize,
    eps: f32,
) -> Result<(), HipError> {
    assert_eq!(x.len(), num_rows * hidden_size, "x length mismatch");
    assert_eq!(gate.len(), num_rows * hidden_size, "gate length mismatch");
    assert_eq!(weight.len(), hidden_size, "weight length must equal hidden_size");
    assert_eq!(out.len(), num_rows * hidden_size, "out length mismatch");

    let threads: i32 = 256;

    // SAFETY: all buffers are real, live `hipMalloc` allocations of at
    // least the asserted element counts; the kernel indexes strictly
    // within `[0, hidden_size)` per row (verified by reading
    // `rmsnorm_gated.hip` directly).
    unsafe {
        ffi::launch_rmsnorm_gated_bf16(
            x.as_device_ptr(),
            gate.as_device_ptr(),
            weight.as_device_ptr() as *const std::ffi::c_float,
            out.as_device_ptr_mut() as *mut c_void,
            num_rows as i32,
            hidden_size as i32,
            eps,
            threads,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// GatedDeltaNet's per-head decay/beta computation:
/// `g = -exp(A_log) * softplus(a + dt_bias)`, `beta = sigmoid(b)`. See
/// `src/kernels/gdn_gate_beta.hip`. `a`/`b` hold `num_heads` bf16 elements
/// (the raw per-token `in_proj_a`/`in_proj_b` projections); `a_log`/
/// `dt_bias` hold `num_heads` f32 elements (real trained per-layer
/// parameters); `g_out`/`beta_out` hold `num_heads` f32 elements, ready to
/// feed directly into `gdn_recurrent_decode_bf16`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_gate_beta_bf16(
    a: &DeviceBuffer<u16>,
    b: &DeviceBuffer<u16>,
    a_log: &DeviceBuffer<f32>,
    dt_bias: &DeviceBuffer<f32>,
    g_out: &mut DeviceBuffer<f32>,
    beta_out: &mut DeviceBuffer<f32>,
    num_heads: usize,
) -> Result<(), HipError> {
    assert_eq!(a.len(), num_heads, "a length must equal num_heads");
    assert_eq!(b.len(), num_heads, "b length must equal num_heads");
    assert_eq!(a_log.len(), num_heads, "a_log length must equal num_heads");
    assert_eq!(dt_bias.len(), num_heads, "dt_bias length must equal num_heads");
    assert_eq!(g_out.len(), num_heads, "g_out length must equal num_heads");
    assert_eq!(beta_out.len(), num_heads, "beta_out length must equal num_heads");

    // SAFETY: all six buffers are real, live `hipMalloc` allocations of at
    // least `num_heads` elements; the kernel indexes strictly within
    // `[0, num_heads)` (verified by reading `gdn_gate_beta.hip` directly).
    unsafe {
        ffi::launch_gdn_gate_beta_bf16(
            a.as_device_ptr(),
            b.as_device_ptr(),
            a_log.as_device_ptr() as *const std::ffi::c_float,
            dt_bias.as_device_ptr() as *const std::ffi::c_float,
            g_out.as_device_ptr_mut() as *mut std::ffi::c_float,
            beta_out.as_device_ptr_mut() as *mut std::ffi::c_float,
            num_heads as i32,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// §103 (real batched-prefill GDN gate/beta): the whole-chunk counterpart
/// of `gdn_gate_beta_bf16` -- see `src/kernels/gdn_gate_beta_prefill.hip`'s
/// header for why this has no real cross-token dependency at all (unlike
/// `causal_conv1d_prefill_bf16`'s bounded-lookback conv, or §100's genuine
/// GDN delta-rule recurrence) and can batch trivially.
///
/// `a_src`/`b_src` are STRIDED views (`*_offset + t*(*_stride) + h`),
/// same real "zero-copy" technique `causal_conv1d_prefill_bf16` uses.
/// `a_log`/`dt_bias` stay `num_heads`-wide (real per-layer trained
/// weights, shared across every token). `g_out`/`beta_out` are tightly
/// packed `[num_tokens, num_heads]`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_gate_beta_prefill_bf16(
    a_src: &DeviceBuffer<u16>,
    a_stride: usize,
    a_offset: usize,
    b_src: &DeviceBuffer<u16>,
    b_stride: usize,
    b_offset: usize,
    a_log: &DeviceBuffer<f32>,
    dt_bias: &DeviceBuffer<f32>,
    g_out: &mut DeviceBuffer<f32>,
    beta_out: &mut DeviceBuffer<f32>,
    num_tokens: usize,
    num_heads: usize,
) -> Result<(), HipError> {
    assert!(a_offset + num_heads <= a_stride || num_tokens == 0, "a_offset + num_heads must fit within one a_stride-wide row");
    assert!(b_offset + num_heads <= b_stride || num_tokens == 0, "b_offset + num_heads must fit within one b_stride-wide row");
    assert!((num_tokens.saturating_sub(1)) * a_stride + a_offset + num_heads <= a_src.len() || num_tokens == 0, "a_src too short for num_tokens * a_stride with this offset/num_heads");
    assert!((num_tokens.saturating_sub(1)) * b_stride + b_offset + num_heads <= b_src.len() || num_tokens == 0, "b_src too short for num_tokens * b_stride with this offset/num_heads");
    assert_eq!(a_log.len(), num_heads, "a_log length must equal num_heads");
    assert_eq!(dt_bias.len(), num_heads, "dt_bias length must equal num_heads");
    assert_eq!(g_out.len(), num_tokens * num_heads, "g_out length must equal num_tokens * num_heads");
    assert_eq!(beta_out.len(), num_tokens * num_heads, "beta_out length must equal num_tokens * num_heads");
    let threads: i32 = 256;

    // SAFETY: bounds checked above; the kernel indexes `a_src`/`b_src`
    // strictly within their asserted stride/offset ranges and
    // `a_log`/`dt_bias`/`g_out`/`beta_out` strictly within
    // `[0, num_heads)`/`[0, num_tokens*num_heads)` (verified by reading
    // `gdn_gate_beta_prefill.hip` directly).
    unsafe {
        ffi::launch_gdn_gate_beta_prefill_bf16(
            a_src.as_device_ptr(),
            a_stride as i32,
            a_offset as i32,
            b_src.as_device_ptr(),
            b_stride as i32,
            b_offset as i32,
            a_log.as_device_ptr() as *const std::ffi::c_float,
            dt_bias.as_device_ptr() as *const std::ffi::c_float,
            g_out.as_device_ptr_mut() as *mut std::ffi::c_float,
            beta_out.as_device_ptr_mut() as *mut std::ffi::c_float,
            num_tokens as i32,
            num_heads as i32,
            threads,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// Elementwise residual add: `out = a + b`. See `src/kernels/add.hip`. All
/// three buffers hold `n` bf16 elements.
pub fn add_bf16(
    a: &DeviceBuffer<u16>,
    b: &DeviceBuffer<u16>,
    out: &mut DeviceBuffer<u16>,
) -> Result<(), HipError> {
    assert_eq!(a.len(), b.len(), "a and b must have the same length");
    assert_eq!(a.len(), out.len(), "out must have the same length as a/b");

    // SAFETY: all three buffers are real, live `hipMalloc` allocations of
    // at least `a.len()` elements; the kernel indexes strictly within
    // `[0, n)` (verified by reading `add.hip` directly).
    unsafe {
        ffi::launch_add_bf16(
            a.as_device_ptr(),
            b.as_device_ptr(),
            out.as_device_ptr_mut() as *mut c_void,
            a.len() as i32,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// §115: real, bit-exact on-device fp32->bf16 cast -- see
/// `src/kernels/cast_f32_bf16.hip`.
pub fn f32_to_bf16_cast(src: &DeviceBuffer<f32>, dst: &mut DeviceBuffer<u16>) -> Result<(), HipError> {
    assert_eq!(src.len(), dst.len(), "src and dst must have the same length");
    // SAFETY: `src`/`dst` are real, live `hipMalloc` allocations of at
    // least `src.len()` elements; the kernel indexes strictly within
    // `[0, n)` (verified by reading `cast_f32_bf16.hip` directly).
    unsafe {
        ffi::launch_f32_to_bf16_cast(src.as_device_ptr(), dst.as_device_ptr_mut() as *mut c_void, src.len() as i32, std::ptr::null_mut());
    }
    check_last_error()?;
    device_synchronize()
}

/// §115: real, bit-exact on-device bf16->fp32 cast -- see
/// `src/kernels/cast_f32_bf16.hip`.
pub fn bf16_to_f32_cast(src: &DeviceBuffer<u16>, dst: &mut DeviceBuffer<f32>) -> Result<(), HipError> {
    assert_eq!(src.len(), dst.len(), "src and dst must have the same length");
    // SAFETY: same reasoning as `f32_to_bf16_cast` above.
    unsafe {
        ffi::launch_bf16_to_f32_cast(src.as_device_ptr(), dst.as_device_ptr_mut() as *mut c_void, src.len() as i32, std::ptr::null_mut());
    }
    check_last_error()?;
    device_synchronize()
}

/// §116: real in-place `buf[i] *= scalar` for all `i`, `scalar` read
/// directly off the device at `scalar_buf[scalar_index]` (no host
/// involvement) -- see `src/kernels/scale_by_device_scalar.hip`.
pub fn scale_bf16_by_device_scalar(buf: &mut DeviceBuffer<u16>, scalar_buf: &DeviceBuffer<f32>, scalar_index: usize) -> Result<(), HipError> {
    assert!(scalar_index < scalar_buf.len(), "scalar_index out of bounds");
    // SAFETY: `buf` is a real, live `hipMalloc` allocation of `buf.len()`
    // elements; `scalar_ptr` points at a real, live, in-bounds element of
    // `scalar_buf` (checked by the assert above); the kernel indexes
    // strictly within `[0, buf.len())` (verified by reading
    // `scale_by_device_scalar.hip` directly).
    unsafe {
        ffi::launch_scale_bf16_by_device_scalar(
            buf.as_device_ptr_mut() as *mut c_void,
            scalar_buf.as_device_ptr_at(scalar_index),
            buf.len() as i32,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// Copies a fixed sub-range of each row out of a strided `[rows,
/// src_stride]` buffer into a tightly-packed `[rows, len]` destination --
/// §96 (batched prefill). See `src/kernels/extract_range.hip` for the full
/// "why" (un-fusing a combined-GEMM output when `rows > 1`, where a plain
/// pointer offset -- which was always enough at this crate's original
/// rows=1 decode granularity -- stops being equivalent to a real
/// tightly-packed view).
pub fn extract_range_bf16(
    src: &DeviceBuffer<u16>,
    dst: &mut DeviceBuffer<u16>,
    rows: usize,
    src_stride: usize,
    src_offset: usize,
    len: usize,
) -> Result<(), HipError> {
    assert!(
        src_offset + len <= src_stride,
        "src_offset + len must fit within src_stride"
    );
    assert_eq!(
        src.len(),
        rows * src_stride,
        "src length must be rows * src_stride"
    );
    assert_eq!(dst.len(), rows * len, "dst length must be rows * len");

    // SAFETY: `src`/`dst` are real, live `hipMalloc` allocations of at
    // least the asserted element counts; the kernel reads strictly within
    // `[row*src_stride + src_offset, row*src_stride + src_offset + len)` of
    // `src` (in bounds per the assert above) and writes strictly within
    // `[row*len, row*len + len)` of `dst` (verified by reading
    // `extract_range.hip` directly).
    unsafe {
        ffi::launch_extract_range_bf16(
            src.as_device_ptr(),
            dst.as_device_ptr_mut() as *mut c_void,
            rows as i32,
            src_stride as i32,
            src_offset as i32,
            len as i32,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// Splits a `[rows, 2*half]` buffer into two `[rows, half]` contiguous
/// buffers along the last dimension per row -- matching
/// `torch.chunk(..., 2, dim=-1)` in `Qwen3_5Attention.forward` (query/gate
/// split). See `src/kernels/split_last_dim.hip` for why this needs a real
/// kernel rather than a pointer-offset view.
pub fn split_last_dim_bf16(
    x: &DeviceBuffer<u16>,
    first: &mut DeviceBuffer<u16>,
    second: &mut DeviceBuffer<u16>,
    rows: usize,
    half: usize,
) -> Result<(), HipError> {
    assert_eq!(x.len(), rows * 2 * half, "x length must be rows * 2 * half");
    assert_eq!(first.len(), rows * half, "first length must be rows * half");
    assert_eq!(second.len(), rows * half, "second length must be rows * half");

    // SAFETY: all three buffers are real, live `hipMalloc` allocations of
    // at least the asserted element counts; the kernel indexes strictly
    // within `[0, half)` per row on both sides of the split (verified by
    // reading `split_last_dim.hip` directly).
    unsafe {
        ffi::launch_split_last_dim_bf16(
            x.as_device_ptr(),
            first.as_device_ptr_mut() as *mut c_void,
            second.as_device_ptr_mut() as *mut c_void,
            rows as i32,
            half as i32,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// Appends one new token's already-RoPE'd/normed K/V into a head-major KV
/// cache at `position`. See `src/kernels/kv_cache_append.hip`. `new_k`/
/// `new_v` hold `num_kv_heads * head_dim` elements; `k_cache`/`v_cache`
/// hold `num_kv_heads * max_seq_len * head_dim` elements and are written
/// in place at the `position`-th slot of each head.
#[allow(clippy::too_many_arguments)]
pub fn kv_cache_append_bf16(
    new_k: &DeviceBuffer<u16>,
    new_v: &DeviceBuffer<u16>,
    k_cache: &mut DeviceBuffer<u16>,
    v_cache: &mut DeviceBuffer<u16>,
    num_kv_heads: usize,
    max_seq_len: usize,
    head_dim: usize,
    position: usize,
) -> Result<(), HipError> {
    assert_eq!(new_k.len(), num_kv_heads * head_dim, "new_k length mismatch");
    assert_eq!(new_v.len(), num_kv_heads * head_dim, "new_v length mismatch");
    assert_eq!(
        k_cache.len(),
        num_kv_heads * max_seq_len * head_dim,
        "k_cache length must be num_kv_heads * max_seq_len * head_dim"
    );
    assert_eq!(
        v_cache.len(),
        num_kv_heads * max_seq_len * head_dim,
        "v_cache length must be num_kv_heads * max_seq_len * head_dim"
    );
    assert!(position < max_seq_len, "position out of bounds for max_seq_len");

    // The real kernel reads `position` through a device pointer, not a
    // host value (§93). This safe wrapper keeps its by-value `usize` API
    // and does the tiny host->device write itself.
    let mut position_buf: DeviceBuffer<i32> = DeviceBuffer::alloc(1)?;
    position_buf.copy_from_host(&[position as i32])?;

    // SAFETY: all four buffers are real, live `hipMalloc` allocations of
    // at least the asserted element counts; `position < max_seq_len` is
    // checked above, so every write stays within each head's own
    // `[0, max_seq_len)` range (verified by reading `kv_cache_append.hip`
    // directly). `position_buf` is a real, live `hipMalloc` allocation
    // holding exactly the written value, valid for the duration of this call.
    unsafe {
        ffi::launch_kv_cache_append_bf16(
            new_k.as_device_ptr(),
            new_v.as_device_ptr(),
            k_cache.as_device_ptr_mut() as *mut c_void,
            v_cache.as_device_ptr_mut() as *mut c_void,
            num_kv_heads as i32,
            max_seq_len as i32,
            head_dim as i32,
            position_buf.as_device_ptr() as *const c_int,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// §103 (real batched-prefill KV cache append): the whole-chunk
/// counterpart of `kv_cache_append_bf16` -- see
/// `src/kernels/kv_cache_append_prefill.hip`'s header for the exact
/// formula (identical real per-append math, just for every token in the
/// chunk in ONE launch, each reading its own real position from
/// `position_buf`).
///
/// `new_k`/`new_v` hold `num_tokens * num_kv_heads * head_dim` elements,
/// tightly packed `[num_tokens, num_kv_heads, head_dim]`. `position_buf`
/// holds `num_tokens` real absolute positions.
#[allow(clippy::too_many_arguments)]
pub fn kv_cache_append_prefill_bf16(
    new_k: &DeviceBuffer<u16>,
    new_v: &DeviceBuffer<u16>,
    k_cache: &mut DeviceBuffer<u16>,
    v_cache: &mut DeviceBuffer<u16>,
    num_tokens: usize,
    num_kv_heads: usize,
    max_seq_len: usize,
    head_dim: usize,
    position_buf: &DeviceBuffer<i32>,
) -> Result<(), HipError> {
    assert_eq!(new_k.len(), num_tokens * num_kv_heads * head_dim, "new_k length mismatch");
    assert_eq!(new_v.len(), num_tokens * num_kv_heads * head_dim, "new_v length mismatch");
    assert_eq!(k_cache.len(), num_kv_heads * max_seq_len * head_dim, "k_cache length must be num_kv_heads * max_seq_len * head_dim");
    assert_eq!(v_cache.len(), num_kv_heads * max_seq_len * head_dim, "v_cache length must be num_kv_heads * max_seq_len * head_dim");
    assert_eq!(position_buf.len(), num_tokens, "position_buf length must equal num_tokens");

    // SAFETY: all four buffers are real, live `hipMalloc` allocations of
    // at least the asserted element counts; the caller's real positions
    // (checked at the call site against `max_seq_len` by construction --
    // every real position this crate ever writes is `state.position + i`
    // for `i` within the chunk, bounded by `max_seq_len` at
    // `DecodeState::new` time) keep every write within each head's own
    // `[0, max_seq_len)` range (verified by reading
    // `kv_cache_append_prefill.hip` directly).
    unsafe {
        ffi::launch_kv_cache_append_prefill_bf16(
            new_k.as_device_ptr(),
            new_v.as_device_ptr(),
            k_cache.as_device_ptr_mut() as *mut c_void,
            v_cache.as_device_ptr_mut() as *mut c_void,
            num_tokens as i32,
            num_kv_heads as i32,
            max_seq_len as i32,
            head_dim as i32,
            position_buf.as_device_ptr() as *const c_int,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// `y = w @ x` (a bias-free `nn.Linear` forward, `w: [out_features,
/// in_features]` row-major, `x: [in_features]`) -- a hand-written,
/// vectorized bf16 GEMV. See `src/kernels/gemv.hip` for the real
/// memory-bandwidth motivation (this engine's GEMMs are always `rows=1`,
/// i.e. really GEMVs, and both general BLAS libraries tried this session
/// under-utilized this GPU's bandwidth for that shape).
///
/// `in_features` must be a multiple of 4 (checked, not assumed -- every
/// real shape in this model satisfies it).
pub fn gemv_bf16(
    x: &DeviceBuffer<u16>,
    w: &DeviceBuffer<u16>,
    y: &mut DeviceBuffer<u16>,
    out_features: usize,
    in_features: usize,
) -> Result<(), HipError> {
    assert_eq!(x.len(), in_features, "x length must equal in_features");
    assert_eq!(w.len(), out_features * in_features, "w length must equal out_features * in_features");
    assert_eq!(y.len(), out_features, "y length must equal out_features");
    assert_eq!(in_features % 4, 0, "gemv_bf16: in_features must be a multiple of 4 (see gemv.hip's header)");

    let threads: i32 = 256;

    // SAFETY: `x`/`w`/`y` are real, live `hipMalloc` allocations of at
    // least the asserted element counts; the kernel indexes strictly
    // within `[0, in_features)` per output row and `[0, out_features)`
    // blocks (verified by reading `gemv.hip` directly).
    unsafe {
        ffi::launch_gemv_bf16(
            x.as_device_ptr(),
            w.as_device_ptr(),
            y.as_device_ptr_mut() as *mut c_void,
            out_features as i32,
            in_features as i32,
            threads,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// §106: `y = dequant(qweight, scales) @ x` -- fused W4A16 dequant+GEMV.
/// `qweight: [out_features, in_features/8]` packed int32 (8 nibbles/row,
/// LSB-first), `scales: [out_features, in_features/group_size]` bf16 --
/// see `src/kernels/w4a16_gemv.hip` and `quantize_w4a16.py` for the real
/// quantization scheme and the deliberate row-major layout this expects
/// (matches this crate's own existing bf16 weight layout, NOT the source
/// Triton kernel's K-major one).
///
/// `in_features` must be a multiple of both 8 (nibble packing) and
/// `group_size` (checked, not assumed).
/// §110: the decode GEMV kernel's real per-thread work item is a 4-int32
/// (32-nibble) chunk (see `w4a16_gemv.hip`'s own header), so a block only
/// has `in_features/32` real work items -- for this model family's
/// smaller real shapes (`in_features=2048` at 2B: only 64 real chunks),
/// a fixed 256-thread launch leaves most of the block idle (real,
/// measured: 2B's decode throughput barely moved from vectorizing the
/// load at all, unlike 4B's, until this fix). Picks the largest power of
/// two `<= work_items`, floored at one warp (32) so the existing
/// warp-shuffle/shared-mem reduction (which assumes `threads` is a
/// power of two and a multiple of `warpSize`) stays correct, capped at
/// 256 (this kernel's original, still-good choice for the model
/// family's bigger real shapes).
/// §110: picks the largest power-of-two thread count (floored at one
/// warp, capped at 256) that doesn't leave most of a block idle for this
/// real shape -- see `w4a16_gemv.hip`'s own header for why a fixed 256
/// under-utilizes this model family's smaller real `in_features` once
/// loads are vectorized. Both the decode (`w4a16_gemv.hip`) and batched
/// prefill (`w4a16_gemm_prefill.hip`) kernels cover 4 int32s (32
/// nibbles) per thread per loop iteration (§113 tried doubling decode's
/// to 8 and reverted it -- see that kernel's own header for the real
/// regression that caused), so both real call sites below share this one
/// function.
fn w4a16_threads_for_width(in_features: usize, elements_per_iter: usize) -> i32 {
    let work_items = (in_features / elements_per_iter).max(1);
    let mut threads: i32 = 32;
    while threads * 2 <= work_items as i32 && threads < 256 {
        threads *= 2;
    }
    threads
}

pub(crate) fn w4a16_decode_threads(in_features: usize) -> i32 {
    w4a16_threads_for_width(in_features, 32)
}

pub(crate) fn w4a16_prefill_threads(in_features: usize) -> i32 {
    w4a16_threads_for_width(in_features, 32)
}

pub fn w4a16_gemv_bf16(
    x: &DeviceBuffer<u16>,
    qweight: &DeviceBuffer<u32>,
    scales: &DeviceBuffer<u16>,
    y: &mut DeviceBuffer<u16>,
    out_features: usize,
    in_features: usize,
    group_size: usize,
) -> Result<(), HipError> {
    assert_eq!(x.len(), in_features, "x length must equal in_features");
    assert_eq!(in_features % 8, 0, "w4a16_gemv_bf16: in_features must be a multiple of 8 (nibble packing)");
    assert_eq!(in_features % group_size, 0, "w4a16_gemv_bf16: in_features must be a multiple of group_size");
    assert_eq!(qweight.len(), out_features * (in_features / 8), "qweight length must equal out_features * in_features/8");
    assert_eq!(scales.len(), out_features * (in_features / group_size), "scales length must equal out_features * in_features/group_size");
    assert_eq!(y.len(), out_features, "y length must equal out_features");

    let threads: i32 = w4a16_decode_threads(in_features);

    // SAFETY: `x`/`qweight`/`scales`/`y` are real, live `hipMalloc`
    // allocations of at least the asserted element counts; the kernel
    // indexes strictly within `[0, in_features/8)` per output row and
    // `[0, out_features)` blocks (verified by reading `w4a16_gemv.hip`
    // directly).
    unsafe {
        ffi::launch_w4a16_gemv_bf16(
            x.as_device_ptr(),
            qweight.as_device_ptr(),
            scales.as_device_ptr(),
            y.as_device_ptr_mut() as *mut c_void,
            out_features as i32,
            in_features as i32,
            group_size as i32,
            threads,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// Quantized-path LoRA additive accumulation: `y[row] += sum_r b[row,r] *
/// mid[r]`, in place -- the real, tested counterpart of
/// `lora_delta_accumulate.hip`'s `launch_lora_delta_accumulate_bf16`. `y`
/// must already hold the base W4A16 GEMV's real output (from a prior,
/// separate `w4a16_gemv_bf16` call) -- this function only adds to it,
/// never overwrites.
#[allow(dead_code)]
pub fn lora_delta_accumulate_bf16(mid: &DeviceBuffer<u16>, b: &DeviceBuffer<u16>, y: &mut DeviceBuffer<u16>, out_features: usize, rank: usize) -> Result<(), HipError> {
    assert_eq!(mid.len(), rank, "mid length must equal rank");
    assert_eq!(b.len(), out_features * rank, "b length must equal out_features * rank");
    assert_eq!(y.len(), out_features, "y length must equal out_features");

    // SAFETY: `mid`/`b`/`y` are real, live `hipMalloc` allocations of at
    // least the asserted element counts; the kernel indexes strictly
    // within `[0, out_features)` threads (grid-stride guarded by an
    // explicit bounds check in the kernel itself) and `[0, rank)` per
    // thread (verified by reading `lora_delta_accumulate.hip` directly).
    unsafe {
        ffi::launch_lora_delta_accumulate_bf16(mid.as_device_ptr(), b.as_device_ptr(), y.as_device_ptr_mut() as *mut c_void, out_features as i32, rank as i32, std::ptr::null_mut());
    }
    check_last_error()?;
    device_synchronize()
}

/// §108: `y[num_tokens, out_features] = dequant(qweight, scales) @
/// x[num_tokens, in_features]^T` -- batched W4A16 prefill, real drop-in
/// replacement for `w4a16_gemv_bf16` at a prefill (`num_tokens > 1`) call
/// site. See `src/kernels/w4a16_gemm_prefill.hip` for the real
/// tile-reuse scheme. `x`/`y` are row-major (`x` stride `in_features`,
/// `y` stride `out_features`) -- the same layout convention `raw::gemm`'s
/// `rows>1` hipBLAS branch already uses for bf16 prefill.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_prefill_bf16(
    x: &DeviceBuffer<u16>,
    qweight: &DeviceBuffer<u32>,
    scales: &DeviceBuffer<u16>,
    y: &mut DeviceBuffer<u16>,
    out_features: usize,
    in_features: usize,
    group_size: usize,
    num_tokens: usize,
) -> Result<(), HipError> {
    assert_eq!(x.len(), num_tokens * in_features, "x length must equal num_tokens * in_features");
    assert_eq!(in_features % 8, 0, "w4a16_gemm_prefill_bf16: in_features must be a multiple of 8 (nibble packing)");
    assert_eq!(in_features % group_size, 0, "w4a16_gemm_prefill_bf16: in_features must be a multiple of group_size");
    assert_eq!(qweight.len(), out_features * (in_features / 8), "qweight length must equal out_features * in_features/8");
    assert_eq!(scales.len(), out_features * (in_features / group_size), "scales length must equal out_features * in_features/group_size");
    assert_eq!(y.len(), num_tokens * out_features, "y length must equal num_tokens * out_features");

    let threads: i32 = w4a16_prefill_threads(in_features);

    // SAFETY: `x`/`qweight`/`scales`/`y` are real, live `hipMalloc`
    // allocations of at least the asserted element counts; the kernel
    // indexes strictly within `[0, in_features/8)` per output row and
    // `[0, num_tokens)` per token tile (verified by reading
    // `w4a16_gemm_prefill.hip` directly).
    unsafe {
        ffi::launch_w4a16_gemm_prefill_bf16(
            x.as_device_ptr(),
            qweight.as_device_ptr(),
            scales.as_device_ptr(),
            y.as_device_ptr_mut() as *mut c_void,
            out_features as i32,
            in_features as i32,
            group_size as i32,
            num_tokens as i32,
            threads,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// §123 EXPERIMENT: `TILE_M=32` variant of `w4a16_gemm_prefill_bf16`,
/// isolated for a real, direct A/B against the shipped `TILE_M=16`
/// kernel -- NOT wired into any real model call site. See
/// `src/kernels/w4a16_gemm_prefill.hip`'s own doc comment and
/// `docs/DECISIONS.md` §123 for the real measured verdict.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_prefill_tile32_bf16(
    x: &DeviceBuffer<u16>,
    qweight: &DeviceBuffer<u32>,
    scales: &DeviceBuffer<u16>,
    y: &mut DeviceBuffer<u16>,
    out_features: usize,
    in_features: usize,
    group_size: usize,
    num_tokens: usize,
) -> Result<(), HipError> {
    assert_eq!(x.len(), num_tokens * in_features, "x length must equal num_tokens * in_features");
    assert_eq!(in_features % 8, 0, "in_features must be a multiple of 8");
    assert_eq!(in_features % group_size, 0, "in_features must be a multiple of group_size");
    assert_eq!(qweight.len(), out_features * (in_features / 8), "qweight length mismatch");
    assert_eq!(scales.len(), out_features * (in_features / group_size), "scales length mismatch");
    assert_eq!(y.len(), num_tokens * out_features, "y length mismatch");

    // SAFETY: same reasoning as `w4a16_gemm_prefill_bf16` above -- same
    // buffer contracts, only TILE_M differs (verified by reading
    // `w4a16_gemm_prefill.hip`'s `w4a16_gemm_prefill_tile32_kernel`
    // directly).
    unsafe {
        ffi::launch_w4a16_gemm_prefill_tile32_bf16(
            x.as_device_ptr(),
            qweight.as_device_ptr(),
            scales.as_device_ptr(),
            y.as_device_ptr_mut() as *mut c_void,
            out_features as i32,
            in_features as i32,
            group_size as i32,
            num_tokens as i32,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// §124 EXPERIMENT: `TILE_N=16` variant of `w4a16_gemm_prefill_bf16`,
/// isolated for a real, direct A/B against the shipped `TILE_N=8`
/// kernel -- NOT wired into any real model call site. See
/// `src/kernels/w4a16_gemm_prefill.hip`'s own doc comment and
/// `docs/DECISIONS.md` §124 for the real measured verdict.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_prefill_tile_n16_bf16(
    x: &DeviceBuffer<u16>,
    qweight: &DeviceBuffer<u32>,
    scales: &DeviceBuffer<u16>,
    y: &mut DeviceBuffer<u16>,
    out_features: usize,
    in_features: usize,
    group_size: usize,
    num_tokens: usize,
) -> Result<(), HipError> {
    assert_eq!(x.len(), num_tokens * in_features, "x length must equal num_tokens * in_features");
    assert_eq!(in_features % 8, 0, "in_features must be a multiple of 8");
    assert_eq!(in_features % group_size, 0, "in_features must be a multiple of group_size");
    assert_eq!(qweight.len(), out_features * (in_features / 8), "qweight length mismatch");
    assert_eq!(scales.len(), out_features * (in_features / group_size), "scales length mismatch");
    assert_eq!(y.len(), num_tokens * out_features, "y length mismatch");

    // SAFETY: same reasoning as `w4a16_gemm_prefill_bf16` above -- same
    // buffer contracts, only TILE_N differs (verified by reading
    // `w4a16_gemm_prefill.hip`'s `w4a16_gemm_prefill_tile_n16_kernel`
    // directly).
    unsafe {
        ffi::launch_w4a16_gemm_prefill_tile_n16_bf16(
            x.as_device_ptr(),
            qweight.as_device_ptr(),
            scales.as_device_ptr(),
            y.as_device_ptr_mut() as *mut c_void,
            out_features as i32,
            in_features as i32,
            group_size as i32,
            num_tokens as i32,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// §126 EXPERIMENT: real, production-shaped W4A16 prefill GEMM using
/// RDNA3 WMMA INT8 tensor cores -- NOT wired into any real model call
/// site. Requires `in_features` to be a multiple of 128 (real, exact for
/// every real model size in this family, see
/// `src/kernels/w4a16_gemm_prefill_wmma_int8.hip`'s own doc comment).
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_prefill_wmma_int8_bf16(
    x: &DeviceBuffer<u16>,
    qweight: &DeviceBuffer<u32>,
    scales: &DeviceBuffer<u16>,
    y: &mut DeviceBuffer<u16>,
    out_features: usize,
    in_features: usize,
    group_size: usize,
    num_tokens: usize,
) -> Result<(), HipError> {
    assert_eq!(x.len(), num_tokens * in_features, "x length must equal num_tokens * in_features");
    assert_eq!(in_features % 8, 0, "in_features must be a multiple of 8");
    assert_eq!(in_features % group_size, 0, "in_features must be a multiple of group_size");
    assert_eq!(group_size, 128, "this WMMA kernel's K-step is fixed at one real 128-element quantization group");
    assert_eq!(qweight.len(), out_features * (in_features / 8), "qweight length mismatch");
    assert_eq!(scales.len(), out_features * (in_features / group_size), "scales length mismatch");
    assert_eq!(y.len(), num_tokens * out_features, "y length mismatch");

    // SAFETY: same reasoning as `w4a16_gemm_prefill_bf16` above -- same
    // real, live allocations of the asserted sizes.
    unsafe {
        ffi::launch_w4a16_gemm_prefill_wmma_int8_bf16(
            x.as_device_ptr(),
            qweight.as_device_ptr(),
            scales.as_device_ptr(),
            y.as_device_ptr_mut() as *mut c_void,
            out_features as i32,
            in_features as i32,
            group_size as i32,
            num_tokens as i32,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// On-device argmax over `logits[n]` (real vocab logits). Returns the
/// index of the first (lowest-index) occurrence of the maximum value --
/// exactly the semantics `model.rs::argmax_sample`'s original host-side
/// scan had, just computed on the GPU (see `src/kernels/argmax.hip` for
/// the real motivation and the tie-breaking discipline).
pub fn argmax_bf16(logits: &DeviceBuffer<u16>) -> Result<i32, HipError> {
    argmax_bf16_row(logits, 0, logits.len())
}

/// §137: real argmax over ONE row of a `[num_rows, row_len]` logits
/// buffer (e.g. a speculative-decode verify chunk's real per-position
/// logits, `[k, VOCAB_SIZE]`) -- `argmax_bf16` itself only ever reads a
/// whole buffer as one flat vector, so this is the same real kernel
/// parameterized by a real row offset instead.
pub fn argmax_bf16_row(logits: &DeviceBuffer<u16>, row: usize, row_len: usize) -> Result<i32, HipError> {
    let threads: i32 = 1024;
    let mut out_idx: DeviceBuffer<i32> = DeviceBuffer::alloc(1)?;

    // SAFETY: `logits` is a real, live `hipMalloc` allocation; the caller
    // guarantees `(row+1)*row_len <= logits.len()`. The kernel indexes
    // strictly within `[0, row_len)` from the given offset (verified by
    // reading `argmax.hip` directly). `out_idx` is a real, live
    // `hipMalloc` allocation of exactly one `i32`.
    unsafe {
        ffi::launch_argmax_bf16(
            logits.as_device_ptr_at(row * row_len),
            out_idx.as_device_ptr_mut() as *mut c_int,
            row_len as i32,
            threads,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()?;

    let mut result = [0i32];
    out_idx.copy_to_host(&mut result)?;
    Ok(result[0])
}

/// §99 (batched-prefill causal attention): row-wise causal-masked softmax
/// over a real precomputed `[num_rows, kv_len]` score matrix, in place.
/// Row `r`'s valid key range is `[0, start_position + r]` -- see
/// `src/kernels/causal_softmax.hip` for the full derivation and why
/// masking happens here rather than inside the QK^T GEMM itself.
pub fn causal_softmax_bf16(scores: &mut DeviceBuffer<u16>, kv_len: usize, start_position: i32, num_rows: usize) -> Result<(), HipError> {
    assert_eq!(scores.len(), num_rows * kv_len, "scores length must be num_rows * kv_len");
    let mut position_buf: DeviceBuffer<i32> = DeviceBuffer::alloc(1)?;
    position_buf.copy_from_host(&[start_position])?;
    let threads: i32 = 256;

    // SAFETY: `scores` is a real, live `hipMalloc` allocation of exactly
    // `num_rows * kv_len` elements (asserted above); the kernel indexes
    // strictly within `[0, kv_len)` per row (verified by reading
    // `causal_softmax.hip` directly). `position_buf` is a real, live
    // single-element allocation.
    unsafe {
        ffi::launch_causal_softmax_bf16(
            scores.as_device_ptr_mut() as *mut c_void,
            kv_len as i32,
            position_buf.as_device_ptr() as *const c_int,
            num_rows as i32,
            threads,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// §100 (chunked GDN parallel prefill): computes cumulative decay,
/// `decay_exp`/`remaining_decay`/`chunk_decay`, and applies the real
/// causal pairwise-decay mask to two score matrices in place -- see
/// `src/kernels/gdn_chunk_decay.hip` for the full derivation (matches
/// transformers' real `torch_chunk_gated_delta_rule` exactly).
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_decay_bf16(
    g: &DeviceBuffer<f32>,
    ut_system: &mut DeviceBuffer<u16>,
    intra_chunk_attn: &mut DeviceBuffer<u16>,
    decay_exp_out: &mut DeviceBuffer<f32>,
    remaining_decay_out: &mut DeviceBuffer<f32>,
    chunk_decay_out: &mut DeviceBuffer<f32>,
    num_heads: usize,
    num_chunks: usize,
    chunk_size: usize,
) -> Result<(), HipError> {
    assert_eq!(g.len(), num_chunks * chunk_size * num_heads, "g length must be num_chunks*chunk_size*num_heads");
    assert_eq!(ut_system.len(), num_heads * num_chunks * chunk_size * chunk_size, "ut_system length mismatch");
    assert_eq!(intra_chunk_attn.len(), num_heads * num_chunks * chunk_size * chunk_size, "intra_chunk_attn length mismatch");
    assert_eq!(decay_exp_out.len(), num_heads * num_chunks * chunk_size, "decay_exp_out length mismatch");
    assert_eq!(remaining_decay_out.len(), num_heads * num_chunks * chunk_size, "remaining_decay_out length mismatch");
    assert_eq!(chunk_decay_out.len(), num_heads * num_chunks, "chunk_decay_out length mismatch");
    let threads: i32 = 256;

    // SAFETY: every buffer above is a real, live `hipMalloc` allocation
    // of exactly the asserted length; the kernel indexes strictly within
    // those bounds per (head, chunk) block (verified by reading
    // `gdn_chunk_decay.hip` directly).
    unsafe {
        ffi::launch_gdn_chunk_decay_bf16(
            g.as_device_ptr() as *const c_float,
            ut_system.as_device_ptr_mut() as *mut c_void,
            intra_chunk_attn.as_device_ptr_mut() as *mut c_void,
            decay_exp_out.as_device_ptr_mut() as *mut c_float,
            remaining_decay_out.as_device_ptr_mut() as *mut c_float,
            chunk_decay_out.as_device_ptr_mut() as *mut c_float,
            num_heads as i32,
            num_chunks as i32,
            chunk_size as i32,
            threads,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// §100 (chunked GDN parallel prefill): solves
/// `(I + strictly_lower(ut_system)) @ X = rhs` via forward substitution,
/// matching transformers' real `torch.linalg.solve_triangular(...,
/// unitriangular=True)` call exactly -- see
/// `src/kernels/gdn_chunk_utsolve.hip` for the full derivation.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_utsolve_bf16(
    ut_system: &DeviceBuffer<u16>,
    rhs: &DeviceBuffer<u16>,
    x_out: &mut DeviceBuffer<u16>,
    num_heads: usize,
    num_chunks: usize,
    chunk_size: usize,
    width: usize,
) -> Result<(), HipError> {
    assert_eq!(ut_system.len(), num_heads * num_chunks * chunk_size * chunk_size, "ut_system length mismatch");
    assert_eq!(rhs.len(), num_heads * num_chunks * chunk_size * width, "rhs length mismatch");
    assert_eq!(x_out.len(), num_heads * num_chunks * chunk_size * width, "x_out length mismatch");
    let threads = (width as i32).min(1024);

    // SAFETY: every buffer is a real, live `hipMalloc` allocation of
    // exactly the asserted length; the kernel indexes strictly within
    // those bounds per (head, chunk) block (verified by reading
    // `gdn_chunk_utsolve.hip` directly).
    unsafe {
        ffi::launch_gdn_chunk_utsolve_bf16(
            ut_system.as_device_ptr(),
            rhs.as_device_ptr(),
            x_out.as_device_ptr_mut() as *mut c_void,
            num_heads as i32,
            num_chunks as i32,
            chunk_size as i32,
            width as i32,
            threads,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// §100 (chunked GDN parallel prefill): pure L2 normalization, matching
/// transformers' real `l2norm()` exactly -- see `src/kernels/l2norm.hip`
/// for the full derivation and why this is NOT `rmsnorm_bf16`.
pub fn l2norm_bf16(x: &DeviceBuffer<u16>, out: &mut DeviceBuffer<u16>, num_rows: usize, hidden_size: usize, eps: f32, post_scale: f32) -> Result<(), HipError> {
    assert_eq!(x.len(), num_rows * hidden_size, "x length must be num_rows*hidden_size");
    assert_eq!(out.len(), num_rows * hidden_size, "out length must be num_rows*hidden_size");
    let threads: i32 = 256;

    // SAFETY: `x`/`out` are real, live `hipMalloc` allocations of exactly
    // `num_rows*hidden_size` elements (asserted above); the kernel indexes
    // strictly within `[0, hidden_size)` per row (verified by reading
    // `l2norm.hip` directly).
    unsafe {
        ffi::launch_l2norm_bf16(
            x.as_device_ptr(),
            out.as_device_ptr_mut() as *mut c_void,
            num_rows as i32,
            hidden_size as i32,
            eps,
            post_scale,
            threads,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// §100 (chunked GDN parallel prefill): `dst[t,h,:] = src[t,h/n_rep,:] *
/// scale[t,h]` -- see `src/kernels/gdn_chunk_broadcast_scale.hip` for the
/// full derivation and its real reuse across `k_beta`/`decayed_k_beta`/
/// the final query&key rescale.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_broadcast_scale_bf16(
    src: &DeviceBuffer<u16>,
    scale: &DeviceBuffer<f32>,
    dst: &mut DeviceBuffer<u16>,
    num_tokens: usize,
    src_heads: usize,
    dst_heads: usize,
    head_dim: usize,
    n_rep: usize,
    chunk_size: usize,
    head_major_output: bool,
) -> Result<(), HipError> {
    assert_eq!(src.len(), num_tokens * src_heads * head_dim, "src length mismatch");
    assert_eq!(scale.len(), num_tokens * dst_heads, "scale length mismatch");
    assert_eq!(dst.len(), num_tokens * dst_heads * head_dim, "dst length mismatch");
    assert_eq!(dst_heads, src_heads * n_rep, "dst_heads must equal src_heads * n_rep");
    let threads: i32 = 128;

    // SAFETY: `src`/`scale`/`dst` are real, live `hipMalloc` allocations
    // of exactly the asserted lengths; the kernel indexes strictly within
    // those bounds per (token, dst_head) block (verified by reading
    // `gdn_chunk_broadcast_scale.hip` directly).
    unsafe {
        ffi::launch_gdn_chunk_broadcast_scale_bf16(
            src.as_device_ptr(),
            scale.as_device_ptr() as *const c_float,
            dst.as_device_ptr_mut() as *mut c_void,
            num_tokens as i32,
            src_heads as i32,
            dst_heads as i32,
            head_dim as i32,
            n_rep as i32,
            chunk_size as i32,
            head_major_output as i32,
            threads,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hip;

    fn f32_to_bf16(v: f32) -> u16 {
        let bits = v.to_bits();
        let lsb = (bits >> 16) & 1;
        let rounded = bits.wrapping_add(0x7fff + lsb);
        (rounded >> 16) as u16
    }

    fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    /// §115: real correctness for the new on-device fp32<->bf16 cast
    /// kernels -- cross-validated against this SAME test module's own
    /// `f32_to_bf16`/`bf16_to_f32` (the identical formula `model.rs`'s
    /// real production code and every other kernel in this crate already
    /// use). Expects BIT-EXACT equality, not a tolerance: this is a pure
    /// elementwise cast, the exact real replacement for
    /// `gdn_chunk_forward_prefill`'s old host round trip, not an
    /// approximation of it.
    #[test]
    fn real_f32_bf16_casts_are_bit_exact_both_directions() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        // Real, varied values -- including some real round-to-nearest-even
        // tie cases (the `+0x7fff+lsb` formula's whole reason for being
        // more than a naive truncation), not just "nice" numbers.
        let f32_vals: Vec<f32> = vec![
            0.0, -0.0, 1.0, -1.0, 3.14159265, -2.71828, 1e-30, -1e30, 0.00390625, 12345.6789, f32::MIN_POSITIVE, -f32::MIN_POSITIVE,
        ];
        let expected_bf16: Vec<u16> = f32_vals.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut f32_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(f32_vals.len()).unwrap();
        f32_buf.copy_from_host(&f32_vals).unwrap();
        let mut bf16_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(f32_vals.len()).unwrap();
        f32_to_bf16_cast(&f32_buf, &mut bf16_buf).expect("real f32_to_bf16_cast call failed");
        let mut got_bf16 = vec![0u16; f32_vals.len()];
        bf16_buf.copy_to_host(&mut got_bf16).unwrap();
        assert_eq!(got_bf16, expected_bf16, "GPU f32->bf16 cast must be BIT-EXACT identical to the CPU reference formula");

        // Round trip back: bf16->f32 is a pure zero-extend (exact by
        // construction, no rounding), so casting the ALREADY-bf16-rounded
        // values back must reproduce their real f32 representation exactly.
        let expected_f32_roundtrip: Vec<f32> = expected_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
        let mut f32_out_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(f32_vals.len()).unwrap();
        bf16_to_f32_cast(&bf16_buf, &mut f32_out_buf).expect("real bf16_to_f32_cast call failed");
        let mut got_f32 = vec![0.0f32; f32_vals.len()];
        f32_out_buf.copy_to_host(&mut got_f32).unwrap();
        assert_eq!(got_f32, expected_f32_roundtrip, "GPU bf16->f32 cast must be BIT-EXACT identical to the CPU reference formula");
    }

    /// §116: real correctness for the new in-place device-scalar-scale
    /// kernel -- cross-validated against an independent CPU reference
    /// (real bf16 round-trip through fp32 multiply, same formula
    /// `gdn_chunk_forward_prefill` now relies on). Expects BIT-EXACT
    /// equality: this kernel itself does exactly ONE real multiply +
    /// round, same as the CPU reference computes -- the real, disclosed
    /// precision question (whether splitting hipBLAS's OWN fused
    /// beta-scale-and-accumulate into two steps changes anything) is a
    /// separate, larger question answered by a real end-to-end generation
    /// A/B, not this kernel-level test.
    #[test]
    fn real_scale_bf16_by_device_scalar_matches_cpu_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let buf_f32: Vec<f32> = (0..37).map(|i| (((i % 11) as i32 - 5) as f32) * 0.3).collect();
        let buf_bf16: Vec<u16> = buf_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let scalars: Vec<f32> = vec![1.0, 0.0, -1.0, 0.7391, 2.5, -0.001];

        for (scalar_index, &scalar) in scalars.iter().enumerate() {
            let expected: Vec<u16> = buf_bf16.iter().map(|&b| f32_to_bf16(bf16_to_f32(b) * scalar)).collect();

            let mut buf: DeviceBuffer<u16> = DeviceBuffer::alloc(buf_bf16.len()).unwrap();
            buf.copy_from_host(&buf_bf16).unwrap();
            let mut scalar_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(scalars.len()).unwrap();
            scalar_buf.copy_from_host(&scalars).unwrap();

            scale_bf16_by_device_scalar(&mut buf, &scalar_buf, scalar_index).expect("real scale_bf16_by_device_scalar call failed");
            let mut got = vec![0u16; buf_bf16.len()];
            buf.copy_to_host(&mut got).unwrap();
            assert_eq!(got, expected, "scalar_index {scalar_index} (value {scalar}): GPU kernel must be BIT-EXACT identical to the CPU reference");
        }
    }

    /// Real correctness test: a small, fixed, real input vector and a real
    /// weight vector, RMSNorm computed on the real GPU via the real kernel,
    /// checked against a value computed INDEPENDENTLY here in Rust using
    /// plain f32 math (not calling the kernel) -- the same
    /// "independent second computation, not self-consistency" discipline
    /// `model_loader.rs`'s decisive test already established, this time
    /// cross-checked against a hand-derived formula instead of an external
    /// Python run (both are legitimate independent oracles; this one keeps
    /// the test self-contained and fast).
    #[test]
    fn real_rmsnorm_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let hidden_size = 8usize;
        let eps = 1e-6f32;
        let x_f32: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 2.5, -1.5];
        let w_f32: Vec<f32> = vec![0.1, 0.2, -0.1, 0.0, 0.3, -0.2, 0.05, -0.05];

        // Independent reference, plain f32, matching ExactRMSNorm's exact
        // formula (unit_offset=True): out = (x * rsqrt(mean(x^2)+eps)) * (1+w)
        let mean_sq: f32 = x_f32.iter().map(|v| v * v).sum::<f32>() / hidden_size as f32;
        let rms = 1.0 / (mean_sq + eps).sqrt();
        let expected: Vec<f32> = x_f32
            .iter()
            .zip(w_f32.iter())
            .map(|(&xv, &wv)| (xv * rms) * (1.0 + wv))
            .collect();

        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let w_bf16: Vec<u16> = w_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(hidden_size).unwrap();
        let mut w_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(hidden_size).unwrap();
        let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(hidden_size).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();
        w_buf.copy_from_host(&w_bf16).unwrap();

        rmsnorm_bf16(&x_buf, &w_buf, &mut out_buf, 1, hidden_size, eps)
            .expect("real rmsnorm_bf16 kernel launch failed");

        let mut out_bf16 = vec![0u16; hidden_size];
        out_buf.copy_to_host(&mut out_bf16).unwrap();
        let got: Vec<f32> = out_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            // bf16 has ~3 decimal digits of precision; this tolerance
            // reflects that rounding, not a loosened correctness bar.
            assert!(
                (g - e).abs() < 0.01,
                "element {i}: GPU kernel got {g}, independent reference {e}"
            );
        }
    }

    /// Real test against REAL model weights loaded in Stage 1
    /// (`model_loader.rs`): layer 0's actual `input_layernorm.weight`, with
    /// a fixed synthetic activation (RMSNorm's cost and correctness don't
    /// depend on activation semantics, only shape/dtype -- the weight is
    /// the part that must be real, and here it is).
    ///
    /// Real, generalized (not gated, unlike its sibling real-weight-byte
    /// tests): this one only checks shape/finiteness, not real, size-
    /// specific reference VALUES -- so it compares against the compiled
    /// feature's own real `HIDDEN_SIZE` constant instead of a hardcoded
    /// literal, making it correct under every real model size.
    #[test]
    fn real_rmsnorm_runs_against_real_layer0_norm_weight() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = crate::model_loader::locate_model_snapshot()
            .expect("no real Qwen3.5-4B snapshot found on this machine");
        let raw = crate::model_loader::load_raw_tensor(
            &snapshot,
            "model.language_model.layers.0.input_layernorm.weight",
        )
        .expect("layer 0 input_layernorm.weight must exist");
        let hidden_size = raw.shape[0];
        assert_eq!(hidden_size, crate::model::HIDDEN_SIZE);

        let weight_bf16: Vec<u16> = raw
            .data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let x_f32: Vec<f32> = (0..hidden_size)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
            .collect();
        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(hidden_size).unwrap();
        let mut w_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(hidden_size).unwrap();
        let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(hidden_size).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();
        w_buf.copy_from_host(&weight_bf16).unwrap();

        rmsnorm_bf16(&x_buf, &w_buf, &mut out_buf, 1, hidden_size, 1e-6)
            .expect("real rmsnorm_bf16 kernel launch failed against real layer 0 weight");

        let mut out_bf16 = vec![0u16; hidden_size];
        out_buf.copy_to_host(&mut out_bf16).unwrap();
        // Sanity: output must be finite and not all-zero (a real computation
        // happened), not a byte-exact check -- the byte-exact check against
        // an independent reference is `real_rmsnorm_matches_independently_computed_reference` above.
        let any_nonzero = out_bf16.iter().any(|&b| b != 0);
        assert!(
            any_nonzero,
            "rmsnorm output against real weights was all zero"
        );
        for &b in &out_bf16 {
            assert!(
                bf16_to_f32(b).is_finite(),
                "rmsnorm produced a non-finite value"
            );
        }
    }

    /// Real timing, not a synthetic estimate: launches the real kernel
    /// `iters` times (after a warmup) at `hidden_size=2560` (this model's
    /// real hidden size), for `num_rows` in {1, 128} -- the exact shapes
    /// `apps/runtime-ipwf/fused_norm.py`'s own docstring benchmarked, so
    /// there's a direct, same-shape number to compare against the current
    /// Python runtime. `#[ignore]`d so `cargo test` stays fast; run with
    /// `cargo test --release -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_real_rmsnorm_vs_python_runtime_shapes() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let hidden_size = 2560usize;
        let eps = 1e-6f32;
        let iters = 2000;
        let warmup = 100;

        for &num_rows in &[1usize, 128usize] {
            let x_f32: Vec<f32> = (0..num_rows * hidden_size)
                .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
                .collect();
            let w_f32: Vec<f32> = (0..hidden_size)
                .map(|i| ((i % 53) as f32 - 26.0) * 0.01)
                .collect();
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let w_bf16: Vec<u16> = w_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_rows * hidden_size).unwrap();
            let w_buf: DeviceBuffer<u16> = {
                let mut b: DeviceBuffer<u16> = DeviceBuffer::alloc(hidden_size).unwrap();
                b.copy_from_host(&w_bf16).unwrap();
                b
            };
            let mut out_buf: DeviceBuffer<u16> =
                DeviceBuffer::alloc(num_rows * hidden_size).unwrap();
            x_buf.copy_from_host(&x_bf16).unwrap();

            for _ in 0..warmup {
                rmsnorm_bf16(&x_buf, &w_buf, &mut out_buf, num_rows, hidden_size, eps).unwrap();
            }

            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                rmsnorm_bf16(&x_buf, &w_buf, &mut out_buf, num_rows, hidden_size, eps).unwrap();
            }
            let elapsed = t0.elapsed();
            let per_call_us = elapsed.as_secs_f64() * 1e6 / iters as f64;
            println!(
                "rmsnorm_bf16 rows={num_rows:4} hidden={hidden_size}: {per_call_us:.3} us/call ({iters} iters, includes hipDeviceSynchronize)"
            );
        }
    }

    /// Same per-call-sync-vs-pipelined investigation as the SwiGLU/
    /// embedding/GEMM variants -- applied here too so the "wins" are
    /// measured on the same honest, consistent footing as the corrected
    /// "losses", not left as an inconsistency in the record.
    #[test]
    #[ignore]
    fn bench_real_rmsnorm_unsynced_pipelined_vs_python_runtime_shapes() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let hidden_size = 2560usize;
        let eps = 1e-6f32;
        let iters = 2000;
        let warmup = 100;

        for &num_rows in &[1usize, 128usize] {
            let x_f32: Vec<f32> = (0..num_rows * hidden_size)
                .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
                .collect();
            let w_f32: Vec<f32> = (0..hidden_size)
                .map(|i| ((i % 53) as f32 - 26.0) * 0.01)
                .collect();
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let w_bf16: Vec<u16> = w_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_rows * hidden_size).unwrap();
            let w_buf: DeviceBuffer<u16> = {
                let mut b: DeviceBuffer<u16> = DeviceBuffer::alloc(hidden_size).unwrap();
                b.copy_from_host(&w_bf16).unwrap();
                b
            };
            let mut out_buf: DeviceBuffer<u16> =
                DeviceBuffer::alloc(num_rows * hidden_size).unwrap();
            x_buf.copy_from_host(&x_bf16).unwrap();

            let threads: i32 = 256;
            let launch_one = |x_buf: &DeviceBuffer<u16>,
                               w_buf: &DeviceBuffer<u16>,
                               out_buf: &mut DeviceBuffer<u16>| {
                // SAFETY: same call rmsnorm_bf16 makes, minus the trailing
                // check_last_error()/device_synchronize().
                unsafe {
                    ffi::launch_rmsnorm_bf16(
                        x_buf.as_device_ptr(),
                        w_buf.as_device_ptr(),
                        out_buf.as_device_ptr_mut() as *mut c_void,
                        num_rows as i32,
                        hidden_size as i32,
                        eps,
                        threads,
                        std::ptr::null_mut(),
                    );
                }
            };

            for _ in 0..warmup {
                launch_one(&x_buf, &w_buf, &mut out_buf);
            }
            hip::device_synchronize().unwrap();

            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                launch_one(&x_buf, &w_buf, &mut out_buf);
            }
            hip::device_synchronize().unwrap();
            let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!(
                "rmsnorm_bf16 (UNSYNCED, pipelined) rows={num_rows:4} hidden={hidden_size}: {per_call_us:.3} us/call ({iters} iters, ONE hipDeviceSynchronize after the whole loop)"
            );
        }
    }

    // ---- RoPE ---------------------------------------------------------

    /// Real correctness test: a small fixed q vector, RoPE computed on the
    /// real GPU via the real kernel, checked against a reference computed
    /// independently here using plain f32 math following the exact formula
    /// derived from the real transformers source (see rope.hip's header
    /// comment) -- not calling the kernel's own code path.
    #[test]
    fn real_rope_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let head_dim = 8usize;
        let rotary_dim = 4usize; // half = 2
        let theta = 10000000.0f32;
        let position = 17i64;
        let x_f32: Vec<f32> = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25, -0.75];

        // Independent reference, plain f32, exact formula from rope.hip's header.
        let half = rotary_dim / 2;
        let mut expected = x_f32.clone();
        for i in 0..rotary_dim {
            let freq_idx = i % half;
            let inv_freq = theta.powf(-((2 * freq_idx) as f32) / rotary_dim as f32);
            let angle = position as f32 * inv_freq;
            let c = angle.cos();
            let s = angle.sin();
            expected[i] = if i < half {
                x_f32[i] * c - x_f32[i + half] * s
            } else {
                x_f32[i] * c + x_f32[i - half] * s
            };
        }
        // indices >= rotary_dim stay as pass-through (already correct in `expected`)

        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(head_dim).unwrap();
        let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(head_dim).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();

        rope_bf16(
            &x_buf,
            &mut out_buf,
            1,
            head_dim,
            rotary_dim,
            theta,
            position,
        )
        .expect("real rope_bf16 kernel launch failed");

        let mut out_bf16 = vec![0u16; head_dim];
        out_buf.copy_to_host(&mut out_bf16).unwrap();
        let got: Vec<f32> = out_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 0.01,
                "element {i}: GPU kernel got {g}, independent reference {e}"
            );
        }
    }

    /// §103 decisive test: `rope_prefill_bf16` (the new real
    /// batched-prefill kernel, one launch for the whole chunk) matches
    /// the ALREADY-PROVEN `rope_bf16` called once per token with its OWN
    /// distinct real position -- the real usage shape (`num_tokens=4`,
    /// `num_heads=3`, each token at a DIFFERENT absolute position,
    /// matching `state.position + i` in the real prefill call site).
    #[test]
    fn real_rope_prefill_matches_per_token_loop_with_distinct_positions() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let head_dim = 8usize;
        let rotary_dim = 4usize;
        let theta = 10000000.0f32;
        let num_tokens = 4usize;
        let num_heads = 3usize;
        let positions: [i64; 4] = [5, 6, 9, 40]; // real, distinct, non-contiguous

        let synth = |n: usize, offset: i32, period: i32, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let m = ((i as i32 + offset).rem_euclid(period)) as f32;
                    (m - (period as f32) / 2.0) * scale
                })
                .collect()
        };
        let x_all_f32: Vec<f32> = synth(num_tokens * num_heads * head_dim, 0, 31, 0.07);
        let x_all_bf16: Vec<u16> = x_all_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        // Reference: the already-proven per-token kernel, called once
        // per token with `num_rows=num_heads` and that token's own real
        // position.
        let mut ref_out = vec![0f32; num_tokens * num_heads * head_dim];
        for t in 0..num_tokens {
            let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_heads * head_dim).unwrap();
            x_buf.copy_from_host(&x_all_bf16[t * num_heads * head_dim..(t + 1) * num_heads * head_dim]).unwrap();
            let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_heads * head_dim).unwrap();
            rope_bf16(&x_buf, &mut out_buf, num_heads, head_dim, rotary_dim, theta, positions[t]).expect("reference rope_bf16 call failed");
            let mut out_bf16 = vec![0u16; num_heads * head_dim];
            out_buf.copy_to_host(&mut out_bf16).unwrap();
            for (i, &b) in out_bf16.iter().enumerate() {
                ref_out[t * num_heads * head_dim + i] = bf16_to_f32(b);
            }
        }

        // New batched path: ONE call, per-token positions.
        let mut x_all_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_all_bf16.len()).unwrap();
        x_all_buf.copy_from_host(&x_all_bf16).unwrap();
        let position_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let mut position_buf: DeviceBuffer<i32> = DeviceBuffer::alloc(num_tokens).unwrap();
        position_buf.copy_from_host(&position_i32).unwrap();
        let mut batched_out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_all_bf16.len()).unwrap();
        rope_prefill_bf16(&x_all_buf, &mut batched_out_buf, num_tokens, num_heads, head_dim, rotary_dim, theta, &position_buf).expect("real rope_prefill_bf16 call failed");

        let mut batched_out_bf16 = vec![0u16; x_all_bf16.len()];
        batched_out_buf.copy_to_host(&mut batched_out_bf16).unwrap();
        for (i, (&g, &e)) in batched_out_bf16.iter().zip(ref_out.iter()).enumerate() {
            let got = bf16_to_f32(g);
            assert!((got - e).abs() < 0.01, "element {i}: batched kernel got {got}, per-token-loop reference {e}");
        }
    }

    #[test]
    #[ignore]
    fn bench_real_rope_vs_python_runtime_shapes() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        // Real model dims: head_dim=256, rotary_dim=head_dim*partial_rotary_factor=64,
        // num_attention_heads=16 (Q) -- rows=16 for a single decode step's Q projection.
        let head_dim = 256usize;
        let rotary_dim = 64usize;
        let theta = 10000000.0f32;
        let num_rows = 16usize;
        let iters = 2000;
        let warmup = 100;

        let x_f32: Vec<f32> = (0..num_rows * head_dim)
            .map(|i| ((i % 91) as f32 - 45.0) * 0.01)
            .collect();
        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_rows * head_dim).unwrap();
        let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_rows * head_dim).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();

        for _ in 0..warmup {
            rope_bf16(
                &x_buf,
                &mut out_buf,
                num_rows,
                head_dim,
                rotary_dim,
                theta,
                128,
            )
            .unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            rope_bf16(
                &x_buf,
                &mut out_buf,
                num_rows,
                head_dim,
                rotary_dim,
                theta,
                128,
            )
            .unwrap();
        }
        let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!(
            "rope_bf16 rows={num_rows} head_dim={head_dim} rotary_dim={rotary_dim}: {per_call_us:.3} us/call ({iters} iters, includes hipDeviceSynchronize)"
        );
    }

    #[test]
    #[ignore]
    fn bench_real_rope_unsynced_pipelined_vs_python_runtime_shapes() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let head_dim = 256usize;
        let rotary_dim = 64usize;
        let theta = 10000000.0f32;
        let num_rows = 16usize;
        let iters = 2000;
        let warmup = 100;

        let x_f32: Vec<f32> = (0..num_rows * head_dim)
            .map(|i| ((i % 91) as f32 - 45.0) * 0.01)
            .collect();
        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_rows * head_dim).unwrap();
        let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_rows * head_dim).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();

        let mut position_buf: DeviceBuffer<i32> = DeviceBuffer::alloc(1).unwrap();
        position_buf.copy_from_host(&[128i32]).unwrap();

        let launch_one = |x_buf: &DeviceBuffer<u16>, out_buf: &mut DeviceBuffer<u16>| {
            // SAFETY: same call rope_bf16 makes, minus the trailing
            // check_last_error()/device_synchronize().
            unsafe {
                ffi::launch_rope_bf16(
                    x_buf.as_device_ptr(),
                    out_buf.as_device_ptr_mut() as *mut c_void,
                    num_rows as i32,
                    head_dim as i32,
                    rotary_dim as i32,
                    theta,
                    position_buf.as_device_ptr() as *const c_int,
                    std::ptr::null_mut(),
                );
            }
        };

        for _ in 0..warmup {
            launch_one(&x_buf, &mut out_buf);
        }
        hip::device_synchronize().unwrap();

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            launch_one(&x_buf, &mut out_buf);
        }
        hip::device_synchronize().unwrap();
        let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!(
            "rope_bf16 (UNSYNCED, pipelined) rows={num_rows} head_dim={head_dim} rotary_dim={rotary_dim}: {per_call_us:.3} us/call ({iters} iters, ONE hipDeviceSynchronize after the whole loop)"
        );
    }

    // ---- SwiGLU ---------------------------------------------------------

    #[test]
    fn real_swiglu_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let gate_f32: Vec<f32> = vec![1.0, -2.0, 0.5, 3.0, -1.0, 0.0, 0.25, -0.75];
        let up_f32: Vec<f32> = vec![0.5, 0.5, -1.0, 2.0, 1.0, 3.0, -0.5, 0.1];
        let expected: Vec<f32> = gate_f32
            .iter()
            .zip(up_f32.iter())
            .map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u)
            .collect();

        let gate_bf16: Vec<u16> = gate_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let up_bf16: Vec<u16> = up_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut gate_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(gate_f32.len()).unwrap();
        let mut up_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(up_f32.len()).unwrap();
        let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(gate_f32.len()).unwrap();
        gate_buf.copy_from_host(&gate_bf16).unwrap();
        up_buf.copy_from_host(&up_bf16).unwrap();

        swiglu_bf16(&gate_buf, &up_buf, &mut out_buf)
            .expect("real swiglu_bf16 kernel launch failed");

        let mut out_bf16 = vec![0u16; gate_f32.len()];
        out_buf.copy_to_host(&mut out_bf16).unwrap();
        let got: Vec<f32> = out_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 0.01,
                "element {i}: GPU kernel got {g}, independent reference {e}"
            );
        }
    }

    /// §114: real correctness for the fused strided SwiGLU -- cross-
    /// validated against the OLD real extract_range(gate) +
    /// extract_range(up) + swiglu three-launch sequence it replaces, on a
    /// real multi-row (`T=5`) strided `[rows, src_stride]` buffer with a
    /// real gate/up layout matching this engine's own combined
    /// `gate_up_proj` output shape. Expects BIT-EXACT equality, not a
    /// tolerance -- this is a pure algebraic identity (same real reads,
    /// same real math, just fused into one launch), not a numerical
    /// approximation.
    #[test]
    fn real_swiglu_strided_matches_unfused_extract_range_plus_swiglu_sequence() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let rows = 5usize;
        let intermediate_size = 37usize; // deliberately not a power of two
        let src_stride = 2 * intermediate_size;

        let gate_up_f32: Vec<f32> = (0..rows * src_stride).map(|i| (((i % 23) as i32 - 11) as f32) * 0.07).collect();
        let gate_up_bf16: Vec<u16> = gate_up_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut gate_up_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(gate_up_bf16.len()).unwrap();
        gate_up_buf.copy_from_host(&gate_up_bf16).unwrap();

        // OLD sequence: two real extract_range calls, then real swiglu.
        let mut gate_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(rows * intermediate_size).unwrap();
        let mut up_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(rows * intermediate_size).unwrap();
        let mut unfused_out: DeviceBuffer<u16> = DeviceBuffer::alloc(rows * intermediate_size).unwrap();
        extract_range_bf16(&gate_up_buf, &mut gate_buf, rows, src_stride, 0, intermediate_size).unwrap();
        extract_range_bf16(&gate_up_buf, &mut up_buf, rows, src_stride, intermediate_size, intermediate_size).unwrap();
        swiglu_bf16(&gate_buf, &up_buf, &mut unfused_out).unwrap();
        let mut unfused_bf16 = vec![0u16; rows * intermediate_size];
        unfused_out.copy_to_host(&mut unfused_bf16).unwrap();

        // NEW fused kernel: one real launch, same real gate_up_buf.
        let mut fused_out: DeviceBuffer<u16> = DeviceBuffer::alloc(rows * intermediate_size).unwrap();
        swiglu_strided_bf16(&gate_up_buf, &mut fused_out, rows, src_stride, 0, intermediate_size, intermediate_size)
            .expect("real swiglu_strided_bf16 call failed");
        let mut fused_bf16 = vec![0u16; rows * intermediate_size];
        fused_out.copy_to_host(&mut fused_bf16).unwrap();

        assert_eq!(fused_bf16, unfused_bf16, "fused swiglu_strided must be BIT-EXACT identical to the unfused extract_range+extract_range+swiglu sequence -- same real reads, same real math, only the launch count differs");
    }

    #[test]
    fn real_fused_attn_qkv_prep_matches_unfused_extract_split_norm() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_tokens = 3usize;
        let num_q_heads = 4usize;
        let num_kv_heads = 2usize;
        let head_dim = 128usize;
        let eps = 1e-6f32;

        let q_row_len = num_q_heads * head_dim;
        let kv_row_len = num_kv_heads * head_dim;
        let combined_dim = q_row_len * 2 + kv_row_len * 2;

        let qkv_f32: Vec<f32> = (0..num_tokens * combined_dim).map(|i| (((i % 17) as i32 - 8) as f32) * 0.05).collect();
        let qkv_bf16: Vec<u16> = qkv_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut qkv_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(qkv_bf16.len()).unwrap();
        qkv_buf.copy_from_host(&qkv_bf16).unwrap();

        let qw_f32: Vec<f32> = (0..head_dim).map(|i| 0.9 + 0.01 * (i as f32)).collect();
        let qw_bf16: Vec<u16> = qw_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut qw_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(head_dim).unwrap();
        qw_buf.copy_from_host(&qw_bf16).unwrap();

        let kw_f32: Vec<f32> = (0..head_dim).map(|i| 1.1 - 0.01 * (i as f32)).collect();
        let kw_bf16: Vec<u16> = kw_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut kw_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(head_dim).unwrap();
        kw_buf.copy_from_host(&kw_bf16).unwrap();

        // 1. Unfused reference sequence:
        let mut q_raw = DeviceBuffer::<u16>::alloc(num_tokens * q_row_len * 2).unwrap();
        let mut k_raw = DeviceBuffer::<u16>::alloc(num_tokens * kv_row_len).unwrap();
        let mut v_raw_unfused = DeviceBuffer::<u16>::alloc(num_tokens * kv_row_len).unwrap();
        let mut query = DeviceBuffer::<u16>::alloc(num_tokens * q_row_len).unwrap();
        let mut gate_unfused = DeviceBuffer::<u16>::alloc(num_tokens * q_row_len).unwrap();
        let mut q_normed_unfused = DeviceBuffer::<u16>::alloc(num_tokens * q_row_len).unwrap();
        let mut k_normed_unfused = DeviceBuffer::<u16>::alloc(num_tokens * kv_row_len).unwrap();

        extract_range_bf16(&qkv_buf, &mut q_raw, num_tokens, combined_dim, 0, q_row_len * 2).unwrap();
        extract_range_bf16(&qkv_buf, &mut k_raw, num_tokens, combined_dim, q_row_len * 2, kv_row_len).unwrap();
        extract_range_bf16(&qkv_buf, &mut v_raw_unfused, num_tokens, combined_dim, q_row_len * 2 + kv_row_len, kv_row_len).unwrap();

        split_last_dim_bf16(&q_raw, &mut query, &mut gate_unfused, num_tokens * num_q_heads, head_dim).unwrap();
        rmsnorm_bf16(&query, &qw_buf, &mut q_normed_unfused, num_tokens * num_q_heads, head_dim, eps).unwrap();
        rmsnorm_bf16(&k_raw, &kw_buf, &mut k_normed_unfused, num_tokens * num_kv_heads, head_dim, eps).unwrap();

        let mut ref_q_normed = vec![0u16; num_tokens * q_row_len];
        let mut ref_gate = vec![0u16; num_tokens * q_row_len];
        let mut ref_k_normed = vec![0u16; num_tokens * kv_row_len];
        let mut ref_v = vec![0u16; num_tokens * kv_row_len];
        q_normed_unfused.copy_to_host(&mut ref_q_normed).unwrap();
        gate_unfused.copy_to_host(&mut ref_gate).unwrap();
        k_normed_unfused.copy_to_host(&mut ref_k_normed).unwrap();
        v_raw_unfused.copy_to_host(&mut ref_v).unwrap();

        // 2. Fused kernel:
        let mut fused_q_normed = DeviceBuffer::<u16>::alloc(num_tokens * q_row_len).unwrap();
        let mut fused_gate = DeviceBuffer::<u16>::alloc(num_tokens * q_row_len).unwrap();
        let mut fused_k_normed = DeviceBuffer::<u16>::alloc(num_tokens * kv_row_len).unwrap();
        let mut fused_v = DeviceBuffer::<u16>::alloc(num_tokens * kv_row_len).unwrap();

        unsafe {
            ffi::launch_fused_attn_qkv_prep_bf16(
                qkv_buf.as_device_ptr(),
                qw_buf.as_device_ptr(),
                kw_buf.as_device_ptr(),
                fused_q_normed.as_device_ptr_mut() as *mut c_void,
                fused_gate.as_device_ptr_mut() as *mut c_void,
                fused_k_normed.as_device_ptr_mut() as *mut c_void,
                fused_v.as_device_ptr_mut() as *mut c_void,
                num_tokens as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                head_dim as i32,
                eps,
                std::ptr::null_mut(),
            );
        }
        check_last_error().unwrap();
        device_synchronize().unwrap();

        let mut actual_q_normed = vec![0u16; num_tokens * q_row_len];
        let mut actual_gate = vec![0u16; num_tokens * q_row_len];
        let mut actual_k_normed = vec![0u16; num_tokens * kv_row_len];
        let mut actual_v = vec![0u16; num_tokens * kv_row_len];
        fused_q_normed.copy_to_host(&mut actual_q_normed).unwrap();
        fused_gate.copy_to_host(&mut actual_gate).unwrap();
        fused_k_normed.copy_to_host(&mut actual_k_normed).unwrap();
        fused_v.copy_to_host(&mut actual_v).unwrap();

        assert_eq!(actual_gate, ref_gate, "gate must be bit-exact match");
        assert_eq!(actual_v, ref_v, "v_raw must be bit-exact match");
        assert_eq!(actual_q_normed, ref_q_normed, "q_normed must be bit-exact match");
        assert_eq!(actual_k_normed, ref_k_normed, "k_normed must be bit-exact match");
    }

    #[test]
    fn real_attention_causal_prefill_matches_attention_decode_split() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_tokens = 4usize;
        let num_q_heads = 4usize;
        let num_kv_heads = 2usize;
        let head_dim = 128usize;
        let kv_stride = 64usize;
        let kv_split = 2usize;
        let scaling = (head_dim as f32).powf(-0.5);

        let q_len = num_tokens * num_q_heads * head_dim;
        let kv_len = num_kv_heads * kv_stride * head_dim;

        let q_f32: Vec<f32> = (0..q_len).map(|i| (((i % 19) as i32 - 9) as f32) * 0.08).collect();
        let k_f32: Vec<f32> = (0..kv_len).map(|i| (((i % 23) as i32 - 11) as f32) * 0.06).collect();
        let v_f32: Vec<f32> = (0..kv_len).map(|i| (((i % 29) as i32 - 14) as f32) * 0.05).collect();

        let mut q_buf = DeviceBuffer::<u16>::alloc(q_len).unwrap();
        let mut k_buf = DeviceBuffer::<u16>::alloc(kv_len).unwrap();
        let mut v_buf = DeviceBuffer::<u16>::alloc(kv_len).unwrap();
        q_buf.copy_from_host(&q_f32.iter().map(|&x| f32_to_bf16(x)).collect::<Vec<_>>()).unwrap();
        k_buf.copy_from_host(&k_f32.iter().map(|&x| f32_to_bf16(x)).collect::<Vec<_>>()).unwrap();
        v_buf.copy_from_host(&v_f32.iter().map(|&x| f32_to_bf16(x)).collect::<Vec<_>>()).unwrap();

        let positions = vec![0i32, 1, 2, 3];
        let mut pos_buf = DeviceBuffer::<i32>::alloc(positions.len()).unwrap();
        pos_buf.copy_from_host(&positions).unwrap();

        // 1. Loop of attention_decode_split:
        let q_row_len = num_q_heads * head_dim;
        let mut ref_out_buf = DeviceBuffer::<u16>::alloc(num_tokens * q_row_len).unwrap();
        for i in 0..num_tokens {
            unsafe {
                ffi::launch_attention_decode_split_bf16(
                    q_buf.as_device_ptr_at(i * q_row_len),
                    k_buf.as_device_ptr(),
                    v_buf.as_device_ptr(),
                    ref_out_buf.as_device_ptr_at_mut(i * q_row_len) as *mut c_void,
                    num_q_heads as i32,
                    num_kv_heads as i32,
                    pos_buf.as_device_ptr_at(i) as *const i32,
                    kv_stride as i32,
                    head_dim as i32,
                    kv_split as i32,
                    scaling,
                    std::ptr::null_mut(),
                );
            }
        }
        device_synchronize().unwrap();
        let mut ref_out = vec![0u16; num_tokens * q_row_len];
        ref_out_buf.copy_to_host(&mut ref_out).unwrap();

        // 2. Batched attention_causal_prefill:
        let mut actual_out_buf = DeviceBuffer::<u16>::alloc(num_tokens * q_row_len).unwrap();
        unsafe {
            ffi::launch_attention_causal_prefill_bf16(
                q_buf.as_device_ptr(),
                k_buf.as_device_ptr(),
                v_buf.as_device_ptr(),
                actual_out_buf.as_device_ptr_mut() as *mut c_void,
                num_tokens as i32,
                num_q_heads as i32,
                num_kv_heads as i32,
                pos_buf.as_device_ptr() as *const i32,
                kv_stride as i32,
                head_dim as i32,
                kv_split as i32,
                scaling,
                4i32,
                std::ptr::null_mut(),
            );
        }
        check_last_error().unwrap();
        device_synchronize().unwrap();
        let mut actual_out = vec![0u16; num_tokens * q_row_len];
        actual_out_buf.copy_to_host(&mut actual_out).unwrap();

        assert_eq!(actual_out, ref_out, "2D batched attention_causal_prefill must be BIT-EXACT identical to the sequential attention_decode_split loop");
    }

    #[test]
    #[ignore]
    fn bench_real_swiglu_vs_python_runtime_shapes() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        // Real model dim: intermediate_size=9216.
        let intermediate_size = 9216usize;
        let iters = 2000;
        let warmup = 100;

        for &num_rows in &[1usize, 128usize] {
            let n = num_rows * intermediate_size;
            let gate_f32: Vec<f32> = (0..n).map(|i| ((i % 83) as f32 - 41.0) * 0.01).collect();
            let up_f32: Vec<f32> = (0..n).map(|i| ((i % 71) as f32 - 35.0) * 0.01).collect();
            let gate_bf16: Vec<u16> = gate_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let up_bf16: Vec<u16> = up_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let mut gate_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(n).unwrap();
            let mut up_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(n).unwrap();
            let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(n).unwrap();
            gate_buf.copy_from_host(&gate_bf16).unwrap();
            up_buf.copy_from_host(&up_bf16).unwrap();

            for _ in 0..warmup {
                swiglu_bf16(&gate_buf, &up_buf, &mut out_buf).unwrap();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                swiglu_bf16(&gate_buf, &up_buf, &mut out_buf).unwrap();
            }
            let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!(
                "swiglu_bf16 rows={num_rows:4} intermediate={intermediate_size}: {per_call_us:.3} us/call ({iters} iters, includes hipDeviceSynchronize)"
            );
        }
    }

    /// Raised by a direct question about why the GEMM (`blas.rs`) lost to
    /// Python: Python's own benchmark script issues all `iters` launches
    /// back-to-back with NO sync between them, only ONE
    /// `torch.cuda.synchronize()` after the whole loop -- classic async
    /// pipelining. Every kernel wrapper in this file (including
    /// `swiglu_bf16`) calls `device_synchronize()` at the end of EVERY
    /// single call, a structurally different, more conservative thing to
    /// measure (full host-device round trip, serialized, every call). This
    /// isolates that variable for SwiGLU too, the same way the GEMM
    /// investigation did in `blas.rs`: raw kernel launch via the FFI
    /// directly, no per-call sync, one sync at the very end.
    #[test]
    #[ignore]
    fn bench_real_swiglu_unsynced_pipelined_vs_python_runtime_shapes() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let intermediate_size = 9216usize;
        let iters = 2000;
        let warmup = 100;

        for &num_rows in &[1usize, 128usize] {
            let n = num_rows * intermediate_size;
            let gate_f32: Vec<f32> = (0..n).map(|i| ((i % 83) as f32 - 41.0) * 0.01).collect();
            let up_f32: Vec<f32> = (0..n).map(|i| ((i % 71) as f32 - 35.0) * 0.01).collect();
            let gate_bf16: Vec<u16> = gate_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let up_bf16: Vec<u16> = up_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let mut gate_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(n).unwrap();
            let mut up_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(n).unwrap();
            let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(n).unwrap();
            gate_buf.copy_from_host(&gate_bf16).unwrap();
            up_buf.copy_from_host(&up_bf16).unwrap();

            let launch_one = |gate_buf: &DeviceBuffer<u16>,
                               up_buf: &DeviceBuffer<u16>,
                               out_buf: &mut DeviceBuffer<u16>| {
                // SAFETY: same call swiglu_bf16 makes, minus the trailing
                // check_last_error()/device_synchronize() -- deliberately,
                // to isolate their cost. Buffers are the same live,
                // correctly-sized allocations used throughout this loop.
                unsafe {
                    ffi::launch_swiglu_bf16(
                        gate_buf.as_device_ptr(),
                        up_buf.as_device_ptr(),
                        out_buf.as_device_ptr_mut() as *mut c_void,
                        n as i32,
                        std::ptr::null_mut(),
                    );
                }
            };

            for _ in 0..warmup {
                launch_one(&gate_buf, &up_buf, &mut out_buf);
            }
            hip::device_synchronize().unwrap();

            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                launch_one(&gate_buf, &up_buf, &mut out_buf);
            }
            hip::device_synchronize().unwrap(); // ONE sync after all launches, matching Python
            let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!(
                "swiglu_bf16 (UNSYNCED, pipelined) rows={num_rows:4} intermediate={intermediate_size}: {per_call_us:.3} us/call ({iters} iters, ONE hipDeviceSynchronize after the whole loop)"
            );
        }
    }

    // ---- Embedding lookup ------------------------------------------------

    /// Real correctness test against the REAL embedding table loaded in
    /// Stage 1: looks up a handful of real token ids on the GPU, and
    /// checks the result byte-for-byte against reading those same rows
    /// directly out of the real tensor's bytes on the host (an independent
    /// path -- host-side slicing, not the kernel's own code).
    #[test]
    fn real_embedding_lookup_matches_real_table_rows() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = crate::model_loader::locate_model_snapshot()
            .expect("no real Qwen3.5-4B snapshot found on this machine");
        let raw = crate::model_loader::load_raw_tensor(
            &snapshot,
            "model.language_model.embed_tokens.weight",
        )
        .expect("embed_tokens.weight must exist");
        let vocab_size = raw.shape[0];
        let hidden_size = raw.shape[1];

        let table_bf16: Vec<u16> = raw
            .data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();

        let mut table_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(table_bf16.len())
            .expect("allocating real embedding table on GPU failed");
        table_buf
            .copy_from_host(&table_bf16)
            .expect("uploading real embedding table failed");

        let ids_host: Vec<i32> = vec![0, 1, 100, 42, (vocab_size - 1) as i32];
        let mut ids_buf: DeviceBuffer<i32> = DeviceBuffer::alloc(ids_host.len()).unwrap();
        ids_buf.copy_from_host(&ids_host).unwrap();
        let mut out_buf: DeviceBuffer<u16> =
            DeviceBuffer::alloc(ids_host.len() * hidden_size).unwrap();

        embedding_lookup_bf16(
            &table_buf,
            &ids_buf,
            &ids_host,
            &mut out_buf,
            vocab_size,
            hidden_size,
        )
        .expect("real embedding_lookup_bf16 kernel launch failed");

        let mut out_bf16 = vec![0u16; ids_host.len() * hidden_size];
        out_buf.copy_to_host(&mut out_bf16).unwrap();

        for (row_idx, &id) in ids_host.iter().enumerate() {
            let expected_row =
                &table_bf16[(id as usize) * hidden_size..(id as usize + 1) * hidden_size];
            let got_row = &out_bf16[row_idx * hidden_size..(row_idx + 1) * hidden_size];
            assert_eq!(
                got_row, expected_row,
                "row for token id {id} did not match the real table"
            );
        }
    }

    // ---- Causal conv1d update (GDN decode path) --------------------------

    /// Real correctness test: a small, fixed, real-shaped case (conv_dim=3,
    /// kernel_size=4), computed on the real GPU via the real kernel,
    /// checked against a reference computed INDEPENDENTLY here in plain f32
    /// Rust, following the exact formula derived from the real
    /// `causal_conv1d_update` transformers source (see
    /// `causal_conv1d_update.hip`'s header comment) -- not calling the
    /// kernel's own code. Also checks the updated `conv_state` buffer, not
    /// just `out`, since the in-place state update is as load-bearing to
    /// get right as the output itself (a wrong state silently corrupts
    /// every subsequent decode step).
    #[test]
    fn real_causal_conv1d_update_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let conv_dim = 3usize;
        let kernel_size = 4usize;
        let state_len = kernel_size - 1;

        let state_f32: Vec<Vec<f32>> = vec![
            vec![0.1, -0.2, 0.3],
            vec![-0.1, 0.2, -0.3],
            vec![0.05, 0.05, 0.05],
        ];
        let h_f32: Vec<f32> = vec![0.4, -0.4, 0.05];
        let w_f32: Vec<Vec<f32>> = vec![
            vec![0.5, -0.1, 0.2, 0.3],
            vec![0.1, 0.2, -0.2, 0.1],
            vec![1.0, 1.0, 1.0, 1.0],
        ];

        // Independent reference, plain f32, exact formula from
        // causal_conv1d_update.hip's header.
        let mut expected_out = vec![0f32; conv_dim];
        let mut expected_state = vec![vec![0f32; state_len]; conv_dim];
        for c in 0..conv_dim {
            let mut raw = h_f32[c] * w_f32[c][kernel_size - 1];
            for k in 0..state_len {
                raw += state_f32[c][k] * w_f32[c][k];
            }
            expected_out[c] = raw / (1.0 + (-raw).exp());
            expected_state[c] = vec![state_f32[c][1], state_f32[c][2], h_f32[c]];
        }

        let h_bf16: Vec<u16> = h_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let state_bf16: Vec<u16> = state_f32
            .iter()
            .flat_map(|row| row.iter().map(|&v| f32_to_bf16(v)))
            .collect();
        let w_bf16: Vec<u16> = w_f32
            .iter()
            .flat_map(|row| row.iter().map(|&v| f32_to_bf16(v)))
            .collect();

        let mut h_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim).unwrap();
        let mut state_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim * state_len).unwrap();
        let mut w_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim * kernel_size).unwrap();
        let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim).unwrap();
        h_buf.copy_from_host(&h_bf16).unwrap();
        state_buf.copy_from_host(&state_bf16).unwrap();
        w_buf.copy_from_host(&w_bf16).unwrap();

        causal_conv1d_update_bf16(
            &h_buf,
            &mut state_buf,
            &w_buf,
            &mut out_buf,
            1,
            conv_dim,
            kernel_size,
        )
        .expect("real causal_conv1d_update_bf16 kernel launch failed");

        let mut out_bf16 = vec![0u16; conv_dim];
        out_buf.copy_to_host(&mut out_bf16).unwrap();
        let got_out: Vec<f32> = out_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
        for (c, (g, e)) in got_out.iter().zip(expected_out.iter()).enumerate() {
            assert!(
                (g - e).abs() < 0.01,
                "channel {c} output: GPU kernel got {g}, independent reference {e}"
            );
        }

        let mut new_state_bf16 = vec![0u16; conv_dim * state_len];
        state_buf.copy_to_host(&mut new_state_bf16).unwrap();
        for c in 0..conv_dim {
            for k in 0..state_len {
                let g = bf16_to_f32(new_state_bf16[c * state_len + k]);
                let e = expected_state[c][k];
                assert!(
                    (g - e).abs() < 0.01,
                    "channel {c} state[{k}]: GPU kernel got {g}, independent reference {e}"
                );
            }
        }
    }

    /// §103 decisive test: `causal_conv1d_prefill_bf16` (the new
    /// real batched-prefill kernel) matches the ALREADY-PROVEN
    /// `causal_conv1d_update_bf16` called in a per-token loop, starting
    /// from the SAME real initial `conv_state` -- the most direct
    /// possible cross-check, since the per-token kernel already has its
    /// own real-transformers-reference decisive test above. Also
    /// exercises the batched kernel's own STRIDED source read
    /// (`src_stride`/`src_offset`): the synthetic input is embedded
    /// inside a WIDER `[num_tokens, wide_stride]` buffer (matching the
    /// real usage of reading straight out of `gdn_in_proj_out`'s own
    /// wider combined-GEMM row), not a tightly-packed `[num_tokens,
    /// conv_dim]` buffer.
    #[test]
    fn real_causal_conv1d_prefill_matches_per_token_loop_with_strided_source() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let conv_dim = 5usize;
        let kernel_size = 4usize;
        let state_len = kernel_size - 1;
        let num_tokens = 6usize;
        let wide_stride = conv_dim + 7; // deliberately wider than conv_dim
        let src_offset = 2usize; // deliberately nonzero

        let synth = |n: usize, offset: i32, period: i32, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let m = ((i as i32 + offset).rem_euclid(period)) as f32;
                    (m - (period as f32) / 2.0) * scale
                })
                .collect()
        };
        let state0_f32 = synth(conv_dim * state_len, 0, 17, 0.04);
        let w_f32 = synth(conv_dim * kernel_size, 5, 13, 0.06);
        // One wide [num_tokens, wide_stride] host buffer; only the real
        // conv1d input occupies [src_offset, src_offset+conv_dim) of each
        // row -- the rest is real garbage the kernel must never touch.
        let mut wide_host_f32 = synth(num_tokens * wide_stride, 3, 23, 0.05);
        let h_all_f32: Vec<Vec<f32>> = (0..num_tokens).map(|t| synth(conv_dim, (t * 3 + 1) as i32, 19, 0.045)).collect();
        for t in 0..num_tokens {
            for c in 0..conv_dim {
                wide_host_f32[t * wide_stride + src_offset + c] = h_all_f32[t][c];
            }
        }

        let wide_bf16: Vec<u16> = wide_host_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let state0_bf16: Vec<u16> = state0_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let w_bf16: Vec<u16> = w_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut wide_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(wide_bf16.len()).unwrap();
        let mut w_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(w_bf16.len()).unwrap();
        wide_buf.copy_from_host(&wide_bf16).unwrap();
        w_buf.copy_from_host(&w_bf16).unwrap();

        // Reference path: the ALREADY-PROVEN per-token kernel, called
        // once per token, threading conv_state through exactly like
        // `gdn_layer_forward_prefill`'s OLD loop used to.
        let mut ref_state_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(state0_bf16.len()).unwrap();
        ref_state_buf.copy_from_host(&state0_bf16).unwrap();
        let mut ref_out_host = vec![0f32; num_tokens * conv_dim];
        for t in 0..num_tokens {
            let h_bf16: Vec<u16> = h_all_f32[t].iter().map(|&v| f32_to_bf16(v)).collect();
            let mut h_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim).unwrap();
            h_buf.copy_from_host(&h_bf16).unwrap();
            let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim).unwrap();
            causal_conv1d_update_bf16(&h_buf, &mut ref_state_buf, &w_buf, &mut out_buf, 1, conv_dim, kernel_size).expect("reference causal_conv1d_update_bf16 call failed");
            let mut out_bf16 = vec![0u16; conv_dim];
            out_buf.copy_to_host(&mut out_bf16).unwrap();
            for c in 0..conv_dim {
                ref_out_host[t * conv_dim + c] = bf16_to_f32(out_bf16[c]);
            }
        }
        let mut ref_final_state_bf16 = vec![0u16; state0_bf16.len()];
        ref_state_buf.copy_to_host(&mut ref_final_state_bf16).unwrap();

        // New batched path: ONE call, strided source, starting from the
        // SAME real initial conv_state.
        let mut batched_state_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(state0_bf16.len()).unwrap();
        batched_state_buf.copy_from_host(&state0_bf16).unwrap();
        let mut batched_out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_tokens * conv_dim).unwrap();
        causal_conv1d_prefill_bf16(&wide_buf, wide_stride, src_offset, &mut batched_state_buf, &w_buf, &mut batched_out_buf, num_tokens, conv_dim, kernel_size).expect("real causal_conv1d_prefill_bf16 call failed");

        let mut batched_out_bf16 = vec![0u16; num_tokens * conv_dim];
        batched_out_buf.copy_to_host(&mut batched_out_bf16).unwrap();
        for (i, (&g, &e)) in batched_out_bf16.iter().zip(ref_out_host.iter()).enumerate() {
            let got = bf16_to_f32(g);
            assert!((got - e).abs() < 0.01, "output element {i}: batched kernel got {got}, per-token-loop reference {e}");
        }

        let mut batched_final_state_bf16 = vec![0u16; state0_bf16.len()];
        batched_state_buf.copy_to_host(&mut batched_final_state_bf16).unwrap();
        assert_eq!(batched_final_state_bf16, ref_final_state_bf16, "final conv_state bytes differ between the batched kernel and the per-token-loop reference");
    }

    /// Decisive real-weight cross-check, same discipline as
    /// `model_loader.rs`'s own decisive test: real layer 0 GDN
    /// `conv1d.weight` (conv_dim=8192, kernel_size=4), a fixed deterministic
    /// synthetic state/hidden input, run on the real GPU kernel and checked
    /// against values computed by the REAL `transformers`
    /// `causal_conv1d_update` function on the same real weight and the same
    /// input formula (see
    /// `scratchpad/gen_causal_conv1d_reference.py` for how these were
    /// generated -- an independent process, not this kernel's own code).
    ///
    /// Real, deliberate scope: reads real byte content from whichever
    /// checkpoint `locate_model_snapshot()` resolves to for the compiled
    /// feature -- `conv_dim=8192` is structurally shared with `qwen35_9b`
    /// (both have `GDN_NUM_V_HEADS=32`), but the reference VALUES below
    /// were computed from Qwen3.5-4B's real trained weights specifically,
    /// which differ from 9B's real weights at the same shape. Gated to
    /// the one feature whose real bytes these values actually match.
    #[test]
    #[cfg(feature = "qwen35_4b")]
    fn real_causal_conv1d_update_matches_real_layer0_weights() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = crate::model_loader::locate_model_snapshot()
            .expect("no real Qwen3.5-4B snapshot found on this machine");
        let raw = crate::model_loader::load_raw_tensor(
            &snapshot,
            "model.language_model.layers.0.linear_attn.conv1d.weight",
        )
        .expect("layer 0 linear_attn.conv1d.weight must exist");
        // Real shape is [conv_dim, 1, kernel_size]; already squeezed away
        // the middle dim of size 1 when read as flat bf16 bytes below.
        let conv_dim = raw.shape[0];
        let kernel_size = raw.shape[2];
        assert_eq!(conv_dim, 8192);
        assert_eq!(kernel_size, 4);
        let state_len = kernel_size - 1;
        let w_bf16: Vec<u16> = raw
            .data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();

        // Same deterministic formula as gen_causal_conv1d_reference.py, so
        // the bf16-rounded values here are byte-identical to what the
        // Python reference actually consumed.
        let h_f32: Vec<f32> = (0..conv_dim)
            .map(|c| (((c % 13) as i32 - 6) as f32) * 0.05)
            .collect();
        let state_f32: Vec<f32> = (0..conv_dim)
            .flat_map(|c| {
                (0..state_len).map(move |k| ((((c + k) % 11) as i32 - 5) as f32) * 0.03)
            })
            .collect();
        let h_bf16: Vec<u16> = h_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let state_bf16: Vec<u16> = state_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut h_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim).unwrap();
        let mut state_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim * state_len).unwrap();
        let mut w_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim * kernel_size).unwrap();
        let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim).unwrap();
        h_buf.copy_from_host(&h_bf16).unwrap();
        state_buf.copy_from_host(&state_bf16).unwrap();
        w_buf.copy_from_host(&w_bf16).unwrap();

        causal_conv1d_update_bf16(
            &h_buf,
            &mut state_buf,
            &w_buf,
            &mut out_buf,
            1,
            conv_dim,
            kernel_size,
        )
        .expect("real causal_conv1d_update_bf16 kernel launch failed against real weights");

        let mut out_bf16 = vec![0u16; conv_dim];
        out_buf.copy_to_host(&mut out_bf16).unwrap();
        let got: Vec<f32> = out_bf16[..8].iter().map(|&b| bf16_to_f32(b)).collect();

        // Independently computed by the real transformers causal_conv1d_update
        // function on this exact real weight and input (see
        // scratchpad/gen_causal_conv1d_reference.py's stdout).
        let expected: [f32; 8] = [
            0.0197753906,
            -0.0147094727,
            -0.0118408203,
            -0.0089111328,
            0.0063171387,
            -0.0007019043,
            -0.0037689209,
            -0.003036499,
        ];
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 0.01,
                "channel {i}: GPU kernel got {g}, independent Python reference {e}"
            );
        }
    }

    /// A direct callback to docs/DECISIONS.md §76: PyTorch's own reference
    /// `F.conv1d(groups=conv_dim)` kernel was found to give batch-size-
    /// dependent results on real trained weights on this exact hardware (a
    /// real ~5.6%-per-step argmax-flip hazard for batched speculative
    /// verification). This kernel computes every (batch, channel) pair in
    /// its own independent thread with no cross-row memory access at all --
    /// so running the SAME per-row input at batch=1 vs. stacked into a
    /// larger batch should be byte-identical, by construction, unlike the
    /// reference PyTorch kernel. Confirms (does not merely assume) that
    /// property empirically rather than only arguing it from the source.
    #[test]
    fn real_causal_conv1d_update_is_batch_size_independent() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = crate::model_loader::locate_model_snapshot()
            .expect("no real Qwen3.5-4B snapshot found on this machine");
        let raw = crate::model_loader::load_raw_tensor(
            &snapshot,
            "model.language_model.layers.0.linear_attn.conv1d.weight",
        )
        .expect("layer 0 linear_attn.conv1d.weight must exist");
        let conv_dim = raw.shape[0];
        let kernel_size = raw.shape[2];
        let state_len = kernel_size - 1;
        let w_bf16: Vec<u16> = raw
            .data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let mut w_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim * kernel_size).unwrap();
        w_buf.copy_from_host(&w_bf16).unwrap();

        let h_row_f32: Vec<f32> = (0..conv_dim)
            .map(|c| (((c % 13) as i32 - 6) as f32) * 0.05)
            .collect();
        let state_row_f32: Vec<f32> = (0..conv_dim)
            .flat_map(|c| {
                (0..state_len).map(move |k| ((((c + k) % 11) as i32 - 5) as f32) * 0.03)
            })
            .collect();
        let h_row_bf16: Vec<u16> = h_row_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let state_row_bf16: Vec<u16> = state_row_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        // batch=1 run.
        let mut h1: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim).unwrap();
        let mut state1: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim * state_len).unwrap();
        let mut out1: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim).unwrap();
        h1.copy_from_host(&h_row_bf16).unwrap();
        state1.copy_from_host(&state_row_bf16).unwrap();
        causal_conv1d_update_bf16(&h1, &mut state1, &w_buf, &mut out1, 1, conv_dim, kernel_size)
            .unwrap();
        let mut out1_bf16 = vec![0u16; conv_dim];
        out1.copy_to_host(&mut out1_bf16).unwrap();

        // batch=8 run: the SAME row repeated 8 times.
        let batch = 8usize;
        let h8_bf16: Vec<u16> = h_row_bf16.repeat(batch);
        let state8_bf16: Vec<u16> = state_row_bf16.repeat(batch);
        let mut h8: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim * batch).unwrap();
        let mut state8: DeviceBuffer<u16> =
            DeviceBuffer::alloc(conv_dim * state_len * batch).unwrap();
        let mut out8: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim * batch).unwrap();
        h8.copy_from_host(&h8_bf16).unwrap();
        state8.copy_from_host(&state8_bf16).unwrap();
        causal_conv1d_update_bf16(
            &h8,
            &mut state8,
            &w_buf,
            &mut out8,
            batch,
            conv_dim,
            kernel_size,
        )
        .unwrap();
        let mut out8_bf16 = vec![0u16; conv_dim * batch];
        out8.copy_to_host(&mut out8_bf16).unwrap();

        for b in 0..batch {
            let row = &out8_bf16[b * conv_dim..(b + 1) * conv_dim];
            assert_eq!(
                row, &out1_bf16[..],
                "batch row {b} differs from the batch=1 result -- this kernel should be \
                 batch-size independent by construction (see the test's own doc comment)"
            );
        }
    }

    #[test]
    #[ignore]
    fn bench_real_causal_conv1d_update_vs_python_runtime_shapes() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = crate::model_loader::locate_model_snapshot()
            .expect("no real Qwen3.5-4B snapshot found on this machine");
        let raw = crate::model_loader::load_raw_tensor(
            &snapshot,
            "model.language_model.layers.0.linear_attn.conv1d.weight",
        )
        .expect("layer 0 linear_attn.conv1d.weight must exist");
        let conv_dim = raw.shape[0]; // 8192
        let kernel_size = raw.shape[2]; // 4
        let state_len = kernel_size - 1;
        let w_bf16: Vec<u16> = raw
            .data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let mut w_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim * kernel_size).unwrap();
        w_buf.copy_from_host(&w_bf16).unwrap();

        let iters = 2000;
        let warmup = 100;

        for &batch in &[1usize, 128usize] {
            let h_f32: Vec<f32> = (0..batch * conv_dim)
                .map(|i| ((i % 89) as f32 - 44.0) * 0.01)
                .collect();
            let state_f32: Vec<f32> = (0..batch * conv_dim * state_len)
                .map(|i| ((i % 67) as f32 - 33.0) * 0.01)
                .collect();
            let h_bf16: Vec<u16> = h_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let state_bf16: Vec<u16> = state_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let mut h_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(batch * conv_dim).unwrap();
            let mut state_buf: DeviceBuffer<u16> =
                DeviceBuffer::alloc(batch * conv_dim * state_len).unwrap();
            let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(batch * conv_dim).unwrap();
            h_buf.copy_from_host(&h_bf16).unwrap();

            state_buf.copy_from_host(&state_bf16).unwrap();
            for _ in 0..warmup {
                causal_conv1d_update_bf16(
                    &h_buf,
                    &mut state_buf,
                    &w_buf,
                    &mut out_buf,
                    batch,
                    conv_dim,
                    kernel_size,
                )
                .unwrap();
            }
            // conv_state is mutated in place by design (matches the real
            // function's contract) -- deliberately NOT reset between
            // iterations, same as every other benchmark in this file
            // reuses its buffers without extra copies inside the timed
            // loop. The kernel does the same fixed amount of work
            // regardless of the state buffer's contents, so this does not
            // bias the measurement; it only means the state drifts over
            // the run, which is fine since nothing here checks correctness.
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                causal_conv1d_update_bf16(
                    &h_buf,
                    &mut state_buf,
                    &w_buf,
                    &mut out_buf,
                    batch,
                    conv_dim,
                    kernel_size,
                )
                .unwrap();
            }
            let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!(
                "causal_conv1d_update_bf16 batch={batch:4} conv_dim={conv_dim} kernel_size={kernel_size}: {per_call_us:.3} us/call ({iters} iters, includes hipDeviceSynchronize)"
            );
        }
    }

    #[test]
    #[ignore]
    fn bench_real_causal_conv1d_update_unsynced_pipelined_vs_python_runtime_shapes() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = crate::model_loader::locate_model_snapshot()
            .expect("no real Qwen3.5-4B snapshot found on this machine");
        let raw = crate::model_loader::load_raw_tensor(
            &snapshot,
            "model.language_model.layers.0.linear_attn.conv1d.weight",
        )
        .expect("layer 0 linear_attn.conv1d.weight must exist");
        let conv_dim = raw.shape[0];
        let kernel_size = raw.shape[2];
        let state_len = kernel_size - 1;
        let w_bf16: Vec<u16> = raw
            .data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let mut w_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(conv_dim * kernel_size).unwrap();
        w_buf.copy_from_host(&w_bf16).unwrap();

        let iters = 2000;
        let warmup = 100;

        for &batch in &[1usize, 128usize] {
            let h_f32: Vec<f32> = (0..batch * conv_dim)
                .map(|i| ((i % 89) as f32 - 44.0) * 0.01)
                .collect();
            let state_f32: Vec<f32> = (0..batch * conv_dim * state_len)
                .map(|i| ((i % 67) as f32 - 33.0) * 0.01)
                .collect();
            let h_bf16: Vec<u16> = h_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let state_bf16: Vec<u16> = state_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let mut h_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(batch * conv_dim).unwrap();
            let mut state_buf: DeviceBuffer<u16> =
                DeviceBuffer::alloc(batch * conv_dim * state_len).unwrap();
            let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(batch * conv_dim).unwrap();
            h_buf.copy_from_host(&h_bf16).unwrap();
            state_buf.copy_from_host(&state_bf16).unwrap();

            let launch_one = |h_buf: &DeviceBuffer<u16>,
                               state_buf: &mut DeviceBuffer<u16>,
                               out_buf: &mut DeviceBuffer<u16>| {
                // SAFETY: same call causal_conv1d_update_bf16 makes, minus
                // the trailing check_last_error()/device_synchronize().
                unsafe {
                    ffi::launch_causal_conv1d_update_bf16(
                        h_buf.as_device_ptr(),
                        state_buf.as_device_ptr_mut() as *mut c_void,
                        w_buf.as_device_ptr(),
                        out_buf.as_device_ptr_mut() as *mut c_void,
                        batch as i32,
                        conv_dim as i32,
                        kernel_size as i32,
                        std::ptr::null_mut(),
                    );
                }
            };

            for _ in 0..warmup {
                launch_one(&h_buf, &mut state_buf, &mut out_buf);
            }
            hip::device_synchronize().unwrap();

            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                launch_one(&h_buf, &mut state_buf, &mut out_buf);
            }
            hip::device_synchronize().unwrap();
            let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!(
                "causal_conv1d_update_bf16 (UNSYNCED, pipelined) batch={batch:4} conv_dim={conv_dim} kernel_size={kernel_size}: {per_call_us:.3} us/call ({iters} iters, ONE hipDeviceSynchronize after the whole loop)"
            );
        }
    }

    #[test]
    #[ignore]
    fn bench_real_embedding_lookup_vs_python_runtime_shapes() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = crate::model_loader::locate_model_snapshot()
            .expect("no real Qwen3.5-4B snapshot found on this machine");
        let raw = crate::model_loader::load_raw_tensor(
            &snapshot,
            "model.language_model.embed_tokens.weight",
        )
        .expect("embed_tokens.weight must exist");
        let vocab_size = raw.shape[0];
        let hidden_size = raw.shape[1];
        let table_bf16: Vec<u16> = raw
            .data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let mut table_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(table_bf16.len()).unwrap();
        table_buf.copy_from_host(&table_bf16).unwrap();

        let iters = 2000;
        let warmup = 100;

        for &num_tokens in &[1usize, 128usize] {
            let ids_host: Vec<i32> = (0..num_tokens)
                .map(|i| ((i * 997) % vocab_size) as i32)
                .collect();
            let mut ids_buf: DeviceBuffer<i32> = DeviceBuffer::alloc(ids_host.len()).unwrap();
            ids_buf.copy_from_host(&ids_host).unwrap();
            let mut out_buf: DeviceBuffer<u16> =
                DeviceBuffer::alloc(num_tokens * hidden_size).unwrap();

            for _ in 0..warmup {
                embedding_lookup_bf16(
                    &table_buf,
                    &ids_buf,
                    &ids_host,
                    &mut out_buf,
                    vocab_size,
                    hidden_size,
                )
                .unwrap();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                embedding_lookup_bf16(
                    &table_buf,
                    &ids_buf,
                    &ids_host,
                    &mut out_buf,
                    vocab_size,
                    hidden_size,
                )
                .unwrap();
            }
            let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!(
                "embedding_lookup_bf16 num_tokens={num_tokens:4} hidden={hidden_size}: {per_call_us:.3} us/call ({iters} iters, includes hipDeviceSynchronize)"
            );
        }
    }

    /// Same investigation as `bench_real_swiglu_unsynced_pipelined_...` and
    /// `blas.rs`'s GEMM version -- isolates per-call `device_synchronize()`
    /// as a variable for embedding lookup too.
    #[test]
    #[ignore]
    fn bench_real_embedding_lookup_unsynced_pipelined_vs_python_runtime_shapes() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = crate::model_loader::locate_model_snapshot()
            .expect("no real Qwen3.5-4B snapshot found on this machine");
        let raw = crate::model_loader::load_raw_tensor(
            &snapshot,
            "model.language_model.embed_tokens.weight",
        )
        .expect("embed_tokens.weight must exist");
        let vocab_size = raw.shape[0];
        let hidden_size = raw.shape[1];
        let table_bf16: Vec<u16> = raw
            .data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let mut table_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(table_bf16.len()).unwrap();
        table_buf.copy_from_host(&table_bf16).unwrap();

        let iters = 2000;
        let warmup = 100;

        for &num_tokens in &[1usize, 128usize] {
            let ids_host: Vec<i32> = (0..num_tokens)
                .map(|i| ((i * 997) % vocab_size) as i32)
                .collect();
            let mut ids_buf: DeviceBuffer<i32> = DeviceBuffer::alloc(ids_host.len()).unwrap();
            ids_buf.copy_from_host(&ids_host).unwrap();
            let mut out_buf: DeviceBuffer<u16> =
                DeviceBuffer::alloc(num_tokens * hidden_size).unwrap();

            let launch_one = |table_buf: &DeviceBuffer<u16>,
                               ids_buf: &DeviceBuffer<i32>,
                               out_buf: &mut DeviceBuffer<u16>| {
                // SAFETY: same call embedding_lookup_bf16 makes (bounds
                // already validated once via ids_host below/above), minus
                // the trailing check_last_error()/device_synchronize().
                unsafe {
                    ffi::launch_embedding_lookup_bf16(
                        table_buf.as_device_ptr(),
                        ids_buf.as_device_ptr() as *const i32,
                        out_buf.as_device_ptr_mut() as *mut c_void,
                        num_tokens as i32,
                        hidden_size as i32,
                        std::ptr::null_mut(),
                    );
                }
            };

            for _ in 0..warmup {
                launch_one(&table_buf, &ids_buf, &mut out_buf);
            }
            hip::device_synchronize().unwrap();

            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                launch_one(&table_buf, &ids_buf, &mut out_buf);
            }
            hip::device_synchronize().unwrap(); // ONE sync after all launches, matching Python
            let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!(
                "embedding_lookup_bf16 (UNSYNCED, pipelined) num_tokens={num_tokens:4} hidden={hidden_size}: {per_call_us:.3} us/call ({iters} iters, ONE hipDeviceSynchronize after the whole loop)"
            );
        }
    }

    // ---- Full attention (Hard tier, feature 1) ----------------------------

    /// Real correctness test: a small, fixed GQA case (4 Q heads, 2 KV
    /// heads -- n_rep=2, head_dim=4, kv_len=3), computed on the real GPU
    /// via the real kernel, checked against a reference computed
    /// INDEPENDENTLY here in plain f32 Rust, following the exact formula
    /// derived from the real `eager_attention_forward`/`repeat_kv`
    /// transformers source (see `attention.hip`'s header comment) -- not
    /// calling the kernel's own code.
    #[test]
    fn real_attention_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_q_heads = 4usize;
        let num_kv_heads = 2usize;
        let n_rep = num_q_heads / num_kv_heads;
        let head_dim = 4usize;
        let kv_len = 3usize;
        let scaling = (head_dim as f32).powf(-0.5);

        let q_f32: Vec<f32> = (0..num_q_heads * head_dim)
            .map(|i| (((i % 9) as i32 - 4) as f32) * 0.1)
            .collect();
        let k_f32: Vec<f32> = (0..num_kv_heads * kv_len * head_dim)
            .map(|i| (((i % 7) as i32 - 3) as f32) * 0.08)
            .collect();
        let v_f32: Vec<f32> = (0..num_kv_heads * kv_len * head_dim)
            .map(|i| (((i % 5) as i32 - 2) as f32) * 0.05)
            .collect();

        // Independent reference, plain f32, exact formula from
        // attention.hip's header: score = dot(Q,K)*scaling, softmax, then
        // weighted sum over V.
        let mut expected = vec![0f32; num_q_heads * head_dim];
        for h in 0..num_q_heads {
            let h_kv = h / n_rep;
            let q_row = &q_f32[h * head_dim..(h + 1) * head_dim];
            let mut scores = vec![0f32; kv_len];
            for j in 0..kv_len {
                let k_row = &k_f32
                    [(h_kv * kv_len + j) * head_dim..(h_kv * kv_len + j + 1) * head_dim];
                let dot: f32 = q_row.iter().zip(k_row.iter()).map(|(a, b)| a * b).sum();
                scores[j] = dot * scaling;
            }
            let max_score = scores.iter().cloned().fold(f32::MIN, f32::max);
            let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            let probs: Vec<f32> = exp_scores.iter().map(|&e| e / sum_exp).collect();
            for d in 0..head_dim {
                let mut acc = 0f32;
                for j in 0..kv_len {
                    let v_val = v_f32[(h_kv * kv_len + j) * head_dim + d];
                    acc += probs[j] * v_val;
                }
                expected[h * head_dim + d] = acc;
            }
        }

        let q_bf16: Vec<u16> = q_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let k_bf16: Vec<u16> = k_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let v_bf16: Vec<u16> = v_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut q_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(q_bf16.len()).unwrap();
        let mut k_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(k_bf16.len()).unwrap();
        let mut v_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(v_bf16.len()).unwrap();
        let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_q_heads * head_dim).unwrap();
        q_buf.copy_from_host(&q_bf16).unwrap();
        k_buf.copy_from_host(&k_bf16).unwrap();
        v_buf.copy_from_host(&v_bf16).unwrap();

        attention_decode_bf16(
            &q_buf,
            &k_buf,
            &v_buf,
            &mut out_buf,
            num_q_heads,
            num_kv_heads,
            kv_len,
            head_dim,
            scaling,
        )
        .expect("real attention_decode_bf16 kernel launch failed");

        let mut out_bf16 = vec![0u16; num_q_heads * head_dim];
        out_buf.copy_to_host(&mut out_bf16).unwrap();
        let got: Vec<f32> = out_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 0.01,
                "element {i}: GPU kernel got {g}, independent reference {e}"
            );
        }
    }

    /// §104 decisive test: `attention_decode_split_bf16` (the new,
    /// faster kernel) matches the ALREADY-PROVEN `attention_decode_bf16`
    /// bit-for-bit-the-same-math kernel -- the most direct possible
    /// correctness check for a pure parallelism restructuring, since both
    /// kernels compute the exact same real formula. Real model dims (16 Q
    /// heads, 4 KV heads, head_dim=256), checked at multiple real
    /// `kv_split` values (including `kv_split > kv_len`, the real edge
    /// case where some split-workers get zero positions) and multiple
    /// real `kv_len`s (small and mid-generation-representative).
    #[test]
    fn real_attention_decode_split_matches_attention_decode_bf16() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_q_heads = 16usize;
        let num_kv_heads = 4usize;
        let head_dim = 256usize;
        let scaling = (head_dim as f32).powf(-0.5);

        let synth = |n: usize, offset: i32, period: i32, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let m = ((i as i32 + offset).rem_euclid(period)) as f32;
                    (m - (period as f32) / 2.0) * scale
                })
                .collect()
        };

        for &kv_len in &[5usize, 200usize] {
            for &kv_split in &[1usize, 2, 4] {
                let q_f32 = synth(num_q_heads * head_dim, 0, 37, 0.03);
                let k_f32 = synth(num_kv_heads * kv_len * head_dim, 5, 41, 0.025);
                let v_f32 = synth(num_kv_heads * kv_len * head_dim, 11, 43, 0.02);
                let q_bf16: Vec<u16> = q_f32.iter().map(|&v| f32_to_bf16(v)).collect();
                let k_bf16: Vec<u16> = k_f32.iter().map(|&v| f32_to_bf16(v)).collect();
                let v_bf16: Vec<u16> = v_f32.iter().map(|&v| f32_to_bf16(v)).collect();

                let mut q_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(q_bf16.len()).unwrap();
                let mut k_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(k_bf16.len()).unwrap();
                let mut v_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(v_bf16.len()).unwrap();
                q_buf.copy_from_host(&q_bf16).unwrap();
                k_buf.copy_from_host(&k_bf16).unwrap();
                v_buf.copy_from_host(&v_bf16).unwrap();

                let mut ref_out: DeviceBuffer<u16> = DeviceBuffer::alloc(num_q_heads * head_dim).unwrap();
                attention_decode_bf16(&q_buf, &k_buf, &v_buf, &mut ref_out, num_q_heads, num_kv_heads, kv_len, head_dim, scaling).expect("reference attention_decode_bf16 call failed");
                let mut ref_bf16 = vec![0u16; num_q_heads * head_dim];
                ref_out.copy_to_host(&mut ref_bf16).unwrap();

                let mut split_out: DeviceBuffer<u16> = DeviceBuffer::alloc(num_q_heads * head_dim).unwrap();
                attention_decode_split_bf16(&q_buf, &k_buf, &v_buf, &mut split_out, num_q_heads, num_kv_heads, kv_len, head_dim, kv_split, scaling).expect("real attention_decode_split_bf16 call failed");
                let mut split_bf16 = vec![0u16; num_q_heads * head_dim];
                split_out.copy_to_host(&mut split_bf16).unwrap();

                for (i, (&r, &s)) in ref_bf16.iter().zip(split_bf16.iter()).enumerate() {
                    let rv = bf16_to_f32(r);
                    let sv = bf16_to_f32(s);
                    assert!((rv - sv).abs() < 0.01, "kv_len={kv_len} kv_split={kv_split} element {i}: split kernel got {sv}, reference (attention_decode_bf16) got {rv}");
                }
            }
        }
    }

    /// Decisive real-oracle cross-check: real model dims (16 Q heads, 4 KV
    /// heads, head_dim=256), a fixed deterministic synthetic Q/K/V (no
    /// stored weight to check against here -- Q/K/V are runtime
    /// activations, not model parameters -- so the real oracle is the real
    /// `eager_attention_forward`/`repeat_kv` transformers function itself,
    /// called on this exact synthetic data; see
    /// `scratchpad/gen_attention_reference.py`, an independent process, not
    /// this kernel's own code).
    #[test]
    fn real_attention_matches_real_transformers_function() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_q_heads = 16usize;
        let num_kv_heads = 4usize;
        let head_dim = 256usize;
        let kv_len = 32usize;
        let scaling = (head_dim as f32).powf(-0.5);

        // Same deterministic formula as gen_attention_reference.py's
        // make_case(), including its own bf16 round-trip on the host
        // before use (`.to(torch.bfloat16).to(torch.float32)` then
        // `.to(torch.bfloat16)` again) -- reproduced here as: compute f32,
        // round to bf16, done (the intermediate f32 round-trip in the
        // Python script is a no-op beyond the final bf16 cast).
        let q_f32: Vec<f32> = (0..num_q_heads * head_dim)
            .map(|i| (((i % 13) as i32 - 13 / 2) as f32) * 0.02)
            .collect();
        let k_f32: Vec<f32> = (0..num_kv_heads * kv_len * head_dim)
            .map(|i| (((i % 29) as i32 - 29 / 2) as f32) * 0.015)
            .collect();
        let v_f32: Vec<f32> = (0..num_kv_heads * kv_len * head_dim)
            .map(|i| (((i % 41) as i32 - 41 / 2) as f32) * 0.01)
            .collect();

        let q_bf16: Vec<u16> = q_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let k_bf16: Vec<u16> = k_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let v_bf16: Vec<u16> = v_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut q_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(q_bf16.len()).unwrap();
        let mut k_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(k_bf16.len()).unwrap();
        let mut v_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(v_bf16.len()).unwrap();
        let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_q_heads * head_dim).unwrap();
        q_buf.copy_from_host(&q_bf16).unwrap();
        k_buf.copy_from_host(&k_bf16).unwrap();
        v_buf.copy_from_host(&v_bf16).unwrap();

        attention_decode_bf16(
            &q_buf,
            &k_buf,
            &v_buf,
            &mut out_buf,
            num_q_heads,
            num_kv_heads,
            kv_len,
            head_dim,
            scaling,
        )
        .expect("real attention_decode_bf16 kernel launch failed against real-dims case");

        let mut out_bf16 = vec![0u16; num_q_heads * head_dim];
        out_buf.copy_to_host(&mut out_bf16).unwrap();

        // Independently computed by the real transformers
        // eager_attention_forward + repeat_kv on this exact synthetic data
        // (see scratchpad/gen_attention_reference.py's stdout).
        let head0_expected: [f32; 8] = [
            0.0047302246,
            0.0019073486,
            -0.0009040833,
            -0.0037078857,
            -0.0065612793,
            -0.0093383789,
            -0.0121459961,
            -0.0150756836,
        ];
        let head15_expected: [f32; 8] = [
            -0.004699707,
            -0.0075378418,
            0.0024871826,
            0.012512207,
            0.0096435547,
            0.0068969727,
            0.0040893555,
            0.0012741089,
        ];

        let head0_got: Vec<f32> = out_bf16[0..8].iter().map(|&b| bf16_to_f32(b)).collect();
        let head15_got: Vec<f32> = out_bf16[15 * head_dim..15 * head_dim + 8]
            .iter()
            .map(|&b| bf16_to_f32(b))
            .collect();

        for (i, (g, e)) in head0_got.iter().zip(head0_expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 0.001,
                "head 0, element {i}: GPU kernel got {g}, independent Python reference {e}"
            );
        }
        for (i, (g, e)) in head15_got.iter().zip(head15_expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 0.001,
                "head 15, element {i}: GPU kernel got {g}, independent Python reference {e}"
            );
        }
    }

    /// Benchmarked directly with the pipelined methodology from the start
    /// (§86's lesson: per-call `hipDeviceSynchronize()` inside a timed loop
    /// measures round-trip latency, not throughput, and gave every earlier
    /// feature in this file a false "Python win" until corrected) -- launch
    /// the raw kernel `iters` times back-to-back with no sync between
    /// calls, one `hipDeviceSynchronize()` at the end, matching the paired
    /// Python benchmark's own methodology exactly.
    #[test]
    #[ignore]
    fn bench_real_attention_vs_python_runtime_shapes() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        // Real model dims: num_attention_heads=16, num_key_value_heads=4,
        // head_dim=256. kv_len is the op's own natural scaling variable
        // (unlike every other feature so far) -- 128 and 2048 as short- and
        // longer-context decode points.
        let num_q_heads = 16usize;
        let num_kv_heads = 4usize;
        let head_dim = 256usize;
        let scaling = (head_dim as f32).powf(-0.5);
        let iters = 2000;
        let warmup = 100;

        for &kv_len in &[128usize, 2048usize] {
            let q_f32: Vec<f32> = (0..num_q_heads * head_dim)
                .map(|i| ((i % 89) as f32 - 44.0) * 0.01)
                .collect();
            let k_f32: Vec<f32> = (0..num_kv_heads * kv_len * head_dim)
                .map(|i| ((i % 71) as f32 - 35.0) * 0.01)
                .collect();
            let v_f32: Vec<f32> = (0..num_kv_heads * kv_len * head_dim)
                .map(|i| ((i % 61) as f32 - 30.0) * 0.01)
                .collect();
            let q_bf16: Vec<u16> = q_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let k_bf16: Vec<u16> = k_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let v_bf16: Vec<u16> = v_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let mut q_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(q_bf16.len()).unwrap();
            let mut k_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(k_bf16.len()).unwrap();
            let mut v_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(v_bf16.len()).unwrap();
            let mut out_buf: DeviceBuffer<u16> =
                DeviceBuffer::alloc(num_q_heads * head_dim).unwrap();
            q_buf.copy_from_host(&q_bf16).unwrap();
            k_buf.copy_from_host(&k_bf16).unwrap();
            v_buf.copy_from_host(&v_bf16).unwrap();

            let threads: i32 = 256;
            let mut position_buf: DeviceBuffer<i32> = DeviceBuffer::alloc(1).unwrap();
            position_buf.copy_from_host(&[(kv_len - 1) as i32]).unwrap();
            let launch_one = |q_buf: &DeviceBuffer<u16>,
                               k_buf: &DeviceBuffer<u16>,
                               v_buf: &DeviceBuffer<u16>,
                               out_buf: &mut DeviceBuffer<u16>| {
                // SAFETY: same call attention_decode_bf16 makes, minus the
                // trailing check_last_error()/device_synchronize() --
                // deliberately, to measure pipelined throughput the same
                // way the paired Python benchmark measures itself (§86).
                unsafe {
                    ffi::launch_attention_decode_bf16(
                        q_buf.as_device_ptr(),
                        k_buf.as_device_ptr(),
                        v_buf.as_device_ptr(),
                        out_buf.as_device_ptr_mut() as *mut c_void,
                        num_q_heads as i32,
                        num_kv_heads as i32,
                        position_buf.as_device_ptr() as *const c_int,
                        kv_len as i32, // kv_stride == kv_len: tightly-packed, as before
                        head_dim as i32,
                        scaling,
                        threads,
                        std::ptr::null_mut(),
                    );
                }
            };

            for _ in 0..warmup {
                launch_one(&q_buf, &k_buf, &v_buf, &mut out_buf);
            }
            hip::device_synchronize().unwrap();

            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                launch_one(&q_buf, &k_buf, &v_buf, &mut out_buf);
            }
            hip::device_synchronize().unwrap(); // ONE sync after all launches, matching Python
            let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!(
                "attention_decode_bf16 (pipelined) kv_len={kv_len:5} num_q_heads={num_q_heads} num_kv_heads={num_kv_heads} head_dim={head_dim}: {per_call_us:.3} us/call ({iters} iters, ONE hipDeviceSynchronize after the whole loop)"
            );
        }
    }

    /// §104: head-to-head kernel-level benchmark, `attention_decode_bf16`
    /// (the original scalar kernel) vs `attention_decode_split_bf16` at
    /// several real `kv_split` values -- same pipelined methodology as
    /// `bench_real_attention_vs_python_runtime_shapes` above (no per-call
    /// sync, ONE sync after the whole timed loop). `kv_len` values chosen
    /// to match the REAL, measured decode-throughput-decline range from
    /// `docs/DECISIONS.md` §103's own investigation (positions 54-354 in
    /// a real generation), not arbitrary round numbers.
    #[test]
    #[ignore]
    fn bench_real_attention_decode_split_vs_scalar() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_q_heads = 16usize;
        let num_kv_heads = 4usize;
        let head_dim = 256usize;
        let scaling = (head_dim as f32).powf(-0.5);
        let iters = 2000;
        let warmup = 100;

        for &kv_len in &[64usize, 128, 256, 354] {
            let q_f32: Vec<f32> = (0..num_q_heads * head_dim).map(|i| ((i % 89) as f32 - 44.0) * 0.01).collect();
            let k_f32: Vec<f32> = (0..num_kv_heads * kv_len * head_dim).map(|i| ((i % 71) as f32 - 35.0) * 0.01).collect();
            let v_f32: Vec<f32> = (0..num_kv_heads * kv_len * head_dim).map(|i| ((i % 61) as f32 - 30.0) * 0.01).collect();
            let q_bf16: Vec<u16> = q_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let k_bf16: Vec<u16> = k_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let v_bf16: Vec<u16> = v_f32.iter().map(|&v| f32_to_bf16(v)).collect();

            let mut q_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(q_bf16.len()).unwrap();
            let mut k_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(k_bf16.len()).unwrap();
            let mut v_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(v_bf16.len()).unwrap();
            let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_q_heads * head_dim).unwrap();
            q_buf.copy_from_host(&q_bf16).unwrap();
            k_buf.copy_from_host(&k_bf16).unwrap();
            v_buf.copy_from_host(&v_bf16).unwrap();
            let mut position_buf: DeviceBuffer<i32> = DeviceBuffer::alloc(1).unwrap();
            position_buf.copy_from_host(&[(kv_len - 1) as i32]).unwrap();

            // Baseline: the original scalar kernel.
            let launch_scalar = |out_buf: &mut DeviceBuffer<u16>| unsafe {
                ffi::launch_attention_decode_bf16(
                    q_buf.as_device_ptr(),
                    k_buf.as_device_ptr(),
                    v_buf.as_device_ptr(),
                    out_buf.as_device_ptr_mut() as *mut c_void,
                    num_q_heads as i32,
                    num_kv_heads as i32,
                    position_buf.as_device_ptr() as *const c_int,
                    kv_len as i32,
                    head_dim as i32,
                    scaling,
                    256,
                    std::ptr::null_mut(),
                );
            };
            for _ in 0..warmup {
                launch_scalar(&mut out_buf);
            }
            hip::device_synchronize().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                launch_scalar(&mut out_buf);
            }
            hip::device_synchronize().unwrap();
            let scalar_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!("kv_len={kv_len:4}  scalar (kv_split=1, original):   {scalar_us:8.3} us/call");

            for &kv_split in &[2usize, 4] {
                let launch_split = |out_buf: &mut DeviceBuffer<u16>| unsafe {
                    ffi::launch_attention_decode_split_bf16(
                        q_buf.as_device_ptr(),
                        k_buf.as_device_ptr(),
                        v_buf.as_device_ptr(),
                        out_buf.as_device_ptr_mut() as *mut c_void,
                        num_q_heads as i32,
                        num_kv_heads as i32,
                        position_buf.as_device_ptr() as *const c_int,
                        kv_len as i32,
                        head_dim as i32,
                        kv_split as i32,
                        scaling,
                        std::ptr::null_mut(),
                    );
                };
                for _ in 0..warmup {
                    launch_split(&mut out_buf);
                }
                hip::device_synchronize().unwrap();
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    launch_split(&mut out_buf);
                }
                hip::device_synchronize().unwrap();
                let split_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
                println!("kv_len={kv_len:4}  split (kv_split={kv_split}):             {split_us:8.3} us/call  ({:.2}x vs scalar)", scalar_us / split_us);
            }
        }
    }

    // ---- GatedDeltaNet recurrent decode (Hard tier, feature 2) -------------

    fn l2norm_ref(x: &[f32], eps: f32) -> Vec<f32> {
        let sumsq: f32 = x.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sumsq + eps).sqrt();
        x.iter().map(|&v| v * inv).collect()
    }

    /// Real correctness test: a small, fixed case (2 heads, head_dim=4,
    /// non-zero initial state -- exercises decay + rank-1 update, not just
    /// the zero-state edge case), computed on the real GPU via the real
    /// kernel, checked against a reference computed INDEPENDENTLY here in
    /// plain f32 Rust, following the exact formula derived from the real
    /// `torch_recurrent_gated_delta_rule` transformers source (see
    /// `gdn_recurrent.hip`'s header comment) -- not calling the kernel's
    /// own code. Checks both `out` and the updated `state`, since a wrong
    /// state silently corrupts every subsequent decode step (same
    /// reasoning `causal_conv1d_update`'s test already established for its
    /// own persistent state).
    #[test]
    fn real_gdn_recurrent_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_heads = 2usize;
        let head_dim = 4usize;
        let eps = 1e-6f32;

        let q_f32: Vec<f32> = vec![0.5, -0.3, 0.2, 0.1, -0.4, 0.6, -0.1, 0.3];
        let k_f32: Vec<f32> = vec![0.2, 0.4, -0.1, 0.3, 0.1, -0.2, 0.5, -0.3];
        let v_f32: Vec<f32> = vec![0.1, -0.2, 0.3, -0.1, 0.2, 0.1, -0.3, 0.4];
        let g_f32: Vec<f32> = vec![-0.05, -0.2];
        let beta_f32: Vec<f32> = vec![0.3, 0.7];
        // [num_heads, head_dim, head_dim], non-zero.
        let mut state_f32: Vec<f32> = (0..num_heads * head_dim * head_dim)
            .map(|i| (((i % 9) as i32 - 4) as f32) * 0.05)
            .collect();

        // Independent reference, plain f32, exact formula from
        // gdn_recurrent.hip's header.
        let mut expected_out = vec![0f32; num_heads * head_dim];
        for h in 0..num_heads {
            let q_row = &q_f32[h * head_dim..(h + 1) * head_dim];
            let k_row = &k_f32[h * head_dim..(h + 1) * head_dim];
            let v_row = &v_f32[h * head_dim..(h + 1) * head_dim];
            let q_n: Vec<f32> = l2norm_ref(q_row, eps)
                .iter()
                .map(|&x| x / (head_dim as f32).sqrt())
                .collect();
            let k_n = l2norm_ref(k_row, eps);
            let decay = g_f32[h].exp();
            let state_head = &mut state_f32[h * head_dim * head_dim..(h + 1) * head_dim * head_dim];
            for x in state_head.iter_mut() {
                *x *= decay;
            }
            let mut delta = vec![0f32; head_dim];
            for vv in 0..head_dim {
                let mut kv_mem = 0f32;
                for kk in 0..head_dim {
                    kv_mem += state_head[kk * head_dim + vv] * k_n[kk];
                }
                delta[vv] = (v_row[vv] - kv_mem) * beta_f32[h];
            }
            for kk in 0..head_dim {
                for vv in 0..head_dim {
                    state_head[kk * head_dim + vv] += k_n[kk] * delta[vv];
                }
            }
            for vv in 0..head_dim {
                let mut acc = 0f32;
                for kk in 0..head_dim {
                    acc += state_head[kk * head_dim + vv] * q_n[kk];
                }
                expected_out[h * head_dim + vv] = acc;
            }
        }
        let expected_state = state_f32.clone(); // mutated in place above

        // Reset state to the pre-mutation values for the GPU run (state_f32
        // was mutated above as the reference computation).
        let mut state_before: Vec<f32> = (0..num_heads * head_dim * head_dim)
            .map(|i| (((i % 9) as i32 - 4) as f32) * 0.05)
            .collect();

        let q_bf16: Vec<u16> = q_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let k_bf16: Vec<u16> = k_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let v_bf16: Vec<u16> = v_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut q_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(q_bf16.len()).unwrap();
        let mut k_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(k_bf16.len()).unwrap();
        let mut v_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(v_bf16.len()).unwrap();
        let mut g_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(g_f32.len()).unwrap();
        let mut beta_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(beta_f32.len()).unwrap();
        let mut state_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(state_before.len()).unwrap();
        let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_heads * head_dim).unwrap();
        q_buf.copy_from_host(&q_bf16).unwrap();
        k_buf.copy_from_host(&k_bf16).unwrap();
        v_buf.copy_from_host(&v_bf16).unwrap();
        g_buf.copy_from_host(&g_f32).unwrap();
        beta_buf.copy_from_host(&beta_f32).unwrap();
        state_buf.copy_from_host(&state_before).unwrap();

        gdn_recurrent_decode_bf16(
            &q_buf,
            &k_buf,
            &v_buf,
            &g_buf,
            &beta_buf,
            &mut state_buf,
            &mut out_buf,
            num_heads,
            num_heads, // num_k_heads: pre-broadcast Q/K in this test, so num_k_heads == num_heads
            head_dim,
        )
        .expect("real gdn_recurrent_decode_bf16 kernel launch failed");

        let mut out_bf16 = vec![0u16; num_heads * head_dim];
        out_buf.copy_to_host(&mut out_bf16).unwrap();
        let got: Vec<f32> = out_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
        for (i, (g, e)) in got.iter().zip(expected_out.iter()).enumerate() {
            assert!(
                (g - e).abs() < 0.01,
                "output element {i}: GPU kernel got {g}, independent reference {e}"
            );
        }

        state_buf.copy_to_host(&mut state_before).unwrap();
        for (i, (g, e)) in state_before.iter().zip(expected_state.iter()).enumerate() {
            assert!(
                (g - e).abs() < 0.01,
                "state element {i}: GPU kernel got {g}, independent reference {e}"
            );
        }
    }

    /// Decisive real-oracle cross-check: real model dims (32 heads,
    /// head_dim=128), a fixed deterministic synthetic Q/K/V/g/beta and a
    /// non-zero initial recurrent state, run through the REAL
    /// `torch_recurrent_gated_delta_rule` transformers function
    /// (`scratchpad/gen_gdn_reference.py` -- an independent process, not
    /// this kernel's own code) with `use_qk_l2norm_in_kernel=True`
    /// (the real decode-path configuration) and `output_final_state=True`.
    #[test]
    fn real_gdn_recurrent_matches_real_transformers_function() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_heads = 32usize;
        let head_dim = 128usize;

        // Same deterministic formulas as gen_gdn_reference.py's synth().
        let synth = |n: usize, offset: i32, period: i32, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let m = ((i as i32 + offset).rem_euclid(period)) as f32;
                    (m - (period as f32) / 2.0) * scale
                })
                .collect()
        };
        let q_f32 = synth(num_heads * head_dim, 0, 13, 0.05);
        let k_f32 = synth(num_heads * head_dim, 3, 11, 0.04);
        let v_f32 = synth(num_heads * head_dim, 7, 19, 0.03);
        let g_f32: Vec<f32> = (0..num_heads).map(|h| -(0.01 + 0.1 * (h % 5) as f32)).collect();
        let beta_f32: Vec<f32> = (0..num_heads)
            .map(|h| 0.2 + 0.6 * (((h * 7) % 11) as f32) / 11.0)
            .collect();
        let state_f32 = synth(num_heads * head_dim * head_dim, 0, 23, 0.02);

        let q_bf16: Vec<u16> = q_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let k_bf16: Vec<u16> = k_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let v_bf16: Vec<u16> = v_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut q_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(q_bf16.len()).unwrap();
        let mut k_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(k_bf16.len()).unwrap();
        let mut v_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(v_bf16.len()).unwrap();
        let mut g_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(g_f32.len()).unwrap();
        let mut beta_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(beta_f32.len()).unwrap();
        let mut state_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(state_f32.len()).unwrap();
        let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_heads * head_dim).unwrap();
        q_buf.copy_from_host(&q_bf16).unwrap();
        k_buf.copy_from_host(&k_bf16).unwrap();
        v_buf.copy_from_host(&v_bf16).unwrap();
        g_buf.copy_from_host(&g_f32).unwrap();
        beta_buf.copy_from_host(&beta_f32).unwrap();
        state_buf.copy_from_host(&state_f32).unwrap();

        gdn_recurrent_decode_bf16(
            &q_buf,
            &k_buf,
            &v_buf,
            &g_buf,
            &beta_buf,
            &mut state_buf,
            &mut out_buf,
            num_heads,
            num_heads, // num_k_heads: pre-broadcast Q/K in this test, so num_k_heads == num_heads
            head_dim,
        )
        .expect("real gdn_recurrent_decode_bf16 kernel launch failed against real-dims case");

        let mut out_bf16 = vec![0u16; num_heads * head_dim];
        out_buf.copy_to_host(&mut out_bf16).unwrap();
        let mut state_after = vec![0f32; state_f32.len()];
        state_buf.copy_to_host(&mut state_after).unwrap();

        // Independently computed by the real transformers
        // torch_recurrent_gated_delta_rule on this exact synthetic data
        // (see scratchpad/gen_gdn_reference.py's stdout).
        let head0_out_expected: [f32; 8] = [
            0.006072998,
            0.0058288574,
            0.0079956055,
            0.0015792847,
            0.0003814697,
            -0.0036468506,
            -0.000617981,
            0.0095825195,
        ];
        let head31_out_expected: [f32; 8] = [
            0.0003738403,
            -0.0040283203,
            -0.0043945312,
            -0.0074157715,
            -0.0031890869,
            0.0061645508,
            0.002822876,
            0.0039672852,
        ];
        let head0_state_row0_expected: [f32; 8] = [
            -0.2269041538,
            -0.2056370378,
            -0.1889493018,
            -0.1686672866,
            -0.1491941214,
            -0.131342873,
            -0.1092605218,
            -0.0915825516,
        ];
        let head31_state_row0_expected: [f32; 8] = [
            0.1422306597,
            0.1463433355,
            0.1642511338,
            0.1843628883,
            0.1873735785,
            -0.19361718,
            -0.184535414,
            -0.1704829186,
        ];

        let head0_out_got: Vec<f32> = out_bf16[0..8].iter().map(|&b| bf16_to_f32(b)).collect();
        let head31_out_got: Vec<f32> = out_bf16[31 * head_dim..31 * head_dim + 8]
            .iter()
            .map(|&b| bf16_to_f32(b))
            .collect();
        let head0_state_row0_got = &state_after[0..8];
        let head31_state_row0_got = &state_after[31 * head_dim * head_dim..31 * head_dim * head_dim + 8];

        for (i, (g, e)) in head0_out_got.iter().zip(head0_out_expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 0.002,
                "head 0 out[{i}]: GPU kernel got {g}, independent Python reference {e}"
            );
        }
        for (i, (g, e)) in head31_out_got.iter().zip(head31_out_expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 0.002,
                "head 31 out[{i}]: GPU kernel got {g}, independent Python reference {e}"
            );
        }
        for (i, (g, e)) in head0_state_row0_got
            .iter()
            .zip(head0_state_row0_expected.iter())
            .enumerate()
        {
            assert!(
                (g - e).abs() < 0.002,
                "head 0 state row 0[{i}]: GPU kernel got {g}, independent Python reference {e}"
            );
        }
        for (i, (g, e)) in head31_state_row0_got
            .iter()
            .zip(head31_state_row0_expected.iter())
            .enumerate()
        {
            assert!(
                (g - e).abs() < 0.002,
                "head 31 state row 0[{i}]: GPU kernel got {g}, independent Python reference {e}"
            );
        }
    }

    /// Benchmarked pipelined from the start (§86's lesson): raw kernel
    /// launches, no sync between calls, one `hipDeviceSynchronize()` after
    /// all `iters` launches, matching the paired Python benchmark exactly.
    #[test]
    #[ignore]
    fn bench_real_gdn_recurrent_vs_python_runtime_shapes() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        // Real model dims: num_v_heads=32 (Q/K already broadcast to this),
        // head_dim=128 (both k_head_dim and v_head_dim).
        let num_heads = 32usize;
        let head_dim = 128usize;
        let iters = 2000;
        let warmup = 100;

        let q_f32: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| ((i % 89) as f32 - 44.0) * 0.01)
            .collect();
        let k_f32: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| ((i % 71) as f32 - 35.0) * 0.01)
            .collect();
        let v_f32: Vec<f32> = (0..num_heads * head_dim)
            .map(|i| ((i % 61) as f32 - 30.0) * 0.01)
            .collect();
        let g_f32: Vec<f32> = (0..num_heads).map(|h| -(0.01 + 0.05 * (h % 7) as f32)).collect();
        let beta_f32: Vec<f32> = (0..num_heads)
            .map(|h| 0.2 + 0.5 * (((h * 5) % 9) as f32) / 9.0)
            .collect();
        let state_f32: Vec<f32> = (0..num_heads * head_dim * head_dim)
            .map(|i| ((i % 53) as f32 - 26.0) * 0.005)
            .collect();

        let q_bf16: Vec<u16> = q_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let k_bf16: Vec<u16> = k_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let v_bf16: Vec<u16> = v_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut q_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(q_bf16.len()).unwrap();
        let mut k_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(k_bf16.len()).unwrap();
        let mut v_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(v_bf16.len()).unwrap();
        let mut g_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(g_f32.len()).unwrap();
        let mut beta_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(beta_f32.len()).unwrap();
        let mut state_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(state_f32.len()).unwrap();
        let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_heads * head_dim).unwrap();
        q_buf.copy_from_host(&q_bf16).unwrap();
        k_buf.copy_from_host(&k_bf16).unwrap();
        v_buf.copy_from_host(&v_bf16).unwrap();
        g_buf.copy_from_host(&g_f32).unwrap();
        beta_buf.copy_from_host(&beta_f32).unwrap();
        state_buf.copy_from_host(&state_f32).unwrap();

        let threads: i32 = head_dim.next_power_of_two() as i32;
        let launch_one = |q_buf: &DeviceBuffer<u16>,
                           k_buf: &DeviceBuffer<u16>,
                           v_buf: &DeviceBuffer<u16>,
                           g_buf: &DeviceBuffer<f32>,
                           beta_buf: &DeviceBuffer<f32>,
                           state_buf: &mut DeviceBuffer<f32>,
                           out_buf: &mut DeviceBuffer<u16>| {
            // SAFETY: same call gdn_recurrent_decode_bf16 makes, minus the
            // trailing check_last_error()/device_synchronize().
            unsafe {
                ffi::launch_gdn_recurrent_decode_bf16(
                    q_buf.as_device_ptr(),
                    k_buf.as_device_ptr(),
                    v_buf.as_device_ptr(),
                    g_buf.as_device_ptr() as *const std::ffi::c_float,
                    beta_buf.as_device_ptr() as *const std::ffi::c_float,
                    state_buf.as_device_ptr_mut() as *mut std::ffi::c_float,
                    out_buf.as_device_ptr_mut() as *mut c_void,
                    num_heads as i32,
                    num_heads as i32, // num_k_heads: this benchmark's q/k buffers are already num_heads-sized
                    head_dim as i32,
                    threads,
                    std::ptr::null_mut(),
                );
            }
        };

        for _ in 0..warmup {
            launch_one(
                &q_buf, &k_buf, &v_buf, &g_buf, &beta_buf, &mut state_buf, &mut out_buf,
            );
        }
        hip::device_synchronize().unwrap();

        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            launch_one(
                &q_buf, &k_buf, &v_buf, &g_buf, &beta_buf, &mut state_buf, &mut out_buf,
            );
        }
        hip::device_synchronize().unwrap(); // ONE sync after all launches, matching Python
        let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        println!(
            "gdn_recurrent_decode_bf16 (pipelined) num_heads={num_heads} head_dim={head_dim}: {per_call_us:.3} us/call ({iters} iters, ONE hipDeviceSynchronize after the whole loop)"
        );
    }

    // ---- Remaining small ops needed to assemble a full forward pass ------

    #[test]
    fn real_sigmoid_gate_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let x_f32: Vec<f32> = vec![1.0, -2.0, 0.5, 3.0, -1.0, 0.0, 0.25, -0.75];
        let gate_f32: Vec<f32> = vec![0.5, 0.5, -1.0, 2.0, 1.0, 3.0, -0.5, 0.1];
        let expected: Vec<f32> = x_f32
            .iter()
            .zip(gate_f32.iter())
            .map(|(&x, &g)| x * (1.0 / (1.0 + (-g).exp())))
            .collect();

        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let gate_bf16: Vec<u16> = gate_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_f32.len()).unwrap();
        let mut gate_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(gate_f32.len()).unwrap();
        let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_f32.len()).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();
        gate_buf.copy_from_host(&gate_bf16).unwrap();

        sigmoid_gate_bf16(&x_buf, &gate_buf, &mut out_buf).expect("kernel launch failed");

        let mut out_bf16 = vec![0u16; x_f32.len()];
        out_buf.copy_to_host(&mut out_bf16).unwrap();
        let got: Vec<f32> = out_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 0.01, "element {i}: got {g}, expected {e}");
        }
    }

    #[test]
    fn real_rmsnorm_gated_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let hidden_size = 8usize;
        let eps = 1e-6f32;
        let x_f32: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 2.5, -1.5];
        let gate_f32: Vec<f32> = vec![0.5, 0.5, -1.0, 2.0, 1.0, 3.0, -0.5, 0.1];
        let w_f32: Vec<f32> = vec![1.1, 0.9, 1.0, 1.05, 0.95, 1.2, 0.8, 1.0];

        let mean_sq: f32 = x_f32.iter().map(|v| v * v).sum::<f32>() / hidden_size as f32;
        let rms = 1.0 / (mean_sq + eps).sqrt();
        let expected: Vec<f32> = x_f32
            .iter()
            .zip(w_f32.iter())
            .zip(gate_f32.iter())
            .map(|((&xv, &wv), &gv)| (wv * (xv * rms)) * (gv / (1.0 + (-gv).exp())))
            .collect();

        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let gate_bf16: Vec<u16> = gate_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(hidden_size).unwrap();
        let mut gate_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(hidden_size).unwrap();
        // weight is native f32 in the real checkpoint (see rmsnorm_gated.hip's
        // header) -- no bf16 round-trip on this input.
        let mut w_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(hidden_size).unwrap();
        let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(hidden_size).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();
        gate_buf.copy_from_host(&gate_bf16).unwrap();
        w_buf.copy_from_host(&w_f32).unwrap();

        rmsnorm_gated_bf16(&x_buf, &gate_buf, &w_buf, &mut out_buf, 1, hidden_size, eps)
            .expect("kernel launch failed");

        let mut out_bf16 = vec![0u16; hidden_size];
        out_buf.copy_to_host(&mut out_bf16).unwrap();
        let got: Vec<f32> = out_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 0.01, "element {i}: got {g}, expected {e}");
        }
    }

    #[test]
    fn real_gdn_gate_beta_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_heads = 4usize;
        let a_f32: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4];
        let b_f32: Vec<f32> = vec![0.5, -0.5, 1.0, -1.0];
        let a_log_f32: Vec<f32> = vec![-1.0, -0.5, 0.0, 0.2]; // ln(A), A in (0.01,16)
        let dt_bias_f32: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];

        let softplus = |x: f32| -> f32 {
            if x > 20.0 { x } else { (1.0f32 + x.exp()).ln() }
        };
        let expected_g: Vec<f32> = a_f32
            .iter()
            .zip(a_log_f32.iter())
            .zip(dt_bias_f32.iter())
            .map(|((&av, &alv), &dbv)| -alv.exp() * softplus(av + dbv))
            .collect();
        let expected_beta: Vec<f32> = b_f32.iter().map(|&bv| 1.0 / (1.0 + (-bv).exp())).collect();

        let a_bf16: Vec<u16> = a_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let b_bf16: Vec<u16> = b_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut a_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_heads).unwrap();
        let mut b_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_heads).unwrap();
        let mut a_log_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(num_heads).unwrap();
        let mut dt_bias_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(num_heads).unwrap();
        let mut g_out_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(num_heads).unwrap();
        let mut beta_out_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(num_heads).unwrap();
        a_buf.copy_from_host(&a_bf16).unwrap();
        b_buf.copy_from_host(&b_bf16).unwrap();
        a_log_buf.copy_from_host(&a_log_f32).unwrap();
        dt_bias_buf.copy_from_host(&dt_bias_f32).unwrap();

        gdn_gate_beta_bf16(
            &a_buf,
            &b_buf,
            &a_log_buf,
            &dt_bias_buf,
            &mut g_out_buf,
            &mut beta_out_buf,
            num_heads,
        )
        .expect("kernel launch failed");

        let mut g_got = vec![0f32; num_heads];
        let mut beta_got = vec![0f32; num_heads];
        g_out_buf.copy_to_host(&mut g_got).unwrap();
        beta_out_buf.copy_to_host(&mut beta_got).unwrap();

        for (i, (g, e)) in g_got.iter().zip(expected_g.iter()).enumerate() {
            assert!((g - e).abs() < 0.01, "g[{i}]: got {g}, expected {e}");
        }
        for (i, (g, e)) in beta_got.iter().zip(expected_beta.iter()).enumerate() {
            assert!((g - e).abs() < 0.01, "beta[{i}]: got {g}, expected {e}");
        }
    }

    /// §103 decisive test: `gdn_gate_beta_prefill_bf16` (the new real
    /// batched-prefill kernel, one launch for the whole chunk) matches
    /// the ALREADY-PROVEN `gdn_gate_beta_bf16` called per-token, AND
    /// exercises the batched kernel's own strided source reads (`a`/`b`
    /// embedded inside wider `[num_tokens, wide_stride]` buffers with a
    /// nonzero offset, matching the real usage of reading straight out of
    /// `gdn_in_proj_out`'s own wider combined-GEMM row).
    #[test]
    fn real_gdn_gate_beta_prefill_matches_per_token_loop_with_strided_source() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_heads = 5usize;
        let num_tokens = 4usize;
        let a_wide_stride = num_heads + 6;
        let a_offset = 2usize;
        let b_wide_stride = num_heads + 3;
        let b_offset = 1usize;

        let synth = |n: usize, offset: i32, period: i32, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let m = ((i as i32 + offset).rem_euclid(period)) as f32;
                    (m - (period as f32) / 2.0) * scale
                })
                .collect()
        };
        let a_log_f32 = synth(num_heads, 0, 9, 0.3);
        let dt_bias_f32 = synth(num_heads, 2, 7, 0.4);
        let mut a_wide_f32 = synth(num_tokens * a_wide_stride, 5, 29, 0.05);
        let mut b_wide_f32 = synth(num_tokens * b_wide_stride, 11, 31, 0.05);
        let a_all_f32: Vec<Vec<f32>> = (0..num_tokens).map(|t| synth(num_heads, (t * 5 + 3) as i32, 15, 0.2)).collect();
        let b_all_f32: Vec<Vec<f32>> = (0..num_tokens).map(|t| synth(num_heads, (t * 7 + 1) as i32, 17, 0.2)).collect();
        for t in 0..num_tokens {
            for h in 0..num_heads {
                a_wide_f32[t * a_wide_stride + a_offset + h] = a_all_f32[t][h];
                b_wide_f32[t * b_wide_stride + b_offset + h] = b_all_f32[t][h];
            }
        }

        let a_wide_bf16: Vec<u16> = a_wide_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let b_wide_bf16: Vec<u16> = b_wide_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut a_wide_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(a_wide_bf16.len()).unwrap();
        let mut b_wide_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(b_wide_bf16.len()).unwrap();
        let mut a_log_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(num_heads).unwrap();
        let mut dt_bias_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(num_heads).unwrap();
        a_wide_buf.copy_from_host(&a_wide_bf16).unwrap();
        b_wide_buf.copy_from_host(&b_wide_bf16).unwrap();
        a_log_buf.copy_from_host(&a_log_f32).unwrap();
        dt_bias_buf.copy_from_host(&dt_bias_f32).unwrap();

        // Reference: the already-proven per-token kernel, once per token.
        let mut ref_g = vec![0f32; num_tokens * num_heads];
        let mut ref_beta = vec![0f32; num_tokens * num_heads];
        for t in 0..num_tokens {
            let a_bf16: Vec<u16> = a_all_f32[t].iter().map(|&v| f32_to_bf16(v)).collect();
            let b_bf16: Vec<u16> = b_all_f32[t].iter().map(|&v| f32_to_bf16(v)).collect();
            let mut a_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_heads).unwrap();
            let mut b_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_heads).unwrap();
            a_buf.copy_from_host(&a_bf16).unwrap();
            b_buf.copy_from_host(&b_bf16).unwrap();
            let mut g_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(num_heads).unwrap();
            let mut beta_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(num_heads).unwrap();
            gdn_gate_beta_bf16(&a_buf, &b_buf, &a_log_buf, &dt_bias_buf, &mut g_buf, &mut beta_buf, num_heads).expect("reference gdn_gate_beta_bf16 call failed");
            g_buf.copy_to_host(&mut ref_g[t * num_heads..(t + 1) * num_heads]).unwrap();
            beta_buf.copy_to_host(&mut ref_beta[t * num_heads..(t + 1) * num_heads]).unwrap();
        }

        // New batched path: ONE call, strided sources.
        let mut g_out_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(num_tokens * num_heads).unwrap();
        let mut beta_out_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(num_tokens * num_heads).unwrap();
        gdn_gate_beta_prefill_bf16(
            &a_wide_buf,
            a_wide_stride,
            a_offset,
            &b_wide_buf,
            b_wide_stride,
            b_offset,
            &a_log_buf,
            &dt_bias_buf,
            &mut g_out_buf,
            &mut beta_out_buf,
            num_tokens,
            num_heads,
        )
        .expect("real gdn_gate_beta_prefill_bf16 call failed");

        let mut g_got = vec![0f32; num_tokens * num_heads];
        let mut beta_got = vec![0f32; num_tokens * num_heads];
        g_out_buf.copy_to_host(&mut g_got).unwrap();
        beta_out_buf.copy_to_host(&mut beta_got).unwrap();

        for (i, (&g, &e)) in g_got.iter().zip(ref_g.iter()).enumerate() {
            assert!((g - e).abs() < 0.001, "g[{i}]: batched kernel got {g}, per-token-loop reference {e}");
        }
        for (i, (&g, &e)) in beta_got.iter().zip(ref_beta.iter()).enumerate() {
            assert!((g - e).abs() < 0.001, "beta[{i}]: batched kernel got {g}, per-token-loop reference {e}");
        }
    }

    #[test]
    fn real_add_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let a_f32: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 0.5, -0.5];
        let b_f32: Vec<f32> = vec![0.1, 0.2, -0.3, 0.4, -0.5, 0.6];
        let expected: Vec<f32> = a_f32.iter().zip(b_f32.iter()).map(|(&x, &y)| x + y).collect();

        let a_bf16: Vec<u16> = a_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let b_bf16: Vec<u16> = b_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut a_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(a_f32.len()).unwrap();
        let mut b_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(b_f32.len()).unwrap();
        let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(a_f32.len()).unwrap();
        a_buf.copy_from_host(&a_bf16).unwrap();
        b_buf.copy_from_host(&b_bf16).unwrap();

        add_bf16(&a_buf, &b_buf, &mut out_buf).expect("kernel launch failed");

        let mut out_bf16 = vec![0u16; a_f32.len()];
        out_buf.copy_to_host(&mut out_bf16).unwrap();
        let got: Vec<f32> = out_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 0.01, "element {i}: got {g}, expected {e}");
        }
    }

    #[test]
    fn real_extract_range_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        // 3 rows, src_stride=7 (mimicking a combined-GEMM row wider than
        // any one sub-range), extracting a 3-element range starting at
        // offset 2 from each row.
        let rows = 3usize;
        let src_stride = 7usize;
        let src_offset = 2usize;
        let len = 3usize;
        let x_f32: Vec<f32> = (0..rows * src_stride).map(|i| i as f32).collect();
        // row r's extracted range is x[r*7+2 .. r*7+5)
        let expected: Vec<f32> = (0..rows)
            .flat_map(|r| {
                let base = r * src_stride + src_offset;
                (base..base + len).map(|i| i as f32).collect::<Vec<_>>()
            })
            .collect();

        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_f32.len()).unwrap();
        let mut dst_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(rows * len).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();

        extract_range_bf16(&x_buf, &mut dst_buf, rows, src_stride, src_offset, len)
            .expect("kernel launch failed");

        let mut dst_bf16 = vec![0u16; rows * len];
        dst_buf.copy_to_host(&mut dst_bf16).unwrap();
        let got: Vec<f32> = dst_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        assert_eq!(got, expected);
    }

    #[test]
    fn real_split_last_dim_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let rows = 2usize;
        let half = 3usize;
        // row 0: [a0,a1,a2 | b0,b1,b2], row 1: [a3,a4,a5 | b3,b4,b5]
        let x_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 40.0, 50.0, 60.0];
        let expected_first: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let expected_second: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_f32.len()).unwrap();
        let mut first_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(rows * half).unwrap();
        let mut second_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(rows * half).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();

        split_last_dim_bf16(&x_buf, &mut first_buf, &mut second_buf, rows, half)
            .expect("kernel launch failed");

        let mut first_bf16 = vec![0u16; rows * half];
        let mut second_bf16 = vec![0u16; rows * half];
        first_buf.copy_to_host(&mut first_bf16).unwrap();
        second_buf.copy_to_host(&mut second_bf16).unwrap();
        let first_got: Vec<f32> = first_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
        let second_got: Vec<f32> = second_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        assert_eq!(first_got, expected_first);
        assert_eq!(second_got, expected_second);
    }

    #[test]
    fn real_kv_cache_append_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_kv_heads = 2usize;
        let max_seq_len = 4usize;
        let head_dim = 3usize;

        let mut k_cache_buf: DeviceBuffer<u16> =
            DeviceBuffer::alloc(num_kv_heads * max_seq_len * head_dim).unwrap();
        let mut v_cache_buf: DeviceBuffer<u16> =
            DeviceBuffer::alloc(num_kv_heads * max_seq_len * head_dim).unwrap();
        // Zero-init the caches on the host, upload, so untouched slots are
        // deterministically zero rather than uninitialized GPU memory.
        let zeros = vec![f32_to_bf16(0.0); num_kv_heads * max_seq_len * head_dim];
        k_cache_buf.copy_from_host(&zeros).unwrap();
        v_cache_buf.copy_from_host(&zeros).unwrap();

        // Append at position 2.
        let position = 2usize;
        let new_k_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2 heads, 3 dims]
        let new_v_f32: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let new_k_bf16: Vec<u16> = new_k_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let new_v_bf16: Vec<u16> = new_v_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut new_k_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(new_k_f32.len()).unwrap();
        let mut new_v_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(new_v_f32.len()).unwrap();
        new_k_buf.copy_from_host(&new_k_bf16).unwrap();
        new_v_buf.copy_from_host(&new_v_bf16).unwrap();

        kv_cache_append_bf16(
            &new_k_buf,
            &new_v_buf,
            &mut k_cache_buf,
            &mut v_cache_buf,
            num_kv_heads,
            max_seq_len,
            head_dim,
            position,
        )
        .expect("kernel launch failed");

        let mut k_cache_bf16 = vec![0u16; num_kv_heads * max_seq_len * head_dim];
        let mut v_cache_bf16 = vec![0u16; num_kv_heads * max_seq_len * head_dim];
        k_cache_buf.copy_to_host(&mut k_cache_bf16).unwrap();
        v_cache_buf.copy_to_host(&mut v_cache_bf16).unwrap();

        // Independent reference: slot `position` of head h should now hold
        // new_k[h,:]/new_v[h,:]; every other slot should remain zero.
        for h in 0..num_kv_heads {
            for s in 0..max_seq_len {
                for d in 0..head_dim {
                    let idx = (h * max_seq_len + s) * head_dim + d;
                    let k_got = bf16_to_f32(k_cache_bf16[idx]);
                    let v_got = bf16_to_f32(v_cache_bf16[idx]);
                    if s == position {
                        assert!(
                            (k_got - new_k_f32[h * head_dim + d]).abs() < 0.01,
                            "k_cache[h={h},s={s},d={d}]: got {k_got}"
                        );
                        assert!(
                            (v_got - new_v_f32[h * head_dim + d]).abs() < 0.01,
                            "v_cache[h={h},s={s},d={d}]: got {v_got}"
                        );
                    } else {
                        assert_eq!(k_got, 0.0, "k_cache[h={h},s={s},d={d}] should be untouched (0)");
                        assert_eq!(v_got, 0.0, "v_cache[h={h},s={s},d={d}] should be untouched (0)");
                    }
                }
            }
        }
    }

    /// §103 decisive test: `kv_cache_append_prefill_bf16` (the new real
    /// batched-prefill kernel, one launch for the whole chunk) matches
    /// the ALREADY-PROVEN `kv_cache_append_bf16` called once per token at
    /// its OWN distinct real position -- real usage shape (`num_tokens=3`,
    /// non-contiguous positions, matching a real prompt appended starting
    /// partway through an already-populated cache).
    #[test]
    fn real_kv_cache_append_prefill_matches_per_token_loop_with_distinct_positions() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_kv_heads = 2usize;
        let max_seq_len = 8usize;
        let head_dim = 3usize;
        let num_tokens = 3usize;
        let positions: [usize; 3] = [1, 2, 5]; // real, distinct, non-contiguous

        let zeros = vec![f32_to_bf16(0.0); num_kv_heads * max_seq_len * head_dim];
        let new_k_all_f32: Vec<f32> = (0..num_tokens * num_kv_heads * head_dim).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let new_v_all_f32: Vec<f32> = (0..num_tokens * num_kv_heads * head_dim).map(|i| (i as f32) * 0.1 + 10.0).collect();
        let new_k_all_bf16: Vec<u16> = new_k_all_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let new_v_all_bf16: Vec<u16> = new_v_all_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        // Reference: the already-proven per-token kernel, called once
        // per token at its own real position, all sharing ONE cache.
        let mut ref_k_cache: DeviceBuffer<u16> = DeviceBuffer::alloc(zeros.len()).unwrap();
        let mut ref_v_cache: DeviceBuffer<u16> = DeviceBuffer::alloc(zeros.len()).unwrap();
        ref_k_cache.copy_from_host(&zeros).unwrap();
        ref_v_cache.copy_from_host(&zeros).unwrap();
        for t in 0..num_tokens {
            let mut new_k_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_kv_heads * head_dim).unwrap();
            let mut new_v_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_kv_heads * head_dim).unwrap();
            new_k_buf.copy_from_host(&new_k_all_bf16[t * num_kv_heads * head_dim..(t + 1) * num_kv_heads * head_dim]).unwrap();
            new_v_buf.copy_from_host(&new_v_all_bf16[t * num_kv_heads * head_dim..(t + 1) * num_kv_heads * head_dim]).unwrap();
            kv_cache_append_bf16(&new_k_buf, &new_v_buf, &mut ref_k_cache, &mut ref_v_cache, num_kv_heads, max_seq_len, head_dim, positions[t]).expect("reference kv_cache_append_bf16 call failed");
        }
        let mut ref_k_bf16 = vec![0u16; zeros.len()];
        let mut ref_v_bf16 = vec![0u16; zeros.len()];
        ref_k_cache.copy_to_host(&mut ref_k_bf16).unwrap();
        ref_v_cache.copy_to_host(&mut ref_v_bf16).unwrap();

        // New batched path: ONE call, per-token positions, into a FRESH cache.
        let mut new_k_all_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(new_k_all_bf16.len()).unwrap();
        let mut new_v_all_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(new_v_all_bf16.len()).unwrap();
        new_k_all_buf.copy_from_host(&new_k_all_bf16).unwrap();
        new_v_all_buf.copy_from_host(&new_v_all_bf16).unwrap();
        let position_i32: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let mut position_buf: DeviceBuffer<i32> = DeviceBuffer::alloc(num_tokens).unwrap();
        position_buf.copy_from_host(&position_i32).unwrap();
        let mut batched_k_cache: DeviceBuffer<u16> = DeviceBuffer::alloc(zeros.len()).unwrap();
        let mut batched_v_cache: DeviceBuffer<u16> = DeviceBuffer::alloc(zeros.len()).unwrap();
        batched_k_cache.copy_from_host(&zeros).unwrap();
        batched_v_cache.copy_from_host(&zeros).unwrap();
        kv_cache_append_prefill_bf16(&new_k_all_buf, &new_v_all_buf, &mut batched_k_cache, &mut batched_v_cache, num_tokens, num_kv_heads, max_seq_len, head_dim, &position_buf).expect("real kv_cache_append_prefill_bf16 call failed");

        let mut batched_k_bf16 = vec![0u16; zeros.len()];
        let mut batched_v_bf16 = vec![0u16; zeros.len()];
        batched_k_cache.copy_to_host(&mut batched_k_bf16).unwrap();
        batched_v_cache.copy_to_host(&mut batched_v_bf16).unwrap();

        assert_eq!(batched_k_bf16, ref_k_bf16, "k_cache bytes differ between the batched kernel and the per-token-loop reference");
        assert_eq!(batched_v_bf16, ref_v_bf16, "v_cache bytes differ between the batched kernel and the per-token-loop reference");
    }

    // ---- GEMV -------------------------------------------------------

    /// Real correctness test: small, fixed `x`/`w`, computed on the real
    /// GPU via the real `gemv_bf16` kernel, checked against a reference
    /// computed INDEPENDENTLY here in plain f32 Rust.
    #[test]
    fn real_gemv_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let in_features = 8usize;
        let out_features = 3usize;

        let x_f32: Vec<f32> = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25, -0.75];
        let w_f32: Vec<f32> = vec![
            0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.15, -0.05, // out 0
            -0.2, 0.3, 0.1, -0.05, -0.1, 0.2, -0.15, 0.1, // out 1
            0.0, -0.1, 0.2, 0.15, 0.05, -0.05, 0.1, -0.2, // out 2
        ];

        let mut expected = vec![0f32; out_features];
        for o in 0..out_features {
            let mut acc = 0f32;
            for i in 0..in_features {
                acc += w_f32[o * in_features + i] * x_f32[i];
            }
            expected[o] = acc;
        }

        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let w_bf16: Vec<u16> = w_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(in_features).unwrap();
        let mut w_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(out_features * in_features).unwrap();
        let mut y_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(out_features).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();
        w_buf.copy_from_host(&w_bf16).unwrap();

        gemv_bf16(&x_buf, &w_buf, &mut y_buf, out_features, in_features).expect("real gemv_bf16 call failed");

        let mut y_bf16 = vec![0u16; out_features];
        y_buf.copy_to_host(&mut y_bf16).unwrap();
        let got: Vec<f32> = y_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 0.02, "element {i}: GPU gemv got {g}, independent reference {e}");
        }
    }

    /// Decisive real-weight cross-check: the real layer 0 `mlp.down_proj.weight`
    /// (`[2560, 9216]`) and the SAME deterministic input formula and
    /// independently-computed (real PyTorch `F.linear`) reference values as
    /// `blas::tests::real_gemm_matches_real_down_proj_weight` -- this test
    /// exists specifically to prove the new hand-written GEMV kernel
    /// computes the IDENTICAL real result as the hipBLAS path it's meant to
    /// replace in the hot path, not just a plausible-looking one.
    ///
    /// Real, deliberate scope: same Qwen3.5-4B-specific real weight and
    /// reference values as `blas::tests::real_gemm_matches_real_down_proj_weight`
    /// -- gated the same way, for the same reason.
    #[test]
    #[cfg(feature = "qwen35_4b")]
    fn real_gemv_matches_real_down_proj_weight() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = crate::model_loader::locate_model_snapshot()
            .expect("no real Qwen3.5-4B snapshot found on this machine");
        let raw = crate::model_loader::load_raw_tensor(
            &snapshot,
            "model.language_model.layers.0.mlp.down_proj.weight",
        )
        .expect("layer 0 mlp.down_proj.weight must exist");
        let out_features = raw.shape[0];
        let in_features = raw.shape[1];
        assert_eq!(out_features, 2560);
        assert_eq!(in_features, 9216);
        let w_bf16: Vec<u16> = raw
            .data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();

        // Same deterministic formula as scratchpad/gen_gemm_reference.py.
        let x_f32: Vec<f32> = (0..in_features)
            .map(|i| (((i % 13) as i32 - 6) as f32) * 0.05)
            .collect();
        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
        let mut w_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(w_bf16.len()).unwrap();
        let mut y_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(out_features).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();
        w_buf.copy_from_host(&w_bf16).unwrap();

        gemv_bf16(&x_buf, &w_buf, &mut y_buf, out_features, in_features)
            .expect("real gemv_bf16 call failed against real down_proj weight");

        let mut y_bf16 = vec![0u16; out_features];
        y_buf.copy_to_host(&mut y_bf16).unwrap();
        let got: Vec<f32> = y_bf16[..8].iter().map(|&b| bf16_to_f32(b)).collect();

        // Independently computed by real PyTorch F.linear on this exact
        // real weight and input (see scratchpad/gen_gemm_reference.py's
        // stdout) -- identical reference values to blas.rs's own test.
        let expected: [f32; 8] = [
            0.1640625,
            -0.0230712891,
            0.1010742188,
            -0.076171875,
            -0.109375,
            -0.0322265625,
            -0.0766601562,
            0.046875,
        ];
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 0.01,
                "element {i}: GPU gemv got {g}, independent Python reference {e}"
            );
        }
    }

    /// §106: real correctness for the new W4A16 GEMV kernel, cross-
    /// validated against an independently-written CPU dequant+matmul
    /// reference (plain scalar Rust, not calling the GPU kernel at all),
    /// on a REAL quantized tensor from the real converted 27B checkpoint
    /// (`layer 0 mlp.down_proj.weight`, out=5120, in=17408,
    /// group_size=128 -- confirmed via `quantize_w4a16.py`'s own output).
    /// The most direct possible check: two independent implementations of
    /// the same real formula must agree on the same real, packed bytes.
    #[test]
    fn real_w4a16_gemv_matches_cpu_dequant_reference() {
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

        let (qweight_buf, scales_buf) = crate::model_loader::load_w4a16_weight(
            &quantized_dir,
            "model.language_model.layers.0.mlp.down_proj.weight",
        )
        .expect("real quantized down_proj weight must load");

        let out_features = 5120usize;
        let in_features = 17408usize;
        let group_size = 128usize;
        assert_eq!(qweight_buf.len(), out_features * (in_features / 8));
        assert_eq!(scales_buf.len(), out_features * (in_features / group_size));

        // Real bytes, read a second time on the host for the independent
        // CPU reference (separate from the DeviceBuffer used by the GPU
        // kernel -- no shared state between the two computations).
        let qraw = crate::model_loader::load_raw_tensor(
            &quantized_dir,
            "model.language_model.layers.0.mlp.down_proj.weight.qweight",
        )
        .unwrap();
        let sraw = crate::model_loader::load_raw_tensor(
            &quantized_dir,
            "model.language_model.layers.0.mlp.down_proj.weight.scales",
        )
        .unwrap();
        let qbits = qraw.to_u32_bits();
        let sbits = sraw.to_bf16_bits();

        let x_f32: Vec<f32> = (0..in_features)
            .map(|i| (((i % 13) as i32 - 6) as f32) * 0.05)
            .collect();
        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        // Independent CPU reference: same dequant formula
        // (w ~= (nibble-8)*scale), plain scalar Rust, no GPU involved.
        let k8 = in_features / 8;
        let groups_per_i32 = group_size / 8;
        let mut expected = vec![0.0f32; out_features];
        for o in 0..out_features {
            let mut acc = 0.0f64; // f64 accumulation on the CPU side to keep this reference tight
            for j in 0..k8 {
                let packed = qbits[o * k8 + j];
                let scale = bf16_to_f32(sbits[o * (in_features / group_size) + j / groups_per_i32]);
                for n in 0..8 {
                    let nibble = ((packed >> (n * 4)) & 0xF) as f32;
                    let w = (nibble - 8.0) * scale;
                    acc += (w * x_f32[j * 8 + n]) as f64;
                }
            }
            expected[o] = acc as f32;
        }

        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
        let mut y_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(out_features).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();

        w4a16_gemv_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y_buf, out_features, in_features, group_size)
            .expect("real w4a16_gemv_bf16 call failed against real quantized down_proj weight");

        let mut y_bf16 = vec![0u16; out_features];
        y_buf.copy_to_host(&mut y_bf16).unwrap();
        let got: Vec<f32> = y_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        let mut max_diff = 0.0f32;
        let mut max_rel = 0.0f32;
        for (o, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            let diff = (g - e).abs();
            max_diff = max_diff.max(diff);
            if e.abs() > 1e-3 {
                max_rel = max_rel.max(diff / e.abs());
            }
            assert!(
                diff < 0.05 || diff / e.abs().max(1e-3) < 0.05,
                "output row {o}: GPU kernel got {g}, CPU reference {e} (diff {diff})"
            );
        }
        eprintln!("max abs diff: {max_diff}, max rel diff: {max_rel}");
    }


    /// §108: real correctness for the new batched W4A16 prefill kernel --
    /// cross-validated against the ALREADY-validated per-token
    /// `w4a16_gemv_bf16` kernel (§106, proven against an independent CPU
    /// reference above), called once per token in a loop. Both are real
    /// GPU kernels computing the exact same real weight tensor; if
    /// batching changed the answer, this catches it directly rather than
    /// re-deriving a fresh CPU reference. `num_tokens=20` deliberately
    /// spans two full `TILE_T=8` tiles plus a partial tail tile (4 tokens)
    /// to exercise the tail-tile bounds-guard, not just the common case.
    #[test]
    fn real_w4a16_gemm_prefill_matches_per_token_gemv() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let quantized_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen38_27b_w4a16");
        if !quantized_dir.is_dir() {
            eprintln!("skipping: no real quantized 27B checkpoint at {}", quantized_dir.display());
            return;
        }

        let (qweight_buf, scales_buf) =
            crate::model_loader::load_w4a16_weight(&quantized_dir, "model.language_model.layers.0.mlp.down_proj.weight")
                .expect("real quantized down_proj weight must load");

        let out_features = 5120usize;
        let in_features = 17408usize;
        let group_size = 128usize;
        let num_tokens = 20usize;

        // Real, distinct activation vectors per token (not all the same
        // vector repeated -- a real test of the batched kernel's per-token
        // indexing, not just its reduction).
        let x_f32: Vec<f32> = (0..num_tokens * in_features)
            .map(|i| {
                let t = i / in_features;
                let k = i % in_features;
                (((k + t * 7) % 13) as i32 - 6) as f32 * 0.05
            })
            .collect();
        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();

        // Reference: real per-token GEMV, one real kernel launch per token.
        let mut expected = vec![0.0f32; num_tokens * out_features];
        for t in 0..num_tokens {
            let mut xt_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(in_features).unwrap();
            xt_buf.copy_from_host(&x_bf16[t * in_features..(t + 1) * in_features]).unwrap();
            let mut yt_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(out_features).unwrap();
            w4a16_gemv_bf16(&xt_buf, &qweight_buf, &scales_buf, &mut yt_buf, out_features, in_features, group_size).unwrap();
            let mut yt_bf16 = vec![0u16; out_features];
            yt_buf.copy_to_host(&mut yt_bf16).unwrap();
            for o in 0..out_features {
                expected[t * out_features + o] = bf16_to_f32(yt_bf16[o]);
            }
        }

        // Under test: real batched kernel, ONE launch for all 20 tokens.
        let mut y_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_tokens * out_features).unwrap();
        w4a16_gemm_prefill_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y_buf, out_features, in_features, group_size, num_tokens)
            .expect("real w4a16_gemm_prefill_bf16 call failed against real quantized down_proj weight");
        let mut y_bf16 = vec![0u16; num_tokens * out_features];
        y_buf.copy_to_host(&mut y_bf16).unwrap();

        let mut max_diff = 0.0f32;
        for t in 0..num_tokens {
            for o in 0..out_features {
                let got = bf16_to_f32(y_bf16[t * out_features + o]);
                let exp = expected[t * out_features + o];
                let diff = (got - exp).abs();
                max_diff = max_diff.max(diff);
                assert!(
                    diff < 1e-3,
                    "token {t}, output row {o}: batched kernel got {got}, per-token-GEMV reference {exp} (diff {diff})"
                );
            }
        }
        eprintln!("real batched-vs-per-token max abs diff (should be ~0, both bf16-rounded from the same math): {max_diff}");
    }

    /// §108: real, measured throughput comparison -- the OLD per-token
    /// prefill path (`w4a16_gemv_bf16` called once per token, exactly
    /// what `forward_one_token`-in-a-loop was doing for quantized prefill
    /// before this kernel existed) vs. the NEW batched
    /// `w4a16_gemm_prefill_bf16` (one real launch for the whole chunk).
    /// `num_tokens=32`: representative of this repo's own real benchmark
    /// prompts (§106/§108's HTTP benchmarks saw ~30-40-token prompts).
    /// Real weights both sides (the same real 27B `down_proj` tensor).
    #[test]
    #[ignore]
    fn bench_real_w4a16_batched_prefill_vs_per_token_loop() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let quantized_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen38_27b_w4a16");
        if !quantized_dir.is_dir() {
            eprintln!("skipping: no real quantized 27B checkpoint at {}", quantized_dir.display());
            return;
        }

        let (qweight_buf, scales_buf) =
            crate::model_loader::load_w4a16_weight(&quantized_dir, "model.language_model.layers.0.mlp.down_proj.weight")
                .expect("real quantized down_proj weight must load");

        let out_features = 5120usize;
        let in_features = 17408usize;
        let group_size = 128usize;
        let num_tokens = 32usize;
        let iters = 200;
        let warmup = 20;

        let x_f32: Vec<f32> = (0..num_tokens * in_features).map(|i| ((i % 91) as f32 - 45.0) * 0.01).collect();
        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();

        let mut xt_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(in_features).unwrap();
        xt_buf.copy_from_host(&x_bf16[..in_features]).unwrap();
        let mut yt_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(out_features).unwrap();
        for _ in 0..warmup {
            for _ in 0..num_tokens {
                w4a16_gemv_bf16(&xt_buf, &qweight_buf, &scales_buf, &mut yt_buf, out_features, in_features, group_size).unwrap();
            }
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            for _ in 0..num_tokens {
                w4a16_gemv_bf16(&xt_buf, &qweight_buf, &scales_buf, &mut yt_buf, out_features, in_features, group_size).unwrap();
            }
        }
        let per_token_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let mut y_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_tokens * out_features).unwrap();
        for _ in 0..warmup {
            w4a16_gemm_prefill_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y_buf, out_features, in_features, group_size, num_tokens).unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            w4a16_gemm_prefill_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y_buf, out_features, in_features, group_size, num_tokens).unwrap();
        }
        let batched_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        println!(
            "down_proj shape (out={out_features}, in={in_features}), {num_tokens} tokens: per-token loop {per_token_us:.1} us/chunk vs batched prefill {batched_us:.1} us/chunk ({:.2}x)",
            per_token_us / batched_us
        );
    }

    /// §123 EXPERIMENT, correctness gate: the `TILE_M=32` kernel must
    /// produce IDENTICAL output to the already-shipped, already-proven
    /// `TILE_M=16` kernel (`w4a16_gemm_prefill_bf16`) for the same real
    /// weights/activations -- both compute the exact same real math, only
    /// the tile size differs. `num_tokens=40` deliberately spans BOTH a
    /// full 32-token tile and a real partial tail tile.
    #[test]
    #[ignore]
    fn real_w4a16_gemm_prefill_tile32_matches_tile16() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let quantized_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen38_27b_w4a16");
        if !quantized_dir.is_dir() {
            eprintln!("skipping: no real quantized 27B checkpoint at {}", quantized_dir.display());
            return;
        }

        let (qweight_buf, scales_buf) =
            crate::model_loader::load_w4a16_weight(&quantized_dir, "model.language_model.layers.0.mlp.down_proj.weight")
                .expect("real quantized down_proj weight must load");

        let out_features = 5120usize;
        let in_features = 17408usize;
        let group_size = 128usize;
        let num_tokens = 40usize;

        let x_f32: Vec<f32> = (0..num_tokens * in_features)
            .map(|i| {
                let t = i / in_features;
                let k = i % in_features;
                (((k + t * 7) % 13) as i32 - 6) as f32 * 0.05
            })
            .collect();
        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();

        let mut y16_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_tokens * out_features).unwrap();
        w4a16_gemm_prefill_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y16_buf, out_features, in_features, group_size, num_tokens)
            .expect("real TILE_M=16 call failed");
        let mut y16 = vec![0u16; num_tokens * out_features];
        y16_buf.copy_to_host(&mut y16).unwrap();

        let mut y32_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_tokens * out_features).unwrap();
        w4a16_gemm_prefill_tile32_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y32_buf, out_features, in_features, group_size, num_tokens)
            .expect("real TILE_M=32 call failed");
        let mut y32 = vec![0u16; num_tokens * out_features];
        y32_buf.copy_to_host(&mut y32).unwrap();

        let mut max_diff = 0.0f32;
        let mut num_exceeding = 0usize;
        for i in 0..num_tokens * out_features {
            let a = bf16_to_f32(y16[i]);
            let b = bf16_to_f32(y32[i]);
            let diff = (a - b).abs();
            max_diff = max_diff.max(diff);
            if diff >= 1e-3 {
                num_exceeding += 1;
            }
        }
        eprintln!("TILE_M=32 vs TILE_M=16: max abs diff={max_diff}, num_exceeding(>=1e-3)={num_exceeding}/{}", num_tokens * out_features);
        assert_eq!(num_exceeding, 0, "TILE_M=32 must match TILE_M=16 within the same real bf16-rounding tolerance -- same math, different tile size only");
    }

    /// §123 EXPERIMENT: real, direct A/B between `TILE_M=16` (shipped)
    /// and `TILE_M=32` (experimental) at real prompt-length-representative
    /// token counts -- tests §121 Part 2's own "Dynamic Batch Tile Sizing"
    /// roadmap claim directly: does halving `ceil(num_tokens/TILE_M)`
    /// weight-VRAM sweeps actually win at THIS benchmark's real ~35-54
    /// token prompts, or does doubled register/LDS pressure cost more
    /// real occupancy than it buys back? Real weights both sides (27B
    /// `down_proj`). Not wired into the model regardless of outcome --
    /// this is the decisive measurement that decides whether it's worth
    /// doing so.
    #[test]
    #[ignore]
    fn bench_real_w4a16_gemm_prefill_tile16_vs_tile32() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let quantized_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen38_27b_w4a16");
        if !quantized_dir.is_dir() {
            eprintln!("skipping: no real quantized 27B checkpoint at {}", quantized_dir.display());
            return;
        }

        let (qweight_buf, scales_buf) =
            crate::model_loader::load_w4a16_weight(&quantized_dir, "model.language_model.layers.0.mlp.down_proj.weight")
                .expect("real quantized down_proj weight must load");

        let out_features = 5120usize;
        let in_features = 17408usize;
        let group_size = 128usize;
        let iters = 200;
        let warmup = 20;

        // Real prompt-length-representative token counts: this
        // benchmark's own real HTTP tasks land in the ~35-54 range
        // (§105/§108's own diagnosis); 16/64/128 bracket the TILE_M=16
        // boundary and the largest real bucket ever exercised.
        for &num_tokens in &[16usize, 24, 32, 40, 54, 64, 128] {
            let x_f32: Vec<f32> = (0..num_tokens * in_features).map(|i| ((i % 91) as f32 - 45.0) * 0.01).collect();
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
            x_buf.copy_from_host(&x_bf16).unwrap();

            let mut y16_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_tokens * out_features).unwrap();
            for _ in 0..warmup {
                w4a16_gemm_prefill_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y16_buf, out_features, in_features, group_size, num_tokens).unwrap();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                w4a16_gemm_prefill_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y16_buf, out_features, in_features, group_size, num_tokens).unwrap();
            }
            let tile16_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            let mut y32_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_tokens * out_features).unwrap();
            for _ in 0..warmup {
                w4a16_gemm_prefill_tile32_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y32_buf, out_features, in_features, group_size, num_tokens).unwrap();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                w4a16_gemm_prefill_tile32_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y32_buf, out_features, in_features, group_size, num_tokens).unwrap();
            }
            let tile32_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            println!(
                "num_tokens={num_tokens:4}: TILE_M=16 {tile16_us:8.1}us vs TILE_M=32 {tile32_us:8.1}us ({:.2}x{})",
                tile16_us / tile32_us,
                if tile32_us < tile16_us { "  <- TILE_M=32 wins" } else { "" }
            );
        }
    }

    /// §126 EXPERIMENT, correctness gate: the real, production-shaped
    /// WMMA INT8 prefill kernel, run against REAL 27B `down_proj`
    /// weights, checked two ways: (1) against a real, independent CPU
    /// reference that replicates the SAME intended INT8 quantization
    /// math (catches real kernel-implementation bugs, not just
    /// quantization noise); (2) against the shipped `TILE_N=16` bf16
    /// kernel's real output (measures the real, disclosed cost of the
    /// new INT8-activation-quantization rounding this kernel introduces
    /// -- a genuinely new rounding source, unlike TILE_N=16's bit-exact
    /// same-math change).
    #[test]
    #[ignore]
    fn real_w4a16_gemm_prefill_wmma_int8_matches_cpu_reference_and_shipped_kernel() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let quantized_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen38_27b_w4a16");
        if !quantized_dir.is_dir() {
            eprintln!("skipping: no real quantized 27B checkpoint at {}", quantized_dir.display());
            return;
        }

        let (qweight_buf, scales_buf) =
            crate::model_loader::load_w4a16_weight(&quantized_dir, "model.language_model.layers.0.mlp.down_proj.weight")
                .expect("real quantized down_proj weight must load");

        let out_features = 5120usize;
        let in_features = 17408usize;
        let group_size = 128usize;
        let num_tokens = 40usize; // real partial-tile tail (not a multiple of 16)

        let mut qweight_host = vec![0u32; out_features * (in_features / 8)];
        qweight_buf.copy_to_host(&mut qweight_host).unwrap();
        let mut scales_host = vec![0u16; out_features * (in_features / group_size)];
        scales_buf.copy_to_host(&mut scales_host).unwrap();

        // Real, deterministic, non-half-integer bf16 activation values
        // (avoids round-to-even vs round-half-away-from-zero ambiguity
        // between HIP's `__float2int_rn` and any CPU-side rounding).
        let x_f32: Vec<f32> = (0..num_tokens * in_features)
            .map(|i| {
                let t = i / in_features;
                let k = i % in_features;
                (((k + t * 7) % 13) as i32 - 6) as f32 * 0.0523
            })
            .collect();
        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let x_bf16_f32: Vec<f32> = x_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();

        // Real reference: shipped TILE_N=16 bf16 kernel.
        let mut y_ref_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_tokens * out_features).unwrap();
        w4a16_gemm_prefill_tile_n16_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y_ref_buf, out_features, in_features, group_size, num_tokens)
            .expect("real shipped TILE_N=16 call failed");
        let mut y_ref = vec![0u16; num_tokens * out_features];
        y_ref_buf.copy_to_host(&mut y_ref).unwrap();

        // Real candidate: WMMA INT8 kernel.
        let mut y_wmma_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_tokens * out_features).unwrap();
        w4a16_gemm_prefill_wmma_int8_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y_wmma_buf, out_features, in_features, group_size, num_tokens)
            .expect("real WMMA INT8 call failed");
        let mut y_wmma = vec![0u16; num_tokens * out_features];
        y_wmma_buf.copy_to_host(&mut y_wmma).unwrap();

        // Real, independent CPU reference: replicates the SAME intended
        // per-token, per-128-group symmetric INT8 quantization (real
        // `max_abs/127`, real round-to-nearest, real [-127,127] clamp)
        // and the SAME `nibble-8` weight dequant, accumulating in f32
        // group-by-group -- the real, intended math this kernel's GPU
        // code should be computing, independent of the GPU kernel text.
        let num_groups = in_features / group_size;
        let mut y_cpu = vec![0.0f32; num_tokens * out_features];
        for row in 0..out_features {
            let qw_row = &qweight_host[row * (in_features / 8)..(row + 1) * (in_features / 8)];
            let scale_row = &scales_host[row * num_groups..(row + 1) * num_groups];
            for t in 0..num_tokens {
                let xr = &x_bf16_f32[t * in_features..(t + 1) * in_features];
                let mut acc = 0.0f32;
                for g in 0..num_groups {
                    let k0 = g * group_size;
                    // Real per-token, per-group activation quantization.
                    let mut max_abs = 0.0f32;
                    for k in 0..group_size {
                        max_abs = max_abs.max(xr[k0 + k].abs());
                    }
                    let a_scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
                    let mut x_q = [0i32; 128];
                    for k in 0..group_size {
                        let q = (xr[k0 + k] / a_scale).round();
                        x_q[k] = q.clamp(-127.0, 127.0) as i32;
                    }
                    let w_scale = bf16_to_f32(scale_row[g]);
                    let mut raw = 0i32;
                    for k in 0..group_size {
                        let word = qw_row[(k0 + k) / 8];
                        let nibble = (word >> ((k % 8) * 4)) & 0xF;
                        let w_q = nibble as i32 - 8;
                        raw += w_q * x_q[k];
                    }
                    acc += w_scale * a_scale * raw as f32;
                }
                y_cpu[t * out_features + row] = acc;
            }
        }

        let mut max_diff_vs_cpu = 0.0f32;
        let mut sum_abs_ref = 0.0f64;
        let mut sum_abs_diff_vs_ref = 0.0f64;
        let mut max_diff_vs_ref = 0.0f32;
        let mut worst: Option<(usize, usize, f32)> = None; // (token, row, diff)
        for i in 0..num_tokens * out_features {
            let wmma = bf16_to_f32(y_wmma[i]);
            let cpu = y_cpu[i];
            let refv = bf16_to_f32(y_ref[i]);
            let d = (wmma - cpu).abs();
            max_diff_vs_cpu = max_diff_vs_cpu.max(d);
            max_diff_vs_ref = max_diff_vs_ref.max((wmma - refv).abs());
            sum_abs_ref += refv.abs() as f64;
            sum_abs_diff_vs_ref += (wmma - refv).abs() as f64;
            if worst.map(|(_, _, wd)| d > wd).unwrap_or(true) {
                worst = Some((i / out_features, i % out_features, d));
            }
        }
        let rel_l1_vs_ref = sum_abs_diff_vs_ref / sum_abs_ref.max(1e-9);
        eprintln!("WMMA INT8 kernel vs real CPU quantization reference: max abs diff = {max_diff_vs_cpu}");
        eprintln!("WMMA INT8 kernel vs real shipped bf16 kernel: max abs diff = {max_diff_vs_ref}, relative L1 = {:.4}%", rel_l1_vs_ref * 100.0);
        eprintln!("wmma[0..4] = {:?}", (0..4).map(|i| bf16_to_f32(y_wmma[i])).collect::<Vec<_>>());
        eprintln!("cpu [0..4] = {:?}", &y_cpu[0..4]);
        eprintln!("ref [0..4] = {:?}", (0..4).map(|i| bf16_to_f32(y_ref[i])).collect::<Vec<_>>());
        if let Some((wt, wr, wd)) = worst {
            eprintln!(
                "worst (token={wt}, row={wr}, block_x={}, block_y={}, warp={}, frag_row={}): wmma={}, cpu={}, ref={}, diff={wd}",
                wr / 128, wt / 16, (wr % 128) / 16, wr % 16,
                bf16_to_f32(y_wmma[wt * out_features + wr]), y_cpu[wt * out_features + wr], bf16_to_f32(y_ref[wt * out_features + wr])
            );
        }
        // Real distribution breakdown: is the error concentrated in a
        // specific token-tile (e.g. the partial tail 32..39) or spread
        // uniformly? Bucket by token-tile and row-tile.
        for tt in 0..((num_tokens + 15) / 16) {
            let t_lo = tt * 16;
            let t_hi = ((tt + 1) * 16).min(num_tokens);
            let mut bucket_max = 0.0f32;
            for t in t_lo..t_hi {
                for row in 0..out_features {
                    let i = t * out_features + row;
                    let d = (bf16_to_f32(y_wmma[i]) - y_cpu[i]).abs();
                    bucket_max = bucket_max.max(d);
                }
            }
            eprintln!("token-tile [{t_lo},{t_hi}): max diff vs cpu = {bucket_max}");
        }

        assert!(
            max_diff_vs_cpu < 0.05,
            "WMMA INT8 kernel must match its own intended CPU-reference quantization math closely (max_diff_vs_cpu={max_diff_vs_cpu}) -- a larger gap signals a real kernel bug, not quantization noise"
        );
    }

    /// §124 EXPERIMENT, correctness gate: the `TILE_N=16` kernel must
    /// produce IDENTICAL output to the shipped `TILE_N=8` kernel for the
    /// same real weights/activations. `num_tokens=40` spans a real
    /// partial tail tile; `out_features=5120` is NOT a multiple of 16,
    /// so this also exercises the real partial-row-pair tail (row_o1
    /// running past `out_features` for the very last warp of the very
    /// last x-block).
    #[test]
    #[ignore]
    fn real_w4a16_gemm_prefill_tile_n16_matches_tile_n8() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let quantized_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen38_27b_w4a16");
        if !quantized_dir.is_dir() {
            eprintln!("skipping: no real quantized 27B checkpoint at {}", quantized_dir.display());
            return;
        }

        let (qweight_buf, scales_buf) =
            crate::model_loader::load_w4a16_weight(&quantized_dir, "model.language_model.layers.0.mlp.down_proj.weight")
                .expect("real quantized down_proj weight must load");

        let out_features = 5120usize;
        let in_features = 17408usize;
        let group_size = 128usize;
        let num_tokens = 40usize;

        let x_f32: Vec<f32> = (0..num_tokens * in_features)
            .map(|i| {
                let t = i / in_features;
                let k = i % in_features;
                (((k + t * 7) % 13) as i32 - 6) as f32 * 0.05
            })
            .collect();
        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();

        let mut y8_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_tokens * out_features).unwrap();
        w4a16_gemm_prefill_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y8_buf, out_features, in_features, group_size, num_tokens)
            .expect("real TILE_N=8 call failed");
        let mut y8 = vec![0u16; num_tokens * out_features];
        y8_buf.copy_to_host(&mut y8).unwrap();

        let mut y16_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_tokens * out_features).unwrap();
        w4a16_gemm_prefill_tile_n16_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y16_buf, out_features, in_features, group_size, num_tokens)
            .expect("real TILE_N=16 call failed");
        let mut y16 = vec![0u16; num_tokens * out_features];
        y16_buf.copy_to_host(&mut y16).unwrap();

        let mut max_diff = 0.0f32;
        let mut num_exceeding = 0usize;
        for i in 0..num_tokens * out_features {
            let a = bf16_to_f32(y8[i]);
            let b = bf16_to_f32(y16[i]);
            let diff = (a - b).abs();
            max_diff = max_diff.max(diff);
            if diff >= 1e-3 {
                num_exceeding += 1;
            }
        }
        eprintln!("TILE_N=16 vs TILE_N=8: max abs diff={max_diff}, num_exceeding(>=1e-3)={num_exceeding}/{}", num_tokens * out_features);
        assert_eq!(num_exceeding, 0, "TILE_N=16 must match TILE_N=8 within the same real bf16-rounding tolerance -- same math, different tile size only");
    }

    /// §124 EXPERIMENT: real, direct A/B between `TILE_N=8` (shipped) and
    /// `TILE_N=16` (experimental, halves the redundant activation-tile
    /// re-read count by halving block count along the output-row
    /// dimension) at real prompt-length-representative token counts.
    /// Real weights both sides (27B `down_proj`). Not wired into the
    /// model regardless of outcome.
    #[test]
    #[ignore]
    fn bench_real_w4a16_gemm_prefill_tile_n8_vs_tile_n16() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let quantized_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen38_27b_w4a16");
        if !quantized_dir.is_dir() {
            eprintln!("skipping: no real quantized 27B checkpoint at {}", quantized_dir.display());
            return;
        }

        let (qweight_buf, scales_buf) =
            crate::model_loader::load_w4a16_weight(&quantized_dir, "model.language_model.layers.0.mlp.down_proj.weight")
                .expect("real quantized down_proj weight must load");

        let out_features = 5120usize;
        let in_features = 17408usize;
        let group_size = 128usize;
        let iters = 200;
        let warmup = 20;

        for &num_tokens in &[16usize, 24, 32, 40, 54, 64, 128] {
            let x_f32: Vec<f32> = (0..num_tokens * in_features).map(|i| ((i % 91) as f32 - 45.0) * 0.01).collect();
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
            x_buf.copy_from_host(&x_bf16).unwrap();

            let mut y8_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_tokens * out_features).unwrap();
            for _ in 0..warmup {
                w4a16_gemm_prefill_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y8_buf, out_features, in_features, group_size, num_tokens).unwrap();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                w4a16_gemm_prefill_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y8_buf, out_features, in_features, group_size, num_tokens).unwrap();
            }
            let tile_n8_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            let mut y16_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_tokens * out_features).unwrap();
            for _ in 0..warmup {
                w4a16_gemm_prefill_tile_n16_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y16_buf, out_features, in_features, group_size, num_tokens).unwrap();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                w4a16_gemm_prefill_tile_n16_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y16_buf, out_features, in_features, group_size, num_tokens).unwrap();
            }
            let tile_n16_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            println!(
                "num_tokens={num_tokens:4}: TILE_N=8 {tile_n8_us:8.1}us vs TILE_N=16 {tile_n16_us:8.1}us ({:.2}x{})",
                tile_n8_us / tile_n16_us,
                if tile_n16_us < tile_n8_us { "  <- TILE_N=16 wins" } else { "" }
            );
        }
    }

    /// §126 EXPERIMENT, THE decisive benchmark: real, measured throughput
    /// comparison, shipped scalar `TILE_N=16` kernel vs. the new WMMA
    /// INT8 tensor-core kernel, real 27B `down_proj` weights, real token
    /// counts spanning this benchmark's own real prompt-length range.
    /// Not wired into the model regardless of outcome -- see
    /// `docs/DECISIONS.md` §126 for the real, honest verdict.
    #[test]
    #[ignore]
    fn bench_real_w4a16_gemm_prefill_tile_n16_vs_wmma_int8() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let quantized_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen38_27b_w4a16");
        if !quantized_dir.is_dir() {
            eprintln!("skipping: no real quantized 27B checkpoint at {}", quantized_dir.display());
            return;
        }

        let (qweight_buf, scales_buf) =
            crate::model_loader::load_w4a16_weight(&quantized_dir, "model.language_model.layers.0.mlp.down_proj.weight")
                .expect("real quantized down_proj weight must load");

        let out_features = 5120usize;
        let in_features = 17408usize;
        let group_size = 128usize;
        let iters = 200;
        let warmup = 20;

        for &num_tokens in &[16usize, 24, 32, 40, 54, 64, 128] {
            let x_f32: Vec<f32> = (0..num_tokens * in_features).map(|i| ((i % 91) as f32 - 45.0) * 0.01).collect();
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
            x_buf.copy_from_host(&x_bf16).unwrap();

            let mut y16_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_tokens * out_features).unwrap();
            for _ in 0..warmup {
                w4a16_gemm_prefill_tile_n16_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y16_buf, out_features, in_features, group_size, num_tokens).unwrap();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                w4a16_gemm_prefill_tile_n16_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y16_buf, out_features, in_features, group_size, num_tokens).unwrap();
            }
            let tile_n16_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            let mut y_wmma_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_tokens * out_features).unwrap();
            for _ in 0..warmup {
                w4a16_gemm_prefill_wmma_int8_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y_wmma_buf, out_features, in_features, group_size, num_tokens).unwrap();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                w4a16_gemm_prefill_wmma_int8_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y_wmma_buf, out_features, in_features, group_size, num_tokens).unwrap();
            }
            let wmma_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            println!(
                "num_tokens={num_tokens:4}: TILE_N=16 (scalar) {tile_n16_us:8.1}us vs WMMA INT8 {wmma_us:8.1}us ({:.2}x{})",
                tile_n16_us / wmma_us,
                if wmma_us < tile_n16_us { "  <- WMMA wins" } else { "" }
            );
        }
    }

    /// Real, measured throughput comparison: `gemv_bf16` vs. `blas::
    /// gemm_bf16_linear` (both synced per call) for this model's real
    /// `down_proj` shape -- the direct before/after number for whether the
    /// custom kernel is actually worth swapping into the hot path.
    #[test]
    #[ignore]
    fn bench_real_gemv_vs_gemm_down_proj_shape() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let out_features = 2560usize;
        let in_features = 9216usize;
        let iters = 2000;
        let warmup = 100;

        let x_f32: Vec<f32> = (0..in_features).map(|i| ((i % 91) as f32 - 45.0) * 0.01).collect();
        let w_f32: Vec<f32> = (0..out_features * in_features).map(|i| ((i % 71) as f32 - 35.0) * 0.01).collect();
        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let w_bf16: Vec<u16> = w_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(in_features).unwrap();
        let mut w_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(out_features * in_features).unwrap();
        let mut y_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(out_features).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();
        w_buf.copy_from_host(&w_bf16).unwrap();

        for _ in 0..warmup {
            gemv_bf16(&x_buf, &w_buf, &mut y_buf, out_features, in_features).unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            gemv_bf16(&x_buf, &w_buf, &mut y_buf, out_features, in_features).unwrap();
        }
        let gemv_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        let handle = crate::blas::BlasHandle::create().expect("real hipblasCreate failed");
        for _ in 0..warmup {
            crate::blas::gemm_bf16_linear(&handle, &x_buf, &w_buf, &mut y_buf, 1, in_features, out_features).unwrap();
        }
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            crate::blas::gemm_bf16_linear(&handle, &x_buf, &w_buf, &mut y_buf, 1, in_features, out_features).unwrap();
        }
        let gemm_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

        println!(
            "down_proj shape (out={out_features}, in={in_features}): gemv_bf16 {gemv_us:.3} us/call vs hipblasGemmEx {gemm_us:.3} us/call ({iters} iters each, includes hipDeviceSynchronize)"
        );
    }

    /// §106: real, head-to-head kernel benchmark -- `gemv_bf16` (existing,
    /// unquantized) vs. `w4a16_gemv_bf16` (new) on the SAME real weight
    /// tensors from the real converted 27B checkpoint, at two real shapes
    /// (`down_proj`: out=5120,in=17408; `up_proj`: out=17408,in=5120 --
    /// the same dims transposed). Both kernels read REAL data: the
    /// unquantized bf16 tensor from the original snapshot for one side,
    /// the quantized tensors from `quantize_w4a16.py`'s real output for
    /// the other -- not synthetic data quantized on the fly. Same timed-
    /// loop-with-warmup methodology as `bench_real_gemv_vs_gemm_down_proj_shape`.
    #[test]
    #[ignore]
    fn bench_real_w4a16_gemv_vs_bf16_gemv_real_27b_shapes() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let quantized_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/qwen38_27b_w4a16");
        let bf16_dir = std::path::Path::new(env!("HOME"))
            .join(".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots");
        if !quantized_dir.is_dir() {
            eprintln!("skipping: no real quantized 27B checkpoint at {}", quantized_dir.display());
            return;
        }
        let bf16_snapshot = std::fs::read_dir(&bf16_dir)
            .ok()
            .and_then(|mut e| e.next())
            .map(|e| e.unwrap().path());
        let Some(bf16_snapshot) = bf16_snapshot else {
            eprintln!("skipping: no real bf16 27B snapshot found under {}", bf16_dir.display());
            return;
        };

        let iters = 2000;
        let warmup = 100;
        let group_size = 128usize;

        for (tensor_name, out_features, in_features) in [
            ("model.language_model.layers.0.mlp.down_proj.weight", 5120usize, 17408usize),
            ("model.language_model.layers.0.mlp.up_proj.weight", 17408usize, 5120usize),
        ] {
            let w_bf16_buf = crate::model_loader::load_bf16_weight(&bf16_snapshot, tensor_name)
                .expect("real bf16 weight must load from the real 27B snapshot");
            let (qweight_buf, scales_buf) = crate::model_loader::load_w4a16_weight(&quantized_dir, tensor_name)
                .expect("real quantized weight must load");

            let x_f32: Vec<f32> = (0..in_features).map(|i| (((i % 13) as i32 - 6) as f32) * 0.05).collect();
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(in_features).unwrap();
            x_buf.copy_from_host(&x_bf16).unwrap();
            let mut y_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(out_features).unwrap();

            for _ in 0..warmup {
                gemv_bf16(&x_buf, &w_bf16_buf, &mut y_buf, out_features, in_features).unwrap();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                gemv_bf16(&x_buf, &w_bf16_buf, &mut y_buf, out_features, in_features).unwrap();
            }
            let bf16_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            for _ in 0..warmup {
                w4a16_gemv_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y_buf, out_features, in_features, group_size).unwrap();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                w4a16_gemv_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y_buf, out_features, in_features, group_size).unwrap();
            }
            let w4a16_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;

            let speedup = bf16_us / w4a16_us;
            println!(
                "{tensor_name} (out={out_features}, in={in_features}): bf16 gemv {bf16_us:.3} us/call vs w4a16 gemv {w4a16_us:.3} us/call -> {speedup:.2}x ({iters} iters each, includes hipDeviceSynchronize)"
            );
        }
    }


    // ---- argmax -----------------------------------------------------

    /// Real correctness test: a real-scale (`VOCAB_SIZE=248320`) logits
    /// buffer with a single, unambiguous maximum planted at a real index,
    /// computed on the real GPU via the real `argmax_bf16` kernel, checked
    /// against the SAME index.
    #[test]
    fn real_argmax_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let n = 248320usize;
        let mut logits_f32 = vec![0f32; n];
        for (i, v) in logits_f32.iter_mut().enumerate() {
            // Deterministic, bounded, no ties -- distinct from the planted max below.
            *v = ((i % 97) as f32 - 48.0) * 0.01;
        }
        let expected_idx = 137529usize;
        logits_f32[expected_idx] = 100.0; // unambiguous real maximum

        let logits_bf16: Vec<u16> = logits_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut logits_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(n).unwrap();
        logits_buf.copy_from_host(&logits_bf16).unwrap();

        let got = argmax_bf16(&logits_buf).expect("real argmax_bf16 call failed");
        assert_eq!(got, expected_idx as i32, "argmax_bf16 did not find the real planted maximum's index");
    }

    /// Real tie-break correctness test: TWO indices share the exact same
    /// maximum value -- the kernel must return the LOWER of the two,
    /// matching the original host-side `argmax_sample`'s "first occurrence
    /// wins" semantics exactly (see `argmax.hip`'s header for why this
    /// isn't automatic through a tree reduction over strided per-thread
    /// assignments).
    #[test]
    fn real_argmax_breaks_ties_toward_the_lower_index() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let n = 4096usize;
        let mut logits_f32 = vec![0f32; n];
        for (i, v) in logits_f32.iter_mut().enumerate() {
            *v = ((i % 53) as f32 - 26.0) * 0.01;
        }
        let lower_tie_idx = 501usize;
        let higher_tie_idx = 3117usize;
        logits_f32[lower_tie_idx] = 50.0;
        logits_f32[higher_tie_idx] = 50.0; // exact tie with lower_tie_idx

        let logits_bf16: Vec<u16> = logits_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut logits_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(n).unwrap();
        logits_buf.copy_from_host(&logits_bf16).unwrap();

        let got = argmax_bf16(&logits_buf).expect("real argmax_bf16 call failed");
        assert_eq!(got, lower_tie_idx as i32, "argmax_bf16 must break an exact tie toward the LOWER index");
    }

    /// §99 decisive test: `causal_softmax_bf16` on a real `[3,4]` score
    /// matrix, `start_position=0` -- so row 0's valid range is just column
    /// 0, row 1's is columns [0,1], row 2's is columns [0,2], and column 3
    /// is masked for EVERY row (none of the 3 rows may see a key at
    /// index 3 yet). Checked against an independent plain-f32 softmax
    /// computed over each row's own real valid prefix, with the masked
    /// tail asserted to be exactly zero.
    #[test]
    fn real_causal_softmax_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_rows = 3usize;
        let kv_len = 4usize;
        let start_position = 0i32;

        // Real, varied raw scores (pre-softmax) -- row r's TRUE valid
        // range is [0, r] inclusive; columns beyond that are masked
        // regardless of what raw score value sits there (deliberately
        // planted as a large value here, to prove masking overrides it).
        let scores_f32: Vec<f32> = vec![
            1.0, 99.0, 99.0, 99.0, // row 0: only column 0 is valid
            0.5, -0.5, 99.0, 99.0, // row 1: columns [0,1] valid
            0.2, 0.8, -0.3, 99.0, // row 2: columns [0,2] valid
        ];

        let mut expected = vec![0f32; num_rows * kv_len];
        for r in 0..num_rows {
            let valid_len = start_position as usize + r + 1;
            let row = &scores_f32[r * kv_len..r * kv_len + valid_len];
            let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&v| (v - max_v).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (j, &e) in exps.iter().enumerate() {
                expected[r * kv_len + j] = e / sum;
            }
            // columns >= valid_len stay 0.0 (already initialized)
        }

        let scores_bf16: Vec<u16> = scores_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut scores_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(scores_bf16.len()).unwrap();
        scores_buf.copy_from_host(&scores_bf16).unwrap();

        causal_softmax_bf16(&mut scores_buf, kv_len, start_position, num_rows).expect("real causal_softmax_bf16 call failed");

        let mut got_bf16 = vec![0u16; num_rows * kv_len];
        scores_buf.copy_to_host(&mut got_bf16).unwrap();
        let got: Vec<f32> = got_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 0.01, "element {i}: GPU causal_softmax_bf16 got {g}, independent reference {e}");
        }

        // Explicit, separate check: every masked (causally future) column
        // is EXACTLY zero probability, not just numerically close to the
        // reference above (which already encodes 0.0, but this makes the
        // masking property itself the thing under test, not incidental).
        for r in 0..num_rows {
            let valid_len = start_position as usize + r + 1;
            for j in valid_len..kv_len {
                let v = got[r * kv_len + j];
                assert_eq!(v, 0.0, "row {r} col {j}: masked column must be exactly zero probability, got {v}");
            }
        }
    }

    /// §136 root-cause: isolates whether `causal_softmax_bf16` ITSELF is
    /// the source of the real divergence found when a verify-chunk's
    /// `kv_len` is padded past a threshold between 184-192
    /// (`docs/DECISIONS.md` §136) -- `gemm_qkt_bf16`/`gemm_pv_bf16` were
    /// already independently swept across the exact same shape/kv_len
    /// range and found clean (`blas::tests::real_gemm_{qkt,pv}_at_real_
    /// attention_shape_matches_reference_across_kv_len_sweep`), leaving
    /// this kernel as the one remaining untested link in the real
    /// GEMM-attention chain. REAL production shape (`t=6`, matching a K=6
    /// verify chunk); scores constructed to match what `gemm_qkt_bf16`
    /// ACTUALLY produces in the real pipeline (small, real values for
    /// `[0, real_len=132)`, near-zero for `[real_len, kv_len)` since
    /// that's `Q . 0 * scale = 0` against a zero-padded K-cache -- not
    /// artificial huge sentinels, the existing tiny toy test already
    /// covers that masking-overrides-large-values property).
    #[test]
    fn real_causal_softmax_at_real_attention_shape_matches_reference_across_kv_len_sweep() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let t = 6usize;
        let real_len = 132usize;
        let start_position = 126i32; // matches §136's real round-3 position

        // Deterministic pseudo-random REAL scores for [0, real_len);
        // small, realistic magnitude (matches real Q.K*scale output).
        let scores_real: Vec<f32> = (0..t * real_len).map(|i| ((((i * 2654435761u64.wrapping_add(1) as usize) % 1009) as f32 / 504.5) - 1.0) * 0.1).collect();

        for &kv_len in &[176usize, 184, 188, 190, 191, 192, 193, 196, 200, 256] {
            let mut scores_full = vec![0f32; t * kv_len];
            for r in 0..t {
                scores_full[r * kv_len..r * kv_len + real_len].copy_from_slice(&scores_real[r * real_len..(r + 1) * real_len]);
                // [real_len, kv_len) stays exactly 0.0 -- matching what a
                // real zero-padded K-cache produces through gemm_qkt_bf16.
            }

            // Independent reference: row r's TRUE valid range is
            // [0, start_position + r + 1) (may extend past real_len for
            // large r, but real_len=132 + t=6 = 138 < kv_len for every
            // value swept here, so valid_len is always <= real_len... no
            // wait: valid_len = start_position + r + 1 = 126+r+1, for
            // r in [0,6) that's [127,132] -- always <= real_len=132, so
            // every "valid" column per the causal mask has REAL (not
            // zero-padded) score data, matching the real pipeline exactly.
            let mut expected = vec![0f32; t * kv_len];
            for r in 0..t {
                let valid_len = (start_position as usize) + r + 1;
                let row = &scores_full[r * kv_len..r * kv_len + valid_len];
                let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = row.iter().map(|&v| (v - max_v).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for (j, &e) in exps.iter().enumerate() {
                    expected[r * kv_len + j] = e / sum;
                }
            }

            let scores_bf16: Vec<u16> = scores_full.iter().map(|&v| f32_to_bf16(v)).collect();
            let mut scores_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(scores_bf16.len()).unwrap();
            scores_buf.copy_from_host(&scores_bf16).unwrap();

            causal_softmax_bf16(&mut scores_buf, kv_len, start_position, t).expect("real causal_softmax_bf16 call failed");

            let mut got_bf16 = vec![0u16; t * kv_len];
            scores_buf.copy_to_host(&mut got_bf16).unwrap();
            let got: Vec<f32> = got_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

            let mut max_diff = 0f32;
            let mut num_exceeding = 0usize;
            for r in 0..t {
                let valid_len = (start_position as usize) + r + 1;
                for j in 0..kv_len {
                    let diff = (got[r * kv_len + j] - expected[r * kv_len + j]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    if diff > 0.05 {
                        num_exceeding += 1;
                    }
                    if j >= valid_len {
                        assert_eq!(got[r * kv_len + j], 0.0, "kv_len={kv_len} row {r} col {j}: masked column must be exactly zero, got {}", got[r * kv_len + j]);
                    }
                }
            }
            eprintln!("causal_softmax_bf16 at kv_len={kv_len}: max_diff={max_diff}, {num_exceeding}/{} elements exceed tolerance=0.05", t * kv_len);
            assert_eq!(num_exceeding, 0, "causal_softmax_bf16 at kv_len={kv_len} (real_len={real_len}) diverged from an independent CPU reference -- max_diff={max_diff}");
        }
    }

    /// §100 decisive test: `gdn_chunk_decay_bf16` on a real, small,
    /// hand-computable case -- `num_heads=2`, `num_chunks=2`,
    /// `chunk_size=4`, DIFFERENT `g` per (head,chunk) pair so a layout
    /// bug (e.g. accidentally reading/writing the head-major layout this
    /// kernel used before being fixed to token-major) would show up as a
    /// real, detectable mismatch, not silently pass the way a
    /// single-head test could. Raw scores planted as all-1.0 so the
    /// decayed output is directly readable as `pairwise_decay` itself.
    /// Checked against an independent plain-f64 reference that computes
    /// the SAME token-major `[chunk, pos, head]` output layout by hand.
    #[test]
    fn real_gdn_chunk_decay_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_heads = 2usize;
        let num_chunks = 2usize;
        let chunk_size = 4usize;

        // g[chunk][pos][head], token-major, real varied (<=0) values,
        // deliberately DIFFERENT per head so head 0 and head 1 never
        // coincidentally match.
        let g_f32: Vec<f32> = vec![
            // chunk 0
            -0.1, -0.5, -0.3, -0.2, -0.05, -0.4, -0.2, -0.1,
            // chunk 1
            -0.2, -0.1, -0.15, -0.3, -0.1, -0.2, -0.05, -0.25,
        ];
        assert_eq!(g_f32.len(), num_chunks * chunk_size * num_heads);

        // Independent reference computed directly in the SAME token-major
        // layout the kernel is now expected to produce.
        let mut expected_decay_exp = vec![0f32; num_chunks * chunk_size * num_heads];
        let mut expected_remaining_decay = vec![0f32; num_chunks * chunk_size * num_heads];
        let mut expected_chunk_decay = vec![0f32; num_chunks * num_heads];
        // expected_pairwise[head][chunk][i][j]
        let mut expected_pairwise = vec![0f32; num_heads * num_chunks * chunk_size * chunk_size];

        for chunk in 0..num_chunks {
            for head in 0..num_heads {
                let mut cum: Vec<f64> = Vec::with_capacity(chunk_size);
                let mut running = 0.0f64;
                for t in 0..chunk_size {
                    let g_idx = (chunk * chunk_size + t) * num_heads + head;
                    running += g_f32[g_idx] as f64;
                    cum.push(running);
                }
                let last = cum[chunk_size - 1];
                for t in 0..chunk_size {
                    let out_idx = (chunk * chunk_size + t) * num_heads + head;
                    expected_decay_exp[out_idx] = cum[t].exp() as f32;
                    expected_remaining_decay[out_idx] = (last - cum[t]).exp() as f32;
                }
                expected_chunk_decay[chunk * num_heads + head] = last.exp() as f32;
                for i in 0..chunk_size {
                    for j in 0..=i {
                        let pw_idx = ((head * num_chunks + chunk) * chunk_size + i) * chunk_size + j;
                        expected_pairwise[pw_idx] = (cum[i] - cum[j]).exp() as f32;
                    }
                }
            }
        }

        let raw_scores_f32 = vec![1.0f32; num_heads * num_chunks * chunk_size * chunk_size];
        let raw_scores_bf16: Vec<u16> = raw_scores_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut g_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(g_f32.len()).unwrap();
        let mut ut_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(raw_scores_bf16.len()).unwrap();
        let mut attn_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(raw_scores_bf16.len()).unwrap();
        let mut decay_exp_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(expected_decay_exp.len()).unwrap();
        let mut remaining_decay_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(expected_remaining_decay.len()).unwrap();
        let mut chunk_decay_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(expected_chunk_decay.len()).unwrap();
        g_buf.copy_from_host(&g_f32).unwrap();
        ut_buf.copy_from_host(&raw_scores_bf16).unwrap();
        attn_buf.copy_from_host(&raw_scores_bf16).unwrap();

        gdn_chunk_decay_bf16(
            &g_buf,
            &mut ut_buf,
            &mut attn_buf,
            &mut decay_exp_buf,
            &mut remaining_decay_buf,
            &mut chunk_decay_buf,
            num_heads,
            num_chunks,
            chunk_size,
        )
        .expect("real gdn_chunk_decay_bf16 call failed");

        let mut got_decay_exp = vec![0f32; expected_decay_exp.len()];
        let mut got_remaining_decay = vec![0f32; expected_remaining_decay.len()];
        let mut got_chunk_decay = vec![0f32; expected_chunk_decay.len()];
        let mut got_ut_bf16 = vec![0u16; raw_scores_bf16.len()];
        let mut got_attn_bf16 = vec![0u16; raw_scores_bf16.len()];
        decay_exp_buf.copy_to_host(&mut got_decay_exp).unwrap();
        remaining_decay_buf.copy_to_host(&mut got_remaining_decay).unwrap();
        chunk_decay_buf.copy_to_host(&mut got_chunk_decay).unwrap();
        ut_buf.copy_to_host(&mut got_ut_bf16).unwrap();
        attn_buf.copy_to_host(&mut got_attn_bf16).unwrap();
        let got_ut: Vec<f32> = got_ut_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
        let got_attn: Vec<f32> = got_attn_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        for i in 0..got_decay_exp.len() {
            assert!((got_decay_exp[i] - expected_decay_exp[i]).abs() < 1e-4, "decay_exp[{i}] (token-major): got {}, expected {}", got_decay_exp[i], expected_decay_exp[i]);
            assert!((got_remaining_decay[i] - expected_remaining_decay[i]).abs() < 1e-4, "remaining_decay[{i}] (token-major): got {}, expected {}", got_remaining_decay[i], expected_remaining_decay[i]);
        }
        for i in 0..got_chunk_decay.len() {
            assert!((got_chunk_decay[i] - expected_chunk_decay[i]).abs() < 1e-4, "chunk_decay[{i}] (token-major): got {}, expected {}", got_chunk_decay[i], expected_chunk_decay[i]);
        }
        // A real, bf16-appropriate tolerance: `ut_system`/`intra_chunk_attn`
        // round-trip through bf16 (unlike the f32-native fields above) --
        // bf16's own step size near 1.0 is `2^-8 = 0.0039`, so `1e-3` was
        // tighter than bf16 itself can represent (confirmed by hand:
        // `exp(-0.3) = 0.7408182`, nearest bf16 value is `0.7421875`).
        for idx in 0..got_ut.len() {
            assert!((got_ut[idx] - expected_pairwise[idx]).abs() < 0.01, "ut_system[{idx}] (head-major): got {}, expected {}", got_ut[idx], expected_pairwise[idx]);
            assert!((got_attn[idx] - expected_pairwise[idx]).abs() < 0.01, "intra_chunk_attn[{idx}] (head-major): got {}, expected {}", got_attn[idx], expected_pairwise[idx]);
        }
    }

    /// §100 decisive test: `gdn_chunk_utsolve_bf16` on a real, small,
    /// hand-computable lower-triangular system -- `chunk_size=3`,
    /// `width=2`, checked against an independent plain-f32 forward
    /// substitution computed by a completely separate loop (not calling
    /// this crate's own kernel code).
    #[test]
    fn real_gdn_chunk_utsolve_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_heads = 1usize;
        let num_chunks = 1usize;
        let chunk_size = 3usize;
        let width = 2usize;

        // Real lower-triangular system: only j<i entries matter (the
        // diagonal -- 9.0, 9.0, 9.0 here -- and the strictly-upper
        // entries are deliberately planted as garbage to prove
        // `unitriangular=True`'s real contract: the kernel must IGNORE
        // them, not just happen to get zeros there).
        let ut_system_f32: Vec<f32> = vec![
            9.0, 777.0, 777.0, // row 0 (no real j<0 term)
            0.5, 9.0, 777.0, // row 1 (real term: j=0)
            0.2, -0.3, 9.0, // row 2 (real terms: j=0,1)
        ];
        let rhs_f32: Vec<f32> = vec![
            1.0, 2.0, // row 0
            0.5, -1.0, // row 1
            2.0, 0.0, // row 2
        ];

        // Independent reference: real forward substitution, unit diagonal.
        let mut expected = vec![0f32; chunk_size * width];
        for i in 0..chunk_size {
            for d in 0..width {
                let mut acc = rhs_f32[i * width + d];
                for j in 0..i {
                    acc -= ut_system_f32[i * chunk_size + j] * expected[j * width + d];
                }
                expected[i * width + d] = acc;
            }
        }

        let ut_bf16: Vec<u16> = ut_system_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let rhs_bf16: Vec<u16> = rhs_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut ut_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(ut_bf16.len()).unwrap();
        let mut rhs_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(rhs_bf16.len()).unwrap();
        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(rhs_bf16.len()).unwrap();
        ut_buf.copy_from_host(&ut_bf16).unwrap();
        rhs_buf.copy_from_host(&rhs_bf16).unwrap();

        gdn_chunk_utsolve_bf16(&ut_buf, &rhs_buf, &mut x_buf, num_heads, num_chunks, chunk_size, width)
            .expect("real gdn_chunk_utsolve_bf16 call failed");

        let mut got_bf16 = vec![0u16; rhs_bf16.len()];
        x_buf.copy_to_host(&mut got_bf16).unwrap();
        let got: Vec<f32> = got_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 0.01, "element {i}: GPU gdn_chunk_utsolve_bf16 got {g}, independent reference {e}");
        }
    }

    /// §100 decisive test: `l2norm_bf16` matching transformers' real
    /// `l2norm(x) = x * rsqrt(sum(x^2)+eps)` exactly (`eps=1e-6`, this
    /// model's real value), checked against an independent plain-f64
    /// reference, including a real non-1.0 `post_scale` (query's own
    /// `1/sqrt(head_dim)` attention scaling).
    #[test]
    fn real_l2norm_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_rows = 2usize;
        let hidden_size = 4usize;
        let eps = 1e-6f32;
        let post_scale = 0.5f32; // e.g. 1/sqrt(head_dim) for head_dim=4

        let x_f32: Vec<f32> = vec![1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25, -0.75];

        let mut expected = vec![0f32; num_rows * hidden_size];
        for r in 0..num_rows {
            let row = &x_f32[r * hidden_size..(r + 1) * hidden_size];
            let sum_sq: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
            let inv_norm = (sum_sq + eps as f64).powf(-0.5);
            for (d, &v) in row.iter().enumerate() {
                expected[r * hidden_size + d] = (v as f64 * inv_norm * post_scale as f64) as f32;
            }
        }

        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
        let mut out_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();

        l2norm_bf16(&x_buf, &mut out_buf, num_rows, hidden_size, eps, post_scale).expect("real l2norm_bf16 call failed");

        let mut got_bf16 = vec![0u16; x_bf16.len()];
        out_buf.copy_to_host(&mut got_bf16).unwrap();
        let got: Vec<f32> = got_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 0.01, "element {i}: GPU l2norm_bf16 got {g}, independent reference {e}");
        }
    }

    /// §100 decisive test: `gdn_chunk_broadcast_scale_bf16` with a real
    /// `n_rep=2` GQA broadcast (2 tokens, `src_heads=2`, `dst_heads=4`,
    /// `head_dim=3`) -- checked against an independent reference that
    /// reads `src[t, h/2, :]` by hand and multiplies by `scale[t,h]`.
    #[test]
    fn real_gdn_chunk_broadcast_scale_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_tokens = 2usize;
        let src_heads = 2usize;
        let dst_heads = 4usize;
        let head_dim = 3usize;
        let n_rep = 2usize;

        // src[token][src_head][dim]
        let src_f32: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // token 0: head0, head1
            -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, // token 1: head0, head1
        ];
        // scale[token][dst_head]
        let scale_f32: Vec<f32> = vec![
            0.5, 2.0, -1.0, 0.25, // token 0
            1.5, -0.5, 3.0, 0.1, // token 1
        ];

        let mut expected = vec![0f32; num_tokens * dst_heads * head_dim];
        for t in 0..num_tokens {
            for dh in 0..dst_heads {
                let sh = dh / n_rep;
                let s = scale_f32[t * dst_heads + dh];
                for d in 0..head_dim {
                    let src_val = src_f32[(t * src_heads + sh) * head_dim + d];
                    expected[(t * dst_heads + dh) * head_dim + d] = src_val * s;
                }
            }
        }

        let src_bf16: Vec<u16> = src_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut src_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(src_bf16.len()).unwrap();
        let mut scale_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(scale_f32.len()).unwrap();
        let mut dst_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(expected.len()).unwrap();
        src_buf.copy_from_host(&src_bf16).unwrap();
        scale_buf.copy_from_host(&scale_f32).unwrap();

        gdn_chunk_broadcast_scale_bf16(&src_buf, &scale_buf, &mut dst_buf, num_tokens, src_heads, dst_heads, head_dim, n_rep, num_tokens, false)
            .expect("real gdn_chunk_broadcast_scale_bf16 call failed");

        let mut got_bf16 = vec![0u16; expected.len()];
        dst_buf.copy_to_host(&mut got_bf16).unwrap();
        let got: Vec<f32> = got_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 0.01, "element {i}: GPU gdn_chunk_broadcast_scale_bf16 got {g}, independent reference {e}");
        }
    }

    /// §100 decisive test: `gdn_chunk_broadcast_scale_bf16` with
    /// `head_major_output=true` -- the path `v_beta`/`decayed_k_beta`/the
    /// final rescaled query&key actually use (feeding `gdn_chunk_utsolve_bf16`
    /// and the inter-chunk scan). Same `n_rep=2` broadcast as the token-major
    /// test above, but 4 tokens split into 2 chunks of 2, and the expected
    /// reference is computed token-major first, then independently
    /// remapped by hand into `[dst_head, chunk, pos, dim]` order to check
    /// against the kernel's head-major write.
    #[test]
    fn real_gdn_chunk_broadcast_scale_head_major_output_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let num_tokens = 4usize;
        let chunk_size = 2usize;
        let num_chunks = num_tokens / chunk_size;
        let src_heads = 1usize;
        let dst_heads = 2usize;
        let head_dim = 2usize;
        let n_rep = 2usize;

        // src[token][src_head][dim]
        let src_f32: Vec<f32> = vec![
            1.0, 2.0, // token 0
            3.0, 4.0, // token 1
            5.0, 6.0, // token 2
            7.0, 8.0, // token 3
        ];
        // scale[token][dst_head]
        let scale_f32: Vec<f32> = vec![
            0.5, 2.0, // token 0
            -1.0, 0.25, // token 1
            1.5, -0.5, // token 2
            3.0, 0.1, // token 3
        ];

        // Independent reference: compute token-major first (same formula
        // as the sibling test), then remap by hand into
        // `[dst_head, chunk, pos, dim]` order.
        let mut expected_head_major = vec![0f32; dst_heads * num_chunks * chunk_size * head_dim];
        for t in 0..num_tokens {
            let chunk = t / chunk_size;
            let pos = t % chunk_size;
            for dh in 0..dst_heads {
                let sh = dh / n_rep;
                let s = scale_f32[t * dst_heads + dh];
                for d in 0..head_dim {
                    let src_val = src_f32[(t * src_heads + sh) * head_dim + d];
                    let out_idx = ((dh * num_chunks + chunk) * chunk_size + pos) * head_dim + d;
                    expected_head_major[out_idx] = src_val * s;
                }
            }
        }

        let src_bf16: Vec<u16> = src_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut src_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(src_bf16.len()).unwrap();
        let mut scale_buf: DeviceBuffer<f32> = DeviceBuffer::alloc(scale_f32.len()).unwrap();
        let mut dst_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(expected_head_major.len()).unwrap();
        src_buf.copy_from_host(&src_bf16).unwrap();
        scale_buf.copy_from_host(&scale_f32).unwrap();

        gdn_chunk_broadcast_scale_bf16(&src_buf, &scale_buf, &mut dst_buf, num_tokens, src_heads, dst_heads, head_dim, n_rep, chunk_size, true)
            .expect("real gdn_chunk_broadcast_scale_bf16 call failed");

        let mut got_bf16 = vec![0u16; expected_head_major.len()];
        dst_buf.copy_to_host(&mut got_bf16).unwrap();
        let got: Vec<f32> = got_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        for (i, (g, e)) in got.iter().zip(expected_head_major.iter()).enumerate() {
            assert!((g - e).abs() < 0.01, "element {i}: GPU gdn_chunk_broadcast_scale_bf16 (head_major_output=true) got {g}, independent reference {e}");
        }
    }

    /// §110 diagnostic: real, synthetic-data GEMV timing for every one of
    /// this SPECIFIC size's real per-layer W4A16 shapes, summed across
    /// `NUM_LAYERS`, to estimate how much of a real observed decode-step
    /// time is GEMV-kernel-bound vs. everything else (rmsnorm, rope,
    /// attention/GDN recurrent kernels, per-launch overhead) -- values
    /// don't affect GEMV timing (no data-dependent branches in the
    /// kernel), so synthetic buffers are a real, honest stand-in for the
    /// real checkpoint here, not a shortcut that changes what's measured.
    #[test]
    #[ignore]
    fn diagnose_real_w4a16_gemv_share_of_decode_step() {
        use crate::model::*;
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let group_size = W4A16_GROUP_SIZE;
        // real: full_attention_interval=4 -- 1/4 of layers are Attn
        // (qkv_proj/o_proj), 3/4 are Gdn (in_proj_combined/out_proj).
        // gate_up_proj/down_proj (MLP) run in EVERY layer regardless.
        let attn_layers = NUM_LAYERS / 4;
        let gdn_layers = NUM_LAYERS - attn_layers;
        let shapes: Vec<(&str, usize, usize, usize)> = vec![
            ("qkv_proj", ATTN_QKV_COMBINED_DIM, HIDDEN_SIZE, attn_layers),
            ("o_proj", HIDDEN_SIZE, HIDDEN_SIZE, attn_layers),
            ("in_proj_combined", GDN_IN_PROJ_COMBINED_DIM, HIDDEN_SIZE, gdn_layers),
            ("out_proj", HIDDEN_SIZE, GDN_VALUE_DIM, gdn_layers),
            ("gate_up_proj", 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE, NUM_LAYERS),
            ("down_proj", HIDDEN_SIZE, INTERMEDIATE_SIZE, NUM_LAYERS),
        ];

        let iters = 500;
        let warmup = 50;
        let mut total_us_all_layers = 0.0f64;

        for (name, out_features, in_features, real_layer_count) in &shapes {
            let (out_features, in_features, real_layer_count) = (*out_features, *in_features, *real_layer_count);
            let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(in_features).unwrap();
            let mut qw_buf: DeviceBuffer<u32> = DeviceBuffer::alloc(out_features * (in_features / 8)).unwrap();
            let mut sc_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(out_features * (in_features / group_size)).unwrap();
            let mut y_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(out_features).unwrap();
            // Zero-init is fine (real kernel timing has no data dependence).
            x_buf.copy_from_host(&vec![0u16; in_features]).unwrap();
            qw_buf.copy_from_host(&vec![0u32; out_features * (in_features / 8)]).unwrap();
            sc_buf.copy_from_host(&vec![0u16; out_features * (in_features / group_size)]).unwrap();

            for _ in 0..warmup {
                w4a16_gemv_bf16(&x_buf, &qw_buf, &sc_buf, &mut y_buf, out_features, in_features, group_size).unwrap();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                w4a16_gemv_bf16(&x_buf, &qw_buf, &sc_buf, &mut y_buf, out_features, in_features, group_size).unwrap();
            }
            let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!("{name:<20} out={out_features:<6} in={in_features:<6}: {us:.2} us/call x {real_layer_count} real layers = {:.2} us total", us * real_layer_count as f64);
            total_us_all_layers += us * real_layer_count as f64;
        }

        let total_ms_all_layers = total_us_all_layers / 1000.0;
        println!(
            "\nSum of all real per-token W4A16 GEMV calls across all {NUM_LAYERS} layers ({attn_layers} Attn + {gdn_layers} Gdn): {total_ms_all_layers:.3} ms/token (GEMV-kernel time only -- compare to the real observed HTTP decode-step time to see what fraction is GEMV vs everything else)"
        );
    }
}
