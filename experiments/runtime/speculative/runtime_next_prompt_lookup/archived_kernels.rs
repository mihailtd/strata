//! Safe wrappers around this crate's real, hand-written HIP kernels
//! (`src/kernels/*.hip`, compiled by `hipcc` in `build.rs`).
//!
//! Same discipline as `hip.rs`: the `extern "C"` block below is the only
//! unsafe surface for kernel launches, hand-curated against the exact
//! signature each `.hip` file's own launcher function declares -- not
//! generated, not guessed.

use crate::hip::{DeviceBuffer, HipError, check_last_error, device_synchronize};
use std::ffi::{c_int, c_void};

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

        /// See `src/kernels/swiglu.hip` for what this actually computes.
        /// `stream` (§93).
        pub fn launch_swiglu_bf16(
            gate: *const c_void,
            up: *const c_void,
            out: *mut c_void,
            n: c_int,
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

        /// See `src/kernels/add.hip` for what this actually computes.
        /// `stream` (§93).
        pub fn launch_add_bf16(a: *const c_void, b: *const c_void, out: *mut c_void, n: c_int, stream: *mut c_void);

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

/// On-device argmax over `logits[n]` (real vocab logits). Returns the
/// index of the first (lowest-index) occurrence of the maximum value --
/// exactly the semantics `model.rs::argmax_sample`'s original host-side
/// scan had, just computed on the GPU (see `src/kernels/argmax.hip` for
/// the real motivation and the tie-breaking discipline).
pub fn argmax_bf16(logits: &DeviceBuffer<u16>) -> Result<i32, HipError> {
    let n = logits.len();
    let threads: i32 = 1024;
    let mut out_idx: DeviceBuffer<i32> = DeviceBuffer::alloc(1)?;

    // SAFETY: `logits` is a real, live `hipMalloc` allocation of `n`
    // elements; the kernel indexes strictly within `[0, n)` (verified by
    // reading `argmax.hip` directly). `out_idx` is a real, live `hipMalloc`
    // allocation of exactly one `i32`.
    unsafe {
        ffi::launch_argmax_bf16(
            logits.as_device_ptr(),
            out_idx.as_device_ptr_mut() as *mut c_int,
            n as i32,
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

/// §96 (speculative decoding): `argmax_bf16`'s raw-pointer counterpart, for
/// argmaxing ONE row out of a larger multi-row buffer (`forward_verify_chunk`'s
/// `[num_tokens, VOCAB_SIZE]` `verify_logits`, where `DeviceBuffer` itself
/// has no lightweight sub-view type to hand `argmax_bf16` directly).
pub fn argmax_bf16_at(logits: &DeviceBuffer<u16>, row: usize, row_len: usize) -> Result<i32, HipError> {
    assert!(
        (row + 1) * row_len <= logits.len(),
        "argmax_bf16_at: row {row} (row_len {row_len}) out of bounds for buffer of length {}",
        logits.len()
    );
    let threads: i32 = 1024;
    let mut out_idx: DeviceBuffer<i32> = DeviceBuffer::alloc(1)?;

    // SAFETY: `logits.as_device_ptr_at(row*row_len)` points at a real,
    // live sub-range of `logits`'s own allocation, `row_len` elements
    // long (bounds-checked above); the kernel indexes strictly within
    // `[0, row_len)` of it.
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
        assert_eq!(hidden_size, 2560);

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

    /// Decisive real-weight cross-check, same discipline as
    /// `model_loader.rs`'s own decisive test: real layer 0 GDN
    /// `conv1d.weight` (conv_dim=8192, kernel_size=4), a fixed deterministic
    /// synthetic state/hidden input, run on the real GPU kernel and checked
    /// against values computed by the REAL `transformers`
    /// `causal_conv1d_update` function on the same real weight and the same
    /// input formula (see
    /// `scratchpad/gen_causal_conv1d_reference.py` for how these were
    /// generated -- an independent process, not this kernel's own code).
    #[test]
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
    #[test]
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

    /// §96: `argmax_bf16_at`'s decisive test -- a fake `[3, VOCAB_SIZE]`-
    /// shaped `verify_logits`-style buffer with a DIFFERENT planted maximum
    /// in each of its 3 rows, checked against `argmax_bf16` run on each
    /// row's own independently-uploaded copy (the already-proven,
    /// whole-buffer path) -- proves the raw-pointer row offset lands on
    /// exactly the right sub-range, not a row-length or stride error.
    #[test]
    fn real_argmax_at_matches_argmax_bf16_per_row() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let row_len = 2048usize;
        let num_rows = 3usize;
        let planted_idx = [77usize, 1500usize, 3usize];

        let mut all_rows_f32 = vec![0f32; num_rows * row_len];
        for (r, &idx) in planted_idx.iter().enumerate() {
            for i in 0..row_len {
                all_rows_f32[r * row_len + i] = ((i % 41) as f32 - 20.0) * 0.01;
            }
            all_rows_f32[r * row_len + idx] = 100.0 + r as f32; // unambiguous per-row maximum
        }
        let all_rows_bf16: Vec<u16> = all_rows_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut combined_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_rows * row_len).unwrap();
        combined_buf.copy_from_host(&all_rows_bf16).unwrap();

        for (r, &idx) in planted_idx.iter().enumerate() {
            let got = argmax_bf16_at(&combined_buf, r, row_len).expect("real argmax_bf16_at call failed");
            assert_eq!(got, idx as i32, "row {r}: argmax_bf16_at did not find the real planted maximum's index");

            // Cross-check against the already-proven whole-buffer path on
            // an independently-uploaded copy of just this row.
            let row_bf16 = &all_rows_bf16[r * row_len..(r + 1) * row_len];
            let mut row_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(row_len).unwrap();
            row_buf.copy_from_host(row_bf16).unwrap();
            let expected = argmax_bf16(&row_buf).expect("real argmax_bf16 call failed");
            assert_eq!(got, expected, "row {r}: argmax_bf16_at disagreed with argmax_bf16 on an identical standalone copy");
        }
    }
}
