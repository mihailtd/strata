//! Safe wrapper around hipBLAS -- this project's first "link a vendor
//! library" feature rather than a hand-written `.hip` kernel. Deliberately
//! different in kind from `hip.rs`/`kernels.rs`: a GEMM is exactly the
//! well-solved-problem case this session's own research (`ECOSYSTEM_NOTES.md`)
//! already argued for linking rather than reimplementing -- the same
//! reasoning that led to using the real `safetensors` crate over hand-
//! rolling a parser in `model_loader.rs`.
//!
//! Same discipline as `hip.rs`: `mod ffi` is the only unsafe surface, hand-
//! curated against `/opt/rocm/include/hipblas/hipblas.h` and
//! `/opt/rocm/include/hipblas-common/hipblas-common.h` and
//! `/opt/rocm/include/hip/library_types.h` (ROCm 7.2, this machine) -- not
//! generated, not guessed.
//!
//! `gemm_bf16_linear` implements exactly `y = x @ W^T` (a PyTorch
//! `nn.Linear`'s forward, no bias -- every real projection in this model's
//! MLP and attention blocks is bias-free), using the standard "row-major
//! via column-major BLAS" trick: BLAS is column-major, PyTorch/safetensors
//! tensors are row-major, and rather than transpose any buffer in memory,
//! the multiplication is expressed with swapped operand order and
//! transpose flags so the SAME row-major bytes are consumed directly. See
//! the doc comment on `gemm_bf16_linear` for the derivation.

use crate::hip::HipError;
use std::ffi::{CStr, c_int, c_void};
use std::fmt;
use std::ptr;

// `pub(crate)`: `model.rs`'s hot decode-loop path calls `hipblasGemmEx`
// directly, skipping `gemm_bf16_linear`'s per-call `device_synchronize()`
// -- same reasoning as `kernels.rs`'s `ffi` module widening.
pub(crate) mod ffi {
    use std::ffi::c_void;

    pub type HipblasHandle = *mut c_void;

    unsafe extern "C" {
        pub fn hipblasCreate(handle: *mut HipblasHandle) -> i32;
        pub fn hipblasDestroy(handle: HipblasHandle) -> i32;
        pub fn hipblasStatusToString(status: i32) -> *const std::ffi::c_char;
        /// §93: binds a hipBLAS handle to an explicit HIP stream -- needed
        /// so a `hipblasGemmEx` call issued through this handle can be
        /// captured into a HIP graph (capture only records operations on
        /// the stream actually being captured; the handle's default stream
        /// is the null/legacy stream otherwise, which cannot be captured).
        pub fn hipblasSetStream(handle: HipblasHandle, stream: *mut c_void) -> i32;

        /// See `hipblas.h`'s own doc comment on `hipblasGemmEx` -- this is
        /// the generic mixed-datatype GEMM, used here with
        /// `aType=bType=cType=HIP_R_16BF` (bf16 I/O) and
        /// `computeType=HIPBLAS_COMPUTE_32F` (fp32 accumulation), matching
        /// this project's established "elementwise/reduction math in fp32,
        /// round to bf16 only at the boundary" convention.
        #[allow(clippy::too_many_arguments)]
        pub fn hipblasGemmEx(
            handle: HipblasHandle,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const c_void,
            a: *const c_void,
            a_type: i32,
            lda: i32,
            b: *const c_void,
            b_type: i32,
            ldb: i32,
            beta: *const c_void,
            c: *mut c_void,
            c_type: i32,
            ldc: i32,
            compute_type: i32,
            algo: i32,
        ) -> i32;

        #[allow(clippy::too_many_arguments)]
        pub fn hipblasGemmStridedBatchedEx(
            handle: HipblasHandle,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const c_void,
            a: *const c_void,
            a_type: i32,
            lda: i32,
            stride_a: i64,
            b: *const c_void,
            b_type: i32,
            ldb: i32,
            stride_b: i64,
            beta: *const c_void,
            c: *mut c_void,
            c_type: i32,
            ldc: i32,
            stride_c: i64,
            batch_count: i32,
            compute_type: i32,
            algo: i32,
        ) -> i32;
    }

    // hipblasOperation_t (hipblas-common.h)
    pub const HIPBLAS_OP_N: i32 = 111;
    pub const HIPBLAS_OP_T: i32 = 112;
    // hipDataType (hip/library_types.h)
    pub const HIP_R_16BF: i32 = 14;
    // hipblasComputeType_t (hipblas-common.h)
    pub const HIPBLAS_COMPUTE_32F: i32 = 2;
    // hipblasGemmAlgo_t (hipblas.h)
    pub const HIPBLAS_GEMM_DEFAULT: i32 = 160;
    // hipblasStatus_t (hipblas-common.h)
    pub const HIPBLAS_STATUS_SUCCESS: i32 = 0;
}

/// A hipBLAS error, carrying both the numeric status code and the string
/// hipBLAS itself provides for it (via `hipblasStatusToString`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlasError {
    pub code: i32,
}

impl BlasError {
    fn from_status(code: i32) -> Result<(), BlasError> {
        if code == ffi::HIPBLAS_STATUS_SUCCESS {
            Ok(())
        } else {
            Err(BlasError { code })
        }
    }

    pub fn message(&self) -> String {
        // SAFETY: hipblasStatusToString returns a pointer to a static,
        // null-terminated string for any hipblasStatus_t value, including
        // out-of-range ones.
        unsafe {
            let ptr = ffi::hipblasStatusToString(self.code);
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

impl fmt::Display for BlasError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "hipBLAS error {}: {}", self.code, self.message())
    }
}

impl std::error::Error for BlasError {}

/// Either a `HipError` (allocation/launch-adjacent) or a `BlasError`
/// (hipBLAS-specific) -- `gemm_bf16_linear` can fail either way, and
/// callers generally want to handle both as "the GEMM failed" rather than
/// match on which library raised it.
#[derive(Debug)]
pub enum BlasOpError {
    Hip(HipError),
    Blas(BlasError),
}

impl fmt::Display for BlasOpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlasOpError::Hip(e) => write!(f, "{e}"),
            BlasOpError::Blas(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for BlasOpError {}

impl From<BlasError> for BlasOpError {
    fn from(e: BlasError) -> Self {
        BlasOpError::Blas(e)
    }
}

/// Owns one real hipBLAS library context (`hipblasHandle_t`). `Drop` calls
/// `hipblasDestroy`, same deterministic-cleanup discipline as
/// `hip::DeviceBuffer`'s `hipFree`.
pub struct BlasHandle {
    raw: ffi::HipblasHandle,
}

// SAFETY: a hipBLAS handle is not tied to a host thread; hipBLAS itself is
// safe to call concurrently from different threads against different
// handles (the same assumption this crate already makes for raw HIP calls).
unsafe impl Send for BlasHandle {}
unsafe impl Sync for BlasHandle {}

impl BlasHandle {
    /// Creates a real hipBLAS library context bound to the currently active
    /// HIP device (`hip::set_device` must have been called first, same as
    /// every other GPU operation in this crate).
    pub fn create() -> Result<Self, BlasError> {
        let mut raw: ffi::HipblasHandle = ptr::null_mut();
        // SAFETY: `raw` is a valid, aligned, writable pointer for the
        // duration of this call; hipBLAS writes exactly one handle through
        // it, or leaves it untouched and returns a nonzero status (checked
        // immediately below).
        let status = unsafe { ffi::hipblasCreate(&mut raw) };
        BlasError::from_status(status)?;
        debug_assert!(
            !raw.is_null(),
            "hipblasCreate returned success with a null handle"
        );
        Ok(BlasHandle { raw })
    }

    /// The raw hipBLAS handle -- for `model.rs`'s hot decode-loop path,
    /// which calls `ffi::hipblasGemmEx` directly (skipping
    /// `gemm_bf16_linear`'s per-call sync). `pub(crate)`, not `pub`: still
    /// not an escape hatch for code outside this crate's own hot path.
    pub(crate) fn raw(&self) -> ffi::HipblasHandle {
        self.raw
    }

    /// Binds this handle to an explicit stream, so subsequent GEMM calls
    /// through it are issued on that stream instead of the null/legacy
    /// stream -- required before this handle's calls can be captured into
    /// a HIP graph (see `crate::hip::begin_capture`).
    pub fn set_stream(&self, stream: &crate::hip::Stream) -> Result<(), BlasError> {
        // SAFETY: `self.raw` is a live hipBLAS handle; `stream.raw()` is a
        // live HIP stream handle for the duration of this call.
        let status = unsafe { ffi::hipblasSetStream(self.raw, stream.raw()) };
        BlasError::from_status(status)
    }
}

impl Drop for BlasHandle {
    fn drop(&mut self) {
        // SAFETY: `self.raw` was returned by a successful `hipblasCreate`
        // above and has not been destroyed elsewhere -- `BlasHandle` is the
        // sole owner (no `Clone` impl) and this is the only `Drop`.
        unsafe {
            ffi::hipblasDestroy(self.raw);
        }
    }
}

/// `y = x @ w^T` (a bias-free `nn.Linear` forward), bf16 in/out, fp32
/// accumulation, computed by real hipBLAS on the real GPU.
///
/// `x` holds `rows * in_features` bf16 elements, row-major (PyTorch's own
/// layout for an activation tensor `[rows, in_features]`). `w` holds
/// `out_features * in_features` bf16 elements, row-major (PyTorch's own
/// `nn.Linear.weight` layout, `[out_features, in_features]` -- e.g. the
/// real `down_proj.weight` loaded by `model_loader.rs`). `y` holds
/// `rows * out_features` bf16 elements, row-major.
///
/// DERIVATION (BLAS is column-major; these buffers are row-major, and
/// nothing here is physically transposed in memory -- only interpreted
/// differently, which is what makes this fast):
///
/// A row-major buffer of logical shape `[p, q]`, read as a column-major
/// buffer of shape `[q, p]`, represents exactly the mathematical transpose
/// of the original matrix (same bytes, different indexing convention).
/// Applying that fact to both `x` (`[rows, in]`) and `w` (`[out, in]`) and
/// solving for the GEMM call that reproduces `y = x @ w^T` in `y`'s own
/// row-major bytes gives:
///
/// ```text
/// hipblasGemmEx(handle,
///     transA = OP_T, transB = OP_N,
///     M = out_features, N = rows, K = in_features,
///     alpha=1, A = w, lda = in_features,
///              B = x, ldb = in_features,
///     beta=0,  C = y, ldc = out_features,
///     computeType = COMPUTE_32F, algo = GEMM_DEFAULT)
/// ```
///
/// This is the same operand-order/transpose-flag pattern any row-major
/// wrapper around a column-major BLAS uses for `y = x @ W^T` (e.g. how
/// `F.linear` itself ultimately calls into hipBLAS/rocBLAS) -- chosen
/// deliberately to make this comparison apples-to-apples: both sides of
/// the upcoming benchmark hit the same underlying vendor GEMM.
#[allow(clippy::too_many_arguments)]
pub fn gemm_bf16_linear(
    handle: &BlasHandle,
    x: &crate::hip::DeviceBuffer<u16>,
    w: &crate::hip::DeviceBuffer<u16>,
    y: &mut crate::hip::DeviceBuffer<u16>,
    rows: usize,
    in_features: usize,
    out_features: usize,
) -> Result<(), BlasOpError> {
    assert_eq!(
        x.len(),
        rows * in_features,
        "x length must be rows * in_features"
    );
    assert_eq!(
        w.len(),
        out_features * in_features,
        "w length must be out_features * in_features"
    );
    assert_eq!(
        y.len(),
        rows * out_features,
        "y length must be rows * out_features"
    );

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    // SAFETY: `x`/`w` are real, live `hipMalloc` allocations of at least
    // the asserted element counts (read-only to hipBLAS here); `y` is
    // likewise real, live, and distinct, sized for the full `[rows,
    // out_features]` result hipBLAS writes. `alpha`/`beta` are valid
    // `f32`s on the host, matching `HIPBLAS_COMPUTE_32F`'s expected scalar
    // type. `handle.raw` is a live hipBLAS context bound to the currently
    // active device (the caller's responsibility, same as every other GPU
    // call in this crate).
    let status = unsafe {
        ffi::hipblasGemmEx(
            handle.raw,
            ffi::HIPBLAS_OP_T,
            ffi::HIPBLAS_OP_N,
            out_features as c_int,
            rows as c_int,
            in_features as c_int,
            &alpha as *const f32 as *const c_void,
            w.as_device_ptr(),
            ffi::HIP_R_16BF,
            in_features as c_int,
            x.as_device_ptr(),
            ffi::HIP_R_16BF,
            in_features as c_int,
            &beta as *const f32 as *const c_void,
            y.as_device_ptr_mut() as *mut c_void,
            ffi::HIP_R_16BF,
            out_features as c_int,
            ffi::HIPBLAS_COMPUTE_32F,
            ffi::HIPBLAS_GEMM_DEFAULT,
        )
    };
    BlasError::from_status(status)?;
    crate::hip::device_synchronize().map_err(BlasOpError::Hip)
}

/// §99 (batched-prefill causal attention): `S = Q @ K^T`, bf16 in/out, fp32
/// accumulation -- the QK^T half of real matrix-core attention, replacing
/// the naive per-token scalar-loop kernel (`attention.hip`) for the
/// prefill case where more than one query row is available at once (see
/// `docs/DECISIONS.md` §99 for the real per-kernel profiling that found
/// this, not attention's launch count, as the actual lever).
///
/// UNLIKE `gemm_bf16_linear`, `q`/`k`/`s` are explicit-stride VIEWS, not
/// necessarily tightly-packed buffers: `q_ld`/`k_ld`/`s_ld` are the real
/// element stride between consecutive logical rows in each buffer's
/// underlying memory, which may be wider than the logical row length
/// itself (e.g. `q` is a `[T, num_heads*head_dim]` buffer and this call
/// only wants ONE head's `head_dim`-wide sub-range per row -- passing
/// `q_ld = num_heads*head_dim` while the logical shape used for the GEMM
/// math is `[t, head_dim]` reads exactly that sub-range, no physical
/// extraction needed, using BLAS's own leading-dimension mechanism the
/// same way `gemm_bf16_linear`'s row-major-via-column-major derivation
/// already relies on `lda`/`ldb`/`ldc` to mean "real memory stride," not
/// "logical dimension").
///
/// Derivation: identical to `gemm_bf16_linear`'s own (`y = x @ w^T` via
/// `OP_T, OP_N`), with `x = q` (`rows = t`, `in_features = head_dim`),
/// `w = k` (`out_features = kv_len`), `y = s` -- just with caller-supplied
/// strides instead of assuming `in_features`/`out_features` themselves.
/// `scale` (real attention's `1/sqrt(head_dim)`) is folded into
/// `hipblasGemmEx`'s own `alpha` scalar -- no separate elementwise scaling
/// pass over `q` needed.
#[allow(clippy::too_many_arguments)]
pub fn gemm_qkt_bf16(
    handle: &BlasHandle,
    q: &crate::hip::DeviceBuffer<u16>,
    q_offset: usize,
    q_ld: usize,
    k: &crate::hip::DeviceBuffer<u16>,
    k_offset: usize,
    s: &mut crate::hip::DeviceBuffer<u16>,
    s_offset: usize,
    s_ld: usize,
    t: usize,
    head_dim: usize,
    kv_len: usize,
    scale: f32,
) -> Result<(), BlasOpError> {
    let alpha: f32 = scale;
    let beta: f32 = 0.0;
    // SAFETY: `q`/`k`/`s` are real, live `hipMalloc` allocations; the
    // caller guarantees `q_offset + (t-1)*q_ld + head_dim <= q.len()`,
    // `k_offset + kv_len*head_dim <= k.len()` (`k`'s own real per-head
    // slice is tightly packed starting at `k_offset` -- e.g. one head's
    // range within a head-major KV cache), and
    // `s_offset + (t-1)*s_ld + kv_len <= s.len()` (checked by the two
    // decisive tests below against independently-computed references, not
    // just asserted here).
    let status = unsafe {
        ffi::hipblasGemmEx(
            handle.raw,
            ffi::HIPBLAS_OP_T,
            ffi::HIPBLAS_OP_N,
            kv_len as c_int,
            t as c_int,
            head_dim as c_int,
            &alpha as *const f32 as *const c_void,
            k.as_device_ptr_at(k_offset),
            ffi::HIP_R_16BF,
            head_dim as c_int,
            q.as_device_ptr_at(q_offset),
            ffi::HIP_R_16BF,
            q_ld as c_int,
            &beta as *const f32 as *const c_void,
            s.as_device_ptr_at_mut(s_offset) as *mut c_void,
            ffi::HIP_R_16BF,
            s_ld as c_int,
            ffi::HIPBLAS_COMPUTE_32F,
            ffi::HIPBLAS_GEMM_DEFAULT,
        )
    };
    BlasError::from_status(status)?;
    crate::hip::device_synchronize().map_err(BlasOpError::Hip)
}

/// §100 (chunked GDN prefill): `Y = A^T @ B` -- a THIRD real transpose
/// pattern, needed for the delta-rule's real state update
/// (`state_delta = key^T @ v_new`, `A=key [chunk,head_dim]`,
/// `B=v_new [chunk,head_dim]`, `Y=[head_dim,head_dim]`). Derivation
/// validated by re-deriving `gemm_pv_bf16`'s own already-proven call from
/// the same general method before trusting it for a new shape (see this
/// function's own derivation below): reading a row-major buffer's bytes
/// as column-major ALWAYS gives that matrix's real transpose "for free"
/// (a `hipMalloc` buffer has no concept of major-order; only the two
/// counted dimensions plus a stride do); an EXTRA `OP_T` flag on top of
/// that then undoes it, handing BLAS back the plain matrix.
///
/// Wanting `Y[K,N]_rowmajor = A^T[K,M] @ B[M,N]` (`A` real shape `[M,K]`,
/// `B` real shape `[M,N]`, both row-major): read `Y` as column-major
/// (`ldc=N`) to get `Y^T[N,K]`; BLAS's own output is therefore an `[N,K]`
/// matrix. Solving `Y^T[n,k] = Y[k,n] = sum_m A[m,k]*B[m,n]` for a plain
/// column-major `C = op(A_blas) @ op(B_blas)`: `A_blas = B` (`OP_N` --
/// reading `B`'s row-major bytes as column-major already gives `B^T[n,m]
/// = B[m,n]`, exactly the factor needed), `B_blas = A` (`OP_T` -- reading
/// `A`'s row-major bytes as column-major gives `A^T[k,m]`, and applying
/// `OP_T` undoes that back to plain `A[m,k]`, the factor needed). `M_blas
/// = N`, `N_blas = K`, `K_blas = M`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_atb_bf16(
    handle: &BlasHandle,
    a: &crate::hip::DeviceBuffer<u16>,
    a_offset: usize,
    a_ld: usize,
    b: &crate::hip::DeviceBuffer<u16>,
    b_offset: usize,
    b_ld: usize,
    y: &mut crate::hip::DeviceBuffer<u16>,
    y_offset: usize,
    y_ld: usize,
    m: usize,
    k: usize,
    n: usize,
    beta: f32,
) -> Result<(), BlasOpError> {
    let alpha: f32 = 1.0;
    // §100: `beta` is a real hipBLAS accumulate scalar -- `Y_new =
    // A^T@B + beta*Y_old`, reading `Y`'s own pre-call contents through
    // the SAME pointer it writes to (a real, well-defined GEMM pattern,
    // not a hazard: `Y` is never also `A` or `B`). Used by the chunked
    // GDN recurrent-state update (`state = state*chunk_decay +
    // key^T@v_new`, `beta = chunk_decay`) -- ordinary callers pass `0.0`
    // for a ordinary fresh write, matching every other GEMM primitive's
    // default.
    // SAFETY: `a`/`b`/`y` are real, live `hipMalloc` allocations; bounds
    // proven by the decisive test below (independent reference + a
    // strided-view case).
    let status = unsafe {
        ffi::hipblasGemmEx(
            handle.raw,
            ffi::HIPBLAS_OP_N,
            ffi::HIPBLAS_OP_T,
            n as c_int,
            k as c_int,
            m as c_int,
            &alpha as *const f32 as *const c_void,
            b.as_device_ptr_at(b_offset),
            ffi::HIP_R_16BF,
            b_ld as c_int,
            a.as_device_ptr_at(a_offset),
            ffi::HIP_R_16BF,
            a_ld as c_int,
            &beta as *const f32 as *const c_void,
            y.as_device_ptr_at_mut(y_offset) as *mut c_void,
            ffi::HIP_R_16BF,
            y_ld as c_int,
            ffi::HIPBLAS_COMPUTE_32F,
            ffi::HIPBLAS_GEMM_DEFAULT,
        )
    };
    BlasError::from_status(status)?;
    crate::hip::device_synchronize().map_err(BlasOpError::Hip)
}

/// §99: `O = P @ V` (NOT `P @ V^T` -- a genuinely different transpose
/// pattern from `gemm_bf16_linear`/`gemm_qkt_bf16`'s `x @ w^T`), the AV
/// half of real matrix-core attention. `p`/`v`/`o` are explicit-stride
/// views, same reasoning as `gemm_qkt_bf16`.
///
/// Derivation (BLAS is column-major; `P`/`V`/`O` are row-major bytes):
/// reading `V`'s row-major `[kv_len, head_dim]` bytes as column-major with
/// `lda = v_ld` gives exactly `V^T` (shape `[head_dim, kv_len]`); reading
/// `P`'s row-major `[t, kv_len]` bytes as column-major with `ldb = p_ld`
/// gives exactly `P^T` (shape `[kv_len, t]`). A plain column-major
/// `C = A @ B` (both `OP_N`) with `A = V^T`, `B = P^T` computes
/// `C[d,row] = sum_j V^T[d,j] * P^T[j,row] = sum_j V[j,d] * P[row,j]`,
/// which written into `O`'s row-major bytes via `ldc = o_ld` is exactly
/// `O[row,d] = sum_j P[row,j] * V[j,d]` -- `O = P @ V`.
#[allow(clippy::too_many_arguments)]
pub fn gemm_pv_bf16(
    handle: &BlasHandle,
    p: &crate::hip::DeviceBuffer<u16>,
    p_offset: usize,
    p_ld: usize,
    v: &crate::hip::DeviceBuffer<u16>,
    v_offset: usize,
    o: &mut crate::hip::DeviceBuffer<u16>,
    o_offset: usize,
    o_ld: usize,
    t: usize,
    kv_len: usize,
    head_dim: usize,
    alpha: f32,
    beta: f32,
) -> Result<(), BlasOpError> {
    // §100: `alpha` (real hipBLAS scale scalar) lets a caller NEGATE the
    // product before it lands -- needed for the chunked GDN state
    // correction (`v_new = new_values - k_cumdecay @ last_recurrent_state`:
    // pre-fill `o` with `new_values`, then call with `p=k_cumdecay`,
    // `v=last_recurrent_state`, `alpha=-1.0`, `beta=1.0` to SUBTRACT the
    // correction in place). Ordinary callers pass `1.0` (an ordinary
    // positive product), matching every other GEMM primitive's default.
    // `beta` (real hipBLAS accumulate scalar) lets a caller ACCUMULATE
    // into `o` instead of overwriting it -- needed for the chunked GDN
    // recurrent output (`out = inter_chunk_attn + intra_chunk_attn@v_new`:
    // write `inter_chunk_attn` with `beta=0`, then accumulate
    // `intra_chunk_attn@v_new` into the SAME buffer with `beta=1.0`).
    // Ordinary callers (§99's attention) pass `0.0`.
    // SAFETY: same reasoning as `gemm_qkt_bf16` -- real, live allocations
    // (`v_offset + kv_len*head_dim <= v.len()`, `v`'s own real per-head
    // slice tightly packed starting at `v_offset`), bounds proven by the
    // decisive tests below.
    let status = unsafe {
        ffi::hipblasGemmEx(
            handle.raw,
            ffi::HIPBLAS_OP_N,
            ffi::HIPBLAS_OP_N,
            head_dim as c_int,
            t as c_int,
            kv_len as c_int,
            &alpha as *const f32 as *const c_void,
            v.as_device_ptr_at(v_offset),
            ffi::HIP_R_16BF,
            head_dim as c_int,
            p.as_device_ptr_at(p_offset),
            ffi::HIP_R_16BF,
            p_ld as c_int,
            &beta as *const f32 as *const c_void,
            o.as_device_ptr_at_mut(o_offset) as *mut c_void,
            ffi::HIP_R_16BF,
            o_ld as c_int,
            ffi::HIPBLAS_COMPUTE_32F,
            ffi::HIPBLAS_GEMM_DEFAULT,
        )
    };
    BlasError::from_status(status)?;
    crate::hip::device_synchronize().map_err(BlasOpError::Hip)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hip::{self, DeviceBuffer};

    fn f32_to_bf16(v: f32) -> u16 {
        let bits = v.to_bits();
        let lsb = (bits >> 16) & 1;
        let rounded = bits.wrapping_add(0x7fff + lsb);
        (rounded >> 16) as u16
    }

    fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    /// §99 decisive test: `gemm_qkt_bf16` computing `S = Q @ K^T` where `Q`
    /// is a STRIDED VIEW into a wider buffer (`q_ld > head_dim`) -- the
    /// exact real usage pattern (`attn_query_roped`'s `[T, num_heads*head_dim]`
    /// layout, extracting one head's `head_dim`-wide sub-range per row
    /// without physically copying it). `T=2` query rows, `head_dim=3`,
    /// `kv_len=4`, `Q` lives inside a `[2, 5]` buffer (2 extra "other head"
    /// columns per row, real stride 5 not 3) at column offset 1, `K` is a
    /// tightly-packed `[4,3]` buffer. Checked against an INDEPENDENT plain
    /// f32 triple-loop reference that reads `Q`'s strided sub-range by
    /// hand, not by calling this GEMM helper's own logic.
    #[test]
    fn real_gemm_qkt_with_strided_q_view_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let t = 2usize;
        let head_dim = 3usize;
        let kv_len = 4usize;
        let q_ld = 5usize; // real row stride wider than head_dim
        let q_offset = 1usize; // this head's sub-range starts at column 1

        // Q lives inside a [2, 5] buffer; columns [1,4) of each row are this
        // head's real Q values, columns 0 and 4 are OTHER heads' data this
        // call must never read.
        let q_full_f32: Vec<f32> = vec![
            999.0, 1.0, -2.0, 0.5, 999.0, // row 0: real Q = [1.0, -2.0, 0.5]
            999.0, 3.0, -0.5, 2.0, 999.0, // row 1: real Q = [3.0, -0.5, 2.0]
        ];
        let k_f32: Vec<f32> = vec![
            0.2, 0.1, -0.3, // key 0
            -0.1, 0.4, 0.2, // key 1
            0.5, -0.2, 0.1, // key 2
            0.3, 0.3, -0.1, // key 3
        ];

        // A real, non-1.0 scale (matching real attention's own
        // `1/sqrt(head_dim)`) -- the exact mechanism `attn_layer_forward_prefill`
        // relies on to avoid a separate elementwise scaling pass over Q.
        let scale = 0.5f32;

        // Independent reference using the REAL (strided) Q sub-range.
        let q_rows: Vec<Vec<f32>> = (0..t)
            .map(|r| q_full_f32[r * q_ld + q_offset..r * q_ld + q_offset + head_dim].to_vec())
            .collect();
        let mut expected = vec![0f32; t * kv_len];
        for r in 0..t {
            for j in 0..kv_len {
                let mut acc = 0f32;
                for d in 0..head_dim {
                    acc += q_rows[r][d] * k_f32[j * head_dim + d];
                }
                expected[r * kv_len + j] = acc * scale;
            }
        }

        let q_bf16: Vec<u16> = q_full_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let k_bf16: Vec<u16> = k_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut q_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(q_bf16.len()).unwrap();
        let mut k_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(k_bf16.len()).unwrap();
        let mut s_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(t * kv_len).unwrap();
        q_buf.copy_from_host(&q_bf16).unwrap();
        k_buf.copy_from_host(&k_bf16).unwrap();

        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        gemm_qkt_bf16(&handle, &q_buf, q_offset, q_ld, &k_buf, 0, &mut s_buf, 0, kv_len, t, head_dim, kv_len, scale)
            .expect("real gemm_qkt_bf16 call failed");

        let mut s_bf16 = vec![0u16; t * kv_len];
        s_buf.copy_to_host(&mut s_bf16).unwrap();
        let got: Vec<f32> = s_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 0.02, "element {i}: GPU gemm_qkt_bf16 got {g}, independent reference {e}");
        }
    }

    /// §99 decisive test: `gemm_pv_bf16` computing `O = P @ V` (NOT
    /// `P @ V^T`) with `O` written into a STRIDED sub-range of a wider
    /// buffer (`o_ld > head_dim`) -- the real usage pattern (writing one
    /// head's AV output directly into its slice of `attn_out`'s
    /// `[T, num_heads*head_dim]` buffer, no separate per-head output
    /// buffer or copy). Checked against an independent plain f32
    /// triple-loop reference.
    #[test]
    fn real_gemm_pv_with_strided_output_view_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let t = 2usize;
        let kv_len = 3usize;
        let head_dim = 2usize;
        let o_ld = 4usize; // real row stride wider than head_dim
        let o_offset = 2usize; // this head's sub-range starts at column 2

        let p_f32: Vec<f32> = vec![
            0.2, 0.5, 0.3, // row 0 (already-normalized attention weights)
            0.1, 0.1, 0.8, // row 1
        ];
        let v_f32: Vec<f32> = vec![
            1.0, -1.0, // value 0
            2.0, 0.5, // value 1
            -0.5, 3.0, // value 2
        ];

        let mut expected_head = vec![0f32; t * head_dim];
        for r in 0..t {
            for d in 0..head_dim {
                let mut acc = 0f32;
                for j in 0..kv_len {
                    acc += p_f32[r * kv_len + j] * v_f32[j * head_dim + d];
                }
                expected_head[r * head_dim + d] = acc;
            }
        }

        let p_bf16: Vec<u16> = p_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let v_bf16: Vec<u16> = v_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut p_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(p_bf16.len()).unwrap();
        let mut v_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(v_bf16.len()).unwrap();
        // O lives inside a [2, 4] buffer; columns [2,4) of each row are
        // this head's real output, columns 0-1 are ANOTHER head's data
        // that must survive this call untouched.
        let o_sentinel: f32 = -777.0;
        let o_full_f32: Vec<f32> = vec![o_sentinel, o_sentinel, 0.0, 0.0, o_sentinel, o_sentinel, 0.0, 0.0];
        let o_full_bf16: Vec<u16> = o_full_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut o_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(o_full_bf16.len()).unwrap();
        p_buf.copy_from_host(&p_bf16).unwrap();
        v_buf.copy_from_host(&v_bf16).unwrap();
        o_buf.copy_from_host(&o_full_bf16).unwrap();

        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        gemm_pv_bf16(&handle, &p_buf, 0, kv_len, &v_buf, 0, &mut o_buf, o_offset, o_ld, t, kv_len, head_dim, 1.0, 0.0)
            .expect("real gemm_pv_bf16 call failed");

        let mut o_full_got_bf16 = vec![0u16; o_full_bf16.len()];
        o_buf.copy_to_host(&mut o_full_got_bf16).unwrap();
        let o_full_got: Vec<f32> = o_full_got_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        // This head's real output landed at the right strided offset.
        for r in 0..t {
            for d in 0..head_dim {
                let g = o_full_got[r * o_ld + o_offset + d];
                let e = expected_head[r * head_dim + d];
                assert!((g - e).abs() < 0.02, "row {r} dim {d}: GPU gemm_pv_bf16 got {g}, independent reference {e}");
            }
        }
        // The OTHER head's untouched columns (0,1 of each row) must be
        // BIT-IDENTICAL to what was uploaded (an exact check, not a
        // tolerance -- a real device write anywhere in that range should
        // never happen, so the bytes must be untouched, not merely
        // "close"; a numeric tolerance here would also be the wrong tool
        // since bf16's absolute precision at this sentinel's magnitude is
        // itself several units, unrelated to whether a write occurred).
        for r in 0..t {
            for c in 0..o_offset {
                let idx = r * o_ld + c;
                assert_eq!(
                    o_full_got_bf16[idx], o_full_bf16[idx],
                    "row {r} col {c}: neighboring head's data was clobbered (got bits {:#06x}, expected untouched bits {:#06x} = {})",
                    o_full_got_bf16[idx], o_full_bf16[idx], o_sentinel
                );
            }
        }
    }

    /// §100 decisive test: `gemm_atb_bf16` computing `Y = A^T @ B` (the
    /// THIRD real transpose pattern this crate needed, beyond
    /// `gemm_bf16_linear`/`gemm_qkt_bf16`'s `x@w^T` and `gemm_pv_bf16`'s
    /// plain `A@B`) -- checked against an independent plain f32
    /// triple-loop reference, with a strided view on BOTH `a` and the
    /// output `y` (the real usage pattern: `key^T @ v_new` reading a
    /// slice of a wider per-token buffer, writing into a slice of a
    /// wider per-head state buffer).
    #[test]
    fn real_gemm_atb_with_strided_views_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let m = 3usize; // contraction dim (e.g. chunk_size)
        let k = 2usize; // A's real "column" dim (e.g. head_dim)
        let n = 4usize; // B's real "column" dim (e.g. head_dim)
        let a_ld = 5usize; // A lives inside a wider [m, 5] buffer, real data at columns [1,3)
        let a_offset = 1usize;

        // A: real logical shape [m, k], strided inside a [m, a_ld] buffer.
        let a_full_f32: Vec<f32> = vec![
            999.0, 1.0, 2.0, 999.0, 999.0, // row 0
            999.0, 3.0, -1.0, 999.0, 999.0, // row 1
            999.0, 0.5, 4.0, 999.0, 999.0, // row 2
        ];
        // B: real logical shape [m, n], tightly packed.
        let b_f32: Vec<f32> = vec![
            0.1, 0.2, 0.3, 0.4, // row 0
            -0.5, 0.6, -0.7, 0.8, // row 1
            1.0, -1.0, 0.5, -0.5, // row 2
        ];

        let a_rows: Vec<Vec<f32>> = (0..m).map(|r| a_full_f32[r * a_ld + a_offset..r * a_ld + a_offset + k].to_vec()).collect();
        let mut expected = vec![0f32; k * n];
        for ki in 0..k {
            for ni in 0..n {
                let mut acc = 0f32;
                for mi in 0..m {
                    acc += a_rows[mi][ki] * b_f32[mi * n + ni];
                }
                expected[ki * n + ni] = acc;
            }
        }

        let a_bf16: Vec<u16> = a_full_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let b_bf16: Vec<u16> = b_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut a_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(a_bf16.len()).unwrap();
        let mut b_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(b_bf16.len()).unwrap();

        // Y lives inside a wider [k, 6] buffer at column offset 2, same
        // "don't clobber a neighbor" check as gemm_pv's own test.
        let y_ld = 6usize;
        let y_offset = 2usize;
        let y_sentinel: f32 = -42.0;
        let mut y_full_f32 = vec![y_sentinel; k * y_ld];
        for row in 0..k {
            for col in 0..n {
                y_full_f32[row * y_ld + y_offset + col] = 0.0;
            }
        }
        let y_full_bf16: Vec<u16> = y_full_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let mut y_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(y_full_bf16.len()).unwrap();
        a_buf.copy_from_host(&a_bf16).unwrap();
        b_buf.copy_from_host(&b_bf16).unwrap();
        y_buf.copy_from_host(&y_full_bf16).unwrap();

        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        gemm_atb_bf16(&handle, &a_buf, a_offset, a_ld, &b_buf, 0, n, &mut y_buf, y_offset, y_ld, m, k, n, 0.0)
            .expect("real gemm_atb_bf16 call failed");

        let mut y_full_got_bf16 = vec![0u16; y_full_bf16.len()];
        y_buf.copy_to_host(&mut y_full_got_bf16).unwrap();
        let y_full_got: Vec<f32> = y_full_got_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        for row in 0..k {
            for col in 0..n {
                let g = y_full_got[row * y_ld + y_offset + col];
                let e = expected[row * n + col];
                assert!((g - e).abs() < 0.02, "row {row} col {col}: GPU gemm_atb_bf16 got {g}, independent reference {e}");
            }
        }
        // Neighboring columns must be untouched (bit-exact).
        for row in 0..k {
            for col in 0..y_offset {
                let idx = row * y_ld + col;
                assert_eq!(
                    y_full_got_bf16[idx], y_full_bf16[idx],
                    "row {row} col {col}: neighboring data was clobbered (got bits {:#06x}, expected untouched bits {:#06x})",
                    y_full_got_bf16[idx], y_full_bf16[idx]
                );
            }
        }
    }

    /// Real correctness test: a small, fixed `x`/`w`, GEMM computed on the
    /// real GPU via real hipBLAS, checked against a reference computed
    /// INDEPENDENTLY here in plain f32 Rust (a direct triple-loop matmul,
    /// not calling hipBLAS or any BLAS routine) -- same discipline as
    /// every other kernel's correctness test in this crate.
    #[test]
    fn real_gemm_matches_independently_computed_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let rows = 2usize;
        let in_features = 4usize;
        let out_features = 3usize;

        // x: [rows, in_features], row-major
        let x_f32: Vec<f32> = vec![
            1.0, -2.0, 0.5, 3.0, // row 0
            -1.0, 2.0, 0.25, -0.75, // row 1
        ];
        // w: [out_features, in_features], row-major (nn.Linear layout)
        let w_f32: Vec<f32> = vec![
            0.1, 0.2, -0.1, 0.05, // out 0
            -0.2, 0.3, 0.1, -0.05, // out 1
            0.0, -0.1, 0.2, 0.15, // out 2
        ];

        // Independent reference: y[r][o] = sum_i x[r][i] * w[o][i]
        let mut expected = vec![0f32; rows * out_features];
        for r in 0..rows {
            for o in 0..out_features {
                let mut acc = 0f32;
                for i in 0..in_features {
                    acc += x_f32[r * in_features + i] * w_f32[o * in_features + i];
                }
                expected[r * out_features + o] = acc;
            }
        }

        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let w_bf16: Vec<u16> = w_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
        let mut w_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(w_bf16.len()).unwrap();
        let mut y_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(rows * out_features).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();
        w_buf.copy_from_host(&w_bf16).unwrap();

        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        gemm_bf16_linear(
            &handle,
            &x_buf,
            &w_buf,
            &mut y_buf,
            rows,
            in_features,
            out_features,
        )
        .expect("real gemm_bf16_linear call failed");

        let mut y_bf16 = vec![0u16; rows * out_features];
        y_buf.copy_to_host(&mut y_bf16).unwrap();
        let got: Vec<f32> = y_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 0.02,
                "element {i}: GPU GEMM got {g}, independent reference {e}"
            );
        }
    }

    /// Decisive real-weight cross-check: the real layer 0 `mlp.down_proj.weight`
    /// (`[2560, 9216]`, the exact GEMM this model runs on every decode step
    /// right after this crate's own SwiGLU kernel), a fixed deterministic
    /// synthetic activation input, run on the real GPU via real hipBLAS and
    /// checked against values computed by real PyTorch's `F.linear` on the
    /// same real weight and the same input formula (see
    /// `scratchpad/gen_gemm_reference.py` -- an independent process, not
    /// this code's own path).
    #[test]
    fn real_gemm_matches_real_down_proj_weight() {
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
        let rows = 1usize;
        let x_f32: Vec<f32> = (0..rows * in_features)
            .map(|i| (((i % 13) as i32 - 6) as f32) * 0.05)
            .collect();
        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
        let mut w_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(w_bf16.len()).unwrap();
        let mut y_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(rows * out_features).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();
        w_buf.copy_from_host(&w_bf16).unwrap();

        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        gemm_bf16_linear(
            &handle,
            &x_buf,
            &w_buf,
            &mut y_buf,
            rows,
            in_features,
            out_features,
        )
        .expect("real gemm_bf16_linear call failed against real down_proj weight");

        let mut y_bf16 = vec![0u16; rows * out_features];
        y_buf.copy_to_host(&mut y_bf16).unwrap();
        let got: Vec<f32> = y_bf16[..8].iter().map(|&b| bf16_to_f32(b)).collect();

        // Independently computed by real PyTorch F.linear on this exact
        // real weight and input (see scratchpad/gen_gemm_reference.py's
        // stdout).
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
                "element {i}: GPU GEMM got {g}, independent Python reference {e}"
            );
        }
    }

    /// Investigates a real question raised about the GEMM result: Python's
    /// benchmark (`scratchpad/bench_gemm_python.py`) issues all `iters`
    /// kernel launches back-to-back with NO synchronization between them,
    /// only calling `torch.cuda.synchronize()` once after the whole loop --
    /// classic async pipelining, where CPU launch overhead for call N+1
    /// overlaps with the GPU still executing call N. `gemm_bf16_linear`
    /// (and every other kernel wrapper in this crate) calls
    /// `hip::device_synchronize()` at the end of EVERY call, which forces a
    /// full host-device round trip serially, back to back -- a
    /// structurally different, more conservative thing to measure. This
    /// test isolates that variable: same handle, same buffers, same
    /// `hipblasGemmEx` call, but launched via the raw FFI directly (no
    /// per-call sync), with ONE `hipDeviceSynchronize()` after all `iters`
    /// launches -- matching the Python benchmark's own methodology exactly,
    /// so the comparison is apples-to-apples for what's actually being
    /// asked ("is the hipBLAS call itself slow, or is per-call sync the
    /// reason"). `#[ignore]`d for the same reason as the other benchmarks.
    #[test]
    #[ignore]
    fn bench_real_gemm_unsynced_pipelined_vs_python_runtime_shapes() {
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
        let w_bf16: Vec<u16> = raw
            .data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let mut w_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(w_bf16.len()).unwrap();
        w_buf.copy_from_host(&w_bf16).unwrap();

        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        let iters = 2000;
        let warmup = 100;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        for &rows in &[1usize, 128usize] {
            let x_f32: Vec<f32> = (0..rows * in_features)
                .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
                .collect();
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
            x_buf.copy_from_host(&x_bf16).unwrap();
            let mut y_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(rows * out_features).unwrap();

            let launch_one = |x_buf: &DeviceBuffer<u16>, y_buf: &mut DeviceBuffer<u16>| {
                // SAFETY: same call as gemm_bf16_linear, minus the trailing
                // device_synchronize() -- deliberately, to isolate its cost.
                // Buffers/handle are the same live, correctly-sized
                // allocations used throughout this file.
                unsafe {
                    ffi::hipblasGemmEx(
                        handle.raw,
                        ffi::HIPBLAS_OP_T,
                        ffi::HIPBLAS_OP_N,
                        out_features as c_int,
                        rows as c_int,
                        in_features as c_int,
                        &alpha as *const f32 as *const c_void,
                        w_buf.as_device_ptr(),
                        ffi::HIP_R_16BF,
                        in_features as c_int,
                        x_buf.as_device_ptr(),
                        ffi::HIP_R_16BF,
                        in_features as c_int,
                        &beta as *const f32 as *const c_void,
                        y_buf.as_device_ptr_mut() as *mut c_void,
                        ffi::HIP_R_16BF,
                        out_features as c_int,
                        ffi::HIPBLAS_COMPUTE_32F,
                        ffi::HIPBLAS_GEMM_DEFAULT,
                    );
                }
            };

            for _ in 0..warmup {
                launch_one(&x_buf, &mut y_buf);
            }
            hip::device_synchronize().unwrap();

            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                launch_one(&x_buf, &mut y_buf);
            }
            hip::device_synchronize().unwrap(); // ONE sync after all launches, matching Python
            let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!(
                "gemm_bf16_linear (UNSYNCED, pipelined) rows={rows:4} in={in_features} out={out_features}: {per_call_us:.3} us/call ({iters} iters, ONE hipDeviceSynchronize after the whole loop)"
            );
        }
    }

    #[test]
    #[ignore]
    fn bench_real_gemm_vs_python_runtime_shapes() {
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
        let out_features = raw.shape[0]; // 2560
        let in_features = raw.shape[1]; // 9216
        let w_bf16: Vec<u16> = raw
            .data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let mut w_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(w_bf16.len()).unwrap();
        w_buf.copy_from_host(&w_bf16).unwrap();

        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        let iters = 2000;
        let warmup = 100;

        for &rows in &[1usize, 128usize] {
            let x_f32: Vec<f32> = (0..rows * in_features)
                .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
                .collect();
            let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
            let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
            x_buf.copy_from_host(&x_bf16).unwrap();
            let mut y_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(rows * out_features).unwrap();

            for _ in 0..warmup {
                gemm_bf16_linear(
                    &handle,
                    &x_buf,
                    &w_buf,
                    &mut y_buf,
                    rows,
                    in_features,
                    out_features,
                )
                .unwrap();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                gemm_bf16_linear(
                    &handle,
                    &x_buf,
                    &w_buf,
                    &mut y_buf,
                    rows,
                    in_features,
                    out_features,
                )
                .unwrap();
            }
            let per_call_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
            println!(
                "gemm_bf16_linear rows={rows:4} in={in_features} out={out_features}: {per_call_us:.3} us/call ({iters} iters, includes hipDeviceSynchronize)"
            );
        }
    }

    /// §93 feasibility check: is `hipblasGemmEx` itself stream-capture-safe
    /// on this hipBLAS version? Not all BLAS libraries historically
    /// supported being captured into a graph (some versions do internal
    /// host-side synchronization or lazy workspace allocation that breaks
    /// capture) -- this is the single biggest unknown standing between
    /// `hip.rs`'s now-proven capture/replay mechanism (see
    /// `hip::tests::real_hip_graph_capture_replay_reflects_new_data_without_recapture`)
    /// and actually wrapping the real decode loop's ~128 GEMMs/token in a
    /// graph, so it's checked in isolation, first, before that much larger
    /// effort. Captures ONE real `hipblasGemmEx` call (via `BlasHandle`,
    /// bound to an explicit stream through `set_stream`) against a small
    /// fixed weight, then replays it against a DIFFERENT input `x` --
    /// WITHOUT re-capturing -- and checks the result against a reference
    /// computed independently in plain f32 Rust for that new `x` (same
    /// independent-reference discipline as
    /// `real_gemm_matches_independently_computed_reference` above).
    #[test]
    fn real_gemm_is_capturable_and_replays_new_input_correctly() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let rows = 1usize;
        let in_features = 4usize;
        let out_features = 3usize;

        // w: [out_features, in_features], row-major (nn.Linear layout) --
        // fixed for both replays; only `x` changes between them.
        let w_f32: Vec<f32> = vec![
            0.1, 0.2, -0.1, 0.05, // out 0
            -0.2, 0.3, 0.1, -0.05, // out 1
            0.0, -0.1, 0.2, 0.15, // out 2
        ];
        let w_bf16: Vec<u16> = w_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let x1_f32: Vec<f32> = vec![1.0, -2.0, 0.5, 3.0];
        let x1_bf16: Vec<u16> = x1_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(in_features).unwrap();
        let mut w_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(w_bf16.len()).unwrap();
        let mut y_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(out_features).unwrap();
        w_buf.copy_from_host(&w_bf16).unwrap();
        x_buf.copy_from_host(&x1_bf16).unwrap();

        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        let stream = hip::Stream::create().expect("Stream::create failed");
        handle.set_stream(&stream).expect("hipblasSetStream failed");

        hip::begin_capture(&stream).expect("begin_capture failed");
        // SAFETY: `handle` is bound to `stream` (just set above), which is
        // the stream currently being captured. Pointers are real, live
        // `hipMalloc` allocations from the `DeviceBuffer`s above, valid for
        // the whole test. Row-major-via-column-major derivation identical
        // to `gemm_bf16_linear` (`m=out_features, n=rows, k=in_features`,
        // `trans_a=T, trans_b=N`).
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let status = unsafe {
            ffi::hipblasGemmEx(
                handle.raw(),
                ffi::HIPBLAS_OP_T,
                ffi::HIPBLAS_OP_N,
                out_features as i32,
                rows as i32,
                in_features as i32,
                &alpha as *const f32 as *const c_void,
                w_buf.as_device_ptr(),
                ffi::HIP_R_16BF,
                in_features as i32,
                x_buf.as_device_ptr(),
                ffi::HIP_R_16BF,
                in_features as i32,
                &beta as *const f32 as *const c_void,
                y_buf.as_device_ptr_mut(),
                ffi::HIP_R_16BF,
                out_features as i32,
                ffi::HIPBLAS_COMPUTE_32F,
                ffi::HIPBLAS_GEMM_DEFAULT,
            )
        };
        BlasError::from_status(status).expect("hipblasGemmEx (captured) failed");
        let graph_exec = hip::end_capture(&stream).expect("end_capture failed -- hipBLAS is NOT stream-capture-safe on this system");

        // Replay 1: same input the GEMM was captured against.
        graph_exec.launch(&stream).expect("graph_exec.launch (replay 1) failed");
        stream.synchronize().expect("stream.synchronize (replay 1) failed");

        let mut expected1 = vec![0f32; out_features];
        for o in 0..out_features {
            let mut acc = 0f32;
            for i in 0..in_features {
                acc += x1_f32[i] * w_f32[o * in_features + i];
            }
            expected1[o] = acc;
        }
        let mut y1_bf16 = vec![0u16; out_features];
        y_buf.copy_to_host(&mut y1_bf16).unwrap();
        let got1: Vec<f32> = y1_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
        for (i, (g, e)) in got1.iter().zip(expected1.iter()).enumerate() {
            assert!((g - e).abs() < 0.02, "replay 1, element {i}: got {g}, expected {e}");
        }

        // THE decisive part: overwrite x_buf with DIFFERENT real data,
        // WITHOUT re-capturing, then replay the SAME graph_exec again.
        let x2_f32: Vec<f32> = vec![-1.0, 2.0, 0.25, -0.75];
        let x2_bf16: Vec<u16> = x2_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        x_buf.copy_from_host(&x2_bf16).unwrap();

        graph_exec.launch(&stream).expect("graph_exec.launch (replay 2) failed");
        stream.synchronize().expect("stream.synchronize (replay 2) failed");

        let mut expected2 = vec![0f32; out_features];
        for o in 0..out_features {
            let mut acc = 0f32;
            for i in 0..in_features {
                acc += x2_f32[i] * w_f32[o * in_features + i];
            }
            expected2[o] = acc;
        }
        let mut y2_bf16 = vec![0u16; out_features];
        y_buf.copy_to_host(&mut y2_bf16).unwrap();
        let got2: Vec<f32> = y2_bf16.iter().map(|&b| bf16_to_f32(b)).collect();
        for (i, (g, e)) in got2.iter().zip(expected2.iter()).enumerate() {
            assert!(
                (g - e).abs() < 0.02,
                "replay 2 (new input, NOT re-captured), element {i}: got {g}, expected {e} -- \
                 if this fails, a captured hipblasGemmEx is NOT correctly re-reading live input"
            );
        }
    }
}
