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
