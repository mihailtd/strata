//! §92 performance pass: hipBLASLt -- a real, algorithm-tuned GEMM library
//! (installed by the user specifically for this: `sudo pacman -S
//! hipblaslt`), distinct from `blas.rs`'s plain `hipblasGemmEx` (which has
//! exactly one algorithm, `HIPBLAS_GEMM_DEFAULT`). §92's own profiling
//! (docs/DECISIONS.md) found the per-layer GEMMs achieving only ~37-42% of
//! this GPU's real memory bandwidth at `rows=1` (a skinny, matrix-vector-
//! shaped GEMM), versus ~67% for the one large `lm_head` GEMM -- exactly
//! the class of shape hipBLASLt's heuristic algorithm search exists to
//! improve, and what production inference engines (vLLM, PyTorch on AMD)
//! actually use for this reason.
//!
//! DESIGN: a real one-time cost, paid once, not per token. hipBLASLt's
//! heuristic search (`hipblasLtMatmulAlgoGetHeuristic`) is itself
//! expensive -- this module runs it exactly ONCE per distinct real GEMM
//! shape this model needs (there are 6, not one per layer: several layers
//! share an identical shape, e.g. every layer's `down_proj` is
//! `[2560,9216]`), at model-load time, and caches the resulting algorithm
//! (`hipblasLtMatmulAlgo_t`, a small POD struct) plus the matmul/matrix-
//! layout descriptors for reuse on every subsequent token. The per-token
//! hot path only ever calls `hipblasLtMatmul` with an already-chosen
//! algorithm -- no search, no allocation, same "queue on the default
//! stream, no host sync" discipline as every other raw launch in
//! `model.rs`.
//!
//! Same row-major-via-column-major derivation `blas.rs`'s `gemm_bf16_linear`
//! already established and proved correct (§85/§86): `transA=OP_T` applied
//! to the weight buffer, `transB=OP_N` applied to the activation buffer,
//! `M=out_features, N=rows, K=in_features` -- reused here unchanged rather
//! than re-derived, since it's already a proven-correct mapping.

use crate::hip::HipError;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;

mod ffi {
    use std::ffi::c_void;

    pub type LtHandle = *mut c_void;
    pub type MatmulDesc = *mut c_void;
    pub type MatrixLayout = *mut c_void;
    pub type MatmulPreference = *mut c_void;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct MatmulAlgo {
        pub data: [u8; 16],
        pub max_workspace_bytes: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct MatmulHeuristicResult {
        pub algo: MatmulAlgo,
        pub workspace_size: usize,
        pub state: i32,
        pub waves_count: f32,
        pub reserved: [i32; 4],
    }

    unsafe extern "C" {
        pub fn hipblasLtCreate(handle: *mut LtHandle) -> i32;
        pub fn hipblasLtDestroy(handle: LtHandle) -> i32;

        pub fn hipblasLtMatmulDescCreate(desc: *mut MatmulDesc, compute_type: i32, scale_type: i32) -> i32;
        pub fn hipblasLtMatmulDescDestroy(desc: MatmulDesc) -> i32;
        pub fn hipblasLtMatmulDescSetAttribute(
            desc: MatmulDesc,
            attr: i32,
            buf: *const c_void,
            size_in_bytes: usize,
        ) -> i32;

        pub fn hipblasLtMatrixLayoutCreate(
            layout: *mut MatrixLayout,
            data_type: i32,
            rows: u64,
            cols: u64,
            ld: i64,
        ) -> i32;
        pub fn hipblasLtMatrixLayoutDestroy(layout: MatrixLayout) -> i32;

        pub fn hipblasLtMatmulPreferenceCreate(pref: *mut MatmulPreference) -> i32;
        pub fn hipblasLtMatmulPreferenceDestroy(pref: MatmulPreference) -> i32;
        pub fn hipblasLtMatmulPreferenceSetAttribute(
            pref: MatmulPreference,
            attr: i32,
            buf: *const c_void,
            size_in_bytes: usize,
        ) -> i32;

        #[allow(clippy::too_many_arguments)]
        pub fn hipblasLtMatmulAlgoGetHeuristic(
            handle: LtHandle,
            matmul_desc: MatmulDesc,
            a_desc: MatrixLayout,
            b_desc: MatrixLayout,
            c_desc: MatrixLayout,
            d_desc: MatrixLayout,
            pref: MatmulPreference,
            requested_algo_count: i32,
            heuristic_results: *mut MatmulHeuristicResult,
            return_algo_count: *mut i32,
        ) -> i32;

        #[allow(clippy::too_many_arguments)]
        pub fn hipblasLtMatmul(
            handle: LtHandle,
            matmul_desc: MatmulDesc,
            alpha: *const c_void,
            a: *const c_void,
            a_desc: MatrixLayout,
            b: *const c_void,
            b_desc: MatrixLayout,
            beta: *const c_void,
            c: *const c_void,
            c_desc: MatrixLayout,
            d: *mut c_void,
            d_desc: MatrixLayout,
            algo: *const MatmulAlgo,
            workspace: *mut c_void,
            workspace_size_in_bytes: usize,
            stream: *mut c_void,
        ) -> i32;
    }

    // hipblasOperation_t (shared with blas.rs's plain hipBLAS -- same enum).
    pub const HIPBLAS_OP_N: i32 = 111;
    pub const HIPBLAS_OP_T: i32 = 112;
    // hipDataType
    pub const HIP_R_32F: i32 = 0;
    pub const HIP_R_16BF: i32 = 14;
    // hipblasComputeType_t
    pub const HIPBLAS_COMPUTE_32F: i32 = 2;
    // hipblasLtMatmulDescAttributes_t
    pub const HIPBLASLT_MATMUL_DESC_TRANSA: i32 = 0;
    pub const HIPBLASLT_MATMUL_DESC_TRANSB: i32 = 1;
    // hipblasLtMatmulPreferenceAttributes_t
    pub const HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: i32 = 1;
    // hipblasStatus_t
    pub const HIPBLAS_STATUS_SUCCESS: i32 = 0;
}

fn check(code: i32, what: &str) -> Result<(), String> {
    if code == ffi::HIPBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what} failed: hipblasStatus_t {code}"))
    }
}

/// One real hipBLASLt library context, owned for the process's lifetime
/// (created once, destroyed on drop).
pub struct LtHandle {
    raw: ffi::LtHandle,
}

unsafe impl Send for LtHandle {}
unsafe impl Sync for LtHandle {}

impl LtHandle {
    pub fn create() -> Result<Self, String> {
        let mut raw: ffi::LtHandle = ptr::null_mut();
        // SAFETY: `raw` is a valid, writable pointer for the call's
        // duration; hipBLASLt writes exactly one handle through it or
        // returns a nonzero status (checked immediately).
        let status = unsafe { ffi::hipblasLtCreate(&mut raw) };
        check(status, "hipblasLtCreate")?;
        Ok(LtHandle { raw })
    }
}

impl Drop for LtHandle {
    fn drop(&mut self) {
        // SAFETY: `self.raw` came from a successful hipblasLtCreate above,
        // sole owner, only Drop.
        unsafe {
            ffi::hipblasLtDestroy(self.raw);
        }
    }
}

/// One real, tuned GEMM plan for a specific `(out_features, in_features)`
/// shape at `rows=1` -- the matmul descriptor, matrix layouts, and the
/// ALREADY-SEARCHED best algorithm, all created once and reused every
/// call. `workspace` is real GPU scratch memory some algorithms need,
/// sized once (from the heuristic result) and never reallocated.
struct GemmPlan {
    matmul_desc: ffi::MatmulDesc,
    a_desc: ffi::MatrixLayout,
    b_desc: ffi::MatrixLayout,
    cd_desc: ffi::MatrixLayout,
    algo: ffi::MatmulAlgo,
    workspace: Option<crate::hip::DeviceBuffer<u8>>,
}

// SAFETY: hipBLASLt descriptors are not tied to a host thread; this
// crate's own single-threaded decode loop is the only real caller.
unsafe impl Send for GemmPlan {}

impl GemmPlan {
    fn new(lt: &LtHandle, out_features: i32, in_features: i32) -> Result<Self, String> {
        let m = out_features;
        let n: i32 = 1; // this model's real decode-step shape, always rows=1
        let k = in_features;

        let mut matmul_desc: ffi::MatmulDesc = ptr::null_mut();
        // SAFETY: valid output pointer; checked below.
        check(
            unsafe { ffi::hipblasLtMatmulDescCreate(&mut matmul_desc, ffi::HIPBLAS_COMPUTE_32F, ffi::HIP_R_32F) },
            "hipblasLtMatmulDescCreate",
        )?;

        let trans_a = ffi::HIPBLAS_OP_T;
        let trans_b = ffi::HIPBLAS_OP_N;
        unsafe {
            check(
                ffi::hipblasLtMatmulDescSetAttribute(
                    matmul_desc,
                    ffi::HIPBLASLT_MATMUL_DESC_TRANSA,
                    &trans_a as *const i32 as *const c_void,
                    size_of::<i32>(),
                ),
                "hipblasLtMatmulDescSetAttribute(TRANSA)",
            )?;
            check(
                ffi::hipblasLtMatmulDescSetAttribute(
                    matmul_desc,
                    ffi::HIPBLASLT_MATMUL_DESC_TRANSB,
                    &trans_b as *const i32 as *const c_void,
                    size_of::<i32>(),
                ),
                "hipblasLtMatmulDescSetAttribute(TRANSB)",
            )?;
        }

        // A = weight buffer, stored column-major [K, M] (ld=K) -- see this
        // module's own doc comment for the row-major-via-column-major
        // derivation this reuses unchanged from blas.rs.
        let mut a_desc: ffi::MatrixLayout = ptr::null_mut();
        check(
            unsafe { ffi::hipblasLtMatrixLayoutCreate(&mut a_desc, ffi::HIP_R_16BF, k as u64, m as u64, k as i64) },
            "hipblasLtMatrixLayoutCreate(A)",
        )?;
        // B = activation buffer, stored column-major [K, N] (ld=K).
        let mut b_desc: ffi::MatrixLayout = ptr::null_mut();
        check(
            unsafe { ffi::hipblasLtMatrixLayoutCreate(&mut b_desc, ffi::HIP_R_16BF, k as u64, n as u64, k as i64) },
            "hipblasLtMatrixLayoutCreate(B)",
        )?;
        // C/D = output buffer, column-major [M, N] (ld=M). beta=0 always
        // in this crate's usage, so C is never actually read -- C and D
        // share one layout and, at call time, the same real buffer.
        let mut cd_desc: ffi::MatrixLayout = ptr::null_mut();
        check(
            unsafe { ffi::hipblasLtMatrixLayoutCreate(&mut cd_desc, ffi::HIP_R_16BF, m as u64, n as u64, m as i64) },
            "hipblasLtMatrixLayoutCreate(C/D)",
        )?;

        let mut pref: ffi::MatmulPreference = ptr::null_mut();
        check(
            unsafe { ffi::hipblasLtMatmulPreferenceCreate(&mut pref) },
            "hipblasLtMatmulPreferenceCreate",
        )?;
        // 32MB workspace ceiling for the heuristic search to consider --
        // negligible next to this GPU's real VRAM, generous enough not to
        // rule out a real algorithm that would otherwise win.
        let max_workspace_bytes: u64 = 32 * 1024 * 1024;
        unsafe {
            check(
                ffi::hipblasLtMatmulPreferenceSetAttribute(
                    pref,
                    ffi::HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &max_workspace_bytes as *const u64 as *const c_void,
                    size_of::<u64>(),
                ),
                "hipblasLtMatmulPreferenceSetAttribute",
            )?;
        }

        // The real, one-time, potentially-slow heuristic search -- paid
        // once per distinct shape at model-load time, never again.
        let mut result = ffi::MatmulHeuristicResult {
            algo: ffi::MatmulAlgo { data: [0; 16], max_workspace_bytes: 0 },
            workspace_size: 0,
            state: 0,
            waves_count: 0.0,
            reserved: [0; 4],
        };
        let mut return_count: i32 = 0;
        let status = unsafe {
            ffi::hipblasLtMatmulAlgoGetHeuristic(
                lt.raw,
                matmul_desc,
                a_desc,
                b_desc,
                cd_desc,
                cd_desc,
                pref,
                1,
                &mut result,
                &mut return_count,
            )
        };
        unsafe {
            ffi::hipblasLtMatmulPreferenceDestroy(pref);
        }
        check(status, "hipblasLtMatmulAlgoGetHeuristic")?;
        if return_count == 0 {
            return Err(format!(
                "hipblasLtMatmulAlgoGetHeuristic returned zero algorithms for shape M={m} N={n} K={k}"
            ));
        }

        let workspace = if result.workspace_size > 0 {
            Some(
                crate::hip::DeviceBuffer::<u8>::alloc(result.workspace_size)
                    .map_err(|e: HipError| format!("allocating hipBLASLt workspace: {e}"))?,
            )
        } else {
            None
        };

        Ok(GemmPlan {
            matmul_desc,
            a_desc,
            b_desc,
            cd_desc,
            algo: result.algo,
            workspace,
        })
    }

    /// Runs this shape's already-tuned GEMM: `y = w^T @ x` (bf16 in/out,
    /// fp32 accumulation), queued on the default stream, no per-call
    /// synchronization -- the caller (`model.rs`'s hot path) syncs once
    /// per token, same discipline as every other raw launch there.
    ///
    /// SAFETY: `x`/`w`/`y` must be real, live device pointers with at
    /// least the element counts this plan was built for
    /// (`out_features*in_features` for `w`, `in_features` for `x`,
    /// `out_features` for `y`) -- the caller's responsibility, same trust
    /// boundary as every other `raw::*` function in `model.rs`.
    unsafe fn matmul(&mut self, lt: &LtHandle, x: *const c_void, w: *const c_void, y: *mut c_void) {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let (workspace_ptr, workspace_bytes) = match &mut self.workspace {
            Some(buf) => (buf.as_device_ptr_mut(), buf.len()),
            None => (ptr::null_mut(), 0),
        };
        unsafe {
            ffi::hipblasLtMatmul(
                lt.raw,
                self.matmul_desc,
                &alpha as *const f32 as *const c_void,
                w,
                self.a_desc,
                x,
                self.b_desc,
                &beta as *const f32 as *const c_void,
                y as *const c_void,
                self.cd_desc,
                y,
                self.cd_desc,
                &self.algo,
                workspace_ptr,
                workspace_bytes,
                ptr::null_mut(),
            );
        }
    }
}

impl Drop for GemmPlan {
    fn drop(&mut self) {
        // SAFETY: each descriptor came from its own successful *Create
        // call above; this is the sole owner, only Drop.
        unsafe {
            ffi::hipblasLtMatmulDescDestroy(self.matmul_desc);
            ffi::hipblasLtMatrixLayoutDestroy(self.a_desc);
            ffi::hipblasLtMatrixLayoutDestroy(self.b_desc);
            ffi::hipblasLtMatrixLayoutDestroy(self.cd_desc);
        }
    }
}

/// All of this model's real GEMM shapes, each tuned exactly once. Built
/// once at model-load time (real, potentially slow: several heuristic
/// searches), then reused for the whole decode loop's lifetime.
pub struct GemmPlans {
    handle: LtHandle,
    plans: HashMap<(i32, i32), GemmPlan>,
}

impl GemmPlans {
    /// `shapes`: every distinct `(out_features, in_features)` this model's
    /// real forward pass needs (duplicates across layers collapse to one
    /// real search -- e.g. every layer's `down_proj` shares one shape).
    pub fn build(shapes: &[(i32, i32)]) -> Result<Self, String> {
        let handle = LtHandle::create()?;
        let mut plans = HashMap::with_capacity(shapes.len());
        for &(out_features, in_features) in shapes {
            let plan = GemmPlan::new(&handle, out_features, in_features)?;
            plans.insert((out_features, in_features), plan);
        }
        Ok(GemmPlans { handle, plans })
    }

    /// `y = w^T @ x` using the pre-tuned algorithm for this exact
    /// `(out_features, in_features)` shape. Panics (a real programming
    /// error, not a runtime condition) if this shape wasn't included in
    /// `build`'s `shapes` -- every real call site in `model.rs` passes a
    /// shape known at compile time, so a missing shape means a real bug
    /// in how `GemmPlans` was built, not user input.
    ///
    /// # Safety
    /// Same contract as `GemmPlan::matmul`: `x`/`w`/`y` must be real, live
    /// device pointers with at least this shape's implied element counts.
    pub unsafe fn matmul(&mut self, out_features: i32, in_features: i32, x: *const c_void, w: *const c_void, y: *mut c_void) {
        let plan = self
            .plans
            .get_mut(&(out_features, in_features))
            .unwrap_or_else(|| panic!("GemmPlans::matmul: no plan built for shape ({out_features}, {in_features})"));
        unsafe {
            plan.matmul(&self.handle, x, w, y);
        }
    }
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

    /// Real correctness test for `GemmPlans`, kept alive and exercised
    /// even though §92's real A/B (docs/DECISIONS.md) found this library
    /// a reproducible REGRESSION for `model.rs`'s actual hot-path shapes
    /// (46.1 tok/s via hipBLASLt vs. 49.2 tok/s via plain `hipblasGemmEx`)
    /// -- reverted out of the hot path for that reason, but the
    /// integration itself is real and correct, so it stays tested rather
    /// than left to bit-rot as unreferenced dead code. Reuses the exact
    /// same real layer-0 `mlp.down_proj.weight` and independently-computed
    /// reference values `blas.rs`'s own `real_gemm_matches_real_down_proj_weight`
    /// test already established (`scratchpad/gen_gemm_reference.py`) --
    /// same real weight, same oracle, different GEMM library underneath.
    ///
    /// Real, deliberate scope: same Qwen3.5-4B-specific real weight and
    /// reference values as `blas::tests::real_gemm_matches_real_down_proj_weight`
    /// -- gated the same way, for the same reason.
    #[test]
    #[cfg(feature = "qwen35_4b")]
    fn real_gemm_plans_matches_real_down_proj_weight() {
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
        let out_features = raw.shape[0] as i32; // 2560
        let in_features = raw.shape[1] as i32; // 9216
        let w_bf16 = raw.to_bf16_bits();

        let rows = 1usize;
        let x_f32: Vec<f32> = (0..rows * in_features as usize)
            .map(|i| (((i % 13) as i32 - 6) as f32) * 0.05)
            .collect();
        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
        let mut w_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(w_bf16.len()).unwrap();
        let mut y_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(out_features as usize).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();
        w_buf.copy_from_host(&w_bf16).unwrap();

        let mut plans = GemmPlans::build(&[(out_features, in_features)])
            .expect("real GemmPlans::build failed (real hipBLASLt heuristic search)");
        unsafe {
            plans.matmul(out_features, in_features, x_buf.as_device_ptr(), w_buf.as_device_ptr(), y_buf.as_device_ptr_mut() as *mut c_void);
        }

        let mut y_bf16 = vec![0u16; out_features as usize];
        y_buf.copy_to_host(&mut y_bf16).unwrap();
        let got: Vec<f32> = y_bf16[..8].iter().map(|&b| bf16_to_f32(b)).collect();

        // Independently computed by real PyTorch F.linear on this exact
        // real weight and input -- same reference values `blas.rs`'s own
        // decisive GEMM test uses (see scratchpad/gen_gemm_reference.py).
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
                "element {i}: hipBLASLt GemmPlans got {g}, independent Python reference {e}"
            );
        }
    }
}
