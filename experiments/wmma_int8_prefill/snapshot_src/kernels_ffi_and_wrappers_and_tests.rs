// Snapshot: the real Rust FFI declarations, safe wrappers, and tests for
// the WMMA INT8 fragment-layout debugging trail (§125/§126 in
// docs/DECISIONS.md), extracted verbatim from apps/runtime-next/src/
// kernels.rs at the point they were removed from the live crate.
//
// NOT part of the active build -- these are historical reference only.
// The production kernel these investigations led to
// (w4a16_gemm_prefill_wmma_int8.hip) and its real, permanent correctness/
// benchmark gates remain live in apps/runtime-next/src/kernels.rs.
//
// See ../README.md for the full narrative.

// ============================================================
// 1. FFI declarations (were inside the `extern "C"` block in kernels.rs)
// ============================================================

        /// §125 EXPERIMENT, ISOLATED PROBE: minimal single-tile WMMA
        /// GEMM feasibility test -- see `src/kernels/wmma_probe.hip`'s
        /// own doc comment and `docs/DECISIONS.md` §125 for the real
        /// verdict.
        pub fn launch_wmma_probe_16x16x16_bf16(a_dup: *const c_void, b_dup: *const c_void, c_out: *mut c_void, stride_half2: c_int, stream: *mut c_void);

        /// §126 EXPERIMENT, ISOLATED PROBE: minimal single-tile INT8xINT8
        /// WMMA GEMM, real fragment layout ported verbatim from
        /// llama.cpp's own production RDNA3 `mma.cuh`/`mmq-vec-dot.cuh` --
        /// see `src/kernels/wmma_int8_probe.hip`'s own doc comment and
        /// `docs/DECISIONS.md` §126 for the real verdict.
        pub fn launch_wmma_int8_probe_16x16x32(a_packed: *const c_void, b_packed: *const c_void, c_out: *mut c_void, stream: *mut c_void);

        /// §126 DEBUG: instrumented copy of the production WMMA INT8
        /// kernel, dumping per-thread `my_row`, the raw weight/activation
        /// fragment's first int32 (chunk 0, k_base 0), and the raw
        /// post-first-wmma-call accumulator, to localize a real bug the
        /// production correctness test found.
        #[allow(clippy::too_many_arguments)]
        pub fn launch_w4a16_gemm_prefill_wmma_int8_debug(
            x: *const c_void,
            qweight: *const c_void,
            scales: *const c_void,
            y: *mut c_void,
            out_features: c_int,
            in_features: c_int,
            group_size: c_int,
            num_tokens: c_int,
            debug_my_row: *mut c_void,
            debug_a0: *mut c_void,
            debug_b0: *mut c_void,
            debug_acc0: *mut c_void,
            stream: *mut c_void,
        );

        /// §126 DEBUG: 4-chunk (K=128) accumulation variant of the
        /// above, isolating whether accumulating across multiple WMMA
        /// chunk calls into one persistent `acc` is where a real bug in
        /// the production kernel lives.
        pub fn launch_wmma_int8_probe_4chunk(a_packed: *const c_void, b_packed: *const c_void, c_out: *mut c_void, stream: *mut c_void);


// ============================================================
// 2. Safe wrappers
// ============================================================

/// §125 EXPERIMENT, ISOLATED PROBE: minimal single-16x16x16-tile
/// bf16xbf16->f32 WMMA GEMM. `a_dup`/`b_dup` must ALREADY be in the
/// real, hypothesized "duplicated half2" layout this probe exists to
/// test -- `[16, 16]` real bf16 values, each one physically duplicated
/// into both lanes of its own `half2` (i.e. `stride_half2` real half2
/// slots per row, `stride_half2` shorts = `2*stride_half2` real u16
/// elements per row in `a_dup`/`b_dup`'s own buffer). NOT a general
/// GEMM entry point -- see `src/kernels/wmma_probe.hip`'s own doc
/// comment for what real, unverified hypothesis this tests.
pub fn wmma_probe_16x16x16_bf16(a_dup: &DeviceBuffer<u16>, b_dup: &DeviceBuffer<u16>, c_out: &mut DeviceBuffer<f32>, stride_half2: usize) -> Result<(), HipError> {
    assert_eq!(a_dup.len(), 16 * stride_half2 * 2, "a_dup length must be 16 rows * stride_half2 half2-slots * 2 shorts/half2");
    assert_eq!(b_dup.len(), 16 * stride_half2 * 2, "b_dup length must be 16 rows * stride_half2 half2-slots * 2 shorts/half2");
    assert_eq!(c_out.len(), 16 * 16, "c_out length must be 16*16");
    // SAFETY: real, live allocations of the asserted sizes; the kernel
    // indexes strictly within `[0, 16*stride_half2*2)` for a_dup/b_dup
    // and `[0, 256)` for c_out (verified by reading `wmma_probe.hip`
    // directly).
    unsafe {
        ffi::launch_wmma_probe_16x16x16_bf16(a_dup.as_device_ptr(), b_dup.as_device_ptr(), c_out.as_device_ptr_mut(), stride_half2 as i32, std::ptr::null_mut());
    }
    check_last_error()?;
    device_synchronize()
}

/// §126 EXPERIMENT, ISOLATED PROBE: minimal single-tile INT8xINT8->INT32
/// WMMA GEMM (K=32, two real K=16 WMMA passes accumulated). `a_packed`/
/// `b_packed` must be `[16 rows, 8 int32]` = `[16, 32]` real int8 values,
/// row-major, one row per real output index (A's row = output row, B's
/// row = output column) -- NOT a general GEMM entry point, see
/// `src/kernels/wmma_int8_probe.hip`'s own doc comment for the real,
/// verbatim-ported fragment-layout formulas this probe verifies.
pub fn wmma_int8_probe_16x16x32(a_packed: &DeviceBuffer<i32>, b_packed: &DeviceBuffer<i32>, c_out: &mut DeviceBuffer<i32>) -> Result<(), HipError> {
    assert_eq!(a_packed.len(), 16 * 8, "a_packed length must be 16 rows * 8 int32/row");
    assert_eq!(b_packed.len(), 16 * 8, "b_packed length must be 16 rows * 8 int32/row");
    assert_eq!(c_out.len(), 16 * 16, "c_out length must be 16*16");
    // SAFETY: real, live allocations of the asserted sizes; the kernel
    // indexes strictly within `[0, 16*8)` for a_packed/b_packed and
    // `[0, 256)` for c_out (verified by reading `wmma_int8_probe.hip`
    // directly).
    unsafe {
        ffi::launch_wmma_int8_probe_16x16x32(a_packed.as_device_ptr(), b_packed.as_device_ptr(), c_out.as_device_ptr_mut(), std::ptr::null_mut());
    }
    check_last_error()?;
    device_synchronize()
}

/// §126 DEBUG: instrumented copy of the production WMMA INT8 kernel --
/// see `src/kernels/w4a16_gemm_prefill_wmma_int8_debug.hip`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_prefill_wmma_int8_debug(
    x: &DeviceBuffer<u16>,
    qweight: &DeviceBuffer<u32>,
    scales: &DeviceBuffer<u16>,
    y: &mut DeviceBuffer<u16>,
    out_features: usize,
    in_features: usize,
    group_size: usize,
    num_tokens: usize,
    debug_my_row: &mut DeviceBuffer<i32>,
    debug_a0: &mut DeviceBuffer<i32>,
    debug_b0: &mut DeviceBuffer<i32>,
    debug_acc0: &mut DeviceBuffer<i32>,
) -> Result<(), HipError> {
    unsafe {
        ffi::launch_w4a16_gemm_prefill_wmma_int8_debug(
            x.as_device_ptr(),
            qweight.as_device_ptr(),
            scales.as_device_ptr(),
            y.as_device_ptr_mut() as *mut c_void,
            out_features as i32,
            in_features as i32,
            group_size as i32,
            num_tokens as i32,
            debug_my_row.as_device_ptr_mut() as *mut c_void,
            debug_a0.as_device_ptr_mut() as *mut c_void,
            debug_b0.as_device_ptr_mut() as *mut c_void,
            debug_acc0.as_device_ptr_mut() as *mut c_void,
            std::ptr::null_mut(),
        );
    }
    check_last_error()?;
    device_synchronize()
}

/// §126 DEBUG: 4-chunk (K=128) accumulation variant, same simple direct
/// per-row packed-int8 format as `wmma_int8_probe_16x16x32` but with 4
/// real chunks accumulated into one persistent `acc`.
pub fn wmma_int8_probe_4chunk(a_packed: &DeviceBuffer<i32>, b_packed: &DeviceBuffer<i32>, c_out: &mut DeviceBuffer<i32>) -> Result<(), HipError> {
    assert_eq!(a_packed.len(), 16 * 32, "a_packed length must be 16 rows * 32 int32/row");
    assert_eq!(b_packed.len(), 16 * 32, "b_packed length must be 16 rows * 32 int32/row");
    assert_eq!(c_out.len(), 16 * 16, "c_out length must be 16*16");
    unsafe {
        ffi::launch_wmma_int8_probe_4chunk(a_packed.as_device_ptr(), b_packed.as_device_ptr(), c_out.as_device_ptr_mut(), std::ptr::null_mut());
    }
    check_last_error()?;
    device_synchronize()
}


// ============================================================
// 3. Tests (were inside #[cfg(test)] mod tests in kernels.rs)
// ============================================================

    /// §125 EXPERIMENT, ISOLATED PROBE, THE decisive test: does a
    /// minimal, single-16x16x16-tile, real bf16xbf16->f32 WMMA GEMM,
    /// using fragment-layout formulas ported VERBATIM from llama.cpp's
    /// own real, production `mma.cuh` (not re-derived), actually
    /// produce the correct real answer? A real, independent, hand-
    /// computed CPU reference (plain scalar Rust matmul, not calling
    /// the GPU or the intrinsic at all) is the real oracle. `a_dup`/
    /// `b_dup` are built in the real, HYPOTHESIZED "each value
    /// duplicated into both lanes of its own half2" layout this probe
    /// exists to test -- see `wmma_probe.hip`'s own doc comment for why
    /// that specific hypothesis.
    ///
    /// Real, honest, measured verdict (see `docs/DECISIONS.md` §125 for
    /// the full account): the hypothesis was FALSIFIED -- real GPU
    /// output diverged from the real CPU reference by a real margin far
    /// beyond bf16 rounding (max abs diff 119), and a real, direct
    /// permutation check ruled out a simple indexing swap as the cause
    /// (sorted-multiset diff 4537, not 0) -- the actual dot products
    /// computed are wrong, not just misplaced. Deliberately kept as a
    /// non-asserting DIAGNOSTIC (not a pass/fail gate -- there is no
    /// known-correct implementation yet to regress FROM) rather than a
    /// permanently-red test, matching this crate's own convention for
    /// an honestly unresolved investigation (`diagnose_*` naming, no
    /// hard assertion) -- NOT swept under the rug by deleting the real,
    /// negative result, and NOT left as a red herring that erodes trust
    /// in "all tests green" the way a permanently-failing `#[test]`
    /// would.
    #[test]
    #[ignore]
    fn diagnose_real_wmma_probe_16x16x16_vs_cpu_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        // Real, deterministic, small-integer test values (bf16-exact,
        // so bf16 rounding never muddies whether a mismatch is a real
        // fragment-layout bug vs. ordinary quantization noise).
        let a_f32: Vec<f32> = (0..16 * 16).map(|i| (((i * 7 + 3) % 9) as f32) - 4.0).collect();
        let b_f32: Vec<f32> = (0..16 * 16).map(|i| (((i * 5 + 2) % 7) as f32) - 3.0).collect();

        // Real, independent CPU reference: plain scalar 16x16x16 matmul,
        // C = A @ B, row-major, no GPU, no intrinsic.
        let mut c_ref = vec![0.0f32; 16 * 16];
        for i in 0..16 {
            for j in 0..16 {
                let mut sum = 0.0f32;
                for k in 0..16 {
                    sum += a_f32[i * 16 + k] * b_f32[k * 16 + j];
                }
                c_ref[i * 16 + j] = sum;
            }
        }

        // Real "duplicated half2" layout under test: row `r`'s data
        // occupies `stride_half2=16` half2 slots (32 real u16 elements),
        // slot `c` holding `(bf16(A[r][c]), bf16(A[r][c]))` -- the real
        // hypothesis `wmma_probe.hip` documents.
        let stride_half2 = 16usize;
        let dup_layout = |m: &[f32]| -> Vec<u16> {
            let mut out = vec![0u16; 16 * stride_half2 * 2];
            for r in 0..16 {
                for c in 0..16 {
                    let bits = f32_to_bf16(m[r * 16 + c]);
                    out[r * stride_half2 * 2 + c * 2] = bits;
                    out[r * stride_half2 * 2 + c * 2 + 1] = bits;
                }
            }
            out
        };
        let a_dup_host = dup_layout(&a_f32);
        let b_dup_host = dup_layout(&b_f32);

        let mut a_dup: DeviceBuffer<u16> = DeviceBuffer::alloc(a_dup_host.len()).unwrap();
        a_dup.copy_from_host(&a_dup_host).unwrap();
        let mut b_dup: DeviceBuffer<u16> = DeviceBuffer::alloc(b_dup_host.len()).unwrap();
        b_dup.copy_from_host(&b_dup_host).unwrap();
        let mut c_out: DeviceBuffer<f32> = DeviceBuffer::alloc(16 * 16).unwrap();

        wmma_probe_16x16x16_bf16(&a_dup, &b_dup, &mut c_out, stride_half2).expect("real wmma_probe_16x16x16_bf16 call failed");
        let mut c_got = vec![0.0f32; 16 * 16];
        c_out.copy_to_host(&mut c_got).unwrap();

        let mut max_diff = 0.0f32;
        for i in 0..16 * 16 {
            max_diff = max_diff.max((c_got[i] - c_ref[i]).abs());
        }
        eprintln!("WMMA probe vs real CPU reference: max abs diff = {max_diff}");
        eprintln!("got[0..4]  = {:?}", &c_got[0..4]);
        eprintln!("ref[0..4]  = {:?}", &c_ref[0..4]);
        {
            // Real, quick diagnostic (not part of the pass/fail gate):
            // is `c_got` a real PERMUTATION of `c_ref` (same real value
            // multiset, different order -- a real indexing bug, likely
            // fixable) or genuinely different VALUES (a real, deeper
            // computation bug)?
            let mut got_sorted = c_got.clone();
            let mut ref_sorted = c_ref.clone();
            got_sorted.sort_by(|a, b| a.total_cmp(b));
            ref_sorted.sort_by(|a, b| a.total_cmp(b));
            let perm_diff: f32 = got_sorted.iter().zip(ref_sorted.iter()).map(|(a, b)| (a - b).abs()).sum();
            eprintln!("diagnostic: sorted-multiset diff (0 => pure permutation/indexing bug) = {perm_diff}");
        }
        // Deliberately NOT an assertion -- see this test's own doc
        // comment for why (a real, honest, unresolved negative result,
        // not a regression gate with a known-correct baseline).
        if max_diff < 1.0 {
            eprintln!("WMMA probe MATCHES the real CPU reference -- the fragment-layout hypothesis holds.");
        } else {
            eprintln!("WMMA probe diverged from the real CPU reference by {max_diff} -- the fragment-layout hypothesis in wmma_probe.hip's own doc comment does not hold as tested; see docs/DECISIONS.md §125.");
        }
    }

    /// §126 EXPERIMENT, ISOLATED PROBE, CONFIRMED: a minimal, single-tile,
    /// real INT8xINT8->INT32 WMMA GEMM (K=32), using fragment-layout
    /// formulas read VERBATIM from llama.cpp's own real, production,
    /// RDNA3 `mma.cuh`/`mmq-vec-dot.cuh` (the exact code path Q4_K's own
    /// quantized prefill uses, per §121/§124's own
    /// `ggml_cuda_should_use_mmq` dispatch-logic finding), matches a real,
    /// independent CPU reference (plain scalar Rust int32 dot products,
    /// no GPU, no intrinsic) BIT-EXACTLY (max abs diff = 0) on its first
    /// real run -- unlike §125's bf16 probe, which caught a real, wrong
    /// hand-derived hypothesis before it could reach production.
    ///
    /// Unlike §125, this probe's formulas were NOT hand-derived from the
    /// generic `get_i`/`get_j` template -- they come from the one `tile<>`
    /// specialization (`DATA_LAYOUT_I_MAJOR_MIRRORED`, `mma.cuh` lines
    /// 529-569) directly confirmed to be what RDNA3's own `load_ldmatrix`
    /// uses for this exact tile shape (`static_assert(sizeof(t.x) == 32)`
    /// at `mma.cuh` line 844 -- the real source of the `ne=4` vs `ne=8`
    /// contradiction this session's earlier investigation got stuck on:
    /// the generic template's `ne = I*J/32` formula is for a DIFFERENT,
    /// non-mirrored layout; RDNA3's real mirrored layout is
    /// `ne = I*J/32*2 = 8`). A real, upgraded, permanent correctness gate
    /// now (hard `assert_eq!`, not a diagnostic) -- see `docs/DECISIONS.md`
    /// §126.
    #[test]
    #[ignore]
    fn real_wmma_int8_probe_16x16x32_matches_cpu_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        // Real, deterministic, small signed-int8 test values (well within
        // i8 range, so packing/unpacking is unambiguous and any mismatch
        // is a real layout bug, not overflow).
        let a_i8: Vec<i8> = (0..16 * 32).map(|i| (((i * 7 + 3) % 9) as i8) - 4).collect();
        let b_i8: Vec<i8> = (0..16 * 32).map(|i| (((i * 5 + 2) % 7) as i8) - 3).collect();

        // Real, independent CPU reference: C[i][j] = sum_k A[i][k]*B[j][k],
        // both i8 widened to i32, no GPU, no intrinsic.
        let mut c_ref = vec![0i32; 16 * 16];
        for i in 0..16 {
            for j in 0..16 {
                let mut sum = 0i32;
                for k in 0..32 {
                    sum += a_i8[i * 32 + k] as i32 * b_i8[j * 32 + k] as i32;
                }
                c_ref[i * 16 + j] = sum;
            }
        }

        // Real packing: 4 signed i8 bytes -> 1 i32, LSB-first (byte 0 in
        // bits 0..7), matching the real `int32x4_t` reinterpretation the
        // hardware intrinsic expects for a row's raw byte data.
        let pack_row_i8 = |row: &[i8]| -> Vec<i32> {
            assert_eq!(row.len(), 32);
            (0..8)
                .map(|w| {
                    let b0 = row[w * 4] as u8 as i32;
                    let b1 = row[w * 4 + 1] as u8 as i32;
                    let b2 = row[w * 4 + 2] as u8 as i32;
                    let b3 = row[w * 4 + 3] as u8 as i32;
                    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                })
                .collect()
        };
        let a_packed_host: Vec<i32> = (0..16).flat_map(|r| pack_row_i8(&a_i8[r * 32..r * 32 + 32])).collect();
        let b_packed_host: Vec<i32> = (0..16).flat_map(|r| pack_row_i8(&b_i8[r * 32..r * 32 + 32])).collect();
        assert_eq!(a_packed_host.len(), 16 * 8);

        let mut a_packed: DeviceBuffer<i32> = DeviceBuffer::alloc(a_packed_host.len()).unwrap();
        a_packed.copy_from_host(&a_packed_host).unwrap();
        let mut b_packed: DeviceBuffer<i32> = DeviceBuffer::alloc(b_packed_host.len()).unwrap();
        b_packed.copy_from_host(&b_packed_host).unwrap();
        let mut c_out: DeviceBuffer<i32> = DeviceBuffer::alloc(16 * 16).unwrap();

        wmma_int8_probe_16x16x32(&a_packed, &b_packed, &mut c_out).expect("real wmma_int8_probe_16x16x32 call failed");
        let mut c_got = vec![0i32; 16 * 16];
        c_out.copy_to_host(&mut c_got).unwrap();

        let max_diff = c_got.iter().zip(c_ref.iter()).map(|(g, r)| (g - r).abs()).max().unwrap_or(0);
        eprintln!("WMMA INT8 probe vs real CPU reference: max abs diff = {max_diff}");
        eprintln!("got[0..4]  = {:?}", &c_got[0..4]);
        eprintln!("ref[0..4]  = {:?}", &c_ref[0..4]);
        assert_eq!(c_got, c_ref, "WMMA INT8 probe must match the real CPU reference bit-exactly (integer dot products, no rounding)");
    }

    /// §126 DEBUG: isolates the 4-chunk (K=128) WMMA accumulation loop
    /// with simple, direct, hardcoded per-row packed-int8 data (same
    /// format as the already bit-exact-verified
    /// `real_wmma_int8_probe_16x16x32_matches_cpu_reference`) -- no
    /// activation quantization, no weight dequant, no LDS. Exists to
    /// determine whether accumulating across 4 chunks specifically is
    /// where a real bug found by the full production-kernel test lives.
    #[test]
    #[ignore]
    fn diagnose_real_wmma_int8_probe_4chunk_vs_cpu_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let a_i8: Vec<i8> = (0..16 * 128).map(|i| (((i * 7 + 3) % 9) as i8) - 4).collect();
        let b_i8: Vec<i8> = (0..16 * 128).map(|i| (((i * 5 + 2) % 7) as i8) - 3).collect();

        let mut c_ref = vec![0i32; 16 * 16];
        for i in 0..16 {
            for j in 0..16 {
                let mut sum = 0i32;
                for k in 0..128 {
                    sum += a_i8[i * 128 + k] as i32 * b_i8[j * 128 + k] as i32;
                }
                c_ref[i * 16 + j] = sum;
            }
        }

        let pack_row_i8 = |row: &[i8]| -> Vec<i32> {
            assert_eq!(row.len(), 128);
            (0..32)
                .map(|w| {
                    let b0 = row[w * 4] as u8 as i32;
                    let b1 = row[w * 4 + 1] as u8 as i32;
                    let b2 = row[w * 4 + 2] as u8 as i32;
                    let b3 = row[w * 4 + 3] as u8 as i32;
                    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
                })
                .collect()
        };
        let a_packed_host: Vec<i32> = (0..16).flat_map(|r| pack_row_i8(&a_i8[r * 128..r * 128 + 128])).collect();
        let b_packed_host: Vec<i32> = (0..16).flat_map(|r| pack_row_i8(&b_i8[r * 128..r * 128 + 128])).collect();

        let mut a_packed: DeviceBuffer<i32> = DeviceBuffer::alloc(a_packed_host.len()).unwrap();
        a_packed.copy_from_host(&a_packed_host).unwrap();
        let mut b_packed: DeviceBuffer<i32> = DeviceBuffer::alloc(b_packed_host.len()).unwrap();
        b_packed.copy_from_host(&b_packed_host).unwrap();
        let mut c_out: DeviceBuffer<i32> = DeviceBuffer::alloc(16 * 16).unwrap();

        wmma_int8_probe_4chunk(&a_packed, &b_packed, &mut c_out).expect("real wmma_int8_probe_4chunk call failed");
        let mut c_got = vec![0i32; 16 * 16];
        c_out.copy_to_host(&mut c_got).unwrap();

        for i in 0..16 {
            for j in 0..16 {
                let got = c_got[i * 16 + j];
                let refv = c_ref[i * 16 + j];
                if got != refv {
                    eprintln!("row={i:2} col={j:2}: got={got:8} ref={refv:8} diff={:8}  <-- BAD", got - refv);
                }
            }
        }
        eprintln!("got[0..4] = {:?}", &c_got[0..4]);
        eprintln!("ref[0..4] = {:?}", &c_ref[0..4]);
        assert_eq!(c_got, c_ref, "4-chunk WMMA accumulation must match the real CPU reference bit-exactly");
    }

    /// §126 DEBUG: dumps per-thread intermediate fragment/accumulator
    /// state from the instrumented kernel for the SAME minimal
    /// single-block, single-group case as
    /// `diagnose_real_wmma_int8_minimal_single_group_single_block`, to
    /// see exactly what data warp0's 32 lanes load and compute.
    #[test]
    #[ignore]
    fn diagnose_real_wmma_int8_debug_dump() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let out_features = 16usize;
        let in_features = 128usize;
        let group_size = 128usize;
        let num_tokens = 16usize;

        // Real, deliberately NON-periodic-in-row weight generator -- see
        // the twin test below (`diagnose_real_wmma_int8_minimal_single_
        // group_single_block`) for why a naive `(i*C+...) % 16` formula
        // with `i = row*16+word` is structurally row-INVARIANT.
        let words_per_row = in_features / 8;
        let qweight_host: Vec<u32> = (0..out_features * words_per_row)
            .map(|i| {
                let row = i / words_per_row;
                let word = i % words_per_row;
                let mut w = 0u32;
                for n in 0..8 {
                    let nibble = (row * 5 + word * 3 + n * 7 + 1) % 16;
                    w |= (nibble as u32) << (n * 4);
                }
                w
            })
            .collect();
        let scales_host: Vec<u16> = (0..out_features * (in_features / group_size)).map(|i| f32_to_bf16(0.01 + (i as f32) * 0.001)).collect();
        let x_f32: Vec<f32> = (0..num_tokens * in_features)
            .map(|i| {
                let t = i / in_features;
                let k = i % in_features;
                (((k + t * 97) % 13) as i32 - 6) as f32 * 0.0523
            })
            .collect();
        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();

        let mut qweight_buf: DeviceBuffer<u32> = DeviceBuffer::alloc(qweight_host.len()).unwrap();
        qweight_buf.copy_from_host(&qweight_host).unwrap();
        let mut scales_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(scales_host.len()).unwrap();
        scales_buf.copy_from_host(&scales_host).unwrap();
        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();
        let mut y_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_tokens * out_features).unwrap();

        let mut debug_my_row: DeviceBuffer<i32> = DeviceBuffer::alloc(256).unwrap();
        let mut debug_a0: DeviceBuffer<i32> = DeviceBuffer::alloc(256).unwrap();
        let mut debug_b0: DeviceBuffer<i32> = DeviceBuffer::alloc(256).unwrap();
        let mut debug_acc0: DeviceBuffer<i32> = DeviceBuffer::alloc(256).unwrap();

        w4a16_gemm_prefill_wmma_int8_debug(
            &x_buf, &qweight_buf, &scales_buf, &mut y_buf, out_features, in_features, group_size, num_tokens,
            &mut debug_my_row, &mut debug_a0, &mut debug_b0, &mut debug_acc0,
        )
        .expect("real debug kernel call failed");

        let mut my_row = vec![0i32; 256];
        debug_my_row.copy_to_host(&mut my_row).unwrap();
        let mut a0 = vec![0i32; 256];
        debug_a0.copy_to_host(&mut a0).unwrap();
        let mut b0 = vec![0i32; 256];
        debug_b0.copy_to_host(&mut b0).unwrap();
        let mut acc0 = vec![0i32; 256];
        debug_acc0.copy_to_host(&mut acc0).unwrap();

        eprintln!("warp0 (tid 0..31), only row_valid lanes matter for this minimal test:");
        for tid in 0..32 {
            eprintln!("tid={tid:3} my_row={:4} a0={:12} b0={:12} acc0_after_first_wmma={:8}", my_row[tid], a0[tid], b0[tid], acc0[tid]);
        }
    }

    /// §126 DEBUG, isolated minimal case: ONE block (16 rows, 16 tokens),
    /// ONE K=128 group, synthetic weights/activations -- eliminates tail
    /// masking, multi-group accumulation, and multi-block confounds to
    /// localize a real bug found by the full production-shape test.
    #[test]
    #[ignore]
    fn diagnose_real_wmma_int8_minimal_single_group_single_block() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let out_features = 16usize;
        let in_features = 128usize;
        let group_size = 128usize;
        let num_tokens = 16usize;

        // Real, deterministic nibble values (0..15), deliberately NOT
        // periodic in `row`: a naive `(i*C+...) % 16` formula with
        // `i = row*words_per_row + word` is structurally row-INVARIANT
        // whenever `words_per_row` is itself a multiple of 16 (true for
        // every real in_features/group_size in this family) -- because
        // `row*words_per_row*C mod 16 == 0` regardless of `row` or `C`.
        // A real, found-the-hard-way test data bug from an earlier pass
        // of this exact debugging session (see `docs/DECISIONS.md` §126)
        // masked the real kernel bug underneath a coincidental "every
        // row has identical weight data" degeneracy. Fixed by deriving
        // the nibble from `row` and `word` SEPARATELY.
        let words_per_row = in_features / 8;
        let qweight_host: Vec<u32> = (0..out_features * words_per_row)
            .map(|i| {
                let row = i / words_per_row;
                let word = i % words_per_row;
                let mut w = 0u32;
                for n in 0..8 {
                    let nibble = (row * 5 + word * 3 + n * 7 + 1) % 16;
                    w |= (nibble as u32) << (n * 4);
                }
                w
            })
            .collect();
        let scales_host: Vec<u16> = (0..out_features * (in_features / group_size)).map(|i| f32_to_bf16(0.01 + (i as f32) * 0.001)).collect();

        let x_f32: Vec<f32> = (0..num_tokens * in_features)
            .map(|i| {
                let t = i / in_features;
                let k = i % in_features;
                (((k + t * 97) % 13) as i32 - 6) as f32 * 0.0523
            })
            .collect();
        let x_bf16: Vec<u16> = x_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let x_bf16_f32: Vec<f32> = x_bf16.iter().map(|&b| bf16_to_f32(b)).collect();

        let mut qweight_buf: DeviceBuffer<u32> = DeviceBuffer::alloc(qweight_host.len()).unwrap();
        qweight_buf.copy_from_host(&qweight_host).unwrap();
        let mut scales_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(scales_host.len()).unwrap();
        scales_buf.copy_from_host(&scales_host).unwrap();
        let mut x_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(x_bf16.len()).unwrap();
        x_buf.copy_from_host(&x_bf16).unwrap();

        let mut y_wmma_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(num_tokens * out_features).unwrap();
        w4a16_gemm_prefill_wmma_int8_bf16(&x_buf, &qweight_buf, &scales_buf, &mut y_wmma_buf, out_features, in_features, group_size, num_tokens)
            .expect("real WMMA INT8 call failed");
        let mut y_wmma = vec![0u16; num_tokens * out_features];
        y_wmma_buf.copy_to_host(&mut y_wmma).unwrap();

        // Real CPU reference, identical intended math.
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

        for t in 0..num_tokens {
            for row in 0..out_features {
                let i = t * out_features + row;
                let wmma = bf16_to_f32(y_wmma[i]);
                let cpu = y_cpu[i];
                let diff = (wmma - cpu).abs();
                eprintln!("t={t:2} row={row:2}: wmma={wmma:10.5} cpu={cpu:10.5} diff={diff:10.5}{}", if diff > 0.05 { "  <-- BAD" } else { "" });
            }
        }
    }
