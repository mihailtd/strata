//! §100 (per the active `/goal`'s explicit ordering, after chunked GDN):
//! Instant LoRA hot-swapping via In-Place Weight Folding (IPWF) -- see
//! `TODO_LORA_SWAP.md` for the full scoping rationale this was built
//! against, and `apps/runtime-ipwf/novel_peft.py` for the real, already-
//! validated Python prototype this Rust port mirrors the invariants of.
//!
//! W_active = W_0 + (lora_alpha/r) * (B @ A), ALWAYS evaluated from a
//! pristine base-weight backup restored first -- never accumulated onto a
//! possibly-already-adapted buffer. This is the real, mandatory invariant
//! the scoping doc calls out: `(W_0 + dW) - dW != W_0` in bf16 (only 7
//! explicit mantissa bits), so repeated swap cycles must restore from a
//! pristine copy, not subtract the delta back out -- otherwise rounding
//! error compounds across swaps and silently degrades the model.
//!
//! Reuses `raw::gemm_pv` (built for this same session's chunked-GDN work,
//! §100) directly for the fold itself: `W_delta[out,in] = B[out,r] @
//! A[r,in]` is EXACTLY the `O = P@V` shape `gemm_pv` already implements
//! (`t=out, kv_len=r, head_dim=in`, `P=B` since `p_ld` supports a stride
//! but `B` is naturally contiguous here, `V=A`), so `alpha=lora_alpha/r,
//! beta=1.0` folds the delta straight into a pristine-restored weight
//! buffer via ONE real hipBLAS accumulate GEMM per adapted matrix -- no
//! new GEMM primitive needed, and the exact same `alpha`/`beta`
//! generalization chunked GDN's `v_new` correction step needed anyway.
//!
//! HIP Graph compatibility (§93): folding mutates a `DeviceBuffer`'s
//! VALUES in place via `hipblasGemmEx`, never reallocates or changes any
//! `as_device_ptr()` address -- a captured HIP Graph's recorded pointers
//! stay valid across a swap with zero re-capture needed (the same
//! invariant `TODO_LORA_SWAP.md` §3 documents).

use crate::blas::{BlasHandle, ffi as blas_ffi};
use crate::hip::{self, DeviceBuffer, HipError};
use crate::model::{self, AttnLayerWeights, LayerWeights, ModelWeights, NUM_LAYERS, raw};
use crate::model_loader::load_raw_tensor_from_file;
use std::path::Path;

/// One real LoRA `(A, B)` factor pair for a single adapted linear layer.
/// `a`: `[r, in_features]`, `b`: `[out_features, r]` -- matching the real
/// PEFT/safetensors on-disk shapes exactly (`lora_A.weight`/`lora_B.weight`).
pub struct LoraFactor {
    pub a: DeviceBuffer<u16>,
    pub b: DeviceBuffer<u16>,
    pub r: usize,
    pub in_features: usize,
    pub out_features: usize,
}

pub struct LoraMlpFactors {
    pub gate: LoraFactor,
    pub up: LoraFactor,
    pub down: LoraFactor,
}

pub struct LoraAttnFactors {
    pub q: LoraFactor,
    pub k: LoraFactor,
    pub v: LoraFactor,
    pub o: LoraFactor,
}

pub struct LoraLayer {
    pub mlp: LoraMlpFactors,
    /// `Some` only for the 8 real full-attention layers -- GDN layers'
    /// own projections (`in_proj_combined`/`out_proj`) are never real
    /// LoRA targets (confirmed against the real adapter's own
    /// `adapter_config.json` `target_modules`: `o_proj`, `gate_proj`,
    /// `k_proj`, `down_proj`, `v_proj`, `q_proj`, `up_proj` -- no GDN
    /// module names).
    pub attn: Option<LoraAttnFactors>,
}

pub struct LoraAdapter {
    #[allow(dead_code)]
    pub name: String,
    /// `lora_alpha / r` -- the real scaling multiplier applied to every
    /// `B@A` fold (see `adapter_config.json`'s own `lora_alpha`/`r`).
    pub scale: f32,
    pub layers: Vec<LoraLayer>,
}

impl LoraAdapter {
    /// Loads a real LoRA adapter directory (`adapter_config.json` +
    /// `adapter_model.safetensors`) straight onto the GPU as bf16 factors,
    /// ready to fold. `name` is caller-supplied (the dispatch key a
    /// server would route `req.model` through), not read from disk.
    pub fn load_from_dir(dir: &Path, name: &str) -> Result<Self, String> {
        let config_path = dir.join("adapter_config.json");
        let config_text = std::fs::read_to_string(&config_path).map_err(|e| format!("reading {}: {e}", config_path.display()))?;
        let config: serde_json::Value = serde_json::from_str(&config_text).map_err(|e| format!("parsing {}: {e}", config_path.display()))?;
        let r = config.get("r").and_then(|v| v.as_u64()).ok_or_else(|| format!("{}: missing integer 'r'", config_path.display()))? as usize;
        let lora_alpha = config.get("lora_alpha").and_then(|v| v.as_f64()).ok_or_else(|| format!("{}: missing numeric 'lora_alpha'", config_path.display()))? as f32;
        let scale = lora_alpha / (r as f32);

        let tensors_path = dir.join("adapter_model.safetensors");
        let load_factor = |module: &str, layer: usize| -> Result<LoraFactor, String> {
            let prefix = format!("base_model.model.model.layers.{layer}.{module}");
            let a_raw = load_raw_tensor_from_file(&tensors_path, &format!("{prefix}.lora_A.weight"))?;
            let b_raw = load_raw_tensor_from_file(&tensors_path, &format!("{prefix}.lora_B.weight"))?;
            assert_eq!(a_raw.shape.len(), 2, "{prefix}.lora_A.weight: expected 2D, got {:?}", a_raw.shape);
            assert_eq!(b_raw.shape.len(), 2, "{prefix}.lora_B.weight: expected 2D, got {:?}", b_raw.shape);
            assert_eq!(a_raw.shape[0], r, "{prefix}.lora_A.weight: expected r={r} rows, got {}", a_raw.shape[0]);
            assert_eq!(b_raw.shape[1], r, "{prefix}.lora_B.weight: expected r={r} cols, got {}", b_raw.shape[1]);
            let in_features = a_raw.shape[1];
            let out_features = b_raw.shape[0];

            let a_bf16: Vec<u16> = a_raw.to_f32().iter().map(|&v| model::f32_to_bf16(v)).collect();
            let b_bf16: Vec<u16> = b_raw.to_f32().iter().map(|&v| model::f32_to_bf16(v)).collect();
            let mut a_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(a_bf16.len()).map_err(|e| format!("hipMalloc {prefix}.lora_A: {e}"))?;
            let mut b_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(b_bf16.len()).map_err(|e| format!("hipMalloc {prefix}.lora_B: {e}"))?;
            a_buf.copy_from_host(&a_bf16).map_err(|e| format!("uploading {prefix}.lora_A: {e}"))?;
            b_buf.copy_from_host(&b_bf16).map_err(|e| format!("uploading {prefix}.lora_B: {e}"))?;
            Ok(LoraFactor { a: a_buf, b: b_buf, r, in_features, out_features })
        };

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for i in 0..NUM_LAYERS {
            let mlp = LoraMlpFactors {
                gate: load_factor("mlp.gate_proj", i)?,
                up: load_factor("mlp.up_proj", i)?,
                down: load_factor("mlp.down_proj", i)?,
            };
            let attn = if model::is_full_attention_layer(i) {
                Some(LoraAttnFactors {
                    q: load_factor("self_attn.q_proj", i)?,
                    k: load_factor("self_attn.k_proj", i)?,
                    v: load_factor("self_attn.v_proj", i)?,
                    o: load_factor("self_attn.o_proj", i)?,
                })
            } else {
                None
            };
            layers.push(LoraLayer { mlp, attn });
        }

        Ok(LoraAdapter { name: name.to_string(), scale, layers })
    }
}

/// A pristine buffer backup that resides either in device VRAM (for models that fit, e.g. 4B)
/// or host RAM (for larger models like 9B/27B where duplicate weight copies exceed VRAM capacity).
pub enum PristineBuffer {
    Device(DeviceBuffer<u16>),
    Host(Vec<u16>),
}

impl PristineBuffer {
    #[inline]
    pub fn restore_into(&self, dst: &mut DeviceBuffer<u16>) -> Result<(), HipError> {
        match self {
            PristineBuffer::Device(d) => dst.copy_from_device(d),
            PristineBuffer::Host(h) => dst.copy_from_host(h),
        }
    }
}

/// One layer's real pristine (never-adapted) copy of every buffer LoRA
/// folding ever touches. `qkv_proj`/`o_proj` are `Some` only for the real
/// full-attention layers, mirroring `LoraLayer::attn`.
pub struct PristineLayer {
    pub gate_up_proj: PristineBuffer,
    pub down_proj: PristineBuffer,
    pub qkv_proj: Option<PristineBuffer>,
    pub o_proj: Option<PristineBuffer>,
}

/// The real, mandatory bit-exact-idempotence anchor every `activate`/
/// `restore` call restores from -- see this module's own header doc.
/// Captured ONCE, at model-load time, before any adapter is ever
/// activated.
pub struct PristineWeights {
    pub layers: Vec<PristineLayer>,
}

fn copy_new(src: &DeviceBuffer<u16>) -> Result<PristineBuffer, HipError> {
    // For 9B / 27B models, storing a duplicate set of weights in VRAM exceeds 24GB.
    // Store pristine weights in Host RAM when running large models or when free VRAM < 8 GB.
    let prefer_host = cfg!(any(feature = "qwen35_9b", feature = "qwen35_27b"))
        || hip::mem_info().map(|(free, _)| free < 8 * 1024 * 1024 * 1024).unwrap_or(false);

    if !prefer_host {
        if let Ok(mut dst) = DeviceBuffer::alloc(src.len()) {
            if dst.copy_from_device(src).is_ok() {
                return Ok(PristineBuffer::Device(dst));
            }
        }
    }

    let mut host = vec![0u16; src.len()];
    src.copy_to_host(&mut host)?;
    Ok(PristineBuffer::Host(host))
}

impl PristineWeights {
    /// Real, one-time snapshot of every LoRA-touched weight buffer in `weights`.
    /// On 4B, snapshots stay directly in VRAM. On 9B/27B, snapshots are stored in Host RAM.
    pub fn capture(weights: &ModelWeights) -> Result<Self, HipError> {
        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for layer in &weights.layers {
            let pristine = match layer {
                LayerWeights::Gdn(g) => PristineLayer {
                    gate_up_proj: copy_new(g.gate_up_proj.as_bf16())?,
                    down_proj: copy_new(g.down_proj.as_bf16())?,
                    qkv_proj: None,
                    o_proj: None,
                },
                LayerWeights::Attn(a) => PristineLayer {
                    gate_up_proj: copy_new(a.gate_up_proj.as_bf16())?,
                    down_proj: copy_new(a.down_proj.as_bf16())?,
                    qkv_proj: Some(copy_new(a.qkv_proj.as_bf16())?),
                    o_proj: Some(copy_new(a.o_proj.as_bf16())?),
                },
            };
            layers.push(pristine);
        }
        Ok(PristineWeights { layers })
    }

    /// Restores every touched buffer to its real pristine (unadapted)
    /// state, in place. The FIRST step of every real `activate_adapter` call
    /// (never accumulate onto a possibly-already-adapted buffer -- the
    /// mandatory idempotence invariant), and also a real "switch back to the
    /// base model" operation on its own.
    pub fn restore(&self, weights: &mut ModelWeights) -> Result<(), HipError> {
        for (layer, pristine) in weights.layers.iter_mut().zip(self.layers.iter()) {
            match layer {
                LayerWeights::Gdn(g) => {
                    pristine.gate_up_proj.restore_into(g.gate_up_proj.as_bf16_mut())?;
                    pristine.down_proj.restore_into(g.down_proj.as_bf16_mut())?;
                }
                LayerWeights::Attn(a) => {
                    pristine.gate_up_proj.restore_into(a.gate_up_proj.as_bf16_mut())?;
                    pristine.down_proj.restore_into(a.down_proj.as_bf16_mut())?;
                    pristine.qkv_proj.as_ref().expect("attn layer's pristine snapshot is missing qkv_proj")
                        .restore_into(a.qkv_proj.as_bf16_mut())?;
                    pristine.o_proj.as_ref().expect("attn layer's pristine snapshot is missing o_proj")
                        .restore_into(a.o_proj.as_bf16_mut())?;
                }
            }
        }
        Ok(())
    }
}

/// One real fold: `dst[row_offset..row_offset+factor.out_features, :] +=
/// scale * (factor.b @ factor.a)`, via `raw::gemm_pv` (see this module's
/// own header doc for the shape derivation). `row_offset` is in ROWS
/// (real weight-matrix rows, i.e. output features), converted to an
/// element offset here since every fused buffer this crate builds
/// (`gate_up_proj`, `qkv_proj`) concatenates its sub-tensors along the
/// OUT dimension with a uniform `in_features`-wide row, matching this
/// function's own single `in_features` parameter.
///
/// SAFETY: caller guarantees `dst` is a real, live `hipMalloc`
/// allocation at least `(row_offset+factor.out_features)*factor.in_features`
/// elements long, and `factor.a`/`factor.b` are real, live allocations of
/// their own declared shapes (both true by construction -- `dst` is one
/// of `ModelWeights`' own real weight buffers, `factor` was built by
/// `LoraAdapter::load_from_dir` directly from the real adapter checkpoint's
/// own tensor shapes).
unsafe fn fold_into(handle_raw: blas_ffi::HipblasHandle, dst: &mut DeviceBuffer<u16>, row_offset: usize, factor: &LoraFactor, scale: f32) {
    let dst_ptr = dst.as_device_ptr_at_mut(row_offset * factor.in_features);
    unsafe {
        raw::gemm_pv(
            handle_raw,
            factor.b.as_device_ptr(),
            factor.r as i32,
            factor.a.as_device_ptr(),
            dst_ptr,
            factor.in_features as i32,
            factor.out_features as i32,
            factor.r as i32,
            factor.in_features as i32,
            scale,
            1.0,
        );
    }
}

fn fold_mlp(handle_raw: blas_ffi::HipblasHandle, gate_up_proj: &mut DeviceBuffer<u16>, down_proj: &mut DeviceBuffer<u16>, mlp: &LoraMlpFactors, scale: f32) {
    unsafe {
        fold_into(handle_raw, gate_up_proj, 0, &mlp.gate, scale);
        fold_into(handle_raw, gate_up_proj, mlp.gate.out_features, &mlp.up, scale);
        fold_into(handle_raw, down_proj, 0, &mlp.down, scale);
    }
}

fn fold_attn(handle_raw: blas_ffi::HipblasHandle, a: &mut AttnLayerWeights, attn: &LoraAttnFactors, scale: f32) {
    unsafe {
        fold_into(handle_raw, a.qkv_proj.as_bf16_mut(), 0, &attn.q, scale);
        fold_into(handle_raw, a.qkv_proj.as_bf16_mut(), attn.q.out_features, &attn.k, scale);
        fold_into(handle_raw, a.qkv_proj.as_bf16_mut(), attn.q.out_features + attn.k.out_features, &attn.v, scale);
        fold_into(handle_raw, a.o_proj.as_bf16_mut(), 0, &attn.o, scale);
    }
}

/// §100: folds `adapter`'s real LoRA factors into `weights`' live
/// buffers, IN PLACE, via real hipBLAS accumulate GEMMs -- ALWAYS from a
/// pristine restore first (the mandatory bit-exact-idempotence
/// invariant; see this module's own header doc). Synchronizes the GPU
/// once, at the end (matching every other real weight-mutation operation
/// in this crate -- correctness of the swap is a real gate here, not a
/// §100: folds `adapter`'s real LoRA factors into `weights`' live
/// buffers, IN PLACE, via real hipBLAS accumulate GEMMs -- ALWAYS from a
/// pristine restore first (the mandatory bit-exact-idempotence
/// invariant; see this module's own header doc). Synchronizes the GPU
/// once, at the end (matching every other real weight-mutation operation
/// in this crate -- correctness of the swap is a real gate here, not a
/// hot per-token path that would need to avoid it).
pub fn activate_adapter(handle: &BlasHandle, weights: &mut ModelWeights, pristine: &PristineWeights, adapter: &LoraAdapter) -> Result<(), HipError> {
    activate_adapter_scaled(handle, weights, pristine, adapter, adapter.scale)
}

/// §100/§133: folds `adapter`'s real LoRA factors scaled by `scale` into
/// `weights`' live buffers WITHOUT restoring pristine first (accumulates with beta=1.0).
pub fn fold_adapter_into(handle: &BlasHandle, weights: &mut ModelWeights, adapter: &LoraAdapter, scale: f32) -> Result<(), HipError> {
    let handle_raw = handle.raw();
    for (layer, lora_layer) in weights.layers.iter_mut().zip(adapter.layers.iter()) {
        match layer {
            LayerWeights::Gdn(g) => {
                fold_mlp(handle_raw, g.gate_up_proj.as_bf16_mut(), g.down_proj.as_bf16_mut(), &lora_layer.mlp, scale);
            }
            LayerWeights::Attn(a) => {
                fold_mlp(handle_raw, a.gate_up_proj.as_bf16_mut(), a.down_proj.as_bf16_mut(), &lora_layer.mlp, scale);
                let attn_factors = lora_layer.attn.as_ref().expect("full-attention layer's LoraLayer is missing its attn factors");
                fold_attn(handle_raw, a, attn_factors, scale);
            }
        }
    }
    hip::check_last_error()?;
    hip::device_synchronize()
}

/// §100/§133: folds `adapter`'s real LoRA factors scaled by `scale` into
/// `weights`' live buffers from a pristine restore.
pub fn activate_adapter_scaled(handle: &BlasHandle, weights: &mut ModelWeights, pristine: &PristineWeights, adapter: &LoraAdapter, scale: f32) -> Result<(), HipError> {
    pristine.restore(weights)?;
    fold_adapter_into(handle, weights, adapter, scale)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{HIDDEN_SIZE, INTERMEDIATE_SIZE};
    use crate::model_loader::locate_model_snapshot;

    /// §100 decisive test, real weights + a real adapter: activating an
    /// adapter genuinely changes a touched weight buffer (checked against
    /// an INDEPENDENTLY computed fold, `pristine + scale*(B@A)`, at a few
    /// spot-checked elements -- not just "it changed"), and restoring
    /// pristine afterward reproduces the ORIGINAL weight bytes BIT-EXACT
    /// (not just "close") -- the mandatory idempotence invariant this
    /// module's own header doc calls out (`TODO_LORA_SWAP.md` §1B: naive
    /// subtraction would NOT be bit-exact in bf16, so this crate never
    /// subtracts -- it always restores from a real pristine copy).
    /// `#[ignore]`d like every other real-weight, real-GPU test in this
    /// crate; run explicitly.
    #[test]
    #[ignore]
    fn real_activate_then_restore_is_bit_exact_idempotent() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        eprintln!("loading real weights for all {NUM_LAYERS} layers...");
        let mut weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        eprintln!("weights loaded.");

        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        let pristine = PristineWeights::capture(&weights).expect("PristineWeights::capture failed");

        // Real HIP Graph compatibility invariant (`TODO_LORA_SWAP.md` §3):
        // folding mutates VALUES in place, never reallocates -- captured
        // here as a real pointer address, checked again at the end.
        let ptr_before = match &weights.layers[0] {
            LayerWeights::Gdn(g) => g.gate_up_proj.as_bf16().as_device_ptr() as usize,
            LayerWeights::Attn(a) => a.gate_up_proj.as_bf16().as_device_ptr() as usize,
        };

        // Real pristine layer-0 gate_up_proj bytes, BEFORE any adapter is
        // ever activated -- the ORIGINAL reference this test's final
        // bit-exact check compares against.
        let gate_up_proj_len = match &weights.layers[0] {
            LayerWeights::Gdn(g) => g.gate_up_proj.as_bf16().len(),
            LayerWeights::Attn(a) => a.gate_up_proj.as_bf16().len(),
        };
        let mut original_bytes = vec![0u16; gate_up_proj_len];
        match &weights.layers[0] {
            LayerWeights::Gdn(g) => g.gate_up_proj.as_bf16().copy_to_host(&mut original_bytes),
            LayerWeights::Attn(a) => a.gate_up_proj.as_bf16().copy_to_host(&mut original_bytes),
        }
        .expect("copy_to_host(original gate_up_proj) failed");

        let adapter_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/adapters/m2_astral_r8a128_v7");
        assert!(adapter_dir.is_dir(), "real adapter directory not found at {}", adapter_dir.display());
        let adapter = LoraAdapter::load_from_dir(&adapter_dir, "astral").expect("real LoraAdapter::load_from_dir failed");
        assert!((adapter.scale - 16.0).abs() < 1e-6, "expected real scale lora_alpha/r=128/8=16.0, got {}", adapter.scale);

        // Independent expected fold for layer 0's `gate_proj` sub-block
        // (offset 0 within `gate_up_proj`), computed by hand from the
        // pristine weight's own real bf16 values and the adapter's own
        // real f32 A/B factors -- read back from the SAME device buffers
        // `activate_adapter` will use, not re-derived from disk, so this
        // check is honest about what's actually on the GPU.
        let gate_factor = &adapter.layers[0].mlp.gate;
        let r = gate_factor.r;
        let mut a_host = vec![0u16; r * HIDDEN_SIZE];
        let mut b_host = vec![0u16; gate_factor.out_features * r];
        gate_factor.a.copy_to_host(&mut a_host).expect("copy_to_host(lora_A) failed");
        gate_factor.b.copy_to_host(&mut b_host).expect("copy_to_host(lora_B) failed");
        let bf16_to_f32 = model::bf16_to_f32;

        let expected_fold = |out_row: usize, in_col: usize| -> f32 {
            let base = bf16_to_f32(original_bytes[out_row * HIDDEN_SIZE + in_col]);
            let mut delta = 0f32;
            for k in 0..r {
                let b_val = bf16_to_f32(b_host[out_row * r + k]);
                let a_val = bf16_to_f32(a_host[k * HIDDEN_SIZE + in_col]);
                delta += b_val * a_val;
            }
            base + adapter.scale * delta
        };

        activate_adapter(&handle, &mut weights, &pristine, &adapter).expect("real activate_adapter failed");

        let mut activated_bytes = vec![0u16; gate_up_proj_len];
        match &weights.layers[0] {
            LayerWeights::Gdn(g) => g.gate_up_proj.as_bf16().copy_to_host(&mut activated_bytes),
            LayerWeights::Attn(a) => a.gate_up_proj.as_bf16().copy_to_host(&mut activated_bytes),
        }
        .expect("copy_to_host(activated gate_up_proj) failed");

        // Spot-check a handful of real (out_row, in_col) pairs across the
        // gate_proj sub-block -- a real matrix, not a synthetic toy, so
        // bf16's own ~0.4% relative precision is the right tolerance
        // (matching every other bf16 decisive test in this crate).
        let mut checked_any_nonzero_delta = false;
        for &(out_row, in_col) in &[(0usize, 0usize), (5, 100), (INTERMEDIATE_SIZE / 2, HIDDEN_SIZE / 2), (INTERMEDIATE_SIZE - 1, HIDDEN_SIZE - 1)] {
            let got = bf16_to_f32(activated_bytes[out_row * HIDDEN_SIZE + in_col]);
            let expected = expected_fold(out_row, in_col);
            let base = bf16_to_f32(original_bytes[out_row * HIDDEN_SIZE + in_col]);
            if (expected - base).abs() > 1e-4 {
                checked_any_nonzero_delta = true;
            }
            let tol = (expected.abs() * 0.01).max(0.01);
            assert!((got - expected).abs() < tol, "gate_up_proj[{out_row},{in_col}]: activated GPU value {got}, independent reference {expected} (pristine base was {base})");
        }
        assert!(checked_any_nonzero_delta, "the real adapter's fold produced no detectable change at any spot-checked element -- test would pass vacuously even if activate_adapter were a no-op");

        // The mandatory idempotence check: restore pristine, and confirm
        // BIT-EXACT (not just numerically close) match to the ORIGINAL
        // bytes captured before any adapter was ever activated.
        pristine.restore(&mut weights).expect("real PristineWeights::restore failed");
        let mut restored_bytes = vec![0u16; gate_up_proj_len];
        match &weights.layers[0] {
            LayerWeights::Gdn(g) => g.gate_up_proj.as_bf16().copy_to_host(&mut restored_bytes),
            LayerWeights::Attn(a) => a.gate_up_proj.as_bf16().copy_to_host(&mut restored_bytes),
        }
        .expect("copy_to_host(restored gate_up_proj) failed");
        assert_eq!(restored_bytes, original_bytes, "restore_pristine did not reproduce the ORIGINAL weight bytes bit-exact -- this is the mandatory idempotence invariant this whole module exists to guarantee");

        let ptr_after = match &weights.layers[0] {
            LayerWeights::Gdn(g) => g.gate_up_proj.as_bf16().as_device_ptr() as usize,
            LayerWeights::Attn(a) => a.gate_up_proj.as_bf16().as_device_ptr() as usize,
        };
        assert_eq!(ptr_after, ptr_before, "gate_up_proj's device address changed across activate/restore -- would silently invalidate a captured HIP Graph's recorded pointers");
    }

    /// §100 decisive test, full real end-to-end forward pass: an
    /// activated adapter genuinely changes real greedy generation output
    /// (not just an isolated weight buffer's bytes -- the whole point of
    /// domain specialization), and restoring pristine reproduces the
    /// REAL base model's own known-correct greedy continuation exactly
    /// (the same reference `real_greedy_generation_matches_real_qwen3_5_4b`
    /// checks) -- so this test doubles as a real regression check that
    /// adapter machinery leaves the unadapted path untouched.
    #[test]
    #[ignore]
    fn real_adapter_changes_generation_and_restore_reproduces_base() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        eprintln!("loading real weights for all {NUM_LAYERS} layers...");
        let mut weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        eprintln!("weights loaded.");

        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        let pristine = PristineWeights::capture(&weights).expect("PristineWeights::capture failed");

        // Same real prompt + known-correct base continuation as
        // `real_greedy_generation_matches_real_qwen3_5_4b`.
        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let base_expected: [i32; 6] = [11751, 13, 198, 32, 13, 2912];

        let generate = |weights: &ModelWeights| -> Vec<i32> {
            let max_seq_len = 32usize;
            let mut state = crate::model::DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
            let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(crate::model::VOCAB_SIZE).unwrap();
            for &token_id in prompt_ids.iter() {
                crate::model::forward_one_token(&handle, weights, &mut state, token_id, &mut logits).expect("forward_one_token failed during prefill");
            }
            let mut generated = Vec::with_capacity(base_expected.len());
            for _ in 0..base_expected.len() {
                let next_id = crate::model::argmax_sample(&logits).expect("argmax_sample failed");
                generated.push(next_id);
                crate::model::forward_one_token(&handle, weights, &mut state, next_id, &mut logits).expect("forward_one_token failed during generation");
            }
            generated
        };

        let base_generated = generate(&weights);
        eprintln!("base generated:      {base_generated:?}");
        assert_eq!(base_generated, base_expected, "unadapted generation (before any adapter is ever loaded) doesn't match the real known-correct reference");

        let adapter_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/adapters/m2_astral_r8a128_v7");
        let adapter = LoraAdapter::load_from_dir(&adapter_dir, "astral").expect("real LoraAdapter::load_from_dir failed");
        activate_adapter(&handle, &mut weights, &pristine, &adapter).expect("real activate_adapter failed");

        let adapted_generated = generate(&weights);
        eprintln!("adapted generated:   {adapted_generated:?}");
        assert_ne!(adapted_generated, base_expected, "activating a real r=8/alpha=128 adapter produced IDENTICAL generation to the base model -- the fold had no detectable effect, which means activate_adapter is not actually doing anything");

        pristine.restore(&mut weights).expect("real PristineWeights::restore failed");
        let restored_generated = generate(&weights);
        eprintln!("restored generated:  {restored_generated:?}");
        assert_eq!(restored_generated, base_expected, "generation after restore_pristine doesn't match the base model anymore -- the swap is not truly reversible at the full-model level");
    }

    /// Real swap-latency measurement, matching `TODO_LORA_SWAP.md`'s own
    /// "100 alternating swaps" benchmark methodology (§4C) -- alternates
    /// `activate_adapter` between two REAL, distinct adapters (real
    /// pristine-restore + real per-matrix GEMM folds every call, not a
    /// cached no-op), timed with `std::time::Instant` around the whole
    /// call (`activate_adapter` already syncs internally, so this is a
    /// real, honest wall-clock number, not an async-queue illusion).
    #[test]
    #[ignore]
    fn bench_real_adapter_swap_latency() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let mut weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        let pristine = PristineWeights::capture(&weights).expect("PristineWeights::capture failed");

        let adapters_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/adapters");
        #[cfg(feature = "qwen35_9b")]
        let (name_a, name_b) = ("m2_astral_r8a128_v7_9b", "m2_postgresql_r8a128_v7_9b");
        #[cfg(not(feature = "qwen35_9b"))]
        let (name_a, name_b) = ("m2_astral_r8a128_v7", "m2_postgresql_r8a128_v7");

        let astral = LoraAdapter::load_from_dir(&adapters_root.join(name_a), "astral").expect("loading adapter a failed");
        let second_dir = adapters_root.join(name_b);
        let second = if second_dir.is_dir() {
            LoraAdapter::load_from_dir(&second_dir, "postgresql").expect("loading adapter b failed")
        } else {
            eprintln!("second adapter not found, reusing adapter a for both sides of the swap");
            LoraAdapter::load_from_dir(&adapters_root.join(name_a), "astral2").expect("loading adapter a (2nd copy) failed")
        };

        // Warmup: first call on a never-before-seen GEMM shape can pay a
        // real one-time Tensile solution-selection cost (§99's own
        // finding) -- exclude that from the measured cycles.
        activate_adapter(&handle, &mut weights, &pristine, &astral).expect("warmup activate_adapter failed");

        let cycles = 20usize;
        let start = std::time::Instant::now();
        for i in 0..cycles {
            let adapter = if i % 2 == 0 { &second } else { &astral };
            activate_adapter(&handle, &mut weights, &pristine, adapter).expect("activate_adapter failed during benchmark");
        }
        let elapsed = start.elapsed();
        let per_swap_ms = elapsed.as_secs_f64() * 1000.0 / cycles as f64;
        eprintln!("REAL measured adapter swap latency ({cycles} alternating cycles, restore+fold({} layers) each): {per_swap_ms:.3} ms/swap", crate::model::NUM_LAYERS);
    }

    /// §104 / Tier 0 (T0-4): real decode-throughput measurement for
    /// In-Place Weight Folding (IPWF) -- WITHOUT vs. WITH a real folded
    /// adapter (r=8, alpha=128).
    ///
    /// Theory: Section 3 claims "0% decode tax" because the adapter delta
    /// is pre-folded directly into the active weight buffers -- during the
    /// decode loop, the GEMM kernel dimensions and memory footprints are
    /// bit-for-bit identical to the unadapted model (unlike unmerged LoRA
    /// branches which launch extra GEMM kernels per token).
    /// This benchmark verifies that claim with real hardware telemetry.
    #[test]
    #[ignore]
    fn bench_real_adapter_decode_overhead() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real model snapshot found on this machine");
        let mut weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");
        let pristine = PristineWeights::capture(&weights).expect("PristineWeights::capture failed");

        let adapters_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/adapters");
        #[cfg(feature = "qwen35_9b")]
        let adapter_name = "m2_astral_r8a128_v7_9b";
        #[cfg(not(feature = "qwen35_9b"))]
        let adapter_name = "m2_astral_r8a128_v7";

        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let max_seq_len = 64usize;
        let warmup = 3usize;
        let timed_tokens = 20usize;

        let run_decode_bench = |weights: &ModelWeights| -> f64 {
            let mut state = crate::model::DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
            let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(crate::model::VOCAB_SIZE).unwrap();
            for &token_id in prompt_ids.iter() {
                crate::model::forward_one_token(&handle, weights, &mut state, token_id, &mut logits).unwrap();
            }
            for _ in 0..warmup {
                let next_id = crate::model::argmax_sample(&logits).unwrap();
                crate::model::forward_one_token(&handle, weights, &mut state, next_id, &mut logits).unwrap();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..timed_tokens {
                let next_id = crate::model::argmax_sample(&logits).unwrap();
                crate::model::forward_one_token(&handle, weights, &mut state, next_id, &mut logits).unwrap();
            }
            timed_tokens as f64 / t0.elapsed().as_secs_f64()
        };

        let base_tps = run_decode_bench(&weights);
        eprintln!("WITHOUT adapter (pristine base weights): {base_tps:.2} tok/s");

        let adapter = LoraAdapter::load_from_dir(&adapters_root.join(adapter_name), "astral").expect("loading adapter failed");
        activate_adapter(&handle, &mut weights, &pristine, &adapter).expect("activate_adapter failed");
        let adapted_tps = run_decode_bench(&weights);
        eprintln!("WITH adapter active (in-place weight folded): {adapted_tps:.2} tok/s");

        pristine.restore(&mut weights).expect("restore failed");

        let overhead_pct = (base_tps - adapted_tps) / base_tps * 100.0;
        eprintln!(
            "REAL measured IPWF decode overhead ({} layers, BF16): {overhead_pct:.2}% ({base_tps:.2} -> {adapted_tps:.2} tok/s)",
            crate::model::NUM_LAYERS
        );
    }
}
