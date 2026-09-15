//! Stage 1 of the real port: load real Qwen3.5-4B weights from disk and get
//! them onto the GPU, byte-for-byte correct -- verified, not assumed.
//!
//! WHY THIS EXISTS
//! ----------------
//! Nothing downstream (a forward pass, a HIP Graph, a server) means anything
//! if the weights it operates on aren't the real ones. This module answers
//! exactly one question first: can we read a real `.safetensors` shard from
//! this project's own Hugging Face cache, extract a named tensor's raw
//! bytes with the offset math right, and move those bytes onto the GPU
//! without corruption -- before any compute touches them at all.
//!
//! Uses the real `safetensors` crate (Hugging Face's own, matches the exact
//! file format the Python runtime already reads via `transformers`) for
//! header/offset parsing -- a well-specified binary format is exactly the
//! kind of well-solved problem worth reusing a canonical parser for, unlike
//! the small hand-curated HIP FFI surface in `hip.rs`, where the reasoning
//! was the opposite (avoid pulling in a large, unaudited surface for a
//! handful of functions). `memmap2` avoids reading a 9.3GB file into RAM to
//! extract one tensor.
//!
//! bf16 -> f32 decoding is hand-written (one line: bf16 is literally the
//! top 16 bits of an f32, this is not a nontrivial problem worth a
//! dependency for) and cross-checked in the test below against values
//! independently computed in Python straight from the same real file, not
//! just checked for internal self-consistency.

use crate::hip::{DeviceBuffer, HipError};
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Real tensor metadata as read from a `.safetensors` header: name, shape,
/// dtype, and the raw bytes exactly as stored (no conversion happens in
/// this module beyond what `to_f32`/`to_bf16_bits` do explicitly). Most
/// weights in this checkpoint are bf16, but a few real parameters (e.g.
/// GatedDeltaNet's `A_log`) are stored as f32 -- read from the real
/// per-tensor header rather than assumed from the tensor's name, so a
/// wrong assumption about a specific tensor's dtype can't silently corrupt
/// its values.
pub struct RawTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    pub data: Vec<u8>,
}

impl RawTensor {
    /// Decodes this tensor's raw bytes to `f32`, using its REAL dtype
    /// (read from the safetensors header, not assumed) to pick the right
    /// conversion: bf16 is the top 16 bits of an f32; f32 needs no
    /// conversion at all, just reinterpreting 4-byte chunks.
    pub fn to_f32(&self) -> Vec<f32> {
        match self.dtype {
            Dtype::BF16 => bf16_bytes_to_f32(&self.data),
            Dtype::F32 => self
                .data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            other => panic!("RawTensor::to_f32: unsupported dtype {other:?} for {:?}", self.name),
        }
    }

    /// Reinterprets this tensor's raw bytes as bf16-bit-pattern `u16`s
    /// (the layout every kernel in this crate expects for bf16 weights).
    /// Panics if this tensor isn't actually bf16 -- calling this on an f32
    /// tensor would silently produce garbage, so it's refused loudly
    /// instead (matches this project's own "no silent fallbacks"
    /// discipline established in `fused_norm.py` and `hip.rs`).
    pub fn to_bf16_bits(&self) -> Vec<u16> {
        assert_eq!(
            self.dtype,
            Dtype::BF16,
            "to_bf16_bits called on non-bf16 tensor {:?} (dtype {:?})",
            self.name,
            self.dtype
        );
        self.data
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect()
    }
}

/// Locates this project's real Qwen3.5-4B snapshot on disk, via the same
/// Hugging Face cache layout `transformers.AutoModelForCausalLM` already
/// reads in the Python runtime. Override with `RUNTIME_NEXT_MODEL_DIR` for
/// a different machine/cache location -- never silently falls back to a
/// synthetic or bundled model.
pub fn locate_model_snapshot() -> Result<PathBuf, String> {
    if let Ok(dir) = std::env::var("RUNTIME_NEXT_MODEL_DIR") {
        return Ok(PathBuf::from(dir));
    }

    let home = std::env::var("HOME").map_err(|_| "HOME not set".to_string())?;
    let base =
        PathBuf::from(home).join(".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots");

    let mut entries: Vec<_> = fs::read_dir(&base)
        .map_err(|e| format!("reading {}: {e}", base.display()))?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());
    entries
        .last()
        .map(|e| e.path())
        .ok_or_else(|| format!("no snapshot directories found under {}", base.display()))
}

/// Reads `model.safetensors.index.json` to map tensor name -> shard
/// filename, exactly the file the Python runtime's own `from_pretrained`
/// reads for a sharded checkpoint.
pub fn load_weight_map(snapshot_dir: &Path) -> Result<HashMap<String, String>, String> {
    let index_path = snapshot_dir.join("model.safetensors.index.json");
    let text = fs::read_to_string(&index_path)
        .map_err(|e| format!("reading {}: {e}", index_path.display()))?;
    let parsed: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parsing index json: {e}"))?;
    let map = parsed
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| "index json has no weight_map object".to_string())?;

    Ok(map
        .iter()
        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
        .collect())
}

/// Reads one real named tensor's raw bytes (still bf16-encoded) out of the
/// shard it actually lives in, via mmap. Returns an error rather than
/// panicking if the tensor name doesn't exist in this checkpoint -- a
/// missing tensor is a real, checkable condition, not something to paper
/// over.
pub fn load_raw_tensor(snapshot_dir: &Path, tensor_name: &str) -> Result<RawTensor, String> {
    let weight_map = load_weight_map(snapshot_dir)?;
    let shard_name = weight_map
        .get(tensor_name)
        .ok_or_else(|| format!("tensor {tensor_name:?} not found in this checkpoint's index"))?;

    let shard_path = snapshot_dir.join(shard_name);
    let file = fs::File::open(&shard_path)
        .map_err(|e| format!("opening {}: {e}", shard_path.display()))?;
    // SAFETY: this is the one `unsafe` in this module. `memmap2::Mmap::map`
    // is unsafe because the file could be truncated/modified concurrently
    // by another process while mapped, which would be undefined behavior on
    // access. This checkpoint is a read-only Hugging Face cache blob this
    // process does not itself write to; nothing else in this project's
    // Python or Rust code mutates a cached model snapshot in place.
    let mmap = unsafe {
        memmap2::Mmap::map(&file).map_err(|e| format!("mmap {}: {e}", shard_path.display()))?
    };

    let tensors = SafeTensors::deserialize(&mmap).map_err(|e| {
        format!(
            "parsing safetensors header in {}: {e}",
            shard_path.display()
        )
    })?;
    let view = tensors
        .tensor(tensor_name)
        .map_err(|e| format!("tensor {tensor_name:?} present in index but not in shard: {e}"))?;

    Ok(RawTensor {
        name: tensor_name.to_string(),
        shape: view.shape().to_vec(),
        dtype: view.dtype(),
        data: view.data().to_vec(),
    })
}

/// Loads a real bf16 weight tensor straight onto the GPU as a
/// `DeviceBuffer<u16>` (the bf16-bit-pattern layout every kernel in this
/// crate expects). Convenience wrapper around `load_raw_tensor` +
/// `RawTensor::to_bf16_bits` + upload, for the common case of "get this
/// named real weight onto the GPU, ready to pass to a kernel."
pub fn load_bf16_weight(
    snapshot_dir: &Path,
    tensor_name: &str,
) -> Result<DeviceBuffer<u16>, String> {
    let raw = load_raw_tensor(snapshot_dir, tensor_name)?;
    let bits = raw.to_bf16_bits();
    let mut buf: DeviceBuffer<u16> =
        DeviceBuffer::alloc(bits.len()).map_err(|e| format!("hipMalloc for {tensor_name:?}: {e}"))?;
    buf.copy_from_host(&bits)
        .map_err(|e| format!("uploading {tensor_name:?}: {e}"))?;
    Ok(buf)
}

/// §92 performance pass: loads several real bf16 weight tensors that share
/// the same `in_features` (e.g. `q_proj`/`k_proj`/`v_proj`, all real
/// `nn.Linear` weights of shape `[out_features_i, in_features]`) and
/// concatenates them along `out_features` into ONE combined
/// `DeviceBuffer<u16>` -- so the forward pass can compute all of them with
/// a single GEMM call (`M = sum(out_features_i)`) instead of one call per
/// tensor, then split the single wider output back into each tensor's own
/// range via a plain contiguous pointer offset (no kernel needed, exactly
/// like `in_proj_qkv`'s existing query/key/value split already worked).
///
/// This is a real concatenation of each tensor's OWN real trained weights,
/// not a new tensor or a reinterpretation -- byte-concatenating two
/// row-major `[out_i, in]` buffers along the row axis produces exactly the
/// `[out_i+out_j, in]` row-major buffer that same concatenation would be
/// if the checkpoint had stored it that way to begin with, since each
/// tensor's own rows are already contiguous. Panics (a real bug, not a
/// runtime condition) if the given tensors don't all share the same
/// `in_features` -- that would silently produce a corrupt weight matrix.
pub fn load_concat_bf16_weights(
    snapshot_dir: &Path,
    tensor_names: &[&str],
) -> Result<DeviceBuffer<u16>, String> {
    let mut combined: Vec<u16> = Vec::new();
    let mut in_features: Option<usize> = None;
    for &name in tensor_names {
        let raw = load_raw_tensor(snapshot_dir, name)?;
        assert_eq!(raw.shape.len(), 2, "load_concat_bf16_weights: {name:?} is not 2D");
        let this_in = raw.shape[1];
        match in_features {
            None => in_features = Some(this_in),
            Some(expected) => assert_eq!(
                this_in, expected,
                "load_concat_bf16_weights: {name:?} has in_features {this_in}, expected {expected} (all concatenated tensors must share in_features)"
            ),
        }
        combined.extend_from_slice(&raw.to_bf16_bits());
    }
    let mut buf: DeviceBuffer<u16> = DeviceBuffer::alloc(combined.len())
        .map_err(|e| format!("hipMalloc for concatenated {tensor_names:?}: {e}"))?;
    buf.copy_from_host(&combined)
        .map_err(|e| format!("uploading concatenated {tensor_names:?}: {e}"))?;
    Ok(buf)
}

/// Loads a real parameter tensor (bf16 OR f32, per its REAL header dtype)
/// straight onto the GPU as a `DeviceBuffer<f32>` -- for the handful of
/// real per-layer scalar parameters (`A_log`, `dt_bias`) the GDN kernels
/// take as f32 regardless of how they happen to be stored in the
/// checkpoint.
pub fn load_f32_param(snapshot_dir: &Path, tensor_name: &str) -> Result<DeviceBuffer<f32>, String> {
    let raw = load_raw_tensor(snapshot_dir, tensor_name)?;
    let values = raw.to_f32();
    let mut buf: DeviceBuffer<f32> =
        DeviceBuffer::alloc(values.len()).map_err(|e| format!("hipMalloc for {tensor_name:?}: {e}"))?;
    buf.copy_from_host(&values)
        .map_err(|e| format!("uploading {tensor_name:?}: {e}"))?;
    Ok(buf)
}

/// Decodes bf16-encoded bytes to `f32`. bf16 is the top 16 bits of an f32
/// (same exponent width, truncated mantissa) -- this is the entire
/// conversion, exactly what PyTorch/safetensors' own BF16 dtype means.
pub fn bf16_bytes_to_f32(data: &[u8]) -> Vec<f32> {
    assert_eq!(data.len() % 2, 0, "bf16 data length must be even");
    data.chunks_exact(2)
        .map(|b| {
            let bits16 = u16::from_le_bytes([b[0], b[1]]);
            f32::from_bits((bits16 as u32) << 16)
        })
        .collect()
}

/// Uploads a raw tensor's bytes to the GPU as-is (no dtype conversion) via
/// the `DeviceBuffer` foundation from `hip.rs`.
pub fn upload_raw_tensor(tensor: &RawTensor) -> Result<DeviceBuffer<u8>, HipError> {
    let mut buf: DeviceBuffer<u8> = DeviceBuffer::alloc(tensor.data.len())?;
    buf.copy_from_host(&tensor.data)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real test: locates the real Qwen3.5-4B snapshot on this machine,
    /// parses the real safetensors index, and confirms embed_tokens (the
    /// largest tensor in the checkpoint) is present with the shape the
    /// model's own config.json implies (vocab_size=248320, hidden_size=2560
    /// -- read directly from the real config.json alongside the weights,
    /// not hardcoded from memory).
    #[test]
    fn real_index_lists_embed_tokens_with_correct_shape() {
        let snapshot =
            locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let raw = load_raw_tensor(&snapshot, "model.language_model.embed_tokens.weight")
            .expect("embed_tokens.weight must exist in a real Qwen3.5-4B checkpoint");
        assert_eq!(raw.shape, vec![248320, 2560]);
        assert_eq!(raw.data.len(), 248320 * 2560 * 2, "bf16 = 2 bytes/element");
    }

    /// The real, decisive test: loads a small real tensor, decodes its
    /// first 8 values from bf16 to f32, and checks them against values
    /// independently computed in Python directly from the same real file
    /// (struct-unpacking the header offsets and shifting bf16 into f32 by
    /// hand, not via this same Rust code) -- proving the offset math and
    /// the bf16 decode are both actually correct, not just internally
    /// self-consistent.
    #[test]
    fn real_tensor_bytes_match_independently_computed_python_values() {
        let snapshot =
            locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let raw = load_raw_tensor(
            &snapshot,
            "model.language_model.layers.0.input_layernorm.weight",
        )
        .expect("layer 0 input_layernorm.weight must exist");

        assert_eq!(raw.shape, vec![2560]);
        assert_eq!(raw.data.len(), 5120);

        let values = bf16_bytes_to_f32(&raw.data[..16]); // first 8 elements

        // Independently computed 2026-09-14 via `struct`/`array` in Python,
        // reading offsets [15360, 20480] directly from the same real shard
        // file's own header -- see docs/DECISIONS.md for the exact command.
        let expected = [
            0.66796875,
            0.38671875,
            0.255859375,
            0.2392578125,
            0.1796875,
            0.56640625,
            0.353515625,
            0.412109375,
        ];
        for (i, (got, want)) in values.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-9,
                "value {i}: got {got}, python-independently-computed {want}"
            );
        }
    }

    /// Real GPU round-trip: the same tensor loaded above, uploaded to the
    /// GPU via `DeviceBuffer`, copied back, and checked byte-for-byte
    /// against the bytes read directly from disk.
    #[test]
    fn real_tensor_survives_a_real_gpu_round_trip() {
        let snapshot =
            locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        if crate::hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        crate::hip::set_device(0)
            .expect("hipSetDevice(0) failed on a machine that reported a device");

        let raw = load_raw_tensor(
            &snapshot,
            "model.language_model.layers.0.input_layernorm.weight",
        )
        .expect("layer 0 input_layernorm.weight must exist");

        let buf = upload_raw_tensor(&raw).expect("uploading real tensor bytes to GPU failed");
        let mut host_out = vec![0u8; raw.data.len()];
        buf.copy_to_host(&mut host_out)
            .expect("copying tensor back from GPU failed");

        assert_eq!(
            host_out, raw.data,
            "GPU round-trip corrupted real model weight bytes"
        );
    }
}
