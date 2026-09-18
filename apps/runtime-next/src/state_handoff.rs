//! §102 (per the active `/goal`'s explicit ordering, the third and final
//! item after chunked GDN and LoRA swap): real O(1) multi-agent tensor
//! state handoff -- see `TODO_TENSOR_STATE_HANDOFF.md` for the full
//! scoping rationale this was built against.
//!
//! THE REAL FOUNDATION THIS BUILDS ON WAS ALREADY THERE: `forward_prefill`/
//! `forward_prefill_chunk` never reset `state.position` themselves -- they
//! read it as the new tokens' starting position and ADVANCE it
//! (`state.position += num_tokens`). Calling `forward_prefill` again on
//! the SAME `DecodeState` (without an intervening `state.reset()`)
//! already IS incremental prefill: the GDN recurrent/conv state and the
//! attention KV cache both carry over exactly as if the whole sequence
//! had been prefilled in one call. This module adds the ONE piece that
//! was genuinely missing: `TensorStateSnapshot` lets that same
//! continuation happen ACROSS two different `DecodeState` instances (a
//! real device-to-device VRAM clone, no host round-trip, no re-prefill)
//! -- the actual mechanism a multi-agent handoff needs (Agent A's state
//! captured once, restored into Agent B's fresh `DecodeState`, generation
//! continues with zero re-prefill of Agent A's own output).
//!
//! Real, deliberate simplification vs. the scoping doc's own
//! position-proportional memory budget: this snapshots the FULL
//! `max_seq_len`-sized K/V cache buffers, not just the first `position`
//! tokens' worth -- a partial copy would need one `hipMemcpy` PER
//! ATTENTION HEAD (the cache's real layout is head-major, so only the
//! first `position` rows within each head's own slice are meaningful,
//! not a single contiguous prefix of the whole buffer), the same "many
//! small calls add up" cost this session already measured for §101's
//! LoRA pristine-restore. A full-buffer copy is fewer, larger calls at
//! the cost of copying some unused tail capacity -- correctness first,
//! matching this session's own standing discipline; a per-head partial
//! copy is a real, identified follow-up if a later profiling pass finds
//! this snapshot latency matters at the `max_seq_len` this crate's real
//! server actually configures.

use crate::hip::{DeviceBuffer, HipError};
use crate::model::{DecodeState, LayerState};

/// One layer's real, owned copy of whatever state that layer type
/// carries -- a GDN layer's conv/recurrent state, or an attention layer's
/// K/V cache.
pub enum LayerStateSnapshot {
    Gdn { conv_state: DeviceBuffer<u16>, recurrent_state: DeviceBuffer<f32> },
    Attn { k_cache: DeviceBuffer<u16>, v_cache: DeviceBuffer<u16> },
}

/// A real, complete, position-independent clone of a `DecodeState`'s
/// entire mutable sequence state -- every GDN layer's recurrent/conv
/// state, every attention layer's K/V cache, and the real `position`
/// counter RoPE/causal-masking/GDN-decay all depend on. Restoring this
/// into ANY `DecodeState` (fresh or already in use) makes subsequent
/// generation from that state indistinguishable from continuing the
/// ORIGINAL session directly -- the real mechanism behind "Agent B
/// continues Agent A's conversation with zero re-prefill."
pub struct TensorStateSnapshot {
    pub position: usize,
    pub layers: Vec<LayerStateSnapshot>,
}

fn copy_new<T: Copy>(src: &DeviceBuffer<T>) -> Result<DeviceBuffer<T>, HipError> {
    let mut dst: DeviceBuffer<T> = DeviceBuffer::alloc(src.len())?;
    dst.copy_from_device(src)?;
    Ok(dst)
}

impl TensorStateSnapshot {
    /// Real device-to-device clone of `state`'s entire sequence state --
    /// no host round-trip (every source and destination buffer already
    /// lives in VRAM), matching §101's `PristineWeights::capture`'s own
    /// reasoning for using `DeviceBuffer::copy_from_device` here too.
    pub fn capture(state: &DecodeState) -> Result<Self, HipError> {
        let mut layers = Vec::with_capacity(state.layers.len());
        for layer in &state.layers {
            let snap = match layer {
                LayerState::Gdn(g) => LayerStateSnapshot::Gdn {
                    conv_state: copy_new(&g.conv_state)?,
                    recurrent_state: copy_new(&g.recurrent_state)?,
                },
                LayerState::Attn(a) => LayerStateSnapshot::Attn {
                    k_cache: copy_new(&a.k_cache)?,
                    v_cache: copy_new(&a.v_cache)?,
                },
            };
            layers.push(snap);
        }
        Ok(TensorStateSnapshot { position: state.position, layers })
    }

    /// Restores this snapshot into `state`, in place, entirely on-device.
    /// `state` must have the SAME layer topology (layer count and
    /// GDN-vs-Attn type per index) as whatever `DecodeState` this
    /// snapshot was captured from -- true by construction for any two
    /// `DecodeState`s built from the same real `ModelWeights` (the layer
    /// topology is a fixed, real architecture fact, not something a
    /// caller can vary per-request), so a mismatch here is a real bug,
    /// not a runtime condition to recover from.
    pub fn restore(&self, state: &mut DecodeState) -> Result<(), HipError> {
        assert_eq!(state.layers.len(), self.layers.len(), "TensorStateSnapshot::restore: layer count mismatch ({} snapshot vs {} state)", self.layers.len(), state.layers.len());
        for (layer, snap) in state.layers.iter_mut().zip(self.layers.iter()) {
            match (layer, snap) {
                (LayerState::Gdn(g), LayerStateSnapshot::Gdn { conv_state, recurrent_state }) => {
                    g.conv_state.copy_from_device(conv_state)?;
                    g.recurrent_state.copy_from_device(recurrent_state)?;
                }
                (LayerState::Attn(a), LayerStateSnapshot::Attn { k_cache, v_cache }) => {
                    a.k_cache.copy_from_device(k_cache)?;
                    a.v_cache.copy_from_device(v_cache)?;
                }
                _ => panic!("TensorStateSnapshot::restore: layer-type mismatch between the snapshot and the target DecodeState -- this is a real architecture-topology bug, not a runtime condition"),
            }
        }
        state.position = self.position;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blas::BlasHandle;
    use crate::hip;
    use crate::model::{ModelWeights, VOCAB_SIZE, argmax_sample, forward_one_token, forward_prefill};
    use crate::model_loader::locate_model_snapshot;

    /// §102 decisive test: the foundation this whole feature rests on --
    /// `forward_prefill` called TWICE in a row on the SAME `DecodeState`
    /// (12 tokens, then 8 more, no `state.reset()` in between) produces
    /// the SAME real argmax continuation, and numerically CLOSE final
    /// logits, as one `forward_prefill` call over all 20 tokens at once.
    /// NOT bit-exact, and correctly so: the chunked-GDN algorithm (§100)
    /// computes token 15's influence on token 18 via two mathematically
    /// equivalent but differently-ORDERED real floating-point paths --
    /// one shot covers positions 0-19 in a single chunk (a direct O(C^2)
    /// intra-chunk pairwise-decay term), while the incremental case's
    /// second call only ever sees its own 8 tokens as a fresh chunk and
    /// reads everything before position 12 through the compressed
    /// `recurrent_state` matrix instead -- the same real
    /// "mathematically equivalent, not bit-identical" property this
    /// session's own `gen_chunked_gdn_reference.py` already found between
    /// the chunked and recurrent PYTHON reference implementations
    /// (`~1e-9` in float32/float64), and the same tolerance-based
    /// pattern `real_batched_prefill_logits_numerically_match_sequential_forward_one_token`
    /// already established for exactly this class of comparison
    /// (different real GPU reduction order, not a correctness bug). If
    /// this property weren't true, snapshotting state between calls would
    /// be pointless -- there'd be nothing correct to continue from.
    #[test]
    #[ignore]
    fn real_incremental_prefill_matches_one_shot_prefill_numerically() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        eprintln!("loading real weights...");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        // Real token ids (same prompt family as the crate's own greedy-
        // generation oracle test), split 12 + 8 for the two-step case.
        let all_tokens: [i32; 20] = [760, 6511, 314, 9338, 369, 11751, 13, 198, 32, 13, 2912, 785, 5591, 5535, 315, 10852, 374, 279, 6094, 3552];
        let (first, second) = all_tokens.split_at(12);

        let max_seq_len = 64usize;
        let mut one_shot_state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut one_shot_logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        forward_prefill(&handle, &weights, &mut one_shot_state, &all_tokens, &mut one_shot_logits).expect("one-shot forward_prefill failed");

        let mut incremental_state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut incremental_logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        forward_prefill(&handle, &weights, &mut incremental_state, first, &mut incremental_logits).expect("incremental forward_prefill (first half) failed");
        assert_eq!(incremental_state.position, 12, "position should be exactly 12 after prefilling the first 12 tokens");
        forward_prefill(&handle, &weights, &mut incremental_state, second, &mut incremental_logits).expect("incremental forward_prefill (second half) failed");
        assert_eq!(incremental_state.position, 20, "position should be exactly 20 after prefilling the remaining 8 tokens");
        assert_eq!(one_shot_state.position, 20, "one-shot prefill position should also be 20");

        let mut one_shot_host = vec![0u16; VOCAB_SIZE];
        let mut incremental_host = vec![0u16; VOCAB_SIZE];
        one_shot_logits.copy_to_host(&mut one_shot_host).unwrap();
        incremental_logits.copy_to_host(&mut incremental_host).unwrap();

        fn bf16_to_f32(bits: u16) -> f32 {
            f32::from_bits((bits as u32) << 16)
        }

        // The strong property: the real argmax (what greedy decoding
        // actually relies on) must agree exactly.
        let one_shot_next = argmax_sample(&one_shot_logits).expect("argmax_sample failed");
        let incremental_next = argmax_sample(&incremental_logits).expect("argmax_sample failed");
        assert_eq!(one_shot_next, incremental_next, "one-shot and incremental prefill disagree on the argmax next token");

        // The numeric property: every logit should be CLOSE, not
        // identical -- same tolerance and honest max-diff reporting as
        // `real_batched_prefill_logits_numerically_match_sequential_forward_one_token`.
        let tolerance = 0.5f32;
        let mut max_diff = 0f32;
        let mut num_exceeding = 0usize;
        for (&a, &b) in one_shot_host.iter().zip(incremental_host.iter()) {
            let diff = (bf16_to_f32(a) - bf16_to_f32(b)).abs();
            max_diff = max_diff.max(diff);
            if diff > tolerance {
                num_exceeding += 1;
            }
        }
        eprintln!("one-shot vs incremental prefill: max_diff={max_diff}, {num_exceeding}/{VOCAB_SIZE} elements exceed tolerance={tolerance}");
        assert_eq!(num_exceeding, 0, "one-shot and incremental prefill diverged beyond tolerance on {num_exceeding} of {VOCAB_SIZE} logits (max_diff={max_diff})");
    }

    /// §102 decisive test, the real new capability: Agent A prefills a
    /// real prompt and generates a few tokens; its state is snapshotted
    /// and restored into a COMPLETELY FRESH, separate `DecodeState`
    /// ("Agent B"); both then continue with the SAME follow-up tokens.
    /// Real success criterion: Agent B's continuation is BIT-EXACT
    /// identical to Agent A's own continuation from that same point --
    /// the whole point of a handoff is that the receiving side cannot
    /// tell the difference from continuing the original session.
    #[test]
    #[ignore]
    fn real_snapshot_restored_into_fresh_state_continues_bit_exact() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot_dir = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        eprintln!("loading real weights...");
        let weights = ModelWeights::load(&snapshot_dir).expect("real weight loading failed");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let max_seq_len = 64usize;

        // "Agent A": real prefill + 3 real generated tokens.
        let mut agent_a = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        forward_prefill(&handle, &weights, &mut agent_a, &prompt_ids, &mut logits).expect("real forward_prefill failed");
        for _ in 0..3 {
            let next_id = argmax_sample(&logits).expect("argmax_sample failed");
            forward_one_token(&handle, &weights, &mut agent_a, next_id, &mut logits).expect("real forward_one_token failed");
        }
        let position_at_handoff = agent_a.position;

        // Real O(1) handoff: snapshot Agent A, restore into a completely
        // fresh "Agent B" DecodeState -- no re-prefill of the prompt or
        // the 3 generated tokens.
        let handoff_snapshot = TensorStateSnapshot::capture(&agent_a).expect("real TensorStateSnapshot::capture failed");
        let mut agent_b = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        handoff_snapshot.restore(&mut agent_b).expect("real TensorStateSnapshot::restore failed");
        assert_eq!(agent_b.position, position_at_handoff, "restored position doesn't match the snapshotted position");

        // Continue BOTH agents with the same 4 follow-up tokens (Agent A
        // continuing its own session directly; Agent B continuing via
        // the handed-off snapshot) and compare every step's logits.
        let mut logits_a: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        let mut logits_b: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        let follow_up_tokens: [i32; 4] = [11751, 13, 198, 32];
        for &tok in follow_up_tokens.iter() {
            forward_one_token(&handle, &weights, &mut agent_a, tok, &mut logits_a).expect("agent A continuation step failed");
            forward_one_token(&handle, &weights, &mut agent_b, tok, &mut logits_b).expect("agent B continuation step failed");
        }

        let mut host_a = vec![0u16; VOCAB_SIZE];
        let mut host_b = vec![0u16; VOCAB_SIZE];
        logits_a.copy_to_host(&mut host_a).unwrap();
        logits_b.copy_to_host(&mut host_b).unwrap();
        assert_eq!(host_a, host_b, "Agent B's continuation after a snapshot handoff diverged from Agent A's own direct continuation -- the handoff lost or corrupted real state");
        assert_eq!(agent_a.position, agent_b.position, "position diverged between the two agents after continuing with the same number of tokens");
    }

    /// Real, decisive verification that state handoff works through the
    /// REAL quantized (W4A16) forward path too, not just bf16 -- direct
    /// answer to a real, previously-unverified question: `TensorStateSnapshot`
    /// only ever touches `DecodeState` (GDN recurrent/conv state, attention
    /// K/V cache, `position`), never `ModelWeights` -- architecturally
    /// quantization-agnostic by construction, since W4A16 only changes
    /// WEIGHT tensors, never the bf16/f32 activation/cache tensors
    /// `DecodeState` owns. This test confirms that reasoning empirically
    /// against the real quantized 27B checkpoint (real per-token
    /// `LinearWeight::Quantized` dispatch, real WMMA-accelerated prefill)
    /// rather than leaving it as an assumption. Same real success
    /// criterion as the bf16 test above: bit-exact continuation.
    #[test]
    #[ignore]
    #[cfg(feature = "qwen35_27b")]
    fn real_snapshot_restored_into_fresh_state_continues_bit_exact_quantized() {
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
        eprintln!("loading real quantized weights for all {} layers...", crate::model::NUM_LAYERS);
        let weights = ModelWeights::load(&quantized_dir).expect("real quantized weight loading failed");
        assert!(weights.is_quantized(), "sanity: this checkpoint must actually be quantized for this test to mean anything");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let max_seq_len = 64usize;

        let mut agent_a = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        forward_prefill(&handle, &weights, &mut agent_a, &prompt_ids, &mut logits).expect("real quantized forward_prefill failed");
        for _ in 0..3 {
            let next_id = argmax_sample(&logits).expect("argmax_sample failed");
            forward_one_token(&handle, &weights, &mut agent_a, next_id, &mut logits).expect("real quantized forward_one_token failed");
        }
        let position_at_handoff = agent_a.position;

        let handoff_snapshot = TensorStateSnapshot::capture(&agent_a).expect("real TensorStateSnapshot::capture failed (quantized)");
        let mut agent_b = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        handoff_snapshot.restore(&mut agent_b).expect("real TensorStateSnapshot::restore failed (quantized)");
        assert_eq!(agent_b.position, position_at_handoff, "restored position doesn't match the snapshotted position");

        let mut logits_a: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        let mut logits_b: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        let follow_up_tokens: [i32; 4] = [11751, 13, 198, 32];
        for &tok in follow_up_tokens.iter() {
            forward_one_token(&handle, &weights, &mut agent_a, tok, &mut logits_a).expect("agent A (quantized) continuation step failed");
            forward_one_token(&handle, &weights, &mut agent_b, tok, &mut logits_b).expect("agent B (quantized) continuation step failed");
        }

        let mut host_a = vec![0u16; VOCAB_SIZE];
        let mut host_b = vec![0u16; VOCAB_SIZE];
        logits_a.copy_to_host(&mut host_a).unwrap();
        logits_b.copy_to_host(&mut host_b).unwrap();
        assert_eq!(host_a, host_b, "Agent B's continuation after a snapshot handoff diverged from Agent A's own direct continuation on the QUANTIZED path -- the handoff lost or corrupted real state");
        assert_eq!(agent_a.position, agent_b.position, "position diverged between the two agents after continuing with the same number of tokens (quantized)");
    }
}
