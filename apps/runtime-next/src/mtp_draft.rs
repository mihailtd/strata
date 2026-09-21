//! §137: real port of the legacy Python engines' shipped MTP (multi-
//! token-prediction) draft head (`apps/runtime-ipwf/mtp_draft.py`'s
//! `Qwen35MTPDraftHead`) -- the trained draft mechanism behind real
//! speculative decoding's economics (`docs/DECISIONS.md` §136 Part 1:
//! real τ=3.52 on real aider-bench prompts, measured through the ACTUAL
//! Python head, clears the real Rust break-even of 2.32 with margin).
//!
//! Real architecture, confirmed directly against the real Qwen3.5-4B
//! checkpoint's own 15 `mtp.*` tensors (never assumed from the Python
//! source alone -- see the real `safe_open` shape dump this port was
//! built against): one ordinary full-attention decoder layer, SAME real
//! dims as this crate's own backbone `AttnLayerWeights` at this model
//! size (`ATTN_NUM_HEADS`/`ATTN_NUM_KV_HEADS`/`ATTN_HEAD_DIM`/
//! `INTERMEDIATE_SIZE` all identical -- confirmed: 4B's real
//! `mtp.layers.0.self_attn.q_proj.weight` is `[8192, 2560]` = `2 *
//! ATTN_NUM_HEADS(16) * ATTN_HEAD_DIM(256)`, the SAME real query+gate-
//! doubled width the backbone's own `q_proj` uses). This module reuses
//! `model::attn_layer_forward` directly for that one layer -- zero new
//! attention/MLP kernel code, rather than re-deriving the same math a
//! second time.
//!
//! What's genuinely new, ported line-for-line from the real, working
//! Python `_fuse`/`_run_layer`/`draft`/`prefill` methods (not
//! re-derived): two RMSNorms (`pre_fc_norm_hidden`, `pre_fc_norm_
//! embedding`), one fuse-GEMM (`fc`, real `nn.Linear(2h, h, bias=False)`,
//! confirmed `[2560, 5120]` on 4B), and the autoregressive draft control
//! flow. The real, empirically-load-bearing fuse ORDER (Python's own
//! `_fuse` docstring: "fc expects `[embedding ; hidden]`, NOT `[hidden ;
//! embedding]`... determined empirically -- with `[hidden ; embedding]`
//! the head degenerates to echoing its input token with 0% real
//! accuracy") is preserved exactly: `fused_concat`'s first half is always
//! the embedding's norm, the second half the hidden state's norm.
//!
//! Real, deliberate scope: reuses the BACKBONE's own `embed_tokens`/
//! `lm_head` (tied, exactly like the Python head does via
//! `model.get_input_embeddings()`/`get_output_embeddings()`) -- never its
//! own separate copies, and RoPE is applied via the SAME real
//! `raw::rope`/`attn_layer_forward` call this crate's backbone attention
//! layers already use (identical rotary config -- `head_cfg` in Python is
//! derived from the SAME base config, only `layer_types`/
//! `num_hidden_layers` overridden). Always bf16 -- this head only ships
//! on the 4B/9B checkpoints this crate's own `qwen35_4b`/`qwen35_9b`
//! features target, never the quantized 27B checkpoint (matching the
//! Python head's own real scope: never quantized there either, and never
//! a real LoRA target -- domain-tuning the head was repeatedly refuted,
//! `docs/DECISIONS.md` §5/§63/§69).
//!
//! Rollback (needed by the real speculative accept/reject loop, not yet
//! built -- see `docs/DECISIONS.md` §136's own real economics case) is
//! deliberately just a position-counter rewind (`rollback_to`), matching
//! this crate's own already-proven KV-cache rollback discipline (no
//! realloc, no crop -- stale entries past the rewound position are
//! allocated but simply never read again, the same property `causal_
//! softmax`'s masking and `kv_cache_append`'s position-indexed writes
//! already guarantee for the backbone's own attention layers). The head's
//! own KV cache entries for an ACCEPTED prefix are already correct and
//! need no recomputation: `draft()`'s step `i` fuses the head's own
//! greedy choice from step `i-1`, which for an accepted token is by
//! definition the SAME real token the verify step confirmed -- only a
//! REJECTED tail needs discarding, which `rollback_to` does for free.

use crate::hip::{DeviceBuffer, HipError};
use crate::model::{
    self, ATTN_HEAD_DIM, ATTN_NUM_KV_HEADS, AttnLayerState, AttnLayerWeights, HIDDEN_SIZE,
    LinearWeight, ModelWeights, RMS_EPS, Scratch, VOCAB_SIZE, argmax_sample, raw,
};
use crate::model_loader::{load_bf16_weight, load_concat_bf16_weights};
use std::ffi::c_void;
use std::path::Path;

/// Real, loaded weights for the MTP draft head's own one real decoder
/// layer plus its four MTP-specific tensors. Never mutated after load
/// (no real LoRA target, no quantized variant -- see this module's own
/// header doc for why).
pub struct MtpDraftHead {
    pre_fc_norm_hidden: DeviceBuffer<u16>,
    pre_fc_norm_embedding: DeviceBuffer<u16>,
    /// Real `nn.Linear(2*HIDDEN_SIZE, HIDDEN_SIZE, bias=False)` weight,
    /// `[HIDDEN_SIZE, 2*HIDDEN_SIZE]` row-major (PyTorch `[out, in]`).
    fc: LinearWeight,
    layer: AttnLayerWeights,
    norm: DeviceBuffer<u16>,
}

impl MtpDraftHead {
    /// Loads the real `mtp.*` tensors from `snapshot_dir` (the SAME real
    /// checkpoint directory `ModelWeights::load` reads the backbone
    /// from). Real, loud failure (not a silent skip) if this checkpoint
    /// ships no `mtp.*` weights at all -- matching the Python head's own
    /// `load_mtp_weights`' `if not wanted: raise RuntimeError`, since
    /// `load_raw_tensor` (underlying `load_bf16_weight`/`load_concat_
    /// bf16_weights`) already errors clearly on a missing tensor name.
    pub fn load(snapshot_dir: &Path) -> Result<Self, String> {
        let p = "mtp.layers.0";
        let q_name = format!("{p}.self_attn.q_proj.weight");
        let k_name = format!("{p}.self_attn.k_proj.weight");
        let v_name = format!("{p}.self_attn.v_proj.weight");
        let qkv_proj = LinearWeight::Bf16(load_concat_bf16_weights(snapshot_dir, &[&q_name, &k_name, &v_name])?);
        let gate_name = format!("{p}.mlp.gate_proj.weight");
        let up_name = format!("{p}.mlp.up_proj.weight");
        let gate_up_proj = LinearWeight::Bf16(load_concat_bf16_weights(snapshot_dir, &[&gate_name, &up_name])?);
        let o_proj = LinearWeight::Bf16(load_bf16_weight(snapshot_dir, &format!("{p}.self_attn.o_proj.weight"))?);
        let down_proj = LinearWeight::Bf16(load_bf16_weight(snapshot_dir, &format!("{p}.mlp.down_proj.weight"))?);

        let layer = AttnLayerWeights {
            input_layernorm: load_bf16_weight(snapshot_dir, &format!("{p}.input_layernorm.weight"))?,
            post_attention_layernorm: load_bf16_weight(snapshot_dir, &format!("{p}.post_attention_layernorm.weight"))?,
            qkv_proj,
            q_norm: load_bf16_weight(snapshot_dir, &format!("{p}.self_attn.q_norm.weight"))?,
            k_norm: load_bf16_weight(snapshot_dir, &format!("{p}.self_attn.k_norm.weight"))?,
            o_proj,
            gate_up_proj,
            down_proj,
        };

        Ok(MtpDraftHead {
            pre_fc_norm_hidden: load_bf16_weight(snapshot_dir, "mtp.pre_fc_norm_hidden.weight")?,
            pre_fc_norm_embedding: load_bf16_weight(snapshot_dir, "mtp.pre_fc_norm_embedding.weight")?,
            fc: LinearWeight::Bf16(load_bf16_weight(snapshot_dir, "mtp.fc.weight")?),
            layer,
            norm: load_bf16_weight(snapshot_dir, "mtp.norm.weight")?,
        })
    }
}

/// Real, per-generation mutable state: the head's own KV cache (genuinely
/// separate from the backbone's -- a different, smaller, single-layer
/// cache, never shared) plus scratch buffers for the fuse/layer/norm
/// pipeline, pre-allocated once and reused every real step (matching this
/// crate's own established `Scratch`/`DecodeState` discipline: no
/// `hipMalloc` in a hot loop).
pub struct MtpDraftState {
    layer_state: AttnLayerState,
    scratch: Scratch,
    /// The head's OWN position counter -- genuinely separate from the
    /// backbone's `DecodeState::position`, though it tracks the same real
    /// sequence 1:1 (every real committed token, whether from the prompt
    /// or an accepted draft, advances both by the same amount).
    position: usize,
    max_seq_len: usize,
    token_buf: DeviceBuffer<i32>,
    position_buf: DeviceBuffer<i32>,
    embed_buf: DeviceBuffer<u16>,
    /// `[2*HIDDEN_SIZE]` -- `[embed_normed ; hidden_normed]`, the real,
    /// empirically-load-bearing order (see this module's header doc).
    fused_concat: DeviceBuffer<u16>,
    fused: DeviceBuffer<u16>,
    layer_out: DeviceBuffer<u16>,
    /// The head's own running hidden state, fed back into `_fuse` as `h`
    /// on every draft step after the first (Python: `h = self._run_layer(
    /// ...)`, reassigned each loop iteration) -- real autoregression in
    /// the head's own feature space, never touching the backbone again
    /// once a draft round starts.
    norm_out: DeviceBuffer<u16>,
    logits: DeviceBuffer<u16>,
}

impl MtpDraftState {
    pub fn new(max_seq_len: usize) -> Result<Self, HipError> {
        let cache_len = ATTN_NUM_KV_HEADS * max_seq_len * ATTN_HEAD_DIM;
        let mut k_cache: DeviceBuffer<u16> = DeviceBuffer::alloc(cache_len)?;
        k_cache.fill_zero()?;
        let mut v_cache: DeviceBuffer<u16> = DeviceBuffer::alloc(cache_len)?;
        v_cache.fill_zero()?;
        Ok(MtpDraftState {
            layer_state: AttnLayerState { k_cache, v_cache },
            scratch: Scratch::new()?,
            position: 0,
            max_seq_len,
            token_buf: DeviceBuffer::alloc(1)?,
            position_buf: DeviceBuffer::alloc(1)?,
            embed_buf: DeviceBuffer::alloc(HIDDEN_SIZE)?,
            fused_concat: DeviceBuffer::alloc(2 * HIDDEN_SIZE)?,
            fused: DeviceBuffer::alloc(HIDDEN_SIZE)?,
            layer_out: DeviceBuffer::alloc(HIDDEN_SIZE)?,
            norm_out: DeviceBuffer::alloc(HIDDEN_SIZE)?,
            logits: DeviceBuffer::alloc(VOCAB_SIZE)?,
        })
    }

    /// Real position the head has observed/drafted up to -- the caller's
    /// own bookkeeping for deciding how far to `rollback_to` after a real
    /// accept/reject decision.
    pub fn position(&self) -> usize {
        self.position
    }

    /// One real fuse -> attention-layer -> final-norm step, exactly
    /// Python's `_fuse` + `_run_layer`. `h` is a raw device pointer (not
    /// `&DeviceBuffer`) specifically so a caller can pass `self.norm_out`'s
    /// own address across successive `draft()` iterations without
    /// fighting the borrow checker over a field of `self` -- safe because
    /// `DeviceBuffer`'s underlying `hipMalloc` allocation never moves for
    /// its own lifetime, the same raw-pointer-captured-before-a-mutable-
    /// call pattern `run_layers_over_chunk`'s own ping-pong buffers use.
    fn step(&mut self, weights: &ModelWeights, head: &MtpDraftHead, h: *const c_void, tok: i32) -> Result<(), HipError> {
        self.token_buf.copy_from_host(&[tok])?;
        self.position_buf.copy_from_host(&[self.position as i32])?;
        unsafe {
            raw::embedding_lookup(
                weights.embed_tokens.as_device_ptr(),
                self.token_buf.as_device_ptr() as *const i32,
                self.embed_buf.as_device_ptr_mut(),
                1,
                HIDDEN_SIZE as i32,
                std::ptr::null_mut(),
            );
            // fused_concat[0..H) = norm(embed(tok)) -- FIRST half.
            raw::rmsnorm(
                self.embed_buf.as_device_ptr(),
                head.pre_fc_norm_embedding.as_device_ptr(),
                self.fused_concat.as_device_ptr_mut(),
                1,
                HIDDEN_SIZE as i32,
                RMS_EPS,
                std::ptr::null_mut(),
            );
            // fused_concat[H..2H) = norm(h) -- SECOND half.
            raw::rmsnorm(
                h,
                head.pre_fc_norm_hidden.as_device_ptr(),
                self.fused_concat.as_device_ptr_at_mut(HIDDEN_SIZE),
                1,
                HIDDEN_SIZE as i32,
                RMS_EPS,
                std::ptr::null_mut(),
            );
            head.fc.apply(self.fused_concat.as_device_ptr(), self.fused.as_device_ptr_mut(), (2 * HIDDEN_SIZE) as i32, HIDDEN_SIZE as i32, std::ptr::null_mut());

            let position_ptr = self.position_buf.as_device_ptr() as *const i32;
            model::attn_layer_forward(&self.fused, &mut self.layer_out, &head.layer, &mut self.layer_state, position_ptr, self.max_seq_len, &mut self.scratch, std::ptr::null_mut());

            raw::rmsnorm(self.layer_out.as_device_ptr(), head.norm.as_device_ptr(), self.norm_out.as_device_ptr_mut(), 1, HIDDEN_SIZE as i32, RMS_EPS, std::ptr::null_mut());
        }
        crate::hip::check_last_error()?;
        crate::hip::device_synchronize()?;
        self.position += 1;
        Ok(())
    }

    /// Real, load-bearing context-building step (Python's own `prefill`
    /// docstring: "drafting from an EMPTY cache gives it one token of
    /// context and it degenerates" -- measured there as 0/6 tokens
    /// matched). Call once per REAL, already-known (prompt or previously-
    /// accepted) token: fuses the backbone's real hidden state at this
    /// position with the REAL next token, extending the head's own KV
    /// cache by one real entry. Discards the logits (nothing to predict
    /// for an already-known token) -- this is Python's teacher-forced
    /// `prefill` loop unrolled one real call at a time, matching this
    /// crate's own incremental (not bulk-buffered) architecture rather
    /// than requiring the caller to have every prompt position's hidden
    /// state resident at once.
    pub fn observe(&mut self, weights: &ModelWeights, head: &MtpDraftHead, hidden: &DeviceBuffer<u16>, next_real_token: i32) -> Result<(), HipError> {
        self.step(weights, head, hidden.as_device_ptr(), next_real_token)
    }

    /// Autoregressively drafts up to `k` real tokens, greedy, entirely in
    /// the head's own feature space (Python's own `draft`: "the base
    /// model is never called here, so no recurrent state advances").
    /// `hidden` is the backbone's real current hidden state (step 1's
    /// `h`); every subsequent step uses the head's own `norm_out` from
    /// the previous step. Does NOT roll back on its own -- the caller
    /// decides how many of the returned tokens were really accepted and
    /// calls `rollback_to` accordingly, exactly matching Python's own
    /// division of responsibility (`draft` never touches `restore_state`
    /// itself).
    pub fn draft(&mut self, weights: &ModelWeights, head: &MtpDraftHead, hidden: &DeviceBuffer<u16>, next_token: i32, k: usize) -> Result<Vec<i32>, HipError> {
        let mut h_ptr = hidden.as_device_ptr();
        let mut tok = next_token;
        let mut drafted = Vec::with_capacity(k);
        for _ in 0..k {
            self.step(weights, head, h_ptr, tok)?;
            unsafe {
                weights.lm_head_apply(self.norm_out.as_device_ptr(), self.logits.as_device_ptr_mut(), std::ptr::null_mut());
            }
            crate::hip::check_last_error()?;
            crate::hip::device_synchronize()?;
            let next_id = argmax_sample(&self.logits)?;
            drafted.push(next_id);
            tok = next_id;
            h_ptr = self.norm_out.as_device_ptr();
        }
        Ok(drafted)
    }

    /// Rolls back the head's own KV cache to `position`, discarding any
    /// rejected speculative tail -- a pure position-counter rewind, no
    /// realloc, no crop (see this module's own header doc for why the
    /// accepted prefix's cache entries never need recomputation).
    pub fn rollback_to(&mut self, position: usize) {
        self.position = position;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hip;
    use crate::model::{DecodeState, forward_one_token, argmax_sample as model_argmax_sample};
    use crate::model_loader::locate_model_snapshot;

    /// §137 cross-validation, step 1 of 2: dumps a REAL extracted
    /// pre-final-norm hidden state (`out.hidden_states[-1]` in HF terms --
    /// confirmed the real input `apps/runtime-ipwf/bucketed_speculative.py`
    /// actually feeds `head.draft()`, not the post-final-norm value) plus
    /// the real token id it corresponds to, to this session's scratchpad,
    /// for an independent REAL Python `Qwen35MTPDraftHead` run to consume
    /// (see `benchmarks/runtime/speculative/mtp_head_folding/` for the
    /// real, already-existing Python-side head-loading pattern this
    /// reference script should mirror). Not a decisive test itself --
    /// produces the real input the actual cross-validation
    /// (`real_mtp_draft_head_matches_python_reference`, written once the
    /// Python-side reference exists) checks against.
    #[test]
    #[ignore]
    fn dump_real_hidden_state_for_mtp_cross_validation() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        let handle = crate::blas::BlasHandle::create().expect("real hipblasCreate failed");

        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let max_seq_len = 64usize;
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        for &token_id in prompt_ids.iter() {
            forward_one_token(&handle, &weights, &mut state, token_id, &mut logits).unwrap();
        }
        // A few real decode steps so the head gets a real, non-trivial
        // position (matching a real mid-generation draft call, not the
        // very first token).
        let mut next_token = model_argmax_sample(&logits).unwrap();
        for _ in 0..3 {
            forward_one_token(&handle, &weights, &mut state, next_token, &mut logits).unwrap();
            next_token = model_argmax_sample(&logits).unwrap();
        }

        // NUM_LAYERS is even for every real size this crate ships (32 for
        // 4B/2B, 24 for 0.8B, 64 for 27B) -- confirmed by tracing
        // `run_decode_body`'s own `use_a_as_input` toggle: it starts
        // `true`, flips once per layer, so an EVEN layer count always
        // ends back at `true`, meaning `hidden_a` holds the real,
        // pre-final-norm final hidden state after any `forward_one_token`
        // call. A real, traced fact, not a guess -- if this crate ever
        // ships an odd-NUM_LAYERS size this assert catches it loudly
        // instead of silently reading the wrong (stale) buffer.
        assert_eq!(model::NUM_LAYERS % 2, 0, "NUM_LAYERS must be even for this test's hidden_a/hidden_b assumption to hold -- re-derive which buffer is active for an odd layer count before trusting this dump");
        let final_hidden = &state.hidden_a;

        let mut hidden_host = vec![0u16; HIDDEN_SIZE];
        final_hidden.copy_to_host(&mut hidden_host).unwrap();

        let out_dir = std::path::Path::new("/tmp/claude-1000/-home-mihai-Projects-gnn-experiment/62408538-8f70-4fd0-a0ab-8c9b65ca71dd/scratchpad");
        std::fs::create_dir_all(out_dir).unwrap();
        let hidden_bytes: Vec<u8> = hidden_host.iter().flat_map(|&v| v.to_le_bytes()).collect();
        std::fs::write(out_dir.join("mtp_xval_hidden.bin"), &hidden_bytes).unwrap();
        let meta = serde_json::json!({
            "hidden_size": HIDDEN_SIZE,
            "position": state.position,
            "next_token": next_token,
            "k": 6,
            "prompt_ids": prompt_ids,
        });
        std::fs::write(out_dir.join("mtp_xval_meta.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();
        eprintln!("dumped real hidden state (position={}, next_token={next_token}) to {}", state.position, out_dir.display());
    }

    /// §137 DECISIVE: does the Rust `MtpDraftHead`/`MtpDraftState` port
    /// agree with the real, original, shipped Python
    /// `Qwen35MTPDraftHead.draft()` -- given the IDENTICAL real hidden-
    /// state bytes (extracted directly from this crate's own real
    /// `forward_one_token` output, not resynthesized)? This is the actual
    /// correctness bar for the port: not "does it run," but "does it
    /// produce the same real drafted tokens a trusted, already-working
    /// implementation does" -- the same standard this crate holds every
    /// other ported component to.
    ///
    /// Reads `mtp_xval_hidden.bin`/`mtp_xval_meta.json` (written by
    /// `dump_real_hidden_state_for_mtp_cross_validation` above) and
    /// `mtp_xval_python_reference.json` (written by the real Python
    /// reference script, `benchmarks/runtime/speculative/mtp_head_folding/
    /// mtp_draft_head_rust_cross_validate.py`, run separately against the
    /// SAME dumped hidden state) -- run this test only after both of
    /// those have been produced fresh.
    #[test]
    #[ignore]
    fn real_mtp_draft_head_matches_python_reference() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let scratch = std::path::Path::new("/tmp/claude-1000/-home-mihai-Projects-gnn-experiment/62408538-8f70-4fd0-a0ab-8c9b65ca71dd/scratchpad");
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(scratch.join("mtp_xval_meta.json")).expect("mtp_xval_meta.json missing -- run dump_real_hidden_state_for_mtp_cross_validation first")).unwrap();
        let python_ref: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(scratch.join("mtp_xval_python_reference.json")).expect("mtp_xval_python_reference.json missing -- run mtp_draft_head_rust_cross_validate.py first")).unwrap();

        let hidden_size = meta["hidden_size"].as_u64().unwrap() as usize;
        assert_eq!(hidden_size, HIDDEN_SIZE, "dumped hidden_size doesn't match this build's HIDDEN_SIZE -- was the dump taken with a different model-size feature?");
        let position = meta["position"].as_u64().unwrap() as usize;
        let next_token = meta["next_token"].as_i64().unwrap() as i32;
        let k = meta["k"].as_u64().unwrap() as usize;

        let hidden_bytes = std::fs::read(scratch.join("mtp_xval_hidden.bin")).unwrap();
        assert_eq!(hidden_bytes.len(), hidden_size * 2);
        let hidden_host: Vec<u16> = hidden_bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let mut hidden_buf: DeviceBuffer<u16> = DeviceBuffer::alloc(hidden_size).unwrap();
        hidden_buf.copy_from_host(&hidden_host).unwrap();

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        let head = MtpDraftHead::load(&snapshot).expect("real MTP head weight loading failed");
        let max_seq_len = 64usize;
        let mut mtp_state = MtpDraftState::new(max_seq_len).expect("MtpDraftState::new failed");
        mtp_state.rollback_to(position); // matches Python's own `start_pos=position`, fresh cache

        let rust_drafted = mtp_state.draft(&weights, &head, &hidden_buf, next_token, k).expect("real draft() call failed");

        let python_drafted: Vec<i32> = python_ref["drafted_ids"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect();
        eprintln!("Rust drafted:   {rust_drafted:?}");
        eprintln!("Python drafted: {python_drafted:?}");
        assert_eq!(rust_drafted, python_drafted, "Rust MTP draft head port diverged from the real, original Python Qwen35MTPDraftHead.draft() on the IDENTICAL real hidden state -- the port has a real correctness bug");
        eprintln!("VERDICT: Rust MTP draft head port matches the real Python reference exactly, token-for-token.");
    }
}
