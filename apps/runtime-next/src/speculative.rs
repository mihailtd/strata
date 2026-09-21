//! §137 continued: the real accept/reject loop tying together the MTP
//! draft head (`mtp_draft.rs`) and the bucketed verify-chunk graph
//! (`model.rs`'s `GraphedVerifyState`, §136) into one real speculative
//! decoding round.
//!
//! The real algorithm's shape is ported from -- not re-derived from --
//! the actual production Python engine's own proven loop
//! (`apps/runtime-ipwf/bucketed_speculative.py`), read directly rather
//! than assumed: the verify chunk is `[next_token] + drafted` (K+1 wide,
//! not K), because `next_token` (the PREVIOUS round's own trusted greedy
//! prediction) has NOT yet been run through the backbone's real forward
//! pass -- this round's verify call is what commits it for real, fused
//! into the same batched call as the K speculative tokens.
//!
//! **Corrected after a real, decisive test caught it** (see
//! `docs/DECISIONS.md` §137): an earlier version of this module tried a
//! deliberate interface simplification -- committing the "bonus"/
//! correction token to the real backbone WITHIN the same round it was
//! produced (via an extra `forward_one_token` call), rather than
//! deferring that commit to the next round the way Python does. That
//! design was WRONG, not just different: since every round's own verify
//! chunk ALWAYS re-commits `next_token` as `chunk[0]` (needed for the
//! genuinely-first round, where `next_token` really is uncommitted), a
//! round that ALSO separately committed its own bonus token left that
//! same logical token committed to TWO real backbone positions -- once
//! as this round's bonus, and again as the NEXT round's `chunk[0]`. The
//! duplicate is invisible in the OUTPUT token list (that bookkeeping was
//! separately correct) but corrupts the backbone's own real internal
//! state (attention context, GDN's irreversible recurrent update) from
//! that point forward. A real end-to-end test (speculative generation
//! vs. plain greedy decode, required to be bit-exact) caught this via a
//! divergence starting several rounds in, once the corruption's effect
//! on an argmax finally flipped a token -- not on round 1, since the
//! model is fairly robust to one extra duplicated context token at
//! first. This module now matches Python's real, proven convention
//! exactly: the bonus token's real backbone commit is DEFERRED to the
//! NEXT round's own verify chunk (as that round's `chunk[0]`), and
//! `state.position` advances by `accepted_count + 1` per round (matching
//! Python's own `pos += n_acc + 1`, not `+2`) -- while the OUTPUT
//! contract stays the interface-simplification part that WAS safe to
//! keep: each round still returns `committed_tokens` (`accepted_count`
//! drafts plus the bonus) as real, decided, new output the caller can
//! use immediately, even though the bonus's own backbone commit hasn't
//! happened yet.
//!
//! Real, deliberate, EXPLAINED divergence from Python's own cache-
//! rebuild mechanism: Python always rebuilds its `DynamicCache`-based
//! draft-head cache via a full `prefill()` on ANY partial reject, because
//! `DynamicCache` is append-only and cannot be truncated. This crate's
//! own `MtpDraftState` uses a FIXED-SIZE, position-indexed cache instead
//! (matching the backbone's own attention layers) -- so a partial reject
//! only needs `MtpDraftState::rollback_to` (a pure counter rewind, no
//! recomputation): `draft()`'s own K sequential steps already wrote real,
//! correct cache entries observing `next_token` and each of
//! `drafted[0..k-1)` (step `i` observes whatever token was FED INTO it,
//! which is `next_token` for `i=0` and `drafted[i-1]` for `i>0`) --
//! keeping the prefix through step `accepted_count` and discarding the
//! rest is exactly right whether the round fully or partially accepted,
//! no branching needed. This optimization is real and safe for the MTP
//! head specifically (pure attention, no recurrent state) -- it does NOT
//! apply to the BACKBONE's own GDN layers, whose chunked recurrent state
//! update is not reversible by a counter rewind once speculative
//! (possibly-wrong) tokens have been mixed into it; a partial/zero accept
//! there needs a real `TensorStateSnapshot` restore, matching Python's
//! own real reasoning for why IT needs a rebuild (just for a structurally
//! different reason: irreversible recurrent state here, an append-only
//! cache class there).
//!
//! Note `drafted[k-1]` (the LAST drafted token) is NEVER observed by
//! `draft()`'s own steps at all (there is no step `k`) -- and neither it
//! nor the bonus token need an explicit separate `MtpDraftState::observe`
//! call: with the bonus commit deferred, the bonus token's own MTP-cache
//! observation happens "for free" as step 0 of the NEXT round's own
//! `draft()` call (whose `tok` argument is `next_token` = this round's
//! bonus) -- provided the `hidden` this module passes to that call is
//! genuinely correct. On the partial-accept path, ordinary
//! `forward_one_token` replay calls naturally leave `state.final_hidden()`
//! correct. On the full-accept path, `verify_chunk`'s own batched
//! computation writes to a DIFFERENT buffer (`PrefillScratch::hidden_a`/
//! `hidden_b`) than the one `state.final_hidden()` exposes -- so this
//! module explicitly copies `verify_chunk`'s own preserved `raw_hidden`
//! (row `k`, the real pre-final-norm hidden state predicting what follows
//! the last drafted token) into `state`'s single-token buffer before
//! returning, so the NEXT round's `draft()` call sees real data instead
//! of stale leftovers.

use crate::blas::BlasHandle;
use crate::hip::{DeviceBuffer, HipError};
use crate::model::{self, DecodeState, GraphedVerifyState, HIDDEN_SIZE, ModelWeights, VOCAB_SIZE};
use crate::mtp_draft::{MtpDraftHead, MtpDraftState};
use crate::state_handoff::TensorStateSnapshot;

/// Real, reusable scratch for one `speculative_decode_round` call --
/// pre-allocated once (matching this crate's own established "no
/// `hipMalloc` in a hot loop" discipline), not per-round. `k_plus_one`
/// must equal the `k` the caller's `GraphedVerifyState` was built with,
/// plus one (the verify chunk width).
pub struct SpeculativeRoundScratch {
    normed_out: DeviceBuffer<u16>,
    verify_logits: DeviceBuffer<u16>,
    raw_hidden: DeviceBuffer<u16>,
    single_logits: DeviceBuffer<u16>,
}

impl SpeculativeRoundScratch {
    pub fn new(k_plus_one: usize) -> Result<Self, HipError> {
        Ok(SpeculativeRoundScratch {
            normed_out: DeviceBuffer::alloc(k_plus_one * HIDDEN_SIZE)?,
            verify_logits: DeviceBuffer::alloc(k_plus_one * VOCAB_SIZE)?,
            raw_hidden: DeviceBuffer::alloc(k_plus_one * HIDDEN_SIZE)?,
            single_logits: DeviceBuffer::alloc(VOCAB_SIZE)?,
        })
    }
}

pub struct SpeculativeRoundResult {
    /// Real, NEW output tokens minted this round, in real sequence order:
    /// `accepted_count` real verified drafts, then exactly one real
    /// bonus/correction token. Always has length `accepted_count + 1`.
    /// Deliberately does NOT include this round's own `next_token` input
    /// -- confirmed against the real production loop
    /// (`bucketed_speculative.py:468`, `yield draft[:n_acc] + [bonus]`):
    /// `next_token` was already real output, either the initial seed
    /// token (emitted once by the caller before the first round) or the
    /// PREVIOUS round's own `bonus`/`next_token` -- re-including it here
    /// would silently duplicate it in the caller's accumulated output.
    pub committed_tokens: Vec<i32>,
    /// How many of the `k` drafted tokens were real, verified matches.
    pub accepted_count: usize,
    /// `committed_tokens`' own last element -- the real, trusted next
    /// token to seed the NEXT round's `speculative_decode_round` call.
    pub next_token: i32,
}

/// Runs one real speculative decoding round. Real, maintained invariant
/// on entry and exit: `mtp_state.position() == state.position - 1` (the
/// draft head's own cache trails the backbone's real position by exactly
/// one -- see this module's header doc, and `docs/DECISIONS.md` §137 for
/// the real derivation, read directly out of `apps/runtime-ipwf/
/// bucketed_speculative.py`'s own `start_pos = pos - 1` convention, not
/// guessed).
pub fn speculative_decode_round(
    handle: &BlasHandle,
    weights: &ModelWeights,
    state: &mut DecodeState,
    graphed_verify: &mut GraphedVerifyState,
    mtp_head: &MtpDraftHead,
    mtp_state: &mut MtpDraftState,
    scratch: &mut SpeculativeRoundScratch,
    next_token: i32,
    k: usize,
) -> Result<SpeculativeRoundResult, HipError> {
    debug_assert_eq!(mtp_state.position(), state.position - 1, "speculative_decode_round: mtp_state/backbone position invariant violated on entry");

    let pre_round_position = state.position;
    let pre_round_mtp_position = mtp_state.position();
    let pre_round_snapshot = TensorStateSnapshot::capture(state)?;

    // 1. Draft K tokens using the MTP head alone -- never touches the
    // backbone (model::attn_layer_forward, called only on the head's own
    // tiny one-layer weights).
    let current_hidden_ptr: *const DeviceBuffer<u16> = state.final_hidden();
    let drafted = mtp_state.draft(weights, mtp_head, unsafe { &*current_hidden_ptr }, next_token, k)?;

    // 2. Verify: real (k+1)-token chunk forward through the FULL
    // backbone -- `next_token` first (not yet committed for real), then
    // the k drafts.
    let mut chunk: Vec<i32> = Vec::with_capacity(k + 1);
    chunk.push(next_token);
    chunk.extend_from_slice(&drafted);
    graphed_verify.verify_chunk(weights, state, &chunk, &mut scratch.normed_out, &mut scratch.verify_logits, &mut scratch.raw_hidden)?;

    // 3. Real per-position argmax + accept-loop: row i's real argmax
    // (predicting what follows chunk[i]) compared against chunk[i+1]
    // (drafted[i] for i in 0..k).
    let mut real_argmax = Vec::with_capacity(k + 1);
    for i in 0..=k {
        real_argmax.push(model::argmax_sample_row(&scratch.verify_logits, i)?);
    }
    let mut accepted_count = 0usize;
    while accepted_count < k && drafted[accepted_count] == real_argmax[accepted_count] {
        accepted_count += 1;
    }
    let bonus_token = real_argmax[accepted_count];

    // 4. MTP head: keep draft()'s own real, correct observations of
    // next_token and each accepted drafted token (steps 0..accepted_count
    // inclusive), discard the rest -- one unconditional rewind, correct
    // whether this round fully or partially accepted (see this module's
    // header doc for why).
    mtp_state.rollback_to(pre_round_mtp_position + accepted_count + 1);

    let mut committed_tokens: Vec<i32> = Vec::with_capacity(accepted_count + 1);
    committed_tokens.extend_from_slice(&drafted[..accepted_count]);

    // §138 CORRECTED (was branched on `accepted_count == k`, trusting
    // verify_chunk's own real computation directly on a full accept and
    // skipping the rebuild below as a real efficiency optimization): a
    // real, decisive diagnostic
    // (`diagnose_full_accept_branch_gdn_state_matches_sequential_decode`)
    // proved that was wrong, not just an optimization with a cost --
    // verify_chunk's CHUNKED (batched, K+1-token-at-once) GDN recurrent-
    // state computation is NOT numerically equivalent to sequential,
    // one-token-at-a-time decode, even for the exact same real tokens.
    // On a real 298-token aider-bench prompt this silently corrupted
    // state that flipped a real argmax several rounds later (this
    // module's own short-prompt correctness test never ran a full-accept
    // round long enough to expose it). This is the SAME CLASS of risk
    // `docs/DECISIONS.md` §136 already found and left open for
    // `gemm_pv_bf16` (kernel-choice-dependent floating-point non-
    // associativity) -- a different kernel, and here PROVEN to actually
    // flip a real token on a real workload, not just a theoretical risk.
    // Until a real fix for the underlying chunked-GDN-continuation
    // numerics exists (a separate, substantial undertaking), correctness
    // requires ALWAYS restoring and sequentially replaying the real
    // confirmed prefix -- no exception for a full accept, even though
    // that gives up whatever efficiency benefit the "no rebuild" case
    // offered.
    pre_round_snapshot.restore(state)?;
    model::forward_one_token(handle, weights, state, next_token, &mut scratch.single_logits)?;
    for &tok in &committed_tokens {
        model::forward_one_token(handle, weights, state, tok, &mut scratch.single_logits)?;
    }
    committed_tokens.push(bonus_token);

    debug_assert_eq!(state.position, pre_round_position + accepted_count + 1, "speculative_decode_round: real position advance didn't match accepted_count + 1");
    debug_assert_eq!(mtp_state.position(), state.position - 1, "speculative_decode_round: mtp_state/backbone position invariant violated on exit");

    let committed_next_token = *committed_tokens.last().unwrap();
    Ok(SpeculativeRoundResult { committed_tokens, accepted_count, next_token: committed_next_token })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hip;
    use crate::model::{ModelWeights, argmax_sample, forward_one_token};
    use crate::model_loader::locate_model_snapshot;

    /// §137 DECISIVE: the defining correctness invariant of speculative
    /// decoding -- real speculative generation must produce EXACTLY the
    /// same token sequence plain greedy decode would have, since
    /// verification only ever accepts a drafted token that matches the
    /// real model's own greedy choice. This is the strongest possible
    /// test of the whole round function's real bookkeeping (position
    /// tracking, the MTP-head-cache rollback optimization, the GDN-
    /// recurrent-state restore-on-reject path) all at once -- any real
    /// bug in any of them would produce a DIFFERENT token somewhere, not
    /// a subtle numerical drift.
    ///
    /// Real prompt, real weights, `k=6` (matching every other real K used
    /// throughout §136/§137). Runs several real speculative rounds
    /// (mixing real full-accept and real partial-accept outcomes,
    /// whichever the actual model/prompt produces -- not staged), then
    /// separately runs plain greedy `forward_one_token` for the SAME
    /// prompt and the SAME total token count, and compares.
    #[test]
    #[ignore]
    fn real_speculative_decode_matches_plain_greedy_decode() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        assert!(!weights.is_quantized(), "this test assumes a bf16 dense checkpoint (4B/9B)");
        let mtp_head = MtpDraftHead::load(&snapshot).expect("real MTP head weight loading failed");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        let prompt_ids: [i32; 5] = [760, 6511, 314, 9338, 369];
        let k = 6usize;
        let max_seq_len = 256usize;
        let min_real_rounds = 3usize;

        // --- SPECULATIVE path ---
        let mut state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        let mut mtp_state = MtpDraftState::new(max_seq_len).expect("MtpDraftState::new failed");

        forward_one_token(&handle, &weights, &mut state, prompt_ids[0], &mut logits).unwrap();
        for i in 1..prompt_ids.len() {
            let hidden_ptr: *const DeviceBuffer<u16> = state.final_hidden();
            mtp_state.observe(&weights, &mtp_head, unsafe { &*hidden_ptr }, prompt_ids[i]).unwrap();
            forward_one_token(&handle, &weights, &mut state, prompt_ids[i], &mut logits).unwrap();
        }
        assert_eq!(mtp_state.position(), state.position - 1, "setup: mtp_state/backbone position invariant must hold before the first real round");

        let mut graphed_verify = GraphedVerifyState::new(k + 1).expect("GraphedVerifyState::new failed");
        let mut scratch = SpeculativeRoundScratch::new(k + 1).expect("SpeculativeRoundScratch::new failed");

        let mut next_token = argmax_sample(&logits).unwrap();
        // Matches the real production loop's own seeding
        // (`bucketed_speculative.py:354/363`, `toks = [nxt]; yield
        // [toks[0]]`): the initial greedy prediction from the prompt's
        // own last-position logits is real output in its own right,
        // emitted once here -- every ROUND's own `committed_tokens`
        // deliberately excludes its `next_token` input (see
        // `SpeculativeRoundResult`'s doc comment).
        let mut speculative_tokens: Vec<i32> = vec![next_token];
        let mut rounds = 0usize;
        let mut total_accepted = 0usize;
        let mut total_drafted = 0usize;
        while speculative_tokens.len() < 40 || rounds < min_real_rounds {
            let result = speculative_decode_round(&handle, &weights, &mut state, &mut graphed_verify, &mtp_head, &mut mtp_state, &mut scratch, next_token, k).expect("real speculative_decode_round failed");
            eprintln!("round {rounds}: accepted {}/{k}, committed {:?}", result.accepted_count, result.committed_tokens);
            total_accepted += result.accepted_count;
            total_drafted += k;
            speculative_tokens.extend_from_slice(&result.committed_tokens);
            next_token = result.next_token;
            rounds += 1;
            if rounds > 20 {
                break;
            }
        }
        eprintln!("real speculative generation: {} rounds, {} tokens, {total_accepted}/{total_drafted} drafts accepted ({:.1}%)", rounds, speculative_tokens.len(), 100.0 * total_accepted as f64 / total_drafted as f64);

        // --- PLAIN GREEDY path, same real prompt, same real token count ---
        let mut greedy_state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut greedy_logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        for &token_id in prompt_ids.iter() {
            forward_one_token(&handle, &weights, &mut greedy_state, token_id, &mut greedy_logits).unwrap();
        }
        let mut greedy_tokens: Vec<i32> = Vec::with_capacity(speculative_tokens.len());
        let mut next_greedy = argmax_sample(&greedy_logits).unwrap();
        for _ in 0..speculative_tokens.len() {
            greedy_tokens.push(next_greedy);
            forward_one_token(&handle, &weights, &mut greedy_state, next_greedy, &mut greedy_logits).unwrap();
            next_greedy = argmax_sample(&greedy_logits).unwrap();
        }

        eprintln!("speculative: {speculative_tokens:?}");
        eprintln!("greedy:      {greedy_tokens:?}");
        assert_eq!(speculative_tokens, greedy_tokens, "real speculative decoding diverged from real plain greedy decode -- the defining correctness invariant is broken, there's a real bug in the round function's bookkeeping");
        eprintln!("VERDICT: real speculative decoding matches real plain greedy decode bit-exact, token-for-token, across {rounds} real rounds.");
    }

    /// §138 continued -- the ONE measurement this repo's own documented
    /// history (`docs/DECISIONS.md` §22-§27, §51) says actually decides
    /// whether speculative decoding is worth shipping, not a projection.
    /// Every prior real attempt at this exact question, in the legacy
    /// Python engine, looked like a real win right up until someone ran
    /// the real, realistic-length, real-serving-path thing -- §25's
    /// measured "+8.5%" was withdrawn by §26 once a 512-token real
    /// generation (instead of a 64-token cap that "manufactured" the
    /// result by only sampling the model's easy, predictable preamble)
    /// showed it was actually 22% SLOWER than the graph-autoregressive
    /// baseline already in production. `SPECULATIVE_DECODE` has stayed
    /// `0` there ever since. This is that same measurement, honestly
    /// built for `runtime-next`, not a repeat of the earlier shortcuts:
    ///
    /// - **Real target workload**: the SAME 10 real aider-bench task
    ///   prompts `docs/DECISIONS.md` §136's own real τ=3.52 acceptance
    ///   number was measured against, rendered with the identical
    ///   system/user template (see
    ///   `benchmarks/runtime/speculative/runtime_next_serving_gate/
    ///   dump_aider_bench_prompts.py`) -- not an arbitrary, unrelated
    ///   prompt the way `real_speculative_decode_matches_plain_greedy_
    ///   decode`'s own correctness-only test used.
    /// - **Real, non-truncated generation length**: `GEN_TOKENS = 256`,
    ///   deliberately not a short cap -- §26's own explicit lesson was
    ///   that a 64-token cap sits entirely inside this model's easy
    ///   thinking-mode preamble, where acceptance is inflated and a
    ///   losing feature looks like a win. 256 tokens reaches past a
    ///   comparable preamble into real generated code, matching the
    ///   region `benchmark_mtp_head_aider_bench_acceptance.py`'s own
    ///   `offsets=[...180]` was built to reach.
    /// - **Real baseline**: this engine's own existing BEST
    ///   autoregressive path, `GraphedDecodeState` -- not eager decode,
    ///   which §24 found the server never runs and which flatters
    ///   speculation by comparing against a strawman.
    /// - **Correctness checked in the same run, not assumed**: each
    ///   task's speculative output must be bit-exact against that SAME
    ///   task's own graphed-decode output. A fast-but-wrong number is
    ///   worthless, and this crate does not report a throughput result
    ///   it has not also verified correct in the same run.
    ///
    /// One real, disclosed approximation: both arms prefill the real
    /// prompt sequentially (`forward_one_token`/`observe`, not the
    /// batched `forward_prefill` path a real server would use) -- an
    /// identical, off-the-clock setup cost paid equally by both arms,
    /// so it does not bias the DECODE-phase comparison this benchmark
    /// actually times. A real bucket's HIP-graph capture cost (§136)
    /// can still land inside a task's timed region if generation
    /// crosses a NEW `KV_LEN_BUCKETS` boundary mid-task (only the
    /// first bucket is warmed up off-clock, matching ARM A's own
    /// fixed warmup) -- reported per-task AND aggregated across all 10
    /// tasks so a rare capture-cost outlier is visible, not hidden.
    #[test]
    #[ignore]
    fn bench_real_speculative_vs_graphed_decode_on_real_aider_bench_tasks() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        assert!(!weights.is_quantized(), "this benchmark assumes a bf16 dense checkpoint (4B/9B)");
        let mtp_head = MtpDraftHead::load(&snapshot).expect("real MTP head weight loading failed");
        let tokenizer = crate::tokenizer::ChatTokenizer::load(&snapshot).expect("real tokenizer loading failed");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        let prompts_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../benchmarks/runtime/speculative/runtime_next_serving_gate/aider_bench_prompts.json");
        let prompts_json = std::fs::read_to_string(&prompts_path).unwrap_or_else(|e| {
            panic!("failed to read {prompts_path:?} -- run dump_aider_bench_prompts.py first: {e}")
        });
        let parsed: serde_json::Value = serde_json::from_str(&prompts_json).expect("aider_bench_prompts.json is not valid JSON");
        let tasks = parsed["tasks"].as_array().expect("aider_bench_prompts.json missing 'tasks' array").clone();
        assert!(!tasks.is_empty(), "aider_bench_prompts.json has zero tasks -- did dump_aider_bench_prompts.py run correctly?");

        const K: usize = 6;
        const GEN_TOKENS: usize = 256;
        const WARMUP_TOKENS: usize = 3;

        let mut per_task: Vec<serde_json::Value> = Vec::new();
        let mut total_baseline_tokens = 0usize;
        let mut total_baseline_secs = 0.0f64;
        let mut total_spec_tokens = 0usize;
        let mut total_spec_secs = 0.0f64;
        let mut total_accepted_all = 0usize;
        let mut total_drafted_all = 0usize;

        for task in &tasks {
            let name = task["name"].as_str().expect("task missing 'name'").to_string();
            let system_prompt = task["system_prompt"].as_str().expect("task missing 'system_prompt'");
            let user_prompt = task["user_prompt"].as_str().expect("task missing 'user_prompt'");
            let templated = tokenizer.apply_chat_template(&[("system", system_prompt), ("user", user_prompt)]);
            let prompt_ids = tokenizer.encode(&templated).expect("real tokenizer encode failed");
            let max_seq_len = prompt_ids.len() + GEN_TOKENS + 4 * K + WARMUP_TOKENS + 32;

            eprintln!("=== task {name}: real prompt {} tokens ===", prompt_ids.len());

            // --- ARM A: this engine's own existing best autoregressive path ---
            let mut graphed = model::GraphedDecodeState::new().expect("GraphedDecodeState::new failed");
            let mut a_state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
            let mut a_logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
            for &tok in &prompt_ids {
                graphed.forward_one_token(&weights, &mut a_state, tok, &mut a_logits).unwrap();
            }
            // Real tokens generated during warmup are KEPT, not discarded --
            // discarding them would silently advance `a_state`'s own real
            // KV-cache/position `WARMUP_TOKENS` further than `b_state`'s
            // before either arm starts recording, making the two token
            // lists represent DIFFERENT real positions in the sequence
            // (an early real run of this benchmark caught exactly this:
            // an apparent "correctness divergence" that was actually just
            // this fixed 3-token misalignment, not a real decoding bug).
            // Only the WALL-CLOCK for the warmup tokens is excluded.
            let mut baseline_tokens: Vec<i32> = Vec::with_capacity(WARMUP_TOKENS + GEN_TOKENS);
            let mut next_id = argmax_sample(&a_logits).unwrap();
            for _ in 0..WARMUP_TOKENS {
                baseline_tokens.push(next_id);
                graphed.forward_one_token(&weights, &mut a_state, next_id, &mut a_logits).unwrap();
                next_id = argmax_sample(&a_logits).unwrap();
            }
            let t0 = std::time::Instant::now();
            for _ in 0..GEN_TOKENS {
                baseline_tokens.push(next_id);
                graphed.forward_one_token(&weights, &mut a_state, next_id, &mut a_logits).unwrap();
                next_id = argmax_sample(&a_logits).unwrap();
            }
            let baseline_elapsed = t0.elapsed();

            // --- ARM B: the new speculative round loop ---
            let mut b_state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
            let mut b_logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
            let mut mtp_state = MtpDraftState::new(max_seq_len).expect("MtpDraftState::new failed");

            forward_one_token(&handle, &weights, &mut b_state, prompt_ids[0], &mut b_logits).unwrap();
            for i in 1..prompt_ids.len() {
                let hidden_ptr: *const DeviceBuffer<u16> = b_state.final_hidden();
                mtp_state.observe(&weights, &mtp_head, unsafe { &*hidden_ptr }, prompt_ids[i]).unwrap();
                forward_one_token(&handle, &weights, &mut b_state, prompt_ids[i], &mut b_logits).unwrap();
            }
            assert_eq!(mtp_state.position(), b_state.position - 1, "task {name}: setup invariant violated");

            let mut graphed_verify = GraphedVerifyState::new(K + 1).expect("GraphedVerifyState::new failed");
            let mut scratch = SpeculativeRoundScratch::new(K + 1).expect("SpeculativeRoundScratch::new failed");

            let mut spec_next_token = argmax_sample(&b_logits).unwrap();
            let mut speculative_tokens: Vec<i32> = vec![spec_next_token];
            let mut total_accepted = 0usize;
            let mut total_drafted = 0usize;
            let mut rounds = 0usize;

            // One real round off the clock -- absorbs the first bucket's
            // real HIP-graph capture cost, matching ARM A's own fixed
            // token warmup for its own first-capture cost.
            {
                let pre_round_pos = b_state.position;
                let result = speculative_decode_round(&handle, &weights, &mut b_state, &mut graphed_verify, &mtp_head, &mut mtp_state, &mut scratch, spec_next_token, K).expect("warmup round failed");
                if std::env::var("SPEC_DEBUG_ROUNDS").is_ok() {
                    let start_idx = speculative_tokens.len();
                    eprintln!(
                        "  round {rounds} (warmup): pre_pos={pre_round_pos} accepted={}/{K} committed={:?} (output idx {}..{})",
                        result.accepted_count, result.committed_tokens, start_idx, start_idx + result.committed_tokens.len()
                    );
                }
                total_accepted += result.accepted_count;
                total_drafted += K;
                speculative_tokens.extend_from_slice(&result.committed_tokens);
                spec_next_token = result.next_token;
                rounds += 1;
            }

            let len_after_warmup = speculative_tokens.len();
            let t1 = std::time::Instant::now();
            while speculative_tokens.len() < GEN_TOKENS {
                let pre_round_pos = b_state.position;
                let result = speculative_decode_round(&handle, &weights, &mut b_state, &mut graphed_verify, &mtp_head, &mut mtp_state, &mut scratch, spec_next_token, K).expect("real speculative_decode_round failed");
                if std::env::var("SPEC_DEBUG_ROUNDS").is_ok() {
                    let start_idx = speculative_tokens.len();
                    eprintln!(
                        "  round {rounds}: pre_pos={pre_round_pos} accepted={}/{K} committed={:?} (output idx {}..{})",
                        result.accepted_count, result.committed_tokens, start_idx, start_idx + result.committed_tokens.len()
                    );
                }
                total_accepted += result.accepted_count;
                total_drafted += K;
                speculative_tokens.extend_from_slice(&result.committed_tokens);
                spec_next_token = result.next_token;
                rounds += 1;
            }
            let spec_elapsed = t1.elapsed();
            let spec_new_tokens = speculative_tokens.len() - len_after_warmup;

            // --- Correctness: checked here, not assumed from the earlier decisive test ---
            if std::env::var("SPEC_DEBUG_ROUNDS").is_ok() {
                for i in 0..GEN_TOKENS.min(speculative_tokens.len()).min(baseline_tokens.len()) {
                    if speculative_tokens[i] != baseline_tokens[i] {
                        eprintln!("  FIRST MISMATCH at output idx {i}: speculative={} baseline={}", speculative_tokens[i], baseline_tokens[i]);
                        break;
                    }
                }
            }
            assert_eq!(
                &speculative_tokens[..GEN_TOKENS],
                &baseline_tokens[..GEN_TOKENS],
                "task {name}: real speculative output diverged from this engine's own real graphed-decode output for the SAME real prompt -- a throughput number is worthless without this"
            );

            let baseline_tok_s = GEN_TOKENS as f64 / baseline_elapsed.as_secs_f64();
            let spec_tok_s = spec_new_tokens as f64 / spec_elapsed.as_secs_f64();
            let tau = total_accepted as f64 / rounds as f64;
            let accept_rate_pct = 100.0 * total_accepted as f64 / total_drafted as f64;
            let ratio = spec_tok_s / baseline_tok_s;

            eprintln!(
                "  baseline (graphed) {baseline_tok_s:.2} tok/s | speculative {spec_tok_s:.2} tok/s | ratio {ratio:.3}x | tau {tau:.3} | accept {accept_rate_pct:.1}% | {rounds} rounds"
            );

            total_baseline_tokens += GEN_TOKENS;
            total_baseline_secs += baseline_elapsed.as_secs_f64();
            total_spec_tokens += spec_new_tokens;
            total_spec_secs += spec_elapsed.as_secs_f64();
            total_accepted_all += total_accepted;
            total_drafted_all += total_drafted;

            per_task.push(serde_json::json!({
                "name": name,
                "prompt_tokens": prompt_ids.len(),
                "baseline_tok_s": baseline_tok_s,
                "speculative_tok_s": spec_tok_s,
                "ratio_vs_baseline": ratio,
                "tau_mean_accepted": tau,
                "accept_rate_pct": accept_rate_pct,
                "rounds": rounds,
            }));
        }

        let overall_baseline_tok_s = total_baseline_tokens as f64 / total_baseline_secs;
        let overall_spec_tok_s = total_spec_tokens as f64 / total_spec_secs;
        let overall_ratio = overall_spec_tok_s / overall_baseline_tok_s;
        let overall_tau = total_accepted_all as f64 / (total_drafted_all as f64 / K as f64);
        let overall_accept_pct = 100.0 * total_accepted_all as f64 / total_drafted_all as f64;

        eprintln!("\n=== OVERALL, {} real aider-bench tasks, K={K}, {GEN_TOKENS} real tokens/task ===", tasks.len());
        eprintln!(
            "baseline (graphed) {overall_baseline_tok_s:.2} tok/s | speculative {overall_spec_tok_s:.2} tok/s | ratio {overall_ratio:.3}x | tau {overall_tau:.3} | accept {overall_accept_pct:.1}%"
        );
        eprintln!(
            "VERDICT: speculative decoding is {} on this real, realistic-length, real-target-workload measurement.",
            if overall_ratio > 1.0 { "a REAL WIN" } else { "NOT a win (at or below the graphed baseline)" }
        );

        let scorecard = serde_json::json!({
            "engine": "runtime-next",
            "model": "Qwen3.5-4B",
            "k": K,
            "gen_tokens": GEN_TOKENS,
            "n_tasks": tasks.len(),
            "overall_baseline_tok_s": overall_baseline_tok_s,
            "overall_speculative_tok_s": overall_spec_tok_s,
            "overall_ratio_vs_baseline": overall_ratio,
            "overall_tau_mean_accepted": overall_tau,
            "overall_accept_rate_pct": overall_accept_pct,
            "per_task": per_task,
        });
        let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/benchmarks/speculative_vs_graphed_decode_aider_bench_qwen35_4b.json");
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&out_path, serde_json::to_string_pretty(&scorecard).unwrap()).expect("failed to write scorecard JSON");
        eprintln!("Wrote real scorecard to {out_path:?}");
    }

    /// DIAGNOSTIC, §138 -- root-causing the exact divergence the serving-
    /// gate benchmark found on the real `bank_account` task: round 2's own
    /// accept loop accepted `drafted[3]` (real position 313) as matching
    /// `real_argmax[3]`, but the value committed ("561") does not match
    /// the true continuation ("6558", confirmed by `GraphedDecodeState`
    /// and, separately, by `diagnose_speculative_aider_bench_divergence_
    /// root_cause` replaying the SAME exact position/bucket from a clean
    /// sequential state -- which found the real captured verify-chunk
    /// graph agrees with the TRUE unpadded math there). That test used a
    /// state built via ordinary sequential `forward_one_token` calls.
    /// Round 2's REAL state is different in one specific way: round 0
    /// (pre_pos=298) was a FULL ACCEPT (6/6), which takes
    /// `speculative_decode_round`'s "no rollback" branch -- trusting
    /// `verify_chunk`'s own CHUNKED (7-token-at-once) GDN recurrent-state
    /// computation directly, never rebuilding it via sequential per-token
    /// decode the way a partial-accept round (or plain decode) always
    /// does. That chunked-vs-sequential GDN equivalence has never been
    /// tested specifically for a verify-chunk continuing from a real,
    /// already-mid-generation position (only for prefill from an empty
    /// state, `real_gdn_chunk_forward_prefill_matches_real_transformers_
    /// function`). This test isolates exactly that: two states reaching
    /// the SAME real position (305) by two different real paths -- (A)
    /// plain sequential `forward_one_token` for each of round 0's own 7
    /// real committed tokens, (B) one real `speculative_decode_round`
    /// call reproducing round 0 itself (drafting for real, confirmed
    /// full-accept, taking the real "no rollback" branch) -- then
    /// continues BOTH with the SAME plain sequential replay of the next
    /// several real, independently-confirmed-correct tokens, and checks
    /// whether their final predictions still agree.
    #[test]
    #[ignore]
    fn diagnose_full_accept_branch_gdn_state_matches_sequential_decode() {
        if hip::device_count().unwrap_or(0) == 0 {
            eprintln!("skipping: no HIP device visible on this machine");
            return;
        }
        hip::set_device(0).expect("hipSetDevice(0) failed on a machine that reported a device");

        let snapshot = locate_model_snapshot().expect("no real Qwen3.5-4B snapshot found on this machine");
        let weights = ModelWeights::load(&snapshot).expect("real weight loading failed");
        let mtp_head = MtpDraftHead::load(&snapshot).expect("real MTP head weight loading failed");
        let tokenizer = crate::tokenizer::ChatTokenizer::load(&snapshot).expect("real tokenizer loading failed");
        let handle = BlasHandle::create().expect("real hipblasCreate failed");

        let prompts_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../benchmarks/runtime/speculative/runtime_next_serving_gate/aider_bench_prompts.json");
        let prompts_json = std::fs::read_to_string(&prompts_path).unwrap_or_else(|e| panic!("failed to read {prompts_path:?}: {e}"));
        let parsed: serde_json::Value = serde_json::from_str(&prompts_json).unwrap();
        let task = parsed["tasks"].as_array().unwrap().iter().find(|t| t["name"] == "bank_account").expect("bank_account task missing");
        let templated = tokenizer.apply_chat_template(&[
            ("system", task["system_prompt"].as_str().unwrap()),
            ("user", task["user_prompt"].as_str().unwrap()),
        ]);
        let prompt_ids = tokenizer.encode(&templated).expect("real tokenizer encode failed");
        let max_seq_len = prompt_ids.len() + 1024;

        // Real, trusted reference (same as the earlier diagnostic).
        let mut graphed = model::GraphedDecodeState::new().expect("GraphedDecodeState::new failed");
        let mut ref_state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut ref_logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        for &tok in &prompt_ids {
            graphed.forward_one_token(&weights, &mut ref_state, tok, &mut ref_logits).unwrap();
        }
        let mut ref_tokens: Vec<i32> = Vec::with_capacity(16);
        let mut next_id = argmax_sample(&ref_logits).unwrap();
        for _ in 0..16 {
            ref_tokens.push(next_id);
            graphed.forward_one_token(&weights, &mut ref_state, next_id, &mut ref_logits).unwrap();
            next_id = argmax_sample(&ref_logits).unwrap();
        }
        eprintln!("real reference tokens: {ref_tokens:?}");

        // --- PATH A: clean, plain sequential decode through round 0's own 7 real committed tokens ---
        let mut a_state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut a_logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        for &tok in &prompt_ids {
            forward_one_token(&handle, &weights, &mut a_state, tok, &mut a_logits).unwrap();
        }
        for &tok in &ref_tokens[..7] {
            forward_one_token(&handle, &weights, &mut a_state, tok, &mut a_logits).unwrap();
        }
        eprintln!("PATH A: real position after plain sequential replay = {}", a_state.position);

        // --- PATH B: one real speculative_decode_round call, reproducing round 0 exactly ---
        let mut b_state = DecodeState::new(max_seq_len).expect("DecodeState allocation failed");
        let mut b_logits: DeviceBuffer<u16> = DeviceBuffer::alloc(VOCAB_SIZE).unwrap();
        let mut mtp_state = MtpDraftState::new(max_seq_len).expect("MtpDraftState::new failed");
        forward_one_token(&handle, &weights, &mut b_state, prompt_ids[0], &mut b_logits).unwrap();
        for i in 1..prompt_ids.len() {
            let hidden_ptr: *const DeviceBuffer<u16> = b_state.final_hidden();
            mtp_state.observe(&weights, &mtp_head, unsafe { &*hidden_ptr }, prompt_ids[i]).unwrap();
            forward_one_token(&handle, &weights, &mut b_state, prompt_ids[i], &mut b_logits).unwrap();
        }
        let seed = argmax_sample(&b_logits).unwrap();
        assert_eq!(seed, ref_tokens[0], "seed token must match the real reference");
        let mut graphed_verify = GraphedVerifyState::new(7).expect("GraphedVerifyState::new failed");
        let mut scratch = SpeculativeRoundScratch::new(7).expect("SpeculativeRoundScratch::new failed");
        let result = speculative_decode_round(&handle, &weights, &mut b_state, &mut graphed_verify, &mtp_head, &mut mtp_state, &mut scratch, seed, 6).expect("round 0 replay failed");
        eprintln!("PATH B: round 0 replay accepted={}/6 committed={:?}", result.accepted_count, result.committed_tokens);
        assert_eq!(result.accepted_count, 6, "expected a real, deterministic full accept, matching the original benchmark run's own round 0");
        assert_eq!(result.committed_tokens, ref_tokens[1..8], "round 0's own committed tokens must match the real reference (already confirmed correct)");
        eprintln!("PATH B: real position after real speculative_decode_round (full-accept, no-rollback branch) = {}", b_state.position);
        assert_eq!(a_state.position, b_state.position, "both paths must reach the same real position");

        // --- Continue BOTH with the SAME plain sequential replay of ref_tokens[7..15] (through real position 313, exactly matching the real benchmark's own divergence point), then compare final predictions ---
        for &tok in &ref_tokens[7..15] {
            forward_one_token(&handle, &weights, &mut a_state, tok, &mut a_logits).unwrap();
            forward_one_token(&handle, &weights, &mut b_state, tok, &mut b_logits).unwrap();
        }
        let a_argmax = argmax_sample(&a_logits).unwrap();
        let b_argmax = argmax_sample(&b_logits).unwrap();
        let mut a_host = vec![0u16; VOCAB_SIZE];
        a_logits.copy_to_host(&mut a_host).unwrap();
        let mut b_host = vec![0u16; VOCAB_SIZE];
        b_logits.copy_to_host(&mut b_host).unwrap();
        let diff = a_host.iter().zip(b_host.iter()).filter(|(x, y)| x != y).count();
        eprintln!("after replaying the SAME {} subsequent real tokens on both paths (real position {}):", 8, a_state.position);
        eprintln!("  PATH A (clean sequential) final argmax = {a_argmax}");
        eprintln!("  PATH B (real full-accept branch) final argmax = {b_argmax}");
        eprintln!("  real true continuation (from GraphedDecodeState) = {}", ref_tokens[15]);
        eprintln!("  {diff}/{VOCAB_SIZE} logits differ between the two paths' final predictions");

        if a_argmax == ref_tokens[15] && b_argmax != ref_tokens[15] {
            eprintln!("VERDICT: CONFIRMED. PATH A (clean sequential) matches the true reference; PATH B (real full-accept branch) does not. The full-accept branch's 'trust verify_chunk's own chunked GDN state, no rebuild' optimization leaves a real, measurable divergence from sequential decode.");
        } else if a_argmax == b_argmax {
            eprintln!("VERDICT: NOT this. Both paths agree at this position -- the real divergence has a different root cause.");
        } else {
            eprintln!("VERDICT: INCONCLUSIVE -- needs further isolation.");
        }
    }
}
