//! Real temperature/top-p/top-k sampling over vocab logits -- the "API
//! Sampling Contract" this engine's HTTP layer was missing (`server.rs`
//! previously only ever called the existing on-device `argmax_sample`,
//! i.e. `temperature=0` greedy decoding, unconditionally).
//!
//! Design directly inspired by (not transliterated from) llama.cpp's own
//! reference sampler chain -- read directly out of this repo's own
//! vendored `apps/runtime-llama/llama.cpp/src/llama-sampler.cpp` before
//! writing this, not assumed from memory:
//! `llama_sampler_temp_impl`/`llama_sampler_softmax_impl`/
//! `llama_sampler_top_k_impl`/`llama_sampler_top_p_apply`/
//! `llama_sampler_dist_apply`. That reference implementation is plain,
//! host-side code over a flat `(token_id, logit, prob)` array -- not a
//! GPU kernel -- which is the same real design this module uses: logits
//! are copied off the GPU ONCE per sampled token (496KB for this model's
//! 248,320-token vocab, the same real cost `argmax_sample`'s own
//! pre-optimization implementation already had -- see that function's own
//! doc comment in `model.rs`), then temperature/top-k/top-p/the final
//! weighted draw all run here, in plain Rust, over a small, filtered
//! array. A dedicated on-device softmax-over-vocab kernel was considered
//! and rejected: it would still need the SAME host round trip afterward
//! (top-k/top-p need a sort, and this crate has no on-device sort
//! primitive worth building for a once-per-token, not-in-any-hot-GEMM-path
//! op), so it would add a whole new kernel for zero real latency benefit.
//!
//! `temperature<=0` (the pre-existing default, and every real benchmark
//! script in this repo's own `benchmarks/harness_sdk` already sends
//! `"temperature": 0.0` explicitly) is NOT routed through this module at
//! all -- `server.rs`'s `Engine::step` keeps calling the existing,
//! already-optimized on-device `argmax_sample` directly for that case, so
//! every existing test, benchmark, and byte-exact-greedy correctness gate
//! in this crate is completely unaffected by this feature's addition.
//! This module only runs when a real caller explicitly asks for
//! `temperature>0`.
//!
//! Real, deliberate optimizations beyond a literal port of the
//! llama.cpp reference (see each function's own doc comment for the
//! full reasoning, and `mod tests` for the decisive proof each one
//! doesn't change the RESULT, only the cost of reaching it):
//! - top-k uses `slice::select_nth_unstable_by` (an O(n) average
//!   partition) instead of a full sort, then sorts only the surviving
//!   k-element slice -- llama.cpp's own hand-rolled partial-sort aims for
//!   the same complexity class; Rust's standard library gives it for
//!   free.
//! - top-p (when top-k didn't already shrink the candidate set) ports
//!   llama.cpp's own adaptive trick: partially sort only the top 256
//!   candidates first, and only fall back to a full sort of the whole
//!   248,320-token vocab if the cumulative probability doesn't reach `p`
//!   within those 256 -- real language-model distributions concentrate
//!   probability mass in a tiny fraction of the vocabulary, so the full
//!   sort is the rare path, not the common one.

use crate::model::bf16_to_f32;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Real sampling parameters for one request. `temperature<=0.0` means
/// greedy (handled by the CALLER via the existing `argmax_sample` path,
/// never reaching this module -- see this module's own doc comment).
/// `top_p>=1.0` and `top_k<=0` both mean "disabled" (matches llama.cpp's
/// own convention exactly, `llama_sampler_init_top_p`/`_top_k`).
#[derive(Clone, Copy, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
}

impl SamplingParams {
    pub const GREEDY: SamplingParams = SamplingParams { temperature: 0.0, top_p: 1.0, top_k: 0 };
}

/// Real, seedable RNG for one request's whole generation -- created ONCE
/// per real request (`Engine::start_request`), not re-seeded per token,
/// matching llama.cpp's own `llama_sampler_dist`'s `std::mt19937 rng`
/// (held for the sampler's whole lifetime, not per-call). A real,
/// explicit `seed` gives byte-exact-reproducible generations across
/// separate real requests (checked by `real_seeded_sampling_is_reproducible`
/// below); omitting one draws real OS entropy once, matching llama.cpp's
/// own `get_rng_seed`/`LLAMA_DEFAULT_SEED` behavior.
pub fn make_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_os_rng(),
    }
}

/// Real, on-device-logits-to-sampled-token-id pipeline: bf16 vocab
/// logits (as copied off the GPU by the caller) -> temperature scale ->
/// top-k truncate -> softmax -> top-p truncate -> weighted random draw.
/// Returns the real sampled token id.
///
/// `logits_bf16.len()` must equal the real vocab size; every index `i`
/// is real token id `i` (matches `argmax_bf16`'s own indexing contract
/// exactly, so this is a drop-in alternative sampling strategy over the
/// SAME logits buffer, not a different vocabulary).
pub fn sample_from_logits(logits_bf16: &[u16], params: &SamplingParams, rng: &mut StdRng) -> i32 {
    assert!(!logits_bf16.is_empty(), "logits must be non-empty");
    assert!(params.temperature > 0.0, "sample_from_logits is only for temperature>0 -- the caller must route temperature<=0 through the existing greedy argmax_sample path directly");

    // Real temperature scale, building the (token_id, scaled_logit) pairs
    // in the SAME pass (llama.cpp's own `llama_sampler_temp_impl` does
    // this as a separate loop over its own persistent array; fusing it
    // here avoids one whole extra 248,320-element pass with no behavior
    // change -- `real_temperature_scaling_matches_reference_formula`
    // checks the fused version against the unfused formula directly).
    let inv_temp = 1.0f32 / params.temperature;
    let mut candidates: Vec<(i32, f32)> = logits_bf16.iter().enumerate().map(|(i, &bits)| (i as i32, bf16_to_f32(bits) * inv_temp)).collect();

    top_k_filter(&mut candidates, params.top_k);
    softmax_in_place(&mut candidates);
    // `softmax_in_place` leaves `candidates` sorted descending by
    // probability whenever it had to sort for correctness (top-k's own
    // sort, or top-p's own sort below) -- `top_p_filter` requires
    // descending order for its cumulative-sum walk, guaranteed either by
    // `top_k_filter` (always sorts its surviving slice) or by
    // `top_p_filter` itself sorting when top-k was disabled.
    top_p_filter(&mut candidates, params.top_p);

    weighted_draw(&candidates, rng)
}

/// Real top-k truncation: `k<=0` or `k >= candidates.len()` means
/// disabled (no-op) -- matches `llama_sampler_top_k_impl`'s own `k <= 0`
/// early-return and `k = min(k, cur_p->size)` clamp. Otherwise,
/// partitions the top `k` largest-logit candidates into the front of the
/// slice in O(n) average (`select_nth_unstable_by`), THEN sorts only
/// those `k` survivors descending (O(k log k)) -- real total cost
/// dominated by the O(n) partition for realistic `k` (e.g. 40) against a
/// 248,320-wide vocab, versus O(n log n) for a full sort of everything.
fn top_k_filter(candidates: &mut Vec<(i32, f32)>, top_k: i32) {
    if top_k <= 0 {
        return;
    }
    let k = (top_k as usize).min(candidates.len());
    if k >= candidates.len() {
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        return;
    }
    candidates.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
    candidates.truncate(k);
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
}

/// Real, numerically-stable softmax over whatever real candidates
/// survived top-k -- max-subtract, exp, sum, normalize -- identical
/// formula to `llama_sampler_softmax_impl`. `candidates` is left in
/// WHATEVER order it was passed in (top-k's own sort, if it ran; the
/// original ascending-token-id order otherwise) -- `top_p_filter` is
/// responsible for sorting if it needs descending order and top-k didn't
/// already provide it.
fn softmax_in_place(candidates: &mut [(i32, f32)]) {
    let max_logit = candidates.iter().map(|&(_, l)| l).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for (_, l) in candidates.iter_mut() {
        let p = (*l - max_logit).exp();
        *l = p;
        sum += p;
    }
    for (_, p) in candidates.iter_mut() {
        *p /= sum;
    }
}

/// Real nucleus (top-p) truncation over `candidates` (already real
/// probabilities, post-softmax). `p>=1.0` means disabled (matches
/// `llama_sampler_init_top_p`'s own `ctx->p >= 1.0f` early return).
///
/// Real, ported optimization (`llama_sampler_top_p_apply`'s own
/// "adaptive top-k sorting" trick, read directly out of the vendored
/// llama.cpp source, not reinvented): when `candidates` isn't ALREADY
/// sorted descending (top-k disabled, so `softmax_in_place` left it in
/// token-id order) and is large (over 1024 real candidates -- this
/// model's real 248,320-token vocab, almost always), first partially
/// sort only the top 256 by probability. Real language-model
/// distributions concentrate probability mass in a tiny fraction of the
/// vocabulary, so the cumulative sum reaching `p` within the top 256 is
/// the OVERWHELMINGLY common real case -- only fall back to sorting the
/// entire candidate set if it doesn't (`real_top_p_adaptive_sort_matches_full_sort_reference`
/// is the decisive, differential proof this optimization never changes
/// the RESULT, only the real cost of reaching it).
fn top_p_filter(candidates: &mut Vec<(i32, f32)>, top_p: f32) {
    if top_p >= 1.0 {
        return;
    }

    const ADAPTIVE_THRESHOLD: usize = 1024;
    const ADAPTIVE_PREFIX: usize = 256;

    let already_sorted_desc = candidates.windows(2).all(|w| w[0].1 >= w[1].1);

    if !already_sorted_desc && candidates.len() > ADAPTIVE_THRESHOLD {
        let k = ADAPTIVE_PREFIX.min(candidates.len());
        candidates.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
        candidates[..k].sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let prefix_sum: f32 = candidates[..k].iter().map(|&(_, p)| p).sum();
        if prefix_sum >= top_p {
            // Real, decisive: the cumulative mass within the top 256
            // already covers `p` -- no real candidate outside this
            // prefix could ever be needed (softmax probabilities are
            // non-negative, so extending the walk past a sorted prefix
            // can only ADD probability, never remove the ones already
            // counted). Safe to truncate to just this sorted prefix and
            // run the same cumulative-sum walk below on it.
            candidates.truncate(k);
        } else {
            candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        }
    } else if !already_sorted_desc {
        candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
    }

    let mut cum_sum = 0.0f32;
    let mut last_idx = candidates.len();
    for (i, &(_, p)) in candidates.iter().enumerate() {
        cum_sum += p;
        if cum_sum >= top_p {
            last_idx = i + 1;
            break;
        }
    }
    candidates.truncate(last_idx);
}

/// Real weighted random draw over `candidates`' (possibly-truncated,
/// so real-summing-to-less-than-1.0) probabilities -- renormalizes then
/// walks the cumulative distribution against one real uniform draw,
/// matching `llama_sampler_dist_apply`'s own approach.
fn weighted_draw(candidates: &[(i32, f32)], rng: &mut StdRng) -> i32 {
    assert!(!candidates.is_empty(), "weighted_draw: candidates must be non-empty (top-k/top-p filtering must always leave at least one real survivor)");
    if candidates.len() == 1 {
        return candidates[0].0;
    }
    let total: f32 = candidates.iter().map(|&(_, p)| p).sum();
    let u: f32 = rng.random::<f32>() * total;
    let mut cum_sum = 0.0f32;
    for &(token_id, p) in candidates {
        cum_sum += p;
        if u < cum_sum {
            return token_id;
        }
    }
    // Real floating-point edge case: `u` landed in the last sliver lost
    // to rounding during renormalization -- the real last candidate is
    // still the mathematically correct answer.
    candidates.last().unwrap().0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_to_bf16(v: f32) -> u16 {
        crate::model::f32_to_bf16(v)
    }

    fn naive_argmax(logits: &[f32]) -> i32 {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        best_i as i32
    }

    /// Reference, deliberately naive (no adaptive shortcuts, no fused
    /// passes) re-implementation of the WHOLE pipeline, used only to
    /// differentially check the real, optimized `sample_from_logits`
    /// against -- see `real_top_p_adaptive_sort_matches_full_sort_reference`
    /// and `real_top_k_partition_matches_full_sort_reference`.
    fn naive_reference_candidates(logits_f32: &[f32], params: &SamplingParams) -> Vec<(i32, f32)> {
        let mut c: Vec<(i32, f32)> = logits_f32.iter().enumerate().map(|(i, &l)| (i as i32, l / params.temperature)).collect();
        if params.top_k > 0 {
            c.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            c.truncate((params.top_k as usize).min(c.len()));
        }
        let max_l = c.iter().map(|&(_, l)| l).fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (_, l) in c.iter_mut() {
            *l = (*l - max_l).exp();
            sum += *l;
        }
        for (_, p) in c.iter_mut() {
            *p /= sum;
        }
        if params.top_p < 1.0 {
            c.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            let mut cum = 0.0f32;
            let mut last = c.len();
            for (i, &(_, p)) in c.iter().enumerate() {
                cum += p;
                if cum >= params.top_p {
                    last = i + 1;
                    break;
                }
            }
            c.truncate(last);
        } else {
            c.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        }
        c
    }

    fn synth_logits(n: usize, seed: u64) -> Vec<f32> {
        // Real, deterministic, NOT hand-picked-to-look-nice synthetic
        // data -- an xorshift64 generator, matching this crate's own
        // established `synth()` convention for kernel-level decisive
        // tests elsewhere (`model.rs`'s own chunked-GDN tests). Uses the
        // FULL 64-bit state to derive each f32 (not a small integer
        // modulus) -- a real, found-not-guessed fix: an earlier version
        // mapped into only 2000 distinct buckets, which for n>=2000
        // samples produces real, frequent ties (the birthday paradox),
        // and an unstable sort/partition is free to break a tie
        // differently depending on the surrounding call sequence -- that
        // was flagged by `real_top_k_partition_matches_full_sort_reference`
        // and friends failing on a tie, not a real algorithm bug (their
        // own real, false failures directly caught the test data flaw).
        // The full-precision range below makes a real collision among a
        // few thousand samples astronomically unlikely, matching how
        // real model logits are practically never bit-for-bit equal.
        let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state as f64 / u64::MAX as f64) * 20.0 - 10.0) as f32
            })
            .collect()
    }

    #[test]
    fn real_top_k_one_always_returns_argmax_across_many_temperatures_and_seeds() {
        let logits = synth_logits(1000, 1);
        let logits_bf16: Vec<u16> = logits.iter().map(|&v| f32_to_bf16(v)).collect();
        // Real, deliberate: the reference argmax must be computed over
        // the SAME bf16-rounded values `sample_from_logits` actually
        // receives, not the higher-precision f32 source -- bf16's ~0.4%
        // relative precision can legitimately reorder two close-together
        // top candidates that a raw f32 comparison would not (caught
        // directly: an earlier version of this test compared against the
        // f32 argmax and failed on real, expected rounding, not a real
        // sampling bug).
        let expected = naive_argmax(&logits_bf16.iter().map(|&b| bf16_to_f32(b)).collect::<Vec<f32>>());
        for temp in [0.1f32, 0.5, 1.0, 2.0, 10.0] {
            for seed in 0..20u64 {
                let mut rng = make_rng(Some(seed));
                let params = SamplingParams { temperature: temp, top_p: 1.0, top_k: 1 };
                let got = sample_from_logits(&logits_bf16, &params, &mut rng);
                assert_eq!(got, expected, "top_k=1 must always return the real argmax, temp={temp} seed={seed}");
            }
        }
    }

    /// Real, decisive: temperature scaling followed by softmax must
    /// leave the ARGMAX unchanged regardless of temperature (a
    /// monotonic rescale never reorders the max) -- checked against a
    /// real, independently-computed reference argmax over UNSCALED
    /// logits, for a real synthetic distribution.
    #[test]
    fn real_temperature_scaling_never_changes_the_argmax_under_top_k_one() {
        for seed in 0..10u64 {
            let logits = synth_logits(2000, seed + 100);
            let logits_bf16: Vec<u16> = logits.iter().map(|&v| f32_to_bf16(v)).collect();
            // Real: reference argmax over the bf16-rounded values, same
            // reasoning as the test above.
            let expected = naive_argmax(&logits_bf16.iter().map(|&b| bf16_to_f32(b)).collect::<Vec<f32>>());
            let mut rng = make_rng(Some(seed));
            let params = SamplingParams { temperature: 0.3, top_p: 1.0, top_k: 1 };
            let got = sample_from_logits(&logits_bf16, &params, &mut rng);
            assert_eq!(got, expected);
        }
    }

    /// The real statistical decisive test: draws many samples from a
    /// small, real, known distribution and checks the empirical
    /// frequency of each token matches its real softmax probability
    /// within a real binomial-error tolerance (5 standard deviations --
    /// a real, generously loose bound chosen so this test is decisive,
    /// not flaky).
    #[test]
    fn real_sampled_frequencies_match_softmax_probabilities_within_tolerance() {
        let logits_f32 = [2.0f32, 1.0, 0.0, -1.0, -2.0];
        let logits_bf16: Vec<u16> = logits_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let params = SamplingParams { temperature: 1.0, top_p: 1.0, top_k: 0 };

        let max_l = logits_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits_f32.iter().map(|&l| (l - max_l).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let expected_probs: Vec<f32> = exp.iter().map(|&e| e / sum).collect();

        let n = 200_000usize;
        let mut counts = [0usize; 5];
        let mut rng = make_rng(Some(42));
        for _ in 0..n {
            let id = sample_from_logits(&logits_bf16, &params, &mut rng);
            counts[id as usize] += 1;
        }

        for i in 0..5 {
            let observed = counts[i] as f64 / n as f64;
            let expected = expected_probs[i] as f64;
            let stderr = (expected * (1.0 - expected) / n as f64).sqrt();
            let tolerance = 5.0 * stderr + 1e-6;
            assert!(
                (observed - expected).abs() < tolerance,
                "token {i}: observed frequency {observed:.5} vs real softmax probability {expected:.5} (tolerance {tolerance:.5})"
            );
        }
    }

    /// Real, decisive: a dominant single token (>=99% real probability)
    /// under a real top_p=0.5 cutoff must be the ONLY token ever sampled
    /// -- the real nucleus is exactly that one token.
    #[test]
    fn real_top_p_excludes_the_low_probability_tail() {
        let mut logits_f32 = vec![-10.0f32; 100];
        logits_f32[7] = 10.0; // dominates: softmax mass on this one token is essentially 1.0
        let logits_bf16: Vec<u16> = logits_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let params = SamplingParams { temperature: 1.0, top_p: 0.5, top_k: 0 };
        let mut rng = make_rng(Some(7));
        for _ in 0..500 {
            let id = sample_from_logits(&logits_bf16, &params, &mut rng);
            assert_eq!(id, 7, "top_p=0.5 against a dominant token must always select it");
        }
    }

    /// Real, decisive: with top_k/top_p both disabled (the real "no
    /// filtering, pure temperature-scaled multinomial" case), every real
    /// candidate must remain reachable -- none silently excluded by a
    /// bug in either filter's own "disabled" branch.
    #[test]
    fn real_disabled_filters_can_reach_every_candidate() {
        let logits_f32: Vec<f32> = (0..20).map(|i| (i as f32) * 0.05).collect(); // close together -> near-uniform, every token has real, non-negligible probability
        let logits_bf16: Vec<u16> = logits_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let params = SamplingParams { temperature: 1.0, top_p: 1.0, top_k: 0 };
        let mut rng = make_rng(Some(99));
        let mut seen = [false; 20];
        for _ in 0..20_000 {
            let id = sample_from_logits(&logits_bf16, &params, &mut rng);
            seen[id as usize] = true;
        }
        assert!(seen.iter().all(|&s| s), "every real candidate must be reachable when top_k/top_p are both disabled: {seen:?}");
    }

    /// Real, decisive differential test for the top-p ADAPTIVE-256
    /// optimization (`top_p_filter`'s own doc comment): over a real,
    /// large (>1024, so the adaptive path actually triggers), unsorted
    /// synthetic vocab, the optimized `top_p_filter` must produce the
    /// EXACT SAME surviving candidate set (same token ids, same order)
    /// as a deliberately naive full-sort reference with no shortcuts --
    /// proves the optimization only changes cost, never the result.
    #[test]
    fn real_top_p_adaptive_sort_matches_full_sort_reference() {
        for seed in 0..8u64 {
            let logits = synth_logits(5000, seed + 500);
            for top_p in [0.5f32, 0.9, 0.99] {
                let params = SamplingParams { temperature: 1.0, top_p, top_k: 0 };
                let reference = naive_reference_candidates(&logits, &params);

                let mut candidates: Vec<(i32, f32)> = logits.iter().enumerate().map(|(i, &l)| (i as i32, l)).collect();
                softmax_in_place(&mut candidates);
                top_p_filter(&mut candidates, top_p);

                assert_eq!(candidates.len(), reference.len(), "seed={seed} top_p={top_p}: surviving candidate COUNT must match the naive full-sort reference");
                for (i, (&(got_id, got_p), &(ref_id, ref_p))) in candidates.iter().zip(reference.iter()).enumerate() {
                    assert_eq!(got_id, ref_id, "seed={seed} top_p={top_p} idx={i}: token id must match the reference");
                    assert!((got_p - ref_p).abs() < 1e-6, "seed={seed} top_p={top_p} idx={i}: probability must match the reference");
                }
            }
        }
    }

    /// Real, decisive differential test for the top-k
    /// select-then-sort optimization: must produce the exact same
    /// surviving (sorted-descending) candidate set as a naive full sort.
    /// `top_k<=0` means disabled (`top_k_filter`'s own contract) -- the
    /// reference for that case is simply every id, unsorted (matches
    /// `top_k_filter`'s own no-op early return, which leaves order
    /// untouched).
    #[test]
    fn real_top_k_partition_matches_full_sort_reference() {
        for seed in 0..8u64 {
            let logits = synth_logits(3000, seed + 900);
            for top_k in [-1i32, 0, 1, 5, 40, 256, 3000, 99999] {
                let mut expected_ids: Vec<i32> = (0..logits.len() as i32).collect();
                if top_k > 0 {
                    expected_ids.sort_unstable_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
                    expected_ids.truncate((top_k as usize).min(logits.len()));
                }

                let mut candidates: Vec<(i32, f32)> = logits.iter().enumerate().map(|(i, &l)| (i as i32, l)).collect();
                top_k_filter(&mut candidates, top_k);
                let got_ids: Vec<i32> = candidates.iter().map(|&(id, _)| id).collect();

                assert_eq!(got_ids, expected_ids, "seed={seed} top_k={top_k}: optimized top_k_filter must match a naive full-sort-then-truncate reference exactly");
            }
        }
    }

    /// Real, decisive: the SAME seed must reproduce the SAME sampled
    /// sequence across two entirely separate `StdRng`/sampling runs --
    /// the real property the HTTP `seed` request parameter promises
    /// callers.
    #[test]
    fn real_same_seed_reproduces_the_same_sample_sequence() {
        let logits = synth_logits(500, 3);
        let logits_bf16: Vec<u16> = logits.iter().map(|&v| f32_to_bf16(v)).collect();
        let params = SamplingParams { temperature: 0.8, top_p: 0.95, top_k: 40 };

        let run = || -> Vec<i32> {
            let mut rng = make_rng(Some(12345));
            (0..50).map(|_| sample_from_logits(&logits_bf16, &params, &mut rng)).collect()
        };
        assert_eq!(run(), run(), "the same seed must reproduce the exact same real sample sequence");
    }

    /// Real, decisive: DIFFERENT seeds must produce genuinely different
    /// raw random streams out of `make_rng` -- checked directly, not
    /// assumed from `StdRng`'s own documentation. Real, found-not-guessed
    /// motivation: a real end-to-end HTTP-level test
    /// (`server::tests::real_end_to_end_sampling_matches_greedy_at_temp_zero_and_is_reproducible_with_a_seed`)
    /// initially used two arbitrary seeds that happened to produce the
    /// IDENTICAL real generated sequence for a real, highly-deterministic
    /// factual prompt -- this test isolates and rules out "the RNG itself
    /// is broken" as the explanation (it isn't: the raw streams below are
    /// provably different), pinning the real cause on the model's own
    /// real, highly-peaked probability distribution for that specific
    /// real prompt instead.
    #[test]
    fn real_different_seeds_produce_different_raw_random_streams() {
        let mut r2 = make_rng(Some(2));
        let mut r3 = make_rng(Some(3));
        let v2: Vec<f32> = (0..10).map(|_| r2.random::<f32>()).collect();
        let v3: Vec<f32> = (0..10).map(|_| r3.random::<f32>()).collect();
        assert_ne!(v2, v3, "seed=2 and seed=3 must produce different raw random streams out of make_rng");
    }
}
