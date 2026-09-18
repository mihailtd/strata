# Speculative decoding on `runtime-next` (Rust/HIP) — built, correctness-proven, and REMOVED: 0.79-0.85x, reinforces this repo's existing finding

**Verdict: do not ship it. Real, measured, reproduced: 15-21% SLOWER than
plain sequential decode, even at a real 58% draft-acceptance rate.** Archived
here rather than left in the production crate — the code is real, tested, and
correct, but does not belong in `apps/runtime-next/src/` since it makes the
engine slower, not faster.

This corroborates, on a second and structurally different engine (a
from-scratch Rust/HIP port of Qwen3.5-4B, vs. this repo's existing PyTorch/CUDA
27B work), the same qualitative conclusion
[`../serving_gate/benchmark_graph_vs_speculative.py`](../serving_gate/benchmark_graph_vs_speculative.py)
and
[`../loop_profile/README.md`](../loop_profile/README.md)
already reached: speculative decoding has to beat an ALREADY-OPTIMIZED
autoregressive decode path (HIP Graph replay here; CUDA Graph replay there),
not naive eager decode, and that bar is genuinely hard to clear once the
non-forward-pass bookkeeping (state snapshot/restore, accept/reject argmax)
is measured rather than assumed cheap.

## What this was

`apps/runtime-next` (a from-scratch Rust/HIP Qwen3.5-4B inference engine,
`docs/DECISIONS.md` §91-§96) got real batched prefill in §96. Verification
during speculative decoding is structurally the same batched-forward-pass
shape as prefill, so §97 built speculative decoding on top of it to see
whether it would help:

- **`prompt_lookup_draft`**: real, model-free drafting (HF transformers' own
  `PromptLookupCandidateGenerator` technique) — search the running context
  for the most recent earlier occurrence of its own last N tokens, draft
  whatever followed that occurrence. No draft model needed, no training.
- **`speculative_round`**: real accept/reject/rollback/replay protocol —
  verify the whole draft in one batched forward pass, walk each drafted
  token against the base model's own real prediction, roll back and replay
  on the first mismatch.
- **GDN state snapshot/restore**: unlike the KV cache (whose unread
  positions are structurally harmless — see `DecodeState::reset()`'s own
  reasoning in `model.rs`), GatedDeltaNet's recurrent/conv state is real
  read-modify-write regardless of position, so a rejected draft's
  speculative state corruption has to be genuinely undone via a real
  device-to-device snapshot/restore.

Full technical writeup: `docs/DECISIONS.md` §97.

## The real result

Two real prompts, `apps/runtime-next`'s own plain eager `forward_one_token`
as the baseline (the fair comparison — `speculative_round`'s own fallback/
replay path already calls it; NOT the faster `GraphedDecodeState`/HIP-Graph
path the HTTP server actually uses for steady-state decode, since
speculative decoding was never wired into that graphed path):

| prompt | sequential | speculative | rounds | drafted/accepted | speedup |
| :--- | ---: | ---: | ---: | ---: | ---: |
| repetitive_boilerplate | 81.27 tok/s | 64.92 tok/s | 54 | 114/66 (57.9%) | **0.799x** |
| creative_low_repetition | 82.10 tok/s | 68.60 tok/s | 103 | 60/17 (28.3%) | **0.836x** |

Investigated before accepting the verdict (same discipline as this repo's own
`loop_profile` experiment): switched 48 blocking device-to-device
GDN-state-snapshot copies per round to async + one sync. Barely moved the
number (0.794x→0.799x) — **the snapshot/restore overhead was NOT the
dominant cost**, matching `loop_profile/README.md`'s own measured breakdown
that `snapshot` is only ~1.1% of that engine's speculative loop. The real
remaining cost here is short average accepted-draft length (~2.1
tokens/round even on a deliberately repetitive prompt — `prompt_lookup_draft`'s
own match availability is the limiter, not the `num_draft` cap), too short to
amortize a batched verify call's real per-position loop overhead (RoPE/
KV-cache-append/attention across 8 layers, causal-conv1d/gate/recurrent-state
across 24 layers — all real per-launch cost even though individually cheap,
the same cost structure `docs/DECISIONS.md` §96 already documents for
prefill) against its one real win (batched GEMM weight-read amortization).

**Not a verdict against speculative decoding in general** — a trained MTP
draft head (this repo's 27B engine's own `tau≈2` acceptance, vs. prompt-lookup's
much weaker ~0.2-1.2 tokens/round here) or a lower-overhead round protocol
could plausibly flip this. Both real, scoped, unattempted.

## Files

- `archived_model.rs`, `archived_kernels.rs`, `archived_hip.rs` — full
  snapshots of `apps/runtime-next/src/{model,kernels,hip}.rs` AS THEY STOOD
  with speculative decoding still built in, before it was stripped back out
  of the production crate. Not compiled, not part of any build — reference
  only. The relevant pieces, if reviving this: `prompt_lookup_draft`,
  `SpeculativeRoundResult`, `speculative_round`, `forward_verify_chunk`,
  `run_verify_chunk_body`, `DecodeState::snapshot_gdn_state`/
  `restore_gdn_state`/`gdn_snapshot`, `PrefillScratch::verify_final_normed`/
  `verify_logits`, `MAX_DRAFT_TOKENS` (all in `archived_model.rs`);
  `argmax_bf16_at` (`archived_kernels.rs`); `copy_from_device`/
  `copy_from_device_async` (`archived_hip.rs`). Two decisive tests are
  preserved in `archived_model.rs`:
  `real_speculative_decode_matches_real_sequential_greedy_generation`
  (correctness: speculative output bit-identical to sequential greedy,
  passed on its first real run) and
  `bench_real_speculative_vs_sequential_decode` (the real A/B above).
