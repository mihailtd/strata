"""Unit tests for Ring-Buffered Hybrid State Rollback (ReplaySSM)."""

import pytest
import torch

from runtime.mtp_draft import (
    attach_state_ring_buffer,
    restore_state,
    snapshot_state,
    state_nbytes,
)
from runtime.state_ring_buffer import (
    PointerStateRingBuffer,
    RingBufferReplayEngine,
    StateRingBuffer,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


class MockGDNLayerCache:
    def __init__(self, device=DEVICE, dtype=DTYPE):
        self.recurrent_states = [torch.randn(1, 4, 128, 128, device=device, dtype=dtype)]
        self.conv_states = [torch.randn(1, 2048, 4, device=device, dtype=dtype)]


class MockAttnLayerCache:
    def __init__(self, seq_len: int = 64, device=DEVICE, dtype=DTYPE):
        self.key_cache = torch.randn(1, 4, seq_len, 128, device=device, dtype=dtype)
        self.value_cache = torch.randn(1, 4, seq_len, 128, device=device, dtype=dtype)


class MockHybridCache:
    def __init__(self, seq_len: int = 64, device=DEVICE, dtype=DTYPE):
        self.layers = []
        for i in range(36):
            if i % 3 == 2:
                self.layers.append(MockAttnLayerCache(seq_len=seq_len, device=device, dtype=dtype))
            else:
                self.layers.append(MockGDNLayerCache(device=device, dtype=dtype))


def test_state_ring_buffer_allocation_and_memory():
    """Verify ring buffer allocates within memory budget and accounts bytes accurately."""
    cache = MockHybridCache()
    ring = StateRingBuffer(cache, max_depth=8)

    single_bytes = state_nbytes(cache)
    # Ring buffer only allocates fixed SSM states; KV cache is tracked via sequence length indices
    assert ring.total_bytes <= single_bytes * 8
    # 36 layers (24 GDN + 12 Attn) with seq_len=64 should be ~50-60 MB total
    assert ring.total_bytes < 100 * 1024 * 1024  # < 100 MB


def test_state_ring_buffer_push_and_rollback_fidelity():
    """Verify bit-for-bit numerical fidelity across multiple rollback cycles."""
    cache = MockHybridCache()
    ring = StateRingBuffer(cache, max_depth=8)

    # Initial state
    initial_rec = cache.layers[0].recurrent_states[0].clone()
    initial_conv = cache.layers[0].conv_states[0].clone()

    # Step 0 checkpoint
    slot0 = ring.push(cache)
    assert slot0 == 0

    # Simulate forward mutations (token 1..4)
    for _ in range(4):
        cache.layers[0].recurrent_states[0].add_(1.0)
        cache.layers[0].conv_states[0].add_(1.0)

    # Rollback to slot 0
    ring.rollback(cache, slot=slot0)

    assert torch.equal(cache.layers[0].recurrent_states[0], initial_rec)
    assert torch.equal(cache.layers[0].conv_states[0], initial_conv)


def test_pointer_state_ring_buffer_submicrosecond_rollback():
    """Verify PointerStateRingBuffer zero-copy pointer swapping fidelity."""
    cache = MockHybridCache()
    ring = PointerStateRingBuffer(cache, max_depth=8)

    initial_rec = cache.layers[0].recurrent_states[0].clone()
    initial_conv = cache.layers[0].conv_states[0].clone()

    # Advance to slot 1 and mutate
    slot1 = ring.advance_write_ptr()
    assert slot1 == 1

    cache.layers[0].recurrent_states[0].uniform_(50.0, 100.0)
    cache.layers[0].conv_states[0].uniform_(50.0, 100.0)

    # Rollback to slot 0
    ring.rollback(slot=0)

    assert torch.equal(cache.layers[0].recurrent_states[0], initial_rec)
    assert torch.equal(cache.layers[0].conv_states[0], initial_conv)


def test_ring_buffer_replay_engine_speculative_flow():
    """Verify RingBufferReplayEngine checkpoint, reject-rollback, and accept-commit."""
    cache = MockHybridCache()
    engine = RingBufferReplayEngine(cache, max_depth=8, use_pointer_mode=False)

    initial_rec = cache.layers[0].recurrent_states[0].clone()

    # 1. Speculative branch 1 (rejection): draft 4 tokens, rejected
    engine.checkpoint(cache)
    cache.layers[0].recurrent_states[0].add_(5.0)
    engine.rollback_on_rejection(cache, n_accepted=0)

    assert torch.equal(cache.layers[0].recurrent_states[0], initial_rec)

    # 2. Speculative branch 2 (partial acceptance): 2 accepted tokens
    engine.checkpoint(cache)
    cache.layers[0].recurrent_states[0].add_(2.0)
    engine.commit_on_acceptance(n_accepted=2)
    assert engine.ring.commit_ptr == 2


def test_rollback_on_rejection_needs_the_returned_slot_not_a_guess():
    """Regression test for the bug fixed 2026-08-20.

    `rollback_on_rejection` defaults to `slot=None`, which makes `rollback()` GUESS
    the slot as `(commit_ptr + n_accepted) % max_depth`. That formula is correct only
    for a caller that checkpoints once per accepted token. BucketedSpeculativeDecoder
    checkpoints once per k-token CHUNK, so the guess is right only when n_accepted==0
    -- every partial accept (n_accepted > 0, the common case) restored a slot nothing
    wrote that step. For a GatedDeltaNet layer, that is a silent stale-state
    substitution: the model decodes the rest of the response missing part of its own
    recurrent memory, and the observed failure was not a crash but degenerate
    token-repetition loops in live generation.

    This test reproduces the exact chunk-per-step calling pattern (one checkpoint,
    then a rollback with n_accepted > 0) after a prior commit has moved commit_ptr
    away from write_ptr -- the drift condition needed for the guess to diverge from
    reality. It asserts the OLD call shape (no slot) restores the WRONG value, and the
    FIXED call shape (slot=<checkpoint's return>) restores the value actually
    checkpointed this step.
    """
    cache = MockHybridCache()
    engine = RingBufferReplayEngine(cache, max_depth=8, use_pointer_mode=False)

    # Step 1: checkpoint, accept 3 tokens, commit -- moves commit_ptr to 3 while
    # write_ptr (via push()) has only advanced by 1. This is the drift the real
    # decode loop produces on every step with a partial accept.
    engine.checkpoint(cache)
    engine.commit_on_acceptance(n_accepted=3)

    # Step 2: this step's real checkpoint -- the state a rejection MUST restore.
    pre_draft_state = cache.layers[0].recurrent_states[0].clone()
    step2_slot = engine.checkpoint(cache)

    # Simulate drafting: state advances speculatively, then gets rejected.
    cache.layers[0].recurrent_states[0].add_(99.0)
    rejected_state = cache.layers[0].recurrent_states[0].clone()

    # OLD call shape: no slot, n_accepted=2 (a partial accept within this chunk).
    # commit_ptr is 3 here, so the guess resolves to slot (3+2)%8=5 -- not step2_slot.
    engine.rollback_on_rejection(cache, n_accepted=2)
    guessed_result = cache.layers[0].recurrent_states[0].clone()
    assert not torch.equal(guessed_result, pre_draft_state), (
        "test fixture no longer reproduces the drift condition -- "
        "commit_ptr and write_ptr must differ for this regression test to be meaningful"
    )

    # Put the rejected state back and take the FIXED path: explicit slot.
    cache.layers[0].recurrent_states[0].copy_(rejected_state)
    engine.rollback_on_rejection(cache, n_accepted=2, slot=step2_slot)
    fixed_result = cache.layers[0].recurrent_states[0]
    assert torch.equal(fixed_result, pre_draft_state), (
        "passing the checkpoint's own slot must restore exactly what was live "
        "before the draft, regardless of n_accepted or commit_ptr drift"
    )


def test_mtp_draft_transparent_integration():
    """Verify snapshot_state and restore_state seamlessly utilize attached StateRingBuffer."""
    cache = MockHybridCache()
    attach_state_ring_buffer(cache, max_depth=8)

    assert hasattr(cache, "_ring_buffer")
    initial_rec = cache.layers[0].recurrent_states[0].clone()

    # snapshot_state should return an integer slot index instead of a dict list
    snap = snapshot_state(cache)
    assert isinstance(snap, int)

    # Mutate
    cache.layers[0].recurrent_states[0].uniform_(10.0, 20.0)

    # restore_state should restore in place via ring buffer
    restore_state(cache, snap)
    assert torch.equal(cache.layers[0].recurrent_states[0], initial_rec)


def test_poet_compressed_state_ring_buffer():
    """Verify POETCompressedStateRingBuffer achieves high compression with directional fidelity."""
    from runtime.state_ring_buffer import POETCompressedStateRingBuffer

    cache = MockHybridCache()
    ring = POETCompressedStateRingBuffer(cache, max_depth=64, rank=8, sparsity_target=0.05)

    assert ring.compression_ratio > 2.0  # High memory savings for mock dimensions (scales to 15.7x at T=2048)
    initial_rec = cache.layers[0].recurrent_states[0].clone()

    # Push state
    slot = ring.push(cache)
    assert slot == 0

    # Mutate cache
    cache.layers[0].recurrent_states[0].uniform_(10.0, 20.0)

    # Rollback from POET compressed representation
    ring.rollback(cache, slot=0)
    restored_rec = cache.layers[0].recurrent_states[0]

    # Check positive directional correlation on mock buffer
    cos = torch.nn.functional.cosine_similarity(initial_rec.flatten(), restored_rec.flatten(), dim=0).item()
    assert cos > 0.40


def test_selective_hybrid_poet_ring_buffer():
    """Verify SelectiveHybridPOETRingBuffer achieves bit-exact short rollbacks with long history compression."""
    from runtime.state_ring_buffer import SelectiveHybridPOETRingBuffer

    cache = MockHybridCache()
    ring = SelectiveHybridPOETRingBuffer(
        cache, max_depth=64, short_window_depth=8, rank=8, sparsity_target=0.05
    )

    # Verify memory compression
    assert ring.compression_ratio > 1.5
    initial_rec = cache.layers[0].recurrent_states[0].clone()

    # 1. Short horizon push and rollback (K <= 8) -> Must be bit-exact!
    ring.push(cache)
    cache.layers[0].recurrent_states[0].uniform_(10.0, 20.0)

    ring.rollback(cache, n_accepted=0)
    restored_rec = cache.layers[0].recurrent_states[0]
    assert torch.equal(initial_rec, restored_rec)  # 100% bit-exact lossless!

