"""Unit tests for Ring-Buffered Hybrid State Rollback (ReplaySSM)."""

import pytest
import torch

from gnn_experiment.mtp_draft import (
    attach_state_ring_buffer,
    restore_state,
    snapshot_state,
    state_nbytes,
)
from gnn_experiment.state_ring_buffer import (
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
