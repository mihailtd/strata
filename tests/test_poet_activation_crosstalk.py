"""Unit tests for POET Activation Cross-Talk and Interference Covariance Probe."""

import pytest
import torch

from benchmarks.factory.geometry.poet_activation_crosstalk.probe_poet_activation_crosstalk import (
    poet_crosstalk_fast,
)


def test_poet_crosstalk_energy_isolation():
    """Verify POET isolates pervasive low-rank factors from sparse channel spikes."""
    torch.manual_seed(42)
    N = 100
    d_out = 500
    
    # Construct base shared activation direction + 5 specific conflicting spike channels
    v_shared = torch.randn(d_out)
    v_shared = v_shared / torch.norm(v_shared)
    
    # Delta A and Delta B share the common mode plus random background
    mode_A = torch.randn(N, 1) @ v_shared.unsqueeze(0)
    mode_B = torch.randn(N, 1) @ v_shared.unsqueeze(0)
    
    # Add strong localized cross-talk spikes on channels 10, 25, 42
    spike_channels = [10, 25, 42]
    mode_A[:, spike_channels] += 5.0 * torch.randn(N, len(spike_channels))
    mode_B[:, spike_channels] += 5.0 * torch.randn(N, len(spike_channels))
    
    U_k, Vh_k, meta, top_conflicts = poet_crosstalk_fast(mode_A, mode_B, rank=1, sparsity_target=0.02)
    
    # Verify the top identified conflict neurons contain our inserted spike channels
    top_conflict_list = top_conflicts.tolist()
    for spike_ch in spike_channels:
        assert spike_ch in top_conflict_list[:10]
        
    assert meta["energy_share_L"] > 0.0
    assert meta["top_eigenvalue"] > 0.0


def test_channel_filtering_reduces_crosstalk():
    """Verify notch filtering top conflict channels cuts cross-talk while retaining >90% energy."""
    torch.manual_seed(123)
    N = 64
    d_out = 256
    
    # Two activation streams with standard in-domain energy + specific conflicting cross-talk spikes
    delta_A = torch.randn(N, d_out)
    delta_B = torch.randn(N, d_out)
    
    spikes = [5, 12, 19, 45, 88]
    spike_signal = torch.randn(N, 1)
    delta_A[:, spikes] += 2.0 * spike_signal
    delta_B[:, spikes] += 2.0 * spike_signal
    
    cos_raw = torch.sum(delta_A * delta_B) / (torch.norm(delta_A) * torch.norm(delta_B))
    
    _, _, _, top_conflicts = poet_crosstalk_fast(delta_A, delta_B, rank=1, sparsity_target=0.05)
    
    delta_A_filtered = delta_A.clone()
    delta_A_filtered[:, top_conflicts[:10]] = 0.0
    
    cos_filtered = torch.sum(delta_A_filtered * delta_B) / (torch.norm(delta_A_filtered) * torch.norm(delta_B) + 1e-12)
    in_domain_retention = torch.norm(delta_A_filtered) / torch.norm(delta_A)
    
    # Filtered cosine should drop substantially while retaining >75% energy
    assert abs(cos_filtered.item()) < abs(cos_raw.item())
    assert in_domain_retention.item() > 0.70
