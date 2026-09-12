"""Unit tests for Asynchronous Double-Buffered Ping-Pong PCIe DMA Streamer."""

from __future__ import annotations

import pytest
import torch
from runtime.async_dma_streamer import (
    AsyncPingPongLayerStreamer,
    DoubleBufferedVRAMStaging,
    PinnedHostLayerStore,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires ROCm / CUDA GPU")
def test_pinned_store_and_vram_staging_allocation():
    """Verify pinned host allocation and double-buffered VRAM staging."""
    num_layers = 8
    layer_mb = 64
    layer_bytes = layer_mb * 1024 * 1024

    store = PinnedHostLayerStore(num_layers, layer_bytes)
    staging = DoubleBufferedVRAMStaging(layer_bytes, device="cuda:0")

    assert len(store.host_layers) == num_layers
    assert store.total_host_mb == num_layers * layer_mb
    assert store.host_layers[0].is_pinned()

    assert len(staging.buffers) == 2
    assert staging.total_vram_mb == 2 * layer_mb
    assert staging.buffers[0].is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires ROCm / CUDA GPU")
def test_pipelined_vs_synchronous_numerical_equivalence():
    """Verify that pipelined ping-pong streaming produces identical outputs to synchronous execution."""
    torch.manual_seed(42)
    device = "cuda:0"
    num_layers = 6
    dim = 256
    layer_bytes = dim * dim * 2  # bfloat16 matrix

    streamer = AsyncPingPongLayerStreamer(num_layers, layer_bytes, device=device)

    # Initialize random layer weights in pinned host RAM
    for i in range(num_layers):
        w_host = torch.randn((dim, dim), dtype=torch.bfloat16)
        streamer.host_store.set_layer_bytes(i, w_host)

    # Define a simple linear layer computation
    def compute_fn(x: torch.Tensor, weight_buf: torch.Tensor) -> torch.Tensor:
        w_mat = weight_buf.view(torch.bfloat16).view(dim, dim)
        return torch.matmul(x, w_mat)

    x_input = torch.randn((8, dim), dtype=torch.bfloat16, device=device)

    # 1. Run Synchronous
    out_sync = streamer.run_synchronous_forward(x_input.clone(), compute_fn)

    # 2. Run Pipelined Ping-Pong
    out_pipe = streamer.run_pipelined_forward(x_input.clone(), compute_fn)

    diff = (out_sync - out_pipe).abs().max().item()
    cos_sim = torch.cosine_similarity(out_sync.flatten().float(), out_pipe.flatten().float(), dim=0).item()

    assert diff == 0.0, f"Pipelined streaming numerical diff {diff} exceeds 0.0"
    assert cos_sim == 1.0, f"Cosine similarity {cos_sim} is not 1.0"
