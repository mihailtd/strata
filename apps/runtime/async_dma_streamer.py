"""Asynchronous Double-Buffered Ping-Pong Layer Streaming Engine over PCIe 4.0 DMA.

Enables running models larger than GPU VRAM (e.g. 70B / 72B in W4A16, ~36-41 GB)
by overlapping background host-to-device PCIe DMA transfers with GPU matrix execution:
- Host memory holds all L layers in pinned DDR5 host RAM (pin_memory=True).
- VRAM holds only 2 layer staging slots (Buffer 0 and Buffer 1) ~900 MB total VRAM.
- Dedicated hardware streams (compute_stream and dma_stream) synchronized via HIP events.
- Layer N+1 streams over PCIe DMA while Layer N executes on RDNA3 WMMA hardware cores.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


class PinnedHostLayerStore:
    """Manages pinned host system RAM storage for full multi-layer model weights."""

    def __init__(self, num_layers: int, layer_size_bytes: int):
        self.num_layers = num_layers
        self.layer_size_bytes = layer_size_bytes
        self.host_layers: list[torch.Tensor] = []

        # Allocate pinned buffers in host RAM
        for _ in range(num_layers):
            t = torch.empty(layer_size_bytes, dtype=torch.uint8, pin_memory=True)
            self.host_layers.append(t)

    @property
    def total_host_mb(self) -> float:
        return (self.num_layers * self.layer_size_bytes) / (1024 * 1024)

    def set_layer_bytes(self, layer_idx: int, data: torch.Tensor) -> None:
        assert 0 <= layer_idx < self.num_layers
        view = data.view(torch.uint8).flatten()
        assert view.numel() == self.layer_size_bytes
        self.host_layers[layer_idx].copy_(view)

    def get_layer_pinned(self, layer_idx: int) -> torch.Tensor:
        return self.host_layers[layer_idx]


class DoubleBufferedVRAMStaging:
    """Manages two pre-allocated VRAM staging slots (Buffer 0 and Buffer 1)."""

    def __init__(self, layer_size_bytes: int, device: str = "cuda:0"):
        self.layer_size_bytes = layer_size_bytes
        self.device = device
        self.buffers: list[torch.Tensor] = [
            torch.empty(layer_size_bytes, dtype=torch.uint8, device=device),
            torch.empty(layer_size_bytes, dtype=torch.uint8, device=device),
        ]

    @property
    def total_vram_mb(self) -> float:
        return (2 * self.layer_size_bytes) / (1024 * 1024)

    def get_slot(self, slot_idx: int) -> torch.Tensor:
        return self.buffers[slot_idx % 2]


class AsyncPingPongLayerStreamer:
    """Orchestrates asynchronous layer streaming over PCIe 4.0 DMA."""

    def __init__(
        self,
        num_layers: int,
        layer_size_bytes: int,
        device: str = "cuda:0",
    ):
        self.num_layers = num_layers
        self.layer_size_bytes = layer_size_bytes
        self.device = device

        self.host_store = PinnedHostLayerStore(num_layers, layer_size_bytes)
        self.vram_staging = DoubleBufferedVRAMStaging(layer_size_bytes, device=device)

        self.compute_stream = torch.cuda.Stream(device=device)
        self.dma_stream = torch.cuda.Stream(device=device)

        self.dma_ready_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_layers)]
        self.compute_done_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_layers)]

    def run_synchronous_forward(
        self,
        hidden_states: torch.Tensor,
        layer_compute_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Baseline single-buffer synchronous execution without overlap."""
        buf = self.vram_staging.get_slot(0)
        for i in range(self.num_layers):
            buf.copy_(self.host_store.get_layer_pinned(i), non_blocking=False)
            hidden_states = layer_compute_fn(hidden_states, buf)
        torch.cuda.synchronize()
        return hidden_states

    def run_pipelined_forward(
        self,
        hidden_states: torch.Tensor,
        layer_compute_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Asynchronous double-buffered execution overlapping DMA transfer of layer N+1 with compute of layer N."""
        # 1. Pre-fetch layer 0 into Slot 0
        with torch.cuda.stream(self.dma_stream):
            self.vram_staging.get_slot(0).copy_(self.host_store.get_layer_pinned(0), non_blocking=True)
            self.dma_ready_events[0].record(self.dma_stream)

        # 2. Ping-pong execution loop
        for i in range(self.num_layers):
            curr_slot = i % 2
            next_slot = (i + 1) % 2

            # Issue background DMA transfer of the NEXT layer on dma_stream
            if i + 1 < self.num_layers:
                # Ensure the previous compute on next_slot has finished before overwriting it
                if i > 0:
                    self.dma_stream.wait_event(self.compute_done_events[i - 1])

                with torch.cuda.stream(self.dma_stream):
                    self.vram_staging.get_slot(next_slot).copy_(
                        self.host_store.get_layer_pinned(i + 1),
                        non_blocking=True,
                    )
                    self.dma_ready_events[i + 1].record(self.dma_stream)

            # Ensure the current layer has arrived in VRAM before computing
            self.compute_stream.wait_event(self.dma_ready_events[i])

            # Execute compute on the current layer
            with torch.cuda.stream(self.compute_stream):
                hidden_states = layer_compute_fn(hidden_states, self.vram_staging.get_slot(curr_slot))
                self.compute_done_events[i].record(self.compute_stream)

        torch.cuda.synchronize()
        return hidden_states
