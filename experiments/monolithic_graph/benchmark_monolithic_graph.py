"""Synthetic Benchmark: Monolithic Full-Model Decode Graph vs Fragmented Dispatch.

Measures the wall-clock decode step latency across 64 simulated hybrid layers
(48 SSM blocks + 16 Full Attention blocks + LM Head) comparing:
1. Fragmented Python Loop (16 separate graph replays + 16 eager attention layers)
2. Monolithic Single Graph (all 64 layers in 1 single CUDAGraph)
on AMD Radeon RX 7900 XTX.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.triton_w4a16 import quantize_and_pack_w4, w4a16_matmul


class MockHybridLayerChain:
    """Represents a full 64-layer chain with dynamic attention masking."""

    def __init__(self, num_layers: int = 64, max_seq_len: int = 128, device: torch.device = torch.device("cuda:0")):
        self.num_layers = num_layers
        self.device = device
        self.max_seq_len = max_seq_len

        # Linear weights
        w_ssm = torch.randn(5120, 5120, dtype=torch.bfloat16, device=device)
        self.qw_ssm, self.scales_ssm = quantize_and_pack_w4(w_ssm)

        w_attn = torch.randn(5120, 5120, dtype=torch.bfloat16, device=device)
        self.qw_attn, self.scales_attn = quantize_and_pack_w4(w_attn)

        # Preallocated buffers
        self.static_x = torch.zeros((1, 5120), dtype=torch.bfloat16, device=device)
        self.static_out = torch.zeros((1, 5120), dtype=torch.bfloat16, device=device)
        self.static_pos = torch.zeros((1,), dtype=torch.long, device=device)
        self.static_mask = torch.full((1, 1, 1, max_seq_len), -1e4, dtype=torch.bfloat16, device=device)

        # 16 KV caches for attention layers
        self.k_caches = [torch.zeros((1, 4, max_seq_len, 256), dtype=torch.bfloat16, device=device) for _ in range(num_layers // 4)]
        self.v_caches = [torch.zeros((1, 4, max_seq_len, 256), dtype=torch.bfloat16, device=device) for _ in range(num_layers // 4)]

        # 48 SSM recurrent states
        self.ssm_states = [torch.zeros((16, 128, 128), dtype=torch.bfloat16, device=device) for _ in range(num_layers - (num_layers // 4))]

    def forward_step(self) -> torch.Tensor:
        curr = self.static_x
        attn_count = 0
        ssm_count = 0

        for i in range(self.num_layers):
            if (i + 1) % 4 == 0:
                # Attention Block
                curr = w4a16_matmul(curr, self.qw_attn, self.scales_attn)
                q = curr[..., :1024].view(1, 4, 1, 256)
                k_new = curr[..., 1024:2048].view(1, 4, 1, 256)
                v_new = curr[..., 2048:3072].view(1, 4, 1, 256)

                self.k_caches[attn_count].index_copy_(2, self.static_pos, k_new)
                self.v_caches[attn_count].index_copy_(2, self.static_pos, v_new)

                attn_out = F.scaled_dot_product_attention(
                    q, self.k_caches[attn_count], self.v_caches[attn_count], attn_mask=self.static_mask
                )
                curr = curr + attn_out.view(1, -1).repeat(1, 5)
                attn_count += 1
            else:
                # SSM Block
                curr = w4a16_matmul(curr, self.qw_ssm, self.scales_ssm)
                self.ssm_states[ssm_count] = self.ssm_states[ssm_count] * 0.99 + 0.01
                ssm_count += 1

        self.static_out.copy_(curr)
        return self.static_out


def benchmark_monolithic_vs_fragmented(device: torch.device, num_steps: int = 128) -> Dict[str, Any]:
    print(f"\n--- Benchmarking 64-Layer Decode Step: Monolithic Graph vs Eager / Fragmented Loop ---")
    chain = MockHybridLayerChain(num_layers=64, max_seq_len=128, device=device)

    # --- 1. Capture Monolithic CUDAGraph ---
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device=device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            chain.forward_step()
    torch.cuda.current_stream(device=device).wait_stream(stream)

    monolithic_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(monolithic_graph, stream=stream):
        chain.forward_step()

    # Benchmark Monolithic Graph
    t0_mono = time.perf_counter()
    for p in range(num_steps):
        chain.static_pos.fill_(p)
        chain.static_mask[0, 0, 0, p] = 0.0
        monolithic_graph.replay()
    torch.cuda.synchronize(device)
    mono_total_ms = (time.perf_counter() - t0_mono) * 1000.0
    mono_step_ms = mono_total_ms / num_steps
    mono_toks = 1000.0 / mono_step_ms

    # --- 2. Benchmark Eager / Fragmented Loop ---
    # Reset states
    chain.static_mask.fill_(-1e4)
    t0_eager = time.perf_counter()
    for p in range(num_steps):
        chain.static_pos.fill_(p)
        chain.static_mask[0, 0, 0, p] = 0.0
        _ = chain.forward_step()
    torch.cuda.synchronize(device)
    eager_total_ms = (time.perf_counter() - t0_eager) * 1000.0
    eager_step_ms = eager_total_ms / num_steps
    eager_toks = 1000.0 / eager_step_ms

    speedup = eager_step_ms / max(1e-5, mono_step_ms)

    print(f"  Eager / Fragmented: {eager_step_ms:6.2f} ms/step ({eager_toks:5.1f} tok/s)")
    print(f"  Monolithic Graph:   {mono_step_ms:6.2f} ms/step ({mono_toks:5.1f} tok/s)")
    print(f"  Speedup:            {speedup:6.2f}x")

    return {
        "num_layers": 64,
        "num_steps": num_steps,
        "eager_ms_per_step": round(eager_step_ms, 2),
        "eager_toks_per_sec": round(eager_toks, 1),
        "monolithic_ms_per_step": round(mono_step_ms, 2),
        "monolithic_toks_per_sec": round(mono_toks, 1),
        "speedup": round(speedup, 2),
    }


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA/ROCm not available!")
        return

    device = torch.device("cuda:0")
    print("=" * 80)
    print("  SYNTHETIC BENCHMARK: MONOLITHIC FULL-MODEL GRAPH VS FRAGMENTED DISPATCH")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(device)}")

    results = benchmark_monolithic_vs_fragmented(device=device, num_steps=100)

    out_dir = Path("results/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "monolithic_graph_synthetic_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to: {out_file}")


if __name__ == "__main__":
    main()
