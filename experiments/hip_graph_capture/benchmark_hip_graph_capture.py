"""Synthetic Benchmark: ROCm HIP Graph Capture vs Eager Python Kernel Launches.

Measures the impact of eliminating CPU host dispatch overhead across multi-layer
W4A16 Triton execution chains (1, 8, 16, and 64 layers) on AMD Radeon RX 7900 XTX.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.native_27b_engine import RMSNorm
from runtime.triton_w4a16 import quantize_and_pack_w4, w4a16_matmul


class SyntheticTritonLayer(nn.Module):
    """Represents a full multi-operator layer with RMSNorm, W4A16 GEMV, and SwiGLU."""

    def __init__(self, dim: int = 5120, intermediate_dim: int = 17408, device: torch.device = torch.device("cuda:0")):
        super().__init__()
        self.device = device
        self.norm1 = RMSNorm(dim, device=device)
        self.norm2 = RMSNorm(dim, device=device)

        # Create packed W4A16 linear weights
        w_gate = torch.randn(dim, intermediate_dim, dtype=torch.bfloat16, device=device)
        w_up = torch.randn(dim, intermediate_dim, dtype=torch.bfloat16, device=device)
        w_down = torch.randn(intermediate_dim, dim, dtype=torch.bfloat16, device=device)

        self.qw_gate, self.scales_gate = quantize_and_pack_w4(w_gate)
        self.qw_up, self.scales_up = quantize_and_pack_w4(w_up)
        self.qw_down, self.scales_down = quantize_and_pack_w4(w_down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. First block: Norm + Proj
        h = self.norm1(x)
        gate = w4a16_matmul(h, self.qw_gate, self.scales_gate)
        up = w4a16_matmul(h, self.qw_up, self.scales_up)
        act = F.silu(gate) * up

        # 2. Down proj + Residual
        y = w4a16_matmul(act, self.qw_down, self.scales_down)
        x = x + y

        # 3. Post Norm + Residual
        x = x + self.norm2(x)
        return x


def benchmark_layer_chain(
    num_layers: int,
    num_iterations: int = 50,
    device: torch.device = torch.device("cuda:0"),
) -> Dict[str, Any]:
    print(f"\nEvaluating Chain Depth: {num_layers} Layers ...")

    # Build layers
    layers = [SyntheticTritonLayer(dim=5120, intermediate_dim=17408, device=device) for _ in range(num_layers)]

    # Static IO buffers
    static_in = torch.randn(1, 5120, dtype=torch.bfloat16, device=device)
    static_out = torch.zeros(1, 5120, dtype=torch.bfloat16, device=device)

    def execute_chain(x: torch.Tensor) -> torch.Tensor:
        curr = x
        for layer in layers:
            curr = layer(curr)
        return curr

    # --- A. Eager Warmup ---
    for _ in range(3):
        _ = execute_chain(static_in)
    torch.cuda.synchronize(device)

    # --- B. Eager Benchmark ---
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    t0_wall = time.perf_counter()
    start_event.record()
    for _ in range(num_iterations):
        res_eager = execute_chain(static_in)
    end_event.record()
    torch.cuda.synchronize(device)
    eager_wall_ms = (time.perf_counter() - t0_wall) * 1000.0
    eager_gpu_ms = start_event.elapsed_time(end_event)

    eager_per_step_ms = eager_gpu_ms / num_iterations
    kernels_per_step = num_layers * 6  # ~6 ops per layer
    eager_launch_overhead_us = (eager_wall_ms - eager_gpu_ms) / (num_iterations * kernels_per_step) * 1000.0

    # --- C. Graph Capture ---
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device=device))

    # Warmup on capture stream
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            out = execute_chain(static_in)
            static_out.copy_(out)
    torch.cuda.current_stream(device=device).wait_stream(capture_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        out = execute_chain(static_in)
        static_out.copy_(out)

    # --- D. Graph Replay Benchmark ---
    t0_wall = time.perf_counter()
    start_event.record()
    for _ in range(num_iterations):
        graph.replay()
    end_event.record()
    torch.cuda.synchronize(device)
    graph_wall_ms = (time.perf_counter() - t0_wall) * 1000.0
    graph_gpu_ms = start_event.elapsed_time(end_event)

    graph_per_step_ms = graph_gpu_ms / num_iterations
    speedup = eager_per_step_ms / max(1e-5, graph_per_step_ms)

    # --- E. Correctness Verification ---
    sim = F.cosine_similarity(res_eager.view(-1).float(), static_out.view(-1).float(), dim=0).item()
    max_diff = torch.max(torch.abs(res_eager - static_out)).item()

    print(f"  -> Eager: {eager_per_step_ms:.2f} ms/step (Wall: {eager_wall_ms/num_iterations:.2f} ms)")
    print(f"  -> Graph: {graph_per_step_ms:.2f} ms/step (Wall: {graph_wall_ms/num_iterations:.2f} ms)")
    print(f"  -> Speedup: {speedup:.2f}x | Cosine Sim: {sim:.6f} | Max Diff: {max_diff:.6f}")

    return {
        "num_layers": num_layers,
        "kernels_per_step": kernels_per_step,
        "eager_per_step_ms": round(eager_per_step_ms, 3),
        "eager_wall_per_step_ms": round(eager_wall_ms / num_iterations, 3),
        "graph_per_step_ms": round(graph_per_step_ms, 3),
        "graph_wall_per_step_ms": round(graph_wall_ms / num_iterations, 3),
        "speedup_factor": round(speedup, 2),
        "cosine_similarity": round(sim, 6),
        "max_diff": round(max_diff, 6),
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("      SYNTHETIC BENCHMARK: ROCm HIP GRAPH CAPTURE VS EAGER DISPATCH         ")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    layer_configs = [1, 8, 16]
    results = []

    for depth in layer_configs:
        res = benchmark_layer_chain(num_layers=depth, num_iterations=40, device=device)
        results.append(res)

    print("\n" + "=" * 80)
    print("                          SYNTHETIC BENCHMARK SUMMARY                          ")
    print("=" * 80)
    print(f"{'Layers':<8} | {'Total Kernels':<14} | {'Eager Step (ms)':<16} | {'Graph Step (ms)':<16} | {'Speedup':<10} | {'Cosine Sim':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['num_layers']:<8} | {r['kernels_per_step']:<14} | {r['eager_per_step_ms']:<16.2f} | {r['graph_per_step_ms']:<16.2f} | {r['speedup_factor']:<10.2f}x | {r['cosine_similarity']:<10.6f}")

    out_dir = Path("results/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "hip_graph_capture_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved synthetic benchmark metrics to: {out_file}")


if __name__ == "__main__":
    main()
