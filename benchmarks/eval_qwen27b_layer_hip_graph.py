"""Real-World Empirical Evaluation: Qwen 3.8-27B Layers with ROCm HIP Graph Capture.

Tests real Layer 0 (SSM) and 3-SSM chunk (Layers 0, 1, 2) weights extracted from GGUF,
running 256 decode steps comparing Eager forward pass vs Replayed HIP Graph forward pass
with in-place state mutation on AMD Radeon RX 7900 XTX.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.runtime.native_27b_engine import (
    PreallocatedKVCache,
    Qwen35FullAttentionBlock,
    Qwen35SSMBlock,
    apply_rotary_emb,
)


def evaluate_single_ssm_layer_hip_graph(device: torch.device, num_steps: int = 256) -> Dict[str, Any]:
    print("\n--- Evaluating Layer 0: Single Gated DeltaNet SSM Block under HIP Graph ---")
    weights_path = Path("models/qwen3.8-27b-triton/layer_0.pt")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {weights_path}")

    layer = Qwen35SSMBlock(layer_idx=0, device=device)
    layer_dict = torch.load(weights_path, map_location=device, weights_only=False)
    layer.load_weights(layer_dict)

    # Static IO buffers for graph
    static_x = torch.zeros((1, 5120), dtype=torch.bfloat16, device=device)
    static_ssm = torch.zeros((16, 128, 128), dtype=torch.bfloat16, device=device)
    static_conv = torch.zeros((10240, 3), dtype=torch.bfloat16, device=device)
    static_out = torch.zeros((1, 5120), dtype=torch.bfloat16, device=device)

    def forward_step(x: torch.Tensor) -> torch.Tensor:
        out, new_ssm, new_conv = layer(x, ssm_state=static_ssm, conv_state=static_conv)
        static_out.copy_(out)
        static_ssm.copy_(new_ssm)
        static_conv.copy_(new_conv)
        return static_out

    # Warmup on side stream
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device=device))
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            forward_step(static_x)
    torch.cuda.current_stream(device=device).wait_stream(capture_stream)

    static_ssm.zero_()
    static_conv.zero_()

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        forward_step(static_x)

    # Run comparative decode steps
    torch.manual_seed(2026)
    simulated_inputs = torch.randn(num_steps, 1, 5120, dtype=torch.bfloat16, device=device)

    # Replay loop
    t0_graph = time.perf_counter()
    for step in range(num_steps):
        static_x.copy_(simulated_inputs[step])
        graph.replay()
    torch.cuda.synchronize(device)
    time_graph_ms = (time.perf_counter() - t0_graph) * 1000.0

    # Eager comparison with separate state
    e_layer = Qwen35SSMBlock(layer_idx=0, device=device)
    e_layer.load_weights(layer_dict)
    e_ssm = None
    e_conv = None
    e_out = None

    t0_eager = time.perf_counter()
    for step in range(num_steps):
        e_out, e_ssm, e_conv = e_layer(simulated_inputs[step], ssm_state=e_ssm, conv_state=e_conv)
    torch.cuda.synchronize(device)
    time_eager_ms = (time.perf_counter() - t0_eager) * 1000.0

    # Check accuracy on final output
    sim = F.cosine_similarity(e_out.view(-1).float(), static_out.view(-1).float(), dim=0).item()
    max_err = torch.max(torch.abs(e_out - static_out)).item()

    print(f"  SSM Eager Total: {time_eager_ms:.2f} ms ({time_eager_ms/num_steps:.3f} ms/step)")
    print(f"  SSM Graph Total: {time_graph_ms:.2f} ms ({time_graph_ms/num_steps:.3f} ms/step)")
    print(f"  SSM Cosine Sim:  {sim:.6f} | Max Error: {max_err:.6f}")

    return {
        "test": "single_ssm_layer",
        "num_steps": num_steps,
        "eager_ms_per_step": round(time_eager_ms / num_steps, 3),
        "graph_ms_per_step": round(time_graph_ms / num_steps, 3),
        "speedup": round(time_eager_ms / max(1e-5, time_graph_ms), 2),
        "cosine_similarity": round(sim, 6),
        "max_abs_error": round(max_err, 6),
        "passed": (sim >= 0.9999 and not torch.isnan(static_out).any()),
    }


def evaluate_3ssm_chunk_hip_graph(device: torch.device, num_steps: int = 128) -> Dict[str, Any]:
    print("\n--- Evaluating 3-SSM Chunk (Layers 0, 1, 2) under HIP Graph ---")
    layers = [Qwen35SSMBlock(i, device=device) for i in range(3)]
    for i, l in enumerate(layers):
        d = torch.load(f"models/qwen3.8-27b-triton/layer_{i}.pt", map_location=device, weights_only=False)
        l.load_weights(d)

    static_x = torch.zeros((1, 5120), dtype=torch.bfloat16, device=device)
    static_ssms = [torch.zeros((16, 128, 128), dtype=torch.bfloat16, device=device) for _ in range(3)]
    static_convs = [torch.zeros((10240, 3), dtype=torch.bfloat16, device=device) for _ in range(3)]
    static_out = torch.zeros((1, 5120), dtype=torch.bfloat16, device=device)

    def forward_3ssm(x: torch.Tensor) -> torch.Tensor:
        curr = x
        for i, l in enumerate(layers):
            out, new_ssm, new_conv = l(curr, ssm_state=static_ssms[i], conv_state=static_convs[i])
            static_ssms[i].copy_(new_ssm)
            static_convs[i].copy_(new_conv)
            curr = out
        static_out.copy_(curr)
        return static_out

    # Warmup
    capture_stream = torch.cuda.Stream(device=device)
    capture_stream.wait_stream(torch.cuda.current_stream(device=device))
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            forward_3ssm(static_x)
    torch.cuda.current_stream(device=device).wait_stream(capture_stream)

    for i in range(3):
        static_ssms[i].zero_()
        static_convs[i].zero_()

    # Capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        forward_3ssm(static_x)

    torch.manual_seed(2026)
    simulated_inputs = torch.randn(num_steps, 1, 5120, dtype=torch.bfloat16, device=device)

    # Replay loop
    t0_graph = time.perf_counter()
    for step in range(num_steps):
        static_x.copy_(simulated_inputs[step])
        graph.replay()
    torch.cuda.synchronize(device)
    time_graph_ms = (time.perf_counter() - t0_graph) * 1000.0

    # Eager comparison
    e_layers = [Qwen35SSMBlock(i, device=device) for i in range(3)]
    for i, l in enumerate(e_layers):
        d = torch.load(f"models/qwen3.8-27b-triton/layer_{i}.pt", map_location=device, weights_only=False)
        l.load_weights(d)
    e_ssms = [None, None, None]
    e_convs = [None, None, None]
    e_out = None

    t0_eager = time.perf_counter()
    for step in range(num_steps):
        curr = simulated_inputs[step]
        for i, l in enumerate(e_layers):
            curr, e_ssms[i], e_convs[i] = l(curr, ssm_state=e_ssms[i], conv_state=e_convs[i])
        e_out = curr
    torch.cuda.synchronize(device)
    time_eager_ms = (time.perf_counter() - t0_eager) * 1000.0

    sim = F.cosine_similarity(e_out.view(-1).float(), static_out.view(-1).float(), dim=0).item()
    max_err = torch.max(torch.abs(e_out - static_out)).item()

    print(f"  3-SSM Eager Total: {time_eager_ms:.2f} ms ({time_eager_ms/num_steps:.3f} ms/step)")
    print(f"  3-SSM Graph Total: {time_graph_ms:.2f} ms ({time_graph_ms/num_steps:.3f} ms/step)")
    print(f"  3-SSM Cosine Sim:  {sim:.6f} | Max Error: {max_err:.6f}")

    return {
        "test": "3ssm_chunk",
        "num_steps": num_steps,
        "eager_ms_per_step": round(time_eager_ms / num_steps, 3),
        "graph_ms_per_step": round(time_graph_ms / num_steps, 3),
        "speedup": round(time_eager_ms / max(1e-5, time_graph_ms), 2),
        "cosine_similarity": round(sim, 6),
        "max_abs_error": round(max_err, 6),
        "passed": (sim >= 0.9999 and not torch.isnan(static_out).any()),
    }


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("      REAL-WORLD EMPIRICAL EVALUATION: QWEN 3.8-27B LAYERS WITH HIP GRAPH    ")
    print("=" * 80)
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    single_ssm = evaluate_single_ssm_layer_hip_graph(device, num_steps=128)
    chunk_ssm = evaluate_3ssm_chunk_hip_graph(device, num_steps=128)

    all_passed = single_ssm["passed"] and chunk_ssm["passed"]

    print("\n================================================================================")
    print(f"Empirical Test Status: {'>>> PASSED <<<' if all_passed else '>>> FAILED <<<'}")
    print("================================================================================")

    out_dir = Path("results/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "layer_hip_graph_eval.json"
    with open(out_file, "w") as f:
        json.dump({"single_ssm": single_ssm, "3ssm_chunk": chunk_ssm, "passed": all_passed}, f, indent=2)
    print(f"Saved empirical evaluation results to: {out_file}")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
