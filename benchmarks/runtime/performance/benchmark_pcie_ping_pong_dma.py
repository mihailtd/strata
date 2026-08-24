"""Empirical Benchmark: Asynchronous Double-Buffered Ping-Pong PCIe 4.0 DMA Streaming for 70B/72B Models.

Evaluates execution of an 80-layer 70B/72B model in W4A16 (~36.2 GB total weights)
where weights are stored in Pinned DDR5 Host RAM and streamed through two 452 MB VRAM staging slots:
- Arm A (Baseline): Synchronous single-buffer layer-by-layer transfer + compute.
- Arm B (Innovation): Asynchronous double-buffered ping-pong PCIe DMA streaming with HIP event synchronization.

Sweeps token batch / speculative verification regimes: M in [1, 2, 4, 8, 16, 32, 64].
Audit-compliant methodology: Warmup, multiple repeats, median + IQR, persisted JSON telemetry.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from runtime.async_dma_streamer import AsyncPingPongLayerStreamer
from runtime.canon import CANON, REPO_ROOT
from runtime.triton_w4a16 import quantize_and_pack_w4, w4a16_matmul


def compute_median_iqr(values: list[float]) -> tuple[float, float, float]:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    median = sorted_vals[n // 2]
    q1 = sorted_vals[max(0, n // 4)]
    q3 = sorted_vals[min(n - 1, 3 * n // 4)]
    return median, q1, q3


def run_pcie_streaming_benchmark(
    num_benchmark_layers: int = 8,
    full_model_layers: int = 80,
    m_values: list[int] | None = None,
    n_repeats: int = 5,
    device: str = "cuda:0",
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm / CUDA GPU is required for PCIe streaming benchmark")

    # Unload server to ensure 100% free VRAM
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:8000/api/engine/unload", data=b"{}", headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30)
    except Exception:
        pass
    torch.cuda.empty_cache()

    if m_values is None:
        m_values = [1, 2, 4, 8, 16, 32, 64]

    torch.manual_seed(42)
    device_name = torch.cuda.get_device_name(0)

    # 70B Layer Dimensions
    D = 8192
    H = 29568
    group_size = 128

    print("=" * 110)
    print(f" ASYNCHRONOUS PING-PONG PCIe DMA STREAMING BENCHMARK ({device_name})")
    print(f" Target Model: 70B/72B (80 Layers in W4A16, ~36.2 GB Host RAM, 905 MB VRAM Staging)")
    print("=" * 110)

    # 1. Synthesize representative 70B layer weights one by one to avoid peak memory spike
    print(f"\n[Setup] Synthesizing 70B layer weights in W4A16 (D={D}, H={H})...")
    w = torch.randn((D, D + 1024 * 2), dtype=torch.bfloat16, device=device)
    qw_qkv, s_qkv = quantize_and_pack_w4(w, group_size=group_size)
    del w
    torch.cuda.empty_cache()

    w = torch.randn((D, D), dtype=torch.bfloat16, device=device)
    qw_o, s_o = quantize_and_pack_w4(w, group_size=group_size)
    del w
    torch.cuda.empty_cache()

    w = torch.randn((D, H * 2), dtype=torch.bfloat16, device=device)
    qw_gate_up, s_gate_up = quantize_and_pack_w4(w, group_size=group_size)
    del w
    torch.cuda.empty_cache()

    w = torch.randn((H, D), dtype=torch.bfloat16, device=device)
    qw_down, s_down = quantize_and_pack_w4(w, group_size=group_size)
    del w
    torch.cuda.empty_cache()

    # Calculate exact layer size in bytes
    layer_bytes = (
        (qw_qkv.numel() + qw_o.numel() + qw_gate_up.numel() + qw_down.numel()) * 4 +
        (s_qkv.numel() + s_o.numel() + s_gate_up.numel() + s_down.numel()) * 2
    )
    layer_mb = layer_bytes / (1024 * 1024)
    total_model_gb = (full_model_layers * layer_bytes) / (1024 ** 3)
    vram_staging_mb = (2 * layer_bytes) / (1024 * 1024)

    print(f"  Layer Size: {layer_mb:.2f} MB (Packed INT4 + Scales)")
    print(f"  Full 80-Layer Model in Host RAM: {total_model_gb:.2f} GB")
    print(f"  Double-Buffered VRAM Staging Footprint: {vram_staging_mb:.2f} MB")

    # Allocate streamer
    streamer = AsyncPingPongLayerStreamer(num_benchmark_layers, layer_bytes, device=device)

    # Define actual 70B layer matrix computation function
    def layer_compute_fn(x: torch.Tensor, weight_buf: torch.Tensor) -> torch.Tensor:
        qkv = w4a16_matmul(x, qw_qkv, s_qkv, group_size=group_size)
        attn_out = w4a16_matmul(qkv[:, :D].contiguous(), qw_o, s_o, group_size=group_size)
        x_norm = x + attn_out
        mlp_in = w4a16_matmul(x_norm, qw_gate_up, s_gate_up, group_size=group_size)
        mlp_act = mlp_in[:, :H].contiguous()
        mlp_out = w4a16_matmul(mlp_act, qw_down, s_down, group_size=group_size)
        return x_norm + mlp_out

    # Measure pure DMA transfer time
    h_buf = streamer.host_store.get_layer_pinned(0)
    d_buf = streamer.vram_staging.get_slot(0)
    for _ in range(5):
        d_buf.copy_(h_buf, non_blocking=True)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(20):
        d_buf.copy_(h_buf, non_blocking=True)
    torch.cuda.synchronize()
    pure_dma_ms = ((time.perf_counter() - t0) / 20) * 1000.0
    dma_bw_gbs = (layer_bytes / 1e9) / (pure_dma_ms / 1000.0)

    print(f"\n[PCIe Hardware Calibration] DMA Transfer per Layer: {pure_dma_ms:.2f} ms ({dma_bw_gbs:.2f} GB/s)")

    results_table = []

    for M in m_values:
        print(f"\n[Evaluating M={M:2d}]...")
        x_input = torch.randn((M, D), dtype=torch.bfloat16, device=device)

        # Pure compute time per layer
        for _ in range(5):
            _ = layer_compute_fn(x_input, d_buf)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(20):
            _ = layer_compute_fn(x_input, d_buf)
        torch.cuda.synchronize()
        pure_compute_ms = ((time.perf_counter() - t0) / 20) * 1000.0

        # Alternating timing loops for Sync vs Pipelined
        sync_times: list[float] = []
        pipe_times: list[float] = []

        for rep in range(n_repeats):
            if rep % 2 == 0:
                # Sync first
                t0 = time.perf_counter()
                _ = streamer.run_synchronous_forward(x_input.clone(), layer_compute_fn)
                torch.cuda.synchronize()
                sync_times.append(((time.perf_counter() - t0) * 1000.0) / num_benchmark_layers)

                t0 = time.perf_counter()
                _ = streamer.run_pipelined_forward(x_input.clone(), layer_compute_fn)
                torch.cuda.synchronize()
                pipe_times.append(((time.perf_counter() - t0) * 1000.0) / num_benchmark_layers)
            else:
                # Pipelined first
                t0 = time.perf_counter()
                _ = streamer.run_pipelined_forward(x_input.clone(), layer_compute_fn)
                torch.cuda.synchronize()
                pipe_times.append(((time.perf_counter() - t0) * 1000.0) / num_benchmark_layers)

                t0 = time.perf_counter()
                _ = streamer.run_synchronous_forward(x_input.clone(), layer_compute_fn)
                torch.cuda.synchronize()
                sync_times.append(((time.perf_counter() - t0) * 1000.0) / num_benchmark_layers)

        sync_med, sync_q1, sync_q3 = compute_median_iqr(sync_times)
        pipe_med, pipe_q1, pipe_q3 = compute_median_iqr(pipe_times)

        # Full 80-layer projections
        sync_80_ms = sync_med * full_model_layers
        pipe_80_ms = pipe_med * full_model_layers

        # Throughput
        sync_tok_s = (1000.0 / sync_80_ms) * M
        pipe_tok_s = (1000.0 / pipe_80_ms) * M

        speedup = sync_med / max(1e-5, pipe_med)

        # Overlap efficiency: how much of the hidden transfer/compute was masked
        theoretical_unoverlapped = pure_dma_ms + pure_compute_ms
        overlap_masked_ms = max(0.0, theoretical_unoverlapped - pipe_med)
        overlap_efficiency_pct = (overlap_masked_ms / max(1e-5, min(pure_dma_ms, pure_compute_ms))) * 100.0
        overlap_efficiency_pct = min(100.0, overlap_efficiency_pct)

        row = {
            "M": M,
            "layer_mb": round(layer_mb, 2),
            "pure_dma_ms": round(pure_dma_ms, 2),
            "pure_compute_ms": round(pure_compute_ms, 2),
            "sync_layer_ms": round(sync_med, 2),
            "pipe_layer_ms": round(pipe_med, 2),
            "sync_80_layers_ms": round(sync_80_ms, 2),
            "pipe_80_layers_ms": round(pipe_80_ms, 2),
            "sync_tok_per_sec": round(sync_tok_s, 2),
            "pipe_tok_per_sec": round(pipe_tok_s, 2),
            "speedup": round(speedup, 2),
            "overlap_efficiency_pct": round(overlap_efficiency_pct, 1),
            "iqr_sync": [round(sync_q1, 2), round(sync_q3, 2)],
            "iqr_pipe": [round(pipe_q1, 2), round(pipe_q3, 2)],
        }
        results_table.append(row)

        print(
            f"  M={M:2d} | DMA: {pure_dma_ms:5.2f}ms, Compute: {pure_compute_ms:5.2f}ms | "
            f"Sync: {sync_med:5.2f}ms ({sync_tok_s:4.2f} tok/s) -> Pipelined: {pipe_med:5.2f}ms ({pipe_tok_s:4.2f} tok/s) | "
            f"Speedup: {speedup:4.2f}x (Overlap Efficiency: {overlap_efficiency_pct:5.1f}%)"
        )

    out_payload = {
        "benchmark": "Asynchronous Double-Buffered Ping-Pong PCIe 4.0 DMA Streaming",
        "target_model": "70B/72B (80 Layers)",
        "hardware": {
            "device": device_name,
            "target_arch": "gfx1100",
            "dma_bandwidth_gbs": round(dma_bw_gbs, 2),
        },
        "canon": CANON.stamp(),
        "config": {
            "layer_size_mb": round(layer_mb, 2),
            "total_model_gb": round(total_model_gb, 2),
            "vram_staging_mb": round(vram_staging_mb, 2),
            "num_benchmark_layers": num_benchmark_layers,
            "full_model_layers": full_model_layers,
            "n_repeats": n_repeats,
        },
        "results": results_table,
    }

    out_file = REPO_ROOT / "results/benchmarks/pcie_ping_pong_dma_perf.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(out_payload, f, indent=2)

    print("\n" + "=" * 110)
    print(f"[Persisted] Benchmark telemetry saved to {out_file}")
    print("=" * 110)
    return out_payload


def main():
    parser = argparse.ArgumentParser(description="PCIe Ping-Pong DMA Streaming Benchmark")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of benchmark layers")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repeats per cell")
    args = parser.parse_args()

    run_pcie_streaming_benchmark(num_benchmark_layers=args.num_layers, n_repeats=args.repeats)


if __name__ == "__main__":
    main()
