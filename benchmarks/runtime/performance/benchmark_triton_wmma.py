"""Benchmark RDNA3 Native WMMA Triton Matrix Multiplication Performance.

Runs performance sweeps across:
- Speculative decoding verification chunks (M=2, 4)
- Micro-batches (M=16)
- Prefill GEMMs (M=64, 256, 512, 1024)

Outputs JSON telemetry to results/benchmarks/triton_wmma_perf.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from runtime.gpu_preflight import ensure_gpu_exclusive
from runtime.triton_wmma import (
    benchmark_wmma_vs_pytorch,
    fused_wmma_lora_matmul,
    inspect_kernel_wmma_isa,
    triton_wmma_matmul,
)


def main():
    print("=================================================================")
    print(" RDNA3 Native WMMA Triton Compilation & Performance Benchmark")
    print("=================================================================")

    ensure_gpu_exclusive(exit_on_conflict=False)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[WMMA Bench] Testing on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. ISA Inspection
    print("\n--- 1. AMDGCN ISA Inspection ---")
    isa_info = inspect_kernel_wmma_isa(m=64, k=4096, n=4096, device=device)
    print(f"Target Architecture: {isa_info.get('target_arch')}")
    print(f"Native v_wmma Instructions Emitted: {isa_info.get('v_wmma_count')}")
    print(f"WMMA Emitted Successfully: {isa_info.get('wmma_emitted')}")
    if isa_info.get("sample_instructions"):
        print("Sample emitted instructions:")
        for line in isa_info["sample_instructions"]:
            print(f"  {line}")

    # 2. Performance Sweep
    print("\n--- 2. Performance Sweep vs. PyTorch Reference ---")
    shapes = [
        (2, 4096, 4096),     # MTP speculative chunk K=2
        (4, 4096, 4096),     # MTP speculative chunk K=4
        (16, 4096, 4096),    # Micro-batch (B=16)
        (64, 4096, 4096),    # Prompt prefill chunk
        (256, 4096, 4096),   # Standard prefill GEMM
        (512, 4096, 4096),   # Long-context prefill
        (1024, 4096, 4096),  # Serving batch GEMM
    ]

    results = benchmark_wmma_vs_pytorch(shapes=shapes, warmup=10, iters=50, device=device)

    print("\n┌──────────┬─────────────┬──────────────┬──────────────┬─────────┬───────────────┬──────────────┐")
    print("│ M x K x N│ Regime      │ PyTorch (ms) │ Triton (ms)  │ Speedup │ Triton TFLOPS │ Torch TFLOPS │")
    print("├──────────┼─────────────┼──────────────┼──────────────┼─────────┼───────────────┼──────────────┤")

    regimes = {
        2: "MTP Spec (K=2)",
        4: "MTP Spec (K=4)",
        16: "Micro-batch 16",
        64: "Prefill Chunk ",
        256: "Prefill Medium",
        512: "Prefill Large ",
        1024: "Serving Batch ",
    }

    for r in results:
        reg = regimes.get(r["M"], "Custom GEMM   ")
        shape_str = f"{r['M']:4d}x{r['K']:4d}x{r['N']:4d}"
        print(f"│ {shape_str} │ {reg} │ {r['torch_ms']:12.4f} │ {r['triton_ms']:12.4f} │ {r['speedup']:6.2f}x │ {r['triton_tflops']:13.2f} │ {r['torch_tflops']:12.2f} │")
    print("└──────────┴─────────────┴──────────────┴──────────────┴─────────┴───────────────┴──────────────┘")

    # 3. Fused LoRA Validation
    print("\n--- 3. Fused Dynamic LoRA Branch Validation ---")
    M, K, N, R = 64, 4096, 4096, 32
    x = torch.randn((M, K), device=device, dtype=torch.bfloat16)
    w = torch.randn((K, N), device=device, dtype=torch.bfloat16)
    lora_a = torch.randn((K, R), device=device, dtype=torch.bfloat16)
    lora_b = torch.randn((R, N), device=device, dtype=torch.bfloat16)
    alpha = 16.0 / R

    fused_out = fused_wmma_lora_matmul(x, w, lora_a, lora_b, alpha=alpha)
    ref_out = torch.matmul(x, w) + alpha * torch.matmul(torch.matmul(x, lora_a), lora_b)

    cos_sim = torch.cosine_similarity(fused_out.flatten().float(), ref_out.flatten().float(), dim=0).item()
    max_diff = torch.max(torch.abs(fused_out - ref_out)).item()
    print(f"Fused LoRA Cosine Similarity: {cos_sim:.6f}")
    print(f"Fused LoRA Max Difference: {max_diff:.4f}")
    print(f"Fused LoRA Verification: {'PASSED' if cos_sim > 0.999 else 'FAILED'}")

    # Save to results JSON
    out_dir = REPO_ROOT / "results" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "triton_wmma_perf.json"

    payload = {
        "isa_info": isa_info,
        "perf_sweep": results,
        "fused_lora_verification": {
            "cosine_similarity": cos_sim,
            "max_difference": max_diff,
            "passed": cos_sim > 0.999,
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[WMMA Bench] Results saved to {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
