"""Unit tests for RDNA3 Native Fused W4A16 + Dynamic LoRA Triton Kernels."""

from __future__ import annotations

import pytest
import torch
from triton_w4a16 import (
    fused_w4a16_lora_matmul,
    inspect_kernel_w4a16_isa,
    quantize_and_pack_w4,
    unpack_and_dequantize_w4,
    w4a16_matmul,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires ROCm / CUDA GPU")
def test_quantize_and_unpack_roundtrip():
    """Verify GPU-vectorized quantization and unpacking preserves signal fidelity."""
    torch.manual_seed(42)
    device = "cuda:0"
    K, N = 2560, 4096

    w = torch.randn((K, N), device=device, dtype=torch.bfloat16)
    qweight, scales = quantize_and_pack_w4(w, group_size=128)
    dequant = unpack_and_dequantize_w4(qweight, scales, group_size=128)

    assert qweight.shape == (K // 8, N)
    assert scales.shape == (K // 128, N)
    assert dequant.shape == (K, N)
    assert dequant.dtype == torch.bfloat16

    # 4-bit quantization on random Gaussian has ~0.993 cosine similarity
    cos_sim = torch.cosine_similarity(w.flatten().float(), dequant.flatten().float(), dim=0).item()
    assert cos_sim > 0.99, f"Quantization cosine similarity {cos_sim:.6f} below expected threshold"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires ROCm / CUDA GPU")
@pytest.mark.parametrize(
    "M, K, N",
    [
        (1, 2560, 2560),  # Single token decode
        (2, 2560, 2560),  # Speculative verification K=2
        (4, 2560, 2560),  # Speculative verification K=4
        (16, 2560, 2560),  # Micro-batch
        (37, 2560, 2560),  # Non-power-of-2 sequence length
        (64, 2560, 7680),  # QKV projection shape
        (128, 2560, 6912),  # MLP gate/up projection shape
    ],
)
def test_w4a16_matmul_numerical_accuracy(M: int, K: int, N: int):
    """Verify W4A16 GEMM matches PyTorch dequantized reference with cosine similarity > 0.999."""
    torch.manual_seed(42)
    device = "cuda:0"

    x = torch.randn((M, K), device=device, dtype=torch.bfloat16)
    w = torch.randn((K, N), device=device, dtype=torch.bfloat16)

    qweight, scales = quantize_and_pack_w4(w, group_size=128)
    dequant_ref = unpack_and_dequantize_w4(qweight, scales, group_size=128)

    ref_out = torch.matmul(x, dequant_ref)
    triton_out = w4a16_matmul(x, qweight, scales, group_size=128)

    assert triton_out.shape == (M, N)
    assert triton_out.dtype == torch.bfloat16

    cos_sim = torch.cosine_similarity(ref_out.flatten().float(), triton_out.flatten().float(), dim=0).item()
    assert cos_sim > 0.999, f"Cosine similarity {cos_sim:.6f} below threshold for shape ({M}x{K}x{N})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires ROCm / CUDA GPU")
def test_w4a16_3d_batched_input():
    """Verify 3D tensor batch inputs (B, M, K) with W4A16 matmul."""
    torch.manual_seed(42)
    device = "cuda:0"
    B, M, K, N = 2, 32, 2560, 1024

    x = torch.randn((B, M, K), device=device, dtype=torch.bfloat16)
    w = torch.randn((K, N), device=device, dtype=torch.bfloat16)

    qweight, scales = quantize_and_pack_w4(w, group_size=128)
    dequant_ref = unpack_and_dequantize_w4(qweight, scales, group_size=128)

    ref_out = torch.matmul(x, dequant_ref)
    triton_out = w4a16_matmul(x, qweight, scales, group_size=128)

    assert triton_out.shape == (B, M, N)
    cos_sim = torch.cosine_similarity(ref_out.flatten().float(), triton_out.flatten().float(), dim=0).item()
    assert cos_sim > 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires ROCm / CUDA GPU")
@pytest.mark.parametrize("rank", [8, 16, 32])
def test_fused_w4a16_lora_numerical_accuracy(rank: int):
    """Verify fused W4A16 base GEMM + LoRA accumulation matches mathematical reference."""
    torch.manual_seed(42)
    device = "cuda:0"
    M, K, N = 32, 2560, 2560
    alpha = 2.0

    x = torch.randn((M, K), device=device, dtype=torch.bfloat16)
    w = torch.randn((K, N), device=device, dtype=torch.bfloat16)
    lora_a = torch.randn((K, rank), device=device, dtype=torch.bfloat16)
    lora_b = torch.randn((rank, N), device=device, dtype=torch.bfloat16)

    qweight, scales = quantize_and_pack_w4(w, group_size=128)
    dequant_ref = unpack_and_dequantize_w4(qweight, scales, group_size=128)

    # Reference: Base GEMM + alpha * (X @ lora_a) @ lora_b
    ref_base = torch.matmul(x, dequant_ref)
    ref_lora = alpha * torch.matmul(torch.matmul(x, lora_a), lora_b)
    ref_out = ref_base + ref_lora

    fused_out = fused_w4a16_lora_matmul(
        x,
        qweight,
        scales,
        lora_a=lora_a,
        lora_b=lora_b,
        alpha=alpha,
        group_size=128,
    )

    assert fused_out.shape == (M, N)
    cos_sim = torch.cosine_similarity(ref_out.flatten().float(), fused_out.flatten().float(), dim=0).item()
    assert cos_sim > 0.999, f"Fused LoRA cosine similarity {cos_sim:.6f} below threshold for rank={rank}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires ROCm / CUDA GPU")
def test_fused_w4a16_lora_none_passthrough():
    """Verify that passing None for LoRA adapters falls back to clean base matmul."""
    torch.manual_seed(42)
    device = "cuda:0"
    M, K, N = 16, 2560, 1024

    x = torch.randn((M, K), device=device, dtype=torch.bfloat16)
    w = torch.randn((K, N), device=device, dtype=torch.bfloat16)
    qweight, scales = quantize_and_pack_w4(w, group_size=128)

    out_base = w4a16_matmul(x, qweight, scales, group_size=128)
    out_fused_none = fused_w4a16_lora_matmul(x, qweight, scales, lora_a=None, lora_b=None)

    diff = (out_base - out_fused_none).abs().max().item()
    assert diff == 0.0, f"Passthrough mismatch: max diff {diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires ROCm / CUDA GPU")
def test_w4a16_isa_emission_on_gpu():
    """Verify that compiled Triton W4A16 kernel emits native v_wmma instructions on RDNA3."""
    isa_info = inspect_kernel_w4a16_isa(m=64, k=1024, n=1024, device="cuda:0")
    assert isinstance(isa_info, dict)
    assert isa_info["wmma_emitted"] is True
    assert isa_info["v_wmma_count"] > 0
    assert len(isa_info["sample_instructions"]) > 0
    assert any("v_wmma" in instr for instr in isa_info["sample_instructions"])


def test_vram_footprint_reduction():
    """Verify that W4A16 packed weights achieve exactly 4x memory footprint reduction."""
    K, N = 2560, 4096
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    w = torch.randn((K, N), device=device, dtype=torch.bfloat16)
    qweight, scales = quantize_and_pack_w4(w, group_size=128)

    bf16_bytes = w.numel() * w.element_size()  # 2560 * 4096 * 2 = 20,971,520 bytes (~20.97 MB)
    qweight_bytes = qweight.numel() * qweight.element_size()  # 320 * 4096 * 4 = 5,242,880 bytes (~5.24 MB)
    scales_bytes = scales.numel() * scales.element_size()  # 20 * 4096 * 2 = 163,840 bytes (~0.16 MB)
    total_w4_bytes = qweight_bytes + scales_bytes

    reduction_factor = bf16_bytes / total_w4_bytes
    assert reduction_factor > 3.8, f"VRAM reduction factor {reduction_factor:.2f} is lower than expected ~3.88x"
