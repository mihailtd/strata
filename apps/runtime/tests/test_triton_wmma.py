"""Unit tests for RDNA3 Native Wave Matrix Multiply-Accumulate (WMMA) Triton Kernels."""

from __future__ import annotations

import pytest
import torch
from runtime.triton_wmma import (
    fused_wmma_lora_matmul,
    inspect_kernel_wmma_isa,
    triton_wmma_matmul,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires ROCm / CUDA GPU")
@pytest.mark.parametrize(
    "M, K, N",
    [
        (16, 128, 128),
        (37, 512, 256),  # Non-power-of-2 test
        (64, 1024, 1024),
        (128, 2048, 1024),
        (256, 4096, 4096),
    ],
)
def test_triton_wmma_numerical_accuracy(M: int, K: int, N: int):
    """Verify Triton WMMA GEMM matches PyTorch reference with cosine similarity > 0.999."""
    torch.manual_seed(42)
    device = "cuda:0"

    a = torch.randn((M, K), device=device, dtype=torch.bfloat16)
    b = torch.randn((K, N), device=device, dtype=torch.bfloat16)

    ref = torch.matmul(a, b)
    out = triton_wmma_matmul(a, b)

    assert out.shape == (M, N)
    assert out.dtype == torch.bfloat16

    cos_sim = torch.cosine_similarity(ref.flatten().float(), out.flatten().float(), dim=0).item()
    assert cos_sim > 0.999, f"Cosine similarity {cos_sim:.6f} below threshold for shape ({M}x{K}x{N})"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires ROCm / CUDA GPU")
def test_triton_wmma_3d_batched_input():
    """Verify 3D tensor batch inputs (B, M, K) @ (K, N) -> (B, M, N)."""
    torch.manual_seed(42)
    device = "cuda:0"
    B, M, K, N = 2, 32, 1024, 512

    a = torch.randn((B, M, K), device=device, dtype=torch.bfloat16)
    b = torch.randn((K, N), device=device, dtype=torch.bfloat16)

    ref = torch.matmul(a, b)
    out = triton_wmma_matmul(a, b)

    assert out.shape == (B, M, N)
    cos_sim = torch.cosine_similarity(ref.flatten().float(), out.flatten().float(), dim=0).item()
    assert cos_sim > 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires ROCm / CUDA GPU")
def test_fused_wmma_lora_matmul():
    """Verify fused LoRA branch calculation matches reference X @ W + alpha * (X @ L_A) @ L_B."""
    torch.manual_seed(42)
    device = "cuda:0"
    M, K, N, R = 32, 1024, 1024, 16
    alpha = 2.0

    x = torch.randn((M, K), device=device, dtype=torch.bfloat16)
    w = torch.randn((K, N), device=device, dtype=torch.bfloat16)
    lora_a = torch.randn((K, R), device=device, dtype=torch.bfloat16)
    lora_b = torch.randn((R, N), device=device, dtype=torch.bfloat16)

    ref = torch.matmul(x, w) + alpha * torch.matmul(torch.matmul(x, lora_a), lora_b)
    out = fused_wmma_lora_matmul(x, w, lora_a, lora_b, alpha=alpha)

    assert out.shape == (M, N)
    cos_sim = torch.cosine_similarity(ref.flatten().float(), out.flatten().float(), dim=0).item()
    assert cos_sim > 0.999

    # Test with no LoRA (pass-through base matmul)
    out_base = fused_wmma_lora_matmul(x, w, None, None)
    ref_base = torch.matmul(x, w)
    cos_sim_base = torch.cosine_similarity(ref_base.flatten().float(), out_base.flatten().float(), dim=0).item()
    assert cos_sim_base > 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires ROCm / CUDA GPU")
def test_wmma_isa_emission_on_gpu():
    """Verify that compiled Triton kernel emits native v_wmma instructions on RDNA3."""
    isa_info = inspect_kernel_wmma_isa(m=64, k=1024, n=1024, device="cuda:0")
    assert isinstance(isa_info, dict)
    assert isa_info["wmma_emitted"] is True
    assert isa_info["v_wmma_count"] > 0
    assert len(isa_info["sample_instructions"]) > 0
    assert any("v_wmma" in instr for instr in isa_info["sample_instructions"])


def test_invalid_tensor_shapes_raise_error():
    """Verify dimension errors are caught cleanly."""
    a = torch.randn((10, 20), dtype=torch.bfloat16)
    b = torch.randn((30, 40), dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="Incompatible matrix dimensions"):
        triton_wmma_matmul(a, b)

    a_1d = torch.randn(10, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="Input tensor `a` must be 2D or 3D"):
        triton_wmma_matmul(a_1d, b)
