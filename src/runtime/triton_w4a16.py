"""RDNA3 Native Fused W4A16 (INT4 Weight, BF16 Activation) + Dynamic LoRA Triton Kernels.

Exploits AMD RDNA3 (gfx1100 / RX 7900 XTX) hardware tensor cores:
- 4-bit packed weights (8 nibbles per uint32) with group-wise scaling (group_size=128).
- Register-level fused dequantization feeding directly into 16x16x16 WMMA hardware tiles.
- Fused dynamic LoRA branch accumulation: Out = X @ dequant(W) + alpha * (X @ L_A) @ L_B.
- Single global VRAM write for both base GEMM and active domain adapter branch.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Vectorized Quantization and Packing Helpers (PyTorch GPU)
# -----------------------------------------------------------------------------

def quantize_and_pack_w4(
    weight: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT4 quantization with group scaling.

    Args:
        weight: (K, N) tensor in bfloat16 or float32.
        group_size: Quantization group size along K dimension (default 128).

    Returns:
        qweight: (K // 8, N) tensor of int32 packed nibbles (LSB-first).
        scales: (K // group_size, N) tensor of bfloat16 scales.
    """
    K, N = weight.shape
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
    assert group_size % 8 == 0, f"group_size ({group_size}) must be divisible by 8"

    n_groups = K // group_size
    w_grouped = weight.view(n_groups, group_size, N)

    max_abs = w_grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
    scales = (max_abs / 7.5).to(torch.bfloat16)

    q = torch.clamp(torch.round(w_grouped / scales.float()) + 8.0, 0, 15).to(torch.int32)
    q = q.view(K, N)

    # Pack 8 nibbles into 1 int32 along K dimension
    q_unpacked = q.view(K // 8, 8, N)
    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=weight.device).view(1, 8, 1)
    qweight = (q_unpacked << shifts).sum(dim=1, dtype=torch.int32)

    return qweight, scales.view(n_groups, N)


def unpack_and_dequantize_w4(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Unpacks int32 packed nibbles and reconstructs bfloat16 weight matrix."""
    K_words, N = qweight.shape
    K = K_words * 8
    n_groups = K // group_size

    shifts = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=qweight.device).view(1, 8, 1)
    q_expanded = (qweight.view(K_words, 1, N) >> shifts) & 0xF
    q = q_expanded.view(n_groups, group_size, N).float()

    s = scales.view(n_groups, 1, N).float()
    dequant = (q - 8.0) * s
    return dequant.view(K, N).to(torch.bfloat16)


# -----------------------------------------------------------------------------
# Autotune Configurations for RDNA3 W4A16 Kernels
# -----------------------------------------------------------------------------

def get_rdna3_w4a16_configs() -> list[triton.Config]:
    """Generates tile configurations tuned for RDNA3 16x16x16 WMMA with INT4 unpacking."""
    return [
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32, "GROUP_M": 8}, num_warps=2, num_stages=2),
    ]


# -----------------------------------------------------------------------------
# Core Triton W4A16 GEMM Kernel
# -----------------------------------------------------------------------------

@triton.jit
def _w4a16_gemm_kernel_raw(
    a_ptr, q_ptr, scale_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_qk, stride_qn,
    stride_sn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """RDNA3 WMMA-accelerated W4A16 GEMM kernel with register-level dequantization."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    offs_kw = tl.arange(0, BLOCK_K // 8)
    shifts = (tl.arange(0, 8) * 4)[None, :, None]

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    q_ptrs = q_ptr + (offs_kw[:, None, None] * stride_qk + offs_n[None, None, :] * stride_qn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)

        # Load packed int32 tile: (BLOCK_K // 8, 1, BLOCK_N)
        q_mask = (offs_kw[:, None, None] < (K - k * BLOCK_K) // 8) & (offs_n[None, None, :] < N)
        q_val = tl.load(q_ptrs, mask=q_mask, other=0)

        # Unpack 8 nibbles per word: (BLOCK_K // 8, 8, BLOCK_N) -> (BLOCK_K, BLOCK_N)
        nibbles = (q_val >> shifts) & 0xF
        b_tile_raw = tl.reshape(nibbles, (BLOCK_K, BLOCK_N))

        # Scale by per-group factor
        group_idx = (k * BLOCK_K) // GROUP_SIZE
        scale_p = scale_ptr + group_idx * stride_sn + offs_n[None, :]
        scale = tl.load(scale_p, mask=offs_n[None, :] < N, other=1.0)

        b_tile = ((b_tile_raw.to(tl.float32) - 8.0) * scale).to(tl.bfloat16)

        # WMMA matrix multiplication accumulator
        accumulator = tl.dot(a, b_tile, accumulator)

        a_ptrs += BLOCK_K * stride_ak
        q_ptrs += (BLOCK_K // 8) * stride_qk

    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


_w4a16_gemm_kernel = triton.autotune(
    configs=get_rdna3_w4a16_configs(),
    key=["M", "N", "K"],
)(_w4a16_gemm_kernel_raw)


def w4a16_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Executes W4A16 GEMM on RDNA3 with on-the-fly register dequantization.

    Args:
        x: (M, K) or (B, M, K) activation tensor in bfloat16.
        qweight: (K // 8, N) packed int32 tensor.
        scales: (K // group_size, N) bfloat16 scales.
        group_size: Quantization group size (default 128).

    Returns:
        (M, N) or (B, M, N) output tensor in bfloat16.
    """
    orig_shape = x.shape
    if x.dim() == 3:
        B, M_orig, K_orig = x.shape
        x_2d = x.reshape(B * M_orig, K_orig)
    elif x.dim() == 2:
        x_2d = x
    else:
        raise ValueError(f"Input tensor `x` must be 2D or 3D, got shape {x.shape}")

    K_words, N = qweight.shape
    K = K_words * 8
    assert x_2d.shape[1] == K, f"Dimension mismatch: x is {x_2d.shape}, qweight implies K={K}"
    assert x_2d.is_contiguous(), "Matrix `x` must be contiguous"
    assert qweight.is_contiguous(), "Matrix `qweight` must be contiguous"
    assert scales.is_contiguous(), "Matrix `scales` must be contiguous"

    M, _ = x_2d.shape
    c_2d = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _w4a16_gemm_kernel[grid](
        x_2d, qweight, scales, c_2d,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0),
        c_2d.stride(0), c_2d.stride(1),
        GROUP_SIZE=group_size,
    )

    if x.dim() == 3:
        return c_2d.reshape(B, M_orig, N)
    return c_2d


# -----------------------------------------------------------------------------
# Fused W4A16 Base GEMM + Dynamic LoRA Branch Kernel
# -----------------------------------------------------------------------------

@triton.jit
def _fused_w4a16_lora_kernel_raw(
    a_ptr, q_ptr, scale_ptr,
    lora_mid_ptr, lora_b_ptr,
    c_ptr,
    M, N, K, R,
    alpha,
    stride_am, stride_ak,
    stride_qk, stride_qn,
    stride_sn,
    stride_lmm, stride_lmr,
    stride_lbr, stride_lbn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """Fused W4A16 Base GEMM + LoRA Accumulation in registers before global write."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    offs_kw = tl.arange(0, BLOCK_K // 8)
    shifts = (tl.arange(0, 8) * 4)[None, :, None]

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    q_ptrs = q_ptr + (offs_kw[:, None, None] * stride_qk + offs_n[None, None, :] * stride_qn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 1. Base W4A16 GEMM accumulation: acc = X @ dequant(W)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)

        q_mask = (offs_kw[:, None, None] < (K - k * BLOCK_K) // 8) & (offs_n[None, None, :] < N)
        q_val = tl.load(q_ptrs, mask=q_mask, other=0)

        nibbles = (q_val >> shifts) & 0xF
        b_tile_raw = tl.reshape(nibbles, (BLOCK_K, BLOCK_N))

        group_idx = (k * BLOCK_K) // GROUP_SIZE
        scale_p = scale_ptr + group_idx * stride_sn + offs_n[None, :]
        scale = tl.load(scale_p, mask=offs_n[None, :] < N, other=1.0)

        b_tile = ((b_tile_raw.to(tl.float32) - 8.0) * scale).to(tl.bfloat16)

        accumulator = tl.dot(a, b_tile, accumulator)

        a_ptrs += BLOCK_K * stride_ak
        q_ptrs += (BLOCK_K // 8) * stride_qk

    # 2. Dynamic LoRA accumulation in registers: acc += alpha * (lora_mid @ lora_b)
    # Evaluated in chunks of 16 along rank dimension R
    offs_r_chunk = tl.arange(0, 16)
    for r_start in range(0, tl.cdiv(R, 16)):
        offs_r = r_start * 16 + offs_r_chunk
        lm_ptrs = lora_mid_ptr + offs_m[:, None] * stride_lmm + offs_r[None, :] * stride_lmr
        lb_ptrs = lora_b_ptr + offs_r[:, None] * stride_lbr + offs_n[None, :] * stride_lbn

        lm_mask = (offs_m[:, None] < M) & (offs_r[None, :] < R)
        lb_mask = (offs_r[:, None] < R) & (offs_n[None, :] < N)

        lm = tl.load(lm_ptrs, mask=lm_mask, other=0.0)
        lb = tl.load(lb_ptrs, mask=lb_mask, other=0.0)

        lora_delta = tl.dot(lm, lb)
        accumulator = accumulator + (alpha * lora_delta)

    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


_fused_w4a16_lora_kernel = triton.autotune(
    configs=get_rdna3_w4a16_configs(),
    key=["M", "N", "K", "R"],
)(_fused_w4a16_lora_kernel_raw)


def fused_w4a16_lora_matmul(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    lora_a: torch.Tensor | None = None,
    lora_b: torch.Tensor | None = None,
    alpha: float = 1.0,
    group_size: int = 128,
) -> torch.Tensor:
    """Computes Out = X @ dequant(W_4bit) + (alpha * (X @ lora_a) @ lora_b if lora present).

    Performs single-pass execution in GPU registers with only 1 global memory write.
    """
    if lora_a is None or lora_b is None:
        return w4a16_matmul(x, qweight, scales, group_size=group_size)

    orig_shape = x.shape
    if x.dim() == 3:
        B, M_orig, K_orig = x.shape
        x_2d = x.reshape(B * M_orig, K_orig)
    elif x.dim() == 2:
        x_2d = x
    else:
        raise ValueError(f"Input tensor `x` must be 2D or 3D, got shape {x.shape}")

    K_words, N = qweight.shape
    K = K_words * 8
    M, _ = x_2d.shape
    R = lora_a.shape[1]

    assert lora_a.shape[0] == K, f"lora_a shape {lora_a.shape} incompatible with K={K}"
    assert lora_b.shape == (R, N), f"lora_b shape {lora_b.shape} incompatible with R={R}, N={N}"

    # Intermediate projection: (M, R) = X @ lora_a
    # Since R is small (8..16), this is a lightweight memory-resident GEMM
    lora_mid = torch.matmul(x_2d, lora_a)

    c_2d = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _fused_w4a16_lora_kernel[grid](
        x_2d, qweight, scales,
        lora_mid, lora_b,
        c_2d,
        M, N, K, R,
        float(alpha),
        x_2d.stride(0), x_2d.stride(1),
        qweight.stride(0), qweight.stride(1),
        scales.stride(0),
        lora_mid.stride(0), lora_mid.stride(1),
        lora_b.stride(0), lora_b.stride(1),
        c_2d.stride(0), c_2d.stride(1),
        GROUP_SIZE=group_size,
    )

    if x.dim() == 3:
        return c_2d.reshape(B, M_orig, N)
    return c_2d


# -----------------------------------------------------------------------------
# ISA Inspection & Verification
# -----------------------------------------------------------------------------

def inspect_kernel_w4a16_isa(
    m: int = 64,
    k: int = 4096,
    n: int = 4096,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Compiles a standalone W4A16 kernel and inspects the emitted AMDGCN ISA for v_wmma instructions."""
    if not torch.cuda.is_available():
        return {"device": "cpu", "v_wmma_count": 0, "wmma_emitted": False}

    a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    q = torch.zeros((k // 8, n), device=device, dtype=torch.int32)
    scales = torch.ones((k // 128, n), device=device, dtype=torch.bfloat16)
    c = torch.empty((m, n), device=device, dtype=torch.bfloat16)

    grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]) * triton.cdiv(n, META["BLOCK_N"]),)
    compiled = _w4a16_gemm_kernel_raw[grid](
        a, q, scales, c,
        m, n, k,
        a.stride(0), a.stride(1),
        q.stride(0), q.stride(1),
        scales.stride(0),
        c.stride(0), c.stride(1),
        BLOCK_M=64,
        BLOCK_N=64,
        BLOCK_K=32,
        GROUP_M=8,
        GROUP_SIZE=128,
        num_warps=4,
        num_stages=2,
    )

    amdgcn_code = ""
    if hasattr(compiled, "asm") and "amdgcn" in compiled.asm:
        amdgcn_code = compiled.asm["amdgcn"]

    wmma_lines = [line.strip() for line in amdgcn_code.split("\n") if "v_wmma" in line]
    wmma_count = len(wmma_lines)

    return {
        "device": torch.cuda.get_device_name(0),
        "target_arch": "gfx1100",
        "v_wmma_count": wmma_count,
        "wmma_emitted": wmma_count > 0,
        "sample_instructions": wmma_lines[:5],
    }
