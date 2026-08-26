"""Benchmark: Fused SwiGLU GEMV Kernel on RDNA3 GPU (RX 7900 XTX).

Fuses Gate + Up Projections and SiLU elementwise activation directly into registers.
"""

from __future__ import annotations

import time
import torch
import triton
import triton.language as tl
from runtime.triton_w4a16 import quantize_and_pack_w4


# -----------------------------------------------------------------------------
# Fused SwiGLU GEMV Kernel (M=1)
# -----------------------------------------------------------------------------

@triton.jit
def _w4a16_fused_swiglu_gemv_kernel(
    x_ptr, qweight_ptr, scales_ptr, act_out_ptr,
    K, FFN_DIM,
    stride_xk,
    stride_qk, stride_qn,
    stride_sn,
    stride_out_n,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """Computes act = silu(X @ W_gate) * (X @ W_up) directly in registers."""
    pid_n = tl.program_id(axis=0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Gate offsets: offs_n
    # Up offsets: offs_n + FFN_DIM
    offs_n_gate = offs_n
    offs_n_up = offs_n + FFN_DIM
    
    shifts = (tl.arange(0, 8) * 4)[:, None]

    acc_gate = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_N,), dtype=tl.float32)

    offs_kw = tl.arange(0, BLOCK_K // 8)
    n_groups_k = tl.cdiv(K, BLOCK_K)

    for k_iter in range(0, n_groups_k):
        # 1. Load activation slice X: (BLOCK_K,)
        x_tile = tl.load(x_ptr + (k_iter * BLOCK_K + tl.arange(0, BLOCK_K)) * stride_xk,
                         mask=(k_iter * BLOCK_K + tl.arange(0, BLOCK_K)) < K, other=0.0)

        # 2. Load Gate weights: (BLOCK_K // 8, BLOCK_N)
        q_gate_ptrs = qweight_ptr + (offs_kw[:, None] * stride_qk + offs_n_gate[None, :] * stride_qn)
        q_gate_val = tl.load(q_gate_ptrs, mask=(offs_kw[:, None] < (K - k_iter * BLOCK_K) // 8) & (offs_n_gate[None, :] < FFN_DIM), other=0)

        # 3. Load Up weights: (BLOCK_K // 8, BLOCK_N)
        q_up_ptrs = qweight_ptr + (offs_kw[:, None] * stride_qk + offs_n_up[None, :] * stride_qn)
        q_up_val = tl.load(q_up_ptrs, mask=(offs_kw[:, None] < (K - k_iter * BLOCK_K) // 8) & (offs_n_up[None, :] < FFN_DIM * 2), other=0)

        # 4. Unpack Gate nibbles
        nibbles_gate = (q_gate_val[:, None, :] >> shifts) & 0xF
        b_gate_raw = tl.reshape(nibbles_gate, (BLOCK_K, BLOCK_N))

        # 5. Unpack Up nibbles
        nibbles_up = (q_up_val[:, None, :] >> shifts) & 0xF
        b_up_raw = tl.reshape(nibbles_up, (BLOCK_K, BLOCK_N))

        # 6. Scale Gate & Up
        group_idx = (k_iter * BLOCK_K) // GROUP_SIZE
        scale_gate = tl.load(scales_ptr + group_idx * stride_sn + offs_n_gate, mask=offs_n_gate < FFN_DIM, other=1.0)
        scale_up = tl.load(scales_ptr + group_idx * stride_sn + offs_n_up, mask=offs_n_up < FFN_DIM * 2, other=1.0)

        b_gate = (b_gate_raw.to(tl.float32) - 8.0) * scale_gate.to(tl.float32)
        b_up = (b_up_raw.to(tl.float32) - 8.0) * scale_up.to(tl.float32)

        # 7. Accumulate Dot Products
        acc_gate += tl.sum(x_tile[:, None] * b_gate, axis=0)
        acc_up += tl.sum(x_tile[:, None] * b_up, axis=0)

        # Advance pointer
        qweight_ptr += (BLOCK_K // 8) * stride_qk

    # 8. Fused SiLU activation in registers: silu(gate) * up
    # silu(z) = z / (1.0 + exp(-z))
    silu_gate = acc_gate / (1.0 + tl.exp(-acc_gate))
    fused_act = silu_gate * acc_up

    # 9. Store only the final fused activation
    tl.store(act_out_ptr + offs_n * stride_out_n, fused_act.to(tl.bfloat16), mask=offs_n < FFN_DIM)


def fused_swiglu_gemv(
    x: torch.Tensor,
    qweight_gate_up: torch.Tensor,
    scales_gate_up: torch.Tensor,
    act_out: torch.Tensor,
    group_size: int = 128,
):
    """Executes fused Gate+Up projection with register SiLU activation."""
    M, K = x.shape
    assert M == 1, "fused_swiglu_gemv is specialized for M=1 decode"
    _, N_total = qweight_gate_up.shape
    FFN_DIM = N_total // 2

    BLOCK_N = 128
    BLOCK_K = 64
    grid = (triton.cdiv(FFN_DIM, BLOCK_N),)

    _w4a16_fused_swiglu_gemv_kernel[grid](
        x, qweight_gate_up, scales_gate_up, act_out,
        K, FFN_DIM,
        x.stride(1),
        qweight_gate_up.stride(0), qweight_gate_up.stride(1),
        scales_gate_up.stride(0),
        act_out.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE=group_size,
        num_warps=4,
        num_stages=2,
    )


def test_fused_swiglu():
    device = "cuda:0"
    print("=" * 80)
    print("⚡ BENCHMARKING FUSED SwiGLU GEMV KERNEL ON RX 7900 XTX")
    print("=" * 80)

    K = 5120
    FFN_DIM = 17408
    group_size = 128

    # Create Gate + Up weights
    w_gate = torch.randn((K, FFN_DIM), dtype=torch.bfloat16, device=device)
    w_up = torch.randn((K, FFN_DIM), dtype=torch.bfloat16, device=device)
    w_gate_up = torch.cat([w_gate, w_up], dim=-1)

    qw_gate_up, scales_gate_up = quantize_and_pack_w4(w_gate_up, group_size=group_size)
    x = torch.randn((1, K), dtype=torch.bfloat16, device=device)
    act_out = torch.empty((1, FFN_DIM), dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(10):
        fused_swiglu_gemv(x, qw_gate_up, scales_gate_up, act_out, group_size=group_size)
    torch.cuda.synchronize()

    # Benchmark Fused Kernel
    n_iters = 500
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fused_swiglu_gemv(x, qw_gate_up, scales_gate_up, act_out, group_size=group_size)
    torch.cuda.synchronize()
    dt_fused_ms = (time.perf_counter() - t0) / n_iters * 1000.0

    print(f"Fused Gate+Up+SiLU (5120 -> 34816) Latency: {dt_fused_ms:.3f} ms")
    
    # Calculate effective bandwidth
    bytes_transferred = (K * FFN_DIM * 2 * 0.5) + (K * FFN_DIM * 2 / group_size * 2) + (K * 2) + (FFN_DIM * 2)
    gb_s = (bytes_transferred / 1e9) / (dt_fused_ms / 1000.0)
    print(f"Effective Memory Bandwidth:                   {gb_s:.1f} GB/s ({gb_s/960.0*100:.1f}% Bus Saturation)")

    # Compare against 2 separate GEMMs + elementwise PyTorch activation
    t0_sep = time.perf_counter()
    for _ in range(n_iters):
        # 2 separate passes
        gate = x @ w_gate
        up = x @ w_up
        act = torch.nn.functional.silu(gate) * up
    torch.cuda.synchronize()
    dt_sep_ms = (time.perf_counter() - t0_sep) / n_iters * 1000.0
    print(f"Separate GEMMs + PyTorch Activation:          {dt_sep_ms:.3f} ms")
    print(f"🚀 Fused SwiGLU Speedup:                      {dt_sep_ms / dt_fused_ms:.2f}x Faster")
    print("=" * 80)


if __name__ == "__main__":
    test_fused_swiglu()
