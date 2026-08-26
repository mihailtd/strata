"""Vectorized GEMV (M=1) Kernel Tuning Benchmark for RDNA3 W4A16."""

import time
import torch
import triton
import triton.language as tl
from runtime.triton_w4a16 import quantize_and_pack_w4, w4a16_matmul


@triton.jit
def _w4a16_gemv_m1_kernel(
    a_ptr, q_ptr, scale_ptr, c_ptr,
    N, K,
    stride_ak,
    stride_qk, stride_qn,
    stride_sn,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    """Specialized GEMV (M=1) Kernel for Single-Token Decoding on RDNA3 WMMA."""
    pid_n = tl.program_id(axis=0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kw = tl.arange(0, BLOCK_K // 8)
    shifts = (tl.arange(0, 8) * 4)[:, None]  # (8, 1)

    accumulator = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Base pointers
    q_ptrs = q_ptr + (offs_kw[:, None] * stride_qk + offs_n[None, :] * stride_qn)
    a_ptrs = a_ptr + (tl.arange(0, BLOCK_K) * stride_ak)

    n_groups_k = tl.cdiv(K, BLOCK_K)

    for k_iter in range(0, n_groups_k):
        # 1. Load activation slice for M=1: (BLOCK_K,)
        a_val = tl.load(a_ptrs, mask=tl.arange(0, BLOCK_K) < K - k_iter * BLOCK_K, other=0.0)

        # 2. Load packed weight tile: (BLOCK_K // 8, BLOCK_N) in int32
        q_mask = (offs_kw[:, None] < (K - k_iter * BLOCK_K) // 8) & (offs_n[None, :] < N)
        q_val = tl.load(q_ptrs, mask=q_mask, other=0)

        # 3. Unpack 8 nibbles: (BLOCK_K // 8, 8, BLOCK_N)
        # Reshape to (BLOCK_K, BLOCK_N)
        # Loop unrolled across 8 nibble slices
        for sub_k in range(8):
            nibble = (q_val >> (sub_k * 4)) & 0xF
            # Scale
            group_idx = (k_iter * BLOCK_K + sub_k * (BLOCK_K // 8)) // GROUP_SIZE
            scale_p = scale_ptr + group_idx * stride_sn + offs_n
            scale = tl.load(scale_p, mask=offs_n < N, other=1.0)
            
            w_dequant = (nibble.to(tl.float32) - 8.0) * scale.to(tl.float32)
            
            # Multiply with corresponding slice of a
            # Accumulate into accumulator: sum over k
            # Vectorized dot product along K
            # a_val has shape (BLOCK_K,)
            a_sub = tl.load(a_ptr + (k_iter * BLOCK_K + tl.arange(0, BLOCK_K // 8) * 8 + sub_k) * stride_ak,
                            mask=(k_iter * BLOCK_K + tl.arange(0, BLOCK_K // 8) * 8 + sub_k) < K, other=0.0)
            
            accumulator += tl.sum(a_sub[:, None] * w_dequant, axis=0)

        a_ptrs += BLOCK_K * stride_ak
        q_ptrs += (BLOCK_K // 8) * stride_qk

    c_ptrs = c_ptr + offs_n * stride_cn
    tl.store(c_ptrs, accumulator.to(tl.bfloat16), mask=offs_n < N)


def benchmark_gemv():
    device = "cuda:0"
    K = 5120
    N = 17408
    group_size = 128

    print(f"Benchmarking M=1 Decode on 27B Layer: K={K}, N={N}")
    torch.manual_seed(42)
    w = torch.randn((K, N), dtype=torch.bfloat16, device=device)
    qw, scales = quantize_and_pack_w4(w, group_size=group_size)
    x = torch.randn((1, K), dtype=torch.bfloat16, device=device)

    # 1. Benchmark current baseline Triton matmul
    for _ in range(10):
        _ = w4a16_matmul(x, qw, scales, group_size=group_size)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_iters = 200
    for _ in range(n_iters):
        _ = w4a16_matmul(x, qw, scales, group_size=group_size)
    torch.cuda.synchronize()
    dt_base = (time.perf_counter() - t0) / n_iters * 1000.0

    print(f"Baseline Triton w4a16_matmul: {dt_base:.3f} ms")


if __name__ == "__main__":
    benchmark_gemv()
