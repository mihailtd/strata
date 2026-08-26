"""RDNA3 WMMA Tile Shape & Stage Sweep for W4A16 GEMM on RX 7900 XTX."""

import time
import torch
import triton
import triton.language as tl
from runtime.triton_w4a16 import quantize_and_pack_w4


@triton.jit
def _w4a16_gemm_tile_kernel(
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

    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def sweep():
    device = "cuda:0"
    K = 5120
    N = 17408
    M = 1
    group_size = 128

    print("=" * 80)
    print(f"🔬 SWEEPING RDNA3 WMMA TILES ON RX 7900 XTX (M={M}, K={K}, N={N})")
    print("=" * 80)

    w = torch.randn((K, N), dtype=torch.bfloat16, device=device)
    qw, scales = quantize_and_pack_w4(w, group_size=group_size)
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=device)

    configs = [
        {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 8, "num_stages": 2},
        {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "num_warps": 8, "num_stages": 2},
        {"BLOCK_M": 16, "BLOCK_N": 512, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 8, "num_stages": 2},
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "num_warps": 4, "num_stages": 2},
        {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 4, "num_stages": 2},
        {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 8, "num_stages": 4},
        {"BLOCK_M": 16, "BLOCK_N": 256, "BLOCK_K": 128, "GROUP_M": 8, "num_warps": 8, "num_stages": 4},
    ]

    best_cfg = None
    best_ms = 9999.0

    for cfg in configs:
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
        
        # Warmup
        try:
            for _ in range(5):
                _w4a16_gemm_tile_kernel[grid](
                    x, qw, scales, out,
                    M, N, K,
                    x.stride(0), x.stride(1),
                    qw.stride(0), qw.stride(1),
                    scales.stride(0),
                    out.stride(0), out.stride(1),
                    BLOCK_M=cfg["BLOCK_M"],
                    BLOCK_N=cfg["BLOCK_N"],
                    BLOCK_K=cfg["BLOCK_K"],
                    GROUP_M=cfg["GROUP_M"],
                    GROUP_SIZE=group_size,
                    num_warps=cfg["num_warps"],
                    num_stages=cfg["num_stages"],
                )
            torch.cuda.synchronize()

            n_iters = 100
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _w4a16_gemm_tile_kernel[grid](
                    x, qw, scales, out,
                    M, N, K,
                    x.stride(0), x.stride(1),
                    qw.stride(0), qw.stride(1),
                    scales.stride(0),
                    out.stride(0), out.stride(1),
                    BLOCK_M=cfg["BLOCK_M"],
                    BLOCK_N=cfg["BLOCK_N"],
                    BLOCK_K=cfg["BLOCK_K"],
                    GROUP_M=cfg["GROUP_M"],
                    GROUP_SIZE=group_size,
                    num_warps=cfg["num_warps"],
                    num_stages=cfg["num_stages"],
                )
            torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) / n_iters * 1000.0

            # Calculate effective GB/s
            bytes_transferred = (K * N * 0.5) + (K * N / group_size * 2) + (M * K * 2) + (M * N * 2)
            gb_s = (bytes_transferred / 1e9) / (dt_ms / 1000.0)

            print(f"Config: {cfg} -> {dt_ms:.3f} ms | Effective Bandwidth: {gb_s:.1f} GB/s ({gb_s/960.0*100:.1f}% Bus Saturation)")

            if dt_ms < best_ms:
                best_ms = dt_ms
                best_cfg = cfg
        except Exception as e:
            print(f"Config {cfg} failed: {e}")

    print("\n" + "=" * 80)
    print(f"🏆 BEST TILE CONFIGURATION: {best_cfg}")
    print(f"   Lowest Latency: {best_ms:.3f} ms")
    print("=" * 80)


if __name__ == "__main__":
    sweep()
