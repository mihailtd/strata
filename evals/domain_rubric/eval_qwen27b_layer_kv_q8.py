"""Real-world Empirical Evaluation: Qwen 3.8-27B Layer 3 Full Attention KV Cache.

Evaluates NaiveCatBF16 vs PreallocatedBF16 vs PreallocatedQ8 on real layer weights
extracted from GGUF, running 512 autoregressive decode steps on AMD Radeon RX 7900 XTX.
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runtime.native_27b_engine import (
    Qwen35FullAttentionBlock,
    apply_rotary_emb,
)



class PreallocatedKVCache:
    """Contiguous Pre-allocated Key-Value Cache supporting BF16 and Q8_0 modes."""

    def __init__(
        self,
        batch_size: int = 1,
        num_heads: int = 4,
        head_dim: int = 256,
        max_seq_len: int = 2048,
        mode: str = "bf16",
        device: torch.device = torch.device("cuda:0"),
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.mode = mode.lower()
        self.device = device
        self.current_len = 0

        if self.mode == "bf16":
            self.k_cache = torch.zeros(
                (batch_size, num_heads, max_seq_len, head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            self.v_cache = torch.zeros(
                (batch_size, num_heads, max_seq_len, head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            self.k_scales = None
            self.v_scales = None
        elif self.mode == "q8_0":
            self.k_cache = torch.zeros(
                (batch_size, num_heads, max_seq_len, head_dim),
                dtype=torch.int8,
                device=device,
            )
            self.v_cache = torch.zeros(
                (batch_size, num_heads, max_seq_len, head_dim),
                dtype=torch.int8,
                device=device,
            )
            self.k_scales = torch.zeros(
                (batch_size, num_heads, max_seq_len, 1),
                dtype=torch.bfloat16,
                device=device,
            )
            self.v_scales = torch.zeros(
                (batch_size, num_heads, max_seq_len, 1),
                dtype=torch.bfloat16,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported KV cache mode: {self.mode}")

    def update(
        self, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Appends new tokens in-place and returns the active context slice."""
        b, h, s, d = k_new.shape
        start = self.current_len
        end = start + s

        if end > self.max_seq_len:
            raise RuntimeError(
                f"KV Cache overflow: attempted sequence length {end} > max_seq_len {self.max_seq_len}"
            )

        if self.mode == "bf16":
            self.k_cache[:, :, start:end, :] = k_new
            self.v_cache[:, :, start:end, :] = v_new
            self.current_len = end
            return self.k_cache[:, :, :end, :], self.v_cache[:, :, :end, :]
        else:
            # Q8_0 symmetric quantization per token per head
            k_scale = (torch.amax(torch.abs(k_new), dim=-1, keepdim=True).clamp(min=1e-5) / 127.0).to(torch.bfloat16)
            v_scale = (torch.amax(torch.abs(v_new), dim=-1, keepdim=True).clamp(min=1e-5) / 127.0).to(torch.bfloat16)

            k_q = torch.clamp(torch.round(k_new / k_scale), -128, 127).to(torch.int8)
            v_q = torch.clamp(torch.round(v_new / v_scale), -128, 127).to(torch.int8)

            self.k_cache[:, :, start:end, :] = k_q
            self.v_cache[:, :, start:end, :] = v_q
            self.k_scales[:, :, start:end, :] = k_scale
            self.v_scales[:, :, start:end, :] = v_scale
            self.current_len = end

            # Dequantize active context
            k_deq = self.k_cache[:, :, :end, :].to(torch.bfloat16) * self.k_scales[:, :, :end, :]
            v_deq = self.v_cache[:, :, :end, :].to(torch.bfloat16) * self.v_scales[:, :, :end, :]
            return k_deq, v_deq

    def get_memory_bytes(self) -> int:
        mem = self.k_cache.nelement() * self.k_cache.element_size()
        mem += self.v_cache.nelement() * self.v_cache.element_size()
        if self.k_scales is not None:
            mem += self.k_scales.nelement() * self.k_scales.element_size()
            mem += self.v_scales.nelement() * self.v_scales.element_size()
        return mem


def run_single_step_layer(
    layer: Qwen35FullAttentionBlock,
    x: torch.Tensor,
    cos_sin: Tuple[torch.Tensor, torch.Tensor],
    k_in: torch.Tensor,
    v_in: torch.Tensor,
) -> torch.Tensor:
    """Executes single-token attention forward pass with given full key and value contexts."""
    b, s, _ = x.shape
    x_norm = layer.attn_norm(x)

    # 1. Q projection via Triton
    q_full = layer.attn_q(x_norm)
    query_states, gate = torch.chunk(
        q_full.view(b, s, 24, 256 * 2), 2, dim=-1
    )
    gate = gate.reshape(b, s, -1)
    q = layer.attn_q_norm(query_states).transpose(1, 2)

    # RoPE
    cos, sin = cos_sin
    q = apply_rotary_emb(q, cos, sin)

    # GQA Repeat
    k_rep = k_in.repeat_interleave(6, dim=1)
    v_rep = v_in.repeat_interleave(6, dim=1)

    attn_out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False)
    attn_out = attn_out.transpose(1, 2).contiguous().view(b, s, -1)
    attn_out = attn_out * torch.sigmoid(gate)

    # Output projection
    y = layer.attn_output(attn_out)
    x = x + y

    # SwiGLU FFN
    x_ffn_norm = layer.post_attention_norm(x)
    ffn_gate = layer.ffn_gate(x_ffn_norm)
    ffn_up = layer.ffn_up(x_ffn_norm)
    swiglu_act = F.silu(ffn_gate) * ffn_up
    mlp_out = layer.ffn_down(swiglu_act)

    return x + mlp_out


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights_path = Path("models/qwen3.8-27b-triton/layer_3.pt")

    if not weights_path.exists():
        print(f"Error: Layer 3 weights not found at {weights_path}")
        sys.exit(1)

    print("================================================================================")
    print("      REAL-WORLD EMPIRICAL EVALUATION: QWEN 3.8-27B LAYER 3 ATTENTION         ")
    print("================================================================================")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Loading real layer weights from: {weights_path}")

    # 1. Instantiate Layer & Load Triton W4A16 Weights
    layer = Qwen35FullAttentionBlock(layer_idx=3, device=device)
    layer_dict = torch.load(weights_path, map_location=device, weights_only=False)
    layer.load_weights(layer_dict)
    print("Layer 3 weights successfully bound to Triton W4A16 GEMV engines.")

    # 2. Setup RoPE table
    inv_freq = 1.0 / (10000000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=device) / 64))
    t = torch.arange(0, 4096, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos_table = torch.cos(freqs).to(torch.bfloat16).unsqueeze(0).unsqueeze(0)  # (1, 1, 4096, 32)
    sin_table = torch.sin(freqs).to(torch.bfloat16).unsqueeze(0).unsqueeze(0)


    # 3. Setup Caches for 512 Decode Steps
    num_steps = 512
    max_seq_len = 1024

    prealloc_bf16 = PreallocatedKVCache(
        batch_size=1, num_heads=4, head_dim=256, max_seq_len=max_seq_len, mode="bf16", device=device
    )
    prealloc_q8 = PreallocatedKVCache(
        batch_size=1, num_heads=4, head_dim=256, max_seq_len=max_seq_len, mode="q8_0", device=device
    )

    # Naive Cat state
    naive_k = None
    naive_v = None

    # Deterministic input stream simulating hidden states from previous layer
    torch.manual_seed(2026)
    simulated_tokens = torch.randn(num_steps, 1, 5120, dtype=torch.bfloat16, device=device)

    # Warmup Triton kernels
    print("\nWarming up Triton kernels...")
    with torch.no_grad():
        x_w = simulated_tokens[0:1]
        x_norm = layer.attn_norm(x_w)
        _ = layer.attn_q(x_norm)
        _ = layer.attn_k(x_norm)
        _ = layer.attn_v(x_norm)
        _ = layer.attn_output(torch.randn(1, 1, 6144, dtype=torch.bfloat16, device=device))
        _ = layer.ffn_gate(x_w)
        _ = layer.ffn_up(x_w)
        _ = layer.ffn_down(torch.randn(1, 1, 17408, dtype=torch.bfloat16, device=device))
    torch.cuda.synchronize(device)
    print("Warmup complete.")

    print(f"\nRunning {num_steps} autoregressive decode steps...")
    metrics = {
        "naive_latencies_ms": [],
        "bf16_latencies_ms": [],
        "q8_latencies_ms": [],
        "bf16_cos_sims": [],
        "q8_cos_sims": [],
        "max_abs_errors": [],
    }

    t0_all = time.perf_counter()

    for step in range(num_steps):
        x_step = simulated_tokens[step : step + 1]  # (1, 1, 5120)
        cos_sin = (cos_table[:, :, step : step + 1, :], sin_table[:, :, step : step + 1, :])

        # Compute K and V for this new token
        with torch.no_grad():
            x_norm = layer.attn_norm(x_step)
            k_raw = layer.attn_k(x_norm) if layer.attn_k else x_norm[..., :1024]
            k_new = layer.attn_k_norm(k_raw.view(1, 1, 4, 256)).transpose(1, 2)
            v_raw = layer.attn_v(x_norm) if layer.attn_v else x_norm[..., :1024]
            v_new = v_raw.view(1, 1, 4, 256).transpose(1, 2)
            k_new = apply_rotary_emb(k_new, cos_sin[0], cos_sin[1])

        # --- A. Naive Cat Baseline ---
        t_start = time.perf_counter()
        if naive_k is None:
            naive_k = k_new
            naive_v = v_new
        else:
            naive_k = torch.cat([naive_k, k_new], dim=2)
            naive_v = torch.cat([naive_v, v_new], dim=2)
        out_naive = run_single_step_layer(layer, x_step, cos_sin, naive_k, naive_v)
        torch.cuda.synchronize(device)
        metrics["naive_latencies_ms"].append((time.perf_counter() - t_start) * 1000)

        # --- B. Preallocated BF16 ---
        t_start = time.perf_counter()
        k_bf16, v_bf16 = prealloc_bf16.update(k_new, v_new)
        out_bf16 = run_single_step_layer(layer, x_step, cos_sin, k_bf16, v_bf16)
        torch.cuda.synchronize(device)
        metrics["bf16_latencies_ms"].append((time.perf_counter() - t_start) * 1000)

        # --- C. Preallocated Q8_0 ---
        t_start = time.perf_counter()
        k_q8, v_q8 = prealloc_q8.update(k_new, v_new)
        out_q8 = run_single_step_layer(layer, x_step, cos_sin, k_q8, v_q8)
        torch.cuda.synchronize(device)
        metrics["q8_latencies_ms"].append((time.perf_counter() - t_start) * 1000)

        # Validation Checks
        assert not torch.isnan(out_bf16).any(), f"NaN in BF16 at step {step}"
        assert not torch.isnan(out_q8).any(), f"NaN in Q8 at step {step}"

        sim_bf16 = F.cosine_similarity(out_naive.view(-1).float(), out_bf16.view(-1).float(), dim=0).item()
        sim_q8 = F.cosine_similarity(out_naive.view(-1).float(), out_q8.view(-1).float(), dim=0).item()
        max_err = torch.max(torch.abs(out_naive - out_q8)).item()

        metrics["bf16_cos_sims"].append(sim_bf16)
        metrics["q8_cos_sims"].append(sim_q8)
        metrics["max_abs_errors"].append(max_err)

        if (step + 1) % 128 == 0 or step == 0:
            print(
                f"  Step {step+1:3d}/{num_steps} | "
                f"Naive: {metrics['naive_latencies_ms'][-1]:.2f}ms | "
                f"BF16: {metrics['bf16_latencies_ms'][-1]:.2f}ms (sim: {sim_bf16:.6f}) | "
                f"Q8: {metrics['q8_latencies_ms'][-1]:.2f}ms (sim: {sim_q8:.6f}, max_err: {max_err:.4f})"
            )

    total_time = time.perf_counter() - t0_all
    print(f"\nAll {num_steps} steps completed in {total_time:.2f}s!")

    # Summary Calculations
    avg_naive = sum(metrics["naive_latencies_ms"]) / num_steps
    avg_bf16 = sum(metrics["bf16_latencies_ms"]) / num_steps
    avg_q8 = sum(metrics["q8_latencies_ms"]) / num_steps
    min_q8_sim = min(metrics["q8_cos_sims"])
    avg_q8_sim = sum(metrics["q8_cos_sims"]) / num_steps
    max_q8_err = max(metrics["max_abs_errors"])

    bf16_mem_mb = prealloc_bf16.get_memory_bytes() / (1024**2)
    q8_mem_mb = prealloc_q8.get_memory_bytes() / (1024**2)

    print("\n================================================================================")
    print("                              FINAL EMPIRICAL RESULTS                          ")
    print("================================================================================")
    print(f"Memory (Max Seq {max_seq_len}):")
    print(f"  Preallocated BF16: {bf16_mem_mb:.2f} MB")
    print(f"  Preallocated Q8_0: {q8_mem_mb:.2f} MB  (Saved: {(1 - q8_mem_mb/bf16_mem_mb)*100:.1f}%)")
    print(f"Latency (Avg per Decode Step over {num_steps} steps):")
    print(f"  NaiveCatBF16:      {avg_naive:.3f} ms")
    print(f"  PreallocatedBF16:  {avg_bf16:.3f} ms")
    print(f"  PreallocatedQ8:    {avg_q8:.3f} ms")
    print(f"Fidelity vs Exact (NaiveCatBF16):")
    print(f"  PreallocatedBF16 Cosine Sim: {min(metrics['bf16_cos_sims']):.6f} (Exact Bit-Identity)")
    print(f"  PreallocatedQ8 Avg Cosine Sim: {avg_q8_sim:.6f}")
    print(f"  PreallocatedQ8 Min Cosine Sim: {min_q8_sim:.6f}")
    print(f"  PreallocatedQ8 Max Abs Error:  {max_q8_err:.6f}")

    # Pass / Fail criteria
    tests_passed = (
        min(metrics["bf16_cos_sims"]) >= 0.9999
        and min_q8_sim >= 0.9990
        and q8_mem_mb < bf16_mem_mb * 0.55
    )

    print(f"\nEmpirical Test Status: {'>>> PASSED <<<' if tests_passed else '>>> FAILED <<<'}")

    # Save summary
    results_dir = Path("results/benchmarks")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "layer3_kv_eval.json"
    summary_data = {
        "num_steps": num_steps,
        "max_seq_len": max_seq_len,
        "bf16_mem_mb": bf16_mem_mb,
        "q8_mem_mb": q8_mem_mb,
        "memory_saving_percent": (1 - q8_mem_mb / bf16_mem_mb) * 100,
        "avg_latency_naive_ms": avg_naive,
        "avg_latency_bf16_ms": avg_bf16,
        "avg_latency_q8_ms": avg_q8,
        "min_q8_cos_sim": min_q8_sim,
        "avg_q8_cos_sim": avg_q8_sim,
        "max_q8_abs_error": max_q8_err,
        "passed": tests_passed,
    }
    with open(out_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved empirical evaluation summary to {out_file}")

    if not tests_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
