"""FlashNorm-style weight folding: benchmark viability on this hardware.

Measures whether absorbing RMSNorm scale weights into downstream Linear
projections provides a meaningful speedup at decode-time (batch=1, seq=1)
on the RX 7900 XTX / ROCm / gfx1100 target.

Three gates, all of which must pass for Phase 2 implementation to proceed:

    Gate 1.1  Latency: torch.cuda.Event profiling of RMSNorm kernel overhead
    Gate 1.2  Numerical: max_relative_diff between original and folded paths < 1e-4
    Gate 1.3  Throughput: folding saves > 2% of total norm+linear time

The benchmark operates on mock modules matching Qwen3.5-4B dimensions:
    hidden_size = 2560
    intermediate_size = 6912
    num_foldable_norms = 73  (36 layers × 2 + 1 final)

    uv run --env-file .env benchmarks/runtime/folding/benchmark_flash_norm_fusion.py
"""

import json
import sys
from pathlib import Path

import torch
from torch import nn

# fla's device probe is @cache'd at import — CUDA touch MUST precede transformers
torch.zeros(1, device="cuda")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.novel_peft import set_hard_vram_cap  # noqa: E402

# ---------------------------------------------------------------------------
# Qwen3.5-4B dimensions
# ---------------------------------------------------------------------------
HIDDEN = 2560
INTERMEDIATE = 6912
HEAD_DIM = 128
NUM_Q_HEADS = 20
NUM_KV_HEADS = 4
EPS = 1e-6
NUM_LAYERS = 36
# Per layer: input_layernorm + post_attention_layernorm = 2, plus 1 final norm
NUM_FOLDABLE_NORMS = NUM_LAYERS * 2 + 1
WARMUP_ITERS = 200
BENCH_ITERS = 1000


# ---------------------------------------------------------------------------
# Mock modules
# ---------------------------------------------------------------------------
class MockRMSNorm(nn.Module):
    """Matches Qwen3_5RMSNorm: unit_offset, weight init zeros."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).type_as(x)


class MockRMSNormScaleFree(nn.Module):
    """RMSNorm without learnable weight — the post-fold replacement."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.type_as(x)


def fold_weights(norm: MockRMSNorm, linear: nn.Linear) -> nn.Linear:
    """Create a new Linear with W_folded[:, i] = (1 + γ[i]) * W[:, i].

    For unit_offset RMSNorm: effective scale is (1 + weight), not weight.
    Linear.weight has shape (out_features, in_features).
    The input vector x is normalized channel-wise, so (x * gamma) @ W.T = x @ (W * gamma).T
    """
    gamma = (1.0 + norm.weight.float()).unsqueeze(0)  # (1, in_features)
    folded = nn.Linear(
        linear.in_features,
        linear.out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    folded.weight.data.copy_((gamma * linear.weight.float()).to(linear.weight.dtype))
    if linear.bias is not None:
        folded.bias.data.copy_(linear.bias.data)
    return folded


# ---------------------------------------------------------------------------
# Timing utility
# ---------------------------------------------------------------------------
def cuda_event_time_ms(fn, warmup: int = WARMUP_ITERS, iters: int = BENCH_ITERS) -> float:
    """Mean kernel time in ms using torch.cuda.Event (not Python time)."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters


# ---------------------------------------------------------------------------
# Gate 1.1: Baseline latency measurement
# ---------------------------------------------------------------------------
def bench_latency():
    """Profile RMSNorm, Linear, and RMSNorm+Linear at decode-time shape."""
    print("\n" + "=" * 70)
    print("GATE 1.1: Baseline Latency Measurement")
    print("=" * 70)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    norm = MockRMSNorm(HIDDEN, EPS).to(device, dtype)
    linear = nn.Linear(HIDDEN, INTERMEDIATE, bias=False).to(device, dtype)
    x = torch.randn(1, 1, HIDDEN, device=device, dtype=dtype)

    # Scale-free norm + folded linear
    norm_sf = MockRMSNormScaleFree(HIDDEN, EPS).to(device, dtype)
    linear_folded = fold_weights(norm, linear).to(device, dtype)

    # Individual kernels
    def norm_only():
        return norm(x)

    def linear_only():
        return linear(x)

    def norm_then_linear():
        return linear(norm(x))

    def scalefree_then_folded():
        return linear_folded(norm_sf(x))

    t_norm = cuda_event_time_ms(norm_only)
    t_linear = cuda_event_time_ms(linear_only)
    t_original = cuda_event_time_ms(norm_then_linear)
    t_folded = cuda_event_time_ms(scalefree_then_folded)

    print(f"  RMSNorm alone:           {t_norm * 1000:.2f} µs")
    print(f"  Linear alone:            {t_linear * 1000:.2f} µs")
    print(f"  RMSNorm + Linear:        {t_original * 1000:.2f} µs")
    print(f"  ScaleFree + FoldedLinear: {t_folded * 1000:.2f} µs")
    print(f"  Per-norm savings:        {(t_original - t_folded) * 1000:.2f} µs")
    print(f"  Norm share of total:     {t_norm / t_original * 100:.1f}%")

    # Extrapolate to full model
    savings_per_norm_us = (t_original - t_folded) * 1000
    total_savings_us = savings_per_norm_us * NUM_FOLDABLE_NORMS
    print(f"\n  Extrapolated model savings ({NUM_FOLDABLE_NORMS} norms):")
    print(f"    {total_savings_us:.1f} µs = {total_savings_us / 1000:.3f} ms per token")

    return {
        "norm_us": t_norm * 1000,
        "linear_us": t_linear * 1000,
        "original_us": t_original * 1000,
        "folded_us": t_folded * 1000,
        "savings_per_norm_us": savings_per_norm_us,
        "total_savings_us": total_savings_us,
        "norm_share_pct": t_norm / t_original * 100,
    }


# ---------------------------------------------------------------------------
# Gate 1.2: Numerical equality test (kill-switch)
# ---------------------------------------------------------------------------
def bench_numerical():
    """Verify folded path matches original to within 1e-4 relative diff."""
    print("\n" + "=" * 70)
    print("GATE 1.2: Numerical Equality Test")
    print("=" * 70)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    results = {}

    # Test multiple gamma distributions
    test_cases = {
        "near_zero": lambda d: torch.zeros(d),           # γ ≈ 0, effective scale ≈ 1
        "near_one": lambda d: torch.ones(d),              # γ ≈ 1, effective scale ≈ 2
        "random_small": lambda d: torch.randn(d) * 0.01,  # γ ≈ small noise
        "random_large": lambda d: torch.randn(d) * 0.5,   # γ ≈ large noise
        "trained_like": lambda d: torch.randn(d) * 0.1,   # realistic trained gamma
    }

    all_pass = True
    for case_name, gamma_fn in test_cases.items():
        norm = MockRMSNorm(HIDDEN, EPS).to(device, dtype)
        norm.weight.data.copy_(gamma_fn(HIDDEN).to(dtype))

        linear = nn.Linear(HIDDEN, INTERMEDIATE, bias=False).to(device, dtype)

        norm_sf = MockRMSNormScaleFree(HIDDEN, EPS).to(device, dtype)
        linear_folded = fold_weights(norm, linear).to(device, dtype)
        norm_f32 = MockRMSNorm(HIDDEN, EPS).to(device, torch.float32)
        norm_f32.weight.data.copy_(gamma_fn(HIDDEN).to(torch.float32))
        lin_f32 = nn.Linear(HIDDEN, INTERMEDIATE, bias=False, device=device, dtype=torch.float32)
        lin_f32.weight.data.copy_(linear.weight.data.float())

        norm_sf_f32 = MockRMSNormScaleFree(HIDDEN, EPS).to(device, torch.float32)
        lin_folded_f32 = fold_weights(norm_f32, lin_f32)

        # Test with multiple input distributions
        max_rel_l2 = 0.0
        max_abs_diff = 0.0
        is_allclose = True
        for _ in range(50):
            x = torch.randn(1, 1, HIDDEN, device=device, dtype=torch.float32)

            with torch.no_grad():
                y_orig = lin_f32(norm_f32(x))
                y_fold = lin_folded_f32(norm_sf_f32(x))

                diff = (y_orig - y_fold).abs()
                rel_l2 = (torch.linalg.norm(diff) / torch.linalg.norm(y_orig)).item()
                max_rel_l2 = max(max_rel_l2, rel_l2)
                max_abs_diff = max(max_abs_diff, diff.max().item())
                is_allclose = is_allclose and torch.allclose(y_orig, y_fold, rtol=1e-4, atol=1e-4)

        passed = max_rel_l2 < 1e-4 and is_allclose
        all_pass = all_pass and passed
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {case_name:20s}  rel L2 = {max_rel_l2:.2e}, max abs = {max_abs_diff:.2e}  {status}")
        results[case_name] = {"rel_l2_diff": max_rel_l2, "max_abs_diff": max_abs_diff, "passed": passed}

    # Also test multi-fan-out correctness (one norm → 4 projections)
    print(f"\n  Multi-fan-out test (1 norm → 4 projections):")
    norm = MockRMSNorm(HIDDEN, EPS).to(device, torch.float32)
    norm.weight.data.copy_((torch.randn(HIDDEN) * 0.1).to(torch.float32))

    linears = [
        nn.Linear(HIDDEN, NUM_Q_HEADS * HEAD_DIM * 2, bias=False, device=device, dtype=torch.float32),  # q_proj
        nn.Linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM, bias=False, device=device, dtype=torch.float32),      # k_proj
        nn.Linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM, bias=False, device=device, dtype=torch.float32),      # v_proj
        nn.Linear(HIDDEN, HIDDEN, bias=False, device=device, dtype=torch.float32),                        # o_proj
    ]
    proj_names = ["q_proj", "k_proj", "v_proj", "o_proj"]

    norm_sf = MockRMSNormScaleFree(HIDDEN, EPS).to(device, torch.float32)
    folded_linears = [fold_weights(norm, l) for l in linears]

    fan_pass = True
    for l_orig, l_fold, pname in zip(linears, folded_linears, proj_names):
        max_rel = 0.0
        max_abs = 0.0
        fan_close = True
        for _ in range(50):
            x = torch.randn(1, 1, HIDDEN, device=device, dtype=torch.float32)
            with torch.no_grad():
                y_orig = l_orig(norm(x))
                y_fold = l_fold(norm_sf(x))
            diff = (y_orig - y_fold).abs()
            rel = (torch.linalg.norm(diff) / torch.linalg.norm(y_orig)).item()
            max_rel = max(max_rel, rel)
            max_abs = max(max_abs, diff.max().item())
            fan_close = fan_close and torch.allclose(y_orig, y_fold, rtol=1e-4, atol=1e-4)
        ok = max_rel < 1e-4 and fan_close
        fan_pass = fan_pass and ok
        status = "✅" if ok else "❌"
        print(f"    {pname:10s}  rel L2 = {max_rel:.2e}, max abs = {max_abs:.2e}  {status}")

    all_pass = all_pass and fan_pass
    results["multi_fan_out"] = {"passed": fan_pass}

    # Also test adapter interaction correctness
    print(f"\n  Adapter delta scaling test:")
    norm = MockRMSNorm(HIDDEN, EPS).to(device, torch.float32)
    norm.weight.data.copy_((torch.randn(HIDDEN) * 0.1).to(torch.float32))
    linear_base = nn.Linear(HIDDEN, INTERMEDIATE, bias=False, device=device, dtype=torch.float32)

    # Simulate adapter: dW = scaling * U @ V
    rank = 8
    scaling = 16.0
    U = torch.randn(INTERMEDIATE, rank, device=device, dtype=torch.float32)
    V = torch.randn(rank, HIDDEN, device=device, dtype=torch.float32)

    # Original path: norm(x) @ (W + scaling * U @ V).T
    W_adapted = linear_base.weight.data + scaling * U @ V

    # Folded path: scale-free-norm(x) @ ((1+γ) * (W + scaling * U @ V)).T
    gamma_scale = (1.0 + norm.weight.float()).unsqueeze(0)  # (1, in_features)

    # CORRECT: fold γ into the ENTIRE adapted weight (or equivalently V_folded = V * gamma_scale)
    W_adapted_folded = gamma_scale * W_adapted

    # WRONG: fold γ only into base, leave adapter unscaled
    W_wrong = fold_weights(norm, linear_base).weight.data + scaling * U @ V

    norm_sf = MockRMSNormScaleFree(HIDDEN, EPS).to(device, torch.float32)
    max_rel_correct = 0.0
    max_rel_wrong = 0.0
    correct_close = True

    for _ in range(50):
        x = torch.randn(1, 1, HIDDEN, device=device, dtype=torch.float32)
        with torch.no_grad():
            normed = norm(x)
            y_orig = torch.nn.functional.linear(normed, W_adapted)

            normed_sf = norm_sf(x)
            y_correct = torch.nn.functional.linear(normed_sf, W_adapted_folded)
            y_wrong = torch.nn.functional.linear(normed_sf, W_wrong)

        diff_c = (y_orig - y_correct).abs()
        diff_w = (y_orig - y_wrong).abs()
        max_rel_correct = max(max_rel_correct, (torch.linalg.norm(diff_c) / torch.linalg.norm(y_orig)).item())
        max_rel_wrong = max(max_rel_wrong, (torch.linalg.norm(diff_w) / torch.linalg.norm(y_orig)).item())

    ok_correct = max_rel_correct < 1e-4
    print(f"    Correct (γ on base+delta):  rel L2 = {max_rel_correct:.2e}  {'✅' if ok_correct else '❌'}")
    print(f"    Wrong (γ on base only):     rel L2 = {max_rel_wrong:.2e}  {'⚠️ EXPECTED LARGE DIVERGENCE' if max_rel_wrong > 1e-4 else '❓ suspiciously small'}")

    results["adapter_correct"] = {"rel_l2_diff": max_rel_correct, "passed": ok_correct}
    results["adapter_wrong_expected_fail"] = {"rel_l2_diff": max_rel_wrong}

    all_pass = all_pass and ok_correct

    print(f"\n  Overall Gate 1.2: {'✅ PASS' if all_pass else '❌ FAIL — ABORT IMPLEMENTATION'}")
    results["overall_pass"] = all_pass
    return results


# ---------------------------------------------------------------------------
# Gate 1.3: Throughput verification
# ---------------------------------------------------------------------------
def bench_throughput():
    """Measure aggregate time for 73 norm+linear pairs vs folded equivalents."""
    print("\n" + "=" * 70)
    print("GATE 1.3: Throughput Verification")
    print("=" * 70)

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Build a chain matching the model's actual foldable norm→linear structure
    # Per layer: input_layernorm → {q,k,v,gate,up}_proj (fan-out)
    #            post_attn_layernorm → {gate,up}_proj (fan-out)
    # Plus final norm → lm_head

    # For throughput, we simplify to the dominant cost: N independent
    # norm→single-linear pairs, because each Linear is a separate kernel
    # regardless of fan-out. The fan-out just means more linears.

    # Count actual downstream linears per foldable norm:
    #   input_layernorm feeds: q_proj(2560→5120), k_proj(2560→512), v_proj(2560→512)
    #     For linear_attn layers: in_proj_qkv(2560→1536), in_proj_z(2560→512),
    #                             in_proj_b(2560→16), in_proj_a(2560→16)
    #   post_attn_layernorm feeds: gate_proj(2560→6912), up_proj(2560→6912)
    #   final norm feeds: lm_head (not a separate Linear, tied to embedding usually)

    # We measure with the dominant shapes
    shapes = [
        ("input_ln→gate_proj", HIDDEN, INTERMEDIATE),  # 36 layers × 1
        ("input_ln→up_proj", HIDDEN, INTERMEDIATE),     # 36 layers × 1
        ("post_attn→q_proj", HIDDEN, NUM_Q_HEADS * HEAD_DIM * 2),  # just representative
    ]

    # Build original and folded chains
    original_norms = []
    original_linears = []
    folded_norms = []
    folded_linears = []

    for _, in_f, out_f in shapes:
        for _ in range(NUM_LAYERS):
            n = MockRMSNorm(in_f, EPS).to(device, dtype)
            n.weight.data.copy_((torch.randn(in_f) * 0.1).to(dtype))
            l = nn.Linear(in_f, out_f, bias=False).to(device, dtype)
            original_norms.append(n)
            original_linears.append(l)

            n_sf = MockRMSNormScaleFree(in_f, EPS).to(device, dtype)
            l_f = fold_weights(n, l).to(device, dtype)
            folded_norms.append(n_sf)
            folded_linears.append(l_f)

    x = torch.randn(1, 1, HIDDEN, device=device, dtype=dtype)

    def run_original():
        for n, l in zip(original_norms, original_linears):
            l(n(x))

    def run_folded():
        for n, l in zip(folded_norms, folded_linears):
            l(n(x))

    t_orig = cuda_event_time_ms(run_original, warmup=50, iters=200)
    t_fold = cuda_event_time_ms(run_folded, warmup=50, iters=200)

    savings_pct = (t_orig - t_fold) / t_orig * 100
    passed = savings_pct > 2.0

    print(f"  {len(original_norms)} norm+linear pairs ({NUM_LAYERS} layers × {len(shapes)} shapes)")
    print(f"  Original chain:   {t_orig:.3f} ms per forward")
    print(f"  Folded chain:     {t_fold:.3f} ms per forward")
    print(f"  Savings:          {savings_pct:.2f}%")
    print(f"  Absolute savings: {(t_orig - t_fold) * 1000:.1f} µs per token")
    print(f"\n  Gate 1.3: {'✅ PASS' if passed else '❌ FAIL — savings < 2%, ABORT'}")

    return {
        "num_pairs": len(original_norms),
        "original_ms": t_orig,
        "folded_ms": t_fold,
        "savings_pct": savings_pct,
        "savings_us": (t_orig - t_fold) * 1000,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_hard_vram_cap(22.0)
    print("FlashNorm Weight Folding Benchmark")
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Hidden: {HIDDEN}, Intermediate: {INTERMEDIATE}")
    print(f"Foldable norms: {NUM_FOLDABLE_NORMS}")

    r1 = bench_latency()
    r2 = bench_numerical()
    r3 = bench_throughput()

    all_pass = r2["overall_pass"] and r3["passed"]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Gate 1.1 (Latency):     {r1['savings_per_norm_us']:.2f} µs/norm saved")
    print(f"  Gate 1.2 (Numerical):   {'✅ PASS' if r2['overall_pass'] else '❌ FAIL'}")
    print(f"  Gate 1.3 (Throughput):  {'✅ PASS' if r3['passed'] else '❌ FAIL'} ({r3['savings_pct']:.2f}% savings)")
    print(f"\n  PROCEED TO PHASE 2: {'✅ YES' if all_pass else '❌ NO — DO NOT IMPLEMENT'}")

    # JSON output for programmatic consumption
    results = {
        "gate_1_1_latency": r1,
        "gate_1_2_numerical": r2,
        "gate_1_3_throughput": r3,
        "all_gates_passed": all_pass,
    }
    print(f"\n--- JSON ---\n{json.dumps(results, indent=2, default=str)}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
