"""Unit Tests for W4A16 Model Loader & 27B/32B Layer Execution on RDNA3 WMMA."""

import pytest
import torch
import torch.nn as nn
from w4a16_loader import W4A16Linear, W4A16ModelLoader


def test_w4a16_linear_numerical_fidelity():
    """Verify W4A16Linear module forward pass against unquantized nn.Linear."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("W4A16 Triton kernels require CUDA/HIP GPU device")

    torch.manual_seed(42)
    in_features = 2560
    out_features = 2560
    group_size = 128

    orig_linear = nn.Linear(in_features, out_features, bias=False, device=device, dtype=torch.bfloat16)
    w4_linear = W4A16Linear.from_linear(orig_linear, group_size=group_size, device=device)

    assert w4_linear.qweight.shape == (in_features // 8, out_features)
    assert w4_linear.scales.shape == (in_features // group_size, out_features)

    # Forward with single token (M=1)
    x = torch.randn((1, in_features), dtype=torch.bfloat16, device=device)
    ref_out = orig_linear(x)
    w4_out = w4_linear(x)

    assert w4_out.shape == (1, out_features)
    assert w4_out.dtype == torch.bfloat16

    # INT4 cosine similarity should exceed 0.98 vs unquantized BF16
    cos_sim = torch.cosine_similarity(ref_out.float().flatten(), w4_out.float().flatten(), dim=0).item()
    assert cos_sim > 0.98, f"Cosine similarity {cos_sim:.4f} is too low"


def test_w4a16_27b_geometry_forward():
    """Verify single forward pass through full 27B/32B attention and MLP layer dimensions."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("W4A16 Triton kernels require CUDA/HIP GPU device")

    torch.manual_seed(42)
    D = 5120  # 27B/32B Hidden dimension
    H_kv = 1024  # 8 KV heads * 128 head dim
    H_mlp = 27392  # 27B/32B Intermediate MLP dimension
    group_size = 128

    # 1. QKV Projections
    q_proj = W4A16Linear(D, D, group_size=group_size, device=device)
    k_proj = W4A16Linear(D, H_kv, group_size=group_size, device=device)
    v_proj = W4A16Linear(D, H_kv, group_size=group_size, device=device)
    o_proj = W4A16Linear(D, D, group_size=group_size, device=device)

    # 2. MLP Projections
    gate_proj = W4A16Linear(D, H_mlp, group_size=group_size, device=device)
    up_proj = W4A16Linear(D, H_mlp, group_size=group_size, device=device)
    down_proj = W4A16Linear(H_mlp, D, group_size=group_size, device=device)

    # Populate dummy quantized weights
    for mod in [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]:
        mod.qweight.random_(-1000, 1000)
        mod.scales.fill_(0.02)

    # Simulate 1 decode token (M=1)
    x = torch.randn((1, D), dtype=torch.bfloat16, device=device)

    # Attention forward
    q = q_proj(x)
    k = k_proj(x)
    v = v_proj(x)
    attn_out = o_proj(q)
    assert attn_out.shape == (1, D)

    # MLP forward
    g = gate_proj(x)
    u = up_proj(x)
    act = g * u  # SwiGLU activation representation
    mlp_out = down_proj(act)
    assert mlp_out.shape == (1, D)


def test_w4a16_fused_lora_dynamic_attachment():
    """Verify setting, executing, and clearing dynamic LoRA adapters on W4A16Linear."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("W4A16 Triton kernels require CUDA/HIP GPU device")

    torch.manual_seed(42)
    in_features = 2560
    out_features = 2560
    rank = 16
    group_size = 128

    mod = W4A16Linear(in_features, out_features, group_size=group_size, device=device)
    mod.qweight.random_(-1000, 1000)
    mod.scales.fill_(0.02)

    x = torch.randn((1, in_features), dtype=torch.bfloat16, device=device)

    # 1. Base forward (no LoRA)
    base_out = mod(x)

    # 2. Attach LoRA adapter
    lora_a = torch.randn((in_features, rank), dtype=torch.bfloat16, device=device) * 0.1
    lora_b = torch.randn((rank, out_features), dtype=torch.bfloat16, device=device) * 0.1
    mod.set_lora_adapter(lora_a, lora_b, alpha=2.0)

    lora_out = mod(x)
    assert lora_out.shape == (1, out_features)
    # LoRA output must differ from base output
    diff = (lora_out - base_out).abs().max().item()
    assert diff > 1e-3, "LoRA branch did not modify output"

    # 3. Clear LoRA adapter
    mod.clear_lora()
    cleared_out = mod(x)
    assert torch.allclose(cleared_out, base_out, atol=1e-5), "Cleared module does not match base forward"


def test_replace_linear_modules_recursive():
    """Verify recursive module replacement in a multi-layer container."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        pytest.skip("W4A16 Triton kernels require CUDA/HIP GPU device")

    class DummyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(512, 512, bias=False, device=device, dtype=torch.bfloat16)
            self.k = nn.Linear(512, 512, bias=False, device=device, dtype=torch.bfloat16)
            self.lm_head = nn.Linear(512, 1000, bias=False, device=device, dtype=torch.bfloat16)

    model = nn.Sequential(DummyBlock(), DummyBlock())
    summary = W4A16ModelLoader.replace_linear_modules(model, group_size=128, skip_modules=["lm_head"])

    assert summary.total_linear_layers == 4  # 2 q + 2 k (lm_head skipped)
    assert summary.compression_ratio > 3.0
    assert isinstance(model[0].q, W4A16Linear)
    assert isinstance(model[0].k, W4A16Linear)
    assert isinstance(model[0].lm_head, nn.Linear)  # lm_head preserved


def test_27b_memory_map_calculation():
    """Verify VRAM memory map calculation for 27B/32B model on 24 GB card."""
    mem_map = W4A16ModelLoader.calculate_27b_32b_memory_map(
        num_layers=64,
        hidden_size=5120,
        intermediate_size=27392,
        num_kv_heads=8,
        head_dim=128,
        max_seq_len=4096,
    )

    assert mem_map["fits_in_24gb_vram"] is True
    assert mem_map["w4a16_weight_gb"] < 17.5
    assert mem_map["static_kv_cache_gb"] < 3.0
    assert mem_map["total_w4_footprint_gb"] < 21.0
    assert mem_map["headroom_vram_gb"] > 3.0
