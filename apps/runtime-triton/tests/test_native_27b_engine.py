import os

import pytest
import torch
import torch.nn.functional as F
from gguf_unpacker import DEFAULT_GGUF_PATH, GGUFStreamingUnpacker
from native_27b_engine import (
    PreallocatedKVCache,
    Qwen35FullAttentionBlock,
    Qwen35SSMBlock,
    RMSNorm,
)


def test_rmsnorm():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    norm = RMSNorm(5120, device=dev)
    x = torch.randn(1, 5120, dtype=torch.bfloat16, device=dev)
    out = norm(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_ssm_block_forward():
    if not os.path.exists(DEFAULT_GGUF_PATH):
        pytest.skip("GGUF file not found")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")

    unpacker = GGUFStreamingUnpacker(DEFAULT_GGUF_PATH)
    l0_dict = unpacker.unpack_layer(0, group_size=128, device="cuda:0")

    dev = torch.device("cuda:0")
    block = Qwen35SSMBlock(0, device=dev)
    block.load_weights(l0_dict)

    x = torch.randn(1, 5120, dtype=torch.bfloat16, device=dev)
    out, ssm_s, conv_s = block(x)

    assert out.shape == (1, 5120)
    assert ssm_s.shape == (48, 128, 128)
    assert not torch.isnan(out).any()


def test_attention_block_forward():
    if not os.path.exists(DEFAULT_GGUF_PATH):
        pytest.skip("GGUF file not found")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")

    unpacker = GGUFStreamingUnpacker(DEFAULT_GGUF_PATH)
    l3_dict = unpacker.unpack_layer(3, group_size=128, device="cuda:0")

    dev = torch.device("cuda:0")
    block = Qwen35FullAttentionBlock(3, device=dev)
    block.load_weights(l3_dict)

    x = torch.randn(1, 1, 5120, dtype=torch.bfloat16, device=dev)
    out, kv_cache = block(x)

    assert out.shape == (1, 1, 5120)
    assert kv_cache[0].shape[1] == 4  # 4 KV heads
    assert not torch.isnan(out).any()


def test_preallocated_kv_cache_bf16():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cache = PreallocatedKVCache(batch_size=1, num_heads=4, head_dim=256, max_seq_len=128, mode="bf16", device=dev)
    assert cache.current_len == 0

    k_new = torch.randn(1, 4, 1, 256, dtype=torch.bfloat16, device=dev)
    v_new = torch.randn(1, 4, 1, 256, dtype=torch.bfloat16, device=dev)
    k_out, v_out = cache.update(k_new, v_new)

    assert cache.current_len == 1
    assert k_out.shape == (1, 4, 1, 256)
    assert v_out.shape == (1, 4, 1, 256)
    assert torch.equal(cache[0], k_out)
    assert torch.equal(cache[1], v_out)

    cache.reset()
    assert cache.current_len == 0


def test_preallocated_kv_cache_q8():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cache_bf16 = PreallocatedKVCache(batch_size=1, num_heads=4, head_dim=256, max_seq_len=128, mode="bf16", device=dev)
    cache_q8 = PreallocatedKVCache(batch_size=1, num_heads=4, head_dim=256, max_seq_len=128, mode="q8_0", device=dev)

    # Memory check: Q8 should take approx 50% of BF16 memory
    mem_bf16 = cache_bf16.get_memory_bytes()
    mem_q8 = cache_q8.get_memory_bytes()
    assert mem_q8 < mem_bf16 * 0.55

    # Quantization fidelity check
    torch.manual_seed(42)
    k_new = torch.randn(1, 4, 4, 256, dtype=torch.bfloat16, device=dev)
    v_new = torch.randn(1, 4, 4, 256, dtype=torch.bfloat16, device=dev)

    k_deq, v_deq = cache_q8.update(k_new, v_new)
    assert cache_q8.current_len == 4
    assert k_deq.shape == (1, 4, 4, 256)
    assert not torch.isnan(k_deq).any()

    # Cosine similarity between original and dequantized
    sim_k = F.cosine_similarity(k_new.view(-1).float(), k_deq.view(-1).float(), dim=0).item()
    sim_v = F.cosine_similarity(v_new.view(-1).float(), v_deq.view(-1).float(), dim=0).item()
    assert sim_k > 0.99
    assert sim_v > 0.99


def test_attention_block_with_preallocated_caches():
    if not os.path.exists(DEFAULT_GGUF_PATH):
        pytest.skip("GGUF file not found")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")

    unpacker = GGUFStreamingUnpacker(DEFAULT_GGUF_PATH)
    l3_dict = unpacker.unpack_layer(3, group_size=128, device="cuda:0")

    dev = torch.device("cuda:0")
    block = Qwen35FullAttentionBlock(3, device=dev)
    block.load_weights(l3_dict)

    # Test BF16 Preallocated Cache over 3 steps
    cache_bf16 = PreallocatedKVCache(batch_size=1, num_heads=4, head_dim=256, max_seq_len=16, mode="bf16", device=dev)
    for _ in range(3):
        x = torch.randn(1, 1, 5120, dtype=torch.bfloat16, device=dev)
        out, cache = block(x, kv_cache=cache_bf16)
        assert out.shape == (1, 1, 5120)
        assert not torch.isnan(out).any()
    assert cache_bf16.current_len == 3

    # Test Q8_0 Preallocated Cache over 3 steps
    cache_q8 = PreallocatedKVCache(batch_size=1, num_heads=4, head_dim=256, max_seq_len=16, mode="q8_0", device=dev)
    for _ in range(3):
        x = torch.randn(1, 1, 5120, dtype=torch.bfloat16, device=dev)
        out, cache = block(x, kv_cache=cache_q8)
        assert out.shape == (1, 1, 5120)
        assert not torch.isnan(out).any()
    assert cache_q8.current_len == 3


def test_ssm_chunk_graph():
    if not os.path.exists(DEFAULT_GGUF_PATH):
        pytest.skip("GGUF file not found")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")

    from native_27b_engine import SSMChunkGraph

    dev = torch.device("cuda:0")

    unpacker = GGUFStreamingUnpacker(DEFAULT_GGUF_PATH)
    layers = [Qwen35SSMBlock(i, device=dev) for i in range(3)]
    for i, l in enumerate(layers):
        l.load_weights(unpacker.unpack_layer(i, group_size=128, device="cuda:0"))

    chunk = SSMChunkGraph(layers, dev)
    x = torch.randn(1, 5120, dtype=torch.bfloat16, device=dev)
    out = chunk.replay(x)

    assert out.shape == (1, 5120)
    assert not torch.isnan(out).any()

    # Reset test
    chunk.reset_states()
    for ssm, conv in zip(chunk.ssm_states, chunk.conv_states):
        assert (ssm == 0).all()
        assert (conv == 0).all()


def test_batched_prefill():
    if not os.path.exists(DEFAULT_GGUF_PATH):
        pytest.skip("GGUF file not found")
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm not available")

    dev = torch.device("cuda:0")
    unpacker = GGUFStreamingUnpacker(DEFAULT_GGUF_PATH)

    # 1. Test SSM Block with Sequence Length 8
    l0_dict = unpacker.unpack_layer(0, group_size=128, device="cuda:0")
    ssm = Qwen35SSMBlock(0, device=dev)
    ssm.load_weights(l0_dict)

    x_seq = torch.randn(1, 8, 5120, dtype=torch.bfloat16, device=dev)
    out_ssm, ssm_state, conv_state = ssm(x_seq)

    assert out_ssm.shape == (1, 8, 5120)
    assert ssm_state.shape == (48, 128, 128)
    assert conv_state.shape == (10240, 3)
    assert not torch.isnan(out_ssm).any()

    # 2. Test Attention Block with Sequence Length 8
    l3_dict = unpacker.unpack_layer(3, group_size=128, device="cuda:0")
    attn = Qwen35FullAttentionBlock(3, device=dev)
    attn.load_weights(l3_dict)

    kv_cache = PreallocatedKVCache(batch_size=1, num_heads=4, head_dim=256, max_seq_len=64, mode="bf16", device=dev)
    out_attn, kv_out = attn(x_seq, kv_cache=kv_cache)

    assert out_attn.shape == (1, 8, 5120)
    assert kv_cache.current_len == 8
    assert not torch.isnan(out_attn).any()
