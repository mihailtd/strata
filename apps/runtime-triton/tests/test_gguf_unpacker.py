import os
import tempfile

import pytest
from gguf_unpacker import DEFAULT_GGUF_PATH, GGUFStreamingUnpacker


def test_gguf_unpacker_metadata():
    if not os.path.exists(DEFAULT_GGUF_PATH):
        pytest.skip("GGUF file not present on this machine")

    unpacker = GGUFStreamingUnpacker(DEFAULT_GGUF_PATH)
    meta = unpacker.get_metadata()
    assert meta["num_layers"] >= 64
    assert meta["hidden_dim"] == 5120
    assert meta["ffn_dim"] == 17408
    assert meta["full_attention_interval"] == 4


def test_gguf_unpack_single_layer():
    if not os.path.exists(DEFAULT_GGUF_PATH):
        pytest.skip("GGUF file not present on this machine")

    with tempfile.TemporaryDirectory() as tmpdir:
        unpacker = GGUFStreamingUnpacker(DEFAULT_GGUF_PATH, cache_dir=tmpdir)
        # Unpack SSM block 0
        l0 = unpacker.unpack_layer(0, group_size=128, device="cpu")
        assert "ffn_gate.weight" in l0
        assert l0["ffn_gate.weight"]["type"] == "w4a16"
        assert l0["ffn_gate.weight"]["qweight"].shape == (5120 // 8, 17408)
        assert l0["ffn_gate.weight"]["scales"].shape == (5120 // 128, 17408)
        assert "ssm_a" in l0
        assert l0["ssm_a"]["type"] == "dense"

        # Unpack Full Attention block 3
        l3 = unpacker.unpack_layer(3, group_size=128, device="cpu")
        assert "attn_q.weight" in l3
        assert "attn_k.weight" in l3
        assert "attn_v.weight" in l3
        assert "attn_output.weight" in l3
        assert l3["attn_q.weight"]["type"] == "w4a16"
