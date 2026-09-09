"""Unit tests for Native Speculative Decoding (NGramDrafter, SSMChunkVerifyGraph, MTP)."""

import pytest
import torch
from pathlib import Path
from runtime.native_27b_engine import (
    NGramDrafter,
    Native27BEngine,
    SSMChunkVerifyGraph,
    Qwen35SSMBlock,
)
from runtime.gguf_unpacker import DEFAULT_CACHE_DIR


def test_ngram_drafter():
    drafter = NGramDrafter(max_n=3, min_n=2, k=2)

    # Context with repeating n-gram [10, 20] -> followed by [30, 40]
    tokens = [1, 2, 10, 20, 30, 40, 5, 6, 10, 20]
    draft = drafter.find_draft(tokens)
    assert draft == [30, 40], f"Expected draft [30, 40], got {draft}"

    # Context without match
    no_match_tokens = [1, 2, 3, 4, 5, 6]
    draft_none = drafter.find_draft(no_match_tokens)
    assert draft_none == [], f"Expected empty draft, got {draft_none}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA / ROCm GPU")
def test_ssm_chunk_verify_graph():
    device = torch.device("cuda:0")
    cache_dir = Path(DEFAULT_CACHE_DIR)
    if not (cache_dir / "layer_0.pt").exists():
        pytest.skip("Layer cache not available")

    layers = [Qwen35SSMBlock(layer_idx=i, device=device) for i in range(3)]
    for i, l in enumerate(layers):
        l_data = torch.load(cache_dir / f"layer_{i}.pt", map_location=str(device), weights_only=False)
        l.load_weights(l_data)
        l.eval()

    verify_graph = SSMChunkVerifyGraph(layers, k=2, device=device)
    x = torch.randn(1, 2, 5120, dtype=torch.bfloat16, device=device)
    init_ssms = [torch.zeros(48, 128, 128, dtype=torch.float32, device=device) for _ in range(3)]
    init_convs = [torch.zeros(10240, 3, dtype=torch.bfloat16, device=device) for _ in range(3)]

    out, ssm_hist, conv_hist = verify_graph.replay(x, init_ssms, init_convs)
    assert out.shape == (1, 2, 5120)
    assert len(ssm_hist) == 3
    assert ssm_hist[0].shape == (2, 48, 128, 128)
    assert conv_hist[0].shape == (2, 10240, 3)
