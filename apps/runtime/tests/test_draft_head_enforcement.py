"""Unit tests verifying strict model-architecture compatibility enforcement for speculative draft heads."""

import pytest
import torch
import torch.nn as nn
from runtime.bucketed_speculative import BucketedSpeculativeDecoder
from runtime.mtp_draft import IncompatibleDraftHeadError
from transformers import AutoConfig


class MockQwenModel(nn.Module):
    def __init__(self, hidden_size: int = 2560):
        super().__init__()
        self.config = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B")
        self.config.hidden_size = hidden_size
        self.config.text_config.hidden_size = hidden_size
        self.dummy_param = nn.Parameter(torch.zeros(1))


class MockDraftHead(nn.Module):
    def __init__(self, model_id: str, hidden_size: int):
        super().__init__()
        self.target_model_id = model_id
        self.target_hidden_size = hidden_size


class MockTokenizer:
    pad_token_id = 0
    eos_token_id = 151643


def test_mismatched_draft_head_raises_incompatible_error():
    """Attempting to pair a 4B draft head (dim=2560) with a 9B model (dim=4096) must raise IncompatibleDraftHeadError."""
    base_9b = MockQwenModel(hidden_size=4096)
    head_4b = MockDraftHead(model_id="Qwen/Qwen3.5-4B", hidden_size=2560)
    tokenizer = MockTokenizer()

    with pytest.raises(IncompatibleDraftHeadError) as exc_info:
        BucketedSpeculativeDecoder(
            model=base_9b,
            tokenizer=tokenizer,
            draft_head=head_4b,
            k=2,
        )

    assert "Speculative draft head incompatible" in str(exc_info.value)
    assert "2560" in str(exc_info.value)
    assert "4096" in str(exc_info.value)


def test_matching_draft_head_initialization_succeeds():
    """Pairing matching 9B draft head (dim=4096) with 9B model (dim=4096) succeeds."""
    base_9b = MockQwenModel(hidden_size=4096)
    head_9b = MockDraftHead(model_id="Qwen/Qwen3.5-9B", hidden_size=4096)
    tokenizer = MockTokenizer()

    decoder = BucketedSpeculativeDecoder(
        model=base_9b,
        tokenizer=tokenizer,
        draft_head=head_9b,
        k=2,
    )
    assert decoder.head.target_hidden_size == 4096
    assert decoder.k == 2
