import asyncio
import json
import threading
import torch
import pytest
from unittest.mock import MagicMock, patch
from runtime.bucketed_speculative import BucketedSpeculativeDecoder


def test_speculative_stream_generate_contract():
    """Verify that BucketedSpeculativeDecoder.generate delegates cleanly to stream_generate."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 151643
    mock_draft_head = MagicMock()

    decoder = BucketedSpeculativeDecoder(
        model=mock_model,
        tokenizer=mock_tokenizer,
        draft_head=mock_draft_head,
        k=4,
        max_seq_len=512,
        device="cpu",
    )

    # Mock stream_generate to yield 3 token batches
    mock_batches = [[101, 102], [103, 104, 105], [106]]
    
    with patch.object(decoder, "stream_generate", return_value=iter(mock_batches)):
        dummy_prompt = torch.tensor([[1, 2, 3]])
        toks, elapsed, st = decoder.generate(dummy_prompt, max_new_tokens=10)
        
        assert toks == [101, 102, 103, 104, 105, 106]
        assert isinstance(st, dict)


@pytest.mark.asyncio
async def test_server_streaming_sse_generator():
    """Test that _build_streaming_response properly handles speculative token batches."""
    from runtime import server
    from runtime.server import _build_streaming_response, ChatCompletionRequest, ChatMessage

    # Setup mocked server state
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 151643
    mock_tokenizer.decode.side_effect = lambda tok_list, **kw: f" tok_{tok_list[0]}"
    
    mock_spec_decoder = MagicMock()
    mock_spec_decoder._locked = True
    mock_spec_decoder.stream_generate.return_value = iter([[101, 102], [103], [104, 105]])

    server.model_state["base_model"] = MagicMock()
    server.model_state["tokenizer"] = mock_tokenizer
    server.model_state["folding_engine"] = MagicMock()
    server.model_state["spec_decoder"] = mock_spec_decoder
    server.model_state["graph_decoder"] = None
    server.model_state["stop_token_ids"] = {151643}
    server.model_state["active_team"] = ["astral"]
    server.model_state["causal_scheduler"] = None
    server.model_state["stop_event"] = threading.Event()

    req = ChatCompletionRequest(
        model="dynamic",
        messages=[ChatMessage(role="user", content="Hello")],
        stream=True,
    )

    qr = MagicMock()
    qr.expert = "dynamic"
    prompt_tokens = torch.tensor([[1, 2, 3]])
    resp = _build_streaming_response(
        req=req,
        expert=None,
        prompt_tokens=prompt_tokens,
        max_new_tokens=32,
        wait_ms=0.5,
        queue_position=0,
        qr=qr,
    )

    chunks = []
    async for chunk_str in resp.body_iterator:
        chunks.append(chunk_str)

    assert len(chunks) >= 3
    assert "data: " in chunks[0]
    assert "[DONE]" in chunks[-1]

    # Verify that spec_decoder.stream_generate was called with engine=None so morphed weights were not reset
    mock_spec_decoder.stream_generate.assert_called_once()
    _, call_kwargs = mock_spec_decoder.stream_generate.call_args
    assert call_kwargs.get("engine") is None
