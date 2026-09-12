"""Context scrubbing: <think>...</think> blocks must not leak into prompt history.

Split out of the former tests/test_category1_hygiene.py (Action 3.1) -- the
KV-cache-precision and deterministic-attention sections of that file were
redundant with apps/runtime-common/tests/test_canon.py and are not repeated
here.
"""

from __future__ import annotations

from runtime.server import (
    ChatMessage,
    extract_thinking_and_content,
    format_prompt,
    scrub_thinking_blocks,
)


def test_scrub_thinking_blocks_clean_tags():
    """Verify standard <think>...</think> blocks are completely stripped."""
    raw = "<think>\nLet me analyze the SQL schema and indexes.\nWe need pgvector.\n</think>\nSELECT * FROM embeddings;"
    expected = "SELECT * FROM embeddings;"
    assert scrub_thinking_blocks(raw) == expected


def test_scrub_thinking_blocks_multiline_and_code():
    """Verify multiline reasoning containing code, quotes, and markdown is cleanly stripped."""
    raw = """<think>
```python
def draft():
    return "test"
```
The user wants an async endpoint with `@app.get("/items")`.
</think>
from fastapi import FastAPI
app = FastAPI()"""
    expected = "from fastapi import FastAPI\napp = FastAPI()"
    assert scrub_thinking_blocks(raw) == expected


def test_scrub_thinking_blocks_unclosed_and_orphaned():
    """Verify unclosed or orphaned think tags do not corrupt text."""
    assert scrub_thinking_blocks("<think>partial reasoning") == "partial reasoning"
    assert scrub_thinking_blocks("answer text</think>") == "answer text"
    assert scrub_thinking_blocks("plain content without think") == "plain content without think"
    assert scrub_thinking_blocks("") == ""


def test_format_prompt_scrubs_prior_assistant_turns():
    """Verify format_prompt removes reasoning traces from historical assistant messages."""
    messages = [
        ChatMessage(role="user", content="Step 1: create table"),
        ChatMessage(
            role="assistant",
            content="<think>\nNeed uuid and vector columns.\n</think>\nCREATE TABLE items (id UUID, emb VECTOR(1536));",
        ),
        ChatMessage(role="user", content="Step 2: add query"),
    ]

    formatted = format_prompt(messages, thinking_effort="low")

    # Historical assistant turn MUST NOT contain historical thinking scratchpad
    assert "Need uuid and vector columns" not in formatted
    assert "CREATE TABLE items (id UUID, emb VECTOR(1536));" in formatted

    # Historical user turns must remain intact
    assert "Step 1: create table" in formatted
    assert "Step 2: add query" in formatted

    # The prompt initiates a new assistant turn with fresh <think> start
    assert formatted.endswith("<|im_start|>assistant\n<think>\n")


def test_extract_thinking_and_content_separation():
    """Verify extract_thinking_and_content splits reasoning from final text."""
    text = "<think>\nComputing cosine distance metric\n</think>\nORDER BY emb <=> target LIMIT 10;"
    reasoning, content = extract_thinking_and_content(text)
    assert reasoning == "Computing cosine distance metric"
    assert content == "ORDER BY emb <=> target LIMIT 10;"
