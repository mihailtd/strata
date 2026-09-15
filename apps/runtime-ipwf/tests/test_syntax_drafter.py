"""Unit tests for Deterministic AST & Syntax Fast-Forwarding Trie Drafter in runtime-ipwf."""

from __future__ import annotations

from pathlib import Path

import pytest
from syntax_drafter import DEFAULT_SYNTAX_MACROS, SyntaxTrieDrafter
from transformers import AutoTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    snap = list(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*"))
    if snap:
        return AutoTokenizer.from_pretrained(str(snap[0]), trust_remote_code=True)
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)


@pytest.fixture(scope="module")
def drafter(tokenizer):
    return SyntaxTrieDrafter(tokenizer)


def test_trie_compilation(drafter):
    """Verifies that all default macros compile into the Trie."""
    assert drafter.macro_count >= len(DEFAULT_SYNTAX_MACROS)
    assert len(drafter.root.children) > 0


def test_trie_canonical_lookups(drafter, tokenizer):
    """Verifies that canonical triggers match their expected continuation."""
    test_cases = [
        ("if __name__ == ", '"__main__":\n    '),
        ("from fastapi import ", "FastAPI, Depends,"),
        ("from pydantic import ", "BaseModel, Field,"),
        ("import pytest\n", "import asyncio\n"),
        ("CREATE EXTENSION IF NOT EXISTS ", '"uuid-ossp";\n'),
        ("QUALIFY ROW_NUMBER() OVER (", "PARTITION BY "),
    ]

    for trigger, expected_substr in test_cases:
        tokens = tokenizer.encode(trigger, add_special_tokens=False)
        draft = drafter.find_draft(tokens, max_k=4)
        assert len(draft) > 0, f"Failed to match trigger: {trigger!r}"
        decoded = tokenizer.decode(draft)
        assert expected_substr[:8] in decoded, f"Mismatch for {trigger!r}: decoded {decoded!r}"


def test_trie_bpe_indentation_robustness(drafter, tokenizer):
    """Verifies that 4-space indented code triggers match correctly."""
    indented_cases = [
        ("    def __init__(self", ", "),
        ("    async with pool.acquire() as ", "conn:\n        "),
        ("    except Exception as ", "e:\n    "),
    ]

    for trigger, expected_substr in indented_cases:
        tokens = tokenizer.encode(trigger, add_special_tokens=False)
        draft = drafter.find_draft(tokens, max_k=4)
        assert len(draft) > 0, f"Failed to match indented trigger: {trigger!r}"
        decoded = tokenizer.decode(draft)
        assert expected_substr[:4] in decoded or expected_substr in decoded, (
            f"Mismatch for {trigger!r}: decoded {decoded!r}"
        )


def test_trie_unknown_suffix_returns_empty(drafter, tokenizer):
    """Verifies that non-macro code returns empty draft without false positives."""
    tokens = tokenizer.encode("some_arbitrary_variable_name = 42 + 99", add_special_tokens=False)
    draft = drafter.find_draft(tokens, max_k=4)
    assert draft == [], f"Expected empty draft for unknown code, got {draft}"


def test_dynamic_regex_fallback(drafter, tokenizer):
    """Verifies that dynamic patterns like class declarations trigger properly."""
    tokens = tokenizer.encode("class CustomerAccount(BaseModel):\n", add_special_tokens=False)
    draft = drafter.find_draft(tokens, max_k=4)
    assert len(draft) > 0
    decoded = tokenizer.decode(draft)
    assert "model_config" in decoded or "ConfigDict" in decoded
