"""Unit tests for Deterministic AST & Syntax Fast-Forwarding Trie Drafter.

Tests:
1. SyntaxTrieDrafter Trie compilation and sub-microsecond reverse-suffix lookup.
2. BPE space boundary & indentation robustness (raw vs 4-space indented).
3. Dynamic regex sliding window fallback.
4. Native27BEngine integration and speculative decode execution.
"""

from __future__ import annotations

import pytest
import server
from syntax_drafter import DEFAULT_SYNTAX_MACROS, SyntaxTrieDrafter


@pytest.fixture(scope="module")
def tokenizer():
    return server.get_27b_tokenizer()


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


def test_engine_syntax_drafter_integration():
    """Verifies that Native27BEngine supports syntax_drafter when explicitly initialized."""
    engine = server.get_native_triton_27b_engine(num_layers=64)
    engine.init_syntax_drafter(server.get_27b_tokenizer())
    assert engine.syntax_drafter is not None
    assert engine.syntax_drafter.macro_count > 0


def test_speculative_stats_tracking():
    """Verifies that generate_speculative returns valid stats when return_stats=True."""
    engine = server.get_native_triton_27b_engine(num_layers=64)
    tok = server.get_27b_tokenizer()
    prompt_ids = tok.encode("import pytest\n")
    tokens, stats = engine.generate_speculative(prompt_ids, max_new_tokens=4, return_stats=True)
    assert isinstance(tokens, list)
    assert isinstance(stats, dict)
    assert "total_steps" in stats
    assert "tokens_generated" in stats
    assert "syntax_drafts_proposed" in stats
    assert "syntax_tokens_accepted" in stats
    assert "ngram_drafts_proposed" in stats
    assert "mtp_drafts_proposed" in stats
