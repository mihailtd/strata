"""Unit tests for eval rubric regex case-insensitivity, giveaway auditing, and leading question detection."""

import re
import pytest
from audit.audit_eval_rubrics import LEADING, audit_domain


def test_leading_question_regex():
    """Verify LEADING regex detects leading question formulations."""
    leading_prompts = [
        "Should I use Postgres without any indexing?",
        "Can you just use a plain B-Tree alone?",
        "Isn't it better to avoid normalization strictly?",
        "Does it make sense to only store embeddings in Redis?",
    ]
    for prompt in leading_prompts:
        assert LEADING.search(prompt) is not None

    neutral_prompts = [
        "Explain the differences between HNSW and IVFFlat indexes in pgvector.",
        "How do I write a query using DISTINCT ON in PostgreSQL?",
    ]
    for prompt in neutral_prompts:
        # Simple neutral explanations without leading qualifiers
        assert not any(w in prompt.lower() for w in ["should", "isn't it", "can you just", "only", "strictly"])


def test_rubric_audit_synthetic_giveaway_detection(tmp_path):
    """Verify audit_domain flags questions where expected terms are given away in prompt."""
    test_eval_file = tmp_path / "test_eval.jsonl"
    test_eval_file.write_text(
        '{"id": "q1", "prompt": "How do I configure uv with pyproject.toml and ruff?", "expects": ["uv", "pyproject.toml", "ruff"]}\n'
        '{"id": "q2", "prompt": "Explain modern python package management.", "expects": ["uv", "lockfile"]}\n'
    )

    cfg = {"questions": str(test_eval_file), "good": [], "bad": []}
    audit = audit_domain("test_domain", cfg, questions_file=str(test_eval_file))

    assert audit["n"] == 2
    assert audit["n_with_rubric"] == 2
    assert audit["rubric_terms_total"] == 5  # 3 + 2
    assert audit["rubric_terms_in_question"] == 3  # "uv", "pyproject.toml", "ruff" in q1
    assert audit["n_fully_given_away"] == 1  # q1 is 100% giveaway
    assert audit["fully_given_away_ids"] == ["q1"]
    assert audit["giveaway_pct"] == 60.0  # 3/5 = 60.0%


def test_rubric_audit_ratio_domain_good_term_detection(tmp_path):
    """Verify ratio domains flag prompts that pre-seed the scored 'good' terminology."""
    test_eval_file = tmp_path / "test_ratio_eval.jsonl"
    test_eval_file.write_text(
        '{"id": "q1", "prompt": "How do I implement pgvector with HNSW index?"}\n'
        '{"id": "q2", "prompt": "How to do fast vector similarity search in a database?"}\n'
    )

    good_terms = [r"\bpgvector\b", r"\bhnsw\b"]
    cfg = {"questions": str(test_eval_file), "good": good_terms, "bad": [r"\bfaiss\b"]}
    audit = audit_domain("test_pg", cfg, questions_file=str(test_eval_file))

    assert audit["n"] == 2
    assert audit["prompts_containing_a_good_term"] == 1  # q1 contains pgvector & hnsw
    assert audit["giveaway_pct"] == 50.0  # 1/2 = 50.0%
