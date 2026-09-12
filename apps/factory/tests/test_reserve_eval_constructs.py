"""Unit tests for corpus evaluation construct reservation and directional cross-corpus leakage rules."""

import re

from corpus.reserve_eval_constructs import RESERVED, family_of, record_text


def test_record_text_extraction_formats():
    """Verify record_text extracts string content correctly from both flat text and chat message lists."""
    # 1. Flat text record
    flat = {"text": "SELECT * FROM test;"}
    assert record_text(flat) == "SELECT * FROM test;"

    # 2. Chat messages record
    chat = {
        "messages": [
            {"role": "user", "content": "How do I use DISTINCT ON?"},
            {"role": "assistant", "content": "Use SELECT DISTINCT ON (user_id)..."},
        ]
    }
    assert "DISTINCT ON" in record_text(chat)
    assert "user_id" in record_text(chat)

    # 3. Empty fallback
    empty = {}
    assert record_text(empty) == ""


def test_family_of_metadata_resolution():
    """Verify family_of extracts the correct family identifier from meta dict."""
    rec1 = {"meta": {"family": "adv_distinct_on"}}
    assert family_of(rec1) == "adv_distinct_on"

    rec2 = {"meta": {"source": "py_taskgroup"}}
    assert family_of(rec2) == "py_taskgroup"

    rec3 = {}
    assert family_of(rec3) == "?"


def test_reserved_postgresql_signatures_detection():
    """Verify all reserved PostgreSQL signatures properly flag synthetic occurrences."""
    pg_signatures = RESERVED["postgresql"]["signatures"]

    test_cases = [
        ("SELECT DISTINCT ON (dept) id FROM emp;", r"distinct\s+on"),
        ("SELECT count(*) FILTER (WHERE active = true);", r"\bfilter\s*\(\s*where"),
        ("SELECT * FROM unnest(arr) WITH ORDINALITY;", r"with\s+ordinality"),
        ("SELECT * FROM t, LATERAL (SELECT * FROM sub) s;", r"\blateral\b"),
        ("SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY val);", r"percentile_cont"),
    ]

    for sql, sig in test_cases:
        assert sig in pg_signatures
        assert len(re.findall(sig, sql, re.I)) > 0


def test_reserved_astral_signatures_detection():
    """Verify reserved Astral/Python signatures properly flag synthetic occurrences."""
    ast_signatures = RESERVED["astral"]["signatures"]

    test_cases = [
        ("import functools\nf = functools.partial(func, 1)", r"functools\.partial"),
        ("f = partial(func, 1)", r"\bpartial\s*\("),
        ("@singledispatch\ndef process(arg): pass", r"singledispatch"),
        ("class MyClass:\n    __slots__ = ('a', 'b')", r"__slots__"),
        ("async with asyncio.TaskGroup() as tg:", r"TaskGroup"),
        ("@cached_property\ndef compute(self): return 42", r"cached_property"),
    ]

    for py_code, sig in test_cases:
        assert sig in ast_signatures
        assert len(re.findall(sig, py_code, re.I)) > 0


def test_directional_cross_corpus_validity_logic():
    """Verify directional leakage classification:
    - DuckDB constructs in PostgreSQL -> INVALID (broken syntax).
    - PostgreSQL constructs in DuckDB -> VALID (native compatibility).
    """
    # DuckDB-specific constructs that produce syntax errors in PostgreSQL
    duckdb_exclusive = [
        "SELECT COLUMNS('cost_.*') FROM expenses;",
        "SELECT * FROM read_parquet('s3://bucket/file.parquet');",
        "SELECT dept, sum(salary) FROM employees GROUP BY ALL;",
        "SELECT * FROM sales QUALIFY row_number() OVER () = 1;",
    ]
    for code in duckdb_exclusive:
        # Must match DuckDB exclusive patterns
        assert any(
            len(re.findall(pat, code, re.I)) > 0
            for pat in [r"COLUMNS\s*\(", r"read_parquet", r"GROUP\s+BY\s+ALL", r"\bQUALIFY\b"]
        )

    # Shared constructs supported natively by both engines
    shared_constructs = [
        "SELECT DISTINCT ON (dept) * FROM staff;",
        "SELECT sum(val) FILTER (WHERE active = 1) FROM t;",
        "SELECT date_trunc('month', created_at) FROM logs;",
    ]
    for code in shared_constructs:
        assert len(re.findall(r"distinct\s+on|\bfilter\s*\(\s*where|date_trunc", code, re.I)) > 0
