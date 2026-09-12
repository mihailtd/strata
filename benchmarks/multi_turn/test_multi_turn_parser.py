"""Unit tests for markdown code fence extraction, formatting cleanup, and pipeline data structures."""

import pytest
from benchmarks.multi_turn.multi_turn_execution_benchmark import (
    BENCHMARK_PIPELINES,
    Pipeline,
    TurnStep,
    extract_code_block,
)


def test_extract_code_block_tagged_markdown():
    """Verify clean extraction from tagged markdown fences."""
    sql_text = """Here is the SQL query:
```sql
SELECT id, name FROM users WHERE active = true;
```
Hope this helps!"""
    extracted = extract_code_block(sql_text, language="sql")
    assert extracted == "SELECT id, name FROM users WHERE active = true;"


def test_extract_code_block_python_tagged_markdown():
    """Verify clean extraction from python tagged markdown fences."""
    py_text = """```python
import asyncio

async def main():
    async with asyncio.TaskGroup() as tg:
        tg.create_task(asyncio.sleep(0.1))
```"""
    extracted = extract_code_block(py_text, language="python")
    assert "asyncio.TaskGroup()" in extracted
    assert not extracted.startswith("```")


def test_extract_code_block_generic_markdown_fallback():
    """Verify extraction falls back to generic untagged code fences if specific tag is absent."""
    text = """```
CREATE TABLE test (id SERIAL PRIMARY KEY);
```"""
    extracted = extract_code_block(text, language="sql")
    assert extracted == "CREATE TABLE test (id SERIAL PRIMARY KEY);"


def test_extract_code_block_unclosed_fence():
    """Verify extraction handles truncated output with unclosed markdown fences."""
    text = """```sql
SELECT user_id, count(*)
FROM events
GROUP BY user_id"""
    extracted = extract_code_block(text, language="sql")
    assert "SELECT user_id, count(*)" in extracted
    assert "GROUP BY user_id" in extracted


def test_extract_code_block_raw_text_fallback():
    """Verify extraction returns raw text stripped when no backticks exist."""
    raw = "SELECT 1 AS num;"
    assert extract_code_block(raw, language="sql") == "SELECT 1 AS num;"


def test_benchmark_pipelines_structure():
    """Verify benchmark pipelines have valid turn step sequencing and target expert bindings."""
    assert len(BENCHMARK_PIPELINES) >= 4
    for pipe in BENCHMARK_PIPELINES:
        assert isinstance(pipe, Pipeline)
        assert len(pipe.steps) >= 2
        for i, step in enumerate(pipe.steps):
            assert isinstance(step, TurnStep)
            assert step.turn_num == i + 1
            assert step.target_expert in ("postgresql", "astral", "duckdb", "financial")
            assert step.eval_type in ("sql", "python")
            assert isinstance(step.is_ood, bool)
            assert len(step.prompt) > 0
