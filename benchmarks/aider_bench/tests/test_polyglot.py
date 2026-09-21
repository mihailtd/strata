"""Unit tests for Aider Polyglot Benchmark task loader and executor."""

from __future__ import annotations

import pytest

from benchmarks.aider_bench.polyglot_tasks import (
    POLYGLOT_DIR,
    SUPPORTED_LANGUAGES,
    list_polyglot_tasks,
    load_polyglot_task,
)
from benchmarks.aider_bench.polyglot_executor import PolyglotExecutor


def test_polyglot_tasks_discovered():
    assert POLYGLOT_DIR.exists()
    all_tasks = list_polyglot_tasks()
    # Polyglot benchmark standard has 225 exercises
    assert len(all_tasks) == 225

    # Check each language track
    for lang in SUPPORTED_LANGUAGES:
        tasks = list_polyglot_tasks(lang)
        assert len(tasks) > 0, f"Expected tasks for {lang}"


def test_load_polyglot_task_rust():
    task = load_polyglot_task("rust/acronym")
    assert task.language == "rust"
    assert task.name == "acronym"
    assert "acronym" in task.instructions.lower()
    assert "src/lib.rs" in task.solution_files
    assert "tests/acronym.rs" in task.test_files
    assert task.test_command == ["cargo", "test", "--", "--include-ignored"]


def test_load_polyglot_task_python():
    task = load_polyglot_task("python/beer-song")
    assert task.language == "python"
    assert task.name == "beer-song"
    assert len(task.all_files) > 0


def test_load_polyglot_task_javascript():
    task = load_polyglot_task("javascript/bowling")
    assert task.language == "javascript"
    assert task.name == "bowling"
    assert len(task.all_files) > 0


def test_polyglot_executor_missing_binary():
    executor = PolyglotExecutor(timeout_seconds=5)
    task = load_polyglot_task("rust/acronym")
    # Test with non-existent command
    task.test_command = ["__non_existent_binary__"]
    result = executor.execute(task, {})
    assert not result.passed
    assert result.exit_code == 127
    assert "Missing toolchain" in result.error_summary
