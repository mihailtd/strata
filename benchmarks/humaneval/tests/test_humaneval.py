"""Unit tests for HumanEval benchmark loader and executor."""

from __future__ import annotations

import pytest

from benchmarks.humaneval import (
    HumanEvalExecutor,
    clean_code,
    get_problem,
    load_humaneval_problems,
)


def test_humaneval_dataset_loads_all_164():
    problems = load_humaneval_problems()
    assert len(problems) == 164
    assert problems[0].task_id == "HumanEval/0"
    assert problems[163].task_id == "HumanEval/163"


def test_get_problem():
    p0 = get_problem("HumanEval/0")
    assert p0.entry_point == "has_close_elements"
    assert "threshold" in p0.prompt

    p1 = get_problem("1")
    assert p1.entry_point == "separate_paren_groups"


def test_humaneval_executor_canonical_solution_passes():
    p = get_problem("HumanEval/0")
    executor = HumanEvalExecutor(timeout_seconds=5.0)
    # Combining prompt and canonical solution
    code = p.prompt + "\n" + p.canonical_solution
    res = executor.execute(p, code)
    assert res.passed is True
    assert res.error == ""


def test_humaneval_executor_failing_solution():
    p = get_problem("HumanEval/0")
    executor = HumanEvalExecutor(timeout_seconds=5.0)
    # Purposely faulty function
    bad_code = "def has_close_elements(numbers, threshold):\n    return False\n"
    res = executor.execute(p, bad_code)
    assert res.passed is False
    assert "AssertionError" in res.error or "assert" in res.error.lower()


def test_humaneval_executor_timeout_detected():
    p = get_problem("HumanEval/0")
    executor = HumanEvalExecutor(timeout_seconds=0.5)
    infinite_loop_code = (
        "def has_close_elements(numbers, threshold):\n"
        "    while True:\n"
        "        pass\n"
    )
    res = executor.execute(p, infinite_loop_code)
    assert res.passed is False
    assert "TimeoutExpired" in res.error


def test_clean_code_varieties():
    entry = "foo"
    prompt = "def foo(x):\n"
    markdown_wrapped = "Here is the code:\n```python\ndef foo(x):\n    return x + 1\n```\nHope that helps!"
    extracted = clean_code(markdown_wrapped, entry, prompt)
    assert "def foo(x):" in extracted
    assert "return x + 1" in extracted
    assert "Here is the code" not in extracted
