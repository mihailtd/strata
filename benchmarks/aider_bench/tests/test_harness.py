"""Strict offline unit tests for the Aider Python benchmark suite.

Verifies task loading, code parsing (whole & diff formats), sandboxed pytest
execution, container command generation, and prompt building without GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.aider_bench.executor import PytestExecutor, PytestResult
from benchmarks.aider_bench.harness_direct import DirectHarness
from benchmarks.aider_bench.parser import (
    apply_edit,
    apply_search_replace_diff,
    extract_whole_file_code,
)
from benchmarks.aider_bench.tasks import list_available_tasks, load_task, load_tasks


# ── 1. Task Loader Tests ───────────────────────────────────────────────────

def test_tasks_load_all_ten():
    available = list_available_tasks()
    expected = [
        "bank_account",
        "binary_search",
        "grains",
        "hello_world",
        "leap",
        "matching_brackets",
        "reverse_string",
        "robot_simulator",
        "two_fer",
        "word_count",
    ]
    assert available == expected

    tasks = load_tasks()
    assert len(tasks) == 10
    for t in tasks:
        assert t.instructions, f"Task {t.name} has empty instructions"
        assert t.stub_code, f"Task {t.name} has empty stub code"
        assert t.test_code, f"Task {t.name} has empty test code"
        assert t.target_file_name.endswith(".py")
        assert t.test_file_name.endswith("_test.py")


def test_task_single_load():
    task = load_task("hello_world")
    assert task.name == "hello_world"
    assert "Hello, World!" in task.instructions
    assert "hello" in task.stub_code
    assert "test_say_hi" in task.test_code


# ── 2. Parser & Diff Engine Tests ──────────────────────────────────────────

def test_extract_whole_file_markdown():
    # Python code fence
    text1 = "Here is the code:\n```python\ndef hello():\n    return 'Hello, World!'\n```\nHope this helps!"
    assert extract_whole_file_code(text1).strip() == "def hello():\n    return 'Hello, World!'"

    # Generic code fence
    text2 = "```\ndef foo():\n    return 42\n```"
    assert extract_whole_file_code(text2).strip() == "def foo():\n    return 42"

    # Raw code without fences
    text3 = "def bar():\n    return 100"
    assert extract_whole_file_code(text3).strip() == "def bar():\n    return 100"


def test_apply_search_replace_diff_single_block():
    original = "def hello():\n    return 'Goodbye, Mars!'\n"
    diff = (
        "<<<<<<< SEARCH\n"
        "    return 'Goodbye, Mars!'\n"
        "=======\n"
        "    return 'Hello, World!'\n"
        ">>>>>>> REPLACE"
    )
    success, updated = apply_search_replace_diff(original, diff)
    assert success is True
    assert updated.strip() == "def hello():\n    return 'Hello, World!'"


def test_apply_search_replace_diff_multi_block():
    original = (
        "def square(number):\n"
        "    pass\n\n"
        "def total():\n"
        "    pass\n"
    )
    diff = (
        "<<<<<<< SEARCH\n"
        "def square(number):\n"
        "    pass\n"
        "=======\n"
        "def square(number):\n"
        "    if not 1 <= number <= 64:\n"
        "        raise ValueError('square must be between 1 and 64')\n"
        "    return 1 << (number - 1)\n"
        ">>>>>>> REPLACE\n\n"
        "<<<<<<< SEARCH\n"
        "def total():\n"
        "    pass\n"
        "=======\n"
        "def total():\n"
        "    return (1 << 64) - 1\n"
        ">>>>>>> REPLACE"
    )
    success, updated = apply_search_replace_diff(original, diff)
    assert success is True
    assert "return 1 << (number - 1)" in updated
    assert "return (1 << 64) - 1" in updated


def test_apply_search_replace_diff_not_found():
    original = "def leap_year(year):\n    pass\n"
    diff = (
        "<<<<<<< SEARCH\n"
        "def non_existent_function():\n"
        "    return False\n"
        "=======\n"
        "def new_function():\n"
        "    return True\n"
        ">>>>>>> REPLACE"
    )
    success, err = apply_search_replace_diff(original, diff)
    assert success is False
    assert "could not be matched" in err


def test_apply_edit_dispatcher():
    original = "x = 1\n"
    # whole
    s1, r1 = apply_edit(original, "```python\nx = 2\n```", "whole")
    assert s1 is True and r1.strip() == "x = 2"

    # diff
    diff_block = "<<<<<<< SEARCH\nx = 1\n=======\nx = 3\n>>>>>>> REPLACE"
    s2, r2 = apply_edit(original, diff_block, "diff")
    assert s2 is True and r2.strip() == "x = 3"


# ── 3. Pytest Executor Tests (Local Subprocess) ─────────────────────────────

def test_executor_failing_stub():
    executor = PytestExecutor(use_container=False)
    task = load_task("hello_world")

    # Initial stub should fail the test
    res = executor.execute(
        target_file_name=task.target_file_name,
        target_code=task.stub_code,
        test_file_name=task.test_file_name,
        test_code=task.test_code,
    )
    assert res.passed is False
    assert res.exit_code != 0
    assert "AssertionError" in (res.stdout + res.stderr)
    assert "Pytest verification FAILED" in res.feedback_message


def test_executor_passing_solution():
    executor = PytestExecutor(use_container=False)
    task = load_task("hello_world")

    correct_code = "def hello():\n    return 'Hello, World!'\n"
    res = executor.execute(
        target_file_name=task.target_file_name,
        target_code=correct_code,
        test_file_name=task.test_file_name,
        test_code=task.test_code,
    )
    assert res.passed is True
    assert res.exit_code == 0
    assert res.duration_s > 0.0


def test_executor_leap_passing_solution():
    executor = PytestExecutor(use_container=False)
    task = load_task("leap")

    correct_leap = (
        "def leap_year(year: int) -> bool:\n"
        "    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n"
    )
    res = executor.execute(
        target_file_name=task.target_file_name,
        target_code=correct_leap,
        test_file_name=task.test_file_name,
        test_code=task.test_code,
    )
    assert res.passed is True
    assert res.exit_code == 0


def test_executor_two_fer_passing_solution():
    executor = PytestExecutor(use_container=False)
    task = load_task("two_fer")

    correct_two_fer = (
        "def two_fer(name: str | None = None) -> str:\n"
        "    if name is None:\n"
        "        return 'One for you, one for me.'\n"
        "    return f'One for {name}, one for me.'\n"
    )
    res = executor.execute(
        target_file_name=task.target_file_name,
        target_code=correct_two_fer,
        test_file_name=task.test_file_name,
        test_code=task.test_code,
    )
    assert res.passed is True
    assert res.exit_code == 0


def test_executor_matching_brackets_passing_solution():
    executor = PytestExecutor(use_container=False)
    task = load_task("matching_brackets")
    sol = (
        "def is_paired(input_string: str) -> bool:\n"
        "    pairs = {')': '(', ']': '[', '}': '{'}\n"
        "    stack = []\n"
        "    for char in input_string:\n"
        "        if char in '([{':\n"
        "            stack.append(char)\n"
        "        elif char in ')]}':\n"
        "            if not stack or stack.pop() != pairs[char]:\n"
        "                return False\n"
        "    return len(stack) == 0\n"
    )
    res = executor.execute(
        target_file_name=task.target_file_name,
        target_code=sol,
        test_file_name=task.test_file_name,
        test_code=task.test_code,
    )
    assert res.passed is True
    assert res.exit_code == 0


def test_executor_binary_search_passing_solution():
    executor = PytestExecutor(use_container=False)
    task = load_task("binary_search")
    sol = (
        "def find(search_list: list[int], value: int) -> int:\n"
        "    lo, hi = 0, len(search_list) - 1\n"
        "    while lo <= hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if search_list[mid] == value:\n"
        "            return mid\n"
        "        elif search_list[mid] < value:\n"
        "            lo = mid + 1\n"
        "        else:\n"
        "            hi = mid - 1\n"
        "    raise ValueError('value not in array')\n"
    )
    res = executor.execute(
        target_file_name=task.target_file_name,
        target_code=sol,
        test_file_name=task.test_file_name,
        test_code=task.test_code,
    )
    assert res.passed is True
    assert res.exit_code == 0


def test_executor_word_count_passing_solution():
    executor = PytestExecutor(use_container=False)
    task = load_task("word_count")
    sol = (
        "import re\n"
        "from collections import Counter\n"
        "def count_words(sentence: str) -> dict[str, int]:\n"
        "    words = re.findall(r\"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?\", sentence.lower())\n"
        "    return dict(Counter(words))\n"
    )
    res = executor.execute(
        target_file_name=task.target_file_name,
        target_code=sol,
        test_file_name=task.test_file_name,
        test_code=task.test_code,
    )
    assert res.passed is True
    assert res.exit_code == 0


# ── 4. Container Command Construction Test ──────────────────────────────────

def test_build_container_command():
    executor = PytestExecutor(
        use_container=True,
        container_engine="podman",
        container_image="python:3.11-slim",
    )
    fake_path = Path("/tmp/test_workspace")
    cmd = executor.build_container_command(fake_path, "hello_world_test.py")

    assert cmd[0] == "podman"
    assert cmd[1] == "run"
    assert "--rm" in cmd
    assert f"{fake_path.resolve()}:/workspace:rw,Z" in cmd
    assert "-w" in cmd and "/workspace" in cmd
    assert "python:3.11-slim" in cmd
    assert "pytest" in cmd
    assert "hello_world_test.py" in cmd


# ── 5. Direct Harness Prompt Builder Tests ─────────────────────────────────

def test_direct_harness_prompt_builder():
    harness_whole = DirectHarness(edit_format="whole")
    task = load_task("reverse_string")
    sys_p, user_p = harness_whole.build_initial_prompt(task)
    assert "```python" in sys_p or "```python" in user_p
    assert "reverse" in user_p.lower()

    harness_diff = DirectHarness(edit_format="diff")
    sys_d, user_d = harness_diff.build_initial_prompt(task)
    assert "<<<<<<< SEARCH" in sys_d
    assert "reverse_string.py" in user_d


def test_resolve_turn_adapter_alternate_and_sequential():
    h_alt = DirectHarness(adapter="alternate:modern@0.25,agentic@0.25")
    assert h_alt._resolve_turn_adapter(1) == "modern@0.25"
    assert h_alt._resolve_turn_adapter(2) == "agentic@0.25"
    assert h_alt._resolve_turn_adapter(3) == "modern@0.25"
    assert h_alt._resolve_turn_adapter(4) == "agentic@0.25"

    h_seq = DirectHarness(adapter="sequential:modern@0.25,agentic@0.25")
    assert h_seq._resolve_turn_adapter(1) == "modern@0.25"
    assert h_seq._resolve_turn_adapter(2) == "agentic@0.25"
    assert h_seq._resolve_turn_adapter(3) == "agentic@0.25"
    assert h_seq._resolve_turn_adapter(4) == "agentic@0.25"

    h_single = DirectHarness(adapter="modern@0.25")
    assert h_single._resolve_turn_adapter(1) == "modern@0.25"
    assert h_single._resolve_turn_adapter(2) == "modern@0.25"

