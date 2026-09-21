"""Task loader and metadata definitions for Aider Python practice benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

TASKS_DIR = Path(__file__).resolve().parent / "tasks"


@dataclass
class AiderTask:
    name: str
    instructions: str
    stub_code: str
    test_code: str
    target_file_name: str
    test_file_name: str

    @property
    def target_module_name(self) -> str:
        return self.target_file_name.removesuffix(".py")


CORE_TASKS = [
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


def list_available_tasks(suite: str = "core") -> list[str]:
    """Returns sorted list of available task directory names for the specified suite ('core', 'full', or 'all')."""
    if not TASKS_DIR.exists():
        return []
    all_tasks = sorted([d.name for d in TASKS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")])
    if suite == "core":
        return [t for t in CORE_TASKS if t in all_tasks]
    return all_tasks


def load_task(task_name: str) -> AiderTask:
    """Loads a single task by name."""
    task_dir = TASKS_DIR / task_name
    if not task_dir.is_dir():
        raise FileNotFoundError(f"Task '{task_name}' not found at {task_dir}")

    # Instructions
    instructions_path = task_dir / ".docs" / "instructions.md"
    if not instructions_path.exists():
        # Fallback to direct instructions.md or README.md
        alt = task_dir / "instructions.md"
        instructions_path = alt if alt.exists() else task_dir / "README.md"
    instructions = instructions_path.read_text(encoding="utf-8").strip()

    # Target stub file
    stub_path = task_dir / f"{task_name}.py"
    if not stub_path.exists():
        raise FileNotFoundError(f"Stub file {stub_path} missing for task {task_name}")
    stub_code = stub_path.read_text(encoding="utf-8")

    # Test file
    test_path = task_dir / f"{task_name}_test.py"
    if not test_path.exists():
        raise FileNotFoundError(f"Test file {test_path} missing for task {task_name}")
    test_code = test_path.read_text(encoding="utf-8")

    return AiderTask(
        name=task_name,
        instructions=instructions,
        stub_code=stub_code,
        test_code=test_code,
        target_file_name=f"{task_name}.py",
        test_file_name=f"{task_name}_test.py",
    )


def load_tasks(task_names: list[str] | None = None) -> list[AiderTask]:
    """Loads specified tasks, or all available tasks if none specified."""
    names = task_names if task_names is not None else list_available_tasks()
    return [load_task(name) for name in names]
