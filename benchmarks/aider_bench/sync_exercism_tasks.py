"""Sync all Exercism Python practice exercises into benchmarks/aider_bench/tasks/."""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TASKS_DIR = REPO_ROOT / "benchmarks" / "aider_bench" / "tasks"
SOURCE_PRACTICE_DIR = Path("/tmp/exercism_python/exercises/practice")


def sync_all_tasks() -> int:
    if not SOURCE_PRACTICE_DIR.exists():
        raise FileNotFoundError(f"Source practice directory not found at {SOURCE_PRACTICE_DIR}")

    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    synced_count = 0

    for exercise_dir in sorted(SOURCE_PRACTICE_DIR.iterdir()):
        if not exercise_dir.is_dir() or exercise_dir.name.startswith("."):
            continue

        raw_name = exercise_dir.name
        # Canonical python task name replaces hyphens with underscores
        task_name = raw_name.replace("-", "_")
        dest_dir = TASKS_DIR / task_name

        # Find python stub and test files
        stub_file = exercise_dir / f"{task_name}.py"
        test_file = exercise_dir / f"{task_name}_test.py"

        if not stub_file.exists() or not test_file.exists():
            # Check for alternative naming
            candidates = list(exercise_dir.glob("*.py"))
            test_candidates = [c for c in candidates if c.name.endswith("_test.py")]
            stub_candidates = [c for c in candidates if not c.name.endswith("_test.py") and not c.name.startswith(".")]

            if test_candidates and stub_candidates:
                stub_file = stub_candidates[0]
                test_file = test_candidates[0]
                task_name = stub_file.stem
                dest_dir = TASKS_DIR / task_name
            else:
                continue

        dest_dir.mkdir(parents=True, exist_ok=True)

        # Copy stub and test
        shutil.copy2(stub_file, dest_dir / f"{task_name}.py")
        shutil.copy2(test_file, dest_dir / f"{task_name}_test.py")

        # Copy docs/instructions
        docs_src = exercise_dir / ".docs"
        docs_dest = dest_dir / ".docs"
        if docs_src.exists() and docs_src.is_dir():
            if docs_dest.exists():
                shutil.rmtree(docs_dest)
            shutil.copytree(docs_src, docs_dest)
        elif (exercise_dir / "instructions.md").exists():
            docs_dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(exercise_dir / "instructions.md", docs_dest / "instructions.md")

        synced_count += 1

    print(f"Successfully synced {synced_count} Exercism Python tasks into {TASKS_DIR}")
    return synced_count


if __name__ == "__main__":
    sync_all_tasks()
