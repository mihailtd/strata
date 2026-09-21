"""Polyglot task loader and metadata definitions for all 225 Aider Polyglot Benchmark exercises."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

POLYGLOT_DIR = Path(__file__).resolve().parent / "polyglot"

SUPPORTED_LANGUAGES = ["cpp", "go", "java", "javascript", "python", "rust"]

TEST_COMMANDS = {
    "python": ["pytest"],
    "rust": ["cargo", "test", "--", "--include-ignored"],
    "go": ["go", "test", "./..."],
    "javascript": ["npm", "test"],
    "cpp": ["make", "test"],
    "java": ["./gradlew", "test"],
}


@dataclass
class PolyglotTask:
    language: str
    name: str
    instructions: str
    solution_files: dict[str, str]  # relative_path -> content
    test_files: dict[str, str]      # relative_path -> content
    all_files: dict[str, str]       # relative_path -> content (including configs/build files)
    task_dir: Path
    test_command: list[str] = field(default_factory=list)


def list_polyglot_tasks(language: str | None = None) -> list[str]:
    """Returns list of task identifiers formatted as '{lang}/{name}' or names if language is specified."""
    if not POLYGLOT_DIR.exists():
        return []

    langs = [language] if language and language in SUPPORTED_LANGUAGES else SUPPORTED_LANGUAGES
    tasks = []

    for lang in langs:
        practice_dir = POLYGLOT_DIR / lang / "exercises" / "practice"
        if not practice_dir.exists():
            continue
        for exercise in sorted(practice_dir.iterdir()):
            if exercise.is_dir() and not exercise.name.startswith("."):
                tasks.append(f"{lang}/{exercise.name}")

    return sorted(tasks)


def load_polyglot_task(task_id: str) -> PolyglotTask:
    """Loads a polyglot task by identifier '{lang}/{name}'."""
    if "/" not in task_id:
        raise ValueError(f"task_id must be in format 'lang/task_name', got '{task_id}'")

    lang, name = task_id.split("/", 1)
    task_dir = POLYGLOT_DIR / lang / "exercises" / "practice" / name

    if not task_dir.is_dir():
        raise FileNotFoundError(f"Polyglot task '{task_id}' not found at {task_dir}")

    # Read instructions
    instructions = ""
    intro_path = task_dir / ".docs" / "introduction.md"
    if intro_path.exists():
        instructions += intro_path.read_text(encoding="utf-8").strip() + "\n\n"

    instr_path = task_dir / ".docs" / "instructions.md"
    if instr_path.exists():
        instructions += instr_path.read_text(encoding="utf-8").strip() + "\n\n"

    append_path = task_dir / ".docs" / "instructions.append.md"
    if append_path.exists():
        instructions += append_path.read_text(encoding="utf-8").strip()

    # Read config.json
    config_path = task_dir / ".meta" / "config.json"
    solution_paths = []
    test_paths = []
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            solution_paths = cfg.get("files", {}).get("solution", [])
            test_paths = cfg.get("files", {}).get("test", [])
        except Exception:
            pass

    solution_files: dict[str, str] = {}
    test_files: dict[str, str] = {}
    all_files: dict[str, str] = {}

    for file_path in task_dir.rglob("*"):
        if not file_path.is_file():
            continue
        rel = str(file_path.relative_to(task_dir))
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        all_files[rel] = content

        if rel in solution_paths:
            solution_files[rel] = content
        elif rel in test_paths:
            test_files[rel] = content

    test_cmd = TEST_COMMANDS.get(lang, ["pytest"])

    return PolyglotTask(
        language=lang,
        name=name,
        instructions=instructions.strip(),
        solution_files=solution_files,
        test_files=test_files,
        all_files=all_files,
        task_dir=task_dir,
        test_command=test_cmd,
    )
