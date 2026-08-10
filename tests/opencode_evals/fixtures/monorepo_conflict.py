"""Generates an isolated, throwaway uv workspace with a real PubGrub
dependency conflict (apps/web-api wants pydantic>=2.0, libs/legacy-connector
wants pydantic<2.0), for testing whether an agent can diagnose/fix it.

Fully self-contained: writes its own pyproject.toml files directly (never
via `uv init`, which would try to register the fixture as a member of
whatever project it's nested under). Give it any base_dir and it produces a
workspace with no relation to this repo's own uv project.
"""

import subprocess
from pathlib import Path

ROOT_PYPROJECT = """[project]
name = "workspace-root"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = []

[tool.uv.workspace]
members = ["apps/*", "libs/*"]
"""

WEB_API_PYPROJECT = """[project]
name = "web-api"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0.0",
]
"""

LEGACY_CONNECTOR_PYPROJECT = """[project]
name = "legacy-connector"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pydantic<2.0.0",
]
"""


def create(base_dir: Path, git: bool = True) -> Path:
    """Writes the broken workspace under base_dir and returns its path.

    With git=True (default), also `git init`s the fixture and commits the
    broken state as the initial commit — this gives you `reset()` (below)
    for free, and `git diff` shows exactly what an agent changed, without
    needing Docker or regenerating the fixture from scratch.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "apps" / "web-api").mkdir(parents=True, exist_ok=True)
    (base_dir / "libs" / "legacy-connector").mkdir(parents=True, exist_ok=True)

    (base_dir / "pyproject.toml").write_text(ROOT_PYPROJECT)
    (base_dir / "apps" / "web-api" / "pyproject.toml").write_text(WEB_API_PYPROJECT)
    (base_dir / "libs" / "legacy-connector" / "pyproject.toml").write_text(
        LEGACY_CONNECTOR_PYPROJECT
    )

    if git:
        _run_git(base_dir, "init", "-q")
        _run_git(base_dir, "config", "user.email", "eval@local")
        _run_git(base_dir, "config", "user.name", "eval")
        _run_git(base_dir, "add", "-A")
        _run_git(base_dir, "commit", "-q", "-m", "broken: pydantic version conflict")

    return base_dir


def reset(base_dir: Path) -> None:
    """Restores base_dir to the broken state committed by create(), discarding
    any changes an agent made (including new/deleted files) — instant, no
    Docker, no regenerating from scratch."""
    _run_git(base_dir, "checkout", "-q", "--", ".")
    _run_git(base_dir, "clean", "-q", "-fdx")


def diff(base_dir: Path) -> str:
    """What an agent actually changed, relative to the broken initial commit."""
    result = subprocess.run(
        ["git", "diff", "HEAD"], cwd=base_dir, capture_output=True, text=True
    )
    return result.stdout


def _run_git(base_dir: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=base_dir, capture_output=True, text=True, check=True)
