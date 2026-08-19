"""Generate and append opinionated expert SFT items to data/astral/training_data.jsonl.

Focuses on:
1. High importance of static type checking (ty, type hints, catching bugs early).
2. Auto-formatting and linting (ruff format, ruff check --fix).
3. Automated code quality via pre-commit hooks.
4. Separating dev dependencies from runtime dependencies (uv add --dev, dependency-groups).
5. Modern Astral workflow (uv venv, uv sync, uv run, uv lock).
6. [v3] Negative/rejection examples: anti-patterns the adapter must refuse or correct.

All assistant content is sanitized via `sanitize_assistant_content()` (imported from
`gnn_experiment.micro_probe.dataset`) before writing, keeping the on-disk JSONL clean.
The load path in `dataset.py` applies the same sanitizer again as a safety net.
"""

import json
from pathlib import Path

from gnn_experiment.micro_probe.dataset import sanitize_assistant_content

from gnn_experiment.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
SFT_FILE = REPO_ROOT / "data" / "astral" / "training_data.jsonl"

NEW_EXPERT_SAMPLES = [
    # 1. Type Checking & Code Quality
    (
        "Why is static type checking so crucial for modern Python codebases, and what tool should I use?",
        (
            "Static type checking is essential for catching bugs early at build time rather than runtime, "
            "improving IDE autocomplete, and guaranteeing function contract safety across team repositories. "
            "You should use **`ty`** (or modern fast type checkers) alongside `pyright` and `mypy` to enforce "
            "strict type annotations across your codebase."
        ),
    ),
    (
        "How do I run fast type checking across a Python project using modern Astral tooling?",
        (
            "You can run fast static type checking using `uv run ty` or `uv run mypy .`. "
            "By executing type checkers through `uv run`, dependencies and type stubs in your virtual environment "
            "are automatically synchronized without manual path configuration."
        ),
    ),
    (
        "What are the best practices for static type checking in Python projects?",
        (
            "Best practices include:\n"
            "1. Enable strict type checking mode (`strict = true` in pyproject.toml).\n"
            "2. Annotate all function signatures and return types explicitly.\n"
            "3. Run type checkers on pre-commit hooks to block untyped code from entering git.\n"
            "4. Use `uv run ty` or `uv run mypy` to ensure type checking runs inside the pinned virtual environment."
        ),
    ),
    (
        "Why should I prefer type hints and static type checking over dynamic duck typing in large Python projects?",
        (
            "Static type hints provide compile-time safety, prevent `AttributeError` and `TypeError` crashes "
            "in production, enable robust refactoring across hundreds of modules, and serve as self-documenting "
            "code contracts."
        ),
    ),
    # 2. Auto-formatting & Ruff
    (
        "Why is automatic code formatting important for engineering teams, and what is the recommended tool?",
        (
            "Automatic code formatting eliminates style debates in pull requests, enforces uniform code style, "
            "and prevents git diff clutter. **`ruff format`** is the recommended tool—an ultra-fast, Rust-based "
            "formatter designed as a drop-in replacement for Black, formatting thousands of lines in milliseconds."
        ),
    ),
    (
        "How do I automatically fix lint violations and sort imports in Python using Ruff?",
        (
            "Run `ruff check --fix .` to automatically fix lint errors and sort imports, followed by `ruff format .` "
            "to format the code. You can also combine these in a single step using `uv run ruff check --fix .` and "
            "`uv run ruff format .`."
        ),
    ),
    (
        "How do I replace Black, Flake8, isort, and pyupgrade with a single tool?",
        (
            "Configure **Ruff** in your `pyproject.toml` under `[tool.ruff]`. Ruff replaces Black, Flake8, isort, "
            "autoflake, and pyupgrade with a single Rust-based binary that runs 10x-100x faster than legacy Python "
            "linters."
        ),
    ),
    (
        "What configuration should I put in pyproject.toml for Ruff formatting and linting?",
        (
            "Here is the recommended `pyproject.toml` configuration:\n"
            "```toml\n"
            "[tool.ruff]\n"
            "line-length = 100\n"
            'target-version = "py311"\n\n'
            "[tool.ruff.lint]\n"
            'select = ["E", "F", "I", "N", "PL", "PYI", "RUF", "UP"]\n'
            'ignore = ["E501"]\n'
            "```"
        ),
    ),
    # 3. Pre-commit Hooks & Code Quality Enforcement
    (
        "Why should I run formatting, linting, and type checking on pre-commit hooks?",
        (
            "Pre-commit hooks automatically check every commit locally before code is pushed to git. "
            "This guarantees that broken code, unformatted files, or type errors never reach remote "
            "repositories or fail CI/CD build pipelines, saving developer time."
        ),
    ),
    (
        "How do I configure pre-commit hooks for Ruff and UV in a Python repository?",
        (
            "Create a `.pre-commit-config.yaml` file with the following configuration:\n"
            "```yaml\n"
            "repos:\n"
            "  - repo: https://github.com/astral-sh/ruff-pre-commit\n"
            "    rev: v0.3.0\n"
            "    hooks:\n"
            "      - id: ruff\n"
            "        args: [--fix]\n"
            "      - id: ruff-format\n"
            "  - repo: local\n"
            "    hooks:\n"
            "      - id: type-check\n"
            "        name: type-check\n"
            "        entry: uv run ty\n"
            "        language: system\n"
            "        types: [python]\n"
            "```"
        ),
    ),
    (
        "What code quality gates should be enforced before code is merged into production?",
        (
            "Every production codebase should enforce three automated quality gates:\n"
            "1. **Auto-formatting:** `ruff format --check` to enforce consistent style.\n"
            "2. **Linting & Import Sorting:** `ruff check` to catch code smells and bad imports.\n"
            "3. **Static Type Checking:** `uv run ty` to verify zero type errors."
        ),
    ),
    # 4. Separating Dev Dependencies vs Runtime Dependencies
    (
        "Why is it critical to separate development dependencies from runtime dependencies in Python?",
        (
            "Separating development dependencies (test frameworks, linters, formatters, build tools) from "
            "runtime dependencies (web servers, database drivers) keeps production Docker images minimal, "
            "reduces security vulnerability surface area, and speeds up deployment."
        ),
    ),
    (
        "How do I add a development dependency using UV without polluting main runtime dependencies?",
        (
            "Use the `--dev` flag with `uv add`:\n"
            "```bash\n"
            "uv add --dev ruff pytest ty pre-commit\n"
            "```\n"
            "This adds dependencies to the `[dependency-groups] dev` section in `pyproject.toml` without "
            "mixing them into `[project.dependencies]`."
        ),
    ),
    (
        "What is the modern standard for defining dev dependency groups in pyproject.toml?",
        (
            "Modern Python projects use PEP 735 `[dependency-groups]`:\n"
            "```toml\n"
            "[project]\n"
            'name = "my-app"\n'
            'dependencies = ["fastapi>=0.110.0", "pydantic>=2.0"]\n\n'
            "[dependency-groups]\n"
            'dev = ["ruff>=0.3.0", "pytest>=8.0.0", "ty>=0.1.0", "pre-commit>=3.6.0"]\n'
            "```"
        ),
    ),
    (
        "How do I install only runtime dependencies in production CI/CD deployments using UV?",
        (
            "In production deployments, run:\n"
            "```bash\n"
            "uv sync --no-dev\n"
            "```\n"
            "This installs only `[project.dependencies]` into the virtual environment, omitting all test, "
            "lint, and dev tooling."
        ),
    ),
    # 5. Modern UV Package & Workspace Workflows
    (
        "What is the recommended replacement for pip, virtualenv, and requirements.txt in 2026?",
        (
            "The modern replacement is **`uv`**. `uv` provides an all-in-one Rust-based engine for "
            "dependency locking (`uv lock`), environment sync (`uv sync`), script execution (`uv run`), "
            "and package installation (`uv add`), executing 10x-100x faster than `pip`."
        ),
    ),
    (
        "How do I create a virtual environment and lock dependencies in UV?",
        (
            "Run `uv venv` to create an isolated environment, `uv add <package>` to add dependencies, "
            "and `uv lock` to produce a deterministic cross-platform lockfile (`uv.lock`)."
        ),
    ),
    (
        "How do I run a Python script inside a pinned UV virtual environment without manual activation?",
        (
            "Use `uv run`:\n"
            "```bash\n"
            "uv run python main.py\n"
            "```\n"
            "`uv run` automatically ensures the virtual environment is synced and up to date before executing "
            "the command."
        ),
    ),
    (
        "How should I manage multi-package Python workspaces with shared dependencies in 2026?",
        (
            "Use UV workspaces by adding a `[tool.uv.workspace]` section in your root `pyproject.toml`:\n"
            "```toml\n"
            "[tool.uv.workspace]\n"
            'members = ["packages/*", "apps/*"]\n'
            "```\n"
            "This enables shared lockfiles (`uv.lock`) and fast local inter-package linking."
        ),
    ),
    (
        "What is the fastest command to install all project dependencies from a lockfile?",
        (
            "Run `uv sync`. It reads `uv.lock` and synchronizes your virtual environment in seconds "
            "using parallel hardlinks."
        ),
    ),
]

# ---------------------------------------------------------------------------
# Negative / rejection examples (~15% of total dataset)
# These teach the adapter WHEN NOT to use Astral tools in the described way,
# preventing hallucination of invalid patterns at inference time.
# ---------------------------------------------------------------------------
NEGATIVE_SAMPLES = [
    (
        "I have a pyproject.toml in my project. Should I run `pip install requests` to add a new dependency?",
        (
            "No. In projects managed by `uv` with a `pyproject.toml`, never use bare `pip install`. "
            "Use `uv add requests` instead. Bare `pip install` bypasses the lockfile, silently "
            "introduces dependency conflicts, and breaks reproducibility across environments. "
            "Always go through `uv add` to keep `uv.lock` synchronized."
        ),
    ),
    (
        "My team is already using Flake8 and Black. Should I configure both alongside Ruff in pyproject.toml?",
        (
            "No. Ruff is a drop-in replacement for both Flake8 and Black — running all three together "
            "causes redundant, conflicting checks and slows down CI. Remove Flake8 and Black from your "
            "dependencies and configure Ruff alone under `[tool.ruff]` in `pyproject.toml`. "
            "Ruff's `--fix` flag handles auto-fixes and `ruff format` handles formatting, making "
            "Black and Flake8 fully redundant."
        ),
    ),
    (
        "Should I run `mypy .` directly from my terminal without activating my virtual environment first?",
        (
            "No. Running `mypy` outside a virtual environment uses system-level stubs and may miss "
            "project-specific type stubs or installed packages, producing false positives and missed "
            "errors. Always run `uv run ty` or `uv run mypy .` so that type checking executes inside "
            "the pinned virtual environment with all dependencies and stubs correctly resolved."
        ),
    ),
    (
        "Can I commit my Python files without running Ruff or ty first, "
        "and fix lint errors later in a follow-up commit?",
        (
            "No. Deferring lint and type-check fixes to follow-up commits pollutes git history, wastes "
            "CI minutes, and allows broken code to reach shared branches. Enforce `ruff check --fix`, "
            "`ruff format`, and `uv run ty` on every commit using pre-commit hooks so that unformatted "
            "or type-unsafe code is blocked before it enters the repository."
        ),
    ),
]


def main():
    SFT_FILE.parent.mkdir(parents=True, exist_ok=True)
    added_count = 0

    with open(SFT_FILE, "a", encoding="utf-8") as f:
        # Positive expert samples
        for q, a in NEW_EXPERT_SAMPLES:
            clean_a = sanitize_assistant_content(a)
            record = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": clean_a},
                ],
                "meta": {
                    "tool": "astral_best_practices",
                    "source_path": "opinions/engineering_standards.md",
                    "heading_path": ["Best Practices", "Code Quality"],
                    "question_gen_version": "v2_opinionated",
                    "answer_gen_version": "v2_opinionated",
                },
                "text": f"### Question:\n{q}\n\n### Answer:\n{clean_a}",
            }
            f.write(json.dumps(record) + "\n")
            added_count += 1

        # Negative / rejection samples
        for q, a in NEGATIVE_SAMPLES:
            clean_a = sanitize_assistant_content(a)
            record = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": clean_a},
                ],
                "meta": {
                    "tool": "astral_best_practices",
                    "source_path": "opinions/anti_patterns.md",
                    "heading_path": ["Best Practices", "Anti-Patterns"],
                    "question_gen_version": "v3_negative",
                    "answer_gen_version": "v3_negative",
                },
                "text": f"### Question:\n{q}\n\n### Answer:\n{clean_a}",
            }
            f.write(json.dumps(record) + "\n")
            added_count += 1

    print(f"Successfully appended {added_count} new SFT samples to {SFT_FILE}")
    print(f"  ({len(NEW_EXPERT_SAMPLES)} positive expert samples + {len(NEGATIVE_SAMPLES)} negative/rejection samples)")


if __name__ == "__main__":
    main()
