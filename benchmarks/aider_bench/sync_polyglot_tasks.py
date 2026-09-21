"""Sync all Aider Polyglot Benchmark exercises into benchmarks/aider_bench/polyglot/."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
POLYGLOT_DIR = REPO_ROOT / "benchmarks" / "aider_bench" / "polyglot"
REMOTE_REPO = "https://github.com/Aider-AI/polyglot-benchmark"
LOCAL_CLONE_CACHE = Path("/tmp/polyglot-benchmark")


def ensure_polyglot_repo() -> Path:
    if LOCAL_CLONE_CACHE.exists() and (LOCAL_CLONE_CACHE / "rust").exists():
        return LOCAL_CLONE_CACHE

    print(f"Cloning {REMOTE_REPO} into {LOCAL_CLONE_CACHE}...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REMOTE_REPO, str(LOCAL_CLONE_CACHE)],
        check=True,
    )
    return LOCAL_CLONE_CACHE


def sync_polyglot_tasks() -> dict[str, int]:
    source_root = ensure_polyglot_repo()
    POLYGLOT_DIR.mkdir(parents=True, exist_ok=True)

    languages = ["cpp", "go", "java", "javascript", "python", "rust"]
    counts = {}

    for lang in languages:
        practice_src = source_root / lang / "exercises" / "practice"
        dest_lang_dir = POLYGLOT_DIR / lang / "exercises" / "practice"
        if not practice_src.exists():
            continue

        dest_lang_dir.mkdir(parents=True, exist_ok=True)
        synced_lang = 0

        for exercise in practice_src.iterdir():
            if not exercise.is_dir() or exercise.name.startswith("."):
                continue
            dest_exercise = dest_lang_dir / exercise.name
            if dest_exercise.exists():
                shutil.rmtree(dest_exercise)
            shutil.copytree(exercise, dest_exercise)
            synced_lang += 1

        counts[lang] = synced_lang
        print(f"Synced {synced_lang} {lang} exercises into {dest_lang_dir}")

    total = sum(counts.values())
    print(f"Total Polyglot exercises synced: {total}")
    return counts


if __name__ == "__main__":
    sync_polyglot_tasks()
