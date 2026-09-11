"""Fetches the `docs/` folder from Astral's tool repos via git sparse-checkout.

Astral's rendered docs site (astral-sh/docs) is a *built* HTML/Next.js output,
not the markdown source — the actual source lives in each tool's own repo.
"""

import shutil
import subprocess
from pathlib import Path

REPOS = {
    "uv": "https://github.com/astral-sh/uv.git",
    "ruff": "https://github.com/astral-sh/ruff.git",
    "ty": "https://github.com/astral-sh/ty.git",
}


def _run_git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result


def fetch_docs(repo_key: str, cache_dir: Path, refresh: bool = False) -> Path:
    """Sparse-checkout `docs/` from an Astral repo into cache_dir/<repo_key>.

    Idempotent: if the target already has a .git, fetches + resets instead of
    re-cloning, unless refresh=True forces a fresh clone. Returns the path to
    the docs/ subdirectory.
    """
    if repo_key not in REPOS:
        raise ValueError(f"Unknown repo '{repo_key}'. Options: {sorted(REPOS)}")

    repo_dir = cache_dir / repo_key
    docs_dir = repo_dir / "docs"

    if refresh and repo_dir.exists():
        shutil.rmtree(repo_dir)

    if (repo_dir / ".git").exists():
        print(f"[fetch] {repo_key}: cache hit, updating")
        _run_git("fetch", "--depth", "1", "origin", "main", cwd=repo_dir)
        _run_git("reset", "--hard", "origin/main", cwd=repo_dir)
    else:
        print(f"[fetch] {repo_key}: cloning (sparse, docs/ only)")
        cache_dir.mkdir(parents=True, exist_ok=True)
        _run_git(
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            REPOS[repo_key],
            str(repo_dir),
        )
        _run_git("sparse-checkout", "set", "docs", cwd=repo_dir)

    if not docs_dir.exists():
        raise RuntimeError(f"docs/ not found in {repo_dir} after checkout")
    return docs_dir


def fetch_all(repo_keys: list[str], cache_dir: Path, refresh: bool = False) -> dict[str, Path]:
    return {key: fetch_docs(key, cache_dir, refresh=refresh) for key in repo_keys}
