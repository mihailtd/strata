"""Adds the monorepo's apps/ dir to sys.path so `import runtime.X` resolves.

apps/factory deliberately does NOT take a package dependency on the root
`runtime` project (see pyproject.toml's comment) -- root's own deps float at
torch>=2.13.0/vllm/flash-attn, which would conflict with this project's
torch==2.11.0 pin in one uv resolution. The handful of apps/runtime modules
genuinely shared with training (novel_peft, mtp_draft, w4a16_loader,
training_db, micro_probe/dataset, utils/logger, eval/eval_suite, datagen/*)
are consumed as source instead, via the same sys.path trick every
benchmarks/*.py script already uses.

Usage: `import _bootstrap` before any `from runtime... import ...` line.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"Could not locate monorepo root walking up from {start}")


_apps_dir = _find_repo_root(Path(__file__).resolve()) / "apps"
if str(_apps_dir) not in sys.path:
    sys.path.insert(0, str(_apps_dir))
