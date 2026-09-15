"""Fail the build when a benchmark hard-codes a value that belongs to canon.py.

The rule this enforces: a benchmark that GRADES QUALITY (execution, linting,
regex rubrics, pass@k) may not carry its own decode budget or its own adapter
version. It must import them from runtime.canon.

Latency/throughput probes are exempt -- a 8- or 32-token decode is the correct
measurement there, and forcing 2048 on them would be its own kind of wrong. The
exemption is by explicit path, never by guess, so a new quality benchmark cannot
sneak in by looking like a probe.

    uv run python audit/check_canon.py          # report, exit 1 on violation
    uv run python audit/check_canon.py --list   # show what is scanned
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from runtime.canon import REPO_ROOT as REPO  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
CANON_MAX_NEW_TOKENS = 2048
CANON_ADAPTER_VERSION = "v7"

# Signals that a script grades OUTPUT QUALITY rather than latency.
QUALITY_MARKERS = (
    "expects", "MODERN_TERMS", "score_one", "ruff check", "compiles",
    "pass@", "linter_score", "POSTGRES_GOOD", "pass_1", "execute",
)

# Latency/throughput probes: short decodes are CORRECT here. Explicit list.
EXEMPT_DIRS = (
    "benchmarks/runtime/speculative/",
    "benchmarks/runtime/memory/",
    "benchmarks/runtime/performance/",
    "benchmarks/superseded/",
    # Stage-1 pre-integration work (see docs/METHODOLOGY.md) -- same latency/gating
    # nature as their benchmarks/ siblings above, just not integrated yet.
    "experiments/runtime/speculative/",
    "experiments/kernel/",
)
EXEMPT_FILES = (
    "benchmarks/runtime/folding/benchmark_weight_folding.py",
    "benchmarks/runtime/folding/evaluate_flash_norm_quality.py",
)

# Three spellings all appear in this repo and all have bitten us. Match each
# explicitly rather than with one loose pattern -- a regex that silently fails to
# match is exactly the failure mode this script exists to prevent.
TOKEN_RES = (
    # parser.add_argument("--max-new-tokens", type=int, default=192)
    re.compile(r"max[_-]new[_-]tokens[\"']\s*,\s*type\s*=\s*int\s*,\s*default\s*=\s*(\d+)"),
    # def f(..., max_new_tokens: int = 768)   and   max_new_tokens = 768
    re.compile(r"max[_-]new[_-]tokens\s*(?::\s*int\s*)?=\s*(\d+)"),
)
ADAPTER_RE = re.compile(r"m2_([a-z]+)_r8a128_(v\d)")


def is_exempt(rel: str) -> bool:
    return rel.startswith(EXEMPT_DIRS) or rel in EXEMPT_FILES


def grades_quality(text: str) -> bool:
    return any(m in text for m in QUALITY_MARKERS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show scanned files and exit")
    args = ap.parse_args()

    violations: list[str] = []
    scanned = 0

    scan_globs = ("benchmarks/**/*.py", "experiments/**/*.py", "evals/**/*.py")
    for path in sorted(p for g in scan_globs for p in REPO.glob(g)):
        rel = str(path.relative_to(REPO))
        if rel == "audit/check_canon.py" or is_exempt(rel):
            continue
        text = path.read_text(errors="ignore")
        if not grades_quality(text):
            continue
        scanned += 1
        if args.list:
            print(f"  {rel}")
            continue

        # imports canon? then the literals below are canon's problem, not ours
        uses_canon = "runtime.canon" in text

        seen: set[int] = set()
        for rx in TOKEN_RES:
            for m in rx.finditer(text):
                if m.start() in seen:
                    continue
                seen.add(m.start())
                val = int(m.group(1))
                if val < CANON_MAX_NEW_TOKENS:
                    line = text[: m.start()].count("\n") + 1
                    violations.append(
                        f"{rel}:{line}  max_new_tokens={val}  -> must be {CANON_MAX_NEW_TOKENS} "
                        f"(import CANON.MAX_NEW_TOKENS)"
                    )

        if not uses_canon:
            for m in ADAPTER_RE.finditer(text):
                domain, ver = m.group(1), m.group(2)
                if ver != CANON_ADAPTER_VERSION:
                    line = text[: m.start()].count("\n") + 1
                    violations.append(
                        f"{rel}:{line}  m2_{domain}_r8a128_{ver}  -> latest is "
                        f"{CANON_ADAPTER_VERSION} (import adapter_path)"
                    )

    if args.list:
        print(f"\n{scanned} quality-grading files scanned")
        return 0

    print("=" * 78)
    print(f" CANON CHECK -- {scanned} quality-grading benchmarks scanned")
    print("=" * 78)
    if not violations:
        print("  clean: no hard-coded decode budgets below 2048, no legacy adapters")
        return 0
    for v in violations:
        print(f"  VIOLATION  {v}")
    print(f"\n  {len(violations)} violation(s). Fix by importing from runtime.canon.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
