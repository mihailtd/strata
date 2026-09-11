"""Compatibility shim -- canon.py now lives in apps/runtime-common.

Re-exported here so existing `from runtime.canon import ...` call sites
(benchmarks/, scripts/, tests/, scratch/) keep working until the Stage 4
repo-wide codemod repoints them at runtime_common directly. Add new code to
apps/runtime-common/src/runtime_common/canon.py, not here -- this file has no
logic of its own to avoid a second, divergent copy of a "single source of
truth" module.
"""

from __future__ import annotations

from runtime_common.canon import (
    CANON,
    DOMAINS,
    REPO_ROOT,
    adapter_path,
    validate_kv_cache_precision,
    configure_deterministic_attention,
)

__all__ = [
    "CANON",
    "DOMAINS",
    "REPO_ROOT",
    "adapter_path",
    "validate_kv_cache_precision",
    "configure_deterministic_attention",
]
