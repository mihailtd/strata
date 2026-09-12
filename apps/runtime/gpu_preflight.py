"""Compatibility shim -- gpu_preflight.py now lives in apps/runtime-common.

Re-exported here so existing `from runtime import gpu_preflight` /
`from runtime.gpu_preflight import ...` call sites (benchmarks/, scripts/,
tests/, scratch/) keep working until the Stage 4 repo-wide codemod repoints
them at runtime_common directly. Add new code to
apps/runtime-common/src/runtime_common/gpu_preflight.py, not here.
"""

from __future__ import annotations

from runtime_common.gpu_preflight import (
    check_gpu_availability,
    ensure_gpu_exclusive,
    find_conflicting_processes,
    get_gpu_vram_info,
    get_sysfs_vram_info,
)

__all__ = [
    "get_sysfs_vram_info",
    "get_gpu_vram_info",
    "find_conflicting_processes",
    "check_gpu_availability",
    "ensure_gpu_exclusive",
]
