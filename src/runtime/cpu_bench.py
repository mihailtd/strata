"""CPU-only benchmark harness: device guard, timing statistics, telemetry persistence.

WHY THIS EXISTS
---------------
The four estimators that use this harness (Vecchia banded precision, CLIME, GEE,
copula tail routing) all run on the CPU side of the request path -- they consume
activation summaries, they do not produce tokens. Benchmarking them while a
training run owns the GPU is therefore legitimate, but only if the benchmark
genuinely never touches the device. "It only allocates a small tensor" is how a
training run loses 400 MB of VRAM and its batch size.

`enforce_cpu_only()` makes that failure impossible rather than unlikely: the device
visibility variables are cleared before torch ever initialises a context, and
`assert_gpu_untouched()` fails the run afterwards if anything did. Both the guard
and the thread cap are written into the telemetry artifact, because a timing number
measured under an unknown thread count is not comparable to anything.

THREAD CAP
----------
Default 8 of 24 cores. These benchmarks run beside a live trainer whose dataloader
needs CPU; taking every core would slow the thing we promised not to interfere with
and would inflate our own numbers with contention noise. The cap must be set before
numpy/OpenBLAS loads, so callers invoke `enforce_cpu_only()` as their first
statement, before importing numpy.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

DEVICE_ENV_VARS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_GUARD_STATE: dict[str, Any] = {"applied": False}


def parse_bootstrap_flags(argv: list[str]) -> dict[str, Any]:
    """Read the two flags that must be honoured BEFORE numpy/torch import.

    argparse runs too late: OpenBLAS reads its thread count when it loads, and torch
    reads device visibility when it first initialises a context. So these two are
    scanned off argv directly, and argparse re-declares them later for --help.
    """
    threads = 8
    if "--threads" in argv:
        i = argv.index("--threads")
        if i + 1 < len(argv):
            with contextlib.suppress(ValueError):
                threads = int(argv[i + 1])
    allow_gpu = "--gpu-capture" in argv or os.environ.get("GNN_ALLOW_GPU") == "1"
    return {"threads": threads, "allow_gpu": allow_gpu}


def enforce_cpu_only(threads: int = 8, allow_gpu: bool = False) -> dict[str, Any]:
    """Cap BLAS threads and, unless `allow_gpu`, hide every GPU. Call before importing numpy.

    `allow_gpu=True` is for the ONE step in this family that legitimately wants the
    device: capturing real activations from a base-model forward pass. The estimators
    themselves stay on the CPU whatever this is set to -- they work on 32x32 matrices and
    linear programs, where a kernel launch costs more than the whole computation.

    Returns the guard description to embed in the telemetry artifact.
    """
    if not allow_gpu:
        for var in DEVICE_ENV_VARS:
            os.environ[var] = ""
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(threads)

    numpy_already_loaded = "numpy" in sys.modules
    torch_already_loaded = "torch" in sys.modules
    if torch_already_loaded:  # thread cap still applies; device vars are read lazily
        import torch

        torch.set_num_threads(threads)

    _GUARD_STATE.update(
        {
            "applied": True,
            "threads": threads,
            "gpu_permitted": allow_gpu,
            "device_env_cleared": [] if allow_gpu else list(DEVICE_ENV_VARS),
            "numpy_imported_before_guard": numpy_already_loaded,
            "torch_imported_before_guard": torch_already_loaded,
        }
    )
    return dict(_GUARD_STATE)


def assert_gpu_untouched() -> dict[str, Any]:
    """Fail the run if a GPU context was created while the CPU-only guard was in force.

    When the guard was started with `allow_gpu=True` this records what happened instead
    of raising -- the artifact still says plainly whether a device was used.
    """
    if "torch" not in sys.modules:
        return {"torch_imported": False, "cuda_initialized": False, "verified": True}

    import torch

    initialized = bool(torch.cuda.is_initialized())
    available = bool(torch.cuda.is_available())
    if initialized and _GUARD_STATE.get("gpu_permitted"):
        return {
            "torch_imported": True,
            "cuda_initialized": True,
            "cuda_available": available,
            "device_name": torch.cuda.get_device_name(0),
            "gpu_permitted": True,
            "verified": True,
        }
    if initialized:
        raise RuntimeError(
            "GPU CONTEXT CREATED. This benchmark promised not to touch the device "
            "(a training run may hold it). Find the allocation and move it to CPU."
        )
    return {"torch_imported": True, "cuda_initialized": initialized, "cuda_available": available, "verified": True}


def median_iqr(values: list[float]) -> dict[str, float]:
    """Median + interquartile range. Linear-interpolated quantiles, not index slicing."""
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    q1, med, q3 = (float(x) for x in np.percentile(arr, [25, 50, 75]))
    return {
        "median": med,
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "n": int(arr.size),
    }


def time_repeats(
    fn: Callable[[], Any],
    repeats: int = 7,
    warmup: int = 2,
) -> tuple[Any, dict[str, float]]:
    """Run `fn` warmup+repeats times, return (last result, millisecond statistics).

    Warmup matters on CPU too: the first call pays page faults, BLAS buffer
    allocation and, for scipy solvers, one-time import of the HiGHS backend.
    """
    result = None
    for _ in range(max(0, warmup)):
        result = fn()
    timings_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)
    return result, median_iqr(timings_ms)


def platform_info() -> dict[str, Any]:
    """What the numbers were measured on. A timing without this is not reproducible."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import numpy as np

        info["numpy"] = np.__version__
    except ImportError:
        pass
    try:
        import scipy

        info["scipy"] = scipy.__version__
    except ImportError:
        pass
    return info


def write_telemetry(
    out_path: str | Path,
    payload: dict[str, Any],
    *,
    extra_config: dict[str, Any] | None = None,
) -> Path:
    """Persist a results artifact, stamped with CANON, the CPU guard and the host."""
    from runtime.canon import CANON

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    stamped = dict(payload)
    stamped["config"] = CANON.stamp()
    stamped["cpu_guard"] = dict(_GUARD_STATE)
    stamped["gpu_verification"] = assert_gpu_untouched()
    stamped["platform"] = platform_info()
    stamped["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if extra_config:
        stamped["run_config"] = extra_config

    out.write_text(json.dumps(stamped, indent=2))
    print(f"\n[telemetry] {out}", flush=True)
    return out


def print_header(title: str, subtitle: str = "", width: int = 104) -> None:
    print("=" * width, flush=True)
    print(f" {title}", flush=True)
    if subtitle:
        print(f" {subtitle}", flush=True)
    print("=" * width, flush=True)
