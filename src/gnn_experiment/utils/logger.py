"""Zero-overhead, lightweight Python logging utilities for benchmark & experiment metrics.

Appends structured metrics directly to JSON Lines (.jsonl) files synchronously
without background threads, HTTP requests, or database overhead.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def log_benchmark_metric(
    data: dict[str, Any],
    filepath: str | Path = "results/benchmark_runs.jsonl",
) -> None:
    """Zero-overhead, pure Python logger that appends experiment metrics to a JSONL file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        **data,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
