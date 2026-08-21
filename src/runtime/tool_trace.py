"""Append-only recorder for routing decisions and detected tool calls.

WHY THIS EXISTS
---------------
`benchmarks/factory/agentic/poet_tool_causal_graph/` fits NOTEARS on
`simulate_tool_execution_data()` -- `np.random.seed(42)`, a hand-built ground-truth
DAG, Gaussian noise. §47's conclusion (never POET-filter a causal graph; raw counts
are optimal) is a real result about the ALGORITHM. It says nothing about this
runtime, because no observed trace has ever been recorded.

The open question needs real data:

    Can the runtime predict the next expert well enough to pre-fold it in the
    background while the current tool runs?  P(Expert_{t+1} | Tool_t) > 0.95?

Pre-folding is worth ~18 ms per transition if the prediction holds and costs a
wasted fold plus a stall if it does not, so the threshold is the whole decision.

WHAT IT COSTS
-------------
One dict and one line of JSON per routed request, on the CPU side of the request
path, after the routing decision is already made. No GPU work, no extra tensor.
Disabled unless GNN_TOOL_TRACE is set, so it cannot surprise a benchmark.

    export GNN_TOOL_TRACE=results/logs/tool_trace.jsonl
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

# Surface markers for tools an answer actually reached for. Deliberately narrow and
# anchored: substring matching is what made the intent classifier fire "rest" inside
# "interest" (DECISIONS.md §60).
TOOL_PATTERNS: dict[str, str] = {
    "uv_add": r"\buv add\b",
    "uv_lock": r"\buv lock\b",
    "uv_sync": r"\buv sync\b",
    "uv_run": r"\buv run\b",
    "uv_init": r"\buv init\b",
    "ruff_check": r"\bruff check\b",
    "ruff_format": r"\bruff format\b",
    "ty_check": r"\bty check\b",
    "pytest": r"\bpytest\b",
    "sql_ddl": r"\bCREATE (TABLE|INDEX|EXTENSION)\b",
    "sql_select": r"\bSELECT\b[\s\S]{0,400}?\bFROM\b",
    "pgvector": r"<=>|<->|<#>|\bhnsw\b|\bivfflat\b",
    "asyncpg": r"\basyncpg\b|\bcreate_pool\b",
    "duckdb_query": r"\bduckdb\b|\bread_parquet\b",
    "export_parquet": r"\bto_parquet\b|\bCOPY\b[\s\S]{0,120}?\bPARQUET\b",
    "fastapi_route": r"@(app|router)\.(get|post|put|patch|delete)\b",
    "pydantic_model": r"\bBaseModel\b",
    "dataclass": r"@dataclass\b",
    "git_commit": r"\bgit commit\b",
}
_COMPILED = {k: re.compile(v, re.I) for k, v in TOOL_PATTERNS.items()}

_lock = threading.Lock()


def trace_path() -> Path | None:
    p = os.environ.get("GNN_TOOL_TRACE")
    return Path(p) if p else None


def detect_tools(text: str) -> list[str]:
    """Which tool surfaces appear in an emitted answer."""
    return sorted(k for k, rx in _COMPILED.items() if rx.search(text))


def record(
    *,
    session: str,
    turn: int,
    prompt: str,
    scores: dict[str, float],
    team: list[str],
    answer: str = "",
    extra: dict | None = None,
) -> None:
    """Append one routing event. Never raises into the request path."""
    path = trace_path()
    if path is None:
        return
    try:
        row = {
            "ts": round(time.time(), 3),
            "session": session,
            "turn": turn,
            # The prompt itself is not stored -- traces get shared, prompts are user
            # data. Length and the intent vector are what the DAG fit needs.
            "prompt_len": len(prompt),
            "scores": {k: round(float(v), 4) for k, v in scores.items()},
            "team": list(team),
            "primary": team[0] if team else None,
            "tools": detect_tools(answer) if answer else [],
        }
        if extra:
            row["extra"] = extra
        line = json.dumps(row, separators=(",", ":"))
        with _lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except Exception as exc:                      # never break serving over telemetry
        print(f"[tool_trace] disabled for this event: {type(exc).__name__}: {exc}")
