"""Real DSH-agent-driven measurement of NOTEARS predictive pre-folding.

WHY THIS EXISTS
---------------
`experiments/agentic/benchmark_predictive_prefold.py` validated the
NotearsCausalScheduler's prediction/scheduling logic against a hand-scripted
15-turn "conversation" with simulated tool-execution time (`time.sleep()`)
and a mocked VRAM fold (`MockVRAMFoldingEngine`, hardcoded 1.9ms). It was
honestly disclosed as a mock in its own code, but its output was cited
elsewhere as unqualified "Empirical Benchmark Results" (see
docs/EXPERIMENT_REAUDIT_2026-09.md's Category 4 / `docs/DECISIONS.md` §64's
correction). This script replaces the mocked halves with real ones:

    REAL: a live DeepSeek Harness (DSH) agent session per step, driving the
          actual runtime-ipwf server (real Qwen3.5-4B, real generation).
    REAL: the actual NotearsCausalScheduler.predict_next_expert /
          async_prefold / record_turn_outcome production code, wired into
          server.py's real request-completion path (this benchmark is also
          what motivated wiring it in -- it was previously instantiated but
          never called).
    REAL: WeightFoldingEngine.activate() -- an actual real GPU weight fold
          (torch.addmm), with server-measured perf_counter() timing, not a
          sleep(). (Building this also found and fixed a real bug: activate()
          used to unconditionally re-fold even when already active, which
          would have silently defeated the entire point of pre-folding.)

HONEST LIMITATION -- what is still NOT real
--------------------------------------------
apps/runtime-ipwf/server.py has no OpenAI-style function/tool-calling support
at all (`ChatCompletionRequest` uses `extra="ignore"`, silently dropping any
`tools=` DSH sends). Confirmed live: DSH's bash tool never actually fires --
the model only describes commands in prose/code-block text, it doesn't
execute them. Implementing real function-calling is a separate, substantial
feature (see docs/EXPERIMENT_REAUDIT_2026-09.md), not attempted here.

What this means for the measurement: `detect_tools()` (real production code,
unchanged) pattern-matches the agent's real generated TEXT for tool markers
(`uv add`, `CREATE TABLE`, etc.) -- this works identically whether the agent
actually executed a tool or just wrote/mentioned the command, since the
scheduler was always driven by text pattern detection, not structured tool-
call events. So the prediction/pre-fold mechanism is exercised for real, on
real model output, in a real multi-turn DSH session -- it just isn't gated on
verified real file changes the way a true coding-agent eval should be. Do not
cite this as "the agent completed a real coding task."

METHODOLOGY
-----------
5 steps mirroring the canonical pipeline the scheduler was seeded with
(astral -> postgresql -> duckdb -> python_web -> python_modern), each a fresh
DSH harness + session (DSH does not support resuming one session_id across
harness instances -- confirmed live, `session "X" already exists`), targeting
the domain-appropriate model alias in sequence. Run twice against the same
live server: prefolding disabled (Arm A) and enabled (Arm B), toggled live via
POST /api/engine/set_predictive_prefold. Real per-transition swap latency and
hit/miss counts come from the server's own `causal_scheduler.telemetry` and
`model_state["last_swap_ms"]` -- read via GET /api/engine/status, never
recomputed here.

Requires apps/runtime-ipwf's server already running with the 4B model loaded:
    cd apps/runtime-ipwf && AUTO_LOAD_MODEL=1 ./run_server.sh

Usage:
    uv run python benchmarks/agentic/predictive_prefold_live/benchmark_predictive_prefold_live.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_FILE = REPO_ROOT / "results" / "benchmarks" / "predictive_prefold_live.json"
SCRATCH_ROOT = Path("/tmp/predictive_prefold_live")

# Mirrors the first canonical pipeline notears_causal_scheduler.py was seeded
# with: (["uv_init", "uv_add"], astral, ["sql_ddl"], postgresql), etc.
STEPS = [
    ("astral", "Explain, with the exact commands, how to initialize a new uv project "
               "named 'service' and add fastapi and asyncpg as dependencies."),
    ("postgresql", "Write the SQL DDL to create a 'users' table with an 'id' primary key, "
                   "an 'email' text column, and an 'embedding vector(768)' column with an "
                   "HNSW index."),
    ("duckdb", "Write a short Python snippet using duckdb to read a partitioned parquet "
               "dataset and export aggregated summary stats to another parquet file."),
    ("python_web", "Write a FastAPI route at /users/{user_id} using a Pydantic response "
                   "model, returning a JSON dict."),
    ("python_modern", "Write a pytest test function for the /users/{user_id} route above "
                      "using FastAPI's TestClient and Python 3.12 type hints."),
]


def _post(base_url: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}{path}", data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _get(base_url: str, path: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def check_server_ready(base_url: str) -> dict[str, Any]:
    health = _get(base_url, "/health", timeout=5)
    if not health.get("engine_loaded"):
        raise RuntimeError(f"Engine not loaded at {base_url} -- POST /api/engine/load first.")
    return _get(base_url, "/api/engine/status", timeout=5)


def run_dsh_step(
    model: str, prompt: str, cwd: Path, dsh_home: Path, session_id: str, base_url: str
) -> tuple[str, float]:
    """Drives one real DSH agent turn against the live server. Returns (response_text, wall_s)."""
    os.environ.setdefault("DEEPSEEK_API_KEY", "sk-local-dev-key")
    os.environ["DEEPSEEK_BASE_URL"] = base_url.rstrip("/") + "/v1"

    from deepseek_harness import DeepSeekHarness

    t0 = time.perf_counter()
    with DeepSeekHarness(
        provider="deepseek-official",
        model=model,
        cwd=str(cwd),
        dsh_home=str(dsh_home),
        profile="sdk-minimal",
        max_tokens=200,
        request_timeout_seconds=120,
    ) as harness:
        result = harness.run(prompt, session_id=session_id)
    return (getattr(result, "final_response", "") or ""), time.perf_counter() - t0


def run_arm(arm_name: str, prefold_enabled: bool, base_url: str, run_id: str) -> dict[str, Any]:
    print(f"\n{'=' * 90}\n  ARM {arm_name}: prefold_enabled={prefold_enabled}\n{'=' * 90}")
    set_resp = _post(base_url, "/api/engine/set_predictive_prefold", {"enabled": prefold_enabled}, timeout=10)
    print(f"  toggle: {set_resp}")

    workspace = SCRATCH_ROOT / run_id / arm_name / "workspace"
    dsh_home = SCRATCH_ROOT / run_id / arm_name / "dsh_home"
    workspace.mkdir(parents=True, exist_ok=True)

    step_records = []
    for i, (domain, prompt) in enumerate(STEPS):
        before = _get(base_url, "/api/engine/status", timeout=10)
        swap_ms_before = float(before.get("cumulative_swap_ms") or 0.0)
        swap_count_before = int(before.get("cumulative_swap_count") or 0)

        response, wall_s = run_dsh_step(
            model=domain, prompt=prompt, cwd=workspace, dsh_home=dsh_home,
            session_id=f"{run_id}-{arm_name}-step{i}", base_url=base_url,
        )
        status = _get(base_url, "/api/engine/status", timeout=10)
        sched = status.get("causal_scheduler_telemetry") or {}
        # The REAL, server-measured cost of this step's expert switch(es) --
        # diffed cumulative counters, not "last swap", since a single DSH
        # .run() call can trigger more than one real completion (and hence
        # more than one _apply_expert call) per step; "last" would silently
        # drop everything but the final one. 0.0 delta means every real
        # activate() this step found the folding engine already correct
        # (a real pre-fold hit, or already active from the previous step).
        swap_ms = float(status.get("cumulative_swap_ms") or 0.0) - swap_ms_before
        swap_count = int(status.get("cumulative_swap_count") or 0) - swap_count_before
        step_records.append({
            "step": i,
            "domain": domain,
            "wall_s": round(wall_s, 2),
            "response_chars": len(response),
            "real_swap_ms": round(swap_ms, 3),
            "real_swap_count": swap_count,
            "scheduler_telemetry_after": dict(sched),
        })
        print(f"  step {i} [{domain:14s}] {wall_s:6.2f}s  real_swap_ms={swap_ms:7.2f} (x{swap_count})  "
              f"hits={sched.get('prefold_hits')} misses={sched.get('prefold_misses')} "
              f"last_pred={sched.get('last_prediction')}({sched.get('last_confidence')})")

    final_status = _get(base_url, "/api/engine/status", timeout=10)
    return {
        "arm": arm_name,
        "prefold_enabled": prefold_enabled,
        "steps": step_records,
        "total_real_swap_ms": round(sum(s["real_swap_ms"] for s in step_records), 3),
        "total_real_swap_count": sum(s["real_swap_count"] for s in step_records),
        "final_scheduler_telemetry": final_status.get("causal_scheduler_telemetry"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8002")
    ap.add_argument("--out", default=str(OUT_FILE))
    args = ap.parse_args()

    run_id = time.strftime("%Y%m%dT%H%M%S")
    print("=" * 90)
    print("  PREDICTIVE PRE-FOLDING -- real DSH-agent-driven measurement")
    print("=" * 90)
    status = check_server_ready(args.base_url)
    print(f"  server: {args.base_url}  model={status.get('model_id')}")
    if status.get("causal_scheduler_telemetry") is None:
        raise RuntimeError("No causal_scheduler on the live server -- is a model loaded?")

    arm_a = run_arm("A_reactive_only", prefold_enabled=False, base_url=args.base_url, run_id=run_id)
    arm_b = run_arm("B_notears_prefold", prefold_enabled=True, base_url=args.base_url, run_id=run_id)

    print("\n" + "=" * 90)
    print("  RESULT")
    print("=" * 90)
    for arm in (arm_a, arm_b):
        t = arm["final_scheduler_telemetry"] or {}
        print(f"  {arm['arm']:20s} total_real_swap_ms={arm['total_real_swap_ms']:.2f}  "
              f"hits={t.get('prefold_hits', 0)} misses={t.get('prefold_misses', 0)}")
    a_ms, b_ms = arm_a["total_real_swap_ms"], arm_b["total_real_swap_ms"]
    if a_ms > 0:
        print(f"\n  Arm B real swap overhead is {b_ms / a_ms:.3f}x of Arm A "
              f"({a_ms - b_ms:+.2f}ms {'saved' if b_ms < a_ms else 'added'} over {len(STEPS)} steps)")

    report = {"run_id": run_id, "base_url": args.base_url, "arms": {"A": arm_a, "B": arm_b}}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out_path}")


if __name__ == "__main__":
    main()
