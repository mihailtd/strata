"""Real, live-server measurement of Weibull Hazard + Bollinger Band speculative
early-exit gating (apps/runtime-ipwf's RangeStatisticGate).

WHY THIS EXISTS
---------------
docs/EXPERIMENT_REAUDIT_2026-09.md flagged this cluster as Critical #1:
`RangeStatisticGate(weibull_hazard_enabled=True, bollinger_bands_enabled=True)`
is the default and is live in the speculative draft loop on both the 4B and
9B serving paths, but its only prior "performance justification" --
`experiments/runtime/speculative/weibull_hazard_gating/` -- never loaded a
model. Both retired scripts hand-built a synthetic acceptance-decay curve
shaped to contain exactly the signal the gate looks for, and computed "tok/s"
from a hardcoded linear formula (`t_base_single * (1.0 + 0.12 * k)`), not a
timed decode. See `benchmarks/superseded/weibull_hazard_gating_fabricated/`
for the retired originals and their retirement banners.

While wiring this up, two real bugs surfaced and were fixed (not benchmark
issues -- actual server bugs, see apps/RUNTIME.md / the runtime-ipwf commit
history): runtime-ipwf's request handlers called methods that did not exist
on the decoder objects (`generate()`/`stream_generate()` vs. the real
`generate_with_graph()`/`generate_tokens_stream()`), the range_gate was
constructed but never passed into the speculative draft call, and
`/api/engine/set_speculative_range_gate` -- the endpoint this benchmark relies
on -- existed only in the legacy `apps/runtime/server.py`, not here. So before
this benchmark, there was no live request path to measure in the first place.

METHODOLOGY
-----------
Real model (Qwen3.5-4B), real prompts (data/astral/evaluation_data.jsonl),
against the ACTUAL live runtime-ipwf server -- not an in-process
reimplementation of the decode loop. Arms are toggled through the real
POST /api/engine/set_speculative_range_gate endpoint; every tok/s number is
the server's own wall-clock measurement (`time.perf_counter()` around the
real speculative decode through the real CUDA graph + real MTP draft head),
read back verbatim from each response's `usage` block -- never recomputed
from a formula here.

Requires apps/runtime-ipwf's server already running with the 4B model loaded
(speculative decode is on by default):

    cd apps/runtime-ipwf && AUTO_LOAD_MODEL=1 ./run_server.sh

Arms (all through the same live speculative decoder -- only the gate config differs):
    A  gate fully disabled            enabled=false
    B  range statistic only           enabled=true,  weibull=False, bollinger=False
    C  + weibull hazard               enabled=true,  weibull=True,  bollinger=False
    D  + weibull + bollinger (SHIP)   enabled=true,  weibull=True,  bollinger=True   <- production default

Usage:
    uv run python benchmarks/runtime/speculative/weibull_hazard_gating/benchmark_weibull_hazard_gating_live.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
PROMPTS_FILE = REPO_ROOT / "data" / "astral" / "evaluation_data.jsonl"
OUT_FILE = REPO_ROOT / "results" / "benchmarks" / "weibull_hazard_gating_live.json"

ARMS: dict[str, dict[str, Any]] = {
    "A_gate_disabled": {"enabled": False},
    "B_range_only": {"enabled": True, "weibull_hazard_enabled": False, "bollinger_bands_enabled": False},
    "C_weibull": {"enabled": True, "weibull_hazard_enabled": True, "bollinger_bands_enabled": False},
    "D_weibull_bollinger_SHIP": {"enabled": True, "weibull_hazard_enabled": True, "bollinger_bands_enabled": True},
}


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
    try:
        health = _get(base_url, "/health", timeout=5)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach {base_url} -- start it first:\n"
            "    cd apps/runtime-ipwf && AUTO_LOAD_MODEL=1 ./run_server.sh"
        ) from exc
    if not health.get("engine_loaded"):
        raise RuntimeError(f"Engine not loaded at {base_url} -- POST /api/engine/load first.")
    status = _get(base_url, "/api/engine/status", timeout=5)
    if not status.get("spec_decode_enabled"):
        raise RuntimeError(
            "spec_decode_enabled=False on the live server -- this benchmark measures the "
            "gate's effect on speculative decode, which requires SPECULATIVE_DECODE=1 (the default)."
        )
    return status


def run_arm(
    base_url: str,
    arm_name: str,
    gate_config: dict[str, Any],
    prompts: list[str],
    max_tokens: int,
    repeats: int,
    timeout: float,
) -> dict[str, Any]:
    set_resp = _post(base_url, "/api/engine/set_speculative_range_gate", gate_config, timeout=10)
    print(f"  [{arm_name}] gate set: {set_resp}")

    tok_s_samples: list[float] = []
    completion_tokens_samples: list[int] = []
    wall_times: list[float] = []
    for rep in range(repeats):
        for prompt in prompts:
            payload = {
                "model": "dynamic",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": False,
            }
            t0 = time.perf_counter()
            resp = _post(base_url, "/v1/chat/completions", payload, timeout=timeout)
            client_wall_s = time.perf_counter() - t0
            usage = resp["usage"]
            tok_s_samples.append(usage["tokens_per_second"])
            completion_tokens_samples.append(usage["completion_tokens"])
            wall_times.append(client_wall_s)
        print(f"  [{arm_name}] repeat {rep + 1}/{repeats} done "
              f"(median so far: {statistics.median(tok_s_samples):.2f} tok/s)")

    return {
        "arm": arm_name,
        "gate_config": gate_config,
        "n_requests": len(tok_s_samples),
        "tok_s_median": statistics.median(tok_s_samples),
        "tok_s_mean": statistics.fmean(tok_s_samples),
        "tok_s_stdev": statistics.stdev(tok_s_samples) if len(tok_s_samples) > 1 else 0.0,
        "tok_s_samples": tok_s_samples,
        "completion_tokens_samples": completion_tokens_samples,
        "client_wall_s_samples": wall_times,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default="http://127.0.0.1:8002")
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", default=str(OUT_FILE))
    args = ap.parse_args()

    print("=" * 100)
    print("  WEIBULL HAZARD + BOLLINGER GATE -- live-server measurement")
    print("=" * 100)

    status = check_server_ready(args.base_url)
    print(f"  server: {args.base_url}  model={status.get('model_id')}  spec_k={status.get('spec_k')}")

    rows = [json.loads(line) for line in PROMPTS_FILE.read_text().splitlines() if line.strip()]
    prompts = [r["prompt"] for r in rows[: args.n_prompts]]
    print(f"  {len(prompts)} real prompts from {PROMPTS_FILE.relative_to(REPO_ROOT)}")
    print(f"  max_tokens={args.max_tokens}  repeats={args.repeats}\n")

    results: dict[str, dict[str, Any]] = {}
    for arm_name, gate_config in ARMS.items():
        results[arm_name] = run_arm(
            args.base_url, arm_name, gate_config, prompts, args.max_tokens, args.repeats, args.timeout
        )

    baseline = results["A_gate_disabled"]["tok_s_median"]
    print("\n" + "=" * 100)
    print("  RESULT (median tok/s, real server-measured decode time)")
    print("=" * 100)
    print(f"  {'arm':<28}{'tok/s (median)':>16}{'vs gate disabled':>20}")
    print("  " + "-" * 64)
    for arm_name, r in results.items():
        ratio = r["tok_s_median"] / baseline if baseline > 0 else float("nan")
        print(f"  {arm_name:<28}{r['tok_s_median']:>16.2f}{ratio:>19.3f}x")

    ship = results["D_weibull_bollinger_SHIP"]["tok_s_median"]
    ship_ratio = ship / baseline if baseline > 0 else float("nan")
    print()
    if ship_ratio > 1.0:
        print(f"  => The shipped default (weibull + bollinger) measured {ship_ratio:.3f}x "
              "over gate-disabled on this real workload.")
    else:
        print(f"  => The shipped default did NOT beat gate-disabled on this real workload "
              f"({ship_ratio:.3f}x). The prior claimed speedup does not replicate here.")
    print("  This is a single real measurement, not a proof for all workloads/hardware --")
    print("  see the README for what would strengthen or falsify it further.")

    report = {
        "device": "AMD RX 7900 XTX (ROCm)",
        "model_id": status.get("model_id"),
        "spec_k": status.get("spec_k"),
        "n_prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
        "arms": results,
        "baseline_tok_s_median": baseline,
        "ship_vs_baseline_ratio": ship_ratio,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Saved -> {out_path.relative_to(REPO_ROOT) if out_path.is_relative_to(REPO_ROOT) else out_path}")


if __name__ == "__main__":
    main()
