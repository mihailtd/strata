"""Run the active statistical-estimator benchmarks and print a consolidated verdict table.

Three benchmarks, not four: CLIME head cross-talk was retired on its own evidence and now
lives in benchmarks/superseded/clime_head_crosstalk/ (docs/DECISIONS.md §66). It is not
run here because the suite is the ACTIVE set -- run it from its own directory for
provenance.

They are independent processes with no shared state, so they run concurrently: the four
together finish in about the time the slowest one takes alone. Each writes its own
telemetry artifact to results/benchmarks/ regardless of what the others do.

  uv run python benchmarks/runtime/statistical/run_all.py
  uv run python benchmarks/runtime/statistical/run_all.py --gpu-capture   # GPU forward pass
  uv run python benchmarks/runtime/statistical/run_all.py --serial        # one at a time

--gpu-capture moves ONLY the base-model activation capture onto the device (roughly 20x
faster, which is what makes a 40k-token real sample affordable). The estimators stay on
the CPU in every mode: they work on 32x32 matrices and linear programs, where a kernel
launch costs more than the computation it would carry.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

BENCHMARKS: dict[str, dict[str, object]] = {
    "vecchia": {
        "script": HERE / "vecchia_layer_horizon" / "benchmark_vecchia_horizon.py",
        "artifact": "vecchia_layer_horizon.json",
        "captures_activations": True,
        "args": [],
    },
    "gee": {
        "script": HERE / "gee_trajectory" / "benchmark_gee_trajectory.py",
        "artifact": "gee_trajectory_drift.json",
        "captures_activations": False,
        "args": ["--replicates", "3000"],
    },
    "copula": {
        "script": HERE / "copula_routing" / "benchmark_copula_tail_routing.py",
        "artifact": "copula_tail_routing.json",
        "captures_activations": False,
        "args": ["--replicates", "300"],
    },
}


def run_one(name: str, spec: dict, threads: int, gpu_capture: bool, log_dir: Path) -> dict:
    cmd = [sys.executable, str(spec["script"]), "--threads", str(threads), *spec["args"]]
    if gpu_capture and spec["captures_activations"]:
        cmd.append("--gpu-capture")

    log_path = log_dir / f"{name}.log"
    t0 = time.perf_counter()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    return {
        "name": name,
        "returncode": proc.returncode,
        "seconds": time.perf_counter() - t0,
        "log": str(log_path),
        "artifact": REPO_ROOT / "results" / "benchmarks" / str(spec["artifact"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--gpu-capture", action="store_true",
                    help="run the base-model activation capture on the GPU (estimators stay on CPU)")
    ap.add_argument("--serial", action="store_true", help="run one at a time instead of concurrently")
    ap.add_argument("--only", nargs="*", choices=sorted(BENCHMARKS), help="run a subset")
    ap.add_argument("--log-dir", default=str(REPO_ROOT / "results" / "logs"))
    args = ap.parse_args()

    selected = {k: v for k, v in BENCHMARKS.items() if not args.only or k in args.only}
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print(" STATISTICAL ESTIMATOR BENCHMARK SUITE")
    print(f" {len(selected)} active benchmarks, {'serially' if args.serial else 'concurrently'}, "
          f"{args.threads} threads each"
          f"{', GPU activation capture' if args.gpu_capture else ''}")
    print("=" * 96, flush=True)

    t0 = time.perf_counter()
    if args.serial:
        results = [run_one(n, s, args.threads, args.gpu_capture, log_dir) for n, s in selected.items()]
    else:
        with ThreadPoolExecutor(max_workers=len(selected)) as pool:
            results = list(pool.map(
                lambda item: run_one(item[0], item[1], args.threads, args.gpu_capture, log_dir),
                selected.items(),
            ))
    wall = time.perf_counter() - t0

    print(f"\n{'benchmark':>12} | {'status':>7} | {'seconds':>8} | artifact")
    print("-" * 96)
    failures = 0
    for r in sorted(results, key=lambda r: r["name"]):
        ok = r["returncode"] == 0 and Path(r["artifact"]).exists()
        failures += int(not ok)
        print(f"{r['name']:>12} | {'ok' if ok else 'FAILED':>7} | {r['seconds']:>8.1f} | "
              f"{Path(r['artifact']).relative_to(REPO_ROOT)}")
        if not ok:
            print(f"{'':>12}   see {r['log']}")
    print("-" * 96)
    print(f"{'total wall':>12} | {'':>7} | {wall:>8.1f}")

    print("\nVERDICTS")
    print("-" * 96)
    for r in sorted(results, key=lambda r: r["name"]):
        path = Path(r["artifact"])
        if not path.exists():
            continue
        verdict = json.loads(path.read_text()).get("verdict", {})
        print(f"\n[{r['name']}]")
        for key, value in verdict.items():
            if isinstance(value, float):
                print(f"  {key:<42} {value:.4f}")
            else:
                print(f"  {key:<42} {value}")

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
