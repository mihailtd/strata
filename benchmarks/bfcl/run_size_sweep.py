"""Sequential BFCL sweep across runtime-next model sizes.

One engine at a time, enforced rather than assumed: before each size the GPU
must have no KFD processes, and after each size the server is stopped and the
GPU is re-checked before moving on (AGENTS.md single-engine invariant).

Each (size, category) pair writes its own scorecard via `benchmarks.bfcl.runner`,
so a crash mid-sweep keeps every completed result.

Usage:
    uv run python -m benchmarks.bfcl.run_size_sweep --bin-dir <dir with runtime-next-qwen35_*>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
SIZES = ["qwen35_0_8b", "qwen35_2b", "qwen35_4b", "qwen35_9b"]
CATEGORIES = ["simple", "multiple", "parallel", "parallel_multiple"]
PORT = 8003


def gpu_pids() -> str:
    out = subprocess.run(["rocm-smi", "--showpids"], capture_output=True, text=True).stdout
    return "" if "No KFD PIDs currently running" in out else out


def wait_gpu_free(timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not gpu_pids():
            return
        time.sleep(1)
    raise RuntimeError(f"GPU still has KFD processes after {timeout_s}s:\n{gpu_pids()}")


def wait_healthy(timeout_s: float = 300.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"server not healthy after {timeout_s}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", required=True, type=Path)
    ap.add_argument("--sizes", default=",".join(SIZES))
    ap.add_argument("--categories", default=",".join(CATEGORIES))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "benchmarks")
    ap.add_argument("--tag", default="", help="suffix for scorecard/log names, e.g. a model variant served via "
                    "RUNTIME_NEXT_MODEL_DIR, so its results never overwrite the base model's")
    args = ap.parse_args()

    for size in args.sizes.split(","):
        binary = args.bin_dir / f"runtime-next-{size}"
        if not binary.is_file():
            print(f"skipping {size}: no binary at {binary}", flush=True)
            continue
        if gpu_pids():
            raise RuntimeError(f"refusing to start {size}: another GPU process is running:\n{gpu_pids()}")

        tag = f"_{args.tag}" if args.tag else ""
        log = open(args.out_dir / f"server_bfcl_toolcalls_{size}{tag}.log", "w")  # noqa: SIM115 - lives for the server's lifetime
        server = subprocess.Popen([str(binary), "--port", str(PORT)], stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_healthy()
            print(f"\n##### {size}: server healthy (pid {server.pid})", flush=True)
            for cat in args.categories.split(","):
                out = args.out_dir / f"scorecard_bfcl_toolcalls_{size}{tag}_{cat}.json"
                cmd = [
                    sys.executable, "-m", "benchmarks.bfcl.runner",
                    "--model", f"{size}-rust",
                    "--categories", cat,
                    "--max-tokens", str(args.max_tokens),
                    "--output", str(out),
                ]
                if args.limit:
                    cmd += ["--limit", str(args.limit)]
                print(f"##### {size} / {cat}", flush=True)
                subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            log.close()
            wait_gpu_free()
            print(f"##### {size}: server stopped, GPU verified free", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
