"""Master benchmark orchestrator: runs Aider, HumanEval, BFCL, and Toolery across all model sizes (0.8B, 2B, 4B, 9B, 27B).

Strictly enforces sequential execution: boots one runtime-next binary at a time,
runs evaluations, releases VRAM, and transitions to the next model tier.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parent.parent

MODEL_CONFIGS = {
    "0.8B": {
        "binary": REPO_ROOT / "apps" / "runtime-next" / "target_0_8b" / "release" / "runtime-next",
        "model_id": "qwen3.5:0.8b-rust",
        "d_model": 1024,
    },
    "2B": {
        "binary": REPO_ROOT / "apps" / "runtime-next" / "target_2b" / "release" / "runtime-next",
        "model_id": "qwen3.5:2b-rust",
        "d_model": 2048,
    },
    "4B": {
        "binary": REPO_ROOT / "apps" / "runtime-next" / "target" / "release" / "runtime-next",
        "model_id": "qwen3.5:4b-rust",
        "d_model": 2560,
    },
    "9B": {
        "binary": REPO_ROOT / "apps" / "runtime-next" / "target_9b" / "release" / "runtime-next",
        "model_id": "qwen3.5:9b-rust",
        "d_model": 4096,
    },
    "27B": {
        "binary": REPO_ROOT / "apps" / "runtime-next" / "target_27b" / "release" / "runtime-next",
        "model_id": "qwen3.8:27b-rust",
        "d_model": 5120,
    },
}


ADAPTER_MAP = {
    "modern": {
        "0.8B": {
            "path": REPO_ROOT / "results" / "adapters" / "m2_python_modern_r8a128_v7_0_8b",
            "name": "m2_python_modern_r8a128_v7_0_8b@0.25",
        },
        "2B": {
            "path": REPO_ROOT / "results" / "adapters" / "m2_python_modern_r8a128_v7_2b",
            "name": "m2_python_modern_r8a128_v7_2b@0.50",
        },
        "4B": {
            "path": REPO_ROOT / "results" / "adapters" / "m2_python_modern_r8a128_v7",
            "name": "m2_python_modern_r8a128_v7",
        },
        "9B": {
            "path": REPO_ROOT / "results" / "adapters" / "m2_python_modern_r8a128_v7_9b",
            "name": "m2_python_modern_r8a128_v7_9b",
        },
        "27B": {
            "path": REPO_ROOT / "results" / "adapters" / "m2_python_modern_r8a128_v7_27b",
            "name": "m2_python_modern_r8a128_v7_27b",
        },
    },
    "agentic": {
        "0.8B": {
            "path": REPO_ROOT / "results" / "adapters" / "m2_agentic_coding_r8a128_v8_0_8b",
            "name": "m2_agentic_coding_r8a128_v8_0_8b",
        },
        "2B": {
            "path": REPO_ROOT / "results" / "adapters" / "m2_agentic_coding_r8a128_v8_2b",
            "name": "m2_agentic_coding_r8a128_v8_2b",
        },
        "4B": {
            "path": REPO_ROOT / "results" / "adapters" / "m2_agentic_coding_r8a128_v8_4b",
            "name": "m2_agentic_coding_r8a128_v8_4b",
        },
        "9B": {
            "path": REPO_ROOT / "results" / "adapters" / "m2_agentic_coding_r8a128_v8_9b",
            "name": "m2_agentic_coding_r8a128_v8_9b",
        },
        "27B": {
            "path": REPO_ROOT / "results" / "adapters" / "m2_agentic_coding_r8a128_v8_27b",
            "name": "m2_agentic_coding_r8a128_v8_27b",
        },
    },
    "combined": {
        "0.8B": {
            "paths": [
                REPO_ROOT / "results" / "adapters" / "m2_agentic_coding_r8a128_v8_0_8b",
                REPO_ROOT / "results" / "adapters" / "m2_python_modern_r8a128_v7_0_8b",
            ],
            "name": "m2_agentic_coding_r8a128_v8_0_8b@0.25+m2_python_modern_r8a128_v7_0_8b@0.25",
        },
        "2B": {
            "paths": [
                REPO_ROOT / "results" / "adapters" / "m2_agentic_coding_r8a128_v8_2b",
                REPO_ROOT / "results" / "adapters" / "m2_python_modern_r8a128_v7_2b",
            ],
            "name": "m2_agentic_coding_r8a128_v8_2b@0.50+m2_python_modern_r8a128_v7_2b@0.50",
        },
        "4B": {
            "paths": [
                REPO_ROOT / "results" / "adapters" / "m2_agentic_coding_r8a128_v8_4b",
                REPO_ROOT / "results" / "adapters" / "m2_python_modern_r8a128_v7",
            ],
            "name": "m2_agentic_coding_r8a128_v8_4b+m2_python_modern_r8a128_v7",
        },
        "9B": {
            "paths": [
                REPO_ROOT / "results" / "adapters" / "m2_agentic_coding_r8a128_v8_9b",
                REPO_ROOT / "results" / "adapters" / "m2_python_modern_r8a128_v7_9b",
            ],
            "name": "m2_agentic_coding_r8a128_v8_9b+m2_python_modern_r8a128_v7_9b",
        },
        "27B": {
            "paths": [
                REPO_ROOT / "results" / "adapters" / "m2_agentic_coding_r8a128_v8_27b",
                REPO_ROOT / "results" / "adapters" / "m2_python_modern_r8a128_v7_27b",
            ],
            "name": "m2_agentic_coding_r8a128_v8_27b+m2_python_modern_r8a128_v7_27b",
        },
    },
    "alternate": {
        "0.8B": {
            "paths": [
                REPO_ROOT / "results" / "adapters" / "m2_python_modern_r8a128_v7_0_8b",
                REPO_ROOT / "results" / "adapters" / "m2_agentic_coding_r8a128_v8_0_8b",
            ],
            "name": "alternate:m2_python_modern_r8a128_v7_0_8b@0.25,m2_agentic_coding_r8a128_v8_0_8b@0.25",
        },
    },
    "sequential": {
        "0.8B": {
            "paths": [
                REPO_ROOT / "results" / "adapters" / "m2_python_modern_r8a128_v7_0_8b",
                REPO_ROOT / "results" / "adapters" / "m2_agentic_coding_r8a128_v8_0_8b",
            ],
            "name": "sequential:m2_python_modern_r8a128_v7_0_8b@0.25,m2_agentic_coding_r8a128_v8_0_8b@0.25",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Master benchmark matrix runner across all model sizes (0.8B to 27B)"
    )
    parser.add_argument(
        "--models",
        default="0.8B,2B,4B,9B,27B",
        help="Comma-separated model tiers to run (default: '0.8B,2B,4B,9B,27B')",
    )
    parser.add_argument(
        "--benchmarks",
        default="aider,humaneval,bfcl,toolery",
        help="Comma-separated benchmarks: 'aider,humaneval,bfcl,toolery' (default: all)",
    )
    parser.add_argument(
        "--adapter",
        choices=["none", "modern", "agentic", "combined", "alternate", "sequential"],
        default="none",
        help="LoRA adapter family to evaluate (default: none)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on tasks per benchmark suite (e.g. 5 or 10 for fast calibration sweep)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8003,
        help="Port for runtime-next server (default: 8003)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "benchmarks"),
        help="Directory to save individual and matrix scorecards",
    )
    return parser.parse_args()


def wait_for_server(base_url: str, timeout_seconds: int = 45) -> bool:
    t0 = time.time()
    url = f"{base_url.rstrip('/')}/models"
    while time.time() - t0 < timeout_seconds:
        try:
            with urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def stop_server(proc: subprocess.Popen, port: int):
    print(f"🛑 Stopping server (PID {proc.pid})...")
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    # Double check no lingering process on port
    subprocess.run(["fuser", "-k", f"{port}/tcp"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    time.sleep(1.5)


def run_benchmark_matrix():
    args = parse_args()
    selected_models = [m.strip().upper() for m in args.models.split(",") if m.strip()]
    selected_benches = [b.strip().lower() for b in args.benchmarks.split(",") if b.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = f"http://127.0.0.1:{args.port}/v1"

    print("=" * 80)
    print("🌐 MASTER CROSS-MODEL BENCHMARK ORCHESTRATOR")
    print(f"   Target Models     : {selected_models}")
    print(f"   Target Benchmarks : {selected_benches}")
    print(f"   Per-Suite Limit   : {args.limit or 'Full Suite'}")
    print(f"   Port              : {args.port}")
    print(f"   Output Directory  : {out_dir}")
    print("=" * 80)

    matrix_summary: dict[str, dict] = {}

    for tier in selected_models:
        if tier not in MODEL_CONFIGS:
            print(f"⚠️ Unknown model tier: '{tier}', skipping.")
            continue

        cfg = MODEL_CONFIGS[tier]
        binary = cfg["binary"]
        model_id = cfg["model_id"]

        if not binary.exists():
            print(f"❌ Binary not found for {tier}: {binary}. Run cargo build first.")
            continue

        print(f"\n================================================================================")
        print(f"🚀 BOOTING MODEL TIER: {tier} ({model_id})")
        print(f"   Binary: {binary}")
        print(f"================================================================================")

        adapter_info = ADAPTER_MAP.get(args.adapter, {}).get(tier) if args.adapter != "none" else None
        tag = args.adapter if args.adapter != "none" else "base"

        server_cmd = [str(binary), "--port", str(args.port)]
        if adapter_info:
            paths = adapter_info.get("paths") or [adapter_info.get("path")]
            for p in paths:
                if p and p.exists():
                    server_cmd.extend(["--lora", str(p)])
            print(f"   LoRA Adapter: {adapter_info['name']}")

        server_log_path = out_dir / f"server_{tier.lower()}_{tag}.log"
        server_log_file = open(server_log_path, "w")
        server_proc = subprocess.Popen(
            server_cmd,
            stdout=server_log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        try:
            print("⏳ Waiting for runtime-next to initialize weights and HIP graph...")
            if not wait_for_server(base_url, timeout_seconds=60):
                print(f"❌ Server failed to respond on {base_url} within timeout.")
                server_proc.kill()
                continue
            print(f"✅ Runtime-next {tier} successfully active on {base_url}!")

            tier_results: dict[str, dict] = {}

            # 1. Aider Python Benchmark
            if "aider" in selected_benches:
                print(f"\n--- [1/4] Running Aider Benchmark on {tier} ({tag}) ---")
                aider_out = out_dir / f"scorecard_aider_{tier.lower()}_{tag}.json"
                cmd = [
                    sys.executable,
                    "-m",
                    "benchmarks.aider_bench.runner",
                    "--base-url",
                    base_url,
                    "--model",
                    model_id,
                    "--suite",
                    "core",
                    "-o",
                    str(aider_out),
                ]
                if adapter_info:
                    cmd.extend(["--adapter", adapter_info["name"]])
                if args.limit:
                    cmd.extend(["--limit", str(args.limit)])
                subprocess.run(cmd)
                if aider_out.exists():
                    with open(aider_out) as f:
                        tier_results["aider"] = json.load(f).get("summary", {})

            # 2. HumanEval Benchmark
            if "humaneval" in selected_benches:
                print(f"\n--- [2/4] Running HumanEval Benchmark on {tier} ({tag}) ---")
                he_out = out_dir / f"scorecard_humaneval_{tier.lower()}_{tag}.json"
                cmd = [
                    sys.executable,
                    "-m",
                    "benchmarks.humaneval.runner",
                    "--base-url",
                    base_url,
                    "--model",
                    model_id,
                    "-o",
                    str(he_out),
                ]
                if adapter_info:
                    cmd.extend(["--adapter", adapter_info["name"]])
                if args.limit:
                    cmd.extend(["--limit", str(args.limit)])
                subprocess.run(cmd)
                if he_out.exists():
                    with open(he_out) as f:
                        tier_results["humaneval"] = json.load(f).get("summary", {})

            # 3. BFCL Benchmark (Strictly Non-Java)
            if "bfcl" in selected_benches:
                print(f"\n--- [3/4] Running BFCL Benchmark (Non-Java) on {tier} ({tag}) ---")
                bfcl_out = out_dir / f"scorecard_bfcl_{tier.lower()}_{tag}.json"
                cmd = [
                    sys.executable,
                    "-m",
                    "benchmarks.bfcl.runner",
                    "--base-url",
                    base_url,
                    "--model",
                    model_id,
                    "--categories",
                    "simple",
                    "-o",
                    str(bfcl_out),
                ]
                if adapter_info:
                    cmd.extend(["--adapter", adapter_info["name"]])
                if args.limit:
                    cmd.extend(["--limit", str(args.limit)])
                subprocess.run(cmd)
                if bfcl_out.exists():
                    with open(bfcl_out) as f:
                        tier_results["bfcl"] = json.load(f).get("summary", {})

            # 4. Toolery Benchmark
            if "toolery" in selected_benches:
                print(f"\n--- [4/4] Running Toolery Benchmark on {tier} ({tag}) ---")
                toolery_out = out_dir / f"scorecard_toolery_{tier.lower()}_{tag}.json"
                cmd = [
                    sys.executable,
                    "-m",
                    "benchmarks.toolery.runner",
                    "--base-url",
                    base_url,
                    "--model",
                    model_id,
                    "--tier",
                    "all",
                    "-o",
                    str(toolery_out),
                ]
                if adapter_info:
                    cmd.extend(["--adapter", adapter_info["name"]])
                if args.limit:
                    cmd.extend(["--limit", str(args.limit)])
                subprocess.run(cmd)
                if toolery_out.exists():
                    with open(toolery_out) as f:
                        tier_results["toolery"] = json.load(f).get("summary", {})

            matrix_summary[tier] = tier_results

        finally:
            stop_server(server_proc, args.port)
            try:
                server_log_file.close()
            except Exception:
                pass

    # Output Grand Summary Matrix
    print("\n" + "=" * 90)
    print("🏆 GRAND CROSS-MODEL BENCHMARK SCORECARD MATRIX")
    print("=" * 90)
    print(f"{'Tier':<8} | {'Aider Pass%':<14} | {'HumanEval Pass%':<18} | {'BFCL Acc%':<12} | {'Toolery Pass%':<14}")
    print("-" * 90)
    for tier, res in matrix_summary.items():
        aider_pct = f"{res.get('aider', {}).get('pass_rate_pct', 'N/A')}%"
        he_pct = f"{res.get('humaneval', {}).get('pass_rate_pct', 'N/A')}%"
        bfcl_pct = f"{res.get('bfcl', {}).get('accuracy_pct', 'N/A')}%"
        toolery_pct = f"{res.get('toolery', {}).get('pass_rate_pct', 'N/A')}%"
        print(f"{tier:<8} | {aider_pct:<14} | {he_pct:<18} | {bfcl_pct:<12} | {toolery_pct:<14}")
    print("=" * 90)

    matrix_file = out_dir / "matrix_cross_model_scorecard.json"
    matrix_file.write_text(json.dumps(matrix_summary, indent=2), encoding="utf-8")
    print(f"💾 Saved full matrix summary to {matrix_file}")


if __name__ == "__main__":
    run_benchmark_matrix()
