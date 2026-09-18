"""§106 Phase A4 (full extension): Real 3-way quantized engine comparison
across the WHOLE real Qwen3.5 dense family (0.8B/2B/4B/9B) -- our W4A16
kernel vs. llama.cpp Q4_K_M vs. Ollama Q4_K_M, real HTTP/SSE, one size at
a time. Direct sibling of `run_27b_quantized_engine_comparison_benchmark.py`
(same 3-arm structure, same real memory monitoring) generalized the same
way `run_quantized_multi_size_benchmark.py` generalized the runtime-next-
only arm across sizes.

The llama.cpp/Ollama Q4_K_M GGUFs for 0.8B/2B/4B/9B did NOT exist on disk
before this script -- the only prior GGUFs for these sizes
(`models/qwen35_gguf_bench/*.gguf`) were real but BF16 (confirmed via
`ollama show`), left over from the original unquantized breakthrough
benchmark. Real Q4_K_M versions were produced via
`apps/runtime-llama/llama.cpp/build/bin/llama-quantize` (CPU-only, no GPU
involved, safe) into `models/qwen35_gguf_q4km/`, then imported into Ollama
the same way the existing 27B Q4_K_M arm does (a Modelfile `FROM` a real
GGUF path).

Deliberately uses the proven-safe pattern from this session's real
GPU-crash incident: every arm is its own separate OS process, killed
cleanly (SIGKILL) before the next one starts, with a real
`wait_for_vram_baseline()` gate between every arm -- never one long-lived
process running many real-model loads back to back.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "harness_sdk"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "runtime-common" / "src"))

import run_4b_engine_comparison_benchmark as base  # noqa: E402
import run_27b_quantized_engine_comparison_benchmark as base27  # noqa: E402
import run_quantized_multi_size_benchmark as base_multi  # noqa: E402
from runtime_common.gpu_preflight import ensure_gpu_exclusive  # noqa: E402

MemoryMonitor = base27.MemoryMonitor
wait_for_vram_baseline = base_multi.wait_for_vram_baseline

RUNTIME_NEXT_BIN_DIR = REPO_ROOT / "models" / "runtime_next_bins"
Q4KM_GGUF_MAP = {
    "0.8B": REPO_ROOT / "models" / "qwen35_gguf_q4km" / "qwen3.5-0.8b-q4km.gguf",
    "2B": REPO_ROOT / "models" / "qwen35_gguf_q4km" / "qwen3.5-2b-q4km.gguf",
    "4B": REPO_ROOT / "models" / "qwen35_gguf_q4km" / "qwen3.5-4b-q4km.gguf",
    "9B": REPO_ROOT / "models" / "qwen35_gguf_q4km" / "qwen3.5-9b-q4km.gguf",
}
QUANT_DIR_MAP = {
    "0.8B": REPO_ROOT / "models" / "qwen35_0_8b_w4a16",
    "2B": REPO_ROOT / "models" / "qwen35_2b_w4a16",
    "4B": REPO_ROOT / "models" / "qwen35_4b_w4a16",
    "9B": REPO_ROOT / "models" / "qwen35_9b_w4a16",
}
FEATURE_MAP = {
    "0.8B": "qwen35_0_8b",
    "2B": "qwen35_2b",
    "4B": "qwen35_4b",
    "9B": "qwen35_9b",
}
OLLAMA_MODEL_MAP = {size: f"qwen3.5:{size.lower()}-q4km-bench" for size in Q4KM_GGUF_MAP}


def run_llamacpp_arm(size: str) -> tuple[list[dict], dict]:
    print(f"\n{'='*80}\n▶ [{size}] LLAMA.CPP Q4_K_M (Port 8001, real GGUF, HTTP/SSE)\n{'='*80}", flush=True)
    ensure_gpu_exclusive()

    gguf = Q4KM_GGUF_MAP[size]
    if not gguf.is_file():
        raise FileNotFoundError(f"Q4_K_M GGUF not found: {gguf}")

    bin_dir = REPO_ROOT / "apps/runtime-llama/llama.cpp/build/bin"
    llama_server = bin_dir / "llama-server"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    with MemoryMonitor(f"llama.cpp-q4km-{size}") as mon:
        proc = subprocess.Popen(
            [str(llama_server), "--model", str(gguf), "--host", "127.0.0.1", "--port", "8001",
             "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "8192",
             "-b", "512", "-ub", "512", "-t", "8", "-np", "1", "--no-webui"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            base.wait_for_endpoint("http://127.0.0.1:8001/v1/models")
            res = base.eval_http_stream("http://127.0.0.1:8001/v1/chat/completions", f"qwen3.5:{size}-llamacpp-q4km", base.BENCHMARK_PROMPTS, f"llama.cpp Q4_K_M {size}")
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except Exception:  # noqa: BLE001
                proc.kill()
            subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
            time.sleep(2.0)
    mon.report()
    wait_for_vram_baseline()
    return res, mon.summary()


def run_ollama_arm(size: str) -> tuple[list[dict], dict]:
    print(f"\n{'='*80}\n▶ [{size}] OLLAMA Q4_K_M (Port 11434, real GGUF, HTTP/SSE)\n{'='*80}", flush=True)
    ensure_gpu_exclusive()

    gguf = Q4KM_GGUF_MAP[size]
    model_name = OLLAMA_MODEL_MAP[size]
    with MemoryMonitor(f"ollama-q4km-{size}") as mon:
        proc = subprocess.Popen(
            ["bash", str(REPO_ROOT / "apps/runtime-ollama/run_server.sh")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            base.wait_for_endpoint("http://127.0.0.1:11434/api/version")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".Modelfile", delete=False) as f:
                f.write(f"FROM {gguf}\n")
                modelfile_path = f.name
            try:
                subprocess.run(["ollama", "create", model_name, "-f", modelfile_path], check=True, timeout=300)
            finally:
                os.unlink(modelfile_path)
            res = base.eval_http_stream("http://127.0.0.1:11434/v1/chat/completions", model_name, base.BENCHMARK_PROMPTS, f"Ollama Q4_K_M {size}")
        finally:
            subprocess.run(["ollama", "stop", model_name], capture_output=True)
            subprocess.run(["bash", str(REPO_ROOT / "apps/runtime-ollama/run_server.sh"), "stop"], capture_output=True)
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except Exception:  # noqa: BLE001
                proc.kill()
            time.sleep(2.0)
    mon.report()
    wait_for_vram_baseline()
    return res, mon.summary()


def run_runtime_next_arm(size: str) -> tuple[list[dict], dict]:
    feature = FEATURE_MAP[size]
    binary = RUNTIME_NEXT_BIN_DIR / f"runtime-next-{feature}"
    quant_dir = QUANT_DIR_MAP[size]
    print(f"\n{'='*80}\n▶ [{size}] RUNTIME-NEXT W4A16 (Port 8003, real quantized weights, HTTP/SSE)\n{'='*80}", flush=True)
    ensure_gpu_exclusive()
    if not binary.is_file():
        raise FileNotFoundError(f"binary not found: {binary}")
    if not quant_dir.is_dir():
        raise FileNotFoundError(f"quantized checkpoint not found: {quant_dir}")

    env = os.environ.copy()
    env["PORT"] = "8003"
    env["RUNTIME_NEXT_MODEL_DIR"] = str(quant_dir)
    with MemoryMonitor(f"runtime-next-w4a16-{size}") as mon:
        proc = subprocess.Popen([str(binary)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            base.wait_for_endpoint("http://127.0.0.1:8003/health", timeout_s=180.0)
            res = base.eval_http_stream("http://127.0.0.1:8003/v1/chat/completions", f"qwen3.5:{size}-w4a16", base.BENCHMARK_PROMPTS, f"runtime-next W4A16 {size}")
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except Exception:  # noqa: BLE001
                proc.kill()
            subprocess.run(["pkill", "-9", "-f", f"runtime-next-{feature}"], capture_output=True)
            subprocess.run(["pkill", "-9", "-f", "target/release/runtime-next"], capture_output=True)
            time.sleep(2.0)
    mon.report()
    wait_for_vram_baseline()
    return res, mon.summary()


def run_one_size(size: str) -> dict:
    llamacpp_res, llamacpp_mem = run_llamacpp_arm(size)
    ollama_res, ollama_mem = run_ollama_arm(size)
    runtime_next_res, runtime_next_mem = run_runtime_next_arm(size)

    avg_llamacpp = sum(x["tok_per_sec"] for x in llamacpp_res) / len(llamacpp_res)
    avg_ollama = sum(x["tok_per_sec"] for x in ollama_res) / len(ollama_res)
    avg_runtime_next = sum(x["tok_per_sec"] for x in runtime_next_res) / len(runtime_next_res)
    avg_ttft_llamacpp = sum(x["ttft_ms"] for x in llamacpp_res) / len(llamacpp_res)
    avg_ttft_ollama = sum(x["ttft_ms"] for x in ollama_res) / len(ollama_res)
    avg_ttft_runtime_next = sum(x["ttft_ms"] for x in runtime_next_res) / len(runtime_next_res)

    print(f"\n--- {size} DONE ---")
    print(f"  llama.cpp Q4_K_M   : {avg_llamacpp:6.1f} tok/s | TTFT {avg_ttft_llamacpp:7.1f}ms")
    print(f"  Ollama Q4_K_M      : {avg_ollama:6.1f} tok/s | TTFT {avg_ttft_ollama:7.1f}ms")
    print(f"  runtime-next W4A16 : {avg_runtime_next:6.1f} tok/s | TTFT {avg_ttft_runtime_next:7.1f}ms | "
          f"speedup vs llama.cpp: {avg_runtime_next/max(1e-5,avg_llamacpp):.2f}x, vs Ollama: {avg_runtime_next/max(1e-5,avg_ollama):.2f}x")

    return {
        "avg_llamacpp_tok_s": round(avg_llamacpp, 2),
        "avg_ollama_tok_s": round(avg_ollama, 2),
        "avg_runtime_next_tok_s": round(avg_runtime_next, 2),
        "runtime_next_speedup_vs_llamacpp": round(avg_runtime_next / max(1e-5, avg_llamacpp), 3),
        "runtime_next_speedup_vs_ollama": round(avg_runtime_next / max(1e-5, avg_ollama), 3),
        "avg_ttft_ms": {
            "llamacpp": round(avg_ttft_llamacpp, 1),
            "ollama": round(avg_ttft_ollama, 1),
            "runtime_next": round(avg_ttft_runtime_next, 1),
        },
        "memory_footprint": {"llamacpp": llamacpp_mem, "ollama": ollama_mem, "runtime_next": runtime_next_mem},
        "tasks": {"llamacpp": llamacpp_res, "ollama": ollama_res, "runtime_next": runtime_next_res},
    }


def main() -> None:
    sizes = sys.argv[1:] if len(sys.argv) > 1 else ["0.8B", "2B", "4B", "9B"]
    print("=" * 80)
    print("🥊 QUANTIZED MULTI-SIZE 3-WAY ENGINE COMPARISON: llama.cpp Q4_K_M vs Ollama Q4_K_M vs runtime-next W4A16")
    print(f"   Sizes: {sizes}")
    print("   Real-time VRAM/GTT/system-RAM monitored per arm; real VRAM-baseline wait between every arm")
    print("=" * 80, flush=True)

    base._kill_all_known_servers()
    wait_for_vram_baseline()

    out_file = REPO_ROOT / "results/benchmarks/quantized_multi_size_engine_comparison_scorecard.json"
    all_results: dict[str, dict] = {}
    if out_file.is_file():
        try:
            all_results = json.loads(out_file.read_text())
        except Exception:  # noqa: BLE001
            all_results = {}

    for size in sizes:
        all_results[size] = run_one_size(size)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(all_results, indent=2))
        print(f"💾 Progress saved to: {out_file}")

    print("\n" + "=" * 100)
    print("🏆 QUANTIZED MULTI-SIZE 3-WAY SCORECARD (all sizes gathered so far)")
    print("=" * 100)
    print(f"{'Size':<6} | {'llama.cpp':<12} | {'Ollama':<12} | {'runtime-next':<12} | {'vs llama.cpp':<12} | {'vs Ollama'}")
    print("-" * 100)
    for size, r in all_results.items():
        print(f"{size:<6} | {r['avg_llamacpp_tok_s']:7.1f} tok/s | {r['avg_ollama_tok_s']:7.1f} tok/s | "
              f"{r['avg_runtime_next_tok_s']:7.1f} tok/s | {r['runtime_next_speedup_vs_llamacpp']:.2f}x       | {r['runtime_next_speedup_vs_ollama']:.2f}x")
    print(f"\n💾 Full results saved to: {out_file}")


if __name__ == "__main__":
    main()
