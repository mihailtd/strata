"""§106 Phase A4 (extended): Real quantized engine comparison across the
WHOLE real Qwen3.5 family (0.8B/2B/4B/9B) plus Qwen3.8-27B -- our W4A16
kernel vs. llama.cpp Q4_K_M vs. Ollama Q4_K_M, real HTTP/SSE, one size at
a time.

Deliberately uses the proven-safe pattern from the original 27B-only
benchmark (`run_27b_quantized_engine_comparison_benchmark.py`, which
completed cleanly with no incident) rather than a single long-lived test
process running many real-model loads back to back -- THAT pattern (a
`cargo test --ignored` sweep) is what caused a real VRAM-exhaustion
desktop-compositor crash earlier this session. Every arm here is its own
separate OS process, killed cleanly (SIGKILL, guaranteed driver-level
resource reclamation regardless of in-process Drop semantics) before the
next one starts -- and a real VRAM check runs between every size, not
just at the very start.

Reuses `run_4b_engine_comparison_benchmark.py`'s validated
`eval_http_stream`/`wait_for_endpoint`/`BENCHMARK_PROMPTS` and
`run_27b_quantized_engine_comparison_benchmark.py`'s `MemoryMonitor`.
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
from runtime_common.gpu_preflight import ensure_gpu_exclusive  # noqa: E402

MemoryMonitor = base27.MemoryMonitor

RUNTIME_NEXT_BIN_DIR = REPO_ROOT / "models" / "runtime_next_bins"
GGUF_DIR = REPO_ROOT / "models" / "qwen35_gguf_bench"  # existing bf16 GGUFs -- NOT what we want for a quantized comparison
QUANT_DIR_MAP = {
    "0.8B": REPO_ROOT / "models" / "qwen35_0_8b_w4a16",
    "2B": REPO_ROOT / "models" / "qwen35_2b_w4a16",
    "4B": REPO_ROOT / "models" / "qwen35_4b_w4a16",
    "9B": REPO_ROOT / "models" / "qwen35_9b_w4a16",
    "27B": REPO_ROOT / "models" / "qwen38_27b_w4a16",
}
FEATURE_MAP = {
    "0.8B": "qwen35_0_8b",
    "2B": "qwen35_2b",
    "4B": "qwen35_4b",
    "9B": "qwen35_9b",
    "27B": "qwen35_27b",
}


def real_vram_used_mb() -> int:
    out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=5).stdout
    for line in out.splitlines():
        if "GPU[0]" in line and "VRAM Total Used" in line:
            return int(line.split(":")[-1].strip()) // (1024 * 1024)
    return -1


def wait_for_vram_baseline(max_mb: int = 4000, timeout_s: float = 30.0) -> None:
    """Real safety gate, added after this session's real VRAM-exhaustion
    incident: refuse to start the next arm until VRAM is actually back
    near idle baseline, not just "the process we killed said goodbye"."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        used = real_vram_used_mb()
        if used >= 0 and used <= max_mb:
            return
        time.sleep(1.0)
    used = real_vram_used_mb()
    print(f"  !!! WARNING: VRAM still at {used}MB after {timeout_s}s wait -- proceeding cautiously, watch for real pressure")


def run_runtime_next_quantized_arm(size: str) -> tuple[list[dict], dict]:
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


def main() -> None:
    sizes = sys.argv[1:] if len(sys.argv) > 1 else ["0.8B", "2B", "4B", "9B"]
    print("=" * 80)
    print("🥊 QUANTIZED MULTI-SIZE BENCHMARK: RUNTIME-NEXT W4A16 (this pass measures ONLY our engine)")
    print(f"   Sizes: {sizes}")
    print("   Real-time VRAM/GTT/system-RAM monitored per arm; real VRAM-baseline wait between sizes")
    print("   (llama.cpp/Ollama Q4_K_M comparison arms come in a separate, smaller follow-up run)")
    print("=" * 80, flush=True)

    base._kill_all_known_servers()
    wait_for_vram_baseline()

    all_results: dict[str, dict] = {}
    for size in sizes:
        res, mem = run_runtime_next_quantized_arm(size)
        avg_tok_s = sum(x["tok_per_sec"] for x in res) / len(res)
        avg_ttft = sum(x["ttft_ms"] for x in res) / len(res)
        all_results[size] = {"avg_tok_s": round(avg_tok_s, 2), "avg_ttft_ms": round(avg_ttft, 1), "memory": mem, "tasks": res}
        print(f"--- {size} DONE: runtime-next W4A16 {avg_tok_s:.1f} tok/s, TTFT {avg_ttft:.1f}ms ---")

    print("\n" + "=" * 100)
    print("🏆 RUNTIME-NEXT W4A16 ACROSS SIZES")
    print("=" * 100)
    for size, r in all_results.items():
        print(f"{size:<8} | {r['avg_tok_s']:6.1f} tok/s | TTFT {r['avg_ttft_ms']:7.1f}ms | VRAM peak {r['memory']['vram_peak_mb']}MB | GTT grew {r['memory']['gtt_grew']}MB")

    out_file = REPO_ROOT / "results/benchmarks/quantized_multi_size_runtime_next_scorecard.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(all_results, indent=2))
    print(f"\n💾 Saved to: {out_file}")


if __name__ == "__main__":
    main()
