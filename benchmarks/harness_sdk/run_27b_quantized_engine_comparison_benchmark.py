"""§106 Phase A4: Real 27B quantized engine comparison -- our W4A16 kernel
vs. llama.cpp Q4_K_M vs. Ollama Q4_K_M, real HTTP/SSE, same real GGUF for
the two upstream arms, our own real converted checkpoint for the third.

Every arm is wrapped in a REAL memory monitor (VRAM/GTT/system-RAM,
sampled every 0.5s via `rocm-smi`/`free`) -- the user explicitly asked for
this after observing real extra system-memory pressure during earlier
manual testing. Peak VRAM/GTT/sys-RAM is reported for every arm, and any
arm whose GTT usage grows in lockstep with its own load is flagged
loudly, not silently trusted.

Reuses `run_4b_engine_comparison_benchmark.py`'s own validated
`eval_http_stream`/`wait_for_endpoint`/`BENCHMARK_PROMPTS` -- same real
methodology, not reinvented.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "harness_sdk"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "runtime-common" / "src"))

import run_4b_engine_comparison_benchmark as base  # noqa: E402
from runtime_common.gpu_preflight import ensure_gpu_exclusive  # noqa: E402

GGUF_27B = "/var/lib/ollama/blobs/sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d"
QUANTIZED_DIR = REPO_ROOT / "models" / "qwen38_27b_w4a16"
RUNTIME_NEXT_BIN = REPO_ROOT / "apps/runtime-next/target/release/runtime-next"
OLLAMA_MODEL_NAME = "qwen3.8:27b-q4km-bench"


class MemoryMonitor:
    """Real-time VRAM/GTT/system-RAM sampler (rocm-smi + free, every
    0.5s) wrapped around one benchmark arm's real run. Reports PEAK usage,
    not a single snapshot -- a transient spike during model load is
    exactly what matters here, and a single before/after check would
    miss it."""

    def __init__(self, label: str):
        self.label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict] = []

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            row = {"t": time.time()}
            try:
                out = subprocess.run(["rocm-smi", "--showmeminfo", "all"], capture_output=True, text=True, timeout=5).stdout
                for line in out.splitlines():
                    if "GPU[0]" in line and "VRAM Total Used" in line:
                        row["vram_mb"] = int(line.split(":")[-1].strip()) // (1024 * 1024)
                    elif "GPU[0]" in line and "GTT Total Used" in line:
                        row["gtt_mb"] = int(line.split(":")[-1].strip()) // (1024 * 1024)
            except Exception:  # noqa: BLE001
                pass
            try:
                free_out = subprocess.run(["free", "-m"], capture_output=True, text=True, timeout=5).stdout
                for line in free_out.splitlines():
                    if line.startswith("Mem:"):
                        parts = line.split()
                        row["sys_used_mb"] = int(parts[2])
            except Exception:  # noqa: BLE001
                pass
            self.samples.append(row)
            self._stop.wait(0.5)

    def __enter__(self) -> "MemoryMonitor":
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def summary(self) -> dict:
        vram = [s["vram_mb"] for s in self.samples if "vram_mb" in s]
        gtt = [s["gtt_mb"] for s in self.samples if "gtt_mb" in s]
        sys_used = [s["sys_used_mb"] for s in self.samples if "sys_used_mb" in s]
        return {
            "vram_peak_mb": max(vram) if vram else None,
            "vram_min_mb": min(vram) if vram else None,
            "gtt_peak_mb": max(gtt) if gtt else None,
            "gtt_min_mb": min(gtt) if gtt else None,
            "gtt_grew": (max(gtt) - min(gtt)) if gtt else 0,
            "sys_used_peak_mb": max(sys_used) if sys_used else None,
            "sys_used_min_mb": min(sys_used) if sys_used else None,
            "n_samples": len(self.samples),
        }

    def report(self) -> None:
        s = self.summary()
        print(f"  [mem/{self.label}] VRAM peak={s['vram_peak_mb']}MB (min={s['vram_min_mb']}MB) | "
              f"GTT peak={s['gtt_peak_mb']}MB grew={s['gtt_grew']}MB | "
              f"sys RAM peak={s['sys_used_peak_mb']}MB (min={s['sys_used_min_mb']}MB) | {s['n_samples']} samples")
        if s["gtt_grew"] and s["gtt_grew"] > 512:
            print(f"  [mem/{self.label}] !!! GTT grew by {s['gtt_grew']}MB during this arm -- "
                  f"real VRAM-to-system-RAM spillover, NOT pure VRAM residency. Disclose this, don't hide it.")


def run_llamacpp_27b_arm() -> tuple[list[dict], dict]:
    print("\n" + "=" * 80)
    print("▶ [27B] LLAMA.CPP Q4_K_M (Port 8001, real GGUF, HTTP/SSE)")
    print("=" * 80, flush=True)
    ensure_gpu_exclusive()

    bin_dir = REPO_ROOT / "apps/runtime-llama/llama.cpp/build/bin"
    llama_server = bin_dir / "llama-server"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    with MemoryMonitor("llama.cpp-27b") as mon:
        proc = subprocess.Popen(
            [str(llama_server), "--model", GGUF_27B, "--host", "127.0.0.1", "--port", "8001",
             "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "8192",
             "-b", "512", "-ub", "512", "-t", "8", "-np", "1", "--no-webui"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            base.wait_for_endpoint("http://127.0.0.1:8001/v1/models")
            res = base.eval_http_stream("http://127.0.0.1:8001/v1/chat/completions", "qwen3.8:27b-llamacpp", base.BENCHMARK_PROMPTS, "llama.cpp 27B")
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except Exception:  # noqa: BLE001
                proc.kill()
            subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
            time.sleep(2.0)
    mon.report()
    return res, mon.summary()


def run_ollama_27b_arm() -> tuple[list[dict], dict]:
    print("\n" + "=" * 80)
    print("▶ [27B] OLLAMA Q4_K_M (Port 11434, real GGUF, HTTP/SSE)")
    print("=" * 80, flush=True)
    ensure_gpu_exclusive()

    with MemoryMonitor("ollama-27b") as mon:
        proc = subprocess.Popen(
            ["bash", str(REPO_ROOT / "apps/runtime-ollama/run_server.sh")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            base.wait_for_endpoint("http://127.0.0.1:11434/api/version")
            with tempfile.NamedTemporaryFile(mode="w", suffix=".Modelfile", delete=False) as f:
                f.write(f"FROM {GGUF_27B}\n")
                modelfile_path = f.name
            try:
                subprocess.run(["ollama", "create", OLLAMA_MODEL_NAME, "-f", modelfile_path], check=True, timeout=300)
            finally:
                os.unlink(modelfile_path)
            res = base.eval_http_stream("http://127.0.0.1:11434/v1/chat/completions", OLLAMA_MODEL_NAME, base.BENCHMARK_PROMPTS, "Ollama 27B")
        finally:
            subprocess.run(["ollama", "stop", OLLAMA_MODEL_NAME], capture_output=True)
            subprocess.run(["bash", str(REPO_ROOT / "apps/runtime-ollama/run_server.sh"), "stop"], capture_output=True)
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except Exception:  # noqa: BLE001
                proc.kill()
            time.sleep(2.0)
    mon.report()
    return res, mon.summary()


def run_runtime_next_27b_arm() -> tuple[list[dict], dict]:
    print("\n" + "=" * 80)
    print("▶ [27B] RUNTIME-NEXT W4A16 (Port 8003, real quantized weights, HTTP/SSE)")
    print("=" * 80, flush=True)
    ensure_gpu_exclusive()

    if not RUNTIME_NEXT_BIN.is_file():
        raise FileNotFoundError(f"runtime-next 27B binary not found at {RUNTIME_NEXT_BIN} -- build with --features qwen35_27b first")

    env = os.environ.copy()
    env["PORT"] = "8003"
    env["RUNTIME_NEXT_MODEL_DIR"] = str(QUANTIZED_DIR)
    with MemoryMonitor("runtime-next-w4a16-27b") as mon:
        proc = subprocess.Popen(
            [str(RUNTIME_NEXT_BIN)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            base.wait_for_endpoint("http://127.0.0.1:8003/health", timeout_s=240.0)
            res = base.eval_http_stream("http://127.0.0.1:8003/v1/chat/completions", "qwen3.8:27b-w4a16", base.BENCHMARK_PROMPTS, "runtime-next W4A16 27B")
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except Exception:  # noqa: BLE001
                proc.kill()
            subprocess.run(["pkill", "-9", "-f", "target/release/runtime-next"], capture_output=True)
            time.sleep(2.0)
    mon.report()
    return res, mon.summary()


def main() -> None:
    print("=" * 80)
    print("🥊 27B QUANTIZED REAL ENGINE COMPARISON: LLAMA.CPP Q4_K_M vs OLLAMA Q4_K_M vs RUNTIME-NEXT W4A16")
    print("   Hardware: AMD Radeon RX 7900 XTX (24GB GDDR6, gfx1100)")
    print("   Model: Qwen3.8-27B, quantized (~4.5-4.85 bits/weight, real per-arm conversion)")
    print("   Every arm real-time memory-monitored (VRAM/GTT/system-RAM) -- see [mem/*] lines")
    print("=" * 80, flush=True)

    base._kill_all_known_servers()

    llamacpp_res, llamacpp_mem = run_llamacpp_27b_arm()
    ollama_res, ollama_mem = run_ollama_27b_arm()
    runtime_next_res, runtime_next_mem = run_runtime_next_27b_arm()

    print("\n" + "=" * 100)
    print("🏆 27B QUANTIZED REAL ENGINE COMPARISON SCORECARD")
    print("=" * 100)
    print(f"{'Task / Challenge':<35} | {'llama.cpp':<14} | {'Ollama':<14} | {'runtime-next':<14} | {'Speedup vs llama.cpp'}")
    print("-" * 100)
    for lc, o, r in zip(llamacpp_res, ollama_res, runtime_next_res, strict=True):
        speedup = r["tok_per_sec"] / max(1e-5, lc["tok_per_sec"])
        print(f"{lc['name']:<35} | {lc['tok_per_sec']:5.1f} tok/s   | {o['tok_per_sec']:5.1f} tok/s   | {r['tok_per_sec']:5.1f} tok/s      | {speedup:.2f}x")

    avg_llamacpp = sum(x["tok_per_sec"] for x in llamacpp_res) / len(llamacpp_res)
    avg_ollama = sum(x["tok_per_sec"] for x in ollama_res) / len(ollama_res)
    avg_runtime_next = sum(x["tok_per_sec"] for x in runtime_next_res) / len(runtime_next_res)
    avg_ttft_llamacpp = sum(x["ttft_ms"] for x in llamacpp_res) / len(llamacpp_res)
    avg_ttft_ollama = sum(x["ttft_ms"] for x in ollama_res) / len(ollama_res)
    avg_ttft_runtime_next = sum(x["ttft_ms"] for x in runtime_next_res) / len(runtime_next_res)

    print("-" * 100)
    print(f"{'AVERAGE STREAMING THROUGHPUT':<35} | {avg_llamacpp:5.1f} tok/s   | {avg_ollama:5.1f} tok/s   | {avg_runtime_next:5.1f} tok/s      | vs llama.cpp: {avg_runtime_next/max(1e-5,avg_llamacpp):.2f}x, vs Ollama: {avg_runtime_next/max(1e-5,avg_ollama):.2f}x")
    print("=" * 100)

    print("\n📊 REAL MEMORY FOOTPRINT PER ARM (peak, real-time monitored):")
    for label, mem in [("llama.cpp Q4_K_M", llamacpp_mem), ("Ollama Q4_K_M", ollama_mem), ("runtime-next W4A16", runtime_next_mem)]:
        print(f"  {label:<22}: VRAM peak={mem['vram_peak_mb']}MB | GTT grew={mem['gtt_grew']}MB | sys RAM peak={mem['sys_used_peak_mb']}MB")

    payload = {
        "hardware": "AMD Radeon RX 7900 XTX (24GB GDDR6, gfx1100)",
        "model": "Qwen3.8-27B, quantized (llama.cpp/Ollama: real Q4_K_M GGUF; runtime-next: real W4A16 via quantize_w4a16.py)",
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
        "memory_footprint": {
            "llamacpp": llamacpp_mem,
            "ollama": ollama_mem,
            "runtime_next": runtime_next_mem,
        },
        "tasks": {"llamacpp": llamacpp_res, "ollama": ollama_res, "runtime_next": runtime_next_res},
    }
    out_file = REPO_ROOT / "results/benchmarks/27b_quantized_engine_comparison_scorecard.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"\n💾 Full results (including per-arm memory traces) saved to: {out_file}")


if __name__ == "__main__":
    main()
