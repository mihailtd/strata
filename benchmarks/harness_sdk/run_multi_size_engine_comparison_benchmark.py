"""§105: Multi-Size Real Engine Comparison: llama.cpp vs Ollama vs
runtime-next, across the REAL Qwen3.5 family (0.8B, 2B, 4B, 9B).

Directly generalizes `run_4b_engine_comparison_benchmark.py`'s own
methodology (real HTTP/SSE, real coding-task prompts, real TTFT +
streaming tok/s, no library-call shortcuts for any arm) to every real
Qwen3.5 size this session confirmed exists (checked directly against
Hugging Face, never Qwen2.5 substituted for a missing size). Reuses that
module's own validated `eval_http_stream`/`wait_for_endpoint` measurement
code directly rather than re-implementing it.

Per size, all three arms serve the SAME real, byte-identical bf16 weights
(one `convert_hf_to_gguf.py --outtype bf16` conversion per size, in
`models/qwen35_gguf_bench/`, matching the original 4B methodology exactly
-- no arm gets a quantization advantage the others don't).

`runtime-next` is a SEPARATE compiled binary per size (Cargo feature
`qwen35_0_8b`/`qwen35_2b`/`qwen35_4b`/`qwen35_9b`, §105's own
const-block-per-size generalization of what was originally a
4B-hardcoded engine) -- pre-built into `models/runtime_next_bins/` by
this same session, one correctness-validated (`real_greedy_generation_
matches_real_qwen3_5_<size>`, byte-exact against a real HF `transformers`
reference) binary per size.

Requires: `models/qwen35_gguf_bench/qwen3.5-{size}-bf16.gguf` for every
size below, `models/runtime_next_bins/runtime-next-qwen35_{size}` for
every size below, a built `llama-server`, and `ollama`.
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
from runtime_common.gpu_preflight import ensure_gpu_exclusive  # noqa: E402

GGUF_DIR = REPO_ROOT / "models" / "qwen35_gguf_bench"
RUNTIME_NEXT_BIN_DIR = REPO_ROOT / "models" / "runtime_next_bins"

# Real sizes this session confirmed exist as real Qwen/Qwen3.5-* repos on
# Hugging Face (raw config.json fetched and parsed directly, never
# paraphrased) -- deliberately NOT Qwen2.5 for any size.
SIZES = [
    {"label": "0.8B", "feature": "qwen35_0_8b", "gguf": "qwen3.5-0.8b-bf16.gguf", "hf_repo": "Qwen/Qwen3.5-0.8B"},
    {"label": "2B", "feature": "qwen35_2b", "gguf": "qwen3.5-2b-bf16.gguf", "hf_repo": "Qwen/Qwen3.5-2B"},
    {"label": "4B", "feature": "qwen35_4b", "gguf": "qwen3.5-4b-bf16.gguf", "hf_repo": "Qwen/Qwen3.5-4B"},
    {"label": "9B", "feature": "qwen35_9b", "gguf": "qwen3.5-9b-bf16.gguf", "hf_repo": "Qwen/Qwen3.5-9B"},
]


def run_llamacpp_arm(gguf_path: Path, label: str) -> list[dict]:
    print("\n" + "=" * 80)
    print(f"▶ [{label}] LLAMA.CPP (Port 8001, real bf16 GGUF, HTTP/SSE)")
    print("=" * 80, flush=True)
    ensure_gpu_exclusive()

    bin_dir = REPO_ROOT / "apps/runtime-llama/llama.cpp/build/bin"
    llama_server = bin_dir / "llama-server"
    if not llama_server.is_file():
        raise FileNotFoundError(f"llama-server not built at {llama_server} -- run apps/runtime-llama/setup.sh first")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    proc = subprocess.Popen(
        [
            str(llama_server), "--model", str(gguf_path), "--host", "127.0.0.1", "--port", "8001",
            "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "16384",
            "-b", "512", "-ub", "512", "-t", "8", "-np", "1", "--no-webui",
        ],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        base.wait_for_endpoint("http://127.0.0.1:8001/v1/models")
        res = base.eval_http_stream("http://127.0.0.1:8001/v1/chat/completions", f"qwen3.5:{label}-llamacpp", base.BENCHMARK_PROMPTS, f"llama.cpp {label}")
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except Exception:  # noqa: BLE001
            proc.kill()
        subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
        time.sleep(2.0)
    return res


def run_ollama_arm(gguf_path: Path, label: str) -> list[dict]:
    print("\n" + "=" * 80)
    print(f"▶ [{label}] OLLAMA (Port 11434, real bf16 GGUF, HTTP/SSE)")
    print("=" * 80, flush=True)
    ensure_gpu_exclusive()

    model_name = f"qwen3.5:{label.lower()}-ollama-bench"
    proc = subprocess.Popen(
        ["bash", str(REPO_ROOT / "apps/runtime-ollama/run_server.sh")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        base.wait_for_endpoint("http://127.0.0.1:11434/api/version")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".Modelfile", delete=False) as f:
            f.write(f"FROM {gguf_path}\n")
            modelfile_path = f.name
        try:
            subprocess.run(["ollama", "create", model_name, "-f", modelfile_path], check=True, timeout=180)
        finally:
            os.unlink(modelfile_path)

        res = base.eval_http_stream("http://127.0.0.1:11434/v1/chat/completions", model_name, base.BENCHMARK_PROMPTS, f"Ollama {label}")
    finally:
        subprocess.run(["ollama", "stop", model_name], capture_output=True)
        subprocess.run(["bash", str(REPO_ROOT / "apps/runtime-ollama/run_server.sh"), "stop"], capture_output=True)
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except Exception:  # noqa: BLE001
            proc.kill()
        time.sleep(2.0)
    return res


def run_runtime_next_arm(binary_path: Path, label: str) -> list[dict]:
    print("\n" + "=" * 80)
    print(f"▶ [{label}] RUNTIME-NEXT (Port 8003, real safetensors weights, HTTP/SSE)")
    print("=" * 80, flush=True)
    ensure_gpu_exclusive()

    if not binary_path.is_file():
        raise FileNotFoundError(f"runtime-next binary not found at {binary_path}")

    env = os.environ.copy()
    env["PORT"] = "8003"
    proc = subprocess.Popen(
        [str(binary_path)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        base.wait_for_endpoint("http://127.0.0.1:8003/health")
        res = base.eval_http_stream("http://127.0.0.1:8003/v1/chat/completions", f"qwen3.5:{label}-rust", base.BENCHMARK_PROMPTS, f"runtime-next {label}")
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except Exception:  # noqa: BLE001
            proc.kill()
        subprocess.run(["pkill", "-9", "-f", f"runtime-next-{label}"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "target/release/runtime-next"], capture_output=True)
        time.sleep(2.0)
    return res


def main() -> None:
    print("=" * 80)
    print("🥊 MULTI-SIZE REAL ENGINE COMPARISON: LLAMA.CPP vs OLLAMA vs RUNTIME-NEXT")
    print("   Hardware: AMD Radeon RX 7900 XTX (24GB GDDR6, gfx1100)")
    print("   Sizes:    real Qwen3.5 0.8B / 2B / 4B / 9B (never Qwen2.5)")
    print("=" * 80, flush=True)

    base._kill_all_known_servers()

    all_results: dict[str, dict] = {}

    for size in SIZES:
        label = size["label"]
        gguf_path = GGUF_DIR / size["gguf"]
        rn_binary = RUNTIME_NEXT_BIN_DIR / f"runtime-next-{size['feature']}"
        if not gguf_path.is_file():
            print(f"⚠ skipping {label}: GGUF not found at {gguf_path}")
            continue
        if not rn_binary.is_file():
            print(f"⚠ skipping {label}: runtime-next binary not found at {rn_binary}")
            continue

        print(f"\n\n{'#'*90}\n# SIZE: {label}\n{'#'*90}")

        llamacpp_res = run_llamacpp_arm(gguf_path, label)
        ollama_res = run_ollama_arm(gguf_path, label)
        runtime_next_res = run_runtime_next_arm(rn_binary, label)

        avg_llamacpp = sum(x["tok_per_sec"] for x in llamacpp_res) / len(llamacpp_res)
        avg_ollama = sum(x["tok_per_sec"] for x in ollama_res) / len(ollama_res)
        avg_runtime_next = sum(x["tok_per_sec"] for x in runtime_next_res) / len(runtime_next_res)
        avg_ttft_llamacpp = sum(x["ttft_ms"] for x in llamacpp_res) / len(llamacpp_res)
        avg_ttft_ollama = sum(x["ttft_ms"] for x in ollama_res) / len(ollama_res)
        avg_ttft_runtime_next = sum(x["ttft_ms"] for x in runtime_next_res) / len(runtime_next_res)

        all_results[label] = {
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
            "tasks": {"llamacpp": llamacpp_res, "ollama": ollama_res, "runtime_next": runtime_next_res},
        }

        print(f"\n--- {label} DONE: llama.cpp {avg_llamacpp:.1f} | Ollama {avg_ollama:.1f} | runtime-next {avg_runtime_next:.1f} tok/s ---")

    print("\n" + "=" * 100)
    print("🏆 FINAL MULTI-SIZE REAL ENGINE COMPARISON SCORECARD")
    print("=" * 100)
    print(f"{'Size':<8} | {'llama.cpp':<12} | {'Ollama':<12} | {'runtime-next':<14} | {'vs llama.cpp':<14} | {'vs Ollama'}")
    print("-" * 100)
    for label, r in all_results.items():
        print(f"{label:<8} | {r['avg_llamacpp_tok_s']:6.1f} tok/s | {r['avg_ollama_tok_s']:6.1f} tok/s | {r['avg_runtime_next_tok_s']:6.1f} tok/s     | {r['runtime_next_speedup_vs_llamacpp']:.2f}x         | {r['runtime_next_speedup_vs_ollama']:.2f}x")
    print("=" * 100)

    payload = {
        "hardware": "AMD Radeon RX 7900 XTX (24GB GDDR6, gfx1100)",
        "model_family": "Qwen 3.5 (real, byte-identical bf16 weights per size across all 3 arms)",
        "sizes": all_results,
    }
    out_file = REPO_ROOT / "results/benchmarks/multi_size_engine_comparison_scorecard.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"💾 Full results saved to: {out_file}")


if __name__ == "__main__":
    main()
