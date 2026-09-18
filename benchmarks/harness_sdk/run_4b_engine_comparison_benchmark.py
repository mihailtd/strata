"""4B-Tier Real Engine Comparison: llama.cpp vs Ollama vs runtime-next (Rust/HIP).

Real end-to-end HTTP/SSE benchmark -- same methodology
`run_direct_harness_plugin_benchmark.py` already established for the 27B
tier (real coding-task prompts, real streaming `/v1/chat/completions`,
real TTFT + streaming tok/s) applied to this session's from-scratch
Rust/HIP Qwen3.5-4B engine (`apps/runtime-next`) against its two upstream
baselines. All three arms serve the SAME real, byte-identical Qwen3.5-4B
bf16 weights (the real bf16 GGUF converted for the original A/B in
`docs/DECISIONS.md` §90; `runtime-next` reads the same real weights
directly from the HF safetensors snapshot -- same model, two file formats,
confirmed byte-identical at conversion time).

This validates whether `runtime-next`'s raw-throughput gains (§91-§94,
31.7 -> ~82 tok/s measured via `llama-bench`/Ollama's `/api/generate`
directly) survive once the engine is actually served over real HTTP/SSE
and driven through the request shape a real client (DSH included) sends --
not just the direct-decode-loop microbenchmark.

Requires: the real bf16 GGUF on disk (see `GGUF_PATH` below -- override via
`QWEN35_4B_BF16_GGUF` if it has moved), `ollama` and a built `llama-server`
(`apps/runtime-llama/setup.sh`) on this machine, and `apps/runtime-next`
built in release mode.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "runtime-common" / "src"))
from runtime_common.gpu_preflight import ensure_gpu_exclusive  # noqa: E402

GGUF_PATH = os.environ.get(
    "QWEN35_4B_BF16_GGUF",
    "/tmp/claude-1000/-home-mihai-Projects-gnn-experiment/62408538-8f70-4fd0-a0ab-8c9b65ca71dd/scratchpad/gguf/qwen3.5-4b-bf16.gguf",
)
OLLAMA_MODEL_NAME = "qwen3.5:4b-ollama"

# Real coding-task prompts -- adapted from run_direct_harness_plugin_benchmark.py's
# own BENCHMARK_PROMPTS (already generic SWE tasks, not tied to the 27B
# LoRA specialist routing this repo's other benchmarks exercise).
BENCHMARK_PROMPTS = [
    {
        "id": "fastapi_pgvector_crud",
        "name": "FastAPI Asyncpg & pgvector CRUD Service",
        "system": "You are a senior PostgreSQL and Python backend architect.",
        "user": "Write a complete production FastAPI application with an asyncpg connection pool, pgvector HNSW search endpoint, and Pydantic response models.",
    },
    {
        "id": "postgres_hnsw_migration",
        "name": "PostgreSQL 17 HNSW Migration & Tuning",
        "system": "You are a database engineer specializing in vector search.",
        "user": "Write a complete PostgreSQL 17 SQL migration that enables the vector extension, creates an items table with a 1536-dim vector column, creates an HNSW index with vector_cosine_ops, and demonstrates a similarity query.",
    },
    {
        "id": "duckdb_parquet_analytics",
        "name": "DuckDB Parquet Window Analytics Pipeline",
        "system": "You are a data engineer specializing in OLAP and DuckDB.",
        "user": "Write a complex DuckDB SQL query that reads from a partitioned Parquet dataset 's3://analytics/events.parquet', computes window aggregates, and uses the QUALIFY clause with ROW_NUMBER() to deduplicate records.",
    },
]

MAX_TOKENS = 350


def wait_for_endpoint(url: str, timeout_s: float = 180.0) -> None:
    """Polls `url` until it responds or `timeout_s` elapses -- real weight
    loading for an 8.65GB bf16 model takes real time on all three engines,
    not instant."""
    t0 = time.perf_counter()
    last_err: Exception | None = None
    while time.perf_counter() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                if r.status == 200:
                    return
        except Exception as e:  # noqa: BLE001
            last_err = e
        time.sleep(1.0)
    raise TimeoutError(f"{url} did not become ready within {timeout_s}s (last error: {last_err})")


def eval_http_stream(endpoint_url: str, model_id: str, prompts: list[dict], arm_label: str) -> list[dict]:
    # Warm up (real cold-start request, matches the original 3-way script).
    warmup_req = urllib.request.Request(
        endpoint_url,
        data=json.dumps({
            "model": model_id,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "stream": False,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(warmup_req, timeout=120) as r:
            r.read()
    except Exception as e:  # noqa: BLE001
        print(f"[{arm_label}] Warmup note: {e}")

    results = []
    for i, p in enumerate(prompts, 1):
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user"]},
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "stream": True,
        }
        req = urllib.request.Request(
            endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        t0 = time.perf_counter()
        tokens = 0
        ttft = None

        with urllib.request.urlopen(req, timeout=90) as resp:
            for line in resp:
                s = line.decode("utf-8").strip()
                if not s.startswith("data: ") or s == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(s[6:])
                except Exception:  # noqa: BLE001
                    continue
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
                if content:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000.0
                    tokens += 1

        t1 = time.perf_counter()
        ttft_val = ttft if ttft is not None else 0.0
        gen_time = (t1 - t0) - (ttft_val / 1000.0)
        tok_s = tokens / max(1e-5, gen_time)

        print(f"  [{i}/{len(prompts)}] {p['name']:<40} | Toks: {tokens:3d} | TTFT: {ttft_val:5.1f}ms | Speed: {tok_s:5.2f} tok/s", flush=True)
        results.append({
            "task_id": p["id"],
            "name": p["name"],
            "tokens": tokens,
            "ttft_ms": round(ttft_val, 1),
            "tok_per_sec": round(tok_s, 2),
        })
    return results


def _kill_all_known_servers() -> None:
    subprocess.run(["ollama", "stop", OLLAMA_MODEL_NAME], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "ollama serve"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "target/release/runtime-next"], capture_output=True)
    time.sleep(2.0)


def run_llamacpp_arm() -> list[dict]:
    print("\n" + "=" * 80)
    print("▶ [ARM 1/3] LLAMA.CPP (Port 8001, real bf16 GGUF, HTTP/SSE)")
    print("=" * 80, flush=True)
    ensure_gpu_exclusive()

    if not Path(GGUF_PATH).is_file():
        raise FileNotFoundError(f"real bf16 GGUF not found at {GGUF_PATH} -- set QWEN35_4B_BF16_GGUF")

    # NOT via run_server.sh directly: that script auto-attaches this repo's
    # real 27B LoRA adapters whenever they exist on disk (`results/adapters/
    # *_27b.gguf`) -- real, found the hard way (llama-server refused to
    # load: "tensor 'blk.0.ffn_down.weight' has incorrect shape", since
    # those adapters were trained for the 27B model's tensor shapes, not
    # this 4B model's). Invoking the real `llama-server` binary directly
    # with the same core serving flags `run_server.sh` uses, minus the
    # 27B-specific LoRA auto-detection -- matches how §90's original A/B
    # used `llama-bench` directly against this exact model with no LoRA.
    bin_dir = REPO_ROOT / "apps/runtime-llama/llama.cpp/build/bin"
    llama_server = bin_dir / "llama-server"
    if not llama_server.is_file():
        raise FileNotFoundError(f"llama-server not built at {llama_server} -- run apps/runtime-llama/setup.sh first")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    proc = subprocess.Popen(
        [
            str(llama_server),
            "--model", GGUF_PATH,
            "--host", "127.0.0.1",
            "--port", "8001",
            "-ngl", "99",
            "-fa", "on",
            "-ctk", "q8_0",
            "-ctv", "q8_0",
            "-c", "16384",
            "-b", "512",
            "-ub", "512",
            "-t", "8",
            "-np", "1",
            "--no-webui",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        wait_for_endpoint("http://127.0.0.1:8001/v1/models")
        res = eval_http_stream("http://127.0.0.1:8001/v1/chat/completions", "qwen3.5:4b-llamacpp", BENCHMARK_PROMPTS, "llama.cpp")
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except Exception:  # noqa: BLE001
            proc.kill()
        subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
        time.sleep(2.0)
    return res


def run_ollama_arm() -> list[dict]:
    print("\n" + "=" * 80)
    print("▶ [ARM 2/3] OLLAMA (Port 11434, real bf16 GGUF, HTTP/SSE)")
    print("=" * 80, flush=True)
    ensure_gpu_exclusive()

    if not Path(GGUF_PATH).is_file():
        raise FileNotFoundError(f"real bf16 GGUF not found at {GGUF_PATH} -- set QWEN35_4B_BF16_GGUF")

    proc = subprocess.Popen(
        ["bash", str(REPO_ROOT / "apps/runtime-ollama/run_server.sh")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        wait_for_endpoint("http://127.0.0.1:11434/api/version")

        # Real model creation from the SAME real GGUF -- the original A/B's
        # test model was removed after benchmarking, so this recreates it.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".Modelfile", delete=False) as f:
            f.write(f"FROM {GGUF_PATH}\n")
            modelfile_path = f.name
        try:
            subprocess.run(
                ["ollama", "create", OLLAMA_MODEL_NAME, "-f", modelfile_path],
                check=True,
                timeout=180,
            )
        finally:
            os.unlink(modelfile_path)

        res = eval_http_stream("http://127.0.0.1:11434/v1/chat/completions", OLLAMA_MODEL_NAME, BENCHMARK_PROMPTS, "Ollama")
    finally:
        subprocess.run(["ollama", "stop", OLLAMA_MODEL_NAME], capture_output=True)
        subprocess.run(["bash", str(REPO_ROOT / "apps/runtime-ollama/run_server.sh"), "stop"], capture_output=True)
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except Exception:  # noqa: BLE001
            proc.kill()
        time.sleep(2.0)
    return res


def run_runtime_next_arm() -> list[dict]:
    print("\n" + "=" * 80)
    print("▶ [ARM 3/3] RUNTIME-NEXT (Port 8003, real safetensors weights, HTTP/SSE)")
    print("   From-scratch Rust/HIP engine -- this session's own port, §91-§95")
    print("=" * 80, flush=True)
    ensure_gpu_exclusive()

    runtime_next_dir = REPO_ROOT / "apps/runtime-next"
    subprocess.run(["cargo", "build", "--release"], cwd=runtime_next_dir, check=True)

    env = os.environ.copy()
    env["PORT"] = "8003"
    proc = subprocess.Popen(
        [str(runtime_next_dir / "target/release/runtime-next")],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        wait_for_endpoint("http://127.0.0.1:8003/health")
        res = eval_http_stream("http://127.0.0.1:8003/v1/chat/completions", "qwen3.5:4b-rust", BENCHMARK_PROMPTS, "runtime-next")
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except Exception:  # noqa: BLE001
            proc.kill()
        subprocess.run(["pkill", "-9", "-f", "target/release/runtime-next"], capture_output=True)
        time.sleep(2.0)
    return res


def main() -> None:
    print("=" * 80)
    print("🥊 4B-TIER REAL ENGINE COMPARISON: LLAMA.CPP vs OLLAMA vs RUNTIME-NEXT")
    print("   Hardware: AMD Radeon RX 7900 XTX (24GB GDDR6, gfx1100)")
    print("   Model:    Qwen 3.5 4B (real bf16, byte-identical weights across all 3 arms)")
    print("=" * 80, flush=True)

    _kill_all_known_servers()

    llamacpp_res = run_llamacpp_arm()
    ollama_res = run_ollama_arm()
    runtime_next_res = run_runtime_next_arm()

    print("\n" + "=" * 100)
    print("🏆 FINAL 4B-TIER REAL ENGINE COMPARISON SCORECARD")
    print("=" * 100)
    print(f"{'Task / Challenge':<35} | {'llama.cpp':<14} | {'Ollama':<14} | {'runtime-next':<14} | {'Speedup vs llama.cpp'}")
    print("-" * 100)

    for lc, o, r in zip(llamacpp_res, ollama_res, runtime_next_res, strict=True):
        speedup = r["tok_per_sec"] / max(1e-5, lc["tok_per_sec"])
        diff_str = f"{speedup:.2f}x ({'+' if r['tok_per_sec'] >= lc['tok_per_sec'] else ''}{r['tok_per_sec'] - lc['tok_per_sec']:.1f} tok/s)"
        print(f"{lc['name']:<35} | {lc['tok_per_sec']:5.1f} tok/s   | {o['tok_per_sec']:5.1f} tok/s   | {r['tok_per_sec']:5.1f} tok/s      | {diff_str}")

    avg_llamacpp = sum(x["tok_per_sec"] for x in llamacpp_res) / len(llamacpp_res)
    avg_ollama = sum(x["tok_per_sec"] for x in ollama_res) / len(ollama_res)
    avg_runtime_next = sum(x["tok_per_sec"] for x in runtime_next_res) / len(runtime_next_res)
    total_speedup_vs_llamacpp = avg_runtime_next / max(1e-5, avg_llamacpp)
    total_speedup_vs_ollama = avg_runtime_next / max(1e-5, avg_ollama)

    print("-" * 100)
    print(f"{'AVERAGE STREAMING THROUGHPUT':<35} | {avg_llamacpp:5.1f} tok/s   | {avg_ollama:5.1f} tok/s   | {avg_runtime_next:5.1f} tok/s      | vs llama.cpp: {total_speedup_vs_llamacpp:.2f}x, vs Ollama: {total_speedup_vs_ollama:.2f}x")
    print("=" * 100)

    avg_ttft_llamacpp = sum(x["ttft_ms"] for x in llamacpp_res) / len(llamacpp_res)
    avg_ttft_ollama = sum(x["ttft_ms"] for x in ollama_res) / len(ollama_res)
    avg_ttft_runtime_next = sum(x["ttft_ms"] for x in runtime_next_res) / len(runtime_next_res)
    print(f"{'AVERAGE TTFT':<35} | {avg_ttft_llamacpp:5.1f} ms      | {avg_ttft_ollama:5.1f} ms      | {avg_ttft_runtime_next:5.1f} ms")

    final_payload = {
        "hardware": "AMD Radeon RX 7900 XTX (24GB GDDR6, gfx1100)",
        "model": "Qwen 3.5 4B bf16 (real, byte-identical weights across all 3 arms)",
        "avg_llamacpp_tok_s": round(avg_llamacpp, 2),
        "avg_ollama_tok_s": round(avg_ollama, 2),
        "avg_runtime_next_tok_s": round(avg_runtime_next, 2),
        "runtime_next_speedup_vs_llamacpp": f"{total_speedup_vs_llamacpp:.2f}x",
        "runtime_next_speedup_vs_ollama": f"{total_speedup_vs_ollama:.2f}x",
        "avg_ttft_ms": {
            "llamacpp": round(avg_ttft_llamacpp, 1),
            "ollama": round(avg_ttft_ollama, 1),
            "runtime_next": round(avg_ttft_runtime_next, 1),
        },
        "tasks": {
            "llamacpp": llamacpp_res,
            "ollama": ollama_res,
            "runtime_next": runtime_next_res,
        },
    }

    out_file = Path("results/benchmarks/4b_engine_comparison_scorecard.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(final_payload, indent=2))
    print(f"💾 Full results saved to: {out_file}")


if __name__ == "__main__":
    main()
