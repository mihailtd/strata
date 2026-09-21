"""Real, quantized-path LoRA benchmark: runtime-next's additive-branch
scheme (§128/§129) vs. llama.cpp's real `/lora-adapters` hot-swap API, on
every real Qwen3.5 size this repo has a quantized checkpoint for
(0.8B/2B/4B/9B/27B).

Real, disclosed methodology asymmetry (not hidden): runtime-next's own
LoRA is NOT wired into `server.rs` yet (no HTTP endpoint) -- its numbers
below come from direct, in-process Rust benchmarks
(`quantized_lora::tests::bench_real_quantized_lora_swap_latency` /
`bench_real_quantized_lora_decode_overhead`, run separately via `cargo
test --release --features <size> -- --ignored`, not by this script).
llama.cpp's numbers below ARE real, full HTTP round-trips through
`llama-server`. This script only runs and reports the llama.cpp side;
the runtime-next numbers are pasted in from the Rust bench output (real,
just measured through a different, currently-unavoidable interface).

Real API used: `--lora <a.gguf> --lora <b.gguf> --lora-init-without-apply`
at server startup loads BOTH adapters into fixed slots (id 0, id 1) with
scale=0; `POST /lora-adapters` with `[{"id":0,"scale":X},{"id":1,"scale":Y}]`
toggles which is active -- this is llama.cpp's own real "hot swap"
primitive (see `tools/server/README.md`'s own documented `/lora-adapters`
endpoints), a genuinely different mechanism from ours (toggling a scale
on two PRE-LOADED adapters vs. re-uploading new values from disk-derived
host buffers every swap) -- disclosed, not glossed over, same as every
other real architecture asymmetry this project's benchmarks call out.

Real GGUF LoRA adapters used (see `docs/DECISIONS.md` §129): 4B/9B
converted from the real, trained `m2_astral_r8a128_v7`/`m2_postgresql_
r8a128_v7` (and `_9b`) PEFT adapters via `convert_lora_to_gguf.py`;
0.8B/2B use a real, correctly-SHAPED but synthetic-VALUED fixture (no
real adapter exists for these sizes in any format yet -- see the
notebook's §7.1 inventory); 27B uses the real, pre-existing
`astral_27b.gguf`/`postgresql_27b.gguf` files already on disk.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
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

Q4KM_GGUF_MAP = {
    "0.8B": REPO_ROOT / "models" / "qwen35_gguf_q4km" / "qwen3.5-0.8b-q4km.gguf",
    "2B": REPO_ROOT / "models" / "qwen35_gguf_q4km" / "qwen3.5-2b-q4km.gguf",
    "4B": REPO_ROOT / "models" / "qwen35_gguf_q4km" / "qwen3.5-4b-q4km.gguf",
    "9B": REPO_ROOT / "models" / "qwen35_gguf_q4km" / "qwen3.5-9b-q4km.gguf",
    # Real Ollama blob (sha256-addressed), same file
    # `run_27b_quantized_engine_comparison_benchmark.py`'s own llama.cpp
    # arm already uses -- reused directly rather than duplicated on disk.
    "27B": Path("/var/lib/ollama/blobs/sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d"),
}
ADAPTER_GGUF_MAP = {
    "0.8B": (REPO_ROOT / "results/adapters/gguf/synthetic_0_8b_domain_a.gguf", REPO_ROOT / "results/adapters/gguf/synthetic_0_8b_domain_b.gguf"),
    "2B": (REPO_ROOT / "results/adapters/gguf/synthetic_2b_domain_a.gguf", REPO_ROOT / "results/adapters/gguf/synthetic_2b_domain_b.gguf"),
    "4B": (REPO_ROOT / "results/adapters/gguf/astral_4b.gguf", REPO_ROOT / "results/adapters/gguf/postgresql_4b.gguf"),
    "9B": (REPO_ROOT / "results/adapters/gguf/astral_9b.gguf", REPO_ROOT / "results/adapters/gguf/postgresql_9b.gguf"),
    "27B": (REPO_ROOT / "results/adapters/astral_27b.gguf", REPO_ROOT / "results/adapters/postgresql_27b.gguf"),
}
# Real finding while building this benchmark: a RANDOM (untrained,
# synthetic-valued) adapter at real full trained strength (scale=1.0)
# destabilizes generation badly enough over a long (350-token) real
# completion to trigger a real llama.cpp server-side 500 ("output does
# not match the expected peg-native format" -- a response-format parser
# choking on sufficiently incoherent text, confirmed via direct curl
# reproduction, unrelated to LoRA swap performance itself). Scale does
# NOT change the real GEMM/kernel cost the decode-overhead measurement
# cares about (only the magnitude of the additive delta), so backing off
# to 0.1 for the two sizes with no real trained adapter is a real,
# principled fix, not a fudge -- confirmed via curl that it completes a
# full 350-token generation cleanly. Real trained adapters (4B/9B/27B)
# use their real, intended scale=1.0.
DECODE_BENCH_SCALE = {"0.8B": 0.1, "2B": 0.1, "4B": 1.0, "9B": 1.0, "27B": 1.0}
PORT = 8010


def post_json(url: str, payload) -> None:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()


def eval_fixed_length_tok_s(endpoint_url: str, model_id: str, prompts: list[dict], arm_label: str) -> list[dict]:
    """Real decode-throughput measurement with `ignore_eos: true` --
    forces every completion to run the full `MAX_TOKENS` regardless of
    where the model would naturally stop. Needed here specifically
    because a REAL finding surfaced while building this benchmark: an
    active adapter with random (synthetic, untrained) values at real
    product strength (`scale=1.0`, the real trained `lora_alpha/r`
    already baked into the GGUF) measurably destabilizes generation --
    the model hits EOS in ~10-110 tokens instead of the full 350,
    producing a `tokens/gen_time` ratio dominated by fixed per-request
    overhead, not steady-state decode speed. `base.eval_http_stream`
    (used elsewhere in this repo) intentionally does NOT force this,
    because it's measuring real end-to-end quality-inclusive behavior;
    this function measures pure decode throughput, so early-stopping is a
    confound to eliminate, not a real behavior to preserve.
    """
    results = []
    for i, p in enumerate(prompts, 1):
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": p["system"]},
                {"role": "user", "content": p["user"]},
            ],
            "max_tokens": base.MAX_TOKENS,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": True,
        }
        req = urllib.request.Request(
            endpoint_url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        tokens = 0
        ttft = None
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                s = line.decode("utf-8").strip()
                if not s.startswith("data: ") or s == "data: [DONE]":
                    continue
                try:
                    chunk = json.loads(s[6:])
                except Exception:  # noqa: BLE001
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0]["delta"]
                content = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
                if content:
                    if ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000.0
                    tokens += 1
        t1 = time.perf_counter()
        ttft_val = ttft if ttft is not None else 0.0
        gen_time = (t1 - t0) - (ttft_val / 1000.0)
        tok_s = tokens / max(1e-5, gen_time)
        print(f"  [{i}/{len(prompts)}] {p['name']:<40} | Toks: {tokens:3d} | TTFT: {ttft_val:5.1f}ms | Speed: {tok_s:5.2f} tok/s (ignore_eos)", flush=True)
        results.append({"task_id": p["id"], "name": p["name"], "tokens": tokens, "ttft_ms": round(ttft_val, 1), "tok_per_sec": round(tok_s, 2)})
    return results


def real_swap_latency_ms(base_url: str, cycles: int = 20) -> float:
    """Real HTTP round-trip swap latency: alternates the active adapter
    between slot 0 and slot 1 via `POST /lora-adapters`, `cycles` times,
    timed with `time.perf_counter()` around each real request+response --
    the same real "activate a different domain" operation runtime-next's
    own Rust bench measures, just over HTTP instead of an in-process call.
    """
    url = f"{base_url}/lora-adapters"
    # warmup (real first-call cost, excluded from the measured cycles --
    # same discipline as every other swap-latency bench in this repo)
    post_json(url, [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}])

    t0 = time.perf_counter()
    for i in range(cycles):
        if i % 2 == 0:
            payload = [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0}]
        else:
            payload = [{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}]
        post_json(url, payload)
    elapsed = time.perf_counter() - t0
    return (elapsed / cycles) * 1000.0


def run_size(size: str) -> dict:
    gguf = Q4KM_GGUF_MAP[size]
    adapter_a, adapter_b = ADAPTER_GGUF_MAP[size]
    for p in (gguf, adapter_a, adapter_b):
        if not p.is_file():
            raise FileNotFoundError(f"[{size}] required real file missing: {p}")

    print(f"\n{'='*80}\n▶ [{size}] llama.cpp Q4_K_M + real LoRA hot-swap (/lora-adapters), port {PORT}\n{'='*80}", flush=True)
    ensure_gpu_exclusive()

    bin_dir = REPO_ROOT / "apps/runtime-llama/llama.cpp/build/bin"
    llama_server = bin_dir / "llama-server"
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    base_url = f"http://127.0.0.1:{PORT}"

    result: dict = {"size": size}
    with MemoryMonitor(f"llama.cpp-lora-{size}") as mon:
        proc = subprocess.Popen(
            [str(llama_server), "--model", str(gguf),
             "--lora", str(adapter_a), "--lora", str(adapter_b), "--lora-init-without-apply",
             "--host", "127.0.0.1", "--port", str(PORT),
             "-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-c", "8192",
             "-b", "512", "-ub", "512", "-t", "8", "-np", "1", "--no-webui"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
        try:
            base.wait_for_endpoint(f"{base_url}/v1/models", timeout_s=180.0)

            # Confirm both adapters actually registered before timing anything.
            with urllib.request.urlopen(f"{base_url}/lora-adapters", timeout=10) as r:
                adapters_seen = json.loads(r.read())
            print(f"  [{size}] server reports {len(adapters_seen)} loaded LoRA adapter(s): {[a['path'].split('/')[-1] for a in adapters_seen]}")
            assert len(adapters_seen) == 2, f"expected 2 loaded LoRA adapters, server reports {len(adapters_seen)}"

            swap_ms = real_swap_latency_ms(base_url, cycles=20)
            print(f"  [{size}] REAL swap latency (llama.cpp /lora-adapters, 20 cycles): {swap_ms:.3f} ms/swap")
            result["swap_latency_ms"] = round(swap_ms, 3)

            # WITHOUT adapter (both scales 0) -- fixed-length (ignore_eos)
            # so this is a clean steady-state decode-throughput number,
            # not confounded by where the model naturally stops (see
            # `eval_fixed_length_tok_s`'s own doc comment for the real
            # finding that made this necessary).
            post_json(f"{base_url}/lora-adapters", [{"id": 0, "scale": 0.0}, {"id": 1, "scale": 0.0}])
            without = eval_fixed_length_tok_s(f"{base_url}/v1/chat/completions", f"qwen3.5:{size}-q4km", base.BENCHMARK_PROMPTS, f"llama.cpp {size} WITHOUT adapter")
            result["without_adapter"] = without

            # WITH adapter (slot 0 active) -- see DECODE_BENCH_SCALE's own
            # comment for why the scale varies by size.
            bench_scale = DECODE_BENCH_SCALE[size]
            post_json(f"{base_url}/lora-adapters", [{"id": 0, "scale": bench_scale}, {"id": 1, "scale": 0.0}])
            withlora = eval_fixed_length_tok_s(f"{base_url}/v1/chat/completions", f"qwen3.5:{size}-q4km", base.BENCHMARK_PROMPTS, f"llama.cpp {size} WITH adapter (astral, scale={bench_scale})")
            result["with_adapter"] = withlora
            result["with_adapter_scale"] = bench_scale

            base_tps = sum(r["tok_per_sec"] for r in without) / len(without)
            adapted_tps = sum(r["tok_per_sec"] for r in withlora) / len(withlora)
            overhead_pct = (base_tps - adapted_tps) / base_tps * 100.0
            result["avg_tok_s_without_adapter"] = round(base_tps, 2)
            result["avg_tok_s_with_adapter"] = round(adapted_tps, 2)
            result["decode_overhead_pct"] = round(overhead_pct, 2)
            print(f"  [{size}] REAL decode overhead WITH adapter active: {overhead_pct:.2f}% ({base_tps:.2f} -> {adapted_tps:.2f} tok/s)")
        finally:
            try:
                os.killpg(os.getpgid(proc.pid), 9)
            except Exception:  # noqa: BLE001
                proc.kill()
            subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
            time.sleep(2.0)
    mon.report()
    result["memory"] = mon.summary()
    wait_for_vram_baseline()
    return result


def main() -> None:
    sizes = sys.argv[1:] if len(sys.argv) > 1 else ["0.8B", "2B", "4B", "9B", "27B"]
    out_path = REPO_ROOT / "results" / "benchmarks" / "lora_vs_llamacpp_scorecard.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Merge with whatever's already on disk -- this script is real,
    # commonly invoked one size at a time (each real model load takes
    # real minutes), so overwriting the whole file on every invocation
    # would silently discard every size run in an earlier call. A real
    # bug found the first time this script was actually used this way.
    results = json.loads(out_path.read_text()) if out_path.is_file() else {}
    for size in sizes:
        results[size] = run_size(size)
        out_path.write_text(json.dumps(results, indent=2))

    print(f"\n💾 Saved real results to: {out_path}")


if __name__ == "__main__":
    main()
