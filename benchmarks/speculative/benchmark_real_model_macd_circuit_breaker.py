"""Real-Model GPU Benchmark for MACD Speculation Circuit-Breaker on AMD RDNA3.

Evaluates the Dual-EMA / MACD Speculation Circuit-Breaker using live GPU inference:
- Real Qwen3.5-4B base model in bfloat16.
- Real genuine MTP draft head from Qwen3.5-4B weights.
- Captured CUDA/HIP graph execution with StaticCache and Selective Ring Buffer.
- Compares 3 live arms across High-Alignment, Low-Alignment, and Mixed-Regime prompts:
    Arm 1: Raw W=1 CUDA Graph Baseline (speed floor)
    Arm 2: Unprotected Speculative Decoding (circuit breaker DISABLED, paying verification tax)
    Arm 3: Protected Speculative Decoding (circuit breaker ENABLED, tripping to W=1 floor)
- Collects real hardware latency, tok/s throughput, acceptance rate (tau), trip events,
  and saves live telemetry scorecard to results/benchmarks/scorecard_macd_circuit_breaker_real_gpu.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Workspace paths
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "runtime-ipwf"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "runtime-common"))

from bucketed_speculative import BucketedSpeculativeDecoder  # type: ignore[unresolved-import]  # noqa: E402
from cuda_graph import FoldedCudaGraphDecoder  # type: ignore[unresolved-import]  # noqa: E402
from mtp_draft import Qwen35MTPDraftHead  # type: ignore[unresolved-import]  # noqa: E402
from runtime_common.gpu_preflight import ensure_gpu_exclusive  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SCORECARD_PATH = RESULTS_DIR / "scorecard_macd_circuit_breaker_real_gpu.json"


PROMPTS = {
    "high_alignment": (
        "<|im_start|>user\n"
        "Write a clean, standard Python class DatabaseConnectionPool that manages a pool of "
        "PostgreSQL connections with acquire and release methods, docstrings, and type annotations:\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    ),
    "low_alignment": (
        "<|im_start|>user\n"
        "Generate a continuous sequence of random hexadecimal noise bytes without spaces or words. "
        "Begin immediately with no introduction:\n"
        "<|im_end|>\n<|im_start|>assistant\n"
        "a4f8c92b"
    ),
    "mixed_regime": (
        "<|im_start|>user\n"
        "Output 16 arbitrary hex characters, and then write standard Python imports "
        "for asyncio, typing, and dataclasses:\n"
        "<|im_end|>\n<|im_start|>assistant\n"
        "7d2f9a1c"
    ),
}


def run_real_model_macd_benchmark(
    model_id: str = "Qwen/Qwen3.5-4B",
    max_tokens: int = 64,
    k: int = 2,
    max_seq_len: int = 2048,
    disengage_threshold: float = 2.3,
    reengage_threshold: float = 2.5,
) -> dict[str, Any]:
    print("=" * 90)
    print("📊 REAL-MODEL GPU BENCHMARK: DUAL-EMA / MACD SPECULATION CIRCUIT-BREAKER")
    print(f"   Model:     {model_id}")
    print(f"   Spec K:    {k}")
    print(f"   Max Tokens: {max_tokens} per scenario")
    print(f"   Thresholds: disengage={disengage_threshold}, reengage={reengage_threshold}")
    print(f"   Device:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 90)

    ensure_gpu_exclusive()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load Tokenizer & Base Model
    print("\n[1/4] Loading base model and tokenizer into GPU VRAM (bfloat16)...")
    t_load0 = time.perf_counter()
    tokenizer: Any = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device.index if device.type == "cuda" else "cpu"},
        trust_remote_code=True,
    )
    base_model.eval()
    t_load = time.perf_counter() - t_load0
    vram_base = torch.cuda.memory_allocated() / (1024**3)
    print(f"      Base model loaded in {t_load:.2f}s ({vram_base:.2f} GB VRAM).")

    # 2. Load Genuine MTP Draft Head
    print("\n[2/4] Initializing genuine Qwen3.5 MTP Draft Head from checkpoint...")
    draft_head = Qwen35MTPDraftHead(base_model, model_id)
    draft_head.eval()
    vram_head = torch.cuda.memory_allocated() / (1024**3)
    print(f"      MTP Draft Head initialized ({vram_head:.2f} GB VRAM total).")

    # 3. Capture CUDA Graphs
    print(f"\n[3/4] Capturing CUDA Graphs (max_seq_len={max_seq_len}, widths 1..{k + 1})...")
    dummy_input = tokenizer(PROMPTS["high_alignment"], return_tensors="pt").input_ids.to(device)

    # 3a. Baseline Raw W=1 Graph Decoder
    graph_decoder = FoldedCudaGraphDecoder(base_model, tokenizer, max_seq_len=max_seq_len, device=device)
    graph_decoder.capture(dummy_input)

    # 3b. Bucketed Speculative Decoder
    spec_decoder = BucketedSpeculativeDecoder(
        base_model,
        tokenizer,
        draft_head,
        k=k,
        max_seq_len=max_seq_len,
        device=device,
    )
    spec_decoder.capture(dummy_input)
    print(f"      Graph capture complete. Buckets: {sorted(spec_decoder.buckets.keys())}")

    # 4. Execute Benchmark Across Scenarios
    print("\n[4/4] Running empirical multi-arm generation across regimes...")

    scorecard: dict[str, Any] = {
        "metadata": {
            "model_id": model_id,
            "spec_k": k,
            "max_tokens": max_tokens,
            "device": torch.cuda.get_device_name(0),
            "vram_allocated_gb": round(vram_head, 2),
            "timestamp": int(time.time()),
        },
        "scenarios": {},
    }

    for scenario_name, prompt_text in PROMPTS.items():
        print("\n" + "-" * 85)
        print(f"▶️  Scenario: [{scenario_name.upper()}]")
        print("-" * 85)

        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        scenario_results: dict[str, Any] = {}

        # ------------------------------------------------------------------
        # Arm 1: Raw W=1 CUDA Graph Baseline
        # ------------------------------------------------------------------
        torch.cuda.synchronize()
        tokens_raw, elapsed_raw, tps_raw, _ = graph_decoder.generate_with_graph(input_ids, max_new_tokens=max_tokens)
        torch.cuda.synchronize()
        print(
            f"  Arm 1 [Raw W=1 Graph Floor]     : {len(tokens_raw)} tokens in {elapsed_raw:.3f}s -> {tps_raw:.2f} tok/s"
        )
        scenario_results["arm_1_raw_graph"] = {
            "tokens": len(tokens_raw),
            "elapsed_s": round(elapsed_raw, 3),
            "tok_s": round(tps_raw, 2),
        }

        # ------------------------------------------------------------------
        # Arm 2: Unprotected Speculation (Circuit-Breaker Disabled)
        # ------------------------------------------------------------------
        spec_decoder.circuit_breaker.enabled = False
        torch.cuda.synchronize()
        tokens_unprot, elapsed_unprot, stats_unprot = spec_decoder.generate(input_ids, max_new_tokens=max_tokens)
        torch.cuda.synchronize()
        tps_unprot = len(tokens_unprot) / max(1e-9, elapsed_unprot)
        tau_unprot = stats_unprot.get("tau", 0.0)
        print(
            f"  Arm 2 [Unprotected Speculation] : {len(tokens_unprot)} tokens in {elapsed_unprot:.3f}s -> "
            f"{tps_unprot:.2f} tok/s (tau={tau_unprot:.2f})"
        )
        scenario_results["arm_2_unprotected_spec"] = {
            "tokens": len(tokens_unprot),
            "elapsed_s": round(elapsed_unprot, 3),
            "tok_s": round(tps_unprot, 2),
            "avg_tau": round(tau_unprot, 2),
            "speedup_vs_raw": round(tps_unprot / max(1e-9, tps_raw), 3),
        }

        # ------------------------------------------------------------------
        # Arm 3: Protected Speculation (Circuit-Breaker Enabled)
        # ------------------------------------------------------------------
        spec_decoder.circuit_breaker.enabled = True
        spec_decoder.circuit_breaker.disengage_threshold = disengage_threshold
        spec_decoder.circuit_breaker.reengage_threshold = reengage_threshold
        spec_decoder.circuit_breaker.reset()
        torch.cuda.synchronize()
        tokens_prot, elapsed_prot, stats_prot = spec_decoder.generate(input_ids, max_new_tokens=max_tokens)
        torch.cuda.synchronize()
        tps_prot = len(tokens_prot) / max(1e-9, elapsed_prot)
        tau_prot = stats_prot.get("tau", 0.0)
        cb_summary = stats_prot.get("circuit_breaker", {})
        trips = cb_summary.get("trips_count", 0)
        reengages = cb_summary.get("reengages_count", 0)
        disengaged_pct = cb_summary.get("disengaged_pct", 0.0)

        dis_pct = f"{disengaged_pct}%"
        print(
            f"  Arm 3 [Protected Spec (MACD CB)]: {len(tokens_prot)} tokens in {elapsed_prot:.3f}s -> "
            f"{tps_prot:.2f} tok/s (tau={tau_prot:.2f}, trips={trips}, reengages={reengages}, disengaged={dis_pct})"
        )
        scenario_results["arm_3_protected_spec"] = {
            "tokens": len(tokens_prot),
            "elapsed_s": round(elapsed_prot, 3),
            "tok_s": round(tps_prot, 2),
            "avg_tau": round(tau_prot, 2),
            "trips_count": trips,
            "reengages_count": reengages,
            "disengaged_pct": disengaged_pct,
            "speedup_vs_raw": round(tps_prot / max(1e-9, tps_raw), 3),
            "speedup_vs_unprotected": round(tps_prot / max(1e-9, tps_unprot), 3),
            "circuit_breaker_telemetry": cb_summary,
        }

        scorecard["scenarios"][scenario_name] = scenario_results

    # Summary table
    print("\n" + "=" * 90)
    print("📋 SUMMARY SCORECARD: LIVE HARDWARE DUAL-EMA / MACD CIRCUIT-BREAKER")
    print("=" * 90)
    print(
        f"{'Scenario':<18} | {'Raw W=1':<10} | {'Unprotected':<12} | "
        f"{'Protected (CB)':<15} | {'CB vs Unprot':<12} | {'Trips':<6}"
    )
    print("-" * 90)
    for sc_name, sc_res in scorecard["scenarios"].items():
        raw_tps = sc_res["arm_1_raw_graph"]["tok_s"]
        unp_tps = sc_res["arm_2_unprotected_spec"]["tok_s"]
        prot_tps = sc_res["arm_3_protected_spec"]["tok_s"]
        gain = sc_res["arm_3_protected_spec"]["speedup_vs_unprotected"]
        tr = sc_res["arm_3_protected_spec"]["trips_count"]
        print(
            f"{sc_name:<20} | {raw_tps:>8.2f} s | {unp_tps:>10.2f} s | {prot_tps:>13.2f} s | {gain:>10.3f}x | {tr:>5}"
        )

    # Save to disk
    with open(SCORECARD_PATH, "w") as f:
        json.dump(scorecard, f, indent=2)
    print(f"\n✅ Scorecard successfully saved to {SCORECARD_PATH}\n")
    return scorecard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live GPU MACD Circuit Breaker Benchmark")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B", help="Hugging Face model ID")
    parser.add_argument("--max-tokens", type=int, default=64, help="Tokens to generate per run")
    parser.add_argument("--k", type=int, default=2, help="Speculative draft depth")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="StaticCache max sequence length")
    parser.add_argument("--disengage-threshold", type=float, default=2.3, help="Disengage threshold")
    parser.add_argument("--reengage-threshold", type=float, default=2.5, help="Re-engage threshold")
    args = parser.parse_args()

    run_real_model_macd_benchmark(
        model_id=args.model_id,
        max_tokens=args.max_tokens,
        k=args.k,
        max_seq_len=args.max_seq_len,
        disengage_threshold=args.disengage_threshold,
        reengage_threshold=args.reengage_threshold,
    )
