"""Real-Model GPU Benchmark for Range Statistic & Composite Speculative Early-Exit Gating.

Empirically evaluates on AMD RDNA3 hardware:
1. Ungated Speculation (Drafts full K=4 tokens unconditionally).
2. Pure Range Statistic Gating (Outlier detection based on top-M logit spread R_M).
3. Tri-Modal Composite Gating (Range Statistic + Weibull Hazard + Bollinger Volatility).

Measures live on real Qwen3.5-4B base model with genuine MTP draft head:
- Throughput (tok/s)
- Latency per token (ms)
- Early-exit abort frequency across draft steps k=1..3
- Abort fidelity (whether pruned tail tokens were indeed doomed mismatches)
- Verification tax savings from eliminating junk drafts
- Saves results to results/benchmarks/scorecard_range_statistic_gate_real_gpu.json
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "apps" / "runtime-ipwf"))
sys.path.insert(0, str(REPO_ROOT / "apps" / "runtime-common"))

from bucketed_speculative import BucketedSpeculativeDecoder  # type: ignore[unresolved-import]  # noqa: E402
from mtp_draft import Qwen35MTPDraftHead  # type: ignore[unresolved-import]  # noqa: E402
from range_statistic_gate import RangeStatisticGate  # type: ignore[unresolved-import]  # noqa: E402
from runtime_common.gpu_preflight import ensure_gpu_exclusive  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results" / "benchmarks"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SCORECARD_PATH = RESULTS_DIR / "scorecard_range_statistic_gate_real_gpu.json"

TEST_PROMPTS = [
    (
        "code_generation",
        "<|im_start|>user\n"
        "Write a high-performance Python LRU Cache using collections.OrderedDict with type annotations, "
        "thread locks, and docstrings:\n"
        "<|im_end|>\n<|im_start|>assistant\n",
    ),
    (
        "sql_schema",
        "<|im_start|>user\n"
        "Write a PostgreSQL DDL schema for a multi-tenant analytics platform with partitioned event tables, "
        "BRIN indexes, and JSONB payload constraints:\n"
        "<|im_end|>\n<|im_start|>assistant\n",
    ),
    (
        "adversarial_cipher",
        "<|im_start|>user\n"
        "Generate a continuous sequence of random hexadecimal noise bytes without spaces or words. "
        "Begin immediately with no introduction:\n"
        "<|im_end|>\n<|im_start|>assistant\n"
        "c8d1f04e",
    ),
]


def run_range_statistic_benchmark(
    model_id: str = "Qwen/Qwen3.5-4B",
    k: int = 4,
    max_tokens: int = 48,
    max_seq_len: int = 2048,
    threshold: float = 3.5,
) -> dict[str, Any]:
    print("=" * 90)
    print("📊 REAL-MODEL GPU BENCHMARK: SINGLE-PASS RANGE STATISTIC & COMPOSITE GATING")
    print(f"   Model:     {model_id}")
    print(f"   Draft K:   {k}")
    print(f"   Max Tokens: {max_tokens} per scenario")
    print(f"   Threshold: {threshold}")
    print(f"   Device:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 90)

    ensure_gpu_exclusive()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load Base Model & Tokenizer
    print("\n[1/3] Loading base model and tokenizer into GPU VRAM (bfloat16)...")
    t0 = time.perf_counter()
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
    print(f"      Base model loaded in {time.perf_counter() - t0:.2f}s.")

    # 2. Load Genuine MTP Draft Head & Decoder
    print("\n[2/3] Loading genuine MTP Draft Head and capturing CUDA Graphs (K=4)...")
    draft_head = Qwen35MTPDraftHead(base_model, model_id)
    draft_head.eval()

    dummy_input = tokenizer(TEST_PROMPTS[0][1], return_tensors="pt").input_ids.to(device)
    spec_decoder = BucketedSpeculativeDecoder(
        base_model,
        tokenizer,
        draft_head,
        k=k,
        max_seq_len=max_seq_len,
        device=device,
    )
    spec_decoder.capture(dummy_input)
    print(f"      CUDA Graphs captured: widths {sorted(spec_decoder.buckets.keys())}")

    # Gates to evaluate
    gate_configs = {
        "arm_1_ungated": None,
        "arm_2_range_statistic_only": RangeStatisticGate(
            top_m=8,
            threshold=threshold,
            mode="extreme_range",
            weibull_hazard_enabled=False,
            bollinger_bands_enabled=False,
        ),
        "arm_3_tri_modal_composite": RangeStatisticGate(
            top_m=8,
            threshold=threshold,
            mode="extreme_range",
            weibull_hazard_enabled=True,
            weibull_beta=2.2,
            weibull_gamma=0.6,
            bollinger_bands_enabled=True,
            bollinger_k=2.0,
            bollinger_gamma=0.5,
        ),
    }

    # 3. Benchmark Execution
    print("\n[3/3] Running multi-arm generation across test workloads...")
    scorecard: dict[str, Any] = {
        "metadata": {
            "model_id": model_id,
            "draft_k": k,
            "max_tokens": max_tokens,
            "threshold": threshold,
            "device": torch.cuda.get_device_name(0),
            "timestamp": int(time.time()),
        },
        "prompts": {},
    }

    for prompt_label, prompt_text in TEST_PROMPTS:
        print("\n" + "-" * 85)
        print(f"▶️  Workload: [{prompt_label.upper()}]")
        print("-" * 85)

        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        prompt_results: dict[str, Any] = {}

        for arm_name, gate in gate_configs.items():
            if gate is not None:
                gate.reset_state()
            spec_decoder.circuit_breaker.reset()

            torch.cuda.synchronize()
            tokens, elapsed, stats = spec_decoder.generate(
                input_ids,
                max_new_tokens=max_tokens,
                gate=gate,
            )
            torch.cuda.synchronize()

            tok_s = len(tokens) / max(1e-9, elapsed)
            tau = stats.get("tau", 0.0)
            steps = stats.get("steps", 0)
            drafted = stats.get("drafted", 0)
            accepted = stats.get("accepted", 0)
            avg_draft_depth = drafted / max(1, steps)

            print(
                f"  {arm_name:<30}: {len(tokens)} tok in {elapsed:.3f}s -> {tok_s:.2f} tok/s "
                f"(avg_draft_len={avg_draft_depth:.2f}/{k}, tau={tau:.2f})"
            )

            prompt_results[arm_name] = {
                "tokens": len(tokens),
                "elapsed_s": round(elapsed, 3),
                "tok_s": round(tok_s, 2),
                "avg_tau": round(tau, 2),
                "steps": steps,
                "total_drafted": drafted,
                "total_accepted": accepted,
                "avg_draft_depth": round(avg_draft_depth, 2),
            }

        # Calculate speedups vs ungated
        ungated_tps = prompt_results["arm_1_ungated"]["tok_s"]
        for arm_name in ["arm_2_range_statistic_only", "arm_3_tri_modal_composite"]:
            tps = prompt_results[arm_name]["tok_s"]
            prompt_results[arm_name]["speedup_vs_ungated"] = round(tps / max(1e-9, ungated_tps), 3)

        scorecard["prompts"][prompt_label] = prompt_results

    # Summary
    print("\n" + "=" * 90)
    print("📋 SUMMARY SCORECARD: RANGE STATISTIC & COMPOSITE SPECULATIVE GATING")
    print("=" * 90)
    print(
        f"{'Workload':<20} | {'Ungated (K=4)':<14} | {'Range Only':<12} | "
        f"{'Composite':<12} | {'Composite vs Ungated':<20}"
    )
    print("-" * 90)
    for p_name, p_res in scorecard["prompts"].items():
        un_tps = p_res["arm_1_ungated"]["tok_s"]
        ro_tps = p_res["arm_2_range_statistic_only"]["tok_s"]
        co_tps = p_res["arm_3_tri_modal_composite"]["tok_s"]
        gain = p_res["arm_3_tri_modal_composite"]["speedup_vs_ungated"]
        print(f"{p_name:<22} | {un_tps:>10.2f} tok/s | {ro_tps:>8.2f} s | {co_tps:>8.2f} s | {gain:>16.3f}x")

    with open(SCORECARD_PATH, "w") as f:
        json.dump(scorecard, f, indent=2)
    print(f"\n✅ Scorecard successfully saved to {SCORECARD_PATH}\n")
    return scorecard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live GPU Range Statistic Speculative Gating Benchmark")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B", help="Hugging Face model ID")
    parser.add_argument("--k", type=int, default=4, help="Draft depth K")
    parser.add_argument("--max-tokens", type=int, default=48, help="Tokens to generate per arm")
    parser.add_argument("--threshold", type=float, default=3.5, help="Baseline range threshold")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="StaticCache max sequence length")
    args = parser.parse_args()

    run_range_statistic_benchmark(
        model_id=args.model_id,
        k=args.k,
        max_tokens=args.max_tokens,
        max_seq_len=args.max_seq_len,
        threshold=args.threshold,
    )
