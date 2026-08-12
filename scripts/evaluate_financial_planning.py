"""Evaluation Harness for Financial Planning Adapters.

Evaluates Base Qwen3.5-4B vs Variant 1 (financial_planning_krona_dora) vs Variant 2 (financial_planning_standard_lora).
Measures:
1. Domain expertise & financial psychology concept adherence across 20 evaluation prompts.
2. In-Place Weight Folding (W_0 ± ΔW) latency comparison (ms) between the micro-adapter and standard LoRA control.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from gnn_experiment.novel_peft import (  # noqa: E402
    FoldedAdapterPayload,
    fold_adapter_in_place,
    prepare_folding_slots,
    set_hard_vram_cap,
    unfold_adapter_in_place,
)
from gnn_experiment.utils.logger import log_benchmark_metric  # noqa: E402

FINANCIAL_TERMS = [
    r"\bmoney script\b",
    r"\bmoney scripts\b",
    r"\bloss aversion\b",
    r"\brisk capacity\b",
    r"\brisk tolerance\b",
    r"\brisk perception\b",
    r"\bbehavioral finance\b",
    r"\bfinancial therapy\b",
    r"\bfinancial psychology\b",
    r"\bactive listening\b",
    r"\bmotivational interviewing\b",
    r"\bmental accounting\b",
    r"\bsequence of returns\b",
    r"\brecency bias\b",
    r"\bconfirmation bias\b",
    r"\bfinancial flashpoint\b",
]

GENERAL_FILLER_TERMS = [
    r"\bconsult a professional\b",
    r"\bseek advice\b",
    r"\bI am an AI\b",
    r"\bgeneral information\b",
    r"\bas an AI language model\b",
]


def count_matches(text: str, patterns: list[str]) -> int:
    total = 0
    text_lower = text.lower()
    for p in patterns:
        matches = re.findall(p, text_lower)
        total += len(matches)
    return total


def evaluate_single_prompt(model, tokenizer, prompt: str, max_new_tokens: int = 192) -> dict:
    formatted = f"### Question:\n{prompt}\n\n### Answer:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    start_t = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen_time_s = time.perf_counter() - start_t

    new_tokens = outputs[0][inputs.input_ids.shape[1] :]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    fin_hits = count_matches(response_text, FINANCIAL_TERMS)
    filler_hits = count_matches(response_text, GENERAL_FILLER_TERMS)

    return {
        "prompt": prompt,
        "response": response_text,
        "gen_time_s": gen_time_s,
        "fin_hits": fin_hits,
        "filler_hits": filler_hits,
        "tok_per_sec": len(new_tokens) / max(1e-5, gen_time_s),
    }


def measure_folding_latency(model, adapter_dir: Path, num_trials: int = 30) -> float:
    """Measures in-place weight folding latency for an adapter directory."""
    if (adapter_dir / "novel_adapter.pt").exists():
        state_dict = torch.load(adapter_dir / "novel_adapter.pt", map_location="cpu")
    else:
        from peft.utils import load_peft_weights
        state_dict = load_peft_weights(str(adapter_dir))

    payload = FoldedAdapterPayload.compute_from_state_dict(state_dict, name=adapter_dir.name)
    slots = prepare_folding_slots(model, payload)

    latencies = []
    for _ in range(num_trials):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fold_adapter_in_place(slots, payload, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        unfold_adapter_in_place(slots, payload, non_blocking=True)

    latencies.sort()
    return latencies[len(latencies) // 2]


def run_evaluation(
    model_name: str = "Qwen/Qwen3.5-4B",
    questions_file: str = "data/financial_planning/evaluation_data.jsonl",
):
    set_hard_vram_cap(22.0)
    questions_path = REPO_ROOT / questions_file
    with open(questions_path) as f:
        questions = [json.loads(line) for line in f if line.strip()]

    print("==================================================")
    print(f" Financial Planning Adapter Evaluation ({model_name})")
    print("==================================================")

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    print(f"Loading tokenizer & base model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=compute_dtype,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    base_model.eval()

    # 1. Base Model Evaluation
    print("\n--- 1. Evaluating Base Model (Generic Baseline) ---")
    base_results = []
    for idx, q in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] Base Model on: '{q['prompt'][:50]}...'")
        res = evaluate_single_prompt(base_model, tokenizer, q["prompt"])
        res["id"] = q["id"]
        base_results.append(res)

    base_fin_hits = sum(r["fin_hits"] for r in base_results)
    base_filler_hits = sum(r["filler_hits"] for r in base_results)

    # 2. Evaluate Variant 1: financial_planning_krona_dora
    var1_dir = REPO_ROOT / "results" / "adapters" / "financial_planning_krona_dora"
    print(f"\n--- 2. Measuring & Evaluating Variant 1 (financial_planning_krona_dora) ---")
    var1_fold_latency = measure_folding_latency(base_model, var1_dir)
    print(f"Variant 1 In-Place Fold Latency (p50): {var1_fold_latency:.3f} ms")

    # Load & evaluate via Peft / Novel adapter
    var1_model = PeftModel.from_pretrained(base_model, str(var1_dir))
    var1_model.eval()
    var1_results = []
    for idx, q in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] Variant 1 on: '{q['prompt'][:50]}...'")
        res = evaluate_single_prompt(var1_model, tokenizer, q["prompt"])
        res["id"] = q["id"]
        var1_results.append(res)

    del var1_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    var1_fin_hits = sum(r["fin_hits"] for r in var1_results)
    var1_filler_hits = sum(r["filler_hits"] for r in var1_results)

    # 3. Evaluate Variant 2: financial_planning_standard_lora
    var2_dir = REPO_ROOT / "results" / "adapters" / "financial_planning_standard_lora"
    print(f"\n--- 3. Measuring & Evaluating Variant 2 (financial_planning_standard_lora) ---")
    var2_fold_latency = measure_folding_latency(base_model, var2_dir)
    print(f"Variant 2 In-Place Fold Latency (p50): {var2_fold_latency:.3f} ms")

    var2_model = PeftModel.from_pretrained(base_model, str(var2_dir))
    var2_model.eval()
    var2_results = []
    for idx, q in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] Variant 2 on: '{q['prompt'][:50]}...'")
        res = evaluate_single_prompt(var2_model, tokenizer, q["prompt"])
        res["id"] = q["id"]
        var2_results.append(res)

    var2_fin_hits = sum(r["fin_hits"] for r in var2_results)
    var2_filler_hits = sum(r["filler_hits"] for r in var2_results)

    print("\n==================================================")
    print(" Financial Planning Adapter Benchmark Summary")
    print("==================================================")
    print(f" Base Model Domain Concept Hits : {base_fin_hits} (Filler: {base_filler_hits})")
    print(f" Variant 1 (KronA micro) Hits   : {var1_fin_hits} (Filler: {var1_filler_hits}) | Fold Latency: {var1_fold_latency:.3f} ms")
    print(f" Variant 2 (Standard LoRA) Hits : {var2_fin_hits} (Filler: {var2_filler_hits}) | Fold Latency: {var2_fold_latency:.3f} ms")

    summary = {
        "model_name": model_name,
        "base_model": {"domain_hits": base_fin_hits, "filler_hits": base_filler_hits},
        "variant_1_krona_dora": {
            "path": str(var1_dir),
            "domain_hits": var1_fin_hits,
            "filler_hits": var1_filler_hits,
            "fold_latency_ms": var1_fold_latency,
        },
        "variant_2_standard_lora": {
            "path": str(var2_dir),
            "domain_hits": var2_fin_hits,
            "filler_hits": var2_filler_hits,
            "fold_latency_ms": var2_fold_latency,
        },
    }

    out_file = REPO_ROOT / "results" / "financial_planning_eval_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log_benchmark_metric(
        {
            "experiment": "financial_planning_evaluation",
            "model_name": model_name,
            "base_domain_hits": base_fin_hits,
            "var1_krona_domain_hits": var1_fin_hits,
            "var1_fold_latency_ms": var1_fold_latency,
            "var2_lora_domain_hits": var2_fin_hits,
            "var2_fold_latency_ms": var2_fold_latency,
        },
        filepath="results/eval_runs.jsonl",
    )

    print(f"\nSaved evaluation summary to {out_file}")


if __name__ == "__main__":
    run_evaluation()
