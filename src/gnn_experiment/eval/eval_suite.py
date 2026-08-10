"""Astral Tooling Evaluation Suite (Base Qwen3.5 vs Fine-Tuned Astral Qwen3.5).

Evaluates modern tooling adherence (uv, ruff, ty) vs legacy fallback (pip, black, flake8, mypy).
"""

import re
import time
from typing import Any

import mlflow
import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODERN_TERMS = [
    r"\buv\b",
    r"\bruff\b",
    r"\bty\b",
    r"uv add",
    r"uv run",
    r"uv sync",
    r"uv lock",
    r"ruff check",
    r"ruff format",
]
LEGACY_TERMS = [
    r"pip install",
    r"requirements\.txt",
    r"\bblack\b",
    r"\bflake8\b",
    r"\bisort\b",
    r"\bmypy\b",
    r"virtualenv",
    r"\bpoetry\b",
]


def count_matches(text: str, patterns: list[str]) -> int:
    total = 0
    text_lower = text.lower()
    for p in patterns:
        matches = re.findall(p, text_lower)
        total += len(matches)
    return total


def evaluate_single_prompt(
    model, tokenizer, prompt: str, max_new_tokens: int = 256
) -> dict[str, Any]:
    formatted_prompt = f"### Question:\n{prompt}\n\n### Answer:\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    start_t = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    gen_time_s = time.perf_counter() - start_t

    new_tokens = outputs[0][inputs.input_ids.shape[1] :]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    modern_hits = count_matches(response_text, MODERN_TERMS)
    legacy_hits = count_matches(response_text, LEGACY_TERMS)
    total_hits = modern_hits + legacy_hits
    adherence_pct = (
        (modern_hits / max(1, total_hits)) * 100.0 if total_hits > 0 else 0.0
    )

    return {
        "prompt": prompt,
        "response": response_text,
        "gen_time_s": gen_time_s,
        "modern_hits": modern_hits,
        "legacy_hits": legacy_hits,
        "adherence_pct": adherence_pct,
    }


def run_astral_evaluation(
    model_name: str = "Qwen/Qwen3.5-4B",
    adapter_path: str = "results/adapters/astral_qwen3.5_micro",
    questions_file: str = "configs/eval_questions.yaml",
    experiment_name: str = "astral_tooling_evaluation",
) -> dict[str, Any]:
    with open(questions_file) as f:
        q_cfg = yaml.safe_load(f)
    questions = q_cfg.get("questions", [])

    mlflow.set_experiment(experiment_name)

    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
    )

    print(f"Loading exact tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 1. Evaluate Base Model ---
    print("\n==================================================")
    print(f" 1. Evaluating Base Model: {model_name}")
    print("==================================================")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    base_model.eval()

    base_results = []
    for idx, q in enumerate(questions, 1):
        print(
            f"[{idx}/{len(questions)}] Evaluating Base Model on prompt: '{q['prompt'][:60]}...'"
        )
        res = evaluate_single_prompt(base_model, tokenizer, q["prompt"])
        res["id"] = q["id"]
        res["category"] = q["category"]
        base_results.append(res)

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 2. Evaluate Fine-Tuned Model ---
    print("\n==================================================")
    print(f" 2. Evaluating Fine-Tuned Astral Model: {adapter_path}")
    print("==================================================")
    base_model_for_adapter = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    ft_model = PeftModel.from_pretrained(base_model_for_adapter, adapter_path)
    ft_model.eval()

    ft_results = []
    for idx, q in enumerate(questions, 1):
        print(
            f"[{idx}/{len(questions)}] Evaluating Astral Model on prompt: '{q['prompt'][:60]}...'"
        )
        res = evaluate_single_prompt(ft_model, tokenizer, q["prompt"])
        res["id"] = q["id"]
        res["category"] = q["category"]
        ft_results.append(res)

    del ft_model, base_model_for_adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- 3. Compute Summary Metrics & Log to MLflow ---
    base_avg_adherence = sum(r["adherence_pct"] for r in base_results) / len(
        base_results
    )
    ft_avg_adherence = sum(r["adherence_pct"] for r in ft_results) / len(ft_results)
    adherence_gain = ft_avg_adherence - base_avg_adherence

    base_modern_total = sum(r["modern_hits"] for r in base_results)
    base_legacy_total = sum(r["legacy_hits"] for r in base_results)
    ft_modern_total = sum(r["modern_hits"] for r in ft_results)
    ft_legacy_total = sum(r["legacy_hits"] for r in ft_results)

    with mlflow.start_run(run_name=f"eval_{model_name.replace('/', '_')}"):
        mlflow.log_params(
            {
                "exact_model_name": model_name,
                "adapter_path": adapter_path,
                "num_questions": len(questions),
            }
        )

        mlflow.log_metrics(
            {
                "base_modern_adherence_pct": base_avg_adherence,
                "finetuned_modern_adherence_pct": ft_avg_adherence,
                "adherence_gain_pct": adherence_gain,
                "base_modern_hits_total": base_modern_total,
                "base_legacy_hits_total": base_legacy_total,
                "finetuned_modern_hits_total": ft_modern_total,
                "finetuned_legacy_hits_total": ft_legacy_total,
            }
        )

        comparison_table = []
        for b, f in zip(base_results, ft_results, strict=True):
            comparison_table.append(
                {
                    "id": b["id"],
                    "prompt": b["prompt"],
                    "base_response": b["response"],
                    "base_adherence_pct": b["adherence_pct"],
                    "finetuned_response": f["response"],
                    "finetuned_adherence_pct": f["adherence_pct"],
                }
            )

        out_summary = {
            "exact_model_name": model_name,
            "adapter_path": adapter_path,
            "base_avg_adherence_pct": base_avg_adherence,
            "finetuned_avg_adherence_pct": ft_avg_adherence,
            "adherence_gain_pct": adherence_gain,
            "comparison": comparison_table,
        }
        mlflow.log_dict(out_summary, "evaluation_comparison.json")

    print("\n==================================================")
    print(f" Evaluation Benchmark Results ({model_name})")
    print("==================================================")
    print(f" Exact Base Model: {model_name}")
    print(f" Base Model Modern Tool Adherence: {base_avg_adherence:.1f}%")
    print(f" Astral Fine-Tuned Model Modern Tool Adherence: {ft_avg_adherence:.1f}%")
    print(f" Net Modern Stack Adherence Gain: +{adherence_gain:.1f}%")
    print(f" Base Legacy Hits Total (pip/black/mypy): {base_legacy_total}")
    print(f" Astral Legacy Hits Total (pip/black/mypy): {ft_legacy_total}")
    print(f" MLflow Run Logged Successfully under '{experiment_name}'!")

    return out_summary
