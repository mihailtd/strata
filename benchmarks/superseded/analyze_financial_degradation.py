"""Diagnostic analysis script for financial_planning model degradation.

Evaluates Base Model vs Stock LoRA financial adapter (ctl_lora_fin_a128) on all 20
evaluation prompts, capturing exact outputs, adherence scoring matches/failures,
and inspecting dataset statistics for training data noise.
"""

import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from gnn_experiment.novel_peft import (  # noqa: E402
    FoldableExpert,
    WeightFoldingEngine,
    set_hard_vram_cap,
)

EVAL_PATH = REPO_ROOT / "data/financial_planning/evaluation_data.jsonl"
TRAIN_PATH = REPO_ROOT / "data/financial_planning/training_data.jsonl"
ADAPTER_PATH = REPO_ROOT / "results/adapters/ctl_lora_fin_a128"


def evaluate_adherence(output_text: str, expects: list[str]) -> bool:
    """Check if all required regex terms in 'expects' are matched in output_text."""
    return all(re.search(pattern, output_text, re.IGNORECASE) for pattern in expects)


@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 128) -> str:
    formatted_prompt = f"### Question:\n{prompt}\n\n### Answer:\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only new tokens
    gen_tokens = output_ids[0][inputs.input_ids.shape[1] :]
    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    # Truncate at ### Question if model hallucinates next question
    if "### Question:" in text:
        text = text.split("### Question:")[0].strip()
    return text.strip()


def analyze_training_data():
    print("═" * 90)
    print("📊 ANALYZING FINANCIAL PLANNING TRAINING DATASET")
    print("═" * 90)

    records = []
    with open(TRAIN_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    total = len(records)
    hey_count = 0
    book_excerpt_count = 0
    bullet_count = 0
    short_answers = 0

    excerpt_patterns = [
        r"excerpt",
        r"section",
        r"according to",
        r"this book",
        r"chapter",
        r"author",
    ]

    for r in records:
        ans = r["messages"][1]["content"]
        user = r["messages"][0]["content"]

        if ans.strip().lower().startswith("hey"):
            hey_count += 1

        if any(re.search(pat, user, re.IGNORECASE) for pat in excerpt_patterns):
            book_excerpt_count += 1

        if "\n*" in ans or "\n-" in ans or "\n1." in ans:
            bullet_count += 1

        if len(ans.split()) < 20:
            short_answers += 1

    print(f"Total training examples: {total}")
    print(f"Answers starting with 'Hey!/Hey,': {hey_count} ({100.0 * hey_count / total:.1f}%)")
    print(
        f"Prompts referencing 'excerpt/section/book': {book_excerpt_count} "
        f"({100.0 * book_excerpt_count / total:.1f}%)"
    )
    print(f"Answers with bullet points / lists: {bullet_count} ({100.0 * bullet_count / total:.1f}%)")
    print(f"Short answers (<20 words): {short_answers} ({100.0 * short_answers / total:.1f}%)")
    print("─" * 90)


def main():
    analyze_training_data()

    set_hard_vram_cap(22.0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading base model onto {device}...")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-4B", dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()

    eval_items = []
    with open(EVAL_PATH) as f:
        for line in f:
            if line.strip():
                eval_items.append(json.loads(line))

    # 1. Evaluate Base Model
    print("\n▶ EVALUATING BASE MODEL (UNADAPTED)...")
    base_results = []
    base_correct = 0
    for item in eval_items:
        ans = generate_response(model, tokenizer, item["prompt"])
        passed = evaluate_adherence(ans, item["expects"])
        if passed:
            base_correct += 1
        base_results.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "expects": item["expects"],
            "ans": ans,
            "passed": passed,
        })

    score_pct = 100.0 * base_correct / len(eval_items)
    print(f"Base Model Score: {base_correct}/{len(eval_items)} ({score_pct:.2f}%)")

    # 2. Evaluate Financial Adapter
    print("\n▶ FOLDING FINANCIAL EXPERT (ctl_lora_fin_a128)...")
    expert = FoldableExpert.from_dir(ADAPTER_PATH, "financial_planning")
    engine = WeightFoldingEngine(model, [expert], keep_pristine=True)
    engine.activate(expert)

    adapter_results = []
    adapter_correct = 0
    for item in eval_items:
        ans = generate_response(model, tokenizer, item["prompt"])
        passed = evaluate_adherence(ans, item["expects"])
        if passed:
            adapter_correct += 1
        adapter_results.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "expects": item["expects"],
            "ans": ans,
            "passed": passed,
        })

    ad_score_pct = 100.0 * adapter_correct / len(eval_items)
    print(f"Financial Adapter Score: {adapter_correct}/{len(eval_items)} ({ad_score_pct:.2f}%)")

    # 3. Detailed Failure Diff Analysis
    print("\n" + "═" * 90)
    print("🔍 DETAILED PROMPT-BY-PROMPT ERROR DIFF (BASE vs ADAPTED)")
    print("═" * 90)

    regressions = []
    improvements = []
    both_failed = []
    both_passed = []

    for b, a in zip(base_results, adapter_results, strict=True):
        if b["passed"] and not a["passed"]:
            regressions.append((b, a))
        elif not b["passed"] and a["passed"]:
            improvements.append((b, a))
        elif not b["passed"] and not a["passed"]:
            both_failed.append((b, a))
        else:
            both_passed.append((b, a))

    print(f"Both Passed: {len(both_passed)}")
    print(f"Both Failed: {len(both_failed)}")
    print(f"Improvements (Base Fail -> Adapter Pass): {len(improvements)}")
    print(f"REGRESSIONS (Base Pass -> Adapter Fail): {len(regressions)}")

    if regressions:
        print("\n" + "🚨 REGRESSION DETAILS (Base Passed, Adapter Failed):")
        print("─" * 90)
        for b, a in regressions:
            print(f"\nID: {b['id']} | Category: {b.get('category', '')}")
            print(f"Prompt: {b['prompt']}")
            print(f"Expects Regex: {b['expects']}")
            print(f"\n[BASE OUTPUT - PASSED]:\n{b['ans']}")
            print(f"\n[ADAPTER OUTPUT - FAILED]:\n{a['ans']}")
            # Find missing expects
            missing = [p for p in b['expects'] if not re.search(p, a['ans'], re.IGNORECASE)]
            print(f"Missing Expects in Adapter Output: {missing}")
            print("─" * 90)

    if improvements:
        print("\n" + "✨ IMPROVEMENT DETAILS (Base Failed, Adapter Passed):")
        print("─" * 90)
        for b, a in improvements:
            print(f"\nID: {b['id']}")
            print(f"Prompt: {b['prompt']}")
            print(f"Expects Regex: {b['expects']}")
            print(f"\n[BASE OUTPUT - FAILED]:\n{b['ans']}")
            print(f"\n[ADAPTER OUTPUT - PASSED]:\n{a['ans']}")
            print("─" * 90)

    if both_failed:
        print("\n" + "❌ BOTH FAILED DETAILS:")
        print("─" * 90)
        for b, a in both_failed:
            print(f"\nID: {b['id']}")
            print(f"Prompt: {b['prompt']}")
            print(f"Expects Regex: {b['expects']}")
            print(f"\n[BASE OUTPUT]:\n{b['ans']}")
            print(f"\n[ADAPTER OUTPUT]:\n{a['ans']}")
            print("─" * 90)


if __name__ == "__main__":
    main()
