"""Strict Head-to-Head Benchmark: 2-Stage Identity-Kronecker (id_kron) vs Stock LoRA

Evaluates 2-Stage Identity-Kronecker (id_kron) adapters against standard Stock LoRA
under identical, unquantized bfloat16 precision and the fixed evaluation harness
(do_sample=False, stop_strings=['### Question']).

USAGE:
    uv run python benchmarks/superseded/benchmark_id_kron_vs_stock_lora.py \
        --model-id Qwen/Qwen3.5-4B \
        --out results/id_kron_vs_stock_lora_benchmark.json

STATUS AFTER REVIEW: SUPERSEDED -- DO NOT CITE ITS CONCLUSIONS
---------------------------------------------------------------
This script evaluates four PRE-EXISTING adapters. It is not a controlled
comparison and its write-up drew three unsupported conclusions:

  * It reported id_kron at "r=64, 49.8M params". The adapter it loads is
    rank_total=8 / 6.26M. That INVERTS the headline -- it scored 50.50% using
    41% FEWER parameters than the Stock LoRA (10.62M) that "beat" it. Every
    "r=64" label is wrong: the LoRAs are r=8, the Kroneckers rank_total 8 / 16.
    TODO.md already contained the correct parameter table.
  * No training records exist for any of the four adapters -- steps, LR, data
    vintage and code version are unknown, including whether they predate the
    ~18x fan-in init fix.
  * Effective scaling was uncontrolled (Stock LoRA a256 = 32.0 vs id_kron_r16
    = 2.0) and scaling is the dominant lever, so architecture and scaling are
    fully confounded.

Its 72.44% for id_kron_r16 does not reproduce under controlled conditions
(best of 15 fresh adapters: 57.83%; rt16 peaks at 56.24%).

Use scripts/eval_controlled_headtohead.py instead: 21 adapters trained fresh
with matched effective scaling, each architecture compared at its own peak.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def load_bfloat16_base_model(model_id: str):
    """Loads base model in unquantized bfloat16 precision."""
    print(f"Loading unquantized bfloat16 base model {model_id} onto GPU...")
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0" if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )


def load_questions(questions_file: str) -> list[dict]:
    path = Path(questions_file)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate_model_on_questions(model, tokenizer, questions: list[dict]) -> dict:
    from runtime.eval.eval_suite import (
        LEGACY_TERMS,
        MODERN_TERMS,
        evaluate_single_prompt,
    )

    results = []
    for idx, q in enumerate(questions, 1):
        res = evaluate_single_prompt(
            model,
            tokenizer,
            q["prompt"],
            modern_terms=MODERN_TERMS,
            legacy_terms=LEGACY_TERMS,
        )
        res["id"] = q.get("id", idx)
        results.append(res)

    avg_adherence = sum(r["adherence_pct"] for r in results) / len(results)
    modern_hits = sum(r["modern_hits"] for r in results)
    legacy_hits = sum(r["legacy_hits"] for r in results)

    return {
        "avg_adherence_pct": avg_adherence,
        "modern_hits_total": modern_hits,
        "legacy_hits_total": legacy_hits,
        "detailed_results": results,
    }


def main():
    from runtime.novel_peft import load_novel_adapter
    parser = argparse.ArgumentParser(
        description="Head-to-Head Benchmark: id_kron vs Stock LoRA (bfloat16, fixed harness)"
    )
    parser.add_argument(
        "--model-id", default="Qwen/Qwen3.5-4B", help="Base model checkpoint"
    )
    parser.add_argument(
        "--questions-file",
        default="data/astral/evaluation_data.jsonl",
        help="Questions JSONL file",
    )
    parser.add_argument(
        "--out",
        default="results/id_kron_vs_stock_lora_benchmark.json",
        help="Output report JSON",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"🔬 AUDITING ID_KRON VS STOCK LORA ON {device} ({dev_name})\n" + "─" * 80)

    questions = load_questions(args.questions_file)
    print(f"Loaded {len(questions)} evaluation prompts from {args.questions_file}")

    print(f"Loading tokenizer for {args.model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Base Model Baseline (bfloat16)
    print("\n1/5. Evaluating Unquantized bfloat16 Base Model...")
    model = load_bfloat16_base_model(args.model_id)
    base_metrics = evaluate_model_on_questions(model, tokenizer, questions)
    print(f"   Base Model Adherence: {base_metrics['avg_adherence_pct']:.2f}%")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Define candidate variants to test head-to-head
    adapters_to_test = [
        {
            "name": "Stock LoRA (a256)",
            "dir": "astral_qwen3.5_micro_lora_a256",
            "type": "peft",
        },
        {
            "name": "Stock LoRA (standard)",
            "dir": "astral_qwen3.5_micro_lora",
            "type": "peft",
        },
        {
            "name": "2-Stage id_kron",
            "dir": "astral_qwen3.5_micro_id_kron",
            "type": "novel",
        },
        {
            "name": "2-Stage id_kron (r16)",
            "dir": "astral_qwen3.5_micro_id_kron_r16",
            "type": "novel",
        },
    ]

    variant_results = {}

    for idx, item in enumerate(adapters_to_test, 2):
        name = item["name"]
        adapter_dir = REPO_ROOT / "results" / "adapters" / item["dir"]

        if not adapter_dir.exists():
            print(f"\n{idx}/5. Skipping {name} ({adapter_dir} not found)")
            continue

        print(f"\n{idx}/5. Evaluating {name} ({item['dir']})...")
        model = load_bfloat16_base_model(args.model_id)

        if item["type"] == "novel":
            load_novel_adapter(model, adapter_dir, velocity_gate=None)
        else:
            model = PeftModel.from_pretrained(model, str(adapter_dir))
        model.eval()

        metrics = evaluate_model_on_questions(model, tokenizer, questions)
        gain_over_base = metrics["avg_adherence_pct"] - base_metrics["avg_adherence_pct"]
        metrics["gain_over_base_pp"] = gain_over_base

        print(
            f"   {name} Adherence: {metrics['avg_adherence_pct']:.2f}% "
            f"(+{gain_over_base:+.2f}pp vs Base)"
        )
        variant_results[name] = metrics

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Comparative decision logic
    stock_lora_score = variant_results.get(
        "Stock LoRA (a256)", variant_results.get("Stock LoRA (standard)", {})
    ).get("avg_adherence_pct", 0.0)

    id_kron_score = variant_results.get("2-Stage id_kron", {}).get(
        "avg_adherence_pct", 0.0
    )

    delta_pp = id_kron_score - stock_lora_score

    if delta_pp >= 3.0:
        outcome = "OUTCOME_1_ID_KRON_WIN"
        board_recommendation = (
            f"id_kron wins by +{delta_pp:.2f}pp (>= +3.0pp). KEEP 🔥 tag."
        )
    elif abs(delta_pp) < 3.0 and id_kron_score >= stock_lora_score - 1.0:
        outcome = "OUTCOME_2_PARITY"
        board_recommendation = (
            f"id_kron and Stock LoRA are at parity (delta: {delta_pp:+.2f}pp). DEMOTE 🔥 to ⭐."
        )
    else:
        outcome = "OUTCOME_3_ID_KRON_LOSS"
        board_recommendation = (
            f"id_kron loses to Stock LoRA by {delta_pp:.2f}pp. DEMOTE/REJECT 🔥 to ❌ (Unearned)."
        )

    print("\n" + "═" * 80)
    print("🎯 HEAD-TO-HEAD BENCHMARK SUMMARY (bfloat16, Fixed Eval Harness)")
    print("─" * 80)
    print(f"  • Base Model (unadapted): {base_metrics['avg_adherence_pct']:.2f}%")
    for name, res in variant_results.items():
        print(
            f"  • {name:<25s}: {res['avg_adherence_pct']:.2f}% (+{res['gain_over_base_pp']:+.2f}pp vs Base)"
        )
    print("─" * 80)
    print(f"  id_kron vs Stock LoRA Delta: {delta_pp:+.2f}pp")
    print(f"  Decision Outcome          : {outcome}")
    print(f"  Board Recommendation      : {board_recommendation}")
    print("═" * 80)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "precision": "bfloat16",
        "harness": "fixed (do_sample=False, stop_strings=['### Question'])",
        "base_model": base_metrics,
        "variants": variant_results,
        "comparison": {
            "stock_lora_a256_score": stock_lora_score,
            "id_kron_score": id_kron_score,
            "delta_pp": delta_pp,
            "outcome": outcome,
            "board_recommendation": board_recommendation,
        },
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved benchmark results to {out_path}")


if __name__ == "__main__":
    main()
