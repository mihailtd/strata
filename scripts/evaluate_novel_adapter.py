"""Evaluate one or more novel-adapter variants (see finetune_novel_adapter.py)
on the same modern-vs-legacy tooling adherence benchmark used for the
original baseline adapter (`gnn_experiment.eval.eval_suite`), so numbers are
directly comparable to the existing MLflow run `eval_Qwen_Qwen3.5-4B`
(finetuned_modern_adherence_pct = 34.50%, base_modern_adherence_pct = 11.99%).

Loads the base model once to score it, then once per variant to score that
variant's adapter (loaded via `gnn_experiment.novel_peft.load_novel_adapter`,
NOT `peft.PeftModel`, since these aren't peft adapters).
"""

import argparse
import json
import sys
from pathlib import Path

import mlflow
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from gnn_experiment.eval.eval_suite import evaluate_single_prompt  # noqa: E402
from gnn_experiment.novel_peft import (  # noqa: E402
    VelocityGate,
    get_decoder_layers,
    load_novel_adapter,
    set_hard_vram_cap,
)


def _load_base_model(model_name: str):
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
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )


def evaluate_novel_variants(
    variants: list[str],
    model_name: str = "Qwen/Qwen3.5-4B",
    adapter_root: str | None = None,
    questions_file: str = "configs/eval_questions.yaml",
    experiment_name: str = "astral_tooling_evaluation",
    out_summary_dir: str | None = None,
    vram_cap_gb: float = 20.0,
) -> dict:
    set_hard_vram_cap(
        vram_cap_gb
    )  # see set_hard_vram_cap docstring for why this is mandatory here

    adapter_root_path = (
        Path(adapter_root) if adapter_root else REPO_ROOT / "results" / "adapters"
    )
    out_summary_dir_path = (
        Path(out_summary_dir) if out_summary_dir else REPO_ROOT / "results"
    )

    questions_path = Path(questions_file)
    if not questions_path.is_absolute():
        questions_path = REPO_ROOT / questions_path
    with open(questions_path) as f:
        questions = yaml.safe_load(f).get("questions", [])

    mlflow_db = REPO_ROOT / "mlruns.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment(experiment_name)

    print(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        "\n================== Evaluating BASE model (shared across all variants) =================="
    )
    base_model = _load_base_model(model_name)
    base_model.eval()
    base_results = []
    for idx, q in enumerate(questions, 1):
        print(f"[{idx}/{len(questions)}] base: '{q['prompt'][:60]}...'")
        res = evaluate_single_prompt(base_model, tokenizer, q["prompt"])
        res["id"] = q["id"]
        res["category"] = q["category"]
        base_results.append(res)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    base_avg_adherence = sum(r["adherence_pct"] for r in base_results) / len(
        base_results
    )
    base_modern_total = sum(r["modern_hits"] for r in base_results)
    base_legacy_total = sum(r["legacy_hits"] for r in base_results)
    print(f"Base model modern adherence: {base_avg_adherence:.2f}%")

    all_summaries = {}
    for variant in variants:
        adapter_dir = adapter_root_path / f"astral_qwen3.5_micro_{variant}"
        print(
            f"\n================== Evaluating variant '{variant}' ({adapter_dir}) =================="
        )
        if not (adapter_dir / "novel_adapter_config.json").exists():
            print(f"  SKIP: no adapter found at {adapter_dir}")
            continue

        model = _load_base_model(model_name)
        meta = json.loads((adapter_dir / "novel_adapter_config.json").read_text())
        gate = None
        if meta.get("gated"):
            # eval is a pure forward pass (no training step ever calls end_step()),
            # so nothing is ever marked quiet -- every adapter fires, matching how
            # a served/deployed version of this adapter would behave post-training.
            gate = VelocityGate(num_layers=len(get_decoder_layers(model)))
        load_novel_adapter(model, adapter_dir, velocity_gate=gate)
        model.eval()

        ft_results = []
        for idx, q in enumerate(questions, 1):
            print(f"[{idx}/{len(questions)}] {variant}: '{q['prompt'][:60]}...'")
            res = evaluate_single_prompt(model, tokenizer, q["prompt"])
            res["id"] = q["id"]
            res["category"] = q["category"]
            ft_results.append(res)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ft_avg_adherence = sum(r["adherence_pct"] for r in ft_results) / len(ft_results)
        adherence_gain = ft_avg_adherence - base_avg_adherence
        ft_modern_total = sum(r["modern_hits"] for r in ft_results)
        ft_legacy_total = sum(r["legacy_hits"] for r in ft_results)

        with mlflow.start_run(
            run_name=f"eval_novel_{variant}_{model_name.replace('/', '_')}"
        ):
            mlflow.set_tags({"variant": variant, "novel_architecture": "true"})
            mlflow.log_params(
                {
                    "exact_model_name": model_name,
                    "adapter_path": str(adapter_dir),
                    "num_questions": len(questions),
                    "variant": variant,
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
            comparison_table = [
                {
                    "id": b["id"],
                    "prompt": b["prompt"],
                    "base_response": b["response"],
                    "base_adherence_pct": b["adherence_pct"],
                    "finetuned_response": f["response"],
                    "finetuned_adherence_pct": f["adherence_pct"],
                }
                for b, f in zip(base_results, ft_results, strict=True)
            ]
            summary = {
                "variant": variant,
                "exact_model_name": model_name,
                "adapter_path": str(adapter_dir),
                "base_avg_adherence_pct": base_avg_adherence,
                "finetuned_avg_adherence_pct": ft_avg_adherence,
                "adherence_gain_pct": adherence_gain,
                "comparison": comparison_table,
            }
            mlflow.log_dict(summary, "evaluation_comparison.json")

        out_summary_dir_path.mkdir(parents=True, exist_ok=True)
        with open(
            out_summary_dir_path / f"astral_eval_summary_{variant}.json", "w"
        ) as f:
            json.dump(summary, f, indent=2)

        print(
            f"[{variant}] modern adherence: {ft_avg_adherence:.2f}% "
            f"(base {base_avg_adherence:.2f}%, gain {adherence_gain:+.2f}pp)"
        )
        all_summaries[variant] = summary

    return {"base_avg_adherence_pct": base_avg_adherence, "variants": all_summaries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        required=True,
        choices=["custom_standard", "tucker", "velocity", "combined"],
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter-root", default=None)
    parser.add_argument("--questions", default="configs/eval_questions.yaml")
    parser.add_argument("--vram-cap-gb", type=float, default=20.0)
    args = parser.parse_args()

    result = evaluate_novel_variants(
        variants=args.variants,
        model_name=args.model_name,
        adapter_root=args.adapter_root,
        questions_file=args.questions,
        vram_cap_gb=args.vram_cap_gb,
    )
    print("\n=== Summary ===")
    print(f"Base: {result['base_avg_adherence_pct']:.2f}%")
    for v, s in result["variants"].items():
        print(
            f"{v}: {s['finetuned_avg_adherence_pct']:.2f}% (gain {s['adherence_gain_pct']:+.2f}pp)"
        )
