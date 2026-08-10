"""Evaluate one or more novel-adapter variants (see finetune_novel_adapter.py)
on the same modern-vs-legacy tooling adherence benchmark used for the
original baseline adapter (`gnn_experiment.eval.eval_suite`), so numbers are
directly comparable to the existing MLflow run `eval_Qwen_Qwen3.5-4B`
(finetuned_modern_adherence_pct = 34.50%, base_modern_adherence_pct = 11.99%).

Loads the base model once to score it, then once per variant to score that
variant's adapter (loaded via `gnn_experiment.novel_peft.load_novel_adapter`,
NOT `peft.PeftModel`, since these aren't peft adapters).

A second `AutoModelForCausalLM.from_pretrained(...)` call in the same process
reproducibly stalls for 1-2+ minutes partway through weight loading on this
machine (observed twice, always around shard ~403/426) -- not a fragile
timeout issue, a real, repeatable slowdown. So: `--base-only` runs exactly one
model load (the base model), scores it, and caches the result to
`results/base_adherence_cache.json`; normal variant runs then load that cache
instead of re-scoring the base model live, keeping every process down to a
single weight load. Run `--base-only` once, then one `--variants <name>`
invocation per adapter (each its own process).
"""

import argparse
import json
import sys
from pathlib import Path

import mlflow
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Piped stdout (e.g. through `tail`, or a background-task capture) is fully
# block-buffered by default, so progress prints can sit invisible for minutes
# and, worse, vanish entirely if the process is killed (e.g. by `timeout`)
# before a normal exit flush. Line-buffer explicitly so progress is visible
# and survives a hard kill -- this bit us once already on this script.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)  # ty: ignore[call-non-callable]  -- real TextIOWrapper method, just missing from the TextIO stub

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from gnn_experiment.eval.eval_suite import evaluate_single_prompt  # noqa: E402
from gnn_experiment.novel_peft import (  # noqa: E402
    load_novel_adapter,
    set_hard_vram_cap,
)


def _load_base_model(model_name: str):
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
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


def _load_questions(questions_file: str) -> list:
    questions_path = Path(questions_file)
    if not questions_path.is_absolute():
        questions_path = REPO_ROOT / questions_path
    with open(questions_path) as f:
        return yaml.safe_load(f).get("questions", [])


def run_base_only(
    model_name: str = "Qwen/Qwen3.5-4B",
    questions_file: str = "configs/eval_questions.yaml",
    cache_path: str | None = None,
    vram_cap_gb: float = 20.0,
) -> dict:
    """Exactly one weight load: score the base model and cache the result so
    later variant runs never need a second from_pretrained() in-process."""
    set_hard_vram_cap(vram_cap_gb)
    questions = _load_questions(questions_file)

    print(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    assert tokenizer is not None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n================== Evaluating BASE model ==================")
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

    base_avg_adherence = sum(r["adherence_pct"] for r in base_results) / len(base_results)
    base_modern_total = sum(r["modern_hits"] for r in base_results)
    base_legacy_total = sum(r["legacy_hits"] for r in base_results)
    print(f"Base model modern adherence: {base_avg_adherence:.2f}%")

    cache = {
        "model_name": model_name,
        "base_avg_adherence_pct": base_avg_adherence,
        "base_modern_hits_total": base_modern_total,
        "base_legacy_hits_total": base_legacy_total,
        "base_results": base_results,
    }
    cache_path = cache_path or str(REPO_ROOT / "results" / "base_adherence_cache.json")
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)
    print(f"Cached base-model eval to {cache_path}")
    return cache


def evaluate_novel_variants(
    variants: list[str],
    model_name: str = "Qwen/Qwen3.5-4B",
    adapter_root: str | None = None,
    questions_file: str = "configs/eval_questions.yaml",
    experiment_name: str = "astral_tooling_evaluation",
    out_summary_dir: str | None = None,
    vram_cap_gb: float = 20.0,
    base_cache_path: str | None = None,
) -> dict:
    set_hard_vram_cap(vram_cap_gb)  # see set_hard_vram_cap docstring for why this is mandatory here

    adapter_root_path = Path(adapter_root) if adapter_root else REPO_ROOT / "results" / "adapters"
    out_summary_dir_path = Path(out_summary_dir) if out_summary_dir else REPO_ROOT / "results"
    base_cache_path = base_cache_path or str(REPO_ROOT / "results" / "base_adherence_cache.json")

    questions = _load_questions(questions_file)

    mlflow_db = REPO_ROOT / "mlruns.db"
    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db}")
    mlflow.set_experiment(experiment_name)

    if not Path(base_cache_path).exists():
        raise FileNotFoundError(
            f"No base-model eval cache at {base_cache_path}. Run this script once with --base-only first "
            "(a second from_pretrained() call in the same process reliably stalls on this machine -- see "
            "module docstring)."
        )
    with open(base_cache_path) as f:
        base_cache = json.load(f)
    if base_cache["model_name"] != model_name:
        raise ValueError(
            f"Base cache was computed for {base_cache['model_name']!r}, not {model_name!r}. Rerun --base-only."
        )
    base_results = base_cache["base_results"]
    base_avg_adherence = base_cache["base_avg_adherence_pct"]
    base_modern_total = base_cache["base_modern_hits_total"]
    base_legacy_total = base_cache["base_legacy_hits_total"]
    print(f"Loaded cached base-model adherence: {base_avg_adherence:.2f}% (from {base_cache_path})")

    print(f"Loading tokenizer for {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    assert tokenizer is not None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_summaries = {}
    for variant in variants:
        adapter_dir = adapter_root_path / f"astral_qwen3.5_micro_{variant}"
        print(f"\n================== Evaluating variant '{variant}' ({adapter_dir}) ==================")
        if not (adapter_dir / "novel_adapter_config.json").exists():
            print(f"  SKIP: no adapter found at {adapter_dir}")
            continue

        model = _load_base_model(model_name)
        # Deliberately never construct/pass a VelocityGate here, even for gated
        # variants: NovelLoraLinear.forward treats "no gate" and "a gate whose
        # quiet_layers is permanently empty" identically (both always active --
        # see its `is_quiet_fn or (lambda idx: False)` default), and since eval
        # never calls end_step(), a real gate's quiet_layers WOULD stay empty
        # forever anyway. The only actual effect of constructing one here would
        # be installing its per-layer forward hooks, which force a host sync
        # every layer of every decode step -- for 256-token generation across
        # 32 layers that's ~8k pointless GPU stalls per question, observed
        # firsthand to blow past a 300s budget without finishing one question.
        load_novel_adapter(model, adapter_dir, velocity_gate=None)
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

        with mlflow.start_run(run_name=f"eval_novel_{variant}_{model_name.replace('/', '_')}"):
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
        with open(out_summary_dir_path / f"astral_eval_summary_{variant}.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(
            f"[{variant}] modern adherence: {ft_avg_adherence:.2f}% "
            f"(base {base_avg_adherence:.2f}%, gain {adherence_gain:+.2f}pp)"
        )
        all_summaries[variant] = summary

    return {"base_avg_adherence_pct": base_avg_adherence, "variants": all_summaries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["custom_standard", "tucker", "velocity", "combined"],
        help="Variant(s) to evaluate. Omit when using --base-only.",
    )
    parser.add_argument("--base-only", action="store_true", help="Score only the base model and cache the result.")
    parser.add_argument("--base-cache", default=None, help="Path to the base-model eval cache JSON.")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter-root", default=None)
    parser.add_argument("--questions", default="configs/eval_questions.yaml")
    parser.add_argument("--vram-cap-gb", type=float, default=20.0)
    args = parser.parse_args()

    if args.base_only:
        run_base_only(
            model_name=args.model_name,
            questions_file=args.questions,
            cache_path=args.base_cache,
            vram_cap_gb=args.vram_cap_gb,
        )
    else:
        if not args.variants:
            parser.error("--variants is required unless --base-only is set")
        result = evaluate_novel_variants(
            variants=args.variants,
            model_name=args.model_name,
            adapter_root=args.adapter_root,
            questions_file=args.questions,
            vram_cap_gb=args.vram_cap_gb,
            base_cache_path=args.base_cache,
        )
        print("\n=== Summary ===")
        print(f"Base: {result['base_avg_adherence_pct']:.2f}%")
        for v, s in result["variants"].items():
            print(f"{v}: {s['finetuned_avg_adherence_pct']:.2f}% (gain {s['adherence_gain_pct']:+.2f}pp)")
