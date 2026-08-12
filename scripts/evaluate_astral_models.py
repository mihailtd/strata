"""CLI Runner for Astral Tooling Evaluation Framework.

Compares Base Qwen3.5-4B vs Fine-Tuned Astral Qwen3.5-4B side-by-side.
Logs output transcripts and adherence metrics to results/eval_runs.jsonl.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from gnn_experiment.eval.eval_suite import run_astral_evaluation  # noqa: E402
from scripts.export_adapter import export_adapter  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--adapter",
        default=str(REPO_ROOT / "results" / "adapters" / "astral_qwen3.5_micro"),
    )
    parser.add_argument("--questions", default=str(REPO_ROOT / "data" / "astral" / "evaluation_data.jsonl"))
    parser.add_argument("--export-if-missing", action="store_true", default=True)
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    config_file = adapter_path / "adapter_config.json"
    if not config_file.exists() and args.export_if_missing:
        print(f"Adapter config not found at {config_file}. Exporting fine-tuned adapter...")
        export_adapter(model_name=args.model_name, out_dir=str(adapter_path), train_steps=150)

    # Run side-by-side evaluation
    results = run_astral_evaluation(
        model_name=args.model_name,
        adapter_path=str(adapter_path),
        questions_file=args.questions,
    )

    out_file = REPO_ROOT / "results" / "astral_eval_summary.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation summary saved to: {out_file}")


if __name__ == "__main__":
    main()
