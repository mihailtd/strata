"""Benchmark LoRA and other PEFT techniques on a single GPU.

    uv run --env-file .env scripts/run_benchmark.py
    uv run --env-file .env scripts/run_benchmark.py --methods lora,dora,qlora --max-steps 50
    uv run --env-file .env scripts/run_benchmark.py --model Qwen/Qwen3.5-2B-Instruct

--env-file .env sets LD_PRELOAD (required for torch to see the GPU on WSL,
see scripts/check_gpu.py) and HF_TOKEN.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from gnn_experiment.bench import run_one

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "benchmark.yaml"))
    p.add_argument("--model")
    p.add_argument("--dataset")
    p.add_argument("--methods", help="comma-separated, e.g. lora,dora,qlora")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--r", type=int)
    p.add_argument(
        "--out", default=str(REPO_ROOT / "results" / "benchmark_results.csv")
    )
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.model:
        cfg["model"] = args.model
    if args.dataset:
        cfg["dataset"] = args.dataset
    if args.methods:
        cfg["methods"] = args.methods.split(",")
    if args.max_steps:
        cfg["max_steps"] = args.max_steps
    if args.r:
        cfg["r"] = args.r

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for method in cfg["methods"]:
        print(f"\n=== {method} ===")
        try:
            row = run_one(
                model_name=cfg["model"],
                method=method,
                dataset_name=cfg["dataset"],
                text_field=cfg["text_field"],
                r=cfg["r"],
                alpha=cfg["alpha"],
                target_modules=cfg["target_modules"],
                max_steps=cfg["max_steps"],
                batch_size=cfg["batch_size"],
                max_length=cfg["max_length"],
                learning_rate=cfg["learning_rate"],
                n_examples=cfg["n_examples"],
                output_dir=str(REPO_ROOT / "results" / "tmp" / method),
            )
        except Exception as e:
            print(f"FAILED: {method}: {e}")
            row = {"method": method, "error": str(e)}
        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
