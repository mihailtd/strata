"""Generate an SFT dataset from Astral's docs (uv, ruff, ty) using the local
llama-server model.

This script only talks to llama-server over plain HTTP and does CPU-side
text processing — unlike run_benchmark.py/chat.py, it never touches
torch/GPU directly, so no --env-file .env / LD_PRELOAD is needed. It does
need llama-server already running (see serving/README.md).

    uv run scripts/run_datagen.py --dry-run
    uv run scripts/run_datagen.py --limit 5
    uv run scripts/run_datagen.py --repos uv,ruff --refresh-docs
    uv run scripts/run_datagen.py

--dry-run samples a small number of chunks (10 by default, or --limit's
value if also given) through the real pipeline, then prints how many chunks
exist, how long each took, and an estimated time for the full run — use
this to check whether a given model is fast enough before committing.

By default, the pipeline stops the llama-server it talked to when it
finishes (frees the GPU automatically) — pass --no-unload to leave it
running, e.g. if you want to chat with it right after.

Inspect results with: mlflow ui --backend-store-uri sqlite:///mlruns.db
"""

import argparse
from pathlib import Path

import yaml

from gnn_experiment.datagen.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(REPO_ROOT / "configs" / "datagen.yaml"))
    p.add_argument("--repos", help="comma-separated subset, e.g. uv,ruff")
    p.add_argument("--model", help="override llm.model, e.g. models/Qwen3.5-4B-Q8_0.gguf")
    p.add_argument("--limit", type=int, help="process only first N chunks (smoke test / dry run sample size)")
    p.add_argument("--dry-run", action="store_true", help="estimate timing only; defaults --limit to 10 if not set")
    p.add_argument("--refresh-docs", action="store_true", help="force re-clone doc repos")
    p.add_argument("--out", help="override output_dir")
    p.add_argument("--nudge", help="extra instruction injected into generation prompts, logged to MLflow")
    p.add_argument("--no-unload", action="store_true", help="leave the llama-server running after the pipeline finishes")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.repos:
        config["repos"] = args.repos.split(",")
    if args.out:
        config["output_dir"] = args.out
    if args.model:
        config["llm"]["model"] = args.model
    if args.nudge:
        config["extraction_nudge"] = args.nudge
    if args.no_unload:
        config["llm"]["unload_after_run"] = False

    limit = args.limit
    if args.dry_run and limit is None:
        limit = 10

    output_path = run_pipeline(
        config, limit=limit, refresh_docs=args.refresh_docs, dry_run=args.dry_run
    )
    print(f"\nDone. Output: {output_path}")
    print("Inspect with: mlflow ui --backend-store-uri sqlite:///mlruns.db")


if __name__ == "__main__":
    main()
