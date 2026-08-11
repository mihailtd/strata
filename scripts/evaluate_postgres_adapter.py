"""Evaluate the PostgreSQL-expert adapter (see finetune_novel_adapter.py run
against the EPUB-derived dataset) on a PostgreSQL-native-vector-stack-vs-
dedicated-vector-db adherence benchmark, analogous to evaluate_novel_adapter.py's
Astral uv/ruff/ty benchmark but scored against a different term-list pair.

Reuses evaluate_novel_adapter.py's run_base_only/evaluate_novel_variants
unchanged (they were generalized to accept modern_terms/legacy_terms and
adapter/output naming prefixes) -- this script only supplies PostgreSQL-
specific defaults so it stays a thin wrapper, no logic duplicated.

Term lists grounded in the actual source book ("AI-Ready PostgreSQL 18" --
Vibhor Kumar): pgvector/HNSW/IVFFlat/embeddings/cosine/semantic-search all
appear dozens to 1000+ times in the real extracted text (see chunk grep done
before writing these), vs Pinecone/Weaviate/Faiss which the book explicitly
frames as the "specialized, bolt-on" alternative it argues against.

Same two-process pattern as evaluate_novel_adapter.py (a second
from_pretrained() in one process reliably stalls on this machine):

    uv run --env-file .env scripts/evaluate_postgres_adapter.py --base-only
    uv run --env-file .env scripts/evaluate_postgres_adapter.py --variants velocity
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from scripts.evaluate_novel_adapter import evaluate_novel_variants, run_base_only  # noqa: E402

POSTGRES_MODERN_TERMS = [
    r"\bpgvector\b",
    r"\bhnsw\b",
    r"\bivfflat\b",
    r"\bembeddings?\b",
    r"cosine (distance|similarity)",
    r"semantic search",
    r"unified (platform|database)",
    r"vector (column|index|extension)",
]
POSTGRES_LEGACY_TERMS = [
    r"\bpinecone\b",
    r"\bweaviate\b",
    r"\bmilvus\b",
    r"\bqdrant\b",
    r"\bchroma(db)?\b",
    r"\bfaiss\b",
    r"(separate|dedicated|standalone) vector (database|store)",
    r"\belasticsearch\b",
]

QUESTIONS_FILE = "data/postgresql/evaluation_data.jsonl"
EXPERIMENT_NAME = "postgres_expert_evaluation"
ADAPTER_DIR_PREFIX = "postgres_qwen3.5_micro"
OUTPUT_PREFIX = "postgres"
DEFAULT_BASE_CACHE = str(REPO_ROOT / "results" / "base_adherence_cache_postgres.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["custom_standard", "tucker", "velocity", "combined", "random_mask", "lora", "dora", "lokr"],
        help="Variant(s) to evaluate. Omit when using --base-only.",
    )
    parser.add_argument("--base-only", action="store_true", help="Score only the base model and cache the result.")
    parser.add_argument("--base-cache", default=DEFAULT_BASE_CACHE, help="Path to the base-model eval cache JSON.")
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--adapter-root", default=None)
    parser.add_argument("--vram-cap-gb", type=float, default=20.0)
    args = parser.parse_args()

    if args.base_only:
        run_base_only(
            model_name=args.model_name,
            questions_file=QUESTIONS_FILE,
            cache_path=args.base_cache,
            vram_cap_gb=args.vram_cap_gb,
            modern_terms=POSTGRES_MODERN_TERMS,
            legacy_terms=POSTGRES_LEGACY_TERMS,
        )
    else:
        if not args.variants:
            parser.error("--variants is required unless --base-only is set")
        result = evaluate_novel_variants(
            variants=args.variants,
            model_name=args.model_name,
            adapter_root=args.adapter_root,
            questions_file=QUESTIONS_FILE,
            experiment_name=EXPERIMENT_NAME,
            vram_cap_gb=args.vram_cap_gb,
            base_cache_path=args.base_cache,
            modern_terms=POSTGRES_MODERN_TERMS,
            legacy_terms=POSTGRES_LEGACY_TERMS,
            adapter_dir_prefix=ADAPTER_DIR_PREFIX,
            output_prefix=OUTPUT_PREFIX,
        )
        print("\n=== Summary ===")
        print(f"Base: {result['base_avg_adherence_pct']:.2f}%")
        for v, s in result["variants"].items():
            print(f"{v}: {s['finetuned_avg_adherence_pct']:.2f}% (gain {s['adherence_gain_pct']:+.2f}pp)")
