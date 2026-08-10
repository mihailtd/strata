"""Dataset loader for Astral documentation (1,000+ items) and SFT QA pairs.

Parses raw markdown docs from data/astral_docs/raw/ (uv, ruff, ty) and SFT pairs
from data/astral_docs/sft/astral_expert_sft.jsonl into instruction dataset items.
"""

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
from datasets import Dataset

from gnn_experiment.datagen.chunk import chunk_all
from gnn_experiment.datagen.schema import Chunk

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_astral_micro_dataset(
    raw_docs_dir: str | None = None,
    sft_file: str | None = None,
    max_samples: int = 2000,
) -> Dataset:
    """Load raw markdown docs and SFT pairs into a Hugging Face Dataset."""
    raw_path = (
        Path(raw_docs_dir)
        if raw_docs_dir
        else (REPO_ROOT / "data" / "astral_docs" / "raw")
    )
    sft_path = (
        Path(sft_file)
        if sft_file
        else (REPO_ROOT / "data" / "astral_docs" / "sft" / "astral_expert_sft.jsonl")
    )

    records: list[dict[str, Any]] = []

    # 1. Load SFT expert Q&A pairs if present
    if sft_path.exists():
        with open(sft_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "text" in obj:
                        records.append({"text": obj["text"], "source": "sft_jsonl"})
                    elif "messages" in obj:
                        msgs = obj["messages"]
                        user_msg = next(
                            (m["content"] for m in msgs if m["role"] == "user"), ""
                        )
                        assistant_msg = next(
                            (m["content"] for m in msgs if m["role"] == "assistant"), ""
                        )
                        text = (
                            f"### Question:\n{user_msg}\n\n### Answer:\n{assistant_msg}"
                        )
                        records.append({"text": text, "source": "sft_jsonl"})
                except Exception:
                    continue

    # 2. Parse raw doc chunks from uv, ruff, ty
    if raw_path.exists():
        docs_roots = {}
        for tool_dir in raw_path.iterdir():
            if tool_dir.is_dir() and not tool_dir.name.startswith("."):
                tool_name = tool_dir.name
                docs_folder = tool_dir / "docs"
                search_dir = docs_folder if docs_folder.exists() else tool_dir
                docs_roots[tool_name] = search_dir

        if docs_roots:
            try:
                chunks: list[Chunk] = chunk_all(docs_roots)
                for c in chunks:
                    heading_str = (
                        " > ".join(c.heading_path) if c.heading_path else c.source_path
                    )
                    formatted = (
                        f"### Tool: {c.tool}\n"
                        f"### Context ({c.source_path}): {heading_str}\n\n"
                        f"{c.text}"
                    )
                    records.append({"text": formatted, "source": f"raw_doc_{c.tool}"})
            except Exception as e:
                print(f"Warning parsing raw doc chunks: {e}")

    if max_samples and len(records) > max_samples:
        records = records[:max_samples]

    print(f"Loaded {len(records)} dataset items from {raw_path} and {sft_path}")

    # Use pyarrow Table + explicit fingerprint to bypass Python 3.14 dill hashing bug
    texts = [r["text"] for r in records]
    sources = [r["source"] for r in records]
    pa_table = pa.Table.from_pydict({"text": texts, "source": sources})
    return Dataset(pa_table, fingerprint="astral_micro_docs_v1")
