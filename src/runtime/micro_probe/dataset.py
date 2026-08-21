"""Dataset loader for doc-derived SFT training data: raw markdown/EPUB chunks
plus generated Q&A pairs, turned into a Hugging Face Dataset.

`load_astral_micro_dataset` (the original, Astral uv/ruff/ty-specific entry
point) is now a thin wrapper around the general `load_micro_dataset` -- use
that one directly for a different corpus (e.g. an EPUB-derived dataset from
`scripts/old/run_datagen.py --epub ...`), same loading/formatting logic either way.

Two output shapes, selected by `conversational`:
- `conversational=False` (default): a `{"text", "source"}` dataset. SFT Q&A
  pairs are flattened to `### Question / ### Answer` text, and raw doc chunks
  (no role structure) are mixed in as plain context text. Use with a plain
  `transformers.Trainer` + `DataCollatorForLanguageModeling`.
- `conversational=True`: a `{"messages"}` dataset (raw role/content turns,
  persona-sanitized, no chat-template rendering applied here). Use with
  `trl.SFTTrainer` + `SFTConfig(assistant_only_loss=True)` -- TRL applies the
  chat template itself internally (with `{% generation %}` markers) to track
  assistant token spans for loss masking, which requires genuinely
  conversational input, not a pre-rendered flat string (confirmed against
  trl==1.9.2's SFTTrainer source: `assistant_only_loss=True` raises unless
  `is_conversational(dataset_sample)`). Raw doc chunks have no message
  structure, so they're excluded in this mode rather than faked -- SFT Q&A
  pairs are the real supervised signal; raw chunks were always a lower-value
  "vocabulary priming" addition, not needed for genuine instruction tuning.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
from datasets import Dataset

from runtime.datagen.chunk import chunk_all
from runtime.datagen.schema import Chunk

# ---------------------------------------------------------------------------
# Persona sanitizer
# ---------------------------------------------------------------------------

# Informal openers that should never appear in the adapter's assistant output.
_INFORMAL_OPENERS: tuple[str, ...] = (
    r"^Hey!\s*",
    r"^Hi!\s*",
    r"^Sure!\s*",
    r"^Sure,\s*",
    r"^Great question!\s*",
    r"^Great!\s*",
    r"^Absolutely!\s*",
    r"^Of course!\s*",
    r"^Certainly!\s*",
    r"^No problem!\s*",
)
_OPENER_RE = re.compile("|".join(_INFORMAL_OPENERS), re.IGNORECASE)

# Informal closers (trailing sign-off phrases).
_INFORMAL_CLOSERS: tuple[str, ...] = (
    r"\s*Hope that helps!\s*$",
    r"\s*Let me know if you have any questions!\s*$",
    r"\s*Feel free to ask if you need more details!\s*$",
    r"\s*Happy to help!\s*$",
)
_CLOSER_RE = re.compile("|".join(_INFORMAL_CLOSERS), re.IGNORECASE)


def sanitize_assistant_content(text: str) -> str:
    """Strip informal greeting/sign-off filler from assistant response text.

    This prevents style-oscillation during fine-tuning: the adapter should
    learn to produce direct, actionable technical answers immediately, not
    conversational preamble that varies sample-to-sample.

    Loops until stable so chained openers like 'Hi! Great question! ...'
    are fully stripped in one call.
    """
    # Strip openers iteratively until no more match (handles chained filler).
    while True:
        stripped = _OPENER_RE.sub("", text).lstrip()
        if stripped == text:
            break
        text = stripped
    text = _CLOSER_RE.sub("", text).rstrip()
    return text


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def load_micro_dataset(
    raw_docs_dir: str | Path | None = None,
    sft_file: str | Path | None = None,
    max_samples: int = 2000,
    fingerprint: str | None = None,
    conversational: bool = False,
) -> Dataset:
    """Load raw markdown/EPUB-derived doc chunks + generated SFT Q&A pairs
    into a Hugging Face Dataset. Either source may be absent -- an EPUB-only
    dataset (see scripts/old/run_datagen.py --epub) has no `raw_docs_dir` (there's
    no directory of markdown files to walk, only the SFT jsonl the pipeline
    wrote), so pass sft_file alone.

    `fingerprint` defaults to a hash of the resolved paths so two different
    datasets never collide on `datasets`' caching -- pass one explicitly only
    to reproduce a prior exact value (e.g. astral's original hardcoded one).

    `conversational` selects the output shape -- see module docstring.
    """
    raw_path = Path(raw_docs_dir) if raw_docs_dir else None
    sft_path = Path(sft_file) if sft_file else None
    if fingerprint is None:
        key = f"{raw_path}|{sft_path}|conversational={conversational}"
        fingerprint = hashlib.sha256(key.encode()).hexdigest()[:16]

    records: list[dict[str, Any]] = []

    # 1. Load SFT expert Q&A pairs if present
    if sft_path and sft_path.exists():
        with open(sft_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "messages" in obj:
                        msgs = obj["messages"]
                        # Apply persona sanitizer to assistant content.
                        cleaned_msgs = [
                            {
                                "role": m["role"],
                                "content": (
                                    sanitize_assistant_content(m["content"])
                                    if m["role"] == "assistant"
                                    else m["content"]
                                ),
                            }
                            for m in msgs
                        ]
                        if conversational:
                            records.append({"messages": cleaned_msgs, "source": "sft_jsonl"})
                        else:
                            user_msg = next((m["content"] for m in cleaned_msgs if m["role"] == "user"), "")
                            assistant_msg = next((m["content"] for m in cleaned_msgs if m["role"] == "assistant"), "")
                            text = f"### Question:\n{user_msg}\n\n### Answer:\n{assistant_msg}"
                            records.append({"text": text, "source": "sft_jsonl"})
                    elif "text" in obj and not conversational:
                        # Legacy records with only a pre-rendered text field
                        # (no messages list) — use as-is. Conversational mode
                        # has nothing to build `messages` from, so these are
                        # skipped there rather than faked.
                        records.append({"text": obj["text"], "source": "sft_jsonl"})
                except Exception:
                    continue

    # 2. Parse raw doc chunks, if a raw docs directory was given (git-clone-style
    # layout: raw_path/<tool_name>/docs/**/*.md -- an EPUB-derived dataset has no
    # such directory, only sft_file, so this whole branch is naturally skipped).
    # Raw chunks have no user/assistant structure, so they're only included in
    # flat-text mode -- see module docstring for why conversational mode skips them.
    if raw_path and raw_path.exists() and not conversational:
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
                    heading_str = " > ".join(c.heading_path) if c.heading_path else c.source_path
                    formatted = f"### Tool: {c.tool}\n### Context ({c.source_path}): {heading_str}\n\n{c.text}"
                    records.append({"text": formatted, "source": f"raw_doc_{c.tool}"})
            except Exception as e:
                print(f"Warning parsing raw doc chunks: {e}")

    if max_samples and len(records) > max_samples:
        records = records[:max_samples]

    print(f"Loaded {len(records)} dataset items from {raw_path} and {sft_path}")

    # Use pyarrow Table + explicit fingerprint to bypass Python 3.14 dill hashing bug
    sources = [r["source"] for r in records]
    if conversational:
        messages = [r["messages"] for r in records]
        pa_table = pa.Table.from_pydict({"messages": messages, "source": sources})
    else:
        texts = [r["text"] for r in records]
        pa_table = pa.Table.from_pydict({"text": texts, "source": sources})
    return Dataset(pa_table, fingerprint=fingerprint)


def load_astral_micro_dataset(
    raw_docs_dir: str | None = None,
    sft_file: str | None = None,
    max_samples: int = 2000,
    conversational: bool = False,
) -> Dataset:
    """Original Astral uv/ruff/ty entry point -- thin wrapper around
    load_micro_dataset for backward compatibility.

    Fingerprint varies by `conversational` since the two modes produce
    different columns (`text` vs `messages`) -- see load_micro_dataset.
    """
    raw_path = raw_docs_dir or str(REPO_ROOT / "data" / "astral" / "raw")
    sft_path = sft_file or str(REPO_ROOT / "data" / "astral" / "training_data.jsonl")
    fingerprint = "astral_micro_docs_v3_conversational" if conversational else "astral_micro_docs_v1"
    return load_micro_dataset(raw_path, sft_path, max_samples, fingerprint=fingerprint, conversational=conversational)
