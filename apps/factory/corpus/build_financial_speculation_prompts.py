"""Builds a 40-prompt financial_planning set for the speculation matrix benchmark.

WHY THIS EXISTS
---------------
The speculation matrix ran on unequal prompt sets: astral 40, postgresql 40,
financial_planning 20. The only production routing change it produced -- disabling
speculation for financial_planning -- came from the half-powered column, on a
measurement 2% away from its threshold. Equal n is a precondition for that call.

WHY NOT JUST EXTEND evaluation_data.jsonl
-----------------------------------------
That file is read by ~18 scripts, and its `expects` field drives keyword accuracy
scoring in `benchmark_m1_vs_m2_regime.py` and `benchmark_alpha_absorption_sweep.py`
-- the benchmarks behind the published m1/m2 and alpha-sweep numbers. Appending
records without `expects` would silently change how those score. So the extra
prompts live in a separate file that ONLY the speculation benchmark reads.

WHERE THE EXTRA 20 COME FROM
----------------------------
`stage_logs/question_gen.jsonl` holds 490 questions the datagen pipeline produced
from the source book. 304 became training data. The rest were generated and never
used. Those are genuinely held out -- same generator, same source, but the
financial adapter has never seen them.

They are filtered to stay comparable to the 20 curated prompts:
  - dropped if they reference the source ("this book", "the excerpt", ...), which
    is a generation artifact that makes a question non-self-contained
  - dropped if they ask about book metadata (author credentials, liability,
    copyright) rather than domain content
  - dropped if they come from front-matter chunks (cover, foreword, preface)
  - length-gated to match the curated prompts' register

Provenance is recorded per row (`source`: curated | heldout_generated) so the
benchmark can report tau on each half separately. If the two halves disagree,
the financial result is a prompt-set artifact rather than a domain property --
and that is exactly the thing the original n=20 run could not check.

USAGE:
    uv run python scripts/corpus/build_financial_speculation_prompts.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from runtime_common.canon import REPO_ROOT  # noqa: E402

# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
DOMAIN_DIR = REPO_ROOT / "apps" / "factory" / "data" / "financial_planning"
EVAL_DIR = REPO_ROOT / "data" / "financial_planning"  # stayed at root -- benchmarks/ consumes these
CURATED = EVAL_DIR / "evaluation_data.jsonl"
QUESTION_LOG = DOMAIN_DIR / "stage_logs" / "question_gen.jsonl"
TRAINING = DOMAIN_DIR / "training_data.jsonl"
OUT = EVAL_DIR / "speculation_prompts.jsonl"  # benchmark input, not training data

TARGET_N = 40
SEED = 42

# Questions that lean on the source text instead of standing alone.
SOURCE_REFERENCING = re.compile(
    r"\b(this book|the book|the excerpt|the document|the text|the chapter|"
    r"the author|the passage|the provided|discussed in|mentioned in|described in|"
    r"according to|in the reading)\b",
    re.I,
)

# Questions about the artifact rather than the domain.
BOOK_METADATA = re.compile(
    r"\b(klontz|isbn|copyright|publisher|liability|warrant|credential|"
    r"acknowledgement|foreword|preface|table of contents|about the author)\b",
    re.I,
)

# Front-matter chunks: cover, forewords (f0*), part dividers (p0*).
FRONT_MATTER = re.compile(r"/(000_cover|00\d_f\d+|0\d\d_p\d+)$", re.I)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def extract_questions(log_rows: list[dict]) -> list[tuple[str, str]]:
    """Pulls individual questions out of each generation response."""
    out = []
    for row in log_rows:
        if FRONT_MATTER.search(row.get("chunk_source", "")):
            continue
        for line in row.get("response_text", "").splitlines():
            q = re.sub(r"^[-*\d.)\s]+", "", line.strip()).strip()
            if q.endswith("?") and len(q) > 30:
                out.append((q, row["chunk_source"]))
    return out


def main() -> None:
    curated = load_jsonl(CURATED)
    print(f"Curated prompts:            {len(curated)}")

    training = load_jsonl(TRAINING)
    trained_on = {m["content"].strip() for r in training for m in r["messages"] if m.get("role") == "user"}
    curated_prompts = {r["prompt"].strip() for r in curated}
    print(f"Questions used in training: {len(trained_on)}")

    candidates = extract_questions(load_jsonl(QUESTION_LOG))
    print(f"Questions parsed from generator (excl. front matter): {len(candidates)}")

    seen: set[str] = set()
    pool: list[tuple[str, str]] = []
    for q, src in candidates:
        if q in trained_on or q in curated_prompts or q in seen:
            continue
        if SOURCE_REFERENCING.search(q) or BOOK_METADATA.search(q):
            continue
        # Match the curated prompts' register: substantive, single-sentence asks.
        if not (60 <= len(q) <= 240):
            continue
        seen.add(q)
        pool.append((q, src))

    print(f"Held-out, self-contained, domain-content candidates:   {len(pool)}")

    need = TARGET_N - len(curated)
    if len(pool) < need:
        raise SystemExit(f"Only {len(pool)} candidates for {need} slots — cannot level to {TARGET_N}.")

    # Spread the picks across source chunks so one section cannot dominate.
    by_chunk: dict[str, list[str]] = {}
    for q, src in pool:
        by_chunk.setdefault(src, []).append(q)

    rng = random.Random(SEED)
    for qs in by_chunk.values():
        rng.shuffle(qs)

    chunks = sorted(by_chunk)
    rng.shuffle(chunks)

    picked: list[tuple[str, str]] = []
    round_idx = 0
    while len(picked) < need:
        progressed = False
        for c in chunks:
            if len(picked) >= need:
                break
            if round_idx < len(by_chunk[c]):
                picked.append((by_chunk[c][round_idx], c))
                progressed = True
        if not progressed:
            break
        round_idx += 1

    print(f"Selected {len(picked)} held-out prompts across {len({c for _, c in picked})} source chunks")

    rows = []
    for r in curated:
        rows.append(
            {
                "id": r["id"],
                "category": r.get("category", "curated"),
                "prompt": r["prompt"],
                "source": "curated",
            }
        )
    for i, (q, src) in enumerate(picked, start=len(curated) + 1):
        rows.append(
            {
                "id": f"fin_{i:02d}",
                "category": "heldout_generated",
                "prompt": q,
                "source": "heldout_generated",
                "chunk_source": src,
            }
        )

    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"\nWrote {len(rows)} prompts -> {OUT.relative_to(REPO_ROOT)}")
    print(f"  curated:           {sum(1 for r in rows if r['source'] == 'curated')}")
    print(f"  heldout_generated: {sum(1 for r in rows if r['source'] == 'heldout_generated')}")
    print("\nSample of the held-out additions:")
    for r in rows[len(curated) : len(curated) + 5]:
        print(f"  {r['id']}: {r['prompt'][:105]}")


if __name__ == "__main__":
    main()
