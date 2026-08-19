"""Merge the new command/disposition data into each domain's training corpus,
deduplicating the existing half on the way in.

WHY DEDUP FIRST
---------------
An adapter learns the PROPORTIONS of what it sees -- that is the whole finding of
the corpus audit (DECISIONS.md §47). A corpus that repeats one phrasing 5x has
taught that phrasing 5x, and the duplicate records also consume gradient steps that
teach nothing new:

    duckdb         1622 -> 801 unique   (51% duplicate)
    python_web      301 -> 141 unique   (53%)
    python_modern   301 -> 161 unique   (47%)
    postgresql     1385 -> 936 unique   (32%)
    financial      1628 -> 328 unique   (80%)

Deduplication is on a NORMALISED answer (digits and punctuation stripped), because
template instances differ only in identifiers and numbers while teaching the exact
same shape.

THE ONE RATIO THAT MATTERS
--------------------------
Only astral changes materially: 20.3% -> ~60% command-shaped answers. That is the
intended effect, not a risk to hedge -- 13.7% command-shaped is why the expert wrote
a `pyproject.toml` when asked to add a dependency. The other domains were already
code-shaped, so the merge barely moves them.

    uv run python scripts/corpus/merge_domain_corpora.py           # report
    uv run python scripts/corpus/merge_domain_corpora.py --write   # emit v5 corpora
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re

from gnn_experiment.canon import REPO_ROOT

MARK = "\n\n### Answer:\n"

DOMAINS = ["astral", "postgresql", "duckdb", "python_modern", "python_web", "financial_planning"]
NEW_FILES = ["training_data_commands.jsonl", "training_data_disposition.jsonl"]

# What "the right form" looks like per domain, for the before/after report.
FORM = {
    "astral": r"\b(uv (add|lock|sync|run|init|python|build|tool|export|remove|tree)|uvx|ruff (check|format)|ty check)\b",
    "postgresql": r"```sql", "duckdb": r"```sql",
    "python_modern": r"```python", "python_web": r"```python",
    "financial_planning": r"\*\*Not ",   # financial is prose; the tell is a rejected alternative
}


def answer_of(rec: dict) -> str:
    if "messages" in rec and len(rec["messages"]) >= 2:
        return rec["messages"][1]["content"]
    t = rec.get("text") or ""
    return t.split(MARK, 1)[1] if MARK in t else t


def norm_key(rec: dict) -> str:
    a = re.sub(r"\d+", "0", answer_of(rec).lower())
    a = re.sub(r"[^a-z0-9\s]", " ", a)
    return hashlib.md5(" ".join(a.split()).encode()).hexdigest()


def load(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("=" * 92)
    print(f" {'domain':14s} {'old':>6s} {'dedup':>6s} {'new':>6s} {'v5':>6s}  "
          f"{'form% before':>12s} {'form% after':>12s}")
    print("-" * 92)

    for dom in DOMAINS:
        d = REPO_ROOT / "data" / dom
        old = load(d / "training_data_v4.jsonl") or load(d / "training_data_v3.jsonl")
        if not old:
            print(f" {dom:14s} -- no base corpus found, skipped")
            continue

        seen, kept = set(), []
        for r in old:
            k = norm_key(r)
            if k in seen:
                continue
            seen.add(k)
            kept.append(r)

        new: list[dict] = []
        for f in NEW_FILES:
            new += load(d / f)

        merged = kept + new
        # Shuffle so the domains/forms interleave. Appending new data as a block
        # means it gets the FINAL gradient steps and is over-weighted by the cosine
        # schedule -- an ordering artifact that looks like a real effect.
        random.Random(args.seed).shuffle(merged)

        rx = FORM.get(dom, r"```")
        fb = sum(bool(re.search(rx, answer_of(r), re.I)) for r in old) * 100.0 / len(old)
        fa = sum(bool(re.search(rx, answer_of(r), re.I)) for r in merged) * 100.0 / len(merged)
        print(f" {dom:14s} {len(old):6d} {len(kept):6d} {len(new):6d} {len(merged):6d}  "
              f"{fb:11.1f}% {fa:11.1f}%")

        if args.write:
            out = d / "training_data_v5.jsonl"
            out.write_text("\n".join(json.dumps(r) for r in merged) + "\n")
            print(f" {'':14s} WROTE {out.relative_to(REPO_ROOT)}")

    print("=" * 92)
    if not args.write:
        print(" (dry run -- pass --write)")
    else:
        print(" v5 corpora written. v4 is untouched, so every existing adapter stays")
        print(" reproducible. Nothing is trained: bump CANON.ADAPTER_VERSION and the")
        print(" trainer's DOMAINS map deliberately when you want v5 adapters.")


if __name__ == "__main__":
    main()
