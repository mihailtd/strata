"""Split the astral corpus into three domains, because it was never one domain.

WHAT THE AUDIT FOUND
--------------------
`data/astral/training_data_v4.jsonl` is 1433 records made of two unrelated halves:

    745  doc-scraped from Astral's own docs (uv 422, ty 167, ruff 139, +17)
         -> 22.0% of answers contain a runnable uv/ruff command
    688  hand-written generators in build_astral_corpus.py
         ->  0.0% of answers contain a runnable uv/ruff command

Of those 688, roughly 88% are generic Python / FastAPI exercises -- `func_lru_cache`,
`fastapi_crud_router`, `py_match_case`, `asyncpg_pool` -- which have nothing to do
with Astral tooling. Only `ruff_idioms` and `pep723_script` do.

That is why the "astral expert", asked to add a dependency with uv, wrote a
`pyproject.toml` and an essay instead of `uv add`: 90% of its training answers are
prose or Python code, and the half we control emits ZERO uv commands.

NOTE ON THE EPUBS
-----------------
The three books in data/astral/ (FastAPI, Functional Python, Modern Python Cookbook)
are INERT for v4 -- no record carries book provenance and no builder reads them. The
generic content is templated, not extracted. Moving the books is housekeeping; the
split below is the actual fix.

THE SPLIT
---------
    astral        Astral tooling only: uv / ruff / ty docs + ruff_idioms + pep723
    python_web    FastAPI / MCP / asyncpg service-building
    python_modern functional & modern-Python idioms

    uv run python scripts/corpus/split_astral_domain.py           # report
    uv run python scripts/corpus/split_astral_domain.py --write   # do it
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter

from gnn_experiment.canon import REPO_ROOT

SRC = "data/astral/training_data_v4.jsonl"

# Families that leave astral. Everything NOT listed here -- including every
# doc-scraped record, which has no family at all -- stays.
WEB = {"fastapi_route", "fastapi_crud_router", "fastapi_query_filters",
       "fastapi_background_tasks", "fastapi_dependency_injection",
       "fastmcp_tool", "asyncpg_pool"}
MODERN = {"func_lru_cache", "func_reduce_compose", "func_immutable_dataclass",
          "func_itertools_pipeline", "py_match_case", "py_contextmanager",
          "modern_typing"}
# pep723_script stays with astral: PEP 723 inline metadata is a `uv run` feature.

BOOKS = {
    "Building Python Web APIs With F - Abdulazeez Abdulazeez Adeshina.epub": "python_web",
    "Functional Python Programming_ - Steven F. Lott.epub": "python_modern",
    "Modern Python Cookbook - Steven F. Lott.epub": "python_modern",
}

CMD = re.compile(r"\b(uv (add|lock|sync|run|init|venv|python|build|tool|export)"
                 r"|uvx|ruff (check|format))\b", re.I)
MARK = "\n\n### Answer:\n"


def answer_of(rec: dict) -> str:
    if "messages" in rec and len(rec["messages"]) >= 2:
        return rec["messages"][1]["content"]
    t = rec.get("text") or ""
    return t.split(MARK, 1)[1] if MARK in t else ""


def bucket(rec: dict) -> str:
    fam = str((rec.get("meta") or {}).get("family") or "")
    if fam in WEB:
        return "python_web"
    if fam in MODERN:
        return "python_modern"
    return "astral"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (REPO_ROOT / SRC).read_text().splitlines() if l.strip()]
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(bucket(r), []).append(r)

    print("=" * 78)
    print(f" splitting {len(rows)} astral records")
    print("=" * 78)
    for name in ("astral", "python_web", "python_modern"):
        g = groups.get(name, [])
        if not g:
            continue
        cmds = sum(bool(CMD.search(answer_of(r))) for r in g)
        docs = sum(1 for r in g if not (r.get("meta") or {}).get("family"))
        fams = Counter(str((r.get("meta") or {}).get("family") or "-doc-") for r in g)
        print(f"\n  {name:14s} {len(g):5d} records"
              f"   doc-scraped {docs:4d}   runnable-command answers {cmds*100.0/len(g):5.1f}%")
        print("     " + ", ".join(f"{k}={v}" for k, v in fams.most_common(6)))

    print("\n  -> astral goes from 13.7% to "
          f"{sum(bool(CMD.search(answer_of(r))) for r in groups['astral'])*100.0/len(groups['astral']):.1f}% "
          "runnable-command answers just by removing what was never astral.")
    print("     Still short of the >=40% target: the doc-scraped half is largely")
    print("     explanatory prose. That gap needs NEW command-shaped data, not a split.")

    if not args.write:
        print("\n  (dry run -- pass --write)")
        return

    # PRESERVE the pre-split corpus. m2_astral_r8a128_v4 was trained on the
    # 1433-record file; overwriting it in place would make that adapter
    # irreproducible, and an adapter you cannot rebuild is an adapter you cannot
    # trust a benchmark against.
    orig = REPO_ROOT / SRC
    keep = orig.with_name("training_data_v4_unsplit.jsonl")
    if not keep.exists():
        shutil.copy2(orig, keep)
        print(f"  BACKED UP {orig.name} -> {keep.name} "
              f"(the corpus m2_astral_r8a128_v4 was trained on)")

    for name, g in groups.items():
        out = REPO_ROOT / f"data/{name}/training_data_v4.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(json.dumps(r) for r in g) + "\n")
        print(f"  WROTE {out}  ({len(g)} records)")

    for book, dest in BOOKS.items():
        src = REPO_ROOT / "data/astral" / book
        if src.exists():
            dst = REPO_ROOT / f"data/{dest}" / book
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            print(f"  MOVED {book[:44]}... -> data/{dest}/")

    print("\n  NEXT: these two new domains have NO evaluation_data.jsonl and are not")
    print("  in canon.DOMAINS. They are corpora, not experts, until both exist.")


if __name__ == "__main__":
    main()
