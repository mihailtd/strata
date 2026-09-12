"""Disposition tasks -- does the model REACH FOR the native tool?

THE RULE FOR EVERY PROMPT HERE
------------------------------
**The prompt must never name the tool, the library, or the function.** It states a
problem and stops. The moment a prompt says "using uv" or "in DuckDB" it stops
measuring disposition and starts measuring capability, which we already measure
elsewhere.

WHY THIS BENCHMARK EXISTS
-------------------------
Every other instrument in this repo asks "can it produce correct output when told
what to use". This one asks "does it pick the right thing when told nothing".

That is roughly half of what an expert is for, and none of what we currently
measure. See docs/WHY_EXPERTS.md.

The failure this catches is invisible to execution gating: a model that solves the
problem CORRECTLY WITH THE WRONG TOOL. Reading 40GB of Parquet with a Python loop
produces the right numbers and passes any correctness check, while being orders of
magnitude slower than the one-liner the expert should have reached for.

CONTAMINATION
-------------
This benchmark is contamination-resistant BY CONSTRUCTION and that is the point.
We are not testing whether the model knows `read_parquet` -- it had better, we
trained it on that deliberately. We are testing whether it VOLUNTEERS it. The
construct being in training is a prerequisite, not a leak. What has to be held out
is the PROBLEM, not the TOOL.

SCORING
-------
    NATIVE  native pattern hit, no manual pattern    -> 1.0   the win
    MIXED   both                                     -> 0.5   knows it, hedges
    MANUAL  manual only                              -> 0.0   THE failure mode
    NEITHER neither                                  -> 0.0   didn't engage

MANUAL and NEITHER are both 0.0 but are reported separately: MANUAL means it
confidently did the wrong thing, which is worse in practice than not answering.
"""

from __future__ import annotations

import re

# Each task: (id, domain, prompt, native_patterns, manual_patterns)
# Patterns are matched case-insensitively against the generated text.

TASKS: list[dict] = [
    # ---------------------------------------------------------------- astral
    {
        "id": "ast_add_dep", "domain": "astral",
        "prompt": "I need to add the requests library to my Python project. What should I do?",
        "native": [r"\buv add\b"],
        "manual": [r"pip install", r"requirements\.txt",
                   r"add.{0,30}to.{0,20}\[project\]", r"\bpoetry\b"],
    },
    {
        "id": "ast_lock", "domain": "astral",
        "prompt": "Make my Python project's dependencies reproducible across machines.",
        "native": [r"\buv lock\b", r"\buv sync\b", r"uv\.lock"],
        "manual": [r"pip freeze", r"requirements\.txt", r"\bpoetry\b", r"pipenv", r"\bpdm\b"],
    },
    {
        "id": "ast_oneoff", "domain": "astral",
        "prompt": "Run a one-off Python script that needs httpx, without installing "
                  "anything into my global environment.",
        "native": [r"\buv run\b", r"\buvx\b", r"# /// script", r"PEP 723",
                   r"inline script metadata"],
        "manual": [r"python -m venv", r"virtualenv", r"pip install", r"conda create"],
    },
    {
        "id": "ast_newproj", "domain": "astral",
        "prompt": "Start a new Python project with a src layout and a pinned interpreter.",
        "native": [r"\buv init\b", r"\buv venv\b", r"uv python (install|pin)"],
        "manual": [r"python -m venv", r"setup\.py", r"pipenv", r"\bpoetry\b", r"conda"],
    },
    {
        "id": "ast_format", "domain": "astral",
        "prompt": "What should I run to auto-format a Python repository and sort its imports? Give me the command.",
        "native": [r"ruff format", r"ruff check --fix", r"\bruff\b"],
        "manual": [r"\bblack\b", r"\bisort\b", r"autopep8", r"yapf"],
    },
    {
        "id": "ast_lint", "domain": "astral",
        "prompt": "What should I run to catch unused imports and undefined names across a Python repository? Give me the command.",
        "native": [r"ruff check", r"\bruff\b"],
        "manual": [r"\bflake8\b", r"\bpylint\b", r"pyflakes"],
    },
    {
        "id": "ast_upgrade", "domain": "astral",
        "prompt": "Upgrade all my Python dependencies to their latest compatible versions.",
        "native": [r"uv lock --upgrade", r"uv sync --upgrade", r"\buv add --upgrade\b"],
        "manual": [r"pip install (-U|--upgrade)", r"pip-review", r"\bpoetry\b",
                   r"pip-compile", r"requirements\.txt"],
    },
    {
        "id": "ast_pyver", "domain": "astral",
        "prompt": "This project needs Python 3.13 but my system has 3.11. What commands get me a working 3.13 environment for it?",
        "native": [r"uv python install", r"uv venv --python", r"uv python pin"],
        "manual": [r"\bpyenv\b", r"deadsnakes", r"conda create", r"download.{0,20}python\.org"],
    },

    # ------------------------------------------------------------ postgresql
    {
        "id": "pg_semantic", "domain": "postgresql",
        "prompt": "I have 1M documents in my database and I want to search them by meaning "
                  "rather than keywords. How should I build that?",
        "native": [r"\bpgvector\b", r"\bhnsw\b", r"\bivfflat\b", r"vector (column|index|extension)",
                   r"<=>", r"<->", r"<#>", r"\bembedding\b"],   # the OPERATORS are the tell
        "manual": [r"\bpinecone\b", r"\bweaviate\b", r"\bmilvus\b", r"\bqdrant\b",
                   r"\bchroma(db)?\b", r"\bfaiss\b", r"\belasticsearch\b",
                   r"(separate|dedicated|standalone) vector (database|store)"],
    },
    {
        "id": "pg_latest_per", "domain": "postgresql",
        "prompt": "Write a query returning, for each customer, their single most recent order.",
        "native": [r"distinct\s+on", r"row_number\s*\(\s*\)\s*over", r"\bwindow function\b"],
        "manual": [r"fetch.{0,30}(python|application|pandas)", r"\.groupby\(",
                   r"for .{0,20} in .{0,20}(customers|rows)", r"then filter in"],
    },
    {
        "id": "pg_conditional_agg", "domain": "postgresql",
        "prompt": "Count active users and inactive users in a single pass over the table.",
        "native": [r"filter\s*\(\s*where", r"count\s*\(\s*case", r"sum\s*\(\s*case"],
        "manual": [r"two (separate )?quer(y|ies)", r"run it twice", r"\.groupby\(",
                   r"in (python|pandas|the application)"],
    },
    {
        "id": "pg_top_n_per", "domain": "postgresql",
        "prompt": "For every user, return their three largest purchases.",
        "native": [r"\blateral\b", r"row_number\s*\(\s*\)\s*over", r"rank\s*\(\s*\)\s*over"],
        "manual": [r"loop over", r"for each user.{0,40}quer", r"N\+1", r"in a loop"],
    },
    {
        "id": "pg_bigtable", "domain": "postgresql",
        "prompt": "My events table has 500 million rows and time-range queries are slow. What change should I make to the schema?",
        "native": [r"partition(ing|ed)? by range", r"\bbrin\b", r"declarative partition"],
        "manual": [r"(move|migrate|switch) to (mongo|cassandra|clickhouse|influx|timescale)",
                   r"nosql", r"shard.{0,20}application"],
    },
    {
        "id": "pg_upsert", "domain": "postgresql",
        "prompt": "Insert a batch of records, updating any that already exist.",
        "native": [r"on conflict", r"\bmerge\b"],
        "manual": [r"select.{0,40}then (insert|update)", r"check if.{0,30}exists",
                   r"try.{0,20}except.{0,30}(insert|update)"],
    },
    {
        "id": "pg_median", "domain": "postgresql",
        "prompt": "Find the median order value for each region.",
        "native": [r"percentile_cont", r"percentile_disc", r"within group"],
        "manual": [r"fetch all", r"\.median\(\)", r"numpy", r"sort.{0,30}in python",
                   r"pull.{0,20}into (python|pandas)"],
    },
    {
        "id": "pg_slowquery", "domain": "postgresql",
        "prompt": "This query got slow after the table grew. How do I find out why?",
        "native": [r"explain\s+analyze", r"\bexplain\b", r"pg_stat_statements"],
        "manual": [r"time it in (python|the app)", r"add (print|log).{0,20}timing",
                   r"guess", r"trial and error"],
    },

    # ---------------------------------------------------------------- duckdb
    {
        "id": "duck_parquet_agg", "domain": "duckdb",
        "prompt": "I have 40GB of Parquet files partitioned by date on local disk. "
                  "Give me revenue per category for last quarter.",
        "native": [r"(?<![.\w])read_parquet", r"hive_partitioning", r"\bduckdb\b"],
        "manual": [r"\bpandas\b", r"\bpd\.", r"for .{0,20}file", r"\bglob\b",
                   r"\bpyarrow\b.{0,30}Table", r"concat"],
    },
    {
        "id": "duck_exclude", "domain": "duckdb",
        "prompt": "Return every column of this wide table except internal_trace_id.",
        "native": [r"\bexclude\b", r"columns\s*\("],
        "manual": [r"select col1", r"list (out |all )?the columns",
                   r"enumerate.{0,20}column", r"name each column"],
    },
    {
        "id": "duck_polars", "domain": "duckdb",
        "prompt": "Get the result of my analytical query into a Polars DataFrame "
                  "without an extra copy.",
        "native": [r"\.pl\s*\(", r"\barrow\b", r"record batch"],
        "manual": [r"to_pandas", r"pl\.from_pandas", r"\.df\(\)", r"csv"],
    },
    {
        "id": "duck_federate", "domain": "duckdb",
        "prompt": "Join a local CSV file against a table living in my Postgres database.",
        "native": [r"\battach\b", r"type\s+postgres", r"(?<![.\w])read_csv", r"postgres_scan",
                   r"INSTALL\s+\w+", r"LOAD\s+\w+"],
        "manual": [r"pandas", r"read_sql.{0,40}merge", r"export.{0,20}csv.{0,20}import",
                   r"load both into"],
    },
    {
        "id": "duck_groupall", "domain": "duckdb",
        "prompt": "Aggregate this table by every non-numeric column at once.",
        "native": [r"group\s+by\s+all", r"columns\s*\("],
        "manual": [r"list (each|every|all).{0,20}column", r"group by col1",
                   r"write out.{0,20}column"],
    },
    {
        "id": "duck_topn", "domain": "duckdb",
        "prompt": "From this sales table, keep only the top 3 rows per category by revenue.",
        "native": [r"\bqualify\b"],
        "manual": [r"\.groupby\(", r"pandas", r"in python"],
        # NOTE: a row_number() subquery is CORRECT here, just not idiomatic. It is
        # deliberately in neither list, so it scores NEITHER rather than MANUAL --
        # this benchmark must not punish valid SQL, only wrong-tool reaching.
    },
    {
        "id": "duck_bigcsv", "domain": "duckdb",
        "prompt": "I have a 5GB CSV that will not fit in memory. How do I compute summary "
                  "statistics per customer segment?",
        "native": [r"(?<![.\w])read_csv", r"\bduckdb\b", r"out-of-core", r"larger than memory",
                   r"INSTALL\s+\w+", r"FROM\s+[\x27\"][^\x27\"]+\.csv"],   # querying the file directly
        "manual": [r"chunksize", r"\bpandas\b", r"iterate.{0,20}chunk", r"\bdask\b",
                   r"split the file"],
    },
    {
        "id": "duck_sample", "domain": "duckdb",
        "prompt": "How do I pull a random 1% sample of a very large table to eyeball?",
        "native": [r"using sample", r"\btablesample\b", r"reservoir"],
        "manual": [r"random\(\)\s*<", r"order by random", r"fetch all.{0,20}then sample",
                   r"pandas.{0,20}sample"],
    },
]


def by_domain() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for t in TASKS:
        out.setdefault(t["domain"], []).append(t)
    return out


def sanity_check() -> list[str]:
    """A prompt that names its own tool measures nothing. Enforce that."""
    banned = {
        "astral": ["uv ", "ruff", "pyproject"],
        "postgresql": ["postgres", "pgvector", "distinct on", "lateral", "sql"],
        # Ban the TOOL UNDER TEST, not the requested output format. "Polars
        # DataFrame" is the destination the user asked for -- it does not hint at
        # DuckDB any more than "a CSV" would, and the thing being measured is
        # whether the model reaches for DuckDB's zero-copy .pl() or round-trips
        # through pandas. Relaxing this deliberately, not to make the check pass.
        "duckdb": ["duckdb", "read_parquet", "qualify", "exclude", " group by all"],
    }
    problems = []
    seen = set()
    for t in TASKS:
        if t["id"] in seen:
            problems.append(f"{t['id']}: duplicate id")
        seen.add(t["id"])
        low = t["prompt"].lower()
        for word in banned.get(t["domain"], []):
            if word in low:
                problems.append(f"{t['id']}: prompt names its own tool ({word!r})")
        if not t["native"] or not t["manual"]:
            problems.append(f"{t['id']}: needs both native and manual patterns")
        # A prompt that only DESCRIBES a situation gets a restatement back, not a
        # tool choice. "This project needs Python 3.13 but my system has 3.11."
        # returned "You need Python 3.13 or later." -- 14 tokens measuring nothing.
        if not re.search(r"\?|\b(write|give|show|run|make|start|upgrade|count|find|"
                         r"return|insert|aggregate|keep|join|get|add|format|check|"
                         r"pull|compute|what|how)\b", low):
            problems.append(f"{t['id']}: prompt requests no action -- a model can "
                            f"simply restate it")
    return problems


if __name__ == "__main__":
    probs = sanity_check()
    d = by_domain()
    print(f"{len(TASKS)} disposition tasks: " +
          ", ".join(f"{k}={len(v)}" for k, v in d.items()))
    if probs:
        print("\nPROBLEMS:")
        for p in probs:
            print(f"  {p}")
        raise SystemExit(1)
    print("sanity check passed: no prompt names its own tool")
