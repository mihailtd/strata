"""Evaluation sets for the disposition corpora.

HOLDOUT DESIGN -- read this before adding items
-----------------------------------------------
These situations do NOT appear in training. That is the point, and it is a
different holdout from the one we rejected earlier.

We do not hold out CONSTRUCTS: `uv add`, `read_parquet` and `pgvector` are all
trained deliberately and saturated, because the tool surface is closed and an
expert that has never seen `uv sync` is broken, not general (see
scripts/corpus/build_astral_commands.py).

We hold out SITUATIONS. The preference is the thing being taught -- "reach for the
database's own features rather than bolting on a second system" -- and a preference
is only demonstrated when it fires on a case the corpus never showed. Every item
below exercises a preference that IS trained, through a scenario that is not.

Each item carries BOTH directions, because "mentioned the right tool" is not the
measurement that matters:

    expects  the right approach appeared
    avoid    the common wrong approach appeared   <- MANUAL, the failure that every
                                                     execution-gated benchmark
                                                     scores as a success

`expects` keeps the file readable by the existing scorers, which look for that key
and ignore the rest. `avoid` is what a disposition-aware scorer needs.

RULE FOR EVERY PROMPT: it must NOT name a tool, library, function or extension.
Enforced below -- the build fails rather than emitting a prompt that gives away its
own answer.

    uv run python scripts/corpus/build_disposition_evals.py
    uv run python scripts/corpus/build_disposition_evals.py --write
"""

from __future__ import annotations

import argparse
import json
import re

from runtime_common.canon import REPO_ROOT

# domain -> [(id, category, prompt, expects[], avoid[])]
EVALS: dict[str, list[tuple[str, str, str, list[str], list[str]]]] = {
    "astral": [
        (
            "ast_e1",
            "deps",
            "Our repo still has a requirements file and a checked-in virtualenv. I want dependency management that a new hire can reproduce exactly. What should I do?",
            [r"\buv\b", r"uv\.lock", r"uv (add|init|lock|sync)"],
            [r"pip install", r"pip freeze", r"\bpoetry\b", r"\bpipenv\b", r"virtualenv"],
        ),
        (
            "ast_e2",
            "deps",
            "I need the HTTP client library available for a script, but it must not end up in what we ship to production.",
            [r"uv add --dev", r"--dev\b", r"dependency-groups|dev group"],
            [r"pip install", r"\bpoetry\b"],
        ),
        (
            "ast_e3",
            "pyver",
            "Our CI image has one Python version and two of our services need different ones. How do I make each project get the right interpreter automatically?",
            [r"uv python (install|pin)", r"\.python-version"],
            [r"\bpyenv\b", r"deadsnakes", r"\bconda\b"],
        ),
        (
            "ast_e4",
            "lint",
            "Our pipeline runs three separate tools for style, imports and dead code, and they disagree with each other. Simplify it.",
            [r"\bruff\b", r"ruff (check|format)"],
            [r"\bblack\b", r"\bisort\b", r"\bflake8\b", r"autopep8"],
        ),
        (
            "ast_e5",
            "run",
            "A maintenance script needs two libraries the main project does not use. I do not want to pollute the project environment.",
            [r"uv run", r"--with\b", r"# /// script|PEP 723|inline script metadata"],
            [r"python -m venv", r"pip install", r"virtualenv"],
        ),
        (
            "ast_e6",
            "ci",
            "Builds pass locally and fail in CI with different package versions. How do I make CI use exactly what I have?",
            [r"uv sync", r"--frozen|--locked", r"uv\.lock"],
            [r"pip install -r", r"requirements\.txt"],
        ),
        (
            "ast_e7",
            "publish",
            "We need to cut a release of an internal library and push it to our private index.",
            [r"uv build", r"uv publish"],
            [r"setup\.py sdist", r"\btwine\b", r"python -m build"],
        ),
        (
            "ast_e8",
            "inspect",
            "Something is pulling an old version of a transitive package into our environment and I cannot tell what.",
            [r"uv tree", r"uv lock --upgrade-package"],
            [r"pipdeptree", r"pip show"],
        ),
    ],
    "financial_planning": [
        (
            "fin_e1",
            "priority",
            "I have a few hundred spare each month. My job puts in 50c for every dollar I contribute to the retirement plan, up to 6% of salary. Where should the money go?",
            [r"match", r"\b6%|full match|at least enough"],
            [r"brokerage first", r"pay (off|down).{0,30}(mortgage|low)", r"savings account"],
        ),
        (
            "fin_e2",
            "debt",
            "I owe 8k on a store card and I also want to start putting money in the market.",
            [r"(pay|clear).{0,30}(card|debt|balance) first", r"interest rate|guaranteed return"],
            [r"invest (first|while)", r"do both equally", r"minimum payment.{0,20}invest"],
        ),
        (
            "fin_e3",
            "selection",
            "I'm picking what to hold in a long-term retirement account and I've shortlisted a few well-known companies.",
            [r"index|total market|diversif", r"low.cost|expense ratio"],
            [r"individual (stock|compan)", r"stock.pick", r"actively managed"],
        ),
        (
            "fin_e4",
            "insurance",
            "Someone is selling me a policy that pays out when I die and also builds a cash value I can borrow against.",
            [r"\bterm\b", r"invest the difference|separate"],
            [r"whole life|permanent|universal life", r"cash value.{0,20}(good|great|invest)"],
        ),
        (
            "fin_e5",
            "timing",
            "Everything looks overpriced right now so I've been sitting in cash waiting for a pullback.",
            [r"(regular|fixed).{0,20}(schedule|interval)|dollar.cost|time in the market"],
            [r"wait (for|until)", r"time the market", r"buy the dip"],
        ),
        (
            "fin_e6",
            "rollover",
            "Changing jobs and the old workplace plan has about 9k in it. Payroll asked what I want done with it.",
            [r"roll.?over|transfer|direct rollover|\bIRA\b"],
            [r"cash (it )?out", r"take the (cash|money)", r"withdraw"],
        ),
        (
            "fin_e7",
            "accounts",
            "I've been putting my retirement savings into a regular investment account at my bank.",
            [r"tax.advantaged|401\(?k\)?|\bIRA\b|contribution (room|limit)"],
            [r"taxable is fine", r"keep (it|them) (there|where)"],
        ),
        (
            "fin_e8",
            "concentration",
            "About 70% of what I've saved is in shares of the company I work for, from grants and the share purchase plan.",
            [r"(sell|reduce|diversif|trim)", r"concentrat|single (stock|company)"],
            [r"hold (it|them)|keep holding", r"you know the company"],
        ),
    ],
    "postgresql": [
        (
            "pg_e1",
            "vector",
            "We store product descriptions and want 'find me similar products' without standing up new infrastructure.",
            [r"\bpgvector\b|\bvector\b", r"<=>|<->|hnsw|ivfflat"],
            [r"\bpinecone\b", r"\bweaviate\b", r"\bfaiss\b", r"\bmilvus\b", r"\bqdrant\b", r"\belasticsearch\b"],
        ),
        (
            "pg_e2",
            "concurrency",
            "Two API replicas both process the same signup row occasionally, sending duplicate emails.",
            [r"skip locked|for update", r"on conflict"],
            [r"select.{0,40}then (insert|update)", r"application[- ]level lock", r"\bredis\b.{0,20}lock"],
        ),
        (
            "pg_e3",
            "schema",
            "Product attributes differ per category and I was going to add a table of attribute name/value rows.",
            [r"\bjsonb\b", r"\bgin\b|@>"],
            [r"entity.attribute.value|\beav\b", r"attribute.{0,10}table"],
        ),
        (
            "pg_e4",
            "perf",
            "Reports filtering on a computed month value scan the whole table despite an index on the date column.",
            [r"expression index|create index.{0,60}\(", r"date_trunc|range|>=.{0,20}<"],
            [r"add more ram", r"denormali[sz]e", r"materiali[sz]ed view.{0,20}only"],
        ),
        (
            "pg_e5",
            "retention",
            "We keep 18 months of telemetry and the monthly purge takes six hours and bloats the table.",
            [r"partition", r"drop table|detach partition"],
            [r"delete from", r"\bvacuum full\b", r"migrate to (mongo|cassandra|clickhouse)"],
        ),
        (
            "pg_e6",
            "search",
            "Users want to search support tickets by keyword and we were about to add a search cluster.",
            [r"tsvector|to_tsvector|websearch_to_tsquery", r"\bgin\b"],
            [r"\belasticsearch\b", r"\bopensearch\b", r"\bsolr\b", r"\bmeilisearch\b"],
        ),
        (
            "pg_e7",
            "keys",
            "New tables need primary keys and half the codebase uses one style and half another.",
            [r"generated always as identity|\bidentity\b", r"uuidv7|uuid_generate_v7"],
            [r"\bserial\b", r"uuid_generate_v4|gen_random_uuid.{0,30}primary key"],
        ),
        (
            "pg_e8",
            "diagnosis",
            "A nightly query started timing out last week and nothing in the code changed.",
            [r"explain\s*\(?\s*analyze", r"pg_stat_statements|\bbuffers\b|\banalyze\b"],
            [r"add (an )?index.{0,20}and see", r"restart the (database|server)", r"increase the timeout"],
        ),
    ],
    "duckdb": [
        (
            "dd_e1",
            "files",
            "Analysts drop gzipped CSVs into a folder each night and want yesterday's totals by region.",
            [r"\bduckdb\b|read_csv|(?<![.\w])read_csv_auto", r"group by all|GROUP BY"],
            [r"\bpandas\b|\bpd\.", r"chunksize", r"for .{0,20}file"],
        ),
        (
            "dd_e2",
            "memory",
            "A 60GB event log needs deduplicating and the box has 32GB of RAM.",
            [r"\bduckdb\b|read_parquet|(?<![.\w])read_csv", r"distinct|group by"],
            [r"\bpandas\b", r"chunksize", r"\bdask\b", r"\bspark\b"],
        ),
        (
            "dd_e3",
            "columns",
            "Our fact table gained fifteen columns and every report query lists columns by hand.",
            [r"\bexclude\b|columns\s*\(", r"group by all"],
            [r"list (each|all|every).{0,20}column", r"maintain.{0,20}column list"],
        ),
        (
            "dd_e4",
            "federation",
            "Reference data lives in the transactional database and the metrics live in files. Analysts export both to spreadsheets.",
            [r"\battach\b", r"type\s+postgres|postgres_scan", r"read_parquet|(?<![.\w])read_csv"],
            [r"export.{0,20}(csv|spreadsheet)", r"\bpandas\b.{0,20}merge", r"\betl\b.{0,20}job"],
        ),
        (
            "dd_e5",
            "interop",
            "The modelling team works in a dataframe library and complains about conversion cost from our query layer.",
            [r"\.pl\s*\(|\barrow\b|zero.copy"],
            [r"to_pandas|\.df\(\)", r"\bcsv\b.{0,20}hand.?off"],
        ),
        (
            "dd_e6",
            "sampling",
            "I need to sanity-check a billion-row table before writing the real query.",
            [r"using sample|\btablesample\b", r"\bsummarize\b|\bdescribe\b"],
            [r"random\(\)\s*<", r"limit 1000.{0,30}not random", r"\bpandas\b.{0,10}sample"],
        ),
        (
            "dd_e7",
            "output",
            "A downstream service wants our aggregate as a typed columnar file, not text.",
            [r"\bcopy\b.{0,40}\bparquet\b|format parquet"],
            [r"to_csv|\bcsv\b", r"to_pandas.{0,20}to_parquet"],
        ),
    ],
    "python_modern": [
        (
            "pm_e1",
            "style",
            "This module joins directory and file names with string concatenation and checks existence with os.path calls.",
            [r"\bpathlib\b|\bPath\(", r"/\s*'|\.exists\(\)|read_text"],
            [r"os\.path\.join", r"os\.path\.exists"],
        ),
        (
            "pm_e2",
            "style",
            "We pass around dictionaries of order fields and a misspelled key reached production.",
            [r"@dataclass|\bdataclass\b|\bBaseModel\b|NamedTuple", r"frozen|slots|: (int|str|float|bool)"],
            [r"plain dict|use a dict", r"\.get\('"],
        ),
        (
            "pm_e3",
            "async",
            "Three independent network calls run one after another and the endpoint is slow.",
            [r"TaskGroup|create_task", r"\basync\b"],
            [r"gather\([^)]*\)(?!.*return_exceptions)", r"thread"],
        ),
        (
            "pm_e4",
            "correctness",
            "A helper accumulates into an argument with a default value and results bleed between calls.",
            [r"=\s*None", r"if .{0,12} is None"],
            [r"=\s*\[\]|=\s*\{\}"],
        ),
        (
            "pm_e5",
            "time",
            "Two services disagree about which of two events happened first.",
            [r"timezone\.utc|\bUTC\b|tz-aware|aware datetime", r"datetime\.now\("],
            [r"utcnow\(\)"],
        ),
        (
            "pm_e6",
            "memory",
            "We hold tens of millions of small records in memory and the process is being OOM-killed.",
            [r"slots|__slots__|\bgenerator\b|\byield\b|iterator"],
            [r"just add ram|increase memory"],
        ),
        (
            "pm_e7",
            "errors",
            "A try block wraps fifty lines and failures disappear without a trace.",
            [r"except \(?[A-Z]", r"\braise\b|logger|logging"],
            [r"except:\s*$|except Exception:\s*\n\s*pass", r"\bpass\b\s*$"],
        ),
        (
            "pm_e8",
            "dispatch",
            "A function branches eight ways on the shape of an incoming payload.",
            [r"\bmatch\b.{0,40}\bcase\b|singledispatch"],
            [r"if .{0,30}elif .{0,30}elif"],
        ),
    ],
    "python_web": [
        (
            "pw_e1",
            "framework",
            "We need a small JSON API in front of an async datastore, with generated docs for the client team.",
            [r"\bfastapi\b", r"async def", r"openapi|/docs"],
            [r"\bdjango\b", r"\bflask\b", r"\bbottle\b"],
        ),
        (
            "pw_e2",
            "validation",
            "Handlers read the raw body and check fields by hand, and the docs no longer match reality.",
            [r"\bpydantic\b|BaseModel", r"response_model|-> [A-Z]"],
            [r"request\.json\(\)", r"if .{0,20}not in (body|payload|data)"],
        ),
        (
            "pw_e3",
            "async",
            "One endpoint calls a third-party API and under load the whole service stops responding.",
            [r"\bhttpx\b|aiohttp|await .{0,20}client", r"run_in_threadpool|to_thread"],
            [r"requests\.(get|post)", r"urllib"],
        ),
        (
            "pw_e4",
            "lifecycle",
            "The connection pool is created at import time and tests leak connections.",
            [r"\blifespan\b|asynccontextmanager", r"app\.state|yield"],
            [r"on_event\(", r"module.level|import time"],
        ),
        (
            "pw_e5",
            "errors",
            "Failures come back as 200 with an error field and the client retry logic never triggers.",
            [r"HTTPException|status_code=(4|5)\d\d", r"\braise\b"],
            [r"return \{.{0,20}error", r"status.{0,10}200"],
        ),
        (
            "pw_e6",
            "testing",
            "The test suite boots a real server on a port and is flaky in CI.",
            [r"ASGITransport|TestClient|AsyncClient", r"dependency_overrides|in-process"],
            [r"uvicorn.{0,20}thread", r"localhost:\d+", r"time\.sleep"],
        ),
        (
            "pw_e7",
            "scale",
            "A list endpoint returns every row and the response has grown to several megabytes.",
            [r"limit|paginat|cursor", r"Query\(|le=|max"],
            [r"return all|fetchall.{0,20}return"],
        ),
        (
            "pw_e8",
            "config",
            "Environment variables are read in a dozen modules and a typo only surfaced in production.",
            [r"BaseSettings|pydantic_settings|Settings\(", r"validat"],
            [r"os\.environ\[", r"os\.getenv"],
        ),
    ],
}

# A prompt that names its own tool measures capability, not disposition.
BANNED = {
    "astral": ["uv ", "uvx", "ruff", " ty ", "pyproject", "lockfile", "uv.lock"],
    "financial_planning": [
        (
            "fin_e1",
            "priority",
            "I have a few hundred spare each month. My job puts in 50c for every dollar I contribute to the retirement plan, up to 6% of salary. Where should the money go?",
            [r"match", r"\b6%|full match|at least enough"],
            [r"brokerage first", r"pay (off|down).{0,30}(mortgage|low)", r"savings account"],
        ),
        (
            "fin_e2",
            "debt",
            "I owe 8k on a store card and I also want to start putting money in the market.",
            [r"(pay|clear).{0,30}(card|debt|balance) first", r"interest rate|guaranteed return"],
            [r"invest (first|while)", r"do both equally", r"minimum payment.{0,20}invest"],
        ),
        (
            "fin_e3",
            "selection",
            "I'm picking what to hold in a long-term retirement account and I've shortlisted a few well-known companies.",
            [r"index|total market|diversif", r"low.cost|expense ratio"],
            [r"individual (stock|compan)", r"stock.pick", r"actively managed"],
        ),
        (
            "fin_e4",
            "insurance",
            "Someone is selling me a policy that pays out when I die and also builds a cash value I can borrow against.",
            [r"\bterm\b", r"invest the difference|separate"],
            [r"whole life|permanent|universal life", r"cash value.{0,20}(good|great|invest)"],
        ),
        (
            "fin_e5",
            "timing",
            "Everything looks overpriced right now so I've been sitting in cash waiting for a pullback.",
            [r"(regular|fixed).{0,20}(schedule|interval)|dollar.cost|time in the market"],
            [r"wait (for|until)", r"time the market", r"buy the dip"],
        ),
        (
            "fin_e6",
            "rollover",
            "Changing jobs and the old workplace plan has about 9k in it. Payroll asked what I want done with it.",
            [r"roll.?over|transfer|direct rollover|\bIRA\b"],
            [r"cash (it )?out", r"take the (cash|money)", r"withdraw"],
        ),
        (
            "fin_e7",
            "accounts",
            "I've been putting my retirement savings into a regular investment account at my bank.",
            [r"tax.advantaged|401\(?k\)?|\bIRA\b|contribution (room|limit)"],
            [r"taxable is fine", r"keep (it|them) (there|where)"],
        ),
        (
            "fin_e8",
            "concentration",
            "About 70% of what I've saved is in shares of the company I work for, from grants and the share purchase plan.",
            [r"(sell|reduce|diversif|trim)", r"concentrat|single (stock|company)"],
            [r"hold (it|them)|keep holding", r"you know the company"],
        ),
    ],
    "postgresql": ["pgvector", "jsonb", "distinct on", "partition", "tsvector", "identity"],
    "duckdb": ["duckdb", "read_parquet", "read_csv", "qualify", "exclude(", "polars"],
    "python_modern": ["pathlib", "dataclass", "taskgroup", "match/case", "__slots__"],
    "python_web": ["fastapi", "pydantic", "httpx", "lifespan", "asgi", "depends"],
    "financial_planning": ["index fund", "term life", "roth", "401(k) match", "rollover"],
}


def check() -> list[str]:
    bad = []
    seen = set()
    for dom, items in EVALS.items():
        for _id, _cat, prompt, exp, avoid in items:
            if _id in seen:
                bad.append(f"{_id}: duplicate id")
            seen.add(_id)
            low = prompt.lower()
            for w in BANNED.get(dom, []):
                if w in low:
                    bad.append(f"{_id}: prompt names its own tool ({w!r})")
            if not exp or not avoid:
                bad.append(f"{_id}: needs both expects and avoid")
            for pat in exp + avoid:
                try:
                    re.compile(pat)
                except re.error as e:
                    bad.append(f"{_id}: bad regex {pat!r}: {e}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    problems = check()
    print("=" * 78)
    print(" DISPOSITION EVALUATION SETS -- held-out SITUATIONS, trained CONSTRUCTS")
    print("=" * 78)
    for dom, items in EVALS.items():
        print(
            f"  {dom:15s} {len(items):3d} items   "
            f"{sum(len(e) for _, _, _, e, _ in items):3d} expects / "
            f"{sum(len(a) for *_, a in items):3d} avoid patterns"
        )
    if problems:
        print("\n  PROBLEMS:")
        for p in problems:
            print(f"    {p}")
        raise SystemExit(1)
    print("\n  sanity: no prompt names its own tool; all regexes compile; ids unique")

    if not args.write:
        print("\n  (dry run -- pass --write)")
        return
    for dom, items in EVALS.items():
        out = REPO_ROOT / f"data/{dom}/evaluation_data_disposition.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "\n".join(
                json.dumps({"id": i, "category": c, "prompt": p, "expects": e, "avoid": a}) for i, c, p, e, a in items
            )
            + "\n"
        )
        print(f"  WROTE {out}  ({len(items)} items)")


if __name__ == "__main__":
    main()
