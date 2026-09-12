"""DuckDB: window functions and quantiles -- the surface it was asked for and lacks.

WHAT THIS FIXES -- measured
---------------------------
apps/factory/data/duckdb/training_data_v5.jsonl, 1184 records:

    read_parquet 190   GROUP BY ALL 160   EXCLUDE 57   COLUMNS( 53
    OVER(         40   PARTITION BY  34   QUALIFY  28   row_number 27
    rank()         3   dense_rank     3   lag(      2   lead(       2
    median        26   quantile       1   PERCENTILE 0   approx_quantile 0

Asked for "parquet files with window queries and quantile aggregations", the v6
expert produced neither: no `read_parquet`, no `OVER`, and `PERCENTILE_CONT(price,
0.95)` -- which is not DuckDB's signature. `PERCENTILE_CONT` takes the fraction and
an ordering clause; `quantile_cont(col, frac)` is the aggregate form.

Zero records is not a weak signal, it is no signal. The expert guessed, and a guess
delivered in the expert's voice is worse than base answering plainly.

DESIGN -- lessons from the two builders written before this one
---------------------------------------------------------------
1. Vary answers LEXICALLY. The dedup key strips digits, so a family that differs only
   in a fraction or a row count collapses to a single record.
2. Cap per family. Otherwise the two highest-variety families supply most of the
   survivors and drag the whole corpus toward their shape.
3. Hold the `**Not X**` share near the 33% this corpus already runs at. python_modern
   at 76% is the adapter that now answers every question with a dataclass.

    uv run python scripts/corpus/build_duckdb_analytics.py [--write]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter

from runtime_common.canon import REPO_ROOT

ANSWER_MARKER = "\n\n### Answer:\n"
GEN = "duckdb_analytics_v1"

TBL = [
    "events",
    "orders",
    "trades",
    "sessions",
    "readings",
    "clicks",
    "shipments",
    "invoices",
    "pageviews",
    "telemetry",
]
GRP = ["region", "customer_id", "device", "symbol", "channel", "warehouse", "tenant_id", "country"]
VAL = ["amount", "latency_ms", "price", "duration", "bytes", "score", "qty", "revenue"]
TS = ["ts", "event_time", "created_at", "occurred_at", "recorded_at"]
GLOB = [
    "s3://bucket/{t}/*.parquet",
    "data/{t}/*.parquet",
    "data/{t}/year=*/*.parquet",
    "/mnt/lake/{t}/**/*.parquet",
    "raw/{t}_*.parquet",
]
FRACS = [("0.5", "median"), ("0.95", "p95"), ("0.99", "p99"), ("0.9", "p90"), ("0.75", "upper quartile")]
NS = [3, 5, 10, 20]

CTX = [
    "",
    " in a nightly report",
    " for a dashboard",
    " on a 60 GB export",
    " without loading it all into pandas",
    " in a notebook",
    " from a cron job",
    " where the files land hourly",
    " across a year of partitions",
    " on a laptop with 16 GB of RAM",
]
PHRASE = [
    "{task}?",
    "How do I {task}?",
    "Need to {task}.",
    "{task} -- query please.",
    "What's the cleanest way to {task}?",
    "{task}. Show the SQL.",
    "Can you {task}?",
]

ALT_QUALIFY = [
    "`QUALIFY` filters on the window function without the subquery a plain `WHERE` "
    "would force, because `WHERE` runs before the window is computed",
    "`QUALIFY` is the clause that can see a window result -- `WHERE` runs first, so "
    "filtering on `row_number()` there needs a wrapping subquery",
    "you cannot put a window function in `WHERE`; `QUALIFY` exists precisely for this and saves the nested select",
]
ALT_QUANT = [
    "`quantile_cont` interpolates between the two bracketing values; `quantile_disc` "
    "returns an actual value from the column",
    "use `quantile_cont` when an interpolated figure is fine and `quantile_disc` when "
    "the answer has to be a value that really occurs",
    "`quantile_cont` gives the interpolated point, `quantile_disc` snaps to a real "
    "observation -- pick by whether an invented value is acceptable",
]
ALT_PUSH = [
    "`read_parquet` pushes the projection and the filter into the scan, so only the "
    "columns and row groups you need are read off disk",
    "the predicate and column list are pushed down into the parquet reader -- whole "
    "row groups are skipped without being decompressed",
    "DuckDB reads parquet lazily: unreferenced columns are never decoded and row "
    "groups outside the filter never leave disk",
]


def rec(family: str, question: str, answer: str) -> dict:
    return {
        "text": f"### Question:\n{question}{ANSWER_MARKER}{answer}",
        "messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
        "meta": {"family": family, "tool": "duckdb", "gen": GEN},
    }


def build(rng: random.Random, n_per: int) -> list[dict]:
    out: list[dict] = []

    def add(fam: str, task: str, ans: str) -> None:
        q = rng.choice(PHRASE).format(task=task + rng.choice(CTX))
        out.append(rec(fam, q[0].upper() + q[1:], ans))

    for _ in range(n_per):
        t = rng.choice(TBL)
        g = rng.choice(GRP)
        v = rng.choice(VAL)
        ts = rng.choice(TS)
        n = rng.choice(NS)
        frac, fname = rng.choice(FRACS)
        src = rng.choice(GLOB).format(t=t)

        add(
            "ddb_quantile_agg",
            f"get the {fname} of {v} per {g} from {t}",
            f"```sql\nSELECT\n    {g},\n    quantile_cont({v}, {frac}) AS {fname}_{v},\n"
            f"    median({v})                 AS median_{v},\n"
            f"    count(*)                    AS n\nFROM read_parquet('{src}')\n"
            f"GROUP BY ALL\nORDER BY {fname}_{v} DESC;\n```\n\n"
            f"{rng.choice(ALT_QUANT)}.\n\n"
            f"**Not `PERCENTILE_CONT({v}, {frac})`** — that signature does not exist. "
            f"The ordered-set form is `PERCENTILE_CONT({frac}) WITHIN GROUP (ORDER BY "
            f"{v})`; the plain aggregate is `quantile_cont({v}, {frac})`.",
        )

        add(
            "ddb_quantile_within_group",
            f"compute several percentiles of {v} in one pass over {t}",
            f"```sql\nSELECT\n    {g},\n"
            f"    PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY {v}) AS p50,\n"
            f"    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY {v}) AS p95,\n"
            f"    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY {v}) AS p99\n"
            f"FROM read_parquet('{src}')\nGROUP BY ALL;\n```\n\n"
            f"You can also pass a list to the aggregate form and get an array back:\n\n"
            f"```sql\nSELECT {g}, quantile_cont({v}, [0.5, 0.95, 0.99]) AS pcts\n"
            f"FROM read_parquet('{src}') GROUP BY ALL;\n```\n\n"
            f"{rng.choice(['One scan either way -- the ordered-set form is clearer when the fractions are fixed', 'both read the file once; the list form is easier to widen later', 'these cost the same scan, so pick on readability'])}.",
        )

        add(
            "ddb_approx_quantile",
            f"get approximate percentiles of {v} over a very large {t} export",
            f"```sql\nSELECT\n    {g},\n"
            f"    approx_quantile({v}, {frac}) AS {fname}_approx,\n"
            f"    approx_count_distinct({g})   AS distinct_{g}\n"
            f"FROM read_parquet('{src}')\nGROUP BY ALL;\n```\n\n"
            f"{rng.choice(['`approx_quantile` uses a t-digest -- bounded memory, and it does not need to sort', '`approx_quantile` sketches the distribution instead of sorting, so memory stays flat', 'the t-digest behind `approx_quantile` keeps memory constant no matter how many rows arrive'])}. "
            f"Exact `quantile_cont` has to materialise and order the group, which is "
            f"what runs you out of memory on a wide export.",
        )

        add(
            "ddb_topn_per_group",
            f"return the top {n} rows of {t} per {g} by {v}",
            f"```sql\nSELECT {g}, id, {v}\nFROM read_parquet('{src}')\n"
            f"QUALIFY row_number() OVER (PARTITION BY {g} ORDER BY {v} DESC) <= {n};\n```\n\n"
            f"{rng.choice(ALT_QUALIFY)}.\n\n"
            f"{rng.choice(['`rank()` leaves gaps after ties and `dense_rank()` does not -- use `row_number()` when you want exactly ' + str(n), 'swap in `rank()` if ties should all come through, `dense_rank()` if you want ties without gaps in the numbering', 'if ties matter, `rank()` keeps them all and `dense_rank()` numbers them without gaps'])}.",
        )

        add(
            "ddb_window_running",
            f"add a running total and a 7-row moving average of {v} to {t}",
            f"```sql\nSELECT\n    {ts},\n    {g},\n    {v},\n"
            f"    sum({v}) OVER w_run                          AS running_{v},\n"
            f"    avg({v}) OVER w_roll                         AS moving_avg\n"
            f"FROM read_parquet('{src}')\nWINDOW\n"
            f"    w_run  AS (PARTITION BY {g} ORDER BY {ts}\n"
            f"               ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),\n"
            f"    w_roll AS (PARTITION BY {g} ORDER BY {ts}\n"
            f"               ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)\n"
            f"ORDER BY {g}, {ts};\n```\n\n"
            f"{rng.choice(['The named `WINDOW` clause is worth it the moment two columns share a frame -- define once, reference twice', 'a named `WINDOW` avoids repeating the frame for every column that uses it', 'once more than one column needs the same frame, name it in `WINDOW` instead of repeating it'])}. "
            f"`ROWS` counts rows; `RANGE` counts by the value of the ordering column, "
            f"which is what you want for a true time window over irregular timestamps.",
        )

        add(
            "ddb_lag_lead",
            f"compute the change in {v} between consecutive {t} rows per {g}",
            f"```sql\nSELECT\n    {ts},\n    {g},\n    {v},\n"
            f"    lag({v}) OVER w                    AS prev_{v},\n"
            f"    {v} - lag({v}) OVER w              AS delta,\n"
            f"    lead({ts}) OVER w                  AS next_{ts}\n"
            f"FROM read_parquet('{src}')\n"
            f"WINDOW w AS (PARTITION BY {g} ORDER BY {ts})\nQUALIFY prev_{v} IS NOT NULL;\n```\n\n"
            f"{rng.choice(['`lag` looks backwards, `lead` forwards; both take an optional offset and default', 'give `lag` a second argument to reach further back and a third for the value when there is no previous row', '`lag(col, 2, 0)` reaches two rows back and substitutes 0 at the partition edge'])}. "
            f"The `QUALIFY` drops the first row of each partition, where there is no "
            f"predecessor.",
        )

        add(
            "ddb_parquet_pushdown",
            f"read only what you need out of a directory of {t} parquet files",
            f"```sql\nSELECT {g}, {ts}, {v}\nFROM read_parquet('{src}', hive_partitioning = true)\n"
            f"WHERE {ts} >= DATE '2026-01-01'\n  AND {v} IS NOT NULL;\n```\n\n"
            f"{rng.choice(ALT_PUSH)}.\n\n"
            f"**Not a glob into pandas then `concat`** — that decompresses every column "
            f"of every file into memory first, and the filter runs after you have "
            f"already paid for all of it.",
        )

        add(
            "ddb_summarize",
            f"see the shape of a {t} export before writing any real query",
            f"```sql\nSUMMARIZE SELECT * FROM read_parquet('{src}');\n```\n\n"
            f"{rng.choice(['One statement gives min, max, approximate quantiles, distinct counts and null percentage per column', 'you get per-column min/max, q25/q50/q75, approx distinct and null share in a single pass', 'it reports type, min, max, approximate quartiles, distinct count and null percentage for every column'])}.\n\n"
            f"**Not `df.describe()`** — that requires loading the frame first, and it "
            f"skips the non-numeric columns that usually explain the problem.",
        )

    return out


def report(rows: list[dict]) -> None:
    ans = [r["messages"][1]["content"] for r in rows]

    def norm(s: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9\s]", " ", re.sub(r"\d+", "0", s.lower())).split())

    print(f"  records          {len(rows)}")
    print(f"  unique questions {len(set(r['messages'][0]['content'] for r in rows)) / len(rows):6.1%}")
    print(f"  unique answers   {len(set(norm(a) for a in ans)) / len(rows):6.1%}")
    print(f"  emits sql        {sum('```sql' in a for a in ans) / len(rows):6.1%}")
    print(f"  '**Not' share    {sum('**Not ' in a for a in ans) / len(rows):6.1%}")
    print("\n  families:")
    for f, c in Counter(r["meta"]["family"] for r in rows).most_common():
        print(f"    {f:28s} {c:5d}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--n-per", type=int, default=200)
    ap.add_argument("--cap", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = build(random.Random(args.seed), args.n_per)
    seen, uniq, per_fam = set(), [], Counter()
    for r in rows:
        k = " ".join(re.sub(r"[^a-z0-9\s]", " ", re.sub(r"\d+", "0", r["messages"][1]["content"].lower())).split())
        fam = r["meta"]["family"]
        if k in seen or per_fam[fam] >= args.cap:
            continue
        seen.add(k)
        per_fam[fam] += 1
        uniq.append(r)
    print("=" * 78)
    print(" DUCKDB: window functions, quantiles, parquet pushdown")
    print("=" * 78)
    print(f"  generated {len(rows)}, {len(uniq)} kept after dedup + per-family cap\n")
    rows = uniq
    report(rows)

    if args.write:
        out = REPO_ROOT / "apps/factory/data/duckdb/training_data_analytics.jsonl"
        out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        print(f"\n  WROTE {out.relative_to(REPO_ROOT)}")
    else:
        print("\n  (dry run -- pass --write)")


if __name__ == "__main__":
    main()
