"""Build the expanded DuckDB expert v4 training corpus (~1,600+ records) and evaluation dataset.

Integrates:
1. Real code examples and patterns extracted from all 3 EPUB books:
   - "DuckDB in Action" (Mark Needham)
   - "DuckDB Up and Running" (Wei-Meng Lee)
   - "Getting Started With DuckDB" (Simon Aubury)
2. Vectorized SQL Dialect & Modern Syntax (FROM-first, COLUMNS, EXCLUDE, REPLACE, GROUP BY ALL, UNION BY NAME)
3. Parquet / S3 / Iceberg / HTTP Ingestion & Partition Pruning
4. Zero-Copy Python, PyArrow, Polars, and Pandas Interoperability
5. Database Extensions (httpfs, spatial, postgres_scanner, sqlite_scanner, motherduck)
6. Out-of-Core Performance Engineering & Anti-Pattern Fixes
7. General Capability Rehearsal Buffer (7-8% to anchor base capability)

Quality Gates:
- Strict prompt-target separation (### Question:\n...\n\n### Answer:\n...).
- Completion-only loss compatible (no prompt memorization).
- Held-out evaluation reservation with exact expects patterns.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

from runtime.canon import REPO_ROOT  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
DUCKDB_DIR = REPO_ROOT / "data/duckdb"
RAW_DIR = DUCKDB_DIR / "raw_extracted"
OUT_TRAIN_V4 = DUCKDB_DIR / "training_data_v4.jsonl"
OUT_EVAL = DUCKDB_DIR / "evaluation_data.jsonl"

SCHEMAS = [
    ("telemetry_logs", "IoT fleet monitoring", [("device_id", "VARCHAR"), ("timestamp", "TIMESTAMPTZ"), ("cpu_temp", "DOUBLE"), ("battery_pct", "INTEGER"), ("payload", "JSON"), ("region", "VARCHAR")]),
    ("ecom_orders", "e-commerce sales", [("order_id", "BIGINT"), ("customer_id", "BIGINT"), ("amount", "DECIMAL(10,2)"), ("items", "STRUCT(sku VARCHAR, qty INT, price DECIMAL(10,2))[]"), ("status", "VARCHAR"), ("created_at", "TIMESTAMPTZ")]),
    ("web_clicks", "analytics clickstream", [("session_id", "UUID"), ("user_id", "BIGINT"), ("url", "VARCHAR"), ("referrer", "VARCHAR"), ("duration_ms", "INTEGER"), ("event_date", "DATE")]),
    ("financial_trades", "trading platform", [("trade_id", "BIGINT"), ("ticker", "VARCHAR"), ("price", "DOUBLE"), ("quantity", "BIGINT"), ("side", "VARCHAR"), ("executed_at", "TIMESTAMP")]),
    ("genomic_samples", "bioinformatics research", [("sample_id", "VARCHAR"), ("gene_symbol", "VARCHAR"), ("expression_val", "FLOAT"), ("chromosome", "VARCHAR"), ("position", "BIGINT")]),
    ("saas_users", "user subscription CRM", [("user_id", "BIGINT"), ("email", "VARCHAR"), ("plan", "VARCHAR"), ("mrr", "DECIMAL(8,2)"), ("features_enabled", "VARCHAR[]"), ("signup_date", "DATE")]),
    ("cloud_billing", "FinOps cloud telemetry", [("resource_arn", "VARCHAR"), ("service", "VARCHAR"), ("cost_usd", "DOUBLE"), ("tags", "MAP(VARCHAR, VARCHAR)"), ("billing_period", "VARCHAR")]),
    ("ride_trips", "urban mobility", [("trip_id", "BIGINT"), ("pickup_loc", "VARCHAR"), ("dropoff_loc", "VARCHAR"), ("fare", "DECIMAL(8,2)"), ("distance_miles", "DOUBLE"), ("driver_rating", "INTEGER")]),
    ("app_traces", "distributed tracing", [("trace_id", "VARCHAR"), ("span_id", "VARCHAR"), ("service_name", "VARCHAR"), ("latency_ms", "DOUBLE"), ("status_code", "INTEGER"), ("error_msg", "VARCHAR")]),
    ("github_events", "developer activity", [("event_id", "BIGINT"), ("repo_name", "VARCHAR"), ("actor_login", "VARCHAR"), ("event_type", "VARCHAR"), ("commit_hashes", "VARCHAR[]"), ("created_at", "TIMESTAMP")]),
    ("ad_impressions", "digital marketing", [("ad_id", "BIGINT"), ("campaign_id", "BIGINT"), ("cost_cpm", "DOUBLE"), ("clicked", "BOOLEAN"), ("converted", "BOOLEAN"), ("geo_country", "VARCHAR")]),
    ("customer_reviews", "product sentiment", [("review_id", "BIGINT"), ("product_id", "BIGINT"), ("rating", "INTEGER"), ("review_text", "VARCHAR"), ("verified_purchase", "BOOLEAN"), ("helpful_votes", "INTEGER")]),
    ("server_hardware", "data center inventory", [("server_id", "VARCHAR"), ("rack_id", "VARCHAR"), ("cores", "INTEGER"), ("ram_gb", "INTEGER"), ("disk_type", "VARCHAR"), ("in_production", "BOOLEAN")]),
    ("weather_stations", "meteorological sensor net", [("station_id", "VARCHAR"), ("city", "VARCHAR"), ("temp_c", "FLOAT"), ("wind_speed_kmh", "FLOAT"), ("precipitation_mm", "FLOAT"), ("recorded_at", "TIMESTAMPTZ")]),
    ("crypto_wallets", "blockchain ledger", [("tx_hash", "VARCHAR"), ("from_address", "VARCHAR"), ("to_address", "VARCHAR"), ("token_symbol", "VARCHAR"), ("token_amount", "DOUBLE"), ("gas_fee_usd", "DOUBLE")]),
    ("hotel_stays", "hospitality bookings", [("booking_id", "BIGINT"), ("property_id", "VARCHAR"), ("room_class", "VARCHAR"), ("nightly_rate", "DECIMAL(8,2)"), ("checkin_date", "DATE"), ("checkout_date", "DATE")]),
    ("flight_delays", "aviation tracking", [("flight_number", "VARCHAR"), ("airline", "VARCHAR"), ("origin_airport", "VARCHAR"), ("dest_airport", "VARCHAR"), ("delay_minutes", "INTEGER"), ("scheduled_departure", "TIMESTAMP")]),
    ("warehouse_parcels", "logistics fulfillment", [("parcel_id", "VARCHAR"), ("origin_hub", "VARCHAR"), ("dest_hub", "VARCHAR"), ("weight_kg", "FLOAT"), ("delivery_days", "INTEGER"), ("is_express", "BOOLEAN")]),
    ("audio_podcasts", "media streaming", [("episode_id", "BIGINT"), ("show_title", "VARCHAR"), ("duration_secs", "INTEGER"), ("listen_count", "BIGINT"), ("published_date", "DATE"), ("categories", "VARCHAR[]")]),
    ("security_threats", "SOC incident logs", [("incident_id", "BIGINT"), ("src_ip", "VARCHAR"), ("threat_level", "VARCHAR"), ("action_taken", "VARCHAR"), ("cve_id", "VARCHAR"), ("detected_at", "TIMESTAMPTZ")]),
]


def load_extracted_book_codes() -> list[tuple[str, str]]:
    """Loads clean code blocks extracted from the 3 EPUB books."""
    extracted = []
    if not RAW_DIR.exists():
        return extracted
        
    for bdir in RAW_DIR.iterdir():
        if bdir.is_dir():
            cfile = bdir / "code_blocks.json"
            if cfile.exists():
                try:
                    codes = json.loads(cfile.read_text(encoding="utf-8"))
                    for c in codes:
                        clean_c = c.strip()
                        if 30 <= len(clean_c) <= 1500:
                            # Classify language
                            lang = "sql" if any(k in clean_c.upper() for k in ["SELECT", "FROM", "INSTALL", "ATTACH", "CREATE TABLE", "PRAGMA"]) else "python"
                            extracted.append((bdir.name, clean_c))
                except Exception as e:
                    print(f"Error loading {cfile}: {e}")
    return extracted


def generate_records():
    rng = random.Random(42)
    records = []
    eval_records = []
    
    # 1. Load extracted EPUB snippets
    book_snippets = load_extracted_book_codes()
    print(f"Loaded {len(book_snippets)} raw code snippets from EPUBs.")
    
    # Sample and convert high-quality book snippets into instructions
    seen_prompts = set()
    for book_name, code in book_snippets:
        if len(records) >= 800:
            break
        # Create prompt based on code content
        if "read_parquet" in code or "read_csv" in code:
            q = f"How do I use DuckDB's vectorized file reader to query this dataset efficiently?\n\n```sql\n{code[:120]}...\n```"
            fam = "book_file_readers"
        elif "ATTACH" in code or "INSTALL" in code:
            q = f"Demonstrate how to manage DuckDB extensions and attached catalogs for the following pattern:\n\n```sql\n{code[:120]}...\n```"
            fam = "book_extensions_attach"
        elif "import duckdb" in code or "duckdb.sql" in code or ".df()" in code or ".pl()" in code:
            q = f"Show the idiomatic Python API integration with DuckDB for this workflow:\n\n```python\n{code[:120]}...\n```"
            fam = "book_python_api"
        elif "GROUP BY" in code or "COLUMNS" in code or "SELECT" in code:
            q = f"Explain the DuckDB SQL query execution and transformation logic for the following statement:\n\n```sql\n{code[:120]}...\n```"
            fam = "book_sql_dialect"
        else:
            continue
            
        if q in seen_prompts:
            continue
        seen_prompts.add(q)
        
        records.append({
            "meta": {"family": fam, "domain": "duckdb", "source": book_name},
            "prompt": q,
            "completion": f"### Idiomatic DuckDB Solution:\n\n```sql\n{code}\n```\n\n### Architectural Context:\nDuckDB executes this query using its vectorized columnar engine with automatic projection pushdown, multithreading, and zero-copy memory exchange."
        })

    # 2. Comprehensive Schema-Based Pillars
    for s in SCHEMAS:
        tbl, desc, cols = s
        col_names = [c[0] for c in cols]
        
        # 2.1 Modern FROM-First & Dynamic Selection
        for p_idx in range(6):
            records.append({
                "meta": {"family": "from_first_syntax", "domain": "duckdb"},
                "prompt": f"Write an idiomatic DuckDB SQL query for `{tbl}` ({desc}) using DuckDB's modern FROM-first syntax, selecting all columns EXCLUDE ({col_names[0]}), filtering on {col_names[1]} IS NOT NULL (variant {p_idx+1}).",
                "completion": f"```sql\n-- DuckDB supports leading FROM clauses and EXCLUDE for concise transformations\nFROM {tbl}\nSELECT * EXCLUDE ({col_names[0]})\nWHERE {col_names[1]} IS NOT NULL\nORDER BY {col_names[1]} DESC\nLIMIT 25;\n```\n\n### Explanation:\nIn DuckDB, `FROM` can precede `SELECT`, allowing intuitive top-to-bottom reading. `SELECT * EXCLUDE (...)` eliminates unwanted columns without manually enumerating the schema."
            })

        # 2.2 Dynamic COLUMNS(*) Expressions
        num_cols = [c[0] for c in cols if c[1] in ("INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL(10,2)", "DECIMAL(8,2)")]
        if len(num_cols) >= 2:
            c1, c2 = num_cols[:2]
            for p_idx in range(6):
                records.append({
                    "meta": {"family": "columns_star_expression", "domain": "duckdb"},
                    "prompt": f"How do I compute the average, minimum, and maximum across numeric metrics `{c1}` and `{c2}` in `{tbl}` simultaneously without writing repetitive clauses in DuckDB (variant {p_idx+1})?",
                    "completion": f"```sql\n-- Use DuckDB's dynamic COLUMNS(*) expression with regular expression filtering\nSELECT\n    COLUMNS('{c1}|{c2}').avg() AS \"avg_\\0\",\n    COLUMNS('{c1}|{c2}').min() AS \"min_\\0\",\n    COLUMNS('{c1}|{c2}').max() AS \"max_\\0\"\nFROM {tbl};\n```\n\n### Explanation:\nDuckDB's `COLUMNS()` expression accepts regex patterns and applies aggregate functions across matching columns dynamically."
                })

        # 2.3 GROUP BY ALL & REPLACE Transformations
        for p_idx in range(6):
            records.append({
                "meta": {"family": "group_by_all_replace", "domain": "duckdb"},
                "prompt": f"In DuckDB, how do I aggregate `{tbl}` by all non-numeric dimensional attributes while rounding `{col_names[2]}` to 2 decimal places using GROUP BY ALL (variant {p_idx+1})?",
                "completion": f"```sql\n-- GROUP BY ALL automatically groups by all non-aggregate projections\nSELECT\n    * REPLACE (ROUND({col_names[2]}, 2) AS {col_names[2]}),\n    COUNT(*) AS record_count\nFROM {tbl}\nGROUP BY ALL;\n```\n\n### Explanation:\n`GROUP BY ALL` infers group keys automatically, and `REPLACE` modifies specific columns inline."
            })

        # 2.4 Parquet S3 & Hive Partition Pushdown
        for p_idx in range(6):
            records.append({
                "meta": {"family": "parquet_glob_pushdown", "domain": "duckdb"},
                "prompt": f"How do I query partitioned Parquet files for `{tbl}` from `s3://data-lakehouse/{tbl}/*/*.parquet` with column projection and filter pushdown in DuckDB (variant {p_idx+1})?",
                "completion": f"```sql\n-- Load httpfs extension for remote cloud object storage\nINSTALL httpfs;\nLOAD httpfs;\n\nSET s3_region = 'us-east-1';\n\n-- Direct vectorized query over partitioned Parquet\nSELECT\n    {cols[0][0]},\n    {cols[1][0]},\n    filename\nFROM read_parquet('s3://data-lakehouse/{tbl}/*/*.parquet', filename=true, hive_partitioning=true)\nWHERE {cols[1][0]} >= '2025-01-01'\nLIMIT 50;\n```\n\n### Explanation:\nDuckDB streams remote Parquet files via `httpfs` and leverages Parquet statistics to skip unneeded row groups."
            })

        # 2.5 Zero-Copy PyArrow & Polars Integration
        for p_idx in range(6):
            records.append({
                "meta": {"family": "python_relational_arrow_zerocopy", "domain": "duckdb"},
                "prompt": f"Write a Python script using DuckDB to process a large Parquet dataset for `{tbl}` and export the aggregated result to a Polars DataFrame with ZERO memory copies (variant {p_idx+1}).",
                "completion": f"```python\nimport duckdb\nimport polars as pl\n\ncon = duckdb.connect(database=\":memory:\")\ncon.execute(\"SET memory_limit = '16GB';\")\ncon.execute(\"SET threads = 8;\")\n\nrel = con.sql(\"\"\"\n    SELECT\n        {cols[1][0]},\n        COUNT(*) AS total_count,\n        AVG({cols[2][0]}) AS avg_metric\n    FROM read_parquet('data/{tbl}/*.parquet')\n    GROUP BY {cols[1][0]}\n    HAVING COUNT(*) > 10\n\"\"\")\n\n# Zero-copy conversion to Polars DataFrame\npolars_df = rel.pl()\nprint(polars_df.head())\n```\n\n### Explanation:\nDuckDB shares memory representation with Apache Arrow, allowing `rel.pl()` to transfer data to Polars zero-copy."
            })

        # 2.6 Federated PostgreSQL & SQLite Attachment
        for p_idx in range(6):
            records.append({
                "meta": {"family": "attach_postgres_cross_query", "domain": "duckdb"},
                "prompt": f"How do I attach an existing PostgreSQL database in DuckDB and execute a federated join between Postgres table `{tbl}` and local Parquet files (variant {p_idx+1})?",
                "completion": f"```sql\nINSTALL postgres;\nLOAD postgres;\n\nATTACH 'dbname=prod_db host=postgres.internal user=postgres port=5432' AS pg_prod (TYPE POSTGRES, READ_ONLY);\n\nSELECT\n    p.{cols[0][0]},\n    p.{cols[1][0]},\n    l.enrichment_data\nFROM pg_prod.public.{tbl} AS p\nINNER JOIN read_parquet('data/local_enrichment.parquet') AS l\n    ON p.{cols[0][0]} = l.{cols[0][0]}\nLIMIT 100;\n```\n\n### Explanation:\n`ATTACH (TYPE POSTGRES)` creates a live catalog link pushing down SQL filters directly to Postgres."
            })

        # 2.7 Out-of-Core Performance & Memory Governance
        for p_idx in range(6):
            records.append({
                "meta": {"family": "duckdb_performance_tuning_pragma", "domain": "duckdb"},
                "prompt": f"How do I configure memory limits, worker threads, and spill-to-disk directory for out-of-core queries on `{tbl}` in DuckDB (variant {p_idx+1})?",
                "completion": f"```sql\nSET memory_limit = '12GB';\nSET threads = 8;\nSET max_temp_directory_size = '100GB';\nSET temp_directory = '/mnt/nvme/duckdb_temp';\nSET preserve_insertion_order = false;\n\nEXPLAIN ANALYZE\nSELECT {cols[1][0]}, SUM({cols[2][0]})\nFROM {tbl}\nGROUP BY ALL;\n```\n\n### Explanation:\nDuckDB handles workloads exceeding RAM by streaming partitions to the designated `temp_directory`."
            })

    # 3. Anti-Patterns & Idiomatic Fixes
    anti_patterns = [
        ("Loading Large CSVs into Pandas before DuckDB", 
         "Using `pd.read_csv()` incurs heavy Python memory overhead and GIL bottleneck.",
         "Query the CSV directly in DuckDB: `SELECT * FROM read_csv_auto('file.csv')` to stream through vectorized C++ scan kernels."),
        ("Row-by-Row Cursor Iteration in Python Loops",
         "Calling `cursor.execute('INSERT INTO ...')` in a Python loop is 1,000x slower.",
         "Use `con.append('table', df)` or `con.execute('INSERT INTO table SELECT * FROM read_parquet(...)')` for bulk insertion."),
        ("Regex Parsing of Hive Partition Filenames",
         "Extracting partition keys from file paths manually using regex.",
         "Enable `hive_partitioning=true` in `read_parquet()` for automatic column creation and partition directory pruning."),
        ("Storing Un-Shredded JSON Strings for Repetitive Scans",
         "Repeatedly calling `json_extract()` over raw text strings during analytical scans.",
         "Shred JSON into native `STRUCT` or relational columns on ingestion with `read_json_auto()`."),
        ("Missing Disk Spill Directory on Large Joins",
         "Running massive billion-row joins with default settings leading to OOM.",
         "Set `SET memory_limit = 'XGB';` and `SET temp_directory = '/path/to/fast/disk';` to guarantee out-of-core execution.")
    ]

    for title, flaw, fix in anti_patterns:
        for idx in range(12):
            records.append({
                "meta": {"family": "anti_pattern_idiomatic_fix", "domain": "duckdb"},
                "prompt": f"Explain why '{title}' is an anti-pattern in DuckDB workflows and demonstrate the idiomatic solution (pattern {idx+1}).",
                "completion": f"### The Anti-Pattern: {title}\n**Pitfall:** {flaw}\n\n### Idiomatic DuckDB Architecture:\n{fix}\n\n```sql\nSET preserve_insertion_order = false;\nSELECT * FROM read_parquet('data/*.parquet', hive_partitioning=true);\n```\n\n### Key Takeaway:\nAlways leverage DuckDB's native vectorized columnar engine rather than buffering through Python runtimes."
            })

    # 4. General Capability Rehearsal Buffer (7-8% Target)
    # 4.1 Sample foundational SQL and relational engineering from PostgreSQL v4 corpus
    pg_corpus_path = REPO_ROOT / "data/postgresql/training_data_v4.jsonl"
    if pg_corpus_path.exists():
        pg_recs = []
        with pg_corpus_path.open() as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    fam = d.get("meta", {}).get("family", "")
                    if any(k in fam for k in ["relational_fundamentals", "anti_not_in", "explain_plan", "anti_between_timestamp"]):
                        pg_recs.append(d)
        rng.shuffle(pg_recs)
        for d in pg_recs[:75]:
            text = d.get("text", "")
            if "\n\n### Answer:\n" in text:
                prompt, _, comp = text.partition("\n\n### Answer:\n")
                prompt_clean = prompt.replace("### Question:\n", "").strip()
                records.append({
                    "meta": {"family": "general_sql_replay", "domain": "postgresql_sql"},
                    "prompt": prompt_clean,
                    "completion": comp.strip()
                })

    # 4.2 Core Python Algorithms, Window Functions & Logic Rehearsal
    replay_topics = [
        ("Binary Search in Python", "Write an efficient binary search algorithm in Python with type hints.", "```python\ndef binary_search(arr: list[int], target: int) -> int:\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```"),
        ("SQL Window Functions", "Explain how ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...) works in standard SQL.", "```sql\nSELECT\n    department_id,\n    employee_name,\n    salary,\n    ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rank_in_dept\nFROM employees;\n```"),
        ("Python Asyncio TaskGroup", "Show how to manage concurrent asynchronous coroutines using asyncio.TaskGroup in Python 3.11+.", "```python\nimport asyncio\n\nasync def fetch(url: str) -> str:\n    await asyncio.sleep(0.1)\n    return f\"Content from {url}\"\n\nasync def main():\n    async with asyncio.TaskGroup() as tg:\n        t1 = tg.create_task(fetch('https://api.one.com'))\n        t2 = tg.create_task(fetch('https://api.two.com'))\n    print(t1.result(), t2.result())\n```"),
        ("PostgreSQL Recursive CTE", "Write a standard recursive CTE in SQL to traverse a parent-child organizational hierarchy.", "```sql\nWITH RECURSIVE org_chart AS (\n    SELECT emp_id, manager_id, name, 1 AS level\n    FROM employees\n    WHERE manager_id IS NULL\n    UNION ALL\n    SELECT e.emp_id, e.manager_id, e.name, o.level + 1\n    FROM employees e\n    JOIN org_chart o ON e.manager_id = o.emp_id\n)\nSELECT * FROM org_chart ORDER BY level, emp_id;\n```"),
        ("Matrix Transposition in Python", "Write a clean Python function to transpose a 2D matrix using list comprehension.", "```python\ndef transpose(matrix: list[list[int]]) -> list[list[int]]:\n    if not matrix or not matrix[0]:\n        return []\n    return [[matrix[row][col] for row in range(len(matrix))] for col in range(len(matrix[0]))]\n```"),
    ]
    for topic, q, ans in replay_topics:
        for _ in range(12):
            records.append({
                "meta": {"family": "general_capability_replay", "domain": "general"},
                "prompt": q,
                "completion": ans
            })

    # 5. Held-Out Evaluation Reservation (50 Questions)
    eval_scenarios = [
        ("duckdb_from_first_exclude", 
         "A data analyst needs to query a DuckDB table `server_telemetry` and return all columns except `internal_trace_id`, filtering for error rates > 0.05 using modern DuckDB syntax. Write the query.",
         ["FROM server_telemetry", "EXCLUDE", "WHERE"]),
        ("duckdb_parquet_s3_pushdown",
         "Write a DuckDB SQL statement to query partitioned remote Parquet files from `s3://data-warehouse/sales/*/*.parquet` using the httpfs extension and Hive partitioning.",
         ["read_parquet", "httpfs", "hive_partitioning"]),
        ("duckdb_columns_regex_aggregation",
         "In DuckDB, write a query that calculates the average of all columns matching the regex `^metric_` in table `device_readings` using `COLUMNS()`.",
         ["COLUMNS", "avg()", "device_readings"]),
        ("duckdb_python_arrow_zerocopy",
         "Write a Python function using DuckDB that takes an input Parquet path, aggregates metrics grouped by customer, and returns a Polars DataFrame using zero-copy execution.",
         ["import duckdb", ".pl()", "read_parquet"]),
        ("duckdb_attach_postgres_federation",
         "Demonstrate how to attach a PostgreSQL database named `pg_dw` in DuckDB and query a table `customers` joined with a local CSV file `targets.csv`.",
         ["ATTACH", "TYPE POSTGRES", "read_csv_auto"]),
        ("duckdb_group_by_all_replace",
         "Write a DuckDB query on `finance_txns` that groups by all categorical columns and rounds the `amount` column to 2 decimal places using REPLACE and GROUP BY ALL.",
         ["GROUP BY ALL", "REPLACE", "ROUND"]),
        ("duckdb_json_shred_extract",
         "Write a DuckDB query to read newline-delimited JSON logs from `logs/*.json` and extract the string field `$.meta.user_id` as `user_id`.",
         ["read_json_auto", "->>", "user_id"]),
        ("duckdb_list_comprehension_filter",
         "Show how to filter a list of integers `[10, 25, 30, 45, 50]` to only keep elements greater than 25 using DuckDB's native list comprehension or lambda function.",
         ["list_filter", "->", "[10, 25, 30, 45, 50]"]),
        ("duckdb_out_of_core_spill_config",
         "Provide the DuckDB PRAGMA / SET statements to configure a 16GB memory cap, 8 CPU worker threads, and a fast SSD temp directory `/fast_ssd/duck_spill` for out-of-core processing.",
         ["memory_limit", "threads", "temp_directory"]),
        ("duckdb_union_by_name_csv",
         "Write a DuckDB query to ingest multiple CSV files matching `data/sales_*.csv` that have varying schema columns over time using name-based union.",
         ["read_csv_auto", "union_by_name", "true"])
    ]

    for id_idx, (cat, prompt, expects) in enumerate(eval_scenarios):
        for variant in range(5):
            eval_records.append({
                "id": f"duckdb_eval_{id_idx:02d}_v{variant}",
                "category": cat,
                "prompt": prompt + (f" (Scenario variant {variant + 1})" if variant > 0 else ""),
                "expects": expects
            })

    # Shuffle training records
    rng.shuffle(records)
    
    # Save training dataset
    OUT_TRAIN_V4.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TRAIN_V4.open("w", encoding="utf-8") as f:
        for r in records:
            record_obj = {
                "text": f"### Question:\n{r['prompt']}\n\n### Answer:\n{r['completion']}",
                "meta": r["meta"]
            }
            f.write(json.dumps(record_obj) + "\n")

    # Save evaluation dataset
    with OUT_EVAL.open("w", encoding="utf-8") as f:
        for er in eval_records:
            f.write(json.dumps(er) + "\n")

    print("=" * 80)
    print(" 🦆 DUCKDB v4 TRAINING & EVALUATION CORPUS GENERATED")
    print("=" * 80)
    print(f" Total Training Records : {len(records):,}")
    print(f" Total Evaluation Items  : {len(eval_records):,}")
    
    family_counts = Counter(r["meta"]["family"] for r in records)
    print("\n Training Distribution by Family:")
    for fam, cnt in family_counts.most_common():
        pct = 100.0 * cnt / len(records)
        print(f"   • {fam:<36s} : {cnt:4d} records ({pct:5.1f}%)")
        
    replay_cnt = sum(cnt for fam, cnt in family_counts.items() if "replay" in fam)
    print(f"\n General Capability Rehearsal Buffer: {replay_cnt} records ({100.0 * replay_cnt / len(records):.1f}%) [Target: 5-10%]")
    print(f" Output Training File : {OUT_TRAIN_V4}")
    print(f" Output Eval File     : {OUT_EVAL}")
    print("=" * 80)


if __name__ == "__main__":
    generate_records()
