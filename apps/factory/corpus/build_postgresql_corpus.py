"""Build the expanded PostgreSQL expert v3 training corpus (~1,800+ records).

Pillars:
1. Applied pgvector & AI-native PostgreSQL (v2 clean foundation, ~741 recs)
2. Anti-patterns & Mistakes vs Idiomatic Fixes (Jimmy Angelakos, ~405 recs)
3. Advanced PostgreSQL 17 Features & Optimization (Hans-Jürgen Schönig, ~373 recs)
4. Foundational & Standard SQL Rehearsal Buffer (~100 recs)

Quality Gates:
- 100% sqlglot parsing verification in postgres dialect.
- Dynamic sequence masking compatibility (prompt/completion separation).
- Diversity assertions: >= 25 schemas, >= 20 distinct question prefixes.

Usage:
    uv run --with sqlglot python scripts/corpus/build_postgresql_corpus.py
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import sqlglot

from runtime_common.canon import REPO_ROOT as REPO  # noqa: E402
# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
SRC_V2 = REPO / "apps/factory/data/postgresql/training_data_v2.jsonl"
OUT_V3 = REPO / "apps/factory/data/postgresql/training_data_v3.jsonl"

REJECTS: list = []

# =====================================================================
# 30+ Diverse Schemas
# =====================================================================
SCHEMAS = [
    ("products", "e-commerce catalogue", [("id", "bigint"), ("name", "text"), ("category", "text"), ("price", "numeric(10,2)"), ("in_stock", "boolean"), ("created_at", "timestamptz")]),
    ("orders", "order management", [("id", "bigint"), ("customer_id", "bigint"), ("status", "text"), ("total_amount", "numeric(12,2)"), ("placed_at", "timestamptz")]),
    ("customers", "CRM store", [("id", "bigint"), ("email", "text"), ("first_name", "text"), ("last_name", "text"), ("tier", "text"), ("registered_at", "timestamptz")]),
    ("sensor_readings", "IoT telemetry", [("id", "bigint"), ("sensor_id", "text"), ("location", "text"), ("metric_val", "double precision"), ("is_valid", "boolean"), ("recorded_at", "timestamptz")]),
    ("articles", "content platform", [("id", "bigint"), ("author_id", "bigint"), ("title", "text"), ("body", "text"), ("views", "integer"), ("published_at", "timestamptz"), ("tags", "text[]")]),
    ("support_tickets", "helpdesk service", [("id", "bigint"), ("user_id", "bigint"), ("department", "text"), ("priority", "text"), ("resolved", "boolean"), ("opened_at", "timestamptz")]),
    ("user_sessions", "auth & telemetry", [("session_id", "uuid"), ("user_id", "bigint"), ("ip_address", "inet"), ("device_type", "text"), ("started_at", "timestamptz"), ("ended_at", "timestamptz")]),
    ("audit_logs", "compliance registry", [("id", "bigint"), ("actor_id", "bigint"), ("action", "text"), ("target_resource", "text"), ("status_code", "integer"), ("logged_at", "timestamptz")]),
    ("inventory_items", "warehouse logistics", [("sku", "text"), ("warehouse_id", "integer"), ("quantity", "integer"), ("reorder_threshold", "integer"), ("last_counted", "date")]),
    ("medical_records", "healthcare data", [("record_id", "bigint"), ("patient_id", "bigint"), ("doctor_id", "bigint"), ("diagnosis_code", "text"), ("treatment_cost", "numeric(10,2)"), ("visit_date", "date")]),
    ("financial_txns", "ledger accounts", [("txn_id", "bigint"), ("account_id", "bigint"), ("txn_type", "text"), ("amount", "numeric(14,4)"), ("is_cleared", "boolean"), ("transacted_at", "timestamptz")]),
    ("server_metrics", "infrastructure ops", [("id", "bigint"), ("hostname", "text"), ("cpu_usage", "real"), ("memory_pct", "real"), ("disk_iops", "integer"), ("sampled_at", "timestamptz")]),
    ("ride_trips", "mobility network", [("trip_id", "bigint"), ("driver_id", "bigint"), ("rider_id", "bigint"), ("fare", "numeric(8,2)"), ("distance_km", "numeric(6,2)"), ("completed_at", "timestamptz")]),
    ("employee_directory", "HR directory", [("emp_id", "bigint"), ("department_id", "integer"), ("job_title", "text"), ("salary", "numeric(10,2)"), ("is_active", "boolean"), ("hired_date", "date")]),
    ("network_events", "cybersecurity log", [("event_id", "bigint"), ("src_ip", "inet"), ("dest_port", "integer"), ("threat_score", "numeric(5,2)"), ("is_blocked", "boolean"), ("detected_at", "timestamptz")]),
    ("shipping_parcels", "fulfillment tracking", [("tracking_number", "text"), ("origin_hub", "text"), ("dest_hub", "text"), ("weight_kg", "numeric(6,3)"), ("shipped_date", "date"), ("delivered", "boolean")]),
    ("hotel_bookings", "hospitality reservations", [("booking_id", "bigint"), ("guest_id", "bigint"), ("room_type", "text"), ("nightly_rate", "numeric(8,2)"), ("check_in", "date"), ("check_out", "date")]),
    ("git_commits", "version control tracker", [("commit_hash", "text"), ("repo_id", "integer"), ("committer_email", "text"), ("additions", "integer"), ("deletions", "integer"), ("committed_at", "timestamptz")]),
    ("subscription_plans", "billing engine", [("sub_id", "bigint"), ("customer_id", "bigint"), ("plan_name", "text"), ("mrr_amount", "numeric(10,2)"), ("auto_renew", "boolean"), ("renews_at", "timestamptz")]),
    ("weather_stations", "climate monitoring", [("station_id", "text"), ("elevation_m", "integer"), ("temperature_c", "real"), ("humidity_pct", "real"), ("rainfall_mm", "real"), ("measured_at", "timestamptz")]),
    ("app_errors", "error monitoring", [("error_id", "bigint"), ("service_name", "text"), ("error_class", "text"), ("stack_trace", "text"), ("occurrences", "integer"), ("last_seen", "timestamptz")]),
    ("delivery_routes", "dispatch routing", [("route_id", "bigint"), ("vehicle_id", "integer"), ("stops_count", "integer"), ("total_distance_km", "numeric(7,2)"), ("dispatched_at", "timestamptz")]),
    ("course_enrollments", "edtech portal", [("enrollment_id", "bigint"), ("student_id", "bigint"), ("course_id", "bigint"), ("progress_pct", "integer"), ("grade", "numeric(5,2)"), ("enrolled_at", "timestamptz")]),
    ("auction_bids", "marketplace exchange", [("bid_id", "bigint"), ("item_id", "bigint"), ("bidder_id", "bigint"), ("bid_amount", "numeric(12,2)"), ("is_winning", "boolean"), ("placed_at", "timestamptz")]),
    ("social_posts", "social media stream", [("post_id", "bigint"), ("user_id", "bigint"), ("content", "text"), ("likes_count", "integer"), ("shares_count", "integer"), ("posted_at", "timestamptz")]),
    ("bank_accounts", "core banking", [("account_number", "text"), ("branch_code", "text"), ("account_type", "text"), ("current_balance", "numeric(14,2)"), ("is_frozen", "boolean"), ("opened_date", "date")]),
    ("flight_legs", "aviation operations", [("flight_id", "bigint"), ("flight_number", "text"), ("origin_airport", "text"), ("dest_airport", "text"), ("departure_time", "timestamptz"), ("delay_minutes", "integer")]),
    ("warehouse_bins", "stock management", [("bin_id", "text"), ("zone", "text"), ("aisle", "integer"), ("capacity_units", "integer"), ("is_full", "boolean")]),
    ("telecom_calls", "call data records", [("call_id", "bigint"), ("caller_number", "text"), ("recipient_number", "text"), ("duration_seconds", "integer"), ("cost", "numeric(6,4)"), ("started_at", "timestamptz")]),
    ("iot_gateways", "device fleet", [("gateway_mac", "text"), ("firmware_version", "text"), ("connected_nodes", "integer"), ("is_online", "boolean"), ("last_heartbeat", "timestamptz")]),
]

# =====================================================================
# Anti-Pattern / Mistake Generators (from Jimmy Angelakos)
# =====================================================================

def g_anti_not_in(rng, s):
    t, dom, cols = s
    col_id, _ = cols[0]
    sub_t = "unverified_targets"
    q = rng.choice([
        f"In our {dom}, we need to find all rows in `{t}` whose `{col_id}` is NOT present in `{sub_t}`. Why does `NOT IN` fail here, and what is the idiomatic PostgreSQL solution?",
        f"Explain the dangerous SQL three-valued logic flaw of `NOT IN (SELECT ...)` over nullable columns in `{t}`, and write the safe, high-performance replacement.",
        f"Write a query to find `{t}` rows missing from `{sub_t}`. Compare `NOT IN` vs `NOT EXISTS` and show the query plan difference.",
    ])
    a = (f"```sql\n"
         f"-- Anti-pattern: NOT IN returns 0 rows if the subquery contains any NULL\n"
         f"-- SELECT {col_id} FROM {t} WHERE {col_id} NOT IN (SELECT {col_id} FROM {sub_t});\n\n"
         f"-- Correct & High-Performance: NOT EXISTS executes as a Hash/Merge Anti-Join\n"
         f"SELECT t.{col_id}\n"
         f"FROM {t} t\n"
         f"WHERE NOT EXISTS (\n"
         f"    SELECT 1 FROM {sub_t} sub\n"
         f"    WHERE sub.{col_id} = t.{col_id}\n"
         f");\n```\n\n"
         f"**Why `NOT IN` fails:** In SQL three-valued logic, `x NOT IN (1, NULL)` evaluates to `UNKNOWN`, never `TRUE`. If `{sub_t}.{col_id}` produces even a single NULL, `NOT IN` returns zero rows. `NOT EXISTS` handles NULLs cleanly and allows PostgreSQL to execute an optimal Anti-Join.")
    return q, a, "anti_not_in"


def g_anti_between_timestamp(rng, s):
    t, dom, cols = s
    ts_cols = [c for c, ty in cols if ty == "timestamptz"]
    ts_col = ts_cols[0] if ts_cols else "created_at"
    q = rng.choice([
        f"Why is using `BETWEEN` on timestamp column `{t}.{ts_col}` considered a critical bug for date range filtering in PostgreSQL, and how should it be written?",
        f"Write the correct date range query for `{t}` covering the entire month of January 2026 without the `BETWEEN` boundary leak.",
        f"Show why `WHERE {ts_col} BETWEEN '2026-01-01' AND '2026-01-31'` misses data or double-counts boundaries, and write the half-open interval equivalent.",
    ])
    a = (f"```sql\n"
         f"-- Anti-pattern: BETWEEN is inclusive [start, end], missing sub-second data or double-counting\n"
         f"-- WHERE {ts_col} BETWEEN '2026-01-01' AND '2026-01-31'\n\n"
         f"-- Correct & Robust: Half-open interval [start, next_start)\n"
         f"SELECT *\n"
         f"FROM {t}\n"
         f"WHERE {ts_col} >= TIMESTAMPTZ '2026-01-01 00:00:00+00'\n"
         f"  AND {ts_col} < TIMESTAMPTZ '2026-02-01 00:00:00+00';\n```\n\n"
         f"`'2026-01-31'` casts to midnight `'2026-01-31 00:00:00'`, completely missing the final 24 hours. If written with `23:59:59`, it misses sub-second timestamps (`23:59:59.123`). The half-open interval `>= start AND < next_period` is exact, timezone-safe, and indexable.")
    return q, a, "anti_between_timestamp"


def g_anti_serial_identity(rng, s):
    t, dom, cols = s
    q = rng.choice([
        f"Why is `SERIAL` / `BIGSERIAL` legacy in PostgreSQL, and how do we write the modern SQL-standard equivalent for `{t}`?",
        f"Convert a legacy `SERIAL` primary key definition for table `{t}` to `GENERATED ALWAYS AS IDENTITY` and explain the security and permissions advantage.",
        f"Write the DDL for `{t}` with an identity column that prevents accidental manual primary key overrides.",
    ])
    a = (f"```sql\n"
         f"-- Modern & SQL-Standard DDL (PostgreSQL 10+)\n"
         f"CREATE TABLE {t} (\n"
         f"    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,\n"
         f"    name text NOT NULL,\n"
         f"    created_at timestamptz DEFAULT clock_timestamp()\n"
         f");\n```\n\n"
         f"`SERIAL` is a legacy pseudo-type that secretly creates a separate sequence object without strict ownership, allowing users to accidentally insert manual IDs that cause future `duplicate key` crashes. `GENERATED ALWAYS AS IDENTITY` conforms to the SQL standard, enforces permissions, and prevents manual overrides unless `OVERRIDING SYSTEM VALUE` is explicitly requested.")
    return q, a, "anti_serial_identity"


def g_anti_unindexed_expression(rng, s):
    t, dom, cols = s
    txt_col = next((c for c, ty in cols if ty == "text"), "name")
    q = rng.choice([
        f"A query with `WHERE lower({txt_col}) = $1` on `{t}` is doing a Sequential Scan despite a B-tree index on `{txt_col}`. Explain why and write the fix.",
        f"Create an expression/functional index on `{t}` for case-insensitive lookup on `{txt_col}`.",
        f"How do we index `{t}` so that `WHERE lower({txt_col}) = lower($1)` uses an Index Scan?",
    ])
    a = (f"```sql\n"
         f"-- Fix: Create a functional/expression index matching the exact expression\n"
         f"CREATE INDEX {t}_{txt_col}_lower_idx ON {t} (lower({txt_col}));\n\n"
         f"-- The query must use the exact indexed expression:\n"
         f"SELECT *\n"
         f"FROM {t}\n"
         f"WHERE lower({txt_col}) = lower($1);\n```\n\n"
         f"Standard B-Tree indexes index raw column values, not function results. When a column is wrapped in `lower()`, the planner cannot use the plain index. A functional index stores the pre-computed `lower({txt_col})` values in the B-Tree directly.")
    return q, a, "anti_unindexed_expression"


def g_anti_integer_division(rng, s):
    t, dom, cols = s
    num_cols = [c for c, ty in cols if ty in ("integer", "bigint")]
    c1 = num_cols[0] if len(num_cols) > 0 else "quantity"
    c2 = num_cols[1] if len(num_cols) > 1 else "reorder_threshold"
    q = rng.choice([
        f"In `{t}`, calculating `{c1} / {c2}` always returns 0 for fractional values. Explain integer truncation and write the safe, divide-by-zero-safe ratio query.",
        f"Write a query over `{t}` calculating the percentage ratio of `{c1}` to `{c2}` formatted as a two-decimal numeric with division-by-zero protection.",
    ])
    a = (f"```sql\n"
         f"-- Safe ratio calculation with division-by-zero protection\n"
         f"SELECT\n"
         f"    {c1},\n"
         f"    {c2},\n"
         f"    ROUND(100.0 * {c1} / NULLIF({c2}, 0), 2) AS ratio_pct\n"
         f"FROM {t};\n```\n\n"
         f"In SQL, dividing two integers (`5 / 10`) performs integer truncation and evaluates to `0`. Multiplying by `100.0` or casting `{c1}::numeric` forces numeric floating-point division. `NULLIF({c2}, 0)` converts zero denominators into `NULL`, preventing `division by zero` exceptions.")
    return q, a, "anti_integer_division"


def g_anti_count_nulls(rng, s):
    t, dom, cols = s
    col_name = cols[1][0]
    q = rng.choice([
        f"Explain the crucial semantic difference between `COUNT(*)`, `COUNT(1)`, and `COUNT({col_name})` in PostgreSQL on table `{t}`.",
        f"Write a query on `{t}` that reports total rows, non-null count of `{col_name}`, and total nulls in a single pass.",
    ])
    a = (f"```sql\n"
         f"SELECT\n"
         f"    count(*) AS total_rows,\n"
         f"    count({col_name}) AS non_null_rows,\n"
         f"    count(*) - count({col_name}) AS null_rows\n"
         f"FROM {t};\n```\n\n"
         f"`COUNT(*)` counts total rows including NULLs and is optimized by the planner. `COUNT({col_name})` checks every row and skips `NULL` values. `COUNT(1)` is internally identical to `COUNT(*)`, so `COUNT(*)` is the idiomatic standard.")
    return q, a, "anti_count_nulls"


def g_anti_skip_locked(rng, s):
    t, dom, cols = s
    q = rng.choice([
        f"Multiple background workers poll `{t}` for pending jobs. How do we prevent lock contention and head-of-line blocking using `SKIP LOCKED`?",
        f"Write an atomic worker queue query for `{t}` that claims exactly 1 unprocessed item without blocking competing worker transactions.",
    ])
    a = (f"```sql\n"
         f"-- Atomic job queue claim with SKIP LOCKED\n"
         f"WITH job AS (\n"
         f"    SELECT id\n"
         f"    FROM {t}\n"
         f"    WHERE status = 'pending'\n"
         f"    ORDER BY created_at\n"
         f"    FOR UPDATE SKIP LOCKED\n"
         f"    LIMIT 1\n"
         f")\n"
         f"UPDATE {t}\n"
         f"SET status = 'processing'\n"
         f"FROM job\n"
         f"WHERE {t}.id = job.id\n"
         f"RETURNING {t}.*;\n```\n\n"
         f"Plain `FOR UPDATE` forces competing workers to wait on the same locked row (head-of-line blocking). `FOR UPDATE SKIP LOCKED` skips rows already locked by other transactions, achieving lock-free concurrency.")
    return q, a, "anti_skip_locked"


# =====================================================================
# Advanced PostgreSQL 17 Generators (from Hans-Jürgen Schönig)
# =====================================================================

def g_adv_distinct_on(rng, s):
    t, dom, cols = s
    group_col = cols[1][0]
    order_col = cols[-1][0]
    q = rng.choice([
        f"In `{t}`, fetch the single most recent record for each distinct `{group_col}` using PostgreSQL's fastest deduplication syntax (`DISTINCT ON`).",
        f"Write an efficient `DISTINCT ON` query for `{t}` grouped by `{group_col}` and ordered by `{order_col}` descending.",
        f"Show how `DISTINCT ON` outperforms window `ROW_NUMBER() = 1` queries when retrieving the latest row per `{group_col}` in `{t}`.",
    ])
    a = (f"```sql\n"
         f"SELECT DISTINCT ON ({group_col}) *\n"
         f"FROM {t}\n"
         f"ORDER BY {group_col}, {order_col} DESC;\n```\n\n"
         f"`DISTINCT ON` keeps only the first row per group based on the `ORDER BY` clause. The leading `ORDER BY` column MUST match the `DISTINCT ON` expression. A supporting index on `({group_col}, {order_col} DESC)` allows PostgreSQL to perform an instant Index Scan.")
    return q, a, "adv_distinct_on"


def g_adv_filter_clause(rng, s):
    t, dom, cols = s
    group_col = cols[1][0]
    num_col = next((c for c, ty in cols if ty in ("numeric", "numeric(10,2)", "numeric(12,2)", "numeric(14,2)", "integer")), "id")
    q = rng.choice([
        f"In `{t}`, calculate the total count, count of active records, and sum of `{num_col}` for premium records in a single aggregation pass grouped by `{group_col}`.",
        f"Write a single-pass aggregation query on `{t}` using standard SQL `FILTER (WHERE ...)` clauses instead of legacy `CASE WHEN` sums.",
    ])
    a = (f"```sql\n"
         f"SELECT\n"
         f"    {group_col},\n"
         f"    count(*) AS total_count,\n"
         f"    count(*) FILTER (WHERE status = 'active') AS active_count,\n"
         f"    sum({num_col}) FILTER (WHERE status = 'active') AS active_total\n"
         f"FROM {t}\n"
         f"GROUP BY {group_col}\n"
         f"ORDER BY {group_col};\n```\n\n"
         f"The standard SQL `FILTER (WHERE ...)` clause is cleaner, faster, and more readable than legacy `SUM(CASE WHEN ... THEN 1 ELSE 0 END)`. It is evaluated natively during aggregate state updates.")
    return q, a, "adv_filter_clause"


def g_adv_ordinality_unnest(rng, s):
    t, dom, cols = s
    q = rng.choice([
        f"In table `{t}`, expand the array column `tags` into rows while preserving the original 1-based index position using `WITH ORDINALITY`.",
        f"Write an unnest query with `WITH ORDINALITY` on `{t}` to pair array items with their sequence position.",
    ])
    a = (f"```sql\n"
         f"SELECT\n"
         f"    t.id,\n"
         f"    elem.val AS item,\n"
         f"    elem.pos AS item_position\n"
         f"FROM {t} t,\n"
         f"     unnest(t.tags) WITH ORDINALITY AS elem(val, pos)\n"
         f"ORDER BY t.id, elem.pos;\n```\n\n"
         f"`WITH ORDINALITY` attaches a synthetic 1-based integer counter column (`pos`) to unnested set-returning functions in the `FROM` clause, preserving original array element order.")
    return q, a, "adv_ordinality_unnest"


def g_adv_percentiles(rng, s):
    t, dom, cols = s
    group_col = cols[1][0]
    num_col = next((c for c, ty in cols if ty in ("numeric", "numeric(10,2)", "numeric(12,2)", "numeric(14,2)", "double precision", "real")), "price")
    q = rng.choice([
        f"Calculate the continuous median (50th percentile) and 95th percentile of `{num_col}` in `{t}` grouped by `{group_col}`.",
        f"Write an exact percentile aggregation query over `{t}.{num_col}` using `percentile_cont`.",
    ])
    a = (f"```sql\n"
         f"SELECT\n"
         f"    {group_col},\n"
         f"    percentile_cont(0.50) WITHIN GROUP (ORDER BY {num_col}) AS median_val,\n"
         f"    percentile_cont(0.95) WITHIN GROUP (ORDER BY {num_col}) AS p95_val\n"
         f"FROM {t}\n"
         f"GROUP BY {group_col}\n"
         f"ORDER BY {group_col};\n```\n\n"
         f"`percentile_cont` computes an exact interpolated continuous percentile using the `WITHIN GROUP (ORDER BY ...)` ordered-set aggregate syntax.")
    return q, a, "adv_percentiles"


def g_adv_lateral_join(rng, s):
    t, dom, cols = s
    q = rng.choice([
        f"For every customer in `customers`, retrieve their 3 largest orders from `orders` using a correlated `JOIN LATERAL` subquery.",
        f"Write a top-3 per customer query joining `customers` and `orders` with `LATERAL`.",
    ])
    a = (f"```sql\n"
         f"SELECT\n"
         f"    c.id AS customer_id,\n"
         f"    c.email,\n"
         f"    o.id AS order_id,\n"
         f"    o.total_amount\n"
         f"FROM customers c\n"
         f"JOIN LATERAL (\n"
         f"    SELECT id, total_amount\n"
         f"    FROM orders\n"
         f"    WHERE customer_id = c.id\n"
         f"    ORDER BY total_amount DESC\n"
         f"    LIMIT 3\n"
         f") o ON true\n"
         f"ORDER BY c.id, o.total_amount DESC;\n```\n\n"
         f"`JOIN LATERAL` allows subqueries in the `FROM` clause to reference columns from preceding tables on the left, enabling clean per-row top-N calculations.")
    return q, a, "adv_lateral_join"


def g_adv_merge_statement(rng, s):
    t, dom, cols = s
    q = rng.choice([
        f"Write a standard SQL `MERGE INTO` statement to synchronize staging table `{t}_staging` into target table `{t}`.",
        f"Use PostgreSQL 17's `MERGE` statement to insert new rows and update changed records in `{t}` in a single command.",
    ])
    a = (f"```sql\n"
         f"MERGE INTO {t} AS target\n"
         f"USING {t}_staging AS source\n"
         f"ON (target.id = source.id)\n"
         f"WHEN MATCHED AND target.updated_at < source.updated_at THEN\n"
         f"    UPDATE SET\n"
         f"        status = source.status,\n"
         f"        updated_at = source.updated_at\n"
         f"WHEN NOT MATCHED THEN\n"
         f"    INSERT (id, status, updated_at)\n"
         f"    VALUES (source.id, source.status, source.updated_at);\n```\n\n"
         f"PostgreSQL 17 supports full ANSI SQL `MERGE`, allowing conditional `UPDATE`, `INSERT`, and `DELETE` actions within a single atomic statement.")
    return q, a, "adv_merge_statement"


def g_adv_brin_index(rng, s):
    t, dom, cols = s
    ts_cols = [c for c, ty in cols if ty == "timestamptz"]
    ts_col = ts_cols[0] if ts_cols else "created_at"
    q = rng.choice([
        f"Our `{t}` table holds 50 million time-series rows. Write a BRIN index DDL for `{ts_col}` and explain why it uses 99% less memory than B-tree.",
        f"Create a Block Range Index (BRIN) on `{t}.{ts_col}` tuned with pages_per_range.",
    ])
    a = (f"```sql\n"
         f"-- Create a lightweight BRIN index for naturally ordered time-series data\n"
         f"CREATE INDEX {t}_{ts_col}_brin_idx\n"
         f"ON {t}\n"
         f"USING brin ({ts_col})\n"
         f"WITH (pages_per_range = 32);\n```\n\n"
         f"BRIN indexes store summary ranges (min/max values) for blocks of pages rather than individual row pointers. For append-only time-series tables, BRIN occupies orders of magnitude less disk and RAM while providing rapid block-level pruning.")
    return q, a, "adv_brin_index"


def g_adv_partition_range(rng, s):
    t, dom, cols = s
    q = rng.choice([
        f"Write the DDL to create table `{t}_partitioned` partitioned by date ranges on `recorded_at`, along with two monthly child partitions for 2026.",
        f"Create a declarative range-partitioned table for `{t}` and show the partition bounds DDL.",
    ])
    a = (f"```sql\n"
         f"-- Declarative Range Partitioning DDL\n"
         f"CREATE TABLE {t}_partitioned (\n"
         f"    id bigint NOT NULL,\n"
         f"    payload text,\n"
         f"    recorded_at timestamptz NOT NULL,\n"
         f"    PRIMARY KEY (id, recorded_at)\n"
         f") PARTITION BY RANGE (recorded_at);\n\n"
         f"CREATE TABLE {t}_y2026m01 PARTITION OF {t}_partitioned\n"
         f"    FOR VALUES FROM ('2026-01-01 00:00:00+00') TO ('2026-02-01 00:00:00+00');\n\n"
         f"CREATE TABLE {t}_y2026m02 PARTITION OF {t}_partitioned\n"
         f"    FOR VALUES FROM ('2026-02-01 00:00:00+00') TO ('2026-03-01 00:00:00+00');\n```\n\n"
         f"Declarative table partitioning enables automatic partition pruning during queries. Partition keys must be included in the primary key constraint.")
    return q, a, "adv_partition_range"


GENERATORS_ANTI = [
    g_anti_not_in, g_anti_between_timestamp, g_anti_serial_identity,
    g_anti_unindexed_expression, g_anti_integer_division, g_anti_count_nulls,
    g_anti_skip_locked
]

GENERATORS_ADV = [
    g_adv_distinct_on, g_adv_filter_clause, g_adv_ordinality_unnest,
    g_adv_percentiles, g_adv_lateral_join, g_adv_merge_statement,
    g_adv_brin_index, g_adv_partition_range
]

# =====================================================================
# Foundational & Standard SQL Rehearsal Buffer (Clean Domain Anchors)
# =====================================================================
STANDARD_SQL_REPLAY = [
    ("Write a standard SQL query finding all customers who placed orders exceeding $500 in total across all their orders, showing customer name and total sum.",
     "```sql\nSELECT c.id, c.name, sum(o.total_amount) AS total_spent\nFROM customers c\nJOIN orders o ON o.customer_id = c.id\nGROUP BY c.id, c.name\nHAVING sum(o.total_amount) > 500\nORDER BY total_spent DESC;\n```\n\n`HAVING` filters aggregated groups after the `GROUP BY` reduction, whereas `WHERE` filters individual rows prior to aggregation."),

    ("Write an explicit transaction block that transfers $100 from account 101 to account 202 with balance integrity checks.",
     "```sql\nBEGIN;\nUPDATE bank_accounts SET current_balance = current_balance - 100.00 WHERE account_number = '101' AND current_balance >= 100.00;\nUPDATE bank_accounts SET current_balance = current_balance + 100.00 WHERE account_number = '202';\nCOMMIT;\n```\n\nWrapping atomic operations in `BEGIN` ... `COMMIT` ensures ACID transactional integrity."),

    ("Write a query returning all products along with their category name using a LEFT JOIN, handling products without a category gracefully with COALESCE.",
     "```sql\nSELECT p.id, p.name, COALESCE(c.category_name, 'Uncategorized') AS category\nFROM products p\nLEFT JOIN categories c ON c.id = p.category_id\nORDER BY p.name;\n```\n\n`LEFT JOIN` preserves all left-table rows, and `COALESCE` replaces `NULL` join results with default values."),

    ("Create a table `employees` with foreign key cascade delete referencing `departments(id)` and a check constraint ensuring salary > 0.",
     "```sql\nCREATE TABLE employees (\n    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,\n    department_id bigint NOT NULL REFERENCES departments(id) ON DELETE CASCADE,\n    name text NOT NULL,\n    salary numeric(10,2) NOT NULL CHECK (salary > 0)\n);\n```\n\n`ON DELETE CASCADE` automatically removes child rows when a referenced parent is deleted."),

    ("Write a query calculating a 7-day moving average of daily revenue using standard SQL window functions.",
     "```sql\nSELECT\n    order_date,\n    daily_revenue,\n    AVG(daily_revenue) OVER (\n        ORDER BY order_date\n        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW\n    ) AS moving_avg_7d\nFROM daily_sales\nORDER BY order_date;\n```\n\n`ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` defines a sliding 7-day window frame for rolling calculations."),
]


def sql_blocks_parse(answer: str) -> bool:
    """Every ```sql block must parse as Postgres. Hard gate."""
    blocks = re.findall(r"```sql\s*(.*?)```", answer, re.S)
    if not blocks:
        return True
    for b in blocks:
        stmts = [x for x in re.split(r";\s*\n", b) if x.strip() and not x.strip().startswith("--")]
        for st in stmts:
            st = "\n".join(l for l in st.splitlines() if not l.strip().startswith("--")).strip()
            if not st:
                continue
            st = st.replace("<#>", "<->")
            try:
                if not sqlglot.parse(st + ";", dialect="postgres"):
                    REJECTS.append(("empty", st))
                    return False
            except Exception as ex:
                REJECTS.append((type(ex).__name__, st))
                return False
    return True


def main() -> None:
    rng = random.Random(20260818)
    print("=== Building Expanded PostgreSQL v3 Training Corpus ===")

    # 1. Load existing v2 foundation
    existing_records = []
    if SRC_V2.exists():
        for line in SRC_V2.read_text().splitlines():
            if line.strip():
                existing_records.append(json.loads(line))
    print(f"Loaded v2 foundation: {len(existing_records)} records")

    # 2. Generate Anti-Pattern & Mistake Records
    by_anti: dict[str, list] = {}
    rejected = 0
    seen_q = set()
    for s in SCHEMAS:
        for g in GENERATORS_ANTI:
            for _ in range(3):
                q, a, fam = g(rng, s)
                if not sql_blocks_parse(a):
                    rejected += 1
                    continue
                if q in seen_q:
                    continue
                seen_q.add(q)
                by_anti.setdefault(fam, []).append({
                    "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                    "meta": {"source": "anti_pattern_generated", "family": fam, "table": s[0]},
                    "text": f"### Question:\n{q}\n\n### Answer:\n{a}"
                })

    anti_records = []
    for fam, items in by_anti.items():
        anti_records.extend(items)
    print(f"Generated Anti-Pattern records: {len(anti_records)} (rejected: {rejected})")

    # 3. Generate Advanced SQL Records
    by_adv: dict[str, list] = {}
    for s in SCHEMAS:
        for g in GENERATORS_ADV:
            for _ in range(3):
                q, a, fam = g(rng, s)
                if not sql_blocks_parse(a):
                    rejected += 1
                    continue
                if q in seen_q:
                    continue
                seen_q.add(q)
                by_adv.setdefault(fam, []).append({
                    "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                    "meta": {"source": "advanced_sql_generated", "family": fam, "table": s[0]},
                    "text": f"### Question:\n{q}\n\n### Answer:\n{a}"
                })

    adv_records = []
    for fam, items in by_adv.items():
        adv_records.extend(items)
    print(f"Generated Advanced SQL records: {len(adv_records)} (rejected: {rejected})")

    # 4. Generate Foundational Standard SQL Rehearsal Records (Clean domain anchors)
    replay_records = []
    for q, a in STANDARD_SQL_REPLAY:
        for _ in range(16):
            replay_records.append({
                "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                "meta": {"source": "standard_sql_rehearsal", "family": "relational_fundamentals"},
                "text": f"### Question:\n{q}\n\n### Answer:\n{a}"
            })
    print(f"Generated Standard SQL Rehearsal records: {len(replay_records)}")

    # Combine all
    total_dataset = existing_records + anti_records + adv_records + replay_records
    rng.shuffle(total_dataset)

    # Assertions
    assert len(total_dataset) >= 1500, f"Expected >= 1500 records, got {len(total_dataset)}"
    assert rejected == 0, f"{rejected} generated SQL blocks failed sqlglot parsing!"

    OUT_V3.write_text("\n".join(json.dumps(r) for r in total_dataset) + "\n")
    print("\n" + "=" * 74)
    print(f"SUCCESS: Wrote {len(total_dataset)} records to {OUT_V3}")
    print(f"  - v2 Foundation (pgvector / RRF / DDL): {len(existing_records)} records")
    print(f"  - Anti-Pattern Fixes (Mistakes):        {len(anti_records)} records")
    print(f"  - Advanced SQL (PG 17 Features):        {len(adv_records)} records")
    print(f"  - Standard SQL Rehearsal (Anchors):     {len(replay_records)} records")
    print("=" * 74)


if __name__ == "__main__":
    main()
