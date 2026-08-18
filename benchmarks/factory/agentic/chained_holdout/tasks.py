"""Chained multi-turn tasks built ONLY from constructs absent from both training corpora.

WHY THE HOLD-OUT MATTERS MORE THAN THE CHAINING
-----------------------------------------------
Multi-step difficulty does NOT defeat train-on-test contamination. If each step
uses a construct the adapter was trained on (HNSW index, RRF query, @mcp.tool()),
a rubric-follower simply scores well three times in a row. What defeats it is
holding out the MACHINERY.

Every construct below was verified to occur ZERO times across
`data/postgresql/training_data_v2.jsonl` and `data/astral/training_data_v2.jsonl`:

    SQL     DISTINCT ON, FILTER (WHERE), WITH ORDINALITY, TABLESAMPLE,
            percentile_cont/WITHIN GROUP, array_agg/unnest, COLLATE, date_trunc,
            LAG/LEAD, LATERAL
    Python  singledispatch, asyncio.TaskGroup, functools.partial, Protocol/ABC,
            __slots__

Rejected as CONTAMINATED (present in the corpora, counts in parentheses):
    WITH RECURSIVE (4), window OVER() (169), LISTEN/NOTIFY (20),
    MATERIALIZED VIEW (68), GROUPING SETS/ROLLUP (62), TRIGGER/plpgsql (33),
    JSONB ops (41), SKIP LOCKED (11), FOR UPDATE (18), CUBE (6),
    argparse (86), asynccontextmanager (86), TypedDict (32), pytest (22),
    contextlib (43), itertools (47), logging (43), Click (11), dataclass (10)

SCORING IS GROUND TRUTH, NOT A RUBRIC
-------------------------------------
Each SQL step is EXECUTED against a seeded database and its returned rows are
compared to a reference solution's rows. Each Python step is EXECUTED and its
stdout compared. A model cannot satisfy this by emitting the right keywords.

The prompts never NAME the construct -- they state the requirement semantically,
so any correct approach scores. That is deliberate: naming it would test
instruction-following, not capability.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Step:
    domain: str          # oracle expert for this step
    kind: str            # "sql" | "python"
    prompt: str
    reference: str       # reference solution, run to derive ground truth
    construct: str       # held-out machinery this exercises (for reporting only)
    order_matters: bool = True


@dataclass
class Task:
    id: str
    seed_sql: str
    steps: list[Step] = field(default_factory=list)


# --------------------------------------------------------------------- seeds
SEED_EVENTS = """
CREATE TABLE authors (id int PRIMARY KEY, name text, country text);
CREATE TABLE articles (
    id int PRIMARY KEY, author_id int, title text,
    published date, views int, tags text[]
);
INSERT INTO authors VALUES
 (1,'Ana','PT'),(2,'Bo','SE'),(3,'Cy','PT'),(4,'Di','DE');
INSERT INTO articles VALUES
 (1,1,'Alpha','2024-01-15',100,ARRAY['db','perf']),
 (2,1,'Beta','2024-03-02',300,ARRAY['db']),
 (3,2,'Gamma','2024-02-11',250,ARRAY['ml','perf']),
 (4,2,'Delta','2024-05-20',150,ARRAY['ml']),
 (5,3,'Eps','2024-04-01',400,ARRAY['db','ml']),
 (6,3,'Zeta','2024-04-15',50,ARRAY['perf']),
 (7,4,'Eta','2024-06-01',220,ARRAY['db']);
"""

SEED_METRICS = """
CREATE TABLE readings (
    id int PRIMARY KEY, sensor text, taken timestamptz, value numeric, ok boolean
);
INSERT INTO readings VALUES
 (1,'s1','2024-01-01 01:00+00',10,true),
 (2,'s1','2024-01-01 05:00+00',20,true),
 (3,'s1','2024-01-02 01:00+00',30,false),
 (4,'s2','2024-01-01 02:00+00',40,true),
 (5,'s2','2024-01-02 03:00+00',50,true),
 (6,'s2','2024-01-02 09:00+00',60,true),
 (7,'s3','2024-01-03 01:00+00',70,false),
 (8,'s3','2024-01-03 02:00+00',80,true);
"""

SEED_ORDERS = """
CREATE TABLE customers (id int PRIMARY KEY, label text);
CREATE TABLE orders (id int PRIMARY KEY, customer_id int, placed date, total numeric);
INSERT INTO customers VALUES (1,'acme'),(2,'Beta'),(3,'ácme'),(4,'zeta');
INSERT INTO orders VALUES
 (1,1,'2024-01-05',50),(2,1,'2024-02-05',70),(3,1,'2024-03-05',90),
 (4,2,'2024-01-20',20),(5,2,'2024-04-20',60),
 (6,3,'2024-02-01',10),(7,4,'2024-05-01',30),(8,4,'2024-05-09',80);
"""

PY_PROTOCOL = (
    "Write a Python module defining a structural interface named `Scorer` with a "
    "method `score(self, rows: list[dict]) -> float`, plus two independent "
    "implementations `SumScorer` (returns the sum of every row's 'value') and "
    "`MaxScorer` (returns the largest 'value'). The interface must be usable for "
    "static type checking WITHOUT the implementations inheriting from it. "
    "Then print, on one line, the two scores separated by a single space, "
    "computed over: {rows}. Print nothing else. Output only a ```python block."
)
PY_SLOTS = (
    "Write a Python module with a memory-compact class `Point` holding exactly two "
    "attributes x and y, declared so that instances carry no per-instance __dict__. "
    "Print `True` if a fresh Point(1,2) has no instance dictionary, else `False`. "
    "Print nothing else. Output only a ```python block."
)
PY_SINGLEDISPATCH = (
    "Write a Python module with a function `render` that dispatches on the TYPE of "
    "its first argument at runtime, using the standard library rather than "
    "if/elif type checks: for int return 'i:<v>', for str return 's:<v>', for list "
    "return 'l:<len>'. Print render(3), render('a'), render([1,2,3]) separated by "
    "single spaces on one line. Print nothing else. Output only a ```python block."
)
PY_PARTIAL = (
    "Write a Python module that builds a specialised callable from a general one by "
    "pre-binding its first argument (use the standard library helper for this, not a "
    "lambda or a nested def). Given `def scale(factor, v): return factor * v`, create "
    "`double` and `triple`, then print double(5) and triple(5) separated by one space "
    "on a single line. Print nothing else. Output only a ```python block."
)
PY_TASKGROUP = (
    "Write a Python module that runs three async coroutines concurrently using the "
    "structured-concurrency primitive added in Python 3.11 (not asyncio.gather), "
    "each returning its index doubled, then prints the three results in order "
    "separated by single spaces on one line. Print nothing else. "
    "Output only a ```python block."
)


def _t(tid, seed, specs):
    return Task(id=tid, seed_sql=seed, steps=[Step(**s) for s in specs])


TASKS: list[Task] = [
    _t("t01_latest_per_author", SEED_EVENTS, [
        dict(domain="postgresql", kind="sql", construct="DISTINCT ON",
             prompt="Return exactly one row per author: the single most recently "
                    "published article. Columns: author_id, title, published. "
                    "Order by author_id ascending. Output only SQL in a ```sql block.",
             reference="SELECT DISTINCT ON (author_id) author_id, title, published "
                       "FROM articles ORDER BY author_id, published DESC;"),
        dict(domain="postgresql", kind="sql", construct="FILTER (WHERE)",
             prompt="Now, in a SINGLE scan of articles with no subquery and no "
                    "WHERE clause, return one row per author_id with: total article "
                    "count, and the count of only those articles with views above 200. "
                    "Columns: author_id, all_count, hot_count. Order by author_id. "
                    "Output only SQL in a ```sql block.",
             reference="SELECT author_id, count(*) AS all_count, "
                       "count(*) FILTER (WHERE views > 200) AS hot_count "
                       "FROM articles GROUP BY author_id ORDER BY author_id;"),
        dict(domain="astral", kind="python", construct="Protocol",
             prompt=PY_PROTOCOL.format(rows="[{'value': 3.0}, {'value': 7.5}, {'value': 1.5}]"),
             reference="12.0 7.5"),
    ]),
    _t("t02_tag_explode", SEED_EVENTS, [
        dict(domain="postgresql", kind="sql", construct="unnest",
             prompt="Expand the tags array so each article contributes one row per "
                    "tag. Columns: id, tag. Order by id, then tag. "
                    "Output only SQL in a ```sql block.",
             reference="SELECT a.id, t AS tag FROM articles a, unnest(a.tags) AS t "
                       "ORDER BY a.id, tag;"),
        dict(domain="postgresql", kind="sql", construct="array_agg",
             prompt="Now invert it: one row per tag, with the sorted list of article "
                    "ids carrying that tag collapsed into a single array-valued "
                    "column. Columns: tag, ids. Order by tag. "
                    "Output only SQL in a ```sql block.",
             reference="SELECT t AS tag, array_agg(a.id ORDER BY a.id) AS ids "
                       "FROM articles a, unnest(a.tags) AS t GROUP BY t ORDER BY tag;"),
        dict(domain="astral", kind="python", construct="singledispatch",
             prompt=PY_SINGLEDISPATCH, reference="i:3 s:a l:3"),
    ]),
    _t("t03_position_in_array", SEED_EVENTS, [
        dict(domain="postgresql", kind="sql", construct="WITH ORDINALITY",
             prompt="For article 1, expand its tags and report each tag together with "
                    "its 1-based position within the original array. Columns: tag, pos. "
                    "Order by pos. Output only SQL in a ```sql block.",
             reference="SELECT t AS tag, ord AS pos FROM articles a, "
                       "unnest(a.tags) WITH ORDINALITY AS x(t, ord) "
                       "WHERE a.id = 1 ORDER BY ord;"),
        dict(domain="postgresql", kind="sql", construct="LATERAL",
             prompt="For every author, attach their two highest-viewed articles by "
                    "correlating a per-author subquery in the FROM clause (not a "
                    "window function, not a subquery in SELECT). Columns: author_id, "
                    "title, views. Order by author_id, views DESC. "
                    "Output only SQL in a ```sql block.",
             reference="SELECT au.id AS author_id, x.title, x.views FROM authors au "
                       "JOIN LATERAL (SELECT title, views FROM articles ar "
                       "WHERE ar.author_id = au.id ORDER BY views DESC LIMIT 2) x ON true "
                       "ORDER BY au.id, x.views DESC;"),
        dict(domain="astral", kind="python", construct="__slots__",
             prompt=PY_SLOTS, reference="True"),
    ]),
    _t("t04_time_buckets", SEED_METRICS, [
        dict(domain="postgresql", kind="sql", construct="date_trunc",
             prompt="Aggregate readings into calendar-day buckets, reporting the "
                    "number of readings per day. Columns: day, n. Order by day. "
                    "Return day as a timestamptz truncated to the day. "
                    "Output only SQL in a ```sql block.",
             reference="SELECT date_trunc('day', taken) AS day, count(*) AS n "
                       "FROM readings GROUP BY 1 ORDER BY 1;"),
        dict(domain="postgresql", kind="sql", construct="FILTER (WHERE)",
             prompt="Same daily buckets, but in one pass and with no WHERE clause, "
                    "report per day: total readings and how many had ok = true. "
                    "Columns: day, n, ok_n. Order by day. "
                    "Output only SQL in a ```sql block.",
             reference="SELECT date_trunc('day', taken) AS day, count(*) AS n, "
                       "count(*) FILTER (WHERE ok) AS ok_n FROM readings "
                       "GROUP BY 1 ORDER BY 1;"),
        dict(domain="astral", kind="python", construct="functools.partial",
             prompt=PY_PARTIAL, reference="10 15"),
    ]),
    _t("t05_percentiles", SEED_METRICS, [
        dict(domain="postgresql", kind="sql", construct="percentile_cont",
             prompt="For each sensor, report the median of value using a continuous "
                    "(interpolating) percentile, not an approximation. Columns: "
                    "sensor, med. Order by sensor. Output only SQL in a ```sql block.",
             reference="SELECT sensor, percentile_cont(0.5) WITHIN GROUP (ORDER BY value) "
                       "AS med FROM readings GROUP BY sensor ORDER BY sensor;"),
        dict(domain="postgresql", kind="sql", construct="LAG",
             prompt="Order readings by sensor then time, and for each row show the "
                    "value of the immediately preceding reading of the SAME sensor "
                    "(null for the first). Columns: id, sensor, value, prev_value. "
                    "Order by sensor, taken. Output only SQL in a ```sql block.",
             reference="SELECT id, sensor, value, lag(value) OVER "
                       "(PARTITION BY sensor ORDER BY taken) AS prev_value "
                       "FROM readings ORDER BY sensor, taken;"),
        dict(domain="astral", kind="python", construct="asyncio.TaskGroup",
             prompt=PY_TASKGROUP, reference="0 2 4"),
    ]),
    _t("t06_collation", SEED_ORDERS, [
        dict(domain="postgresql", kind="sql", construct="COLLATE",
             prompt="List customer labels sorted so that accents and letter case do "
                    "not affect the ordering (acme, ácme and Beta must sort "
                    "naturally). Column: label. Output only SQL in a ```sql block.",
             reference="SELECT label FROM customers ORDER BY label COLLATE \"und-x-icu\";",
             order_matters=True),
        dict(domain="postgresql", kind="sql", construct="DISTINCT ON",
             prompt="Return each customer's single most recent order. Columns: "
                    "customer_id, id, placed. Order by customer_id. "
                    "Output only SQL in a ```sql block.",
             reference="SELECT DISTINCT ON (customer_id) customer_id, id, placed "
                       "FROM orders ORDER BY customer_id, placed DESC;"),
        dict(domain="astral", kind="python", construct="Protocol",
             prompt=PY_PROTOCOL.format(rows="[{'value': 2.0}, {'value': 4.0}]"),
             reference="6.0 4.0"),
    ]),
    _t("t07_running_delta", SEED_ORDERS, [
        dict(domain="postgresql", kind="sql", construct="LAG",
             prompt="For each customer, list orders chronologically with the change "
                    "in total versus that customer's previous order (null for the "
                    "first). Columns: customer_id, id, total, delta. Order by "
                    "customer_id, placed. Output only SQL in a ```sql block.",
             reference="SELECT customer_id, id, total, total - lag(total) OVER "
                       "(PARTITION BY customer_id ORDER BY placed) AS delta "
                       "FROM orders ORDER BY customer_id, placed;"),
        dict(domain="postgresql", kind="sql", construct="LATERAL",
             prompt="For every customer row, attach their single largest order by "
                    "correlating a subquery in the FROM clause. Columns: id, label, "
                    "top_total. Order by id. Output only SQL in a ```sql block.",
             reference="SELECT c.id, c.label, x.total AS top_total FROM customers c "
                       "LEFT JOIN LATERAL (SELECT total FROM orders o "
                       "WHERE o.customer_id = c.id ORDER BY total DESC LIMIT 1) x "
                       "ON true ORDER BY c.id;"),
        dict(domain="astral", kind="python", construct="singledispatch",
             prompt=PY_SINGLEDISPATCH, reference="i:3 s:a l:3"),
    ]),
    _t("t08_monthly", SEED_ORDERS, [
        dict(domain="postgresql", kind="sql", construct="date_trunc",
             prompt="Total order value per calendar month. Columns: month, revenue. "
                    "Order by month. Return month as a date truncated to the month. "
                    "Output only SQL in a ```sql block.",
             reference="SELECT date_trunc('month', placed) AS month, sum(total) AS revenue "
                       "FROM orders GROUP BY 1 ORDER BY 1;"),
        dict(domain="postgresql", kind="sql", construct="array_agg",
             prompt="One row per customer with their order ids gathered into a single "
                    "array column, ascending. Columns: customer_id, ids. Order by "
                    "customer_id. Output only SQL in a ```sql block.",
             reference="SELECT customer_id, array_agg(id ORDER BY id) AS ids "
                       "FROM orders GROUP BY customer_id ORDER BY customer_id;"),
        dict(domain="astral", kind="python", construct="functools.partial",
             prompt=PY_PARTIAL, reference="10 15"),
    ]),
    _t("t09_quartiles", SEED_EVENTS, [
        dict(domain="postgresql", kind="sql", construct="percentile_cont",
             prompt="Report the 25th and 75th percentile of views across all "
                    "articles, interpolating between values. Columns: q1, q3. "
                    "Output only SQL in a ```sql block.",
             reference="SELECT percentile_cont(0.25) WITHIN GROUP (ORDER BY views) AS q1, "
                       "percentile_cont(0.75) WITHIN GROUP (ORDER BY views) AS q3 "
                       "FROM articles;"),
        dict(domain="postgresql", kind="sql", construct="FILTER (WHERE)",
             prompt="In a single pass over articles with no WHERE clause, return the "
                    "overall count and the count published after 2024-03-31. "
                    "Columns: n, recent_n. Output only SQL in a ```sql block.",
             reference="SELECT count(*) AS n, count(*) FILTER "
                       "(WHERE published > DATE '2024-03-31') AS recent_n FROM articles;"),
        dict(domain="astral", kind="python", construct="__slots__",
             prompt=PY_SLOTS, reference="True"),
    ]),
    _t("t10_lead", SEED_EVENTS, [
        dict(domain="postgresql", kind="sql", construct="LEAD",
             prompt="For each author's articles in publication order, show the title "
                    "of the NEXT article by that same author (null for the last). "
                    "Columns: author_id, title, next_title. Order by author_id, "
                    "published. Output only SQL in a ```sql block.",
             reference="SELECT author_id, title, lead(title) OVER "
                       "(PARTITION BY author_id ORDER BY published) AS next_title "
                       "FROM articles ORDER BY author_id, published;"),
        dict(domain="postgresql", kind="sql", construct="unnest",
             prompt="Count how many articles carry each tag. Columns: tag, n. "
                    "Order by tag. Output only SQL in a ```sql block.",
             reference="SELECT t AS tag, count(*) AS n FROM articles a, "
                       "unnest(a.tags) AS t GROUP BY t ORDER BY tag;"),
        dict(domain="astral", kind="python", construct="asyncio.TaskGroup",
             prompt=PY_TASKGROUP, reference="0 2 4"),
    ]),
    _t("t11_join_filter", SEED_EVENTS, [
        dict(domain="postgresql", kind="sql", construct="FILTER (WHERE)",
             prompt="Per country, in one pass and with no WHERE clause, report the "
                    "number of authors and how many of them are Portuguese. Columns: "
                    "country, n, pt_n. Order by country. "
                    "Output only SQL in a ```sql block.",
             reference="SELECT country, count(*) AS n, count(*) FILTER "
                       "(WHERE country = 'PT') AS pt_n FROM authors "
                       "GROUP BY country ORDER BY country;"),
        dict(domain="postgresql", kind="sql", construct="DISTINCT ON",
             prompt="One row per country: the author with the lowest id in that "
                    "country. Columns: country, id, name. Order by country. "
                    "Output only SQL in a ```sql block.",
             reference="SELECT DISTINCT ON (country) country, id, name FROM authors "
                       "ORDER BY country, id;"),
        dict(domain="astral", kind="python", construct="Protocol",
             prompt=PY_PROTOCOL.format(rows="[{'value': 1.0}, {'value': 2.0}, {'value': 9.0}]"),
             reference="12.0 9.0"),
    ]),
    _t("t12_ordinality_join", SEED_ORDERS, [
        dict(domain="postgresql", kind="sql", construct="WITH ORDINALITY",
             prompt="Expand the literal array ARRAY['a','b','c'] into rows carrying "
                    "each element's 1-based position. Columns: v, pos. Order by pos. "
                    "Output only SQL in a ```sql block.",
             reference="SELECT v, ord AS pos FROM unnest(ARRAY['a','b','c']) "
                       "WITH ORDINALITY AS x(v, ord) ORDER BY ord;"),
        dict(domain="postgresql", kind="sql", construct="LAG",
             prompt="List all orders by date with the previous order's total across "
                    "the whole table (null for the first). Columns: id, placed, "
                    "total, prev_total. Order by placed, id. "
                    "Output only SQL in a ```sql block.",
             reference="SELECT id, placed, total, lag(total) OVER "
                       "(ORDER BY placed, id) AS prev_total FROM orders "
                       "ORDER BY placed, id;"),
        dict(domain="astral", kind="python", construct="singledispatch",
             prompt=PY_SINGLEDISPATCH, reference="i:3 s:a l:3"),
    ]),
    _t("t13_daily_median", SEED_METRICS, [
        dict(domain="postgresql", kind="sql", construct="date_trunc+percentile",
             prompt="Per calendar day, report the interpolated median value. Columns: "
                    "day, med. Order by day. Return day truncated to the day. "
                    "Output only SQL in a ```sql block.",
             reference="SELECT date_trunc('day', taken) AS day, "
                       "percentile_cont(0.5) WITHIN GROUP (ORDER BY value) AS med "
                       "FROM readings GROUP BY 1 ORDER BY 1;"),
        dict(domain="postgresql", kind="sql", construct="LATERAL",
             prompt="For each sensor, attach its earliest reading by correlating a "
                    "subquery in the FROM clause. Columns: sensor, id, value. "
                    "Order by sensor. Output only SQL in a ```sql block.",
             reference="SELECT s.sensor, x.id, x.value FROM (SELECT DISTINCT sensor "
                       "FROM readings) s JOIN LATERAL (SELECT id, value FROM readings r "
                       "WHERE r.sensor = s.sensor ORDER BY taken LIMIT 1) x ON true "
                       "ORDER BY s.sensor;"),
        dict(domain="astral", kind="python", construct="functools.partial",
             prompt=PY_PARTIAL, reference="10 15"),
    ]),
    _t("t14_array_roundtrip", SEED_EVENTS, [
        dict(domain="postgresql", kind="sql", construct="array_agg",
             prompt="One row per author with the titles of their articles gathered "
                    "into a single array, ordered by publication date. Columns: "
                    "author_id, titles. Order by author_id. "
                    "Output only SQL in a ```sql block.",
             reference="SELECT author_id, array_agg(title ORDER BY published) AS titles "
                       "FROM articles GROUP BY author_id ORDER BY author_id;"),
        dict(domain="postgresql", kind="sql", construct="LEAD",
             prompt="Across all articles ordered by publication date, show each "
                    "article's views and the views of the next article (null at the "
                    "end). Columns: id, views, next_views. Order by published, id. "
                    "Output only SQL in a ```sql block.",
             reference="SELECT id, views, lead(views) OVER (ORDER BY published, id) "
                       "AS next_views FROM articles ORDER BY published, id;"),
        dict(domain="astral", kind="python", construct="__slots__",
             prompt=PY_SLOTS, reference="True"),
    ]),
    _t("t15_mixed", SEED_METRICS, [
        dict(domain="postgresql", kind="sql", construct="DISTINCT ON",
             prompt="For each sensor return only its latest reading. Columns: sensor, "
                    "id, value. Order by sensor. Output only SQL in a ```sql block.",
             reference="SELECT DISTINCT ON (sensor) sensor, id, value FROM readings "
                       "ORDER BY sensor, taken DESC;"),
        dict(domain="postgresql", kind="sql", construct="array_agg",
             prompt="One row per sensor with all its reading ids gathered ascending "
                    "into a single array column. Columns: sensor, ids. Order by "
                    "sensor. Output only SQL in a ```sql block.",
             reference="SELECT sensor, array_agg(id ORDER BY id) AS ids FROM readings "
                       "GROUP BY sensor ORDER BY sensor;"),
        dict(domain="astral", kind="python", construct="asyncio.TaskGroup",
             prompt=PY_TASKGROUP, reference="0 2 4"),
    ]),
]
