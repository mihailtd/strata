"""Anti-pattern traps: does the expert have TASTE, and does that taste generalise?

TIER 3 of the evaluation pyramid. Tier 1 (held-out gate) asks "did the adapter
damage the base?". Tier 2 (in-domain) asks "can it solve the problem?". Neither
measures the reason the adapter exists: a base model already solves most standard
problems, but it carries a decade of bad habits -- `SERIAL`, `NOT IN` over
nullable columns, `pip install` + virtualenv, mutable default arguments.

⚠️ THE SPLIT THAT MAKES THIS HONEST
Every trap is labelled `trained` or `novel`, verified against both corpora:

    trained  the corpus explicitly teaches this fix (NOT EXISTS 223 hits,
             half-open BETWEEN 309, IDENTITY 164, lower() 390, NULLIF 106,
             COUNT(*) 393). Measures whether the preference INSTALLED.
    novel    ZERO hits in either corpus. Measures whether the preference
             GENERALISES to habits nobody trained.

Reporting them together would repeat the contamination mistake that invalidated
the earlier gates: a high trained score proves only that fine-tuning works, while
the novel score is the one that says something about taste.

SCORING
-------
Each trap carries a `bad` and a `good` detector. AST-based where the structure is
expressible (`ast` for Python, `sqlglot` for SQL), regex where it is not -- each
detector records which method it used, because a regex detector is weaker evidence
and should not be presented as if it were structural.

    bad present            -> 0   (fell into the trap)
    good present, no bad   -> 1   (idiomatic)
    neither                -> NO SIGNAL, excluded from the rate

Excluding no-signal items matters: an answer that never emits code should not
count as "rejected the anti-pattern".
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class Trap:
    id: str
    tier: str            # "trained" | "novel"
    lang: str            # "sql" | "python"
    prompt: str
    bad: Callable[[str], bool]
    good: Callable[[str], bool]
    method: str          # "ast" | "regex" | "mixed"
    note: str = ""


# ----------------------------------------------------------------- helpers
def _py(code: str):
    try:
        return ast.parse(code)
    except Exception:
        return None


def py_has_mutable_default(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in list(n.args.defaults) + [x for x in n.args.kw_defaults if x]:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    return True
    return False


def py_has_none_default_guard(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in list(n.args.defaults) + [x for x in n.args.kw_defaults if x]:
                if isinstance(d, ast.Constant) and d.value is None:
                    return True
    return False


def py_has_bare_except(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    return any(isinstance(n, ast.ExceptHandler) and n.type is None for n in ast.walk(t))


def py_has_typed_except(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    return any(isinstance(n, ast.ExceptHandler) and n.type is not None for n in ast.walk(t))


def py_uses_ospath(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    for n in ast.walk(t):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute):
            if getattr(n.value.value, "id", "") == "os" and n.value.attr == "path":
                return True
    return bool(re.search(r"\bos\.path\.", code))


def py_uses_pathlib(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    for n in ast.walk(t):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mod = getattr(n, "module", "") or ""
            names = [a.name for a in n.names]
            if mod == "pathlib" or "pathlib" in names:
                return True
    return False


def py_uses_type_eq(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    for n in ast.walk(t):
        if isinstance(n, ast.Compare) and isinstance(n.left, ast.Call):
            if getattr(n.left.func, "id", "") == "type":
                return True
    return False


def py_uses_isinstance(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    return any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "isinstance"
               for n in ast.walk(t))


def py_uses_time_time(code: str) -> bool:
    return bool(re.search(r"\btime\.time\s*\(", code))


def py_uses_perf_counter(code: str) -> bool:
    return bool(re.search(r"perf_counter|monotonic", code))


def rx(p):
    return lambda s: bool(re.search(p, s, re.I | re.S))


# ------------------------------------------------------------------- traps
TRAPS: list[Trap] = [
    # ---------------- TRAINED: does the taught preference install? ----------
    Trap("not_in", "trained", "sql",
         "Table `app_users(id bigint, email text)` and `orders(id bigint, user_id bigint)`. "
         "`orders.user_id` is nullable. Write the query returning users who have never "
         "placed an order. Output only SQL in a ```sql block.",
         bad=rx(r"\bnot\s+in\s*\("), good=rx(r"\bnot\s+exists\b"), method="regex",
         note="NOT IN over a nullable column returns zero rows (three-valued logic)"),
    Trap("between_ts", "trained", "sql",
         "Table `events(id bigint, created_at timestamptz)`. Write the query selecting "
         "every event that occurred during January 2026. Output only SQL in a ```sql block.",
         bad=rx(r"\bbetween\b"),
         good=rx(r">=\s*'?2026-01-01|>=\s*timestamptz|<\s*'?2026-02-01"),
         method="regex", note="BETWEEN on timestamps leaks the final day / sub-second rows"),
    Trap("serial_identity", "trained", "sql",
         "Write the DDL for a table `parts` on PostgreSQL 17 with an auto-incrementing "
         "primary key and a name column. Output only SQL in a ```sql block.",
         bad=rx(r"\b(big)?serial\b"), good=rx(r"generated\s+always\s+as\s+identity"),
         method="regex", note="SERIAL is a legacy pseudo-type"),
    Trap("func_index", "trained", "sql",
         "Queries on `accounts(email text)` filter with a case-insensitive comparison on "
         "email and are doing a sequential scan despite an index on email. Write the DDL "
         "that fixes it. Output only SQL in a ```sql block.",
         bad=rx(r"create\s+index[^;]*\(\s*email\s*\)"),
         good=rx(r"create\s+index[^;]*lower\s*\(\s*email"), method="regex",
         note="a plain B-tree cannot serve lower(col)"),
    Trap("int_division", "trained", "sql",
         "Table `stock(on_hand integer, capacity integer)`. Write the query reporting the "
         "fill ratio as a percentage with two decimals, safe against a zero capacity. "
         "Output only SQL in a ```sql block.",
         bad=rx(r"on_hand\s*/\s*capacity(?!\s*::)"),
         good=rx(r"nullif\s*\(\s*capacity|100\.0\s*\*|::\s*numeric"), method="regex",
         note="integer/integer truncates to 0; zero denominator raises"),

    # ---------------- NOVEL: does taste generalise? -------------------------
    Trap("timestamptz", "novel", "sql",
         "Write the DDL for a table `sessions` recording when each session started, for a "
         "service with users in several countries. Output only SQL in a ```sql block.",
         bad=rx(r"timestamp\s*(?!with)(?!tz)\b(?!.*with\s+time\s+zone)"),
         good=rx(r"timestamptz|timestamp\s+with\s+time\s+zone"), method="regex",
         note="naive timestamp loses the offset; nothing in either corpus teaches this"),
    Trap("varchar255", "novel", "sql",
         "Write the DDL for a table `profiles` storing a display name and a short bio. "
         "Output only SQL in a ```sql block.",
         bad=rx(r"varchar\s*\(\s*\d+\s*\)"), good=rx(r"\btext\b"), method="regex",
         note="varchar(n) has no performance benefit in PostgreSQL"),
    Trap("explicit_join", "novel", "sql",
         "Tables `a(id, b_id)` and `b(id, label)`. Write the query returning a.id with "
         "b.label for matching rows. Output only SQL in a ```sql block.",
         bad=rx(r"from\s+a\s*,\s*b\b"), good=rx(r"\bjoin\b[^;]*\bon\b"), method="regex",
         note="comma-join hides the join condition in WHERE"),
    Trap("mutable_default", "novel", "python",
         "Write a Python function `collect(item, bucket)` that appends an item to a bucket "
         "list and returns it, where the bucket argument is optional. "
         "Output only Python in a ```python block.",
         bad=py_has_mutable_default, good=py_has_none_default_guard, method="ast",
         note="a mutable default is shared across every call"),
    Trap("bare_except", "novel", "python",
         "Write a Python function `load(path)` that reads JSON from a file and returns an "
         "empty dict if the file is missing or malformed. "
         "Output only Python in a ```python block.",
         bad=py_has_bare_except, good=py_has_typed_except, method="ast",
         note="bare except swallows KeyboardInterrupt and SystemExit"),
    Trap("pathlib", "novel", "python",
         "Write a Python function `newest(directory)` returning the most recently modified "
         "file in a directory. Output only Python in a ```python block.",
         bad=py_uses_ospath, good=py_uses_pathlib, method="ast",
         note="os.path is the legacy API"),
    Trap("isinstance", "novel", "python",
         "Write a Python function `describe(value)` returning 'int', 'str' or 'other' "
         "depending on the runtime type of its argument. "
         "Output only Python in a ```python block.",
         bad=py_uses_type_eq, good=py_uses_isinstance, method="ast",
         note="type(x) == T breaks on subclasses"),
    Trap("perf_counter", "novel", "python",
         "Write a Python function `timed(fn)` that runs a callable and returns the elapsed "
         "duration in seconds. Output only Python in a ```python block.",
         bad=py_uses_time_time, good=py_uses_perf_counter, method="regex",
         note="time.time() is wall clock and can go backwards"),

    # ---------------- NOVEL: modern tooling preference ----------------------
    Trap("uv_stack", "novel", "python",
         "Give the shell commands to start a new Python project, add the `httpx` "
         "dependency, and run `main.py`. Output only a shell block.",
         bad=rx(r"python\s+-m\s+venv|virtualenv|pip\s+install|poetry\s+add|pipenv"),
         good=rx(r"\buv\s+(init|add|run|sync)\b"), method="regex",
         note="Astral stack preference; uv appears in the corpus but venv/pip is the "
              "habit being tested"),
]



# ============================ EXPANSION ============================
# 14 traps yielded only 3 discordant pairs, and McNemar's floor at 3 is p=0.25 --
# the design could not reach significance however well the expert did. >=6
# discordant pairs are needed for p<0.05, so the set is enlarged here. Traps are
# pure content: no GPU to author, ~5 s each to run.

def py_uses_with_open(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    return any(isinstance(n, ast.With) or isinstance(n, ast.AsyncWith) for n in ast.walk(t))


def py_manual_close(code: str) -> bool:
    t = _py(code)
    if t is None:
        return bool(re.search(r"\.close\s*\(", code))
    has_close = any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "close"
                    for n in ast.walk(t))
    return has_close and not py_uses_with_open(code)


def py_uses_comprehension(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    return any(isinstance(n, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp))
               for n in ast.walk(t))


def py_uses_map_filter_lambda(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    for n in ast.walk(t):
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") in ("map", "filter"):
            if any(isinstance(a, ast.Lambda) for a in n.args):
                return True
    return False


def py_index_without_get(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    has_sub = any(isinstance(n, ast.Subscript) for n in ast.walk(t))
    has_get = any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "get"
                  for n in ast.walk(t))
    return has_sub and not has_get


def py_uses_get(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    return any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "get"
               for n in ast.walk(t))


def py_uses_star_import(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    return any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)
               for n in ast.walk(t))


def py_explicit_imports(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    return any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(t)) \
        and not py_uses_star_import(code)


TRAPS += [
    # ---- SQL, novel ----
    Trap("on_delete", "novel", "sql",
         "Write the DDL for `comments(id, post_id)` referencing `posts(id)`, such that "
         "removing a post cannot leave orphaned comments. Output only SQL in a ```sql block.",
         bad=rx(r"references\s+posts\s*\(\s*id\s*\)\s*(,|\)|;|$)"),
         good=rx(r"on\s+delete\s+(cascade|set\s+null|restrict)"), method="regex",
         note="a bare REFERENCES leaves deletion behaviour to chance"),
    Trap("offset_pagination", "novel", "sql",
         "Table `feed(id bigserial, created_at timestamptz)` has 10 million rows. Write the "
         "query fetching page 5000 of an infinite scroll, 20 rows per page, newest first. "
         "Output only SQL in a ```sql block.",
         bad=rx(r"\boffset\s+\d{3,}"),
         good=rx(r"where\s+\(?\s*(created_at|id)\s*[<>]"), method="regex",
         note="deep OFFSET scans and discards every skipped row; keyset seeks"),
    Trap("money_numeric", "novel", "sql",
         "Write the DDL for `invoices` storing an invoice total in euros. "
         "Output only SQL in a ```sql block.",
         bad=rx(r"\b(float|real|double\s+precision|money)\b"),
         good=rx(r"numeric\s*\(|\bnumeric\b|\bdecimal\b"), method="regex",
         note="binary floats cannot represent decimal currency exactly"),
    Trap("implicit_cast", "novel", "sql",
         "Table `logs(id bigint, trace_id text)`. Write the query fetching the row whose "
         "trace_id equals a supplied integer-looking value $1. "
         "Output only SQL in a ```sql block.",
         bad=rx(r"trace_id\s*=\s*\$1\s*::\s*(int|bigint)"),
         good=rx(r"trace_id\s*=\s*\$1(\s*::\s*text)?"), method="regex",
         note="casting the COLUMN side disables the index"),
    Trap("count_star_exists", "novel", "sql",
         "Write the query that checks whether table `jobs` contains any row with "
         "status = 'failed'. Return a boolean. Output only SQL in a ```sql block.",
         bad=rx(r"count\s*\(\s*\*\s*\)\s*>\s*0"),
         good=rx(r"\bexists\s*\("), method="regex",
         note="COUNT(*) scans everything; EXISTS short-circuits"),
    Trap("upsert_race", "novel", "sql",
         "Write the statement that inserts a row into `settings(key text unique, value text)` "
         "or updates it if the key already exists, safe under concurrency. "
         "Output only SQL in a ```sql block.",
         bad=rx(r"select[^;]*if\s+not\s+exists|begin[^;]*select[^;]*insert"),
         good=rx(r"on\s+conflict\s*\([^)]*\)\s*do\s+update"), method="regex",
         note="SELECT-then-INSERT races; ON CONFLICT is atomic"),
    Trap("text_search", "novel", "sql",
         "Table `docs(id, body text)` with 5 million rows. Write the query finding documents "
         "containing the word in $1, fast. Output only SQL in a ```sql block.",
         bad=rx(r"like\s*'%|ilike\s*'%"),
         good=rx(r"to_tsvector|@@|tsquery|pg_trgm|gin"), method="regex",
         note="leading-wildcard LIKE cannot use a B-tree"),
    # ---- Python, novel ----
    Trap("context_manager", "novel", "python",
         "Write a Python function `read_lines(path)` returning the lines of a text file. "
         "Output only Python in a ```python block.",
         bad=py_manual_close, good=py_uses_with_open, method="ast",
         note="a manual close leaks the handle on exception"),
    Trap("dict_get", "novel", "python",
         "Write a Python function `port_of(cfg)` returning cfg's 'port' entry, defaulting to "
         "8080 when absent. Output only Python in a ```python block.",
         bad=py_index_without_get, good=py_uses_get, method="ast",
         note="cfg['port'] raises KeyError instead of defaulting"),
    Trap("comprehension", "novel", "python",
         "Write a Python function `evens_squared(nums)` returning the squares of the even "
         "numbers in a list. Output only Python in a ```python block.",
         bad=py_uses_map_filter_lambda, good=py_uses_comprehension, method="ast",
         note="map/filter with lambdas where a comprehension reads better"),
    Trap("star_import", "novel", "python",
         "Write a Python module that computes the great-circle distance between two "
         "latitude/longitude pairs. Output only Python in a ```python block.",
         bad=py_uses_star_import, good=py_explicit_imports, method="ast",
         note="star imports pollute the namespace"),
    Trap("fstring", "novel", "python",
         "Write a Python function `greet(name, count)` returning a message embedding both "
         "values. Output only Python in a ```python block.",
         bad=rx(r"%\s*\(|\.format\s*\(|['\"]\s*\+\s*str\s*\("),
         good=rx(r"f['\"]"), method="regex",
         note="%-formatting and .format() where an f-string is clearer"),
    Trap("decimal_money", "novel", "python",
         "Write a Python function `total(prices)` summing a list of monetary amounts "
         "exactly. Output only Python in a ```python block.",
         bad=rx(r"\bfloat\s*\(|:\s*float\b"), good=rx(r"\bDecimal\b"),
         method="regex", note="float arithmetic on money accumulates error"),
    Trap("zoneinfo", "novel", "python",
         "Write a Python function `now_in(tz_name)` returning the current time in a named "
         "timezone. Output only Python in a ```python block.",
         bad=rx(r"utcnow\s*\(|datetime\.now\s*\(\s*\)"),
         good=rx(r"ZoneInfo|zoneinfo|timezone\.utc|tzinfo\s*="), method="regex",
         note="naive datetimes and utcnow() are deprecated footguns"),
    # ---- trained, more coverage ----
    Trap("skip_locked", "trained", "sql",
         "Several workers poll `jobs(id, status)` for pending work. Write the statement one "
         "worker uses to claim exactly one job without blocking the others. "
         "Output only SQL in a ```sql block.",
         bad=rx(r"for\s+update\s*;|for\s+update\s*\n"),
         good=rx(r"skip\s+locked"), method="regex",
         note="FOR UPDATE without SKIP LOCKED serialises the workers"),
    Trap("count_col", "trained", "sql",
         "Table `signups(id, referrer text)` where referrer is nullable. Write the query "
         "returning total rows and how many have a referrer. "
         "Output only SQL in a ```sql block.",
         bad=rx(r"count\s*\(\s*1\s*\)"),
         good=rx(r"count\s*\(\s*\*\s*\)[^;]*count\s*\(\s*referrer"), method="regex",
         note="COUNT(col) skips NULLs -- that is the point being tested"),
    Trap("pep723", "trained", "python",
         "Write a single-file Python script that fetches a URL with httpx and prints the "
         "status, runnable directly with no separate install step. "
         "Output only Python in a ```python block.",
         bad=rx(r"requirements\.txt|pip\s+install"),
         good=rx(r"#\s*///\s*script"), method="regex",
         note="PEP 723 inline metadata is the uv-native answer"),
    Trap("ruff_over_flake8", "trained", "python",
         "Give the command that lints and formats a Python project. "
         "Output only a shell block.",
         bad=rx(r"\b(flake8|black|isort|pylint|autopep8)\b"),
         good=rx(r"\bruff\b"), method="regex",
         note="Astral stack preference"),
]




# ==================== EXPANSION 2 ====================
# 32 traps produced 25 scorable and only +4/-0 discordant, and McNemar's floor at
# 4 discordant IS p=0.125 -- the design could not reach significance however well
# the expert performed. >=6 discordant pairs are required for p<0.05. These 18
# were each verified at ZERO occurrences across all three v3/v4 corpora.


def py_open_without_encoding(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    for n in ast.walk(t):
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "open":
            if not any(k.arg == "encoding" for k in n.keywords):
                return True
    return False


def py_open_with_encoding(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    return any(isinstance(n, ast.Call) and getattr(n.func, "id", "") == "open"
               and any(k.arg == "encoding" for k in n.keywords) for n in ast.walk(t))


def py_swallows_exception(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    for n in ast.walk(t):
        if isinstance(n, ast.ExceptHandler) and len(n.body) == 1 and isinstance(n.body[0], ast.Pass):
            return True
    return False


def py_handles_exception(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    for n in ast.walk(t):
        if isinstance(n, ast.ExceptHandler) and not (
                len(n.body) == 1 and isinstance(n.body[0], ast.Pass)):
            return True
    return False


def py_str_concat_in_loop(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    for n in ast.walk(t):
        if isinstance(n, (ast.For, ast.While)):
            for sub in ast.walk(n):
                if isinstance(sub, ast.AugAssign) and isinstance(sub.op, ast.Add):
                    return True
    return False


def py_uses_join(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    return any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "join"
               for n in ast.walk(t))


def py_uses_eval(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    return any(isinstance(n, ast.Call) and getattr(n.func, "id", "") in ("eval", "exec")
               for n in ast.walk(t))


def py_safe_parse(code: str) -> bool:
    return bool(re.search(r"json\.loads|literal_eval", code))


def py_os_system(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    for n in ast.walk(t):
        if isinstance(n, ast.Attribute) and n.attr == "system" \
                and getattr(n.value, "id", "") == "os":
            return True
    return False


def py_subprocess_run(code: str) -> bool:
    return bool(re.search(r"subprocess\.(run|Popen|check_output)", code))


def py_eq_none(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    for n in ast.walk(t):
        if isinstance(n, ast.Compare) and any(isinstance(o, (ast.Eq, ast.NotEq)) for o in n.ops):
            if any(isinstance(c, ast.Constant) and c.value is None for c in n.comparators):
                return True
    return False


def py_is_none(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    for n in ast.walk(t):
        if isinstance(n, ast.Compare) and any(isinstance(o, (ast.Is, ast.IsNot)) for o in n.ops):
            if any(isinstance(c, ast.Constant) and c.value is None for c in n.comparators):
                return True
    return False


def py_len_eq_zero(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    for n in ast.walk(t):
        if isinstance(n, ast.Compare) and isinstance(n.left, ast.Call) \
                and getattr(n.left.func, "id", "") == "len":
            if any(isinstance(c, ast.Constant) and c.value == 0 for c in n.comparators):
                return True
    return False


def py_truthiness(code: str) -> bool:
    t = _py(code)
    if t is None:
        return False
    for n in ast.walk(t):
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return True
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name):
            return True
    return False


TRAPS += [
    # ---------------- SQL, novel ----------------
    Trap("union_all", "novel", "sql",
         "Tables `eu_orders(id, total)` and `us_orders(id, total)` never share rows. "
         "Write the query returning every order from both. "
         "Output only SQL in a ```sql block.",
         bad=rx(r"\bunion\b(?!\s+all)"), good=rx(r"\bunion\s+all\b"), method="regex",
         note="plain UNION pays for a dedup pass that cannot find duplicates"),
    Trap("order_by_random", "novel", "sql",
         "Table `events` has 50 million rows. Write the query that pulls roughly 100 "
         "rows at random for a spot check, cheaply. "
         "Output only SQL in a ```sql block.",
         bad=rx(r"order\s+by\s+random\s*\("),
         good=rx(r"tablesample|where\s+random\s*\(\s*\)\s*<"), method="regex",
         note="ORDER BY random() sorts the whole table to take 100 rows"),
    Trap("date_on_column", "novel", "sql",
         "Table `logs(id, created_at timestamptz)` with an index on created_at. Write "
         "the query counting rows recorded on 2026-03-04, using the index. "
         "Output only SQL in a ```sql block.",
         bad=rx(r"\bdate\s*\(\s*created_at\s*\)|created_at::date"),
         good=rx(r"created_at\s*>=[^;]*created_at\s*<"), method="regex",
         note="wrapping the column in a function makes the index unusable"),
    Trap("any_array", "novel", "sql",
         "Write the query selecting rows of `items` whose id appears in a caller-supplied "
         "list of several thousand ids passed as one parameter $1. "
         "Output only SQL in a ```sql block.",
         bad=rx(r"\bin\s*\(\s*\$1\s*\)|\bin\s*\([^)]{80,}"),
         good=rx(r"=\s*any\s*\(|unnest\s*\(\s*\$1"), method="regex",
         note="a giant IN list re-plans per call; = ANY(array) takes one parameter"),
    Trap("at_time_zone", "novel", "sql",
         "Table `visits(started_at timestamptz)`. Write the query grouping visits by "
         "calendar day as observed in Europe/Lisbon, not UTC. "
         "Output only SQL in a ```sql block.",
         bad=rx(r"date_trunc\s*\(\s*'day'\s*,\s*started_at\s*\)(?![^;]*time\s+zone)"),
         good=rx(r"at\s+time\s+zone"), method="regex",
         note="truncating a timestamptz without a zone silently uses UTC"),
    Trap("case_insensitive", "novel", "sql",
         "Table `accounts(email text)` with millions of rows. Users log in with any "
         "capitalisation. Write the schema/index change that makes lookup exact and "
         "indexed. Output only SQL in a ```sql block.",
         bad=rx(r"where[^;]*ilike|lower\s*\(\s*email\s*\)\s*=\s*lower"),
         good=rx(r"citext|create\s+index[^;]*lower\s*\(\s*email"), method="regex",
         note="ILIKE on a plain column cannot use a B-tree"),
    Trap("enum_type", "novel", "sql",
         "Column `orders.status` may only ever hold 'new', 'paid' or 'shipped'. Write "
         "the DDL that enforces this in the database. "
         "Output only SQL in a ```sql block.",
         bad=rx(r"status\s+text\s*(,|\)|;|$)"),
         good=rx(r"create\s+type[^;]*as\s+enum|\bcheck\s*\("), method="regex",
         note="a bare text column enforces nothing"),
    Trap("count_distinct", "novel", "sql",
         "Table `pageviews(user_id bigint)`. Write the query returning how many distinct "
         "users appear. Output only SQL in a ```sql block.",
         bad=rx(r"count\s*\(\s*user_id\s*\)(?![^;]*distinct)"),
         good=rx(r"count\s*\(\s*distinct"), method="regex",
         note="COUNT(col) counts rows, not distinct values"),
    # ---------------- Python, novel ----------------
    Trap("open_encoding", "novel", "python",
         "Write a Python function `read_config(path)` returning the text of a UTF-8 file. "
         "Output only Python in a ```python block.",
         bad=py_open_without_encoding, good=py_open_with_encoding, method="ast",
         note="open() without encoding uses a platform-dependent default"),
    Trap("swallow_exception", "novel", "python",
         "Write a Python function `parse_port(raw)` returning an int port, or 8080 when "
         "the input cannot be parsed. Log why it failed. "
         "Output only Python in a ```python block.",
         bad=py_swallows_exception, good=py_handles_exception, method="ast",
         note="except: pass discards the diagnosis"),
    Trap("str_join", "novel", "python",
         "Write a Python function `render_csv(rows)` building one comma-separated string "
         "from a list of lists. Output only Python in a ```python block.",
         bad=py_str_concat_in_loop, good=py_uses_join, method="ast",
         note="repeated += on a str is quadratic"),
    Trap("secrets_token", "novel", "python",
         "Write a Python function `new_session_id()` returning an unguessable session "
         "identifier. Output only Python in a ```python block.",
         bad=rx(r"\brandom\.(choice|randint|random|sample)"),
         good=rx(r"\bsecrets\.|uuid4"), method="regex",
         note="random is a predictable PRNG; secrets is the CSPRNG"),
    Trap("no_eval", "novel", "python",
         "Write a Python function `load_record(text)` turning a serialised record from an "
         "untrusted source into a dict. Output only Python in a ```python block.",
         bad=py_uses_eval, good=py_safe_parse, method="ast",
         note="eval on untrusted input is arbitrary code execution"),
    Trap("subprocess", "novel", "python",
         "Write a Python function `disk_free(path)` that shells out to get free space and "
         "returns the command's output. Output only Python in a ```python block.",
         bad=py_os_system, good=py_subprocess_run, method="ast",
         note="os.system offers no argument quoting and no captured output"),
    Trap("is_none", "novel", "python",
         "Write a Python function `label(value)` returning 'missing' when the argument is "
         "None and 'present' otherwise. Output only Python in a ```python block.",
         bad=py_eq_none, good=py_is_none, method="ast",
         note="== None goes through __eq__; identity is the correct test"),
    Trap("truthiness", "novel", "python",
         "Write a Python function `first_or_default(items)` returning the first element of "
         "a list, or None when it has none. Output only Python in a ```python block.",
         bad=py_len_eq_zero, good=py_truthiness, method="ast",
         note="len(x) == 0 where truthiness reads better"),
    Trap("deepcopy", "novel", "python",
         "Write a Python function `with_override(config, key, value)` returning a NEW "
         "nested config dict with one key changed, leaving the original untouched. "
         "Output only Python in a ```python block.",
         bad=rx(r"\.copy\s*\(\s*\)|dict\s*\([^)]*\)"),
         good=rx(r"deepcopy"), method="regex",
         note="a shallow copy still aliases the nested dicts"),
    Trap("pyproject", "novel", "python",
         "Give the file a small Python library needs to declare its name, version and "
         "dependencies for publishing. Output only the file contents in a block.",
         bad=rx(r"setup\s*\(|setup\.py|requirements\.txt"),
         good=rx(r"\[project\]|pyproject"), method="regex",
         note="setup.py is superseded by PEP 621 pyproject metadata"),
]

# the zoneinfo trap scored NO SIGNAL for both arms -- its detectors were too narrow
for _t in TRAPS:
    if _t.id == "zoneinfo":
        _t.bad = rx(r"utcnow\s*\(|datetime\.now\s*\(\s*\)|\.now\s*\(\s*\)")
        _t.good = rx(r"ZoneInfo|zoneinfo|pytz|timezone\.utc|tz\s*=|tzinfo")


def by_tier(tier: str) -> list[Trap]:
    return [t for t in TRAPS if t.tier == tier]
