"""Build the Agentic Coding, Diffs, Regex, and Terminal Commands corpus.

Four Core Pillars:
1. Deterministic Linux / Bash Terminal Commands: Non-interactive, cross-platform POSIX bash.
2. Aider SEARCH / REPLACE Diffs: Exact verbatim anchoring, minimal localized hunks.
3. Traceback-to-Fix Self-Repair: Pytest / compiler error localization and targeted repair.
4. Regex & String Boundary Precision: Apostrophe contractions, quote stripping, and anti-regex stack discipline.

Usage:
    uv run python apps/factory/corpus/build_agentic_coding_corpus.py           # report stats
    uv run python apps/factory/corpus/build_agentic_coding_corpus.py --write   # emit jsonl datasets
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Canonical monorepo root
REPO_ROOT = Path(__file__).resolve().parents[3]
ANSWER_MARKER = "\n\n### Answer:\n"

# --------------------------------------------------------------------------
# 1. LINUX / BASH COMMANDS POOLS
# --------------------------------------------------------------------------
SHELL_PHRASES_TRAIN = [
    "What is the command to {task}?",
    "How do I {task}?",
    "Show me the command to {task}.",
    "I need to {task}. What should I run?",
    "Quick terminal command to {task}?",
    "Give me the exact shell command to {task}.",
    "What command runs {task}?",
    "In bash, how do I {task}?",
    "Working in the terminal, how do I {task}?",
    "Terminal command please: {task}.",
]

SHELL_PHRASES_EVAL = [
    "Can you provide the command to {task}?",
    "What should I execute in the terminal to {task}?",
    "Please give me the shell command to {task}.",
]

# --------------------------------------------------------------------------
# 2. DIFF & SEARCH/REPLACE POOLS
# --------------------------------------------------------------------------
DIFF_PHRASES_TRAIN = [
    "Apply a minimal SEARCH/REPLACE diff to {task}.",
    "Write an Aider-style SEARCH/REPLACE block to {task}.",
    "Provide a surgical SEARCH/REPLACE block for {task}.",
    "Emit the exact SEARCH/REPLACE block to {task}.",
    "Fix this issue using a minimal SEARCH/REPLACE diff: {task}.",
    "Produce a clean SEARCH/REPLACE hunk to {task}.",
    "Use a SEARCH/REPLACE block to {task}.",
]

DIFF_PHRASES_EVAL = [
    "Show the SEARCH/REPLACE diff required to {task}.",
    "Can you give the exact SEARCH/REPLACE hunk to {task}?",
    "What is the minimal SEARCH/REPLACE diff to {task}?",
]

# --------------------------------------------------------------------------
# 3. TRACEBACK REPAIR POOLS
# --------------------------------------------------------------------------
REPAIR_PHRASES_TRAIN = [
    "The test suite failed with the following traceback. Emit a minimal SEARCH/REPLACE patch to fix it:\n\n```\n{traceback}\n```\n\nTarget file snippet:\n```python\n{code}\n```",
    "Fix the failing test below by providing a SEARCH/REPLACE block:\n\nPytest error:\n```\n{traceback}\n```\n\nSource code:\n```python\n{code}\n```",
    "Here is a traceback from pytest. Provide the minimal diff to fix the bug:\n\n```\n{traceback}\n```\n\nFile:\n```python\n{code}\n```",
    "Diagnose the following test failure and produce the exact SEARCH/REPLACE repair block:\n\nTraceback:\n```\n{traceback}\n```\n\nSnippet:\n```python\n{code}\n```",
]

REPAIR_PHRASES_EVAL = [
    "Pytest reported this error. Emit the exact SEARCH/REPLACE fix:\n\n```\n{traceback}\n```\n\nCode:\n```python\n{code}\n```",
    "Test assertion failed with traceback. Give the surgical SEARCH/REPLACE patch:\n\n```\n{traceback}\n```\n\nFile snippet:\n```python\n{code}\n```",
]

# --------------------------------------------------------------------------
# 4. REGEX & STRING BOUNDARY POOLS
# --------------------------------------------------------------------------
REGEX_PHRASES_TRAIN = [
    "How do I {task} with Python's re module?",
    "Write a Python function using regular expressions to {task}.",
    "What regex pattern correctly handles {task}?",
    "Show me how to {task} in Python without breaking edge cases.",
    "Idiomatic Python regex for: {task}.",
    "Provide the Python implementation using re to {task}.",
]

REGEX_PHRASES_EVAL = [
    "Can you provide a robust Python regex solution to {task}?",
    "What is the idiomatic Python way to {task}?",
    "How would you solve {task} reliably in Python?",
]


def make_record(family: str, question: str, answer: str, subgen: str) -> dict:
    return {
        "text": f"### Question:\n{question}{ANSWER_MARKER}{answer}",
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "meta": {
            "family": family,
            "tool": "agentic_coding",
            "gen": subgen,
        },
    }


def generate_shell_records(rng: random.Random, phrases: list[str], is_eval: bool) -> list[dict]:
    records: list[dict] = []

    tasks = [
        # Git workflows
        ("git_status", "check git status in machine-readable porcelain format", "```bash\ngit status --porcelain\n```\n\nProduces deterministic, parseable output without decoration or branch hints."),
        ("git_diff_staged", "inspect the exact staged changes before committing", "```bash\ngit diff --cached\n```\n\nShows only changes in the index ready for commit."),
        ("git_diff_head", "view diff against the previous commit", "```bash\ngit diff HEAD~1\n```\n\nCompares the current working tree with parent commit HEAD~1."),
        ("git_log_oneline", "view the last 5 commits in compact oneline format", "```bash\ngit log -n 5 --oneline\n```\n\nReturns the 5 most recent commit hashes and titles."),
        ("git_branch_create", "create and switch to a new branch named 'feature/patch'", "```bash\ngit checkout -b feature/patch\n```\n\nCreates the branch and switches the HEAD pointer in a single step."),
        ("git_stash_keep", "stash uncommitted changes including untracked files", "```bash\ngit stash -u\n```\n\nStashes both modified tracked files and untracked files."),
        ("git_stash_pop", "restore the most recently stashed changes and drop it from the stack", "```bash\ngit stash pop\n```\n\nApplies `stash@{0}` and immediately drops it upon success."),
        ("git_restore_file", "discard unstaged working tree changes to src/model.py", "```bash\ngit restore src/model.py\n```\n\nRestores the working tree file to match the index without affecting other files."),
        ("git_cherry_pick", "apply commit e4f1a23 to the current branch", "```bash\ngit cherry-pick e4f1a23\n```\n\nApplies the exact delta from commit e4f1a23 onto the active branch."),
        ("git_clean_dry_run", "preview which untracked files would be removed without deleting them", "```bash\ngit clean -nd\n```\n\nPerforms a dry run listing untracked files and directories."),

        # Search and inspection
        ("grep_python_defs", "search recursively for function definitions matching 'def calculate' in src/", "```bash\ngrep -rn \"def calculate\" src/\n```\n\nRecursively scans src/ printing line numbers and matching signatures."),
        ("grep_ignore_case", "case-insensitively find 'TODO' comments in all python files under apps/", "```bash\ngrep -rni --include=\"*.py\" \"todo\" apps/\n```\n\nLimits matches strictly to Python source files while ignoring case."),
        ("find_py_files", "find all Python files excluding hidden dot directories", "```bash\nfind . -type f -name \"*.py\" -not -path \"*/.*\"\n```\n\nTraverses directory tree excluding `.git`, `.venv`, and hidden caches."),
        ("head_first_lines", "inspect the first 25 lines of a source file", "```bash\nhead -n 25 app.py\n```\n\nReads the top 25 lines without reading the entire file into memory."),
        ("tail_last_lines", "inspect the last 30 lines of a log file", "```bash\ntail -n 30 logs/app.log\n```\n\nOutputs the trailing 30 lines cleanly."),
        ("du_dir_size", "check human-readable disk usage of the results directory", "```bash\ndu -sh results/\n```\n\nSummarizes total disk consumption of results/ in MB/GB."),
        ("diff_unified", "compare two files using unified diff format", "```bash\ndiff -u old_impl.py new_impl.py\n```\n\nEmits standard unified diff showing lines added and removed."),

        # Testing & Python tools
        ("pytest_quiet", "run pytest quietly without verbose header banners", "```bash\nuv run pytest -q\n```\n\nRuns test suite in quiet mode using the project's virtual environment."),
        ("pytest_single", "run only a single specific test method in tests/test_parser.py", "```bash\nuv run pytest tests/test_parser.py::TestParser::test_empty_input -v\n```\n\nDirectly targets the test method without executing unrelated tests."),
        ("pytest_keyword", "run tests matching the substring 'telemetry_probe'", "```bash\nuv run pytest -k \"telemetry_probe\"\n```\n\nFilters test discovery to tests matching the keyword expression."),
        ("uv_sync_frozen", "install dependencies in CI exactly as declared in uv.lock", "```bash\nuv sync --frozen\n```\n\nPrevents lockfile modifications and fails if lockfile is out of sync."),
        ("uv_run_script", "execute a standalone script using uv without manual activation", "```bash\nuv run python scripts/bench.py\n```\n\nResolves dependencies and runs the script in project context."),
        ("ruff_check_fix", "run ruff linter and apply safe automatic fixes across the repository", "```bash\nuv run ruff check --fix .\n```\n\nAutomatically corrects fixable lint violations across all Python files."),

        # Process & Diagnostics
        ("lsof_port", "find which process is listening on port 8003", "```bash\nlsof -i :8003\n```\n\nIdentifies PID and process name bound to TCP port 8003."),
        ("kill_pid", "terminate a hanging process with PID 14205", "```bash\nkill -15 14205\n```\n\nSends SIGTERM allowing the process to perform graceful cleanup; use `kill -9` if unresponsive."),
        ("mkdir_nested", "create nested directories 'artifacts/benchmarks/json' safely", "```bash\nmkdir -p artifacts/benchmarks/json\n```\n\nCreates parent directories as needed without error if they already exist."),
        ("tar_extract", "extract a tar.gz archive into output/ directory", "```bash\ntar -xzf data.tar.gz -C output/\n```\n\nDecompresses gzip and extracts archive contents into output/."),
        ("cargo_test_single", "run the specific test 'real_activate' in release mode showing stdout", "```bash\ncargo test real_activate --release -- --nocapture\n```\n\nRuns matching Rust unit test in release profile without capturing output."),
        ("curl_json_health", "check the health endpoint of a local server at port 8003", "```bash\ncurl -sSL http://127.0.0.1:8003/health\n```\n\nFetches health response silently while following redirects."),
    ]

    multiplier = 12 if not is_eval else 2
    for _ in range(multiplier):
        for fam, task, ans in tasks:
            q = rng.choice(phrases).format(task=task)
            records.append(make_record(f"sh_{fam}", q, ans, "agentic_shell_v1"))

    return records


def generate_diff_records(rng: random.Random, phrases: list[str], is_eval: bool) -> list[dict]:
    records: list[dict] = []

    diff_cases = [
        (
            "diff_sliding_window_bounds",
            "fix the sliding window boundary calculation when computing slice end",
            """<<<<<<< SEARCH
        window_end = start_idx + window_size - 1
        batch = stream[start_idx:window_end]
=======
        window_end = start_idx + window_size
        batch = stream[start_idx:window_end]
>>>>>>>""",
            "Python slice end indices are half-open [start, end), so subtracting 1 dropped the last element of the window."
        ),
        (
            "diff_null_coalesce",
            "handle None input safely by defaulting to an empty list",
            """<<<<<<< SEARCH
def process_items(items: list[str]) -> list[str]:
    return [item.strip() for item in items]
=======
def process_items(items: list[str] | None = None) -> list[str]:
    if items is None:
        return []
    return [item.strip() for item in items]
>>>>>>>""",
            "Adds a None guard to prevent AttributeError when callers pass None."
        ),
        (
            "diff_config_port_range",
            "validate that server port is within unprivileged network range (1024 to 65535)",
            """<<<<<<< SEARCH
    self.port = int(port_str)
=======
    port = int(port_str)
    if not (1024 <= port <= 65535):
        raise ValueError(f"port {port} out of range: must be between 1024 and 65535")
    self.port = port
>>>>>>>""",
            "Validates that the port number is within the valid TCP unprivileged port range."
        ),
        (
            "diff_modern_pathlib",
            "replace legacy open() file reading with Path.read_text(encoding='utf-8')",
            """<<<<<<< SEARCH
    with open(filepath, "r") as f:
        data = f.read()
=======
    data = Path(filepath).read_text(encoding="utf-8")
>>>>>>>""",
            "Uses modern pathlib with explicit UTF-8 encoding for safety and conciseness."
        ),
        (
            "diff_dict_get_fallback",
            "prevent KeyError by using dict.get with a default fallback",
            """<<<<<<< SEARCH
    user_role = config["roles"][user_id]
=======
    user_role = config.get("roles", {}).get(user_id, "guest")
>>>>>>>""",
            "Prevents KeyError on missing users or unconfigured roles dictionary."
        ),
        (
            "diff_exception_zero_division",
            "add a check to prevent division by zero in average calculation",
            """<<<<<<< SEARCH
    return total / count
=======
    if count == 0:
        return 0.0
    return total / count
>>>>>>>""",
            "Guards against ZeroDivisionError on empty collections."
        ),
        (
            "diff_filter_active_records",
            "replace manual loop with list comprehension when filtering active records",
            """<<<<<<< SEARCH
    active = []
    for record in records:
        if record.is_active:
            active.append(record.id)
    return active
=======
    return [record.id for record in records if record.is_active]
>>>>>>>""",
            "List comprehension reduces bytecode overhead and eliminates mutable accumulator list."
        ),
        (
            "diff_buffer_capacity_pow2",
            "calculate next power-of-two buffer capacity with maximum size validation",
            """<<<<<<< SEARCH
def calculate_buffer_size(requested: int) -> int:
    return requested
=======
def calculate_buffer_size(requested: int) -> int:
    if requested <= 0 or requested > (1 << 30):
        raise ValueError("buffer size must be positive and <= 1GB")
    return 1 << (requested - 1).bit_length()
>>>>>>>""",
            "Rounds buffer size up to the next power of two and validates upper capacity limits."
        ),
        (
            "diff_case_insensitive_lookup",
            "normalize search keys to lowercase before dictionary lookup",
            """<<<<<<< SEARCH
    def get_count(self, word: str) -> int:
        return self.counts.get(word, 0)
=======
    def get_count(self, word: str) -> int:
        return self.counts.get(word.lower(), 0)
>>>>>>>""",
            "Ensures case-insensitive word frequency matching."
        ),
        (
            "diff_ring_buffer_cursor",
            "advance ring buffer write index with wraparound modulo buffer capacity",
            """<<<<<<< SEARCH
    def advance_write_slot(self) -> None:
        self.write_index = self.write_index + 1
=======
    def advance_write_slot(self) -> None:
        self.write_index = (self.write_index + 1) % self.capacity
>>>>>>>""",
            "Modulo buffer capacity ensures the write cursor wraps around cleanly without indexing out of bounds."
        ),
        (
            "diff_json_loads_try_except",
            "safely handle malformed JSON strings by catching JSONDecodeError",
            """<<<<<<< SEARCH
    payload = json.loads(raw_text)
=======
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
>>>>>>>""",
            "Catches invalid JSON payloads without terminating the process."
        ),
        (
            "diff_enumerate_index",
            "replace range(len(items)) with idiomatic enumerate()",
            """<<<<<<< SEARCH
    for i in range(len(items)):
        print(f"{i}: {items[i]}")
=======
    for i, item in enumerate(items):
        print(f"{i}: {item}")
>>>>>>>""",
            "Uses enumerate for direct tuple unpacking and cleaner iteration."
        ),
        (
            "diff_set_membership",
            "optimize repeated list lookups by converting to a set",
            """<<<<<<< SEARCH
def filter_valid(candidates: list[str], allowed: list[str]) -> list[str]:
    return [c for c in candidates if c in allowed]
=======
def filter_valid(candidates: list[str], allowed: list[str]) -> list[str]:
    allowed_set = set(allowed)
    return [c for c in candidates if c in allowed_set]
>>>>>>>""",
            "Reduces lookup complexity from O(N*M) to O(N+M) using hash set lookup."
        ),
        (
            "diff_fstring_formatting",
            "modernize old-style % string formatting to f-strings",
            """<<<<<<< SEARCH
    msg = "User %s (id: %d) logged in" % (username, user_id)
=======
    msg = f"User {username} (id: {user_id}) logged in"
>>>>>>>""",
            "f-strings are faster, more readable, and evaluated inline."
        ),
        (
            "diff_dataclass_kw_only",
            "enforce keyword-only arguments on a configuration dataclass",
            """<<<<<<< SEARCH
@dataclass(frozen=True)
class Config:
    host: str
    port: int = 8080
=======
@dataclass(frozen=True, kw_only=True)
class Config:
    host: str
    port: int = 8080
>>>>>>>""",
            "`kw_only=True` prevents positional argument confusion at call sites."
        ),
        (
            "diff_strip_comparison",
            "strip leading and trailing whitespace before checking string equality",
            """<<<<<<< SEARCH
    if input_str == target:
=======
    if input_str.strip() == target.strip():
>>>>>>>""",
            "Prevents false mismatches caused by trailing newlines or extra spaces."
        ),
        (
            "diff_tuple_unpack",
            "unpack coordinate tuple directly in function signature",
            """<<<<<<< SEARCH
def move(coord: tuple[int, int], dx: int, dy: int) -> tuple[int, int]:
    x = coord[0]
    y = coord[1]
    return (x + dx, y + dy)
=======
def move(coord: tuple[int, int], dx: int, dy: int) -> tuple[int, int]:
    x, y = coord
    return (x + dx, y + dy)
>>>>>>>""",
            "Direct tuple unpacking avoids redundant indexing."
        ),
        (
            "diff_list_comprehension_filter",
            "refactor manual loop with append into a list comprehension",
            """<<<<<<< SEARCH
    evens = []
    for num in numbers:
        if num % 2 == 0:
            evens.append(num)
    return evens
=======
    return [num for num in numbers if num % 2 == 0]
>>>>>>>""",
            "List comprehension reduces bytecode overhead and simplifies logic."
        ),
    ]

    multiplier = 20 if not is_eval else 3
    for _ in range(multiplier):
        for fam, task, diff, explanation in diff_cases:
            q = rng.choice(phrases).format(task=task)
            ans = f"{diff}\n\n{explanation}"
            records.append(make_record(fam, q, ans, "agentic_diff_v1"))

    return records


def generate_repair_records(rng: random.Random, phrases: list[str], is_eval: bool) -> list[dict]:
    records: list[dict] = []

    repair_cases = [
        (
            "repair_assertion_payload_bounds",
            """Traceback (most recent call last):
  File "test_packet.py", line 22, in test_payload_size_limit
    with self.assertRaises(ValueError):
AssertionError: ValueError not raised by parse_packet""",
            """def parse_packet(payload_bytes: int) -> bytes:
    if payload_bytes < 0:
        raise ValueError("size cannot be negative")
    return b"\\x00" * payload_bytes""",
            """<<<<<<< SEARCH
    if payload_bytes < 0:
        raise ValueError("size cannot be negative")
=======
    if not (0 <= payload_bytes <= 65535):
        raise ValueError("payload size must be between 0 and 65535 bytes")
>>>>>>>""",
            "Adds an upper boundary check so packets exceeding maximum frame size raise ValueError."
        ),
        (
            "repair_assertion_sanitize_slug",
            """Traceback (most recent call last):
  File "test_slug.py", line 14, in test_empty_input
    self.assertEqual(clean_slug(""), "")
AssertionError: None != ''""",
            """def clean_slug(raw: str) -> str:
    if not raw:
        return None
    return raw.strip().lower()""",
            """<<<<<<< SEARCH
    if not raw:
        return None
=======
    if not raw:
        return ""
>>>>>>>""",
            "Returns an empty string instead of None when given empty input."
        ),
        (
            "repair_queue_peek_empty",
            """Traceback (most recent call last):
  File "test_queue.py", line 31, in test_peek_empty_list
    peek_item([])
IndexError: list index out of range""",
            """def peek_item(items: list[str]) -> str | None:
    return items[0]""",
            """<<<<<<< SEARCH
def peek_item(items: list[str]) -> str | None:
    return items[0]
=======
def peek_item(items: list[str]) -> str | None:
    if not items:
        return None
    return items[0]
>>>>>>>""",
            "Guards against IndexError when peeking from an empty collection."
        ),
        (
            "repair_db_session_closed",
            """Traceback (most recent call last):
  File "test_session.py", line 45, in test_cannot_query_closed_session
    session.execute_query()
AssertionError: RuntimeError not raised""",
            """    def execute_query(self) -> list:
        return self.cursor.fetchall()""",
            """<<<<<<< SEARCH
    def execute_query(self) -> list:
        return self.cursor.fetchall()
=======
    def execute_query(self) -> list:
        if not self.is_active:
            raise RuntimeError("cannot execute query on closed session")
        return self.cursor.fetchall()
>>>>>>>""",
            "Checks session active state and raises RuntimeError if attempting to query a closed connection."
        ),
        (
            "repair_type_error_concat",
            """Traceback (most recent call last):
  File "logger.py", line 14, in log_event
    return "Event " + event_id + " processed"
TypeError: can only concatenate str (not "int") to str""",
            """def log_event(event_id: int) -> str:
    return "Event " + event_id + " processed" """,
            """<<<<<<< SEARCH
    return "Event " + event_id + " processed"
=======
    return f"Event {event_id} processed"
>>>>>>>""",
            "Replaces raw string addition with an f-string to safely convert integers."
        ),
        (
            "repair_key_error_user_meta",
            """Traceback (most recent call last):
  File "auth.py", line 25, in get_user_role
    return user["metadata"]["role"]
KeyError: 'metadata'""",
            """def get_user_role(user: dict) -> str:
    return user["metadata"]["role"]""",
            """<<<<<<< SEARCH
    return user["metadata"]["role"]
=======
    return user.get("metadata", {}).get("role", "default")
>>>>>>>""",
            "Uses chained `.get()` lookups with fallback to avoid KeyError when metadata is omitted."
        ),
        (
            "repair_zero_division_metrics",
            """Traceback (most recent call last):
  File "stats.py", line 8, in compute_rate
    return successes / attempts
ZeroDivisionError: division by zero""",
            """def compute_rate(successes: int, attempts: int) -> float:
    return successes / attempts""",
            """<<<<<<< SEARCH
    return successes / attempts
=======
    if attempts == 0:
        return 0.0
    return successes / attempts
>>>>>>>""",
            "Returns 0.0 when attempts is zero to prevent ZeroDivisionError."
        ),
    ]

    multiplier = 35 if not is_eval else 5
    for _ in range(multiplier):
        for fam, tb, code, diff, explanation in repair_cases:
            q = rng.choice(phrases).format(traceback=tb.strip(), code=code.strip())
            ans = f"{diff}\n\n{explanation}"
            records.append(make_record(fam, q, ans, "agentic_repair_v1"))

    return records


def generate_regex_records(rng: random.Random, phrases: list[str], is_eval: bool) -> list[dict]:
    records: list[dict] = []

    regex_cases = [
        (
            "re_parse_log_identifiers",
            "extract system event identifiers with optional internal hyphen or colon segments while stripping punctuation",
            r"""```python
import re


def extract_identifiers(log_line: str) -> list[str]:
    # Matches alphanumeric tokens allowing internal hyphens or colons (e.g. "auth:user-login")
    # while ignoring trailing and leading delimiters.
    return re.findall(r"[a-zA-Z0-9]+(?:[-:][a-zA-Z0-9]+)*", log_line)
```

Key invariant: Non-capturing group `(?:[-:][a-zA-Z0-9]+)*` matches compound identifiers without splitting on internal delimiters or matching trailing punctuation."""
        ),
        (
            "re_anti_pattern_tag_hierarchy",
            "validate whether opening and closing XML tags are properly nested and balanced",
            r"""```python
import re


def validate_xml_hierarchy(document: str) -> bool:
    # IMPORTANT: Do NOT use regular expressions to validate recursive tag nesting.
    # Arbitrarily nested hierarchies require context-free grammar parsing via a stack.
    tags = re.findall(r"<(/?[a-zA-Z0-9_]+)>", document)
    stack: list[str] = []

    for tag in tags:
        if not tag.startswith("/"):
            stack.append(tag)
        else:
            expected = tag[1:]
            if not stack or stack[-1] != expected:
                return False
            stack.pop()

    return len(stack) == 0
```

Architectural note: Regular expressions cannot track arbitrarily nested recursive structures (Chomsky Type-2 grammars). A LIFO stack is required."""
        ),
        (
            "re_extract_dollar_amounts",
            "extract integer dollar amounts preceded by '$' without capturing the currency symbol",
            r"""```python
import re


def extract_amounts(text: str) -> list[int]:
    # Lookbehind (?<=\$) ensures '$' is present before digits without including it in match
    matches = re.findall(r"(?<=\$)\d+", text)
    return [int(m) for m in matches]
```

Using a positive lookbehind `(?<=\$)` leaves the dollar sign out of the extracted token."""
        ),
        (
            "re_non_greedy_html_strip",
            "strip HTML tags from a string without eating intermediate content between tags",
            r"""```python
import re


def strip_tags(html: str) -> str:
    # Non-greedy .*? stops at the first closing '>' rather than matching across multiple tags
    return re.sub(r"<.*?>", "", html)
```

Greedy `<.*>` would collapse `<p>hello</p><p>world</p>` into an empty string; non-greedy `<.*?>` removes each tag individually."""
        ),
        (
            "re_clean_spacing",
            "normalize multiple consecutive whitespaces and tabs into a single space and strip edges",
            r"""```python
import re


def clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
```

Matches any sequence of whitespace characters (spaces, tabs, newlines) and replaces them with a single space."""
        ),
        (
            "re_semver_parse",
            "parse semantic version strings into major, minor, patch integers",
            r"""```python
import re


def parse_semver(version: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version.strip())
    if not match:
        raise ValueError(f"Invalid semver: {version}")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)
```

Anchored `^...$` ensures the entire string conforms to semantic versioning."""
        ),
        (
            "re_camel_to_snake",
            "convert CamelCase identifier strings into snake_case",
            r"""```python
import re


def camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
```

Zero-width lookahead `(?=[A-Z])` with negative lookbehind `(?<!^)` inserts underscores before capital letters without prefixing the initial letter."""
        ),
        (
            "re_split_delimiters",
            "split text on commas, semicolons, or whitespace into clean non-empty items",
            r"""```python
import re


def split_items(text: str) -> list[str]:
    return [token for token in re.split(r"[,;\s]+", text) if token]
```

Splits on one or more punctuation/space characters and filters empty strings."""
        ),
    ]

    multiplier = 30 if not is_eval else 4
    for _ in range(multiplier):
        for fam, task, code in regex_cases:
            q = rng.choice(phrases).format(task=task)
            records.append(make_record(fam, q, code, "agentic_regex_v1"))

    return records


def build_corpora() -> tuple[list[dict], list[dict]]:
    rng_train = random.Random(42)
    rng_eval = random.Random(1337)

    train_records: list[dict] = []
    eval_records: list[dict] = []

    # 1. Shell commands
    train_records.extend(generate_shell_records(rng_train, SHELL_PHRASES_TRAIN, is_eval=False))
    eval_records.extend(generate_shell_records(rng_eval, SHELL_PHRASES_EVAL, is_eval=True))

    # 2. Diff records
    train_records.extend(generate_diff_records(rng_train, DIFF_PHRASES_TRAIN, is_eval=False))
    eval_records.extend(generate_diff_records(rng_eval, DIFF_PHRASES_EVAL, is_eval=True))

    # 3. Traceback repair records
    train_records.extend(generate_repair_records(rng_train, REPAIR_PHRASES_TRAIN, is_eval=False))
    eval_records.extend(generate_repair_records(rng_eval, REPAIR_PHRASES_EVAL, is_eval=True))

    # 4. Regex & String boundary records
    train_records.extend(generate_regex_records(rng_train, REGEX_PHRASES_TRAIN, is_eval=False))
    eval_records.extend(generate_regex_records(rng_eval, REGEX_PHRASES_EVAL, is_eval=True))

    rng_train.shuffle(train_records)
    rng_eval.shuffle(eval_records)

    return train_records, eval_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate agentic coding corpus")
    parser.add_argument("--write", action="store_true", help="Write JSONL files to disk")
    args = parser.parse_args()

    train, eval_set = build_corpora()

    print(f"Generated {len(train)} training records and {len(eval_set)} evaluation records.")

    # Breakdown by pillar
    train_pillars = {}
    for r in train:
        prefix = r["meta"]["family"].split("_")[0]
        train_pillars[prefix] = train_pillars.get(prefix, 0) + 1

    print("\nPillar Distribution (Training):")
    for pillar, count in sorted(train_pillars.items()):
        print(f"  - {pillar:8s}: {count:4d} records ({count / len(train) * 100:4.1f}%)")

    if args.write:
        data_dir = REPO_ROOT / "apps" / "factory" / "data" / "agentic_coding"
        data_dir.mkdir(parents=True, exist_ok=True)

        v3_path = data_dir / "training_data_v3.jsonl"
        v6_path = data_dir / "training_data_v6.jsonl"
        eval_path = data_dir / "eval_data_v3.jsonl"

        with v3_path.open("w", encoding="utf-8") as f:
            for r in train:
                f.write(json.dumps(r) + "\n")

        with v6_path.open("w", encoding="utf-8") as f:
            for r in train:
                f.write(json.dumps(r) + "\n")

        with eval_path.open("w", encoding="utf-8") as f:
            for r in eval_set:
                f.write(json.dumps(r) + "\n")

        print(f"\n[OK] Emitted datasets to {data_dir.relative_to(REPO_ROOT)}:")
        print(f"  - {v3_path.name} ({v3_path.stat().st_size / 1024:.1f} KB)")
        print(f"  - {v6_path.name} ({v6_path.stat().st_size / 1024:.1f} KB)")
        print(f"  - {eval_path.name} ({eval_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
