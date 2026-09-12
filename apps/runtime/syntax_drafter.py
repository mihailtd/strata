"""Deterministic AST & Syntax Fast-Forwarding Trie Drafter for Speculative Decoding.

Matches high-frequency, deterministic code skeletons and syntax macros in sub-microsecond
host time (0 bytes VRAM) and emits speculative draft token chains directly into forward_verify.
"""

from __future__ import annotations

import re
from typing import Any


class TrieNode:
    __slots__ = ("children", "continuation", "macro_name")

    def __init__(self) -> None:
        self.children: dict[int, TrieNode] = {}
        self.continuation: list[int] | None = None
        self.macro_name: str | None = None


# Canonical high-frequency code syntax macros across Python, FastAPI, SQL, DuckDB, PyTest
DEFAULT_SYNTAX_MACROS: list[dict[str, str]] = [
    # Python Core & Idioms
    {
        "name": "py_main_guard",
        "trigger": "if __name__ == ",
        "continuation": '"__main__":\n    ',
    },
    {
        "name": "py_init_self",
        "trigger": "def __init__(self",
        "continuation": ", ",
    },
    {
        "name": "py_except_exc",
        "trigger": "except Exception as ",
        "continuation": "e:\n    ",
    },
    {
        "name": "py_typing_imports",
        "trigger": "from typing import ",
        "continuation": "List, Dict, Optional, Any, Tuple, Union\n",
    },
    {
        "name": "py_status_dict",
        "trigger": 'return {"status": ',
        "continuation": '"ok", ',
    },
    {
        "name": "py_error_dict",
        "trigger": 'return {"error": ',
        "continuation": '"not_found", ',
    },
    {
        "name": "py_super_init",
        "trigger": "super().__init__(",
        "continuation": "*args, **kwargs)\n        ",
    },
    {
        "name": "py_dict_items",
        "trigger": "for key, value in ",
        "continuation": "data.items():\n        ",
    },
    {
        "name": "py_logger_init",
        "trigger": "logger = logging.getLogger(",
        "continuation": "__name__)\n",
    },
    # PyTest & Async Testing
    {
        "name": "pytest_import_asyncio",
        "trigger": "import pytest\n",
        "continuation": "import asyncio\n",
    },
    {
        "name": "pytest_async_test",
        "trigger": "@pytest.mark.asyncio\n",
        "continuation": "async def test_",
    },
    {
        "name": "pytest_fixture_def",
        "trigger": "@pytest.fixture\n",
        "continuation": "def ",
    },
    {
        "name": "pytest_assert_status_200",
        "trigger": "assert response.status_code == ",
        "continuation": "200\n    ",
    },
    {
        "name": "pytest_assert_status_201",
        "trigger": "assert response.status_code == 2",
        "continuation": "01\n    ",
    },
    {
        "name": "pytest_assert_status_404",
        "trigger": "assert response.status_code == 4",
        "continuation": "04\n    ",
    },
    {
        "name": "pytest_assert_json_status",
        "trigger": 'assert response.json()["status"] == ',
        "continuation": '"ok"\n    ',
    },
    {
        "name": "pytest_client_get",
        "trigger": "response = await client.get(",
        "continuation": '"/api/v1/',
    },
    {
        "name": "pytest_client_post",
        "trigger": "response = await client.post(",
        "continuation": '"/api/v1/',
    },
    # FastAPI & Pydantic
    {
        "name": "fastapi_core_imports",
        "trigger": "from fastapi import ",
        "continuation": "FastAPI, Depends, HTTPException, status, Query, Path\n",
    },
    {
        "name": "pydantic_core_imports",
        "trigger": "from pydantic import ",
        "continuation": "BaseModel, Field, ConfigDict\n",
    },
    {
        "name": "pydantic_config_dict",
        "trigger": "model_config = ",
        "continuation": "ConfigDict(from_attributes=True)\n",
    },
    {
        "name": "fastapi_status_200",
        "trigger": "status_code=status.HTTP_2",
        "continuation": "00_OK",
    },
    {
        "name": "fastapi_status_201",
        "trigger": "status_code=status.HTTP_2",
        "continuation": "01_CREATED",
    },
    {
        "name": "fastapi_status_404",
        "trigger": "status_code=status.HTTP_4",
        "continuation": "04_NOT_FOUND",
    },
    {
        "name": "fastapi_raise_404",
        "trigger": "raise HTTPException(status_code=404, ",
        "continuation": 'detail="Resource not found")\n',
    },
    {
        "name": "fastapi_raise_status",
        "trigger": "raise HTTPException(status_code=status.HTTP_",
        "continuation": '404_NOT_FOUND, detail="Not found")\n',
    },
    {
        "name": "fastapi_router_get",
        "trigger": '@router.get("/',
        "continuation": '", response_model=',
    },
    {
        "name": "fastapi_router_post",
        "trigger": '@router.post("/',
        "continuation": '", status_code=status.HTTP_201_CREATED)\n',
    },
    # PostgreSQL & asyncpg
    {
        "name": "asyncpg_acquire",
        "trigger": "async with pool.acquire() as ",
        "continuation": "conn:\n        ",
    },
    {
        "name": "asyncpg_transaction",
        "trigger": "async with conn.transaction(",
        "continuation": "readonly=True):\n            ",
    },
    {
        "name": "pg_create_extension",
        "trigger": "CREATE EXTENSION IF NOT EXISTS ",
        "continuation": '"uuid-ossp";\n',
    },
    {
        "name": "pg_hnsw_ops",
        "trigger": "USING hnsw (",
        "continuation": "embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);",
    },
    # DuckDB & Streaming OLAP
    {
        "name": "duckdb_qualify_row_number",
        "trigger": "QUALIFY ROW_NUMBER() OVER (",
        "continuation": "PARTITION BY ",
    },
    {
        "name": "duckdb_rows_between",
        "trigger": "ROWS BETWEEN ",
        "continuation": "UNBOUNDED PRECEDING AND CURRENT ROW",
    },
    {
        "name": "duckdb_read_parquet",
        "trigger": "SELECT * FROM ",
        "continuation": "read_parquet('",
    },
]

# Regex triggers for dynamic variable patterns that cannot be fully static in a Trie
DYNAMIC_REGEX_PATTERNS = [
    {
        "name": "pydantic_model_declaration",
        "pattern": re.compile(r"class\s+[A-Za-z0-9_]+\s*\(\s*BaseModel\s*\):\s*$"),
        "continuation": "\n    model_config = ConfigDict(from_attributes=True)\n",
    },
    {
        "name": "async_def_header",
        "pattern": re.compile(r"async\s+def\s+[A-Za-z0-9_]+\s*\(\s*$"),
        "continuation": "self, ",
    },
]


class SyntaxTrieDrafter:
    """Fast, zero-VRAM Deterministic AST & Syntax Fast-Forwarding Drafter.

    Maintains a suffix Trie of token IDs for sub-microsecond matching of common syntax macros,
    with a lightweight regex fallback for dynamic variable patterns.
    """

    def __init__(self, tokenizer: Any, macros: list[dict[str, str]] | None = None) -> None:
        self.tokenizer = tokenizer
        self.root = TrieNode()
        self.macro_count = 0
        self.macros = macros or DEFAULT_SYNTAX_MACROS
        self._compiled_regexes = DYNAMIC_REGEX_PATTERNS
        self._build_trie()

    def _build_trie(self) -> None:
        """Encodes syntax macros and populates the reverse-suffix Trie."""
        for macro in self.macros:
            trigger_str = macro["trigger"]
            cont_str = macro["continuation"]
            name = macro.get("name", "macro")

            # Register standard trigger and 4-space indented trigger for BPE space boundary robustness
            triggers_to_register = [trigger_str]
            if not trigger_str.startswith(" ") and not trigger_str.startswith("\n"):
                triggers_to_register.append("    " + trigger_str)
                triggers_to_register.append(" " + trigger_str)

            for trig in triggers_to_register:
                trigger_ids: list[int] = self.tokenizer.encode(trig, add_special_tokens=False)
                cont_ids: list[int] = self.tokenizer.encode(cont_str, add_special_tokens=False)

                if not trigger_ids or not cont_ids:
                    continue

                full_ids = trigger_ids + cont_ids
                start_idx = len(trigger_ids)
                # Index the base trigger and every intermediate continuation checkpoint
                for cp_idx in range(start_idx, len(full_ids)):
                    prefix_window = full_ids[max(0, cp_idx - 16) : cp_idx]
                    remaining = full_ids[cp_idx : cp_idx + 8]
                    if not remaining:
                        continue

                    curr = self.root
                    for tok_id in reversed(prefix_window):
                        if tok_id not in curr.children:
                            curr.children[tok_id] = TrieNode()
                        curr = curr.children[tok_id]

                    curr.continuation = remaining
                    curr.macro_name = f"{name}_cp{cp_idx}"
                    self.macro_count += 1

    def find_draft(self, tokens: list[int], max_k: int = 3) -> list[int]:
        """Finds speculative continuation for the current token suffix.

        Args:
            tokens: Current context token IDs including the newly accepted token.
            max_k: Maximum candidate tokens to return (default 3, matching K=4 verify graph).

        Returns:
            List of candidate token IDs (length 0 to max_k).
        """
        if not tokens:
            return []

        # 1. Reverse-Suffix Trie Traversal (< 0.5 µs)
        curr = self.root
        match_cont: list[int] | None = None

        # Look back up to 16 tokens
        limit = min(len(tokens), 16)
        for i in range(1, limit + 1):
            tok = tokens[-i]
            if tok in curr.children:
                curr = curr.children[tok]
                if curr.continuation is not None:
                    match_cont = curr.continuation
                    # Keep searching in case a longer suffix matches
            else:
                break

        if match_cont:
            return match_cont[:max_k]

        # 2. Dynamic Regex Fallback (< 5.0 µs on sliding window of last 15 tokens)
        if len(tokens) >= 4 and self._compiled_regexes:
            window_text = self.tokenizer.decode(tokens[-15:])
            for dyn in self._compiled_regexes:
                if dyn["pattern"].search(window_text):
                    cont_ids = self.tokenizer.encode(dyn["continuation"], add_special_tokens=False)
                    if cont_ids:
                        return cont_ids[:max_k]

        return []
