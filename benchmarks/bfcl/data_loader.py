"""Data loader for the Berkeley Function Calling Leaderboard (BFCL).

CRITICAL INVARIANT: Java is entirely ignored and excluded from this benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

# Explicitly excluded: "java"
ALLOWED_CATEGORIES = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "javascript",
]


@dataclass
class BFCLTestCase:
    id: str
    category: str
    messages: list[dict]
    tools: list[dict]
    ground_truth: list[dict] = field(default_factory=list)


def load_category(category: str) -> list[BFCLTestCase]:
    cat = category.strip().lower()
    if cat == "java":
        raise ValueError("Java is strictly ignored and excluded from BFCL evaluations.")
    if cat not in ALLOWED_CATEGORIES:
        raise ValueError(f"Category '{category}' is not supported. Allowed: {ALLOWED_CATEGORIES}")

    q_file = DATA_DIR / f"BFCL_v3_{cat}.json"
    a_file = DATA_DIR / f"possible_answer_BFCL_v3_{cat}.json"

    if not q_file.exists():
        raise FileNotFoundError(f"Question file for category '{cat}' not found at {q_file}")

    # Load ground truths
    answers: dict[str, list[dict]] = {}
    if a_file.exists():
        with open(a_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                answers[item["id"]] = item.get("ground_truth", [])

    test_cases: list[BFCLTestCase] = []
    with open(q_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cid = item["id"]

            # Format messages
            raw_q = item.get("question", [])
            messages = raw_q[0] if raw_q and isinstance(raw_q[0], list) else raw_q

            # Format tools as OpenAI tools
            raw_funcs = item.get("function", [])
            tools = []
            for fn in raw_funcs:
                # Convert python types to json schema types if needed
                params = fn.get("parameters", {})
                if params.get("type") == "dict":
                    params["type"] = "object"
                tools.append({
                    "type": "function",
                    "function": {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "parameters": params,
                    },
                })

            test_cases.append(
                BFCLTestCase(
                    id=cid,
                    category=cat,
                    messages=messages,
                    tools=tools,
                    ground_truth=answers.get(cid, []),
                )
            )

    return test_cases


def load_bfcl_data(categories: list[str] | None = None) -> list[BFCLTestCase]:
    """Loads BFCL test cases across specified categories (ignoring Java completely)."""
    cats = categories if categories is not None else ALLOWED_CATEGORIES
    all_cases: list[BFCLTestCase] = []
    for cat in cats:
        if "java" in cat.lower() and "javascript" not in cat.lower():
            # Skip Java completely
            continue
        all_cases.extend(load_category(cat))
    return all_cases
