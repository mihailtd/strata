"""Evaluation logic for BFCL model tool calls against ground truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class BFCLEvalResult:
    passed: bool
    error: str
    matched_calls: int
    expected_calls: int


def _normalize_name(name: str) -> str:
    """Normalizes function name by lowercasing and replacing dots or hyphens."""
    return name.strip().lower().replace(".", "_").replace("-", "_")


def _structure_matches(actual_val: Any, expected: Any) -> bool | None:
    """Matches one BFCL answer candidate that is itself a STRUCTURE.

    BFCL's possible-answer format encodes nested values with per-leaf allowed
    lists: a dict parameter's candidate is `{"name": ["John Doe"], ...}` and a
    list-of-dicts candidate is `[{"field": ["age"], ...}, ...]`. Comparing the
    model's `{"name": "John Doe"}` to that with `==` can never succeed, which
    silently failed every correct nested-argument call (found by reading the
    raw failures: simple_89/94/96 were all exactly right). This descends the
    way the official AST checker does.

    Returns None when `expected` is not a structure this function owns, so the
    caller falls through to its scalar comparisons.
    """
    if isinstance(expected, dict):
        if not isinstance(actual_val, dict):
            return False
        for key, allowed in expected.items():
            allowed_list = allowed if isinstance(allowed, list) else [allowed]
            if key in actual_val:
                if not _arg_matches(actual_val[key], allowed_list):
                    return False
            elif "" not in allowed_list and None not in allowed_list:
                return False
        # A key the answer key never mentions is a hallucinated argument.
        return all(key in expected for key in actual_val)
    if isinstance(expected, list) and expected and all(isinstance(e, dict) for e in expected):
        if not isinstance(actual_val, list) or len(actual_val) != len(expected):
            return False
        return all(_structure_matches(a, e) for a, e in zip(actual_val, expected, strict=True))
    return None


def _arg_matches(actual_val: Any, allowed_list: list[Any]) -> bool:
    """Checks if actual argument value matches any of the allowed values."""
    if actual_val in allowed_list:
        return True

    for expected in allowed_list:
        if _structure_matches(actual_val, expected):
            return True

    # Numeric comparison with type flexibility (e.g. 5.0 == 5)
    if isinstance(actual_val, (int, float)):
        for expected in allowed_list:
            if isinstance(expected, (int, float)) and actual_val == expected:
                return True

    # String comparison case-insensitive
    if isinstance(actual_val, str):
        actual_str = actual_val.strip().lower()
        for expected in allowed_list:
            if isinstance(expected, str) and actual_str == expected.strip().lower():
                return True

    # Substring / list comparison
    for expected in allowed_list:
        if str(actual_val) == str(expected):
            return True

    return False


def _match_single_call(
    cand_name: str,
    cand_args: dict[str, Any],
    exp_name: str,
    exp_params: dict[str, list[Any]],
) -> tuple[bool, str]:
    if _normalize_name(cand_name) != _normalize_name(exp_name):
        return False, f"Name mismatch: got '{cand_name}', expected '{exp_name}'"

    for param_name, allowed_values in exp_params.items():
        if param_name in cand_args:
            val = cand_args[param_name]
            if not _arg_matches(val, allowed_values):
                return (
                    False,
                    f"Param '{param_name}' value mismatch: got {val}, allowed: {allowed_values}",
                )
        else:
            # Parameter not provided. Valid only if empty string or None is allowed
            if "" not in allowed_values and None not in allowed_values:
                return False, f"Required parameter '{param_name}' missing from call."

    return True, ""


def evaluate_tool_calls(
    model_tool_calls: list[dict],
    ground_truth: list[dict],
) -> BFCLEvalResult:
    """Evaluates a list of tool calls against ground truth."""
    if not ground_truth:
        # If no ground truth specified, passed if no tools called or any tool called
        return BFCLEvalResult(passed=True, error="", matched_calls=0, expected_calls=0)

    parsed_calls: list[tuple[str, dict[str, Any]]] = []
    for call in model_tool_calls:
        fn = call.get("function", call)
        name = fn.get("name", "")
        raw_args = fn.get("arguments", {})
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        parsed_calls.append((name, args))

    if len(parsed_calls) != len(ground_truth):
        return BFCLEvalResult(
            passed=False,
            error=f"Count mismatch: generated {len(parsed_calls)} calls, expected {len(ground_truth)}",
            matched_calls=0,
            expected_calls=len(ground_truth),
        )

    # For each expected call in ground truth, find a matching generated call
    remaining_candidates = list(parsed_calls)
    matched = 0

    for exp_dict in ground_truth:
        for exp_name, exp_params in exp_dict.items():
            matched_idx = None
            last_err = ""
            for idx, (cand_name, cand_args) in enumerate(remaining_candidates):
                ok, err = _match_single_call(cand_name, cand_args, exp_name, exp_params)
                if ok:
                    matched_idx = idx
                    break
                else:
                    last_err = err

            if matched_idx is not None:
                remaining_candidates.pop(matched_idx)
                matched += 1
            else:
                return BFCLEvalResult(
                    passed=False,
                    error=f"No matching tool call for '{exp_name}': {last_err}",
                    matched_calls=matched,
                    expected_calls=len(ground_truth),
                )

    return BFCLEvalResult(
        passed=True,
        error="",
        matched_calls=matched,
        expected_calls=len(ground_truth),
    )
