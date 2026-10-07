"""Unit tests for BFCL data loader and evaluator with strict Java exclusion invariant."""

from __future__ import annotations

import pytest

from benchmarks.bfcl import (
    ALLOWED_CATEGORIES,
    evaluate_tool_calls,
    load_bfcl_data,
    load_category,
)


def test_java_strictly_excluded_invariant():
    # 1. 'java' must never be in ALLOWED_CATEGORIES
    assert "java" not in ALLOWED_CATEGORIES
    for cat in ALLOWED_CATEGORIES:
        assert cat != "java"

    # 2. Attempting to load 'java' directly raises ValueError
    with pytest.raises(ValueError, match="Java is strictly ignored"):
        load_category("java")

    # 3. Loading default or all data must never load any Java test case
    cases = load_bfcl_data()
    for tc in cases:
        assert tc.category != "java"
        assert not tc.id.startswith("java_")


def test_load_categories():
    simple_cases = load_category("simple")
    assert len(simple_cases) == 400
    assert simple_cases[0].id == "simple_0"
    assert len(simple_cases[0].tools) > 0

    multiple_cases = load_category("multiple")
    assert len(multiple_cases) == 200

    js_cases = load_category("javascript")
    assert len(js_cases) == 50


def test_evaluator_exact_match():
    ground_truth = [
        {
            "calculate_triangle_area": {
                "base": [10],
                "height": [5],
                "unit": ["units", ""],
            }
        }
    ]
    model_calls = [
        {
            "function": {
                "name": "calculate_triangle_area",
                "arguments": '{"base": 10, "height": 5, "unit": "units"}',
            }
        }
    ]
    res = evaluate_tool_calls(model_calls, ground_truth)
    assert res.passed is True
    assert res.matched_calls == 1
    assert res.error == ""


def test_evaluator_optional_param_omitted():
    ground_truth = [
        {
            "calculate_triangle_area": {
                "base": [10],
                "height": [5],
                "unit": ["units", ""],
            }
        }
    ]
    # 'unit' is omitted, but "" is allowed in ground truth
    model_calls = [
        {
            "function": {
                "name": "calculate_triangle_area",
                "arguments": '{"base": 10, "height": 5}',
            }
        }
    ]
    res = evaluate_tool_calls(model_calls, ground_truth)
    assert res.passed is True


def test_evaluator_wrong_arguments():
    ground_truth = [
        {
            "calculate_triangle_area": {
                "base": [10],
                "height": [5],
                "unit": ["units", ""],
            }
        }
    ]
    # Wrong base: 99 instead of 10
    model_calls = [
        {
            "function": {
                "name": "calculate_triangle_area",
                "arguments": '{"base": 99, "height": 5}',
            }
        }
    ]
    res = evaluate_tool_calls(model_calls, ground_truth)
    assert res.passed is False
    assert "Param 'base' value mismatch" in res.error


def test_evaluator_parallel_calls():
    ground_truth = [
        {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
        {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}},
    ]
    model_calls = [
        {
            "function": {
                "name": "spotify_play",
                "arguments": '{"artist": "Taylor Swift", "duration": 20}',
            }
        },
        {
            "function": {
                "name": "spotify_play",
                "arguments": '{"artist": "Maroon 5", "duration": 15}',
            }
        },
    ]
    res = evaluate_tool_calls(model_calls, ground_truth)
    assert res.passed is True
    assert res.matched_calls == 2


# --- failure classification: decides whether grammar decoding could help ---

from benchmarks.bfcl.runner import classify_failure  # noqa: E402


def _resp(content: str, finish: str = "stop", calls: list | None = None) -> dict:
    return {"content": content, "finish_reason": finish, "tool_calls": calls or []}


def test_budget_exhausted_inside_think_is_truncated_not_no_call():
    assert classify_failure(_resp("<think>\nstill going", "length"), "") == "truncated"


def test_prose_answer_is_no_call():
    assert classify_failure(_resp("<think>\nx\n</think>\nThe area is 25."), "") == "no_call"


def test_tool_call_tag_with_nothing_parsed_is_unparseable():
    assert classify_failure(_resp("</think>\n<tool_call>\ngarbage\n</tool_call>"), "") == "unparseable"


def test_tool_call_tag_only_inside_think_is_no_call_not_unparseable():
    content = "<think>\n<tool_call>draft</tool_call>\n</think>\nAnswer in prose."
    assert classify_failure(_resp(content), "") == "no_call"


def test_well_formed_but_wrong_is_bucketed_by_evaluator_reason():
    call = [{"function": {"name": "f", "arguments": "{}"}}]
    assert classify_failure(_resp("x", calls=call), "Count mismatch: generated 1 calls, expected 2") == "wrong_count"
    name_err = "No matching tool call for 'g': Name mismatch: got 'f'"
    arg_err = "No matching tool call for 'f': Param 'a' value mismatch"
    assert classify_failure(_resp("x", calls=call), name_err) == "wrong_function"
    assert classify_failure(_resp("x", calls=call), arg_err) == "wrong_args"


# --- nested-argument matching (BFCL per-leaf allowed-list format) ---

import json  # noqa: E402

from benchmarks.bfcl.evaluator import evaluate_tool_calls as _eval  # noqa: E402


def _call(name: str, args: dict) -> list[dict]:
    return [{"function": {"name": name, "arguments": json.dumps(args)}}]


def test_nested_dict_argument_matches_per_leaf_allowed_lists():
    # Real case simple_94: this exact answer used to be scored wrong.
    truth = [{"update_user_info": {"user_id": [43523],
                                   "update_info": [{"name": ["John Doe"], "email": ["johndoe@email.com"]}],
                                   "database": ["CustomerInfo", ""]}}]
    info = {"email": "johndoe@email.com", "name": "John Doe"}
    ok = _call("update_user_info", {"user_id": 43523, "update_info": info})
    assert _eval(ok, truth).passed


def test_list_of_dicts_argument_matches_elementwise():
    # Real case simple_96.
    truth = [{"database.query": {"table": ["user"], "conditions": [[
        {"field": ["age"], "operation": [">"], "value": ["25"]},
        {"field": ["job"], "operation": ["="], "value": ["engineer"]}]]}}]
    ok = _call("database.query", {"table": "user", "conditions": [
        {"field": "age", "operation": ">", "value": "25"}, {"field": "job", "operation": "=", "value": "engineer"}]})
    assert _eval(ok, truth).passed


def test_nested_dict_still_rejects_wrong_leaf_and_hallucinated_key():
    truth = [{"f": {"d": [{"name": ["John Doe"]}]}}]
    assert not _eval(_call("f", {"d": {"name": "Jane"}}), truth).passed
    assert not _eval(_call("f", {"d": {"name": "John Doe", "extra": 1}}), truth).passed


def test_nested_optional_leaf_may_be_omitted():
    truth = [{"f": {"d": [{"name": ["John Doe"], "age": ["", 30]}]}}]
    assert _eval(_call("f", {"d": {"name": "John Doe"}}), truth).passed
