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
