"""Unit tests for Toolery deterministic tool-use benchmark loader and evaluator."""

from __future__ import annotations

import pytest

from benchmarks.toolery import (
    SCENARIOS_DIR,
    TIERS,
    evaluate,
    list_scenario_paths,
    load_all_scenarios,
    load_scenario,
)
from toolery.core.models import CheckResult, ScenarioResult, ToolCall, TraceResult


def test_toolery_scenarios_count():
    assert SCENARIOS_DIR.exists()
    all_paths = list_scenario_paths()
    # 143 total scenarios: 40 easy, 45 medium, 34 hard, 24 very_hard
    assert len(all_paths) == 143

    assert len(list_scenario_paths("easy")) == 40
    assert len(list_scenario_paths("medium")) == 45
    assert len(list_scenario_paths("hard")) == 34
    assert len(list_scenario_paths("very_hard")) == 24


def test_load_single_scenario():
    sc_path = SCENARIOS_DIR / "easy" / "easy-01-direct-weather.yaml"
    assert sc_path.exists()
    sc = load_scenario(sc_path)
    assert sc.id == "easy-01-direct-weather"
    assert sc.tier == "easy"
    assert "get_weather" in sc.tools
    assert "Warsaw" in sc.prompt
    assert sc.budget.max_tool_calls == 1


def test_evaluate_mock_passing_trace():
    sc_path = SCENARIOS_DIR / "easy" / "easy-01-direct-weather.yaml"
    sc = load_scenario(sc_path)

    # Simulate a trace that called get_weather and answered properly
    trace = TraceResult(
        scenario_id=sc.id,
        adapter="test",
        trial_index=0,
        messages=[],
        final_response="The current temperature in Warsaw is 7°C and it is cloudy.",
        tool_calls=[
            ToolCall(
                index=0,
                name="get_weather",
                args={"location": "Warsaw"},
                result={"temp_c": 7, "condition": "cloudy"},
                result_kind="json",
                latency_ms=10.0,
            )
        ],
        started_at_iso="2026-09-21T20:00:00Z",
        duration_ms=120.0,
    )

    res: ScenarioResult = evaluate(sc, trace)
    assert res.status == "pass"
    assert res.score == 1.0
    assert res.call_count == 1


def test_evaluate_mock_failing_trace():
    sc_path = SCENARIOS_DIR / "easy" / "easy-01-direct-weather.yaml"
    sc = load_scenario(sc_path)

    # Hallucinated answer with 0 tool calls
    trace = TraceResult(
        scenario_id=sc.id,
        adapter="test",
        trial_index=0,
        messages=[],
        final_response="I think it's sunny and 25°C in Warsaw.",
        tool_calls=[],
        started_at_iso="2026-09-21T20:00:00Z",
        duration_ms=80.0,
    )

    res: ScenarioResult = evaluate(sc, trace)
    assert res.status == "fail"
    assert res.score == 0.0
