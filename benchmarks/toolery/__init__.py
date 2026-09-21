"""Toolery deterministic tool-use benchmark suite (143 scenarios across 4 difficulty tiers)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure vendor toolery package is importable
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import toolery
import toolery.tools.generic
import toolery.tools.terminal
import toolery.tools.api_db
import toolery.tools.domain
from toolery.core.models import Scenario, ScenarioResult, TraceResult
from toolery.core.scenario import load_scenario
from toolery.core.scorer import evaluate

SCENARIOS_DIR = PACKAGE_DIR / "scenarios"
TIERS = ["easy", "medium", "hard", "very_hard"]


def list_scenario_paths(tier: str | None = None) -> list[Path]:
    """Returns sorted list of YAML scenario paths, optionally filtered by tier."""
    if not SCENARIOS_DIR.exists():
        return []
    target_tiers = [tier] if tier and tier in TIERS else TIERS
    paths = []
    for t in target_tiers:
        tier_dir = SCENARIOS_DIR / t
        if tier_dir.exists():
            paths.extend(sorted(tier_dir.glob("*.yaml")))
    return paths


def load_all_scenarios(tier: str | None = None) -> list[Scenario]:
    """Loads all scenarios, optionally filtered by tier."""
    paths = list_scenario_paths(tier)
    return [load_scenario(p) for p in paths]


__all__ = [
    "TIERS",
    "SCENARIOS_DIR",
    "Scenario",
    "ScenarioResult",
    "TraceResult",
    "load_scenario",
    "load_all_scenarios",
    "list_scenario_paths",
    "evaluate",
]
