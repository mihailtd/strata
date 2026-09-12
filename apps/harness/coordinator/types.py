"""Type definitions for Harness Coordinator & LoRA Subagent Orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class DomainSpecialist(str, Enum):
    POSTGRESQL = "postgresql"
    PYTHON_WEB = "python_web"
    DUCKDB = "duckdb"
    ASTRAL = "astral"
    PYTHON_MODERN = "python_modern"
    FINANCIAL_PLANNING = "financial_planning"
    STACKED_CROSS_DOMAIN = "stacked_cross_domain"
    GENERAL = "general"


@dataclass
class SubagentTask:
    """A discrete unit of work assigned to a LoRA-specialized subagent."""

    task_id: str
    title: str
    specialist: DomainSpecialist
    adapter_weights: dict[str, float]
    prompt: str
    expected_artifacts: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    system_override: str | None = None


@dataclass
class SubagentResult:
    """The execution output of a specialized subagent.

    `success` means the expected artifact(s) actually exist on disk and are
    syntactically valid -- never a hardcoded assumption. There is no template
    fallback: if the agent didn't produce a valid artifact, this is False and
    `error_message` says why.
    """

    task_id: str
    specialist: DomainSpecialist
    success: bool
    generated_text: str
    artifacts_created: list[str]
    swap_latency_ms: float
    duration_s: float
    tool_calls: int
    tool_output: str | None = None
    error_message: str | None = None


@dataclass
class CoordinatedProjectManifest:
    """The complete project state managed by the Harness Coordinator."""

    goal: str
    project_dir: Path
    tasks: list[SubagentTask]
    results: list[SubagentResult] = field(default_factory=list)
    total_duration_s: float = 0.0
    all_tests_passed: bool = False
