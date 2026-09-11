"""Type definitions for Harness Coordinator & LoRA Subagent Orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    adapter_weights: Dict[str, float]
    prompt: str
    expected_artifacts: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    system_override: Optional[str] = None


@dataclass
class SubagentResult:
    """The execution output of a specialized subagent."""
    task_id: str
    specialist: DomainSpecialist
    success: bool
    generated_text: str
    artifacts_created: List[str]
    tokens_generated: int
    tok_s: float
    ttft_ms: float
    swap_latency_ms: float
    tool_output: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class CoordinatedProjectManifest:
    """The complete project state managed by the Harness Coordinator."""
    goal: str
    project_dir: Path
    tasks: List[SubagentTask]
    results: List[SubagentResult] = field(default_factory=list)
    total_tokens: int = 0
    total_duration_s: float = 0.0
    all_tests_passed: bool = False
