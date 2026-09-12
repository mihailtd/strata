"""LoRA-Specialized Subagent Execution Unit.

Drives a real DSH agent (bash + editor tools) to write its assigned
artifact(s) in `project_dir`, then verifies what it actually produced --
no template fallback. An earlier version of this file silently substituted
a pre-written "reference" implementation whenever the model's real output
didn't match a hardcoded set of expected substrings (e.g. requiring the
exact string "class VectorRecord" to appear), which meant the artifacts
`orchestrator.py` went on to lint/test were frequently NOT what the model
generated at all -- the model call still happened and its tok/s got logged,
but the code being graded was fabricated. That's the "Zero-Simulation
Mandate" violation this repo's own README already retracted once (the
SWE-bench mock-delay incident). See docs/DECISIONS.md and
`evals/dsh_agent/README.md` for the same principle applied elsewhere.

NOT YET LIVE-VERIFIED end-to-end (no model server was running when this was
written) -- see evals/dsh_agent/README.md's smoke-check step, which applies
here too before trusting a real run.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

from harness.coordinator.types import DomainSpecialist, SubagentResult, SubagentTask
from runtime.adapter_stacker import DynamicAdapterStacker
from runtime.long_context_engine import LongContextAgentEngine

SPECIALIST_SYSTEM_PROMPTS = {
    DomainSpecialist.ASTRAL: (
        "You are an Astral ecosystem tooling specialist. "
        "Strictly generate valid `pyproject.toml` configurations using `[project]` and `[tool.ruff.lint]` "
        "with explicit `select` rules, target-version py312, and zero legacy setuptools/flake8 cruft."
    ),
    DomainSpecialist.PYTHON_MODERN: (
        "You are a modern Python 3.12+ specialist. "
        "Strictly use PEP 695 type parameter syntax (`class Box[T]:`, `type Alias[T] = ...`, `def func[T](...)`) "
        "and Pydantic v2 `model_config = ConfigDict(from_attributes=True)` without legacy TypeVar."
    ),
    DomainSpecialist.POSTGRESQL: (
        "You are a PostgreSQL 17 and pgvector principal architect. "
        "Strictly write asyncpg database modules with connection pools, parameterized `$1` queries, "
        "and HNSW index creation using the Cosine distance operator `<=>`."
    ),
    DomainSpecialist.DUCKDB: (
        "You are a DuckDB vectorized OLAP analytics specialist. "
        "Strictly write high-performance in-memory SQL queries utilizing native `QUALIFY` window clauses, "
        "ROW_NUMBER() ranking, and percentile calculations."
    ),
    DomainSpecialist.PYTHON_WEB: (
        "You are a FastAPI async web architect. "
        "Strictly use `@asynccontextmanager async def lifespan(app: FastAPI):` for lifecycle management, "
        "clean dependency injection with `Annotated`, and Pydantic v2 request/response validation."
    ),
    DomainSpecialist.FINANCIAL_PLANNING: (
        "You are a quantitative financial risk modeling expert. "
        "Strictly implement vectorized numpy Value-at-Risk (VaR) and Conditional VaR (Expected Shortfall) algorithms."
    ),
    DomainSpecialist.STACKED_CROSS_DOMAIN: (
        "You are a cross-domain integration engineering specialist with multi-expert domain knowledge. "
        "Synthesize FastAPI lifespan architectures, PostgreSQL pgvector cosine queries (<=>), DuckDB QUALIFY window analytics, "
        "and Pydantic v2 models into clean, isolated Pytest unit tests."
    ),
    DomainSpecialist.GENERAL: (
        "You are an expert autonomous software engineer writing clean, robust, modern Python code."
    ),
}

# str_replace_editor is opt-in for sdk-minimal (bash-only by default) -- see
# evals/dsh_agent/editor.patch.yml, the same patch reused here.
EDITOR_PATCH = Path(__file__).resolve().parents[3] / "evals" / "dsh_agent" / "editor.patch.yml"


class LoRASubagent:
    """An autonomous subagent pinned to a specific LoRA adapter specialty."""

    def __init__(
        self,
        task: SubagentTask,
        engine: LongContextAgentEngine,
        stacker: DynamicAdapterStacker,
        project_dir: Path,
    ):
        self.task = task
        self.engine = engine
        self.stacker = stacker
        self.project_dir = project_dir

    def execute(self, existing_artifacts: dict[str, str]) -> SubagentResult:
        """Drives a real DSH agent to write this task's artifact(s), then verifies what it actually wrote."""
        import time

        t_start = time.perf_counter()

        # 1. Measure and perform In-Place Adapter Fusion/Swap (unchanged -- real math, real latency).
        t_swap = time.perf_counter()
        if len(self.task.adapter_weights) > 1:
            out_file = self.project_dir / ".adapters" / f"{self.task.task_id}.safetensors"
            self.stacker.stack_adapters(self.task.adapter_weights, output_path=out_file)
        swap_latency_ms = round((time.perf_counter() - t_swap) * 1000.0, 2)

        base_system_prompt = self.task.system_override or SPECIALIST_SYSTEM_PROMPTS.get(
            self.task.specialist, SPECIALIST_SYSTEM_PROMPTS[DomainSpecialist.GENERAL]
        )
        prompt = (
            f"{base_system_prompt}\n\n"
            f"### TASK: {self.task.title}\n{self.task.prompt}\n\n"
            f"Create exactly these file(s), relative to the current directory: "
            f"{', '.join(self.task.expected_artifacts)}. "
            "Use the editor or bash to write them directly -- do not just print code in your response."
        )
        if existing_artifacts:
            context = "\n".join(
                f"--- Existing file `{name}` ---\n{content}" for name, content in existing_artifacts.items()
            )
            prompt = f"### EXISTING PROJECT ARTIFACTS:\n{context}\n\n{prompt}"

        generated_text = ""
        tool_calls = 0
        error_message = None
        try:
            from deepseek_harness import DeepSeekHarness

            dsh_home = self.project_dir / ".dsh_home" / self.task.task_id
            with DeepSeekHarness(
                provider="deepseek-official",
                model=self.engine.model_name,
                cwd=str(self.project_dir),
                dsh_home=str(dsh_home),
                profile="sdk-minimal",
                patches=(str(EDITOR_PATCH),) if EDITOR_PATCH.exists() else (),
            ) as harness:
                result = harness.run(prompt, session_id=f"subagent-{self.task.task_id}")
            generated_text = getattr(result, "final_response", "") or str(result)
            tool_calls = self._count_tool_calls(dsh_home)
        except Exception as exc:  # noqa: BLE001 -- any harness failure is an honest subagent failure, not a crash
            error_message = str(exc)

        # 2. Verify what was ACTUALLY written -- no fallback, no substitution.
        created_files: list[str] = []
        all_valid = True
        for artifact_rel_path in self.task.expected_artifacts:
            target_file = self.project_dir / artifact_rel_path
            if not target_file.exists():
                all_valid = False
                continue
            created_files.append(str(target_file))
            if not self._is_syntactically_valid(artifact_rel_path, target_file.read_text()):
                all_valid = False

        duration_s = round(time.perf_counter() - t_start, 2)
        success = error_message is None and all_valid and bool(created_files)
        if error_message is None and not success:
            error_message = "expected artifact(s) missing or syntactically invalid after the agent run"

        return SubagentResult(
            task_id=self.task.task_id,
            specialist=self.task.specialist,
            success=success,
            generated_text=generated_text,
            artifacts_created=created_files,
            swap_latency_ms=swap_latency_ms,
            duration_s=duration_s,
            tool_calls=tool_calls,
            error_message=error_message,
        )

    def _count_tool_calls(self, dsh_home: Path) -> int:
        """Counts tool/call events in this run's session JSONL, for telemetry only."""
        sessions_dir = dsh_home / "sessions"
        if not sessions_dir.exists():
            return 0
        jsonl_files = sorted(sessions_dir.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime)
        if not jsonl_files:
            return 0
        import json

        count = 0
        for line in jsonl_files[-1].read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                if json.loads(line).get("type") == "tool/call":
                    count += 1
            except json.JSONDecodeError:
                continue
        return count

    def _is_syntactically_valid(self, filename: str, content: str) -> bool:
        """Honest syntax check -- no content-shape requirements, no substring matching."""
        if filename.endswith(".toml"):
            try:
                tomllib.loads(content)
                return True
            except Exception:
                return False
        if filename.endswith(".py"):
            try:
                ast.parse(content)
                return True
            except SyntaxError:
                return False
        return bool(content.strip())
