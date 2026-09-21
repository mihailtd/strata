"""Agentic DSH harness integration testing autonomous tool-calling and self-verification."""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from benchmarks.aider_bench.executor import PytestExecutor, PytestResult
from benchmarks.aider_bench.tasks import AiderTask

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORDIS_PATCH = REPO_ROOT / "apps" / "harness" / "cordis.patch.yml"


@dataclass
class AgenticTaskResult:
    task_name: str
    passed: bool
    total_duration_s: float
    tool_calls_count: int
    final_response: str
    pytest_result: PytestResult | None
    error_message: str | None = None


class AgenticHarness:
    """Evaluates tasks by driving the DeepSeekHarness agent with custom verification plugins."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8003/v1",
        model: str = "qwen3.5:4b-rust",
        executor: PytestExecutor | None = None,
        timeout: int = 180,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.executor = executor or PytestExecutor()
        self.timeout = timeout

    def run_task(self, task: AiderTask) -> AgenticTaskResult:
        """Executes a task inside an isolated temporary directory using DeepSeekHarness."""
        from deepseek_harness import DeepSeekHarness

        t_start = time.perf_counter()

        with tempfile.TemporaryDirectory(prefix="aider_agentic_") as tmp_workspace:
            ws_path = Path(tmp_workspace)
            dsh_home = ws_path / ".dsh"
            dsh_home.mkdir(parents=True, exist_ok=True)

            # Write the initial stub and test files
            target_path = ws_path / task.target_file_name
            target_path.write_text(task.stub_code, encoding="utf-8")

            test_path = ws_path / task.test_file_name
            test_path.write_text(task.test_code, encoding="utf-8")

            # Create pyproject.toml so tool-code-verify detects Python pytest framework
            (ws_path / "pyproject.toml").write_text(
                "[project]\nname = 'aider-task'\nversion = '0.1.0'\n\n[tool.pytest.ini_options]\npython_files = '*_test.py'\n",
                encoding="utf-8",
            )

            os.environ["DEEPSEEK_API_KEY"] = "sk-local-bench"
            os.environ["DEEPSEEK_BASE_URL"] = self.base_url

            prompt = (
                f"Solve this Python exercise in `{task.target_file_name}`.\n\n"
                f"Instructions:\n{task.instructions}\n\n"
                f"Current `{task.target_file_name}` contents:\n```python\n{task.stub_code}\n```\n\n"
                f"Test suite `{task.test_file_name}` is already in the workspace.\n"
                "Use your tools (like tool-code-verify) to verify and ensure all tests pass."
            )

            patches = (str(CORDIS_PATCH),) if CORDIS_PATCH.exists() else ()
            final_response = ""
            error = None
            tool_calls = 0

            try:
                with DeepSeekHarness(
                    provider="runtime-next" if "8003" in self.base_url else "deepseek-official",
                    model=self.model,
                    cwd=str(ws_path),
                    dsh_home=str(dsh_home),
                    profile="sdk-minimal",
                    patches=patches,
                ) as harness:
                    session_id = f"bench_{task.name}_{int(time.time())}"
                    result = harness.run(prompt, session_id=session_id)
                    final_response = getattr(result, "final_response", "") or str(result)
            except Exception as ex:
                error = str(ex)

            # Independently verify what was actually written to the file
            updated_code = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
            pytest_res = self.executor.execute(
                target_file_name=task.target_file_name,
                target_code=updated_code,
                test_file_name=task.test_file_name,
                test_code=task.test_code,
            )

            total_duration = time.perf_counter() - t_start
            return AgenticTaskResult(
                task_name=task.name,
                passed=pytest_res.passed,
                total_duration_s=total_duration,
                tool_calls_count=tool_calls,
                final_response=final_response,
                pytest_result=pytest_res,
                error_message=error,
            )
