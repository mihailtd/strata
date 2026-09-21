"""Sandboxed pytest execution environment supporting both local subprocess and Docker/Podman container modes."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PytestResult:
    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    error_summary: str

    @property
    def feedback_message(self) -> str:
        """Concise feedback message with only failing assertion/error lines.

        Strips pytest boilerplate (collection banners, PASSED lines, summary bars)
        and returns only the lines that matter for a repair prompt.
        """
        summary = self.error_summary or (self.stdout + "\n" + self.stderr).strip()
        return (
            f"Tests failed (exit code {self.exit_code}):\n\n"
            f"```\n{summary}\n```\n\n"
            "Fix the errors and output the corrected complete Python file."
        )


class PytestExecutor:
    """Executes pytest against test suites in isolated local temp dirs or containers."""

    def __init__(
        self,
        use_container: bool = False,
        container_image: str = "python:3.11-slim",
        timeout_seconds: int = 15,
        container_engine: str | None = None,
    ):
        self.use_container = use_container
        self.container_image = container_image
        self.timeout_seconds = timeout_seconds
        self.container_engine = container_engine or self._detect_container_engine()

    @staticmethod
    def _detect_container_engine() -> str:
        """Detects podman or docker binary on host."""
        if shutil.which("podman"):
            return "podman"
        if shutil.which("docker"):
            return "docker"
        return "docker"

    def build_container_command(self, host_dir: Path, test_file_name: str) -> list[str]:
        """Constructs the container execution CLI command."""
        return [
            self.container_engine,
            "run",
            "--rm",
            "-v",
            f"{host_dir.resolve()}:/workspace:rw,Z",
            "-w",
            "/workspace",
            self.container_image,
            "pytest",
            test_file_name,
            "-v",
            "--tb=short",
        ]

    def execute(
        self,
        target_file_name: str,
        target_code: str,
        test_file_name: str,
        test_code: str,
    ) -> PytestResult:
        """Runs pytest on the provided target and test code in an isolated environment."""
        with tempfile.TemporaryDirectory(prefix="aider_bench_") as tmp_dir:
            env_path = Path(tmp_dir)

            # Write target module
            (env_path / target_file_name).write_text(target_code, encoding="utf-8")

            # Write test file
            (env_path / test_file_name).write_text(test_code, encoding="utf-8")

            # Create an empty __init__.py so relative imports work seamlessly
            (env_path / "__init__.py").write_text("", encoding="utf-8")

            t0 = time.perf_counter()

            if self.use_container:
                cmd = self.build_container_command(env_path, test_file_name)
            else:
                cmd = [
                    sys.executable,
                    "-m",
                    "pytest",
                    test_file_name,
                    "-v",
                    "--tb=short",
                ]

            try:
                res = subprocess.run(
                    cmd,
                    cwd=str(env_path),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                duration = time.perf_counter() - t0
                passed = res.returncode == 0
                stdout = res.stdout
                stderr = res.stderr
                exit_code = res.returncode

                # Generate concise error summary if failed
                error_summary = ""
                if not passed:
                    output = stdout + "\n" + stderr
                    for line in output.splitlines():
                        if "FAILED" in line or "ERROR" in line or "AssertionError" in line:
                            error_summary += line + "\n"
                    if not error_summary:
                        error_summary = "\n".join(output.splitlines()[-10:])

                return PytestResult(
                    passed=passed,
                    exit_code=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    duration_s=duration,
                    error_summary=error_summary.strip(),
                )

            except subprocess.TimeoutExpired:
                duration = time.perf_counter() - t0
                return PytestResult(
                    passed=False,
                    exit_code=-1,
                    stdout="",
                    stderr=f"Pytest timed out after {self.timeout_seconds} seconds.",
                    duration_s=duration,
                    error_summary=f"TimeoutExpired: {self.timeout_seconds}s",
                )
            except Exception as e:
                duration = time.perf_counter() - t0
                return PytestResult(
                    passed=False,
                    exit_code=-2,
                    stdout="",
                    stderr=f"Executor exception: {e}",
                    duration_s=duration,
                    error_summary=str(e),
                )
