"""Multi-language execution environment for Aider Polyglot Benchmark exercises."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from .polyglot_tasks import PolyglotTask


@dataclass
class PolyglotResult:
    passed: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    error_summary: str

    @property
    def feedback_message(self) -> str:
        output = (self.stdout + "\n" + self.stderr).strip()
        return (
            f"Test verification FAILED (exit code {self.exit_code}):\n\n"
            f"```\n{output}\n```\n\n"
            "Please fix the errors and provide an updated solution."
        )


class PolyglotExecutor:
    """Executes multi-language test suites in an isolated temporary directory."""

    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds

    def execute(
        self,
        task: PolyglotTask,
        edited_files: dict[str, str],
    ) -> PolyglotResult:
        with tempfile.TemporaryDirectory(prefix="aider_polyglot_") as tmp_dir:
            env_path = Path(tmp_dir)

            # Copy all files from original task directory
            for rel_path, content in task.all_files.items():
                target_file = env_path / rel_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(content, encoding="utf-8")

            # Apply edited solution files
            for rel_path, content in edited_files.items():
                target_file = env_path / rel_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(content, encoding="utf-8")

            # Restore fresh test files to prevent accidental test alteration
            for rel_path, content in task.test_files.items():
                target_file = env_path / rel_path
                target_file.parent.mkdir(parents=True, exist_ok=True)
                target_file.write_text(content, encoding="utf-8")

            t0 = time.perf_counter()
            cmd = list(task.test_command)
            binary = cmd[0]

            # Check if required runtime/compiler exists on host
            if not shutil.which(binary) and not (env_path / binary).exists():
                return PolyglotResult(
                    passed=False,
                    exit_code=127,
                    stdout="",
                    stderr=f"Required test toolchain '{binary}' not found on host environment.",
                    duration_s=0.0,
                    error_summary=f"Missing toolchain: {binary}",
                )

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=env_path,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                duration = time.perf_counter() - t0
                passed = proc.returncode == 0
                error_summary = "" if passed else (proc.stderr or proc.stdout)[-300:].strip()

                return PolyglotResult(
                    passed=passed,
                    exit_code=proc.returncode,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    duration_s=duration,
                    error_summary=error_summary,
                )
            except subprocess.TimeoutExpired as exc:
                return PolyglotResult(
                    passed=False,
                    exit_code=-1,
                    stdout=exc.stdout or "",
                    stderr=exc.stderr or "Execution timed out.",
                    duration_s=time.perf_counter() - t0,
                    error_summary=f"Timeout after {self.timeout_seconds}s",
                )
            except Exception as exc:
                return PolyglotResult(
                    passed=False,
                    exit_code=-2,
                    stdout="",
                    stderr=str(exc),
                    duration_s=time.perf_counter() - t0,
                    error_summary=str(exc),
                )
