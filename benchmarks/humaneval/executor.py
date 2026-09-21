"""Safe isolated execution harness for HumanEval code evaluations."""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

from .dataset import HumanEvalProblem


@dataclass
class ExecutionResult:
    passed: bool
    error: str
    duration_s: float
    stdout: str
    stderr: str


def clean_code(raw_response: str, entry_point: str, prompt: str) -> str:
    """Extracts executable Python code from raw model response."""
    text = raw_response.strip()

    # If wrapped in markdown code blocks, extract the python block
    code_blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if code_blocks:
        # Choose block containing entry_point if possible
        for block in code_blocks:
            if f"def {entry_point}" in block:
                return block
        return code_blocks[0]

    # If it begins with def entry_point or has it
    if f"def {entry_point}" in text:
        # Keep everything from first def entry_point onwards up to next markdown or stop
        idx = text.index(f"def {entry_point}")
        return text[idx:]

    # Otherwise assume it is completion to prompt
    return prompt + "\n" + text


class HumanEvalExecutor:
    """Executes candidate solution against the HumanEval test harness in an isolated subprocess."""

    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout_seconds = timeout_seconds

    def execute(self, problem: HumanEvalProblem, candidate_code: str) -> ExecutionResult:
        full_candidate = clean_code(candidate_code, problem.entry_point, problem.prompt)

        # Assemble full test script
        test_script = (
            f"{full_candidate}\n\n"
            f"{problem.test}\n\n"
            f"check({problem.entry_point})\n"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(test_script)
            tmp_path = tmp.name

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-u", tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            duration = time.perf_counter() - t0
            passed = proc.returncode == 0
            error = "" if passed else (proc.stderr or proc.stdout).strip()

            return ExecutionResult(
                passed=passed,
                error=error,
                duration_s=duration,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - t0
            return ExecutionResult(
                passed=False,
                error=f"TimeoutExpired after {self.timeout_seconds}s",
                duration_s=duration,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "Timed out",
            )
        except Exception as exc:
            duration = time.perf_counter() - t0
            return ExecutionResult(
                passed=False,
                error=str(exc),
                duration_s=duration,
                stdout="",
                stderr=str(exc),
            )
        finally:
            import os
            try:
                os.remove(tmp_path)
            except OSError:
                pass
