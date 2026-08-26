"""Autonomous SWE-Bench Execution Harness.

Provides dynamic environment isolation, multi-turn repair loops, pytest verification,
and detailed timing/token telemetry.
"""

from __future__ import annotations

import ast
import asyncio
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.swe_bench.tasks import SWEBenchTask, SWE_BENCH_TASKS


@dataclass
class TaskAttemptResult:
    task_id: str
    domain: str
    arm: str  # "ollama_generalist_27b" vs "our_specialist_lora_27b"
    passed: bool
    pass_turn: int  # 1 for Pass@1, 2 for Turn 2, 0 if failed
    total_turns: int
    wall_clock_time_s: float
    total_generated_tokens: int
    average_throughput_tok_s: float
    ttft_ms_first_turn: float
    ttft_ms_last_turn: float
    error_trace: Optional[str] = None
    generated_patch: str = ""


class SWEBenchHarness:
    """Executes SWE-Bench tasks with multi-turn autonomous refinement."""

    def __init__(self, temp_dir: Optional[Path] = None):
        self.base_dir = temp_dir or Path(tempfile.mkdtemp(prefix="swe_bench_"))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def execute_pytest(self, code: str, test_code: str) -> Tuple[bool, str]:
        """Runs pytest on combined code and test suite in an isolated temp directory."""
        with tempfile.TemporaryDirectory(dir=self.base_dir) as tmp_env:
            env_path = Path(tmp_env)
            test_file = env_path / "test_suite.py"

            # Combine code under test with pytest assertions
            full_content = f"{code}\n\n{test_code}\n"
            test_file.write_text(full_content)

            try:
                cmd = [sys.executable, "-m", "pytest", str(test_file), "-q", "--tb=short"]
                res = subprocess.run(
                    cmd,
                    cwd=str(env_path),
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                passed = res.returncode == 0
                output = res.stdout + res.stderr
                return passed, output
            except subprocess.TimeoutExpired:
                return False, "Error: PyTest execution timed out after 15 seconds."
            except Exception as ex:
                return False, f"Harness execution error: {ex}"

    @staticmethod
    def extract_code_block(response_text: str) -> str:
        """Extracts Python code from Markdown code fences (```python ... ```)."""
        if "```python" in response_text:
            parts = response_text.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()
        elif "```" in response_text:
            parts = response_text.split("```")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()
        return response_text.strip()
