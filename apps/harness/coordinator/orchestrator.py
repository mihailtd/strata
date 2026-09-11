"""Master Harness Coordinator & Subagent Orchestrator."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.coordinator.planner import HarnessPlanner
from harness.coordinator.subagent import LoRASubagent
from harness.coordinator.types import (
    CoordinatedProjectManifest,
    DomainSpecialist,
    SubagentResult,
    SubagentTask,
)
from harness.state_compactor import SemanticStateCompactor
from runtime.adapter_stacker import DynamicAdapterStacker
from runtime.long_context_engine import LongContextAgentEngine

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class HarnessCoordinator:
    """Orchestrates multi-subagent pipelines using specialized LoRA adapters."""

    def __init__(
        self,
        model_name: str = "ornith-1.5:35b",
        adapters_dir: Path = REPO_ROOT / "results" / "adapters",
    ):
        self.model_name = model_name
        self.adapters_dir = Path(adapters_dir)
        self.planner = HarnessPlanner()
        self.stacker = DynamicAdapterStacker(self.adapters_dir)
        self.engine = LongContextAgentEngine(model_name=model_name, max_context=32768, kv_quant_bits=4)
        self.compactor = SemanticStateCompactor(max_raw_tool_lines=15, preserve_last_n_turns=2)

    def execute_goal(self, goal: str, project_dir: Path) -> CoordinatedProjectManifest:
        """Plans, provisions LoRA subagents, executes pipeline, and verifies with real OS tools."""
        t_start = time.perf_counter()
        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "__init__.py").write_text('"""Coordinated Project Package."""\n')

        print("=" * 105)
        print("🤖 HARNESS MASTER COORDINATOR: LAUNCHING SUBAGENT ORCHESTRATION PIPELINE")
        print(f"   Goal       : '{goal}'")
        print(f"   Project Dir: {project_dir}")
        print(f"   Engine     : {self.model_name} (24 GB Zero-Spill 32k)")
        print("=" * 105, flush=True)

        # 1. Plan Subagent DAG
        tasks = self.planner.plan_project(goal, project_dir)
        print(f"\n📋 [PLANNER] Generated {len(tasks)} Specialized Subagent Tasks:")
        for idx, t in enumerate(tasks, 1):
            exp_desc = ", ".join(f"{k} ({v*100:.0f}%)" for k, v in t.adapter_weights.items())
            print(f"   {idx}. [{t.specialist.value.upper()}] {t.title} ➔ LoRA: {exp_desc}")

        manifest = CoordinatedProjectManifest(goal=goal, project_dir=project_dir, tasks=tasks)
        existing_artifacts: Dict[str, str] = {}

        # 2. Sequential Subagent Execution
        for idx, task in enumerate(tasks, 1):
            print("\n" + "-" * 105)
            print(f"▶ [SUBAGENT {idx}/{len(tasks)}] Executing: {task.title}")
            print(f"  • Specialist LoRA : {task.specialist.value} ({task.adapter_weights})")
            print("-" * 105)

            subagent = LoRASubagent(task, self.engine, self.stacker, project_dir)
            result = subagent.execute(existing_artifacts)
            manifest.results.append(result)
            manifest.total_tokens += result.tokens_generated

            print(f"  ✅ Completed in {result.tok_s:5.1f} tok/s | TTFT: {result.ttft_ms:5.1f}ms | Swap Latency: {result.swap_latency_ms}ms")
            for art in result.artifacts_created:
                p = Path(art)
                if p.exists():
                    rel_name = p.relative_to(project_dir).as_posix()
                    existing_artifacts[rel_name] = p.read_text()
                    print(f"     📁 Artifact Created: `{rel_name}` ({p.stat().st_size} bytes)")

        # 3. Real OS Tool Verification (Ruff & Pytest)
        print("\n" + "=" * 105)
        print("⚡ [VERIFICATION] EXECUTING REAL OPERATING SYSTEM TOOLS")
        print("=" * 105)

        # Format and Lint
        self._run_command("uvx ruff format .", cwd=project_dir)
        lint_res = self._run_command("uvx ruff check --fix .", cwd=project_dir)
        print(f"  • Ruff Linter Exit Code: {lint_res['exit_code']} ({lint_res['elapsed_s']}s)")
        if lint_res["stdout"]:
            for l in lint_res["stdout"].splitlines()[:5]:
                print(f"    {l}")

        # Pytest
        test_res = self._run_command("uv run pytest tests/ -v", cwd=project_dir)
        print(f"\n  • Pytest Exit Code: {test_res['exit_code']} ({test_res['elapsed_s']}s)")
        for l in test_res["stdout"].splitlines():
            print(f"    {l}")

        manifest.all_tests_passed = (test_res["exit_code"] == 0)
        manifest.total_duration_s = round(time.perf_counter() - t_start, 2)

        # 4. Save Manifest & Telemetry Log
        log_file = REPO_ROOT / "results" / "benchmarks" / "harness_coordinator_execution_log.json"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            json.dump(
                {
                    "goal": manifest.goal,
                    "project_dir": str(manifest.project_dir),
                    "total_tokens": manifest.total_tokens,
                    "total_duration_s": manifest.total_duration_s,
                    "all_tests_passed": manifest.all_tests_passed,
                    "tasks_count": len(manifest.tasks),
                    "subagent_telemetry": [
                        {
                            "task_id": r.task_id,
                            "specialist": r.specialist.value,
                            "tok_s": r.tok_s,
                            "ttft_ms": r.ttft_ms,
                            "swap_latency_ms": r.swap_latency_ms,
                            "tokens": r.tokens_generated,
                            "artifacts": r.artifacts_created,
                        }
                        for r in manifest.results
                    ],
                },
                f,
                indent=2,
            )

        print("\n" + "=" * 105)
        print("🏆 HARNESS COORDINATOR PIPELINE COMPLETE")
        print(f"   • Total Subagents Executed : {len(manifest.tasks)}")
        print(f"   • Real OS Verification     : {'✅ ALL TESTS PASSED' if manifest.all_tests_passed else '⚠️ VERIFICATION INCOMPLETE'}")
        print(f"   • Total Pipeline Duration  : {manifest.total_duration_s}s")
        print(f"   • Execution Telemetry Log  : {log_file}")
        print("=" * 105)

        return manifest

    def _run_command(self, cmd: str, cwd: Path) -> Dict[str, Any]:
        """Runs a real OS command on the host filesystem."""
        t0 = time.perf_counter()
        res = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return {
            "command": cmd,
            "exit_code": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }
