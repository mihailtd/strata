"""State-Handoff-Aware Harness Coordinator for 27B Triton Engine.

Integrates True O(1) Tensor State Handoff ($S_t$) into the Master Harness Coordinator,
enabling multi-subagent autonomous software engineering pipelines without text re-prefill penalties.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from harness.coordinator.planner import HarnessPlanner
from harness.coordinator.subagent import SPECIALIST_SYSTEM_PROMPTS
from harness.coordinator.types import (
    DomainSpecialist,
)
from runtime.native_27b_engine import Native27BEngine
from runtime.server import get_27b_tokenizer
from runtime.state_handoff_27b import AgentTurn, StateHandoffSession

logger = logging.getLogger("StateHandoffHarness")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] (StateHandoffHarness) %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


SPECIALIST_TO_LORA_MAP = {
    DomainSpecialist.ASTRAL: "astral",
    DomainSpecialist.PYTHON_MODERN: "python_modern",
    DomainSpecialist.POSTGRESQL: "postgresql",
    DomainSpecialist.DUCKDB: "duckdb",
    DomainSpecialist.PYTHON_WEB: "python_web",
    DomainSpecialist.FINANCIAL_PLANNING: "financial",
    DomainSpecialist.STACKED_CROSS_DOMAIN: "postgresql+duckdb+python_web",  # Dynamic Multi-Expert Stack
    DomainSpecialist.GENERAL: "astral",
}


@dataclass
class HarnessTurnTelemetry:
    task_id: str
    specialist: str
    lora_adapter: str
    prefill_ms: float
    decode_ms: float
    total_ms: float
    handoff_ms: float
    lora_swap_ms: float
    prompt_tokens: int
    tokens_avoided: int
    tokens_generated: int
    tok_per_sec: float
    artifacts_created: list[str] = field(default_factory=list)


class StateHandoffHarnessCoordinator:
    """Master Subagent Orchestrator powered by 27B Tensor State Handoff."""

    def __init__(
        self,
        engine: Native27BEngine | None = None,
        clone_on_handoff: bool = True,
    ):
        self.engine = engine or Native27BEngine(num_layers=64)
        if not self.engine.layers or len(self.engine.layers) < 64:
            self.engine.load_from_cache()

        self.planner = HarnessPlanner()
        self.tokenizer = get_27b_tokenizer()
        self.clone_on_handoff = clone_on_handoff

    def _extract_code(self, text: str) -> str:
        """Extracts code enclosed in triple backticks."""
        pattern = r"```(?:[a-zA-Z0-9_\-]+)?\s*([\s\S]*?)```"
        matches = re.findall(pattern, text)
        if matches:
            return matches[0].strip()
        lines = [l for l in text.splitlines() if not l.startswith("<think>") and not l.startswith("</think>")]
        return "\n".join(lines).strip()

    def _extract_file_content(self, raw_text: str, target_rel_path: str) -> str:
        """Extracts whatever code the model actually generated for this artifact.

        No fallback: an earlier version of this method silently substituted a
        pre-written "curated" implementation whenever the model's real output
        didn't parse -- which meant `_verify_project`'s ruff/pytest run was
        frequently grading fabricated code, not the model's. This benchmark's
        actual subject is tensor-handoff *timing* (prefill_ms/decode_ms/
        handoff_ms, all measured from the real forward pass regardless of what
        gets written here), so there's no reason to fake the artifact content
        too -- write exactly what came out, valid or not, and let the real
        verification step report the honest result.
        """
        return self._extract_code(raw_text)

    def execute_project(
        self,
        goal: str,
        project_dir: Path,
        mode: str = "tensor_handoff",  # "tensor_handoff" | "text_reprefill"
        max_tokens_per_subagent: int = 256,
    ) -> tuple[list[HarnessTurnTelemetry], dict[str, Any]]:
        """Executes the subagent project pipeline under either tensor handoff or text re-prefill."""
        project_dir = Path(project_dir)
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "__init__.py").write_text('"""Coordinated Project Package."""\n')

        tasks = self.planner.plan_project(goal, project_dir)
        telemetry_records: list[HarnessTurnTelemetry] = []

        logger.info(f"Starting pipeline execution with mode='{mode}', {len(tasks)} tasks.")

        if mode == "tensor_handoff":
            # Warmup: amortize the first HIP graph capture before the measured turn loop.
            # Without this, Turn 1 of Arm A pays a one-time 2.6s graph-capture cost that
            # Arm B avoids by pre-loading all LoRAs at startup. This makes the comparison fair.
            logger.info("[Warmup] Running dummy forward to prime HIP graph capture...")
            _dummy_tokens = self.tokenizer.encode("<|im_start|>user\nwarmup<|im_end|>\n<|im_start|>assistant\n")
            _, _ws = self.engine.forward_prompt(_dummy_tokens)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("[Warmup] HIP graph capture complete.")

            session = StateHandoffSession(
                engine=self.engine,
                clone_on_handoff=self.clone_on_handoff,
            )
            for idx, task in enumerate(tasks):
                if getattr(task, "adapter_weights", None) and len(task.adapter_weights) >= 2:
                    lora_name = "+".join(task.adapter_weights.keys())
                else:
                    lora_name = SPECIALIST_TO_LORA_MAP.get(task.specialist, "astral")
                sys_prompt = SPECIALIST_SYSTEM_PROMPTS.get(task.specialist, "")
                prompt = (
                    f"{sys_prompt}\n"
                    f"### TASK [{task.task_id}]: {task.title}\n"
                    f"{task.prompt}\n"
                    f"Generate strictly the code for: {task.expected_artifacts[0] if task.expected_artifacts else 'target code'}."
                )

                agent_turn = AgentTurn(
                    agent_id=task.task_id,
                    role=task.title,
                    instruction=prompt,
                    expert_lora=lora_name,
                    max_new_tokens=max_tokens_per_subagent,
                    temperature=0.0,
                )

                turn_res = session.execute_turn(agent_turn)

                # Write artifact(s) to project_dir
                created_artifacts = []
                for art_name in task.expected_artifacts:
                    target_file = project_dir / art_name
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    content = self._extract_file_content(turn_res.output_text, art_name)
                    target_file.write_text(content)
                    created_artifacts.append(str(target_file))

                telemetry_records.append(
                    HarnessTurnTelemetry(
                        task_id=task.task_id,
                        specialist=task.specialist.value,
                        lora_adapter=lora_name,
                        prefill_ms=turn_res.prefill_ms,
                        decode_ms=turn_res.decode_ms,
                        total_ms=turn_res.total_ms,
                        handoff_ms=turn_res.handoff_ms,
                        lora_swap_ms=turn_res.lora_swap_ms,
                        prompt_tokens=turn_res.prefill_tokens,
                        tokens_avoided=turn_res.tokens_avoided,
                        tokens_generated=turn_res.tokens_generated,
                        tok_per_sec=turn_res.tok_per_sec,
                        artifacts_created=created_artifacts,
                    )
                )

        else:  # text_reprefill baseline
            cumulative_history = ""
            for idx, task in enumerate(tasks):
                t_turn_start = time.perf_counter()
                lora_name = SPECIALIST_TO_LORA_MAP.get(task.specialist, "astral")

                # Swap LoRA
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_lora0 = time.perf_counter()
                self.engine.set_active_lora(lora_name)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                lora_swap_ms = (time.perf_counter() - t_lora0) * 1000.0

                sys_prompt = SPECIALIST_SYSTEM_PROMPTS.get(task.specialist, "")
                prompt_delta = (
                    f"{sys_prompt}\n"
                    f"### TASK [{task.task_id}]: {task.title}\n"
                    f"{task.prompt}\n"
                    f"Generate strictly the code for: {task.expected_artifacts[0] if task.expected_artifacts else 'target code'}."
                )

                if idx == 0:
                    cumulative_history = f"<|im_start|>user\n{prompt_delta}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    cumulative_history += (
                        f"<|im_end|>\n<|im_start|>user\n{prompt_delta}<|im_end|>\n<|im_start|>assistant\n"
                    )

                prompt_tokens = self.tokenizer.encode(cumulative_history)

                # Prefill full accumulated context
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_pref0 = time.perf_counter()
                l_first, state_dict = self.engine.forward_prompt(prompt_tokens)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                prefill_ms = (time.perf_counter() - t_pref0) * 1000.0

                # Decode
                first_token = int(torch.argmax(l_first[0, :]).item())
                gen_tokens = [first_token]
                curr_token = first_token
                pos = len(prompt_tokens)
                if self.engine.hip_graph_captured:
                    self.engine.sync_states_to_graphs(state_dict)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_dec0 = time.perf_counter()
                for _ in range(max_tokens_per_subagent - 1):
                    if first_token in self.engine.STOP_TOKEN_IDS or max_tokens_per_subagent <= 1:
                        break
                    logits, state_dict = self.engine.forward_token(curr_token, state_dict, pos=pos, use_graph=True)
                    next_token = int(torch.argmax(logits[0, :]).item())
                    gen_tokens.append(next_token)
                    if next_token in self.engine.STOP_TOKEN_IDS:
                        break
                    curr_token = next_token
                    pos += 1

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                decode_ms = (time.perf_counter() - t_dec0) * 1000.0
                total_ms = (time.perf_counter() - t_turn_start) * 1000.0

                output_text = self.tokenizer.decode(gen_tokens)
                cumulative_history += output_text
                tok_s = (len(gen_tokens) / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0

                # Write artifact(s)
                created_artifacts = []
                for art_name in task.expected_artifacts:
                    target_file = project_dir / art_name
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    content = self._extract_file_content(output_text, art_name)
                    target_file.write_text(content)
                    created_artifacts.append(str(target_file))

                telemetry_records.append(
                    HarnessTurnTelemetry(
                        task_id=task.task_id,
                        specialist=task.specialist.value,
                        lora_adapter=lora_name,
                        prefill_ms=prefill_ms,
                        decode_ms=decode_ms,
                        total_ms=total_ms,
                        handoff_ms=0.0,
                        lora_swap_ms=lora_swap_ms,
                        prompt_tokens=len(prompt_tokens),
                        tokens_avoided=0,
                        tokens_generated=len(gen_tokens),
                        tok_per_sec=tok_s,
                        artifacts_created=created_artifacts,
                    )
                )

        # OS Verification: check linter and tests
        verification = self._verify_project(project_dir)

        return telemetry_records, verification

    def _verify_project(self, project_dir: Path) -> dict[str, Any]:
        """Runs ruff and pytest on the generated project to verify real code correctness."""
        results: dict[str, Any] = {"ruff_exit": -1, "pytest_exit": -1, "passed": False}
        try:
            # Auto-format and fix imports
            subprocess.run(
                ["uvx", "ruff", "check", "--fix", "."],
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                timeout=15,
            )
            # 1. Ruff syntax check
            res_ruff = subprocess.run(
                ["uvx", "ruff", "check", "."],
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                timeout=15,
            )
            results["ruff_exit"] = res_ruff.returncode
            results["ruff_stdout"] = res_ruff.stdout[:300]
        except Exception as e:
            results["ruff_error"] = str(e)

        try:
            # 2. Pytest execution
            res_test = subprocess.run(
                ["uv", "run", "pytest", "tests/", "-v"],
                cwd=str(project_dir),
                capture_output=True,
                text=True,
                timeout=20,
            )
            results["pytest_exit"] = res_test.returncode
            results["pytest_stdout"] = res_test.stdout[:300]
        except Exception as e:
            results["pytest_error"] = str(e)

        results["passed"] = (results["ruff_exit"] == 0) and (results["pytest_exit"] == 0)
        return results
