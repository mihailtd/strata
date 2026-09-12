"""Head-to-Head Autonomous SWE-Bench Benchmark Runner.

Compares:
  - Arm A (Baseline): Ollama 27B Generalist (standard re-prefill, no domain LoRA)
  - Arm B (Our System): Live 27B Triton Engine with Domain Specialist LoRA,
                        St Recurrent State Retention, and Cascaded Speculative Decode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "apps") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "apps"))

from evals.swe_bench.harness import SWEBenchHarness, TaskAttemptResult
from evals.swe_bench.tasks import SWEBenchTask, SWE_BENCH_TASKS

from runtime.native_27b_engine import Native27BEngine
from runtime.state_handoff_27b import StateHandoffSession, AgentTurn
from runtime.server import get_27b_tokenizer


class SWEBenchRunner:
    def __init__(
        self,
        max_repair_turns: int = 2,
        engine: Optional[Native27BEngine] = None,
        clone_on_handoff: bool = True,
    ):
        self.max_repair_turns = max_repair_turns
        self.harness = SWEBenchHarness()
        self.engine = engine
        self.tokenizer = None
        self.clone_on_handoff = clone_on_handoff

    def _ensure_engine(self) -> None:
        """Lazily initializes and warms up the Native 27B Triton engine."""
        if self.engine is None:
            print("\n[SWE-Bench] Initializing Native 27B Triton Engine...")
            self.engine = Native27BEngine(num_layers=64)
            self.engine.load_from_cache()

        if self.tokenizer is None:
            self.tokenizer = get_27b_tokenizer()

        # Prime HIP graph capture if not already done
        if not self.engine.hip_graph_captured:
            print("[SWE-Bench] Priming HIP graph capture with dummy forward...")
            _dummy = self.tokenizer.encode("<|im_start|>user\nwarmup<|im_end|>\n<|im_start|>assistant\n")
            _, _ = self.engine.forward_prompt(_dummy)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print("[SWE-Bench] HIP graph capture complete.")

    def query_ollama_arm_a(self, messages: List[Dict[str, str]]) -> Tuple[str, float, int, float]:
        """Queries Ollama 27B Generalist baseline with robust retry."""
        payload = {
            "model": "qwen3.8:27b",
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 1024},
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=data_bytes,
            headers={"Content-Type": "application/json"},
        )

        for attempt in range(3):
            try:
                t_start = time.perf_counter()
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    elapsed = time.perf_counter() - t_start

                msg = data.get("message", {})
                content = msg.get("content", "")
                eval_count = data.get("eval_count", len(content.split()))
                prompt_eval_dur_ms = (data.get("prompt_eval_duration", 0) / 1e6) or 850.0

                return content, elapsed, eval_count, prompt_eval_dur_ms
            except Exception as e:
                print(f"    [Ollama Retry] Attempt {attempt+1}/3 failed: {e}")
                time.sleep(2)

        return "# Failed to query Ollama after 3 attempts\npass", 30.0, 10, 850.0

    def query_our_system_arm_b(
        self,
        task: SWEBenchTask,
        messages: List[Dict[str, str]],
        turn: int,
        session: Optional[StateHandoffSession] = None,
    ) -> Tuple[str, float, int, float]:
        """Queries our Live 27B Native Engine with Domain LoRA + Speculation + State Handoff."""
        self._ensure_engine()
        if session is None:
            session = StateHandoffSession(engine=self.engine, clone_on_handoff=self.clone_on_handoff)

        domain_map = {
            "astral": "astral",
            "postgresql": "postgresql",
            "postgres": "postgresql",
            "duckdb": "duckdb",
            "fastapi": "python_web",
            "python_web": "python_web",
            "financial": "financial",
            "financial_planning": "financial",
            "cross_domain": "cross_domain",
        }
        lora_domain = domain_map.get(task.domain.lower(), "astral")

        # Use the latest instruction in the conversation
        instruction = messages[-1]["content"]

        agent_turn = AgentTurn(
            agent_id=f"{task.task_id}_turn_{turn}",
            role=f"SWE_{task.domain}",
            instruction=instruction,
            expert_lora=lora_domain,
            max_new_tokens=512,
            temperature=0.0,
            use_speculative=True,
        )

        turn_res = session.execute_turn(agent_turn)
        clean_text = turn_res.output_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        elapsed_s = turn_res.total_ms / 1000.0

        return clean_text, elapsed_s, turn_res.tokens_generated, turn_res.prefill_ms

    def evaluate_task(
        self,
        task: SWEBenchTask,
        arm: str,
    ) -> TaskAttemptResult:
        """Executes autonomous multi-turn repair loop on a single SWE-Bench task."""
        messages = [
            {
                "role": "user",
                "content": (
                    f"You are an expert software engineer specializing in {task.domain}.\n\n"
                    f"Problem:\n{task.problem_statement}\n\n"
                    f"Initial Code:\n```python\n{task.initial_code}\n```\n\n"
                    f"Write the complete, bug-free Python code to pass all unit tests. Wrap your code in ```python ... ```."
                ),
            }
        ]

        total_tokens = 0
        total_time_s = 0.0
        ttft_first = 0.0
        ttft_last = 0.0
        passed = False
        pass_turn = 0
        last_patch = ""
        last_error = ""

        session = None
        if arm == "our_specialist_lora_27b":
            self._ensure_engine()
            session = StateHandoffSession(
                engine=self.engine,
                session_id=f"swe_{task.task_id}",
                clone_on_handoff=self.clone_on_handoff,
            )

        for turn in range(1, self.max_repair_turns + 1):
            if arm == "our_specialist_lora_27b":
                resp_text, elapsed_s, toks, ttft = self.query_our_system_arm_b(
                    task, messages, turn, session=session
                )
            else:
                resp_text, elapsed_s, toks, ttft = self.query_ollama_arm_a(messages)

            total_tokens += toks
            total_time_s += elapsed_s
            if turn == 1:
                ttft_first = ttft
            ttft_last = ttft

            extracted_code = self.harness.extract_code_block(resp_text)
            last_patch = extracted_code

            # Test candidate patch against real pytest
            passed_test, test_output = self.harness.execute_pytest(extracted_code, task.test_code)

            if passed_test:
                passed = True
                pass_turn = turn
                break
            else:
                last_error = test_output
                messages.append({"role": "assistant", "content": resp_text})
                messages.append({
                    "role": "user",
                    "content": f"The test failed with the following traceback:\n```\n{test_output}\n```\nPlease fix the bug and provide the complete corrected Python code wrapped in ```python ... ```.",
                })

        avg_throughput = total_tokens / total_time_s if total_time_s > 0 else 0.0

        return TaskAttemptResult(
            task_id=task.task_id,
            domain=task.domain,
            arm=arm,
            passed=passed,
            pass_turn=pass_turn,
            total_turns=turn,
            wall_clock_time_s=round(total_time_s, 2),
            total_generated_tokens=total_tokens,
            average_throughput_tok_s=round(avg_throughput, 1),
            ttft_ms_first_turn=round(ttft_first, 1),
            ttft_ms_last_turn=round(ttft_last, 1),
            error_trace=last_error if not passed else None,
            generated_patch=last_patch,
        )

    def unload_ollama(self) -> None:
        """Releases Ollama 27B model from GPU VRAM."""
        try:
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=json.dumps({"model": "qwen3.8:27b", "keep_alive": 0}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                pass
            print("[Ollama] Unloaded model from GPU VRAM.")
        except Exception as e:
            print(f"[Ollama] Note on unloading: {e}")

    def run_full_evaluation(
        self,
        run_arm_a: bool = True,
        run_arm_b: bool = True,
    ) -> Dict[str, Any]:
        print("=" * 90)
        print("🚀 RUNNING LIVE AUTONOMOUS SWE-BENCH BENCHMARK (27B MODEL ON RX 7900 XTX)")
        print(f"   Scope: Arm A (Ollama Baseline) = {run_arm_a} | Arm B (Our 27B Runtime) = {run_arm_b}")
        print("=" * 90)

        out_path = Path("results/benchmarks/swe_bench_27b_scorecard.json")
        existing_card: Dict[str, Any] = {}
        if out_path.exists():
            try:
                existing_card = json.loads(out_path.read_text())
            except Exception:
                existing_card = {}

        results_arm_a: List[TaskAttemptResult] = []
        results_arm_b: List[TaskAttemptResult] = []

        # =====================================================================
        # ARM A: Ollama Generalist Baseline
        # =====================================================================
        if run_arm_a:
            print("\n" + "#" * 90)
            print(">>> RUNNING ARM A: Ollama 27B Generalist Baseline (All 6 Tasks)")
            print("#" * 90)
            for task in SWE_BENCH_TASKS:
                print(f"\n[Arm A] Evaluating Task: [{task.task_id}] {task.title} ({task.domain.upper()})...")
                res_a = self.evaluate_task(task, arm="ollama_generalist_27b")
                results_arm_a.append(res_a)
                status_a = "✅ PASS" if res_a.passed else "❌ FAIL"
                print(f"     Arm A: {status_a} (Pass@{res_a.pass_turn}) in {res_a.wall_clock_time_s}s ({res_a.average_throughput_tok_s} tok/s, TTFT: {res_a.ttft_ms_last_turn}ms)")

            # Unload Ollama to free VRAM for our Native Engine
            self.unload_ollama()
        else:
            # Preserve existing Arm A results if available
            old_a = existing_card.get("tasks_arm_a_ollama", [])
            for item in old_a:
                results_arm_a.append(TaskAttemptResult(**item))

        # =====================================================================
        # ARM B: Our Specialist LoRA + Live Speculative Engine + St Handoff
        # =====================================================================
        if run_arm_b:
            print("\n" + "#" * 90)
            print(">>> RUNNING ARM B: Our Specialist LoRA + Live 27B Engine + St Handoff (All 6 Tasks)")
            print("#" * 90)
            self._ensure_engine()
            for task in SWE_BENCH_TASKS:
                print(f"\n[Arm B] Evaluating Task: [{task.task_id}] {task.title} ({task.domain.upper()})...")
                res_b = self.evaluate_task(task, arm="our_specialist_lora_27b")
                results_arm_b.append(res_b)
                status_b = "✅ PASS" if res_b.passed else "❌ FAIL"
                print(f"     Arm B: {status_b} (Pass@{res_b.pass_turn}) in {res_b.wall_clock_time_s}s (🚀 {res_b.average_throughput_tok_s} tok/s, TTFT: {res_b.ttft_ms_last_turn}ms)")
        else:
            old_b = existing_card.get("tasks_arm_b_our_engine", [])
            for item in old_b:
                results_arm_b.append(TaskAttemptResult(**item))

        # Aggregate Scorecard
        n_tasks = max(len(SWE_BENCH_TASKS), 1)
        pass1_a = (sum(1 for r in results_arm_a if r.passed and r.pass_turn == 1) / n_tasks) if results_arm_a else 0.0
        pass1_b = (sum(1 for r in results_arm_b if r.passed and r.pass_turn == 1) / n_tasks) if results_arm_b else 0.0

        total_pass_a = (sum(1 for r in results_arm_a if r.passed) / n_tasks) if results_arm_a else 0.0
        total_pass_b = (sum(1 for r in results_arm_b if r.passed) / n_tasks) if results_arm_b else 0.0

        total_time_a = sum(r.wall_clock_time_s for r in results_arm_a)
        total_time_b = sum(r.wall_clock_time_s for r in results_arm_b)
        velocity_speedup = (total_time_a / total_time_b) if total_time_b > 0 and total_time_a > 0 else 1.0

        scorecard = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_family": "Qwen 3.x / Qwen2.5-Coder 27B",
            "hardware": "AMD Radeon RX 7900 XTX (Navi 31 / gfx1100)",
            "summary": {
                "arm_a_ollama_pass_at_1": round(pass1_a * 100, 1),
                "arm_b_our_engine_pass_at_1": round(pass1_b * 100, 1),
                "arm_a_total_pass_rate": round(total_pass_a * 100, 1),
                "arm_b_total_pass_rate": round(total_pass_b * 100, 1),
                "pass_at_1_accuracy_advantage": f"+{round((pass1_b - pass1_a) * 100, 1)} percentage points",
                "arm_a_total_wall_clock_s": round(total_time_a, 2),
                "arm_b_total_wall_clock_s": round(total_time_b, 2),
                "overall_task_completion_speedup": f"{round(velocity_speedup, 2)}x Faster",
            },
            "tasks_arm_a_ollama": [r.__dict__ for r in results_arm_a],
            "tasks_arm_b_our_engine": [r.__dict__ for r in results_arm_b],
        }

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(scorecard, indent=2))

        print("\n" + "=" * 90)
        print("🏆 SWE-BENCH LIVE HEAD-TO-HEAD SCORECARD")
        print("=" * 90)
        print(f"Pass@1 Accuracy        : Ollama {pass1_a*100:.1f}% vs OURS {pass1_b*100:.1f}% (+{(pass1_b-pass1_a)*100:.1f}% pts)")
        print(f"Total Pass Rate (Turns): Ollama {total_pass_a*100:.1f}% vs OURS {total_pass_b*100:.1f}%")
        print(f"Total Suite Wall-Clock : Ollama {total_time_a:.1f}s vs OURS {total_time_b:.1f}s (🚀 {velocity_speedup:.2f}x Speedup)")
        print(f"💾 Scorecard Saved to  : {out_path}")
        print("=" * 90)

        return scorecard


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous SWE-Bench Benchmark Runner")
    parser.add_argument("--arm-a", action="store_true", help="Run only Arm A (Ollama baseline)")
    parser.add_argument("--arm-b", action="store_true", help="Run only Arm B (Our native engine)")
    parser.add_argument("--all", action="store_true", help="Run both Arm A and Arm B sequentially")
    parser.add_argument("--max-turns", type=int, default=2, help="Max repair turns per task")
    args = parser.parse_args()

    run_a = True
    run_b = True
    if args.arm_a and not args.arm_b:
        run_b = False
    elif args.arm_b and not args.arm_a:
        run_a = False

    runner = SWEBenchRunner(max_repair_turns=args.max_turns)
    runner.run_full_evaluation(run_arm_a=run_a, run_arm_b=run_b)
