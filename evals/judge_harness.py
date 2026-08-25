"""E2E Judge Harness for Qualitative & Quantitative LLM Output Evaluation.

Executes prompts sequentially (multi-turn or single-turn), extracts runtime telemetry,
runs automated sanity checks (syntax, repetition, code blocks), and formats
complete, un-truncated outputs for Agent/Human qualitative evaluation.
"""

from __future__ import annotations

import ast
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any
import urllib.request
import urllib.error


@dataclass
class TurnResult:
    turn_idx: int
    prompt_id: str
    domain: str
    user_prompt: str
    assistant_response: str
    active_team: list[str]
    tokens_per_second: float
    total_tokens: int
    elapsed_seconds: float
    ttft_ms: float = 0.0
    prefolded_expert: str | None = None
    prefolded_confidence: float = 0.0
    # Automated Quality Indicators
    python_syntax_valid: bool | None = None
    python_syntax_error: str | None = None
    repetition_ratio_4gram: float = 0.0
    has_code_blocks: bool = False
    code_blocks_count: int = 0
    raw_telemetry: dict[str, Any] = field(default_factory=dict)


class E2EJudgeHarness:
    """Harness for running multi-turn and single-turn judge evaluations."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def check_server_health(self) -> bool:
        """Checks if the local inference server is up and responsive."""
        try:
            req = urllib.request.Request(f"{self.base_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def get_server_status(self) -> dict[str, Any]:
        """Fetches current server engine status."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/engine/status", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"error": str(e)}

    def execute_turn_live(
        self,
        conversation_history: list[dict[str, str]],
        turn_idx: int,
        prompt_id: str,
        domain: str,
        expert: str = "dynamic",
        max_tokens: int = 400,
        thinking_effort: str = "off",
    ) -> TurnResult:
        """Executes a single turn against the live server using SSE streaming."""
        payload = {
            "model": "qwen3.5-9b",
            "messages": conversation_history,
            "expert": expert,
            "max_tokens": max_tokens,
            "stream": True,
            "thinking_effort": thinking_effort,
        }

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        t_start = time.perf_counter()
        t_first_token: float | None = None
        collected_text_chunks: list[str] = []
        active_team: list[str] = []
        usage_data: dict[str, Any] = {}
        headers_info: dict[str, str] = {}

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                headers_info = dict(response.headers)
                gpu_state = headers_info.get("X-Router-GPU-State", "")
                if gpu_state:
                    active_team = [gpu_state.replace("expert_", "")]

                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(data_str)
                        if "choices" in chunk_json and chunk_json["choices"]:
                            delta = chunk_json["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                if t_first_token is None:
                                    t_first_token = time.perf_counter()
                                collected_text_chunks.append(content)
                        if "usage" in chunk_json and chunk_json["usage"]:
                            usage_data = chunk_json["usage"]
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            elapsed = time.perf_counter() - t_start
            return TurnResult(
                turn_idx=turn_idx,
                prompt_id=prompt_id,
                domain=domain,
                user_prompt=conversation_history[-1]["content"],
                assistant_response=f"[ERROR: Server Request Failed: {e}]",
                active_team=active_team,
                tokens_per_second=0.0,
                total_tokens=0,
                elapsed_seconds=elapsed,
            )

        elapsed = time.perf_counter() - t_start
        full_text = "".join(collected_text_chunks)
        ttft_ms = ((t_first_token - t_start) * 1000.0) if t_first_token else 0.0

        # Extract usage telemetry
        tok_s = usage_data.get("tokens_per_second", 0.0)
        tok_count = usage_data.get("completion_tokens", len(full_text.split()))
        if tok_s == 0.0 and elapsed > 0 and tok_count > 0:
            tok_s = tok_count / elapsed

        prefold = usage_data.get("predicted_next_expert")
        conf = usage_data.get("predicted_confidence", 0.0)

        # Automated Quality Checks
        syntax_ok, syntax_err, code_blocks = self._check_python_syntax(full_text)
        rep_ratio = self._calculate_repetition(full_text)

        return TurnResult(
            turn_idx=turn_idx,
            prompt_id=prompt_id,
            domain=domain,
            user_prompt=conversation_history[-1]["content"],
            assistant_response=full_text,
            active_team=active_team,
            tokens_per_second=round(tok_s, 2),
            total_tokens=tok_count,
            elapsed_seconds=round(elapsed, 3),
            ttft_ms=round(ttft_ms, 1),
            prefolded_expert=prefold,
            prefolded_confidence=round(conf * 100.0, 1),
            python_syntax_valid=syntax_ok,
            python_syntax_error=syntax_err,
            repetition_ratio_4gram=round(rep_ratio, 3),
            has_code_blocks=len(code_blocks) > 0,
            code_blocks_count=len(code_blocks),
            raw_telemetry=usage_data,
        )

    def _check_python_syntax(self, text: str) -> tuple[bool | None, str | None, list[str]]:
        """Extracts python code blocks and verifies AST syntax validity."""
        code_blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL)
        if not code_blocks:
            return None, None, []

        for idx, block in enumerate(code_blocks, start=1):
            try:
                ast.parse(block)
            except SyntaxError as e:
                return False, f"Block #{idx} SyntaxError: {e.msg} at line {e.lineno}", code_blocks
        return True, None, code_blocks

    def _calculate_repetition(self, text: str) -> float:
        """Computes 4-gram repetition ratio (0.0 = completely unique, 1.0 = degenerate loop)."""
        words = re.findall(r"\w+", text.lower())
        if len(words) < 8:
            return 0.0
        four_grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
        if not four_grams:
            return 0.0
        unique_grams = set(four_grams)
        return 1.0 - (len(unique_grams) / len(four_grams))

    @staticmethod
    def format_turn_markdown(result: TurnResult) -> str:
        """Formats a single turn into rich Markdown suitable for human/agent review."""
        team_str = " + ".join(result.active_team) if result.active_team else "base/pristine"
        syntax_badge = "N/A"
        if result.python_syntax_valid is True:
            syntax_badge = "✅ VALID AST"
        elif result.python_syntax_valid is False:
            syntax_badge = f"❌ SYNTAX ERROR ({result.python_syntax_error})"

        rep_badge = "✅ CLEAN" if result.repetition_ratio_4gram < 0.20 else f"⚠️ REPETITIVE ({result.repetition_ratio_4gram:.1%})"

        prefold_str = f"{result.prefolded_expert} ({result.prefolded_confidence}%)" if result.prefolded_expert else "None"

        lines = [
            f"### 📍 Turn {result.turn_idx}: [{result.domain.upper()}] `{result.prompt_id}`",
            f"> **Prompt**: {result.user_prompt}",
            "",
            "**Runtime Telemetry & Quality Indicators:**",
            f"- **Active Expert Team**: `{team_str}`",
            f"- **Throughput**: `{result.tokens_per_second} tok/s` | **TTFT**: `{result.ttft_ms} ms` | **Total Tokens**: `{result.total_tokens}` ({result.elapsed_seconds}s)",
            f"- **Next-Turn Pre-fold Prediction**: `{prefold_str}`",
            f"- **Code Syntax**: {syntax_badge}",
            f"- **Repetition Index (4-gram)**: {rep_badge}",
            "",
            "**Full Assistant Completion:**",
            "```markdown",
            result.assistant_response,
            "```",
            "",
            "---",
        ]
        return "\n".join(lines)

    @staticmethod
    def format_summary_table(results: list[TurnResult]) -> str:
        """Generates a summary markdown table across all executed turns."""
        headers = [
            "Turn",
            "Domain",
            "Active Team",
            "tok/s",
            "Tokens",
            "TTFT (ms)",
            "Syntax",
            "Repetition",
            "Predicted Next",
        ]
        rows = []
        for r in results:
            team_str = "+".join(r.active_team) if r.active_team else "base"
            syntax_str = "OK" if r.python_syntax_valid is True else ("ERR" if r.python_syntax_valid is False else "-")
            rep_str = f"{r.repetition_ratio_4gram:.1%}"
            next_str = f"{r.prefolded_expert} ({r.prefolded_confidence}%)" if r.prefolded_expert else "-"

            rows.append(
                f"| {r.turn_idx} | {r.domain} | `{team_str}` | {r.tokens_per_second:.1f} | {r.total_tokens} | {r.ttft_ms:.1f} | {syntax_str} | {rep_str} | {next_str} |"
            )

        header_row = "| " + " | ".join(headers) + " |"
        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        return "\n".join([header_row, separator_row] + rows)
