"""Direct HTTP client evaluating OpenAI-compatible runtimes (/v1/chat/completions).

Agentic multi-turn loop with tensor handoff (TensorStateSnapshot / resume).

Key design decisions:
  - Turn 1: full cold prefill (system + user prompt) with python_modern adapter.
  - After Turn 1: snapshot the KV/GDN state via POST /v1/state/snapshot.
  - Turns 2+: resume from the snapshot with ONLY the new feedback turn.
      This means the engine restores KV cache + GDN state from VRAM in ~1 ms
      and prefills only the new ~50-token feedback message, instead of
      re-prefilling the full growing conversation history.
  - LoRA switching: even-numbered repair turns switch to agentic_coding adapter
      for error analysis and routing, odd repair turns use python_modern for
      code generation -- both share the same tensor state via resume.
  - Concise feedback: only the specific failing assertion/error lines are sent,
      not the full pytest console output.
"""

from __future__ import annotations

import json
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from benchmarks.aider_bench.executor import PytestExecutor, PytestResult
from benchmarks.aider_bench.parser import apply_edit
from benchmarks.aider_bench.tasks import AiderTask


@dataclass
class TurnTelemetry:
    turn: int
    ttft_ms: float
    decode_tok_s: float
    generated_tokens: int
    wall_clock_s: float
    edit_applied: bool
    edit_error: str | None
    pytest_result: PytestResult | None
    adapter: str | None = None
    # Tensor handoff metadata
    used_resume: bool = False
    snapshot_name: str | None = None
    prefill_token_count: int = 0


@dataclass
class DirectTaskResult:
    task_name: str
    passed: bool
    pass_turn: int  # 1-indexed turn where tests passed; 0 if failed
    total_turns: int
    total_tokens: int
    average_ttft_ms: float
    average_tok_s: float
    total_duration_s: float
    turns: list[TurnTelemetry] = field(default_factory=list)
    final_code: str = ""
    error_message: str | None = None


def _extract_concise_error(pytest_result: PytestResult) -> str:
    """Extracts a concise error summary from pytest output.

    Strips pytest boilerplate (collection banners, PASSED lines, summary bars)
    and returns only the lines that matter for a repair prompt:
      - AssertionError lines
      - FAILED lines  
      - E  (pytest short-form error lines)
      - ImportError / SyntaxError lines
      - The final short_summary section

    Caps at 20 lines to avoid flooding the context window.
    """
    output = (pytest_result.stdout + "\n" + pytest_result.stderr).strip()
    lines = output.splitlines()

    priority_lines: list[str] = []
    in_short_test_summary = False

    for line in lines:
        stripped = line.strip()
        # Enter short test summary section
        if "short test summary info" in stripped.lower():
            in_short_test_summary = True
        if in_short_test_summary:
            priority_lines.append(line)
            continue
        # pytest error detail lines (E   AssertionError: ...)
        if stripped.startswith("E ") or stripped.startswith("E\t"):
            priority_lines.append(line)
            continue
        # FAILED/ERROR summary markers
        if stripped.startswith("FAILED") or stripped.startswith("ERROR"):
            priority_lines.append(line)
            continue
        # AssertionError / SyntaxError / ImportError in tracebacks
        if any(exc in stripped for exc in ("AssertionError", "SyntaxError", "ImportError", "NameError", "TypeError", "AttributeError")):
            priority_lines.append(line)
            continue

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for line in priority_lines:
        k = line.strip()
        if k and k not in seen:
            seen.add(k)
            deduped.append(line)

    # If nothing specific found, fall back to last 10 output lines
    if not deduped:
        deduped = lines[-10:]

    # Cap at 20 lines
    deduped = deduped[:20]
    return "\n".join(deduped)


class SnapshotClient:
    """Lightweight client for the runtime-next snapshot API.

    POST /v1/state/snapshot  -> creates a named TensorStateSnapshot (KV+GDN clone)
    DELETE /v1/state/snapshot -> deletes a named snapshot
    """

    def __init__(self, base_url: str, timeout: int = 30):
        self.snapshot_url = f"{base_url.rstrip('/')}/state/snapshot"
        self.timeout = timeout

    def create(self, name: str) -> tuple[bool, str]:
        """Creates a snapshot with the given name. Returns (ok, error_or_empty)."""
        payload = json.dumps({"name": name}).encode("utf-8")
        req = urllib.request.Request(
            self.snapshot_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-local-bench",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read()
            return True, ""
        except Exception as e:
            return False, str(e)

    def delete(self, name: str) -> None:
        """Best-effort delete of a named snapshot."""
        payload = json.dumps({"name": name}).encode("utf-8")
        req = urllib.request.Request(
            self.snapshot_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-local-bench",
            },
            method="DELETE",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp.read()
        except Exception:
            pass  # Best-effort; snapshot TTL will clean it up anyway


class DirectHarness:
    """Evaluates tasks directly against an OpenAI-compatible HTTP runtime.

    Agentic loop with tensor handoff:
      - Turn 1: cold prefill; state snapshot taken after completion.
      - Turns 2+: resume from snapshot with only the incremental feedback message.
        The runtime restores KV cache + GDN state in VRAM (~1 ms device-to-device)
        and prefills only the new feedback string (~50 tokens) instead of the
        full growing conversation history.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8003/v1",
        model: str = "qwen3.5:4b-rust",
        adapter: str | None = None,
        edit_format: str = "whole",
        max_turns: int = 5,
        temperature: float = 0.0,
        executor: PytestExecutor | None = None,
        timeout: int = 90,
        # Enough to generate any Aider task implementation without truncation.
        # Server default (512) was cutting code mid-function body.
        max_tokens: int = 4096,
        # Slight temperature for repair turns breaks the deterministic loop
        # that occurs when tensor handoff restores the same KV state each turn
        # at temperature=0 — the model would always produce the identical wrong answer.
        repair_temperature: float = 0.15,
        # Tensor handoff: enabled by default, can disable for ablation
        use_tensor_handoff: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.completions_url = f"{self.base_url}/chat/completions"
        self.model = model
        self.adapter = adapter
        self.edit_format = edit_format.lower()
        self.max_turns = max_turns
        self.temperature = temperature
        self.executor = executor or PytestExecutor()
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.repair_temperature = repair_temperature
        self.use_tensor_handoff = use_tensor_handoff
        self.snapshot_client = SnapshotClient(self.base_url, timeout=30)

    def build_initial_prompt(self, task: AiderTask) -> tuple[str, str]:
        """Constructs system prompt and initial user prompt for the task."""
        if self.edit_format == "diff":
            system_prompt = (
                "You are an expert software engineer. Follow the user's instructions carefully.\n"
                "Provide code edits exclusively using SEARCH/REPLACE blocks in this exact format:\n\n"
                "```python\n"
                "<<<<<<< SEARCH\n"
                "# exact lines to match from the existing file\n"
                "=======\n"
                "# replacement lines\n"
                ">>>>>>> REPLACE\n"
                "```\n"
            )
            user_prompt = (
                f"Instructions:\n{task.instructions}\n\n"
                f"Existing file `{task.target_file_name}`:\n"
                f"```python\n{task.stub_code}\n```\n\n"
                f"Please provide the SEARCH/REPLACE block(s) to implement the required functionality in `{task.target_file_name}`."
            )
        else:
            system_prompt = (
                "You are an expert software engineer. Follow the user's instructions carefully.\n"
                f"Implement the requested solution for `{task.target_file_name}`.\n"
                "Respond ONLY with the complete updated Python code inside a ```python ... ``` code block."
            )
            user_prompt = (
                f"Instructions:\n{task.instructions}\n\n"
                f"Starting stub in `{task.target_file_name}`:\n"
                f"```python\n{task.stub_code}\n```\n\n"
                f"Please implement the full code for `{task.target_file_name}`."
            )

        return system_prompt, user_prompt

    def _resolve_turn_adapter(self, turn_idx: int) -> str | None:
        """Resolves active adapter for the given turn.

        Supports:
          - "alternate:a,b"  -- cycles through adapters each turn (turn 1=a, turn 2=b, ...)
          - "alternating:a,b" -- alias for alternate
          - "sequential:a,b"  -- turn 1 uses a, all subsequent turns use b
          - "agentic:gen,analysis" -- turn 1 uses gen, even repair turns use analysis adapter
          - plain name -- same adapter every turn
        """
        if not self.adapter:
            return None
        if self.adapter.startswith(("alternate:", "alternating:")):
            parts = self.adapter.split(":", 1)[1].split(",")
            if len(parts) >= 2:
                idx = (turn_idx - 1) % len(parts)
                return parts[idx].strip()
        elif self.adapter.startswith("sequential:"):
            parts = self.adapter.split(":", 1)[1].split(",")
            if len(parts) >= 2:
                return parts[0].strip() if turn_idx == 1 else parts[1].strip()
        elif self.adapter.startswith("agentic:"):
            # "agentic:gen_adapter,analysis_adapter"
            # Turn 1 (generation): gen_adapter
            # Turn 2+ even (analysis): analysis_adapter
            # Turn 2+ odd (repair generation): gen_adapter
            parts = self.adapter.split(":", 1)[1].split(",")
            if len(parts) >= 2:
                gen_adapter = parts[0].strip()
                analysis_adapter = parts[1].strip()
                if turn_idx == 1:
                    return gen_adapter
                # Repair turns: even=analysis, odd=generation
                return analysis_adapter if turn_idx % 2 == 0 else gen_adapter
        return self.adapter

    def _call_streaming_api(
        self,
        messages: list[dict[str, str]],
        adapter: str | None = None,
        resume: str | None = None,
        temperature: float | None = None,
    ) -> tuple[str, float, float, int, int]:
        """Sends chat completion request and streams tokens.

        Returns (response_text, ttft_ms, tok_s, token_count, prompt_tokens).
        When `resume` is set, the engine restores KV/GDN state from that snapshot
        and prefills only the messages in this request (not the full history).
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": self.max_tokens,
            # Stop after closing code fence to prevent runaway generation.
            # The model should emit ``` after the implementation, then stop.
            "stop": ["```\n", "```\r\n"],
            "stream": True,
        }
        active_adapter = adapter if adapter is not None else self.adapter
        # Resolve plain adapter string (not policy prefix) for the payload
        if active_adapter and any(active_adapter.startswith(p) for p in ("alternate:", "alternating:", "sequential:", "agentic:")):
            active_adapter = None  # already resolved to turn-specific adapter above
        if active_adapter:
            payload["adapter"] = active_adapter
        if resume:
            payload["resume"] = resume

        req = urllib.request.Request(
            self.completions_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-local-bench",
            },
        )

        t0 = time.perf_counter()
        tokens = 0
        prompt_tokens = 0
        ttft_ms: float | None = None
        full_content: list[str] = []

        # Early-exit fence tracker: once we see a complete ```...``` code block,
        # we drop the connection. The server detects the write failure and stops
        # within one extra token (see server.rs §real_disconnected_client_stops_generation_within_one_token).
        # This prevents 19-second runaway turns when the model never emits EOS.
        _open_fence_seen = False
        _tail_buf = ""  # trailing text buffer to detect the closing fence across chunks

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for line in resp:
                    s = line.decode("utf-8").strip()
                    if not s.startswith("data: ") or s == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(s[6:])
                    except Exception:
                        continue

                    # Capture prompt token count from the final usage chunk
                    if "usage" in chunk and chunk["usage"]:
                        prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})
                    content = delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")
                    if content:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000.0
                        tokens += 1
                        full_content.append(content)

                        # Fence tracking: maintain a 16-char rolling tail buffer.
                        # Detect: opening fence (```python or ```) then closing fence (```).
                        _tail_buf = (_tail_buf + content)[-64:]
                        if not _open_fence_seen:
                            if "```python" in _tail_buf or "```py\n" in _tail_buf or "```\n" in _tail_buf:
                                _open_fence_seen = True
                                # Consume the opening fence from the tail so it doesn't
                                # immediately re-trigger the closing check below.
                                for marker in ("```python", "```py\n", "```\n"):
                                    idx = _tail_buf.find(marker)
                                    if idx != -1:
                                        _tail_buf = _tail_buf[idx + len(marker):]
                                        break
                        else:
                            # Look for a standalone closing ``` (not part of an opening fence)
                            # Must appear after at least some code content (newline before it).
                            if "\n```" in _tail_buf:
                                # Complete code block received — drop the connection now.
                                # Server stops within 1 extra token on write failure.
                                break
                        # Safety net: if we've read >1200 tokens with no complete code
                        # fence yet, drop the connection. This caps analysis-only adapter
                        # turns (agentic_coding generating text without code fences) at
                        # ~5s instead of 25s. The content collected so far is still usable
                        # as diagnostic context for the next turn.
                        if tokens > 1200 and not _open_fence_seen:
                            break
        except Exception:
            # Connection dropped early (our break above) or network error — both fine.
            pass

        t1 = time.perf_counter()
        ttft_val = ttft_ms if ttft_ms is not None else 0.0
        gen_time = (t1 - t0) - (ttft_val / 1000.0)
        tok_s = tokens / max(1e-5, gen_time)
        return "".join(full_content), ttft_val, tok_s, tokens, prompt_tokens

    def run_task(
        self,
        task: AiderTask,
        on_turn_complete: Callable[[int, TurnTelemetry], None] | None = None,
    ) -> DirectTaskResult:
        """Executes the agentic multi-turn benchmark loop for a single task.

        Agentic tensor-handoff loop:
          Turn 1: cold prefill [system, user]. Snapshot state after completion.
          Turn N>1: POST /v1/chat/completions with:
            - resume="<snapshot_name>"  -- engine restores KV+GDN state in ~1ms
            - messages=[{"role": "user", "content": "<concise feedback only>"}]
            This drops per-turn prefill from O(total_tokens) to O(feedback_tokens).
        """
        system_prompt, user_prompt = self.build_initial_prompt(task)

        # Full initial message list for Turn 1 (cold prefill)
        turn1_messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        current_code = task.stub_code
        turns_log: list[TurnTelemetry] = []
        passed = False
        pass_turn = 0
        t_task_start = time.perf_counter()

        # Snapshot name for this task instance (unique per task run)
        snap_name = f"bench_{task.name}_{uuid.uuid4().hex[:8]}"
        snap_taken = False

        for turn_idx in range(1, self.max_turns + 1):
            t_turn_start = time.perf_counter()
            turn_adapter = self._resolve_turn_adapter(turn_idx)
            used_resume = False
            active_snapshot: str | None = None

            # Determine messages and resume target for this turn
            if turn_idx == 1 or not snap_taken:
                # Cold prefill: send full message history
                messages_to_send = turn1_messages
                resume_name = None
            else:
                # Tensor handoff: send only the new incremental feedback message.
                # The engine will restore KV+GDN state from the snapshot and
                # prefill only this one feedback message.
                #
                # If the previous turn was an analysis-only response (no code fence),
                # prepend it as a diagnosis hint so the generation adapter has context.
                analysis_prefix = ""
                if "_analysis_context" in dir() and _analysis_context:  # type: ignore[used-before-assignment]
                    analysis_prefix = f"Diagnosis from previous analysis:\n{_analysis_context}\n\n"

                if not success:  # type: ignore[used-before-assignment]
                    if edit_error:  # type: ignore[used-before-assignment]
                        feedback_content = (
                            f"{analysis_prefix}The code block couldn't be applied: {edit_error}.\n"  # type: ignore[used-before-assignment]
                            "Please output the complete corrected Python file inside a ```python ... ``` block."
                        )
                    else:
                        # No code was emitted (analysis-only turn) — ask for actual code
                        feedback_content = (
                            f"{analysis_prefix}"
                            "Now output the complete corrected Python file implementing the fix "
                            "inside a ```python ... ``` code block."
                        )
                elif pytest_res is not None:  # type: ignore[used-before-assignment]
                    concise = _extract_concise_error(pytest_res)
                    feedback_content = (
                        f"{analysis_prefix}Tests failed. Fix the following errors:\n\n```\n{concise}\n```\n\n"
                        "Output the corrected complete Python file inside a ```python ... ``` block."
                    )
                else:
                    feedback_content = (
                        f"{analysis_prefix}Verification failed. Please output the corrected complete Python file "
                        "inside a ```python ... ``` block."
                    )
                messages_to_send = [{"role": "user", "content": feedback_content}]
                resume_name = snap_name
                used_resume = True
                active_snapshot = snap_name

            try:
                response_text, ttft_ms, tok_s, tokens, prompt_tokens = self._call_streaming_api(
                    messages_to_send,
                    adapter=turn_adapter,
                    resume=resume_name,
                    temperature=self.repair_temperature if used_resume else None,
                )
            except Exception as e:
                err_msg = f"HTTP Error on turn {turn_idx}: {e}"
                turns_log.append(
                    TurnTelemetry(
                        turn=turn_idx,
                        ttft_ms=0.0,
                        decode_tok_s=0.0,
                        generated_tokens=0,
                        wall_clock_s=time.perf_counter() - t_turn_start,
                        edit_applied=False,
                        edit_error=err_msg,
                        pytest_result=None,
                        adapter=turn_adapter,
                        used_resume=used_resume,
                        snapshot_name=active_snapshot,
                        prefill_token_count=0,
                    )
                )
                # Clean up snapshot on failure
                if snap_taken and self.use_tensor_handoff:
                    self.snapshot_client.delete(snap_name)
                return DirectTaskResult(
                    task_name=task.name,
                    passed=False,
                    pass_turn=0,
                    total_turns=turn_idx,
                    total_tokens=sum(t.generated_tokens for t in turns_log),
                    average_ttft_ms=0.0,
                    average_tok_s=0.0,
                    total_duration_s=time.perf_counter() - t_task_start,
                    turns=turns_log,
                    final_code=current_code,
                    error_message=err_msg,
                )

            # Detect whether the response contains a code block at all.
            # Agentic-coding adapter turns often output pure analysis text (no code fence).
            # In that case: treat the response as diagnostic context for the NEXT turn
            # instead of running pytest on the analysis text (which would always fail with
            # a SyntaxError and produce confusing feedback).
            _has_code_fence = "```" in response_text

            if not _has_code_fence:
                # Analysis-only turn: capture the diagnosis, don't touch current_code,
                # don't run pytest. The analysis will be prepended to the next turn's
                # feedback so python_modern can act on the diagnosis.
                success = False
                edit_error = None   # Not an edit failure, just no code emitted
                pytest_res = None
                _analysis_context = response_text.strip()
            else:
                _analysis_context = ""
                # Apply edit to current code
                success, updated_or_err = apply_edit(current_code, response_text, self.edit_format)
                if not success:
                    edit_error = updated_or_err
                    pytest_res = None
                else:
                    edit_error = None
                    current_code = updated_or_err
                    # Run isolated pytest
                    pytest_res = self.executor.execute(
                        target_file_name=task.target_file_name,
                        target_code=current_code,
                        test_file_name=task.test_file_name,
                        test_code=task.test_code,
                    )
                    if pytest_res.passed:
                        passed = True
                        pass_turn = turn_idx

            turn_data = TurnTelemetry(
                turn=turn_idx,
                ttft_ms=ttft_ms,
                decode_tok_s=tok_s,
                generated_tokens=tokens,
                wall_clock_s=time.perf_counter() - t_turn_start,
                edit_applied=success,
                edit_error=edit_error,
                pytest_result=pytest_res,
                adapter=turn_adapter,
                used_resume=used_resume,
                snapshot_name=active_snapshot,
                prefill_token_count=prompt_tokens,
            )
            turns_log.append(turn_data)

            if on_turn_complete:
                on_turn_complete(turn_idx, turn_data)

            if passed:
                break

            # After Turn 1 (regardless of pass/fail), snapshot the engine state
            # so subsequent turns can resume from this point via incremental prefill.
            if turn_idx == 1 and self.use_tensor_handoff and not snap_taken:
                ok, err = self.snapshot_client.create(snap_name)
                if ok:
                    snap_taken = True
                # If snapshot fails, we fall back to full history re-prefill gracefully
                # (turn 2+ will see snap_taken=False and send full messages)

        # Clean up snapshot after task completes (passes or exhausts turns)
        if snap_taken and self.use_tensor_handoff:
            self.snapshot_client.delete(snap_name)

        total_duration = time.perf_counter() - t_task_start
        total_tokens = sum(t.generated_tokens for t in turns_log)
        avg_ttft = sum(t.ttft_ms for t in turns_log) / max(1, len(turns_log))
        avg_tok_s = sum(t.decode_tok_s for t in turns_log) / max(1, len(turns_log))

        return DirectTaskResult(
            task_name=task.name,
            passed=passed,
            pass_turn=pass_turn,
            total_turns=len(turns_log),
            total_tokens=total_tokens,
            average_ttft_ms=avg_ttft,
            average_tok_s=avg_tok_s,
            total_duration_s=total_duration,
            turns=turns_log,
            final_code=current_code,
        )
