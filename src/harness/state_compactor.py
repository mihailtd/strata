"""Harness Semantic State Compactor for Long-Horizon Multi-Turn Agent Tasks.

Prevents context window explosion during multi-turn coding sessions by automatically
compressing verbose tool outputs (greps, file reads, test runs, diffs) from previous turns
into high-density structural digests, while preserving the active turn in full raw fidelity.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class SemanticStateCompactor:
    """Intelligent tool-call and multi-turn state compressor."""

    def __init__(
        self,
        max_raw_tool_lines: int = 15,
        preserve_last_n_turns: int = 2,
    ) -> None:
        self.max_raw_tool_lines = max_raw_tool_lines
        self.preserve_last_n_turns = preserve_last_n_turns

    def compact_tool_output(self, tool_name: str, raw_output: str) -> str:
        """Compact a single historical tool output based on tool semantics."""
        if not raw_output or len(raw_output.strip()) == 0:
            return "[Empty Output]"

        lines = raw_output.strip().splitlines()
        if len(lines) <= self.max_raw_tool_lines:
            return raw_output

        tool_lower = tool_name.lower()

        # 1. Grep / Search compaction
        if "grep" in tool_lower or "search" in tool_lower:
            match_files = set()
            match_count = 0
            sample_matches = []
            for line in lines:
                if ":" in line:
                    match_count += 1
                    parts = line.split(":", 2)
                    match_files.add(parts[0].strip())
                    if len(sample_matches) < 3:
                        sample_matches.append(line.strip())

            files_str = ", ".join(sorted(list(match_files))[:5])
            if len(match_files) > 5:
                files_str += f" (+{len(match_files)-5} more)"

            sample_str = "\n  ".join(sample_matches)
            return (
                f"[Grep Digest: {match_count} matches in {len(match_files)} files: {files_str}]\n"
                f"  Sample matches:\n  {sample_str}\n  ... ({len(lines)-len(sample_matches)} lines folded)"
            )

        # 2. View / Read File compaction
        if "view" in tool_lower or "read" in tool_lower or "file" in tool_lower:
            first_lines = "\n".join(lines[:4])
            last_lines = "\n".join(lines[-3:])
            return (
                f"[File Read Digest: {len(lines)} lines total]\n"
                f"{first_lines}\n"
                f"  ... [{len(lines)-7} lines folded for brevity] ...\n"
                f"{last_lines}"
            )

        # 3. Test Runner / Pytest / Cargo / UV compaction
        if "test" in tool_lower or "pytest" in tool_lower or "cargo" in tool_lower or "run" in tool_lower:
            failed_lines = [line for line in lines if any(k in line.lower() for k in ["fail", "error", "traceback", "assert", "exit code"])]
            passed_lines = [line for line in lines if "passed" in line.lower() or "ok" in line.lower() or "success" in line.lower()]

            digest_parts = [f"[Execution Digest: {len(lines)} lines total]"]
            if failed_lines:
                digest_parts.append(f"  Failures/Errors ({len(failed_lines)}):\n  " + "\n  ".join(failed_lines[:5]))
            if passed_lines:
                digest_parts.append(f"  Passed summary: {passed_lines[-1]}")
            if not failed_lines and not passed_lines:
                digest_parts.append("  Output head: " + "\n  ".join(lines[:3]))
                digest_parts.append(f"  ... ({len(lines)-3} lines folded)")
            return "\n".join(digest_parts)

        # 4. Generic fallback truncation
        head = "\n".join(lines[:5])
        tail = "\n".join(lines[-3:])
        return f"{head}\n  ... [{len(lines)-8} lines folded] ...\n{tail}"

    def compact_turn_history(
        self,
        messages: List[Dict[str, Any]],
        current_turn_index: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Compact older turns while keeping the active turn and system prompt intact."""
        if not messages:
            return []

        compacted_messages = []
        total_msgs = len(messages)

        for idx, msg in enumerate(messages):
            # Never modify system prompt
            if msg.get("role") == "system":
                compacted_messages.append(msg)
                continue

            # Keep the last N messages completely untouched in full fidelity
            is_recent = (total_msgs - idx) <= (self.preserve_last_n_turns * 2)
            if is_recent:
                compacted_messages.append(msg)
                continue

            # Process older turns
            content = msg.get("content", "")
            role = msg.get("role")

            if role == "tool" or role == "assistant" or role == "user":
                # Check for tool results embedded in markdown codeblocks
                if "```" in content and len(content.splitlines()) > self.max_raw_tool_lines:
                    folded_content = self._fold_embedded_tool_blocks(content)
                    compacted_messages.append({**msg, "content": folded_content})
                elif role == "tool" and len(content.splitlines()) > self.max_raw_tool_lines:
                    folded_content = self.compact_tool_output(msg.get("name", "tool"), content)
                    compacted_messages.append({**msg, "content": folded_content})
                else:
                    compacted_messages.append(msg)
            else:
                compacted_messages.append(msg)

        return compacted_messages

    def _fold_embedded_tool_blocks(self, text: str) -> str:
        """Fold large codeblocks in assistant or user messages from past turns."""
        pattern = r"```([a-zA-Z0-9_-]*)\n([\s\S]*?)```"

        def _replace_match(match: re.Match) -> str:
            lang = match.group(1)
            code = match.group(2)
            lines = code.splitlines()
            if len(lines) > self.max_raw_tool_lines:
                head = "\n".join(lines[:4])
                tail = "\n".join(lines[-3:])
                return f"```{lang}\n{head}\n// ... [{len(lines)-7} lines folded for past context efficiency] ...\n{tail}\n```"
            return match.group(0)

        return re.sub(pattern, _replace_match, text)
