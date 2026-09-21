"""Code extraction and patch application for whole-file and SEARCH/REPLACE diff formats."""

from __future__ import annotations

import re


def extract_whole_file_code(response_text: str) -> str:
    """Extracts python code from markdown code fences.

    Supports ```python ... ```, ```py ... ```, or plain ``` ... ```.
    If no code fence is found, returns the stripped response text.
    """
    code_fence_pattern = re.compile(
        r"```(?:python|py)?\r?\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    matches = code_fence_pattern.findall(response_text)
    if matches:
        # If multiple code blocks exist, choose the largest block (most likely the code)
        largest = max(matches, key=len)
        return largest.strip() + "\n"

    # Fallback: if text starts with triple backticks without closing or vice versa
    if response_text.strip().startswith("```"):
        lines = response_text.strip().splitlines()
        first_line = lines[0].strip()
        if first_line.startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip() + "\n"

    return response_text.strip() + "\n"


def apply_search_replace_diff(original_content: str, diff_text: str) -> tuple[bool, str]:
    """Applies one or more Aider SEARCH/REPLACE blocks to the original content.

    Format:
    <<<<<<< SEARCH
    old code
    =======
    new code
    >>>>>>> REPLACE

    Returns (success, result_text_or_error_message).
    """
    pattern = re.compile(
        r"<<<<<<<\s*SEARCH\r?\n(.*?)\r?\n=======\r?\n(.*?)\r?\n>>>>>>>\s*REPLACE",
        re.DOTALL,
    )
    blocks = pattern.findall(diff_text)
    if not blocks:
        # Check if model returned a whole-file block instead of search/replace
        if "```" in diff_text:
            extracted = extract_whole_file_code(diff_text)
            if extracted.strip() != diff_text.strip():
                return True, extracted
        return False, "No valid <<<<<<< SEARCH / ======= / >>>>>>> REPLACE blocks found in model response."

    content = original_content
    for idx, (search_block, replace_block) in enumerate(blocks, 1):
        # 1. Exact match
        if search_block in content:
            content = content.replace(search_block, replace_block, 1)
            continue

        # 2. Match ignoring carriage returns
        search_norm = search_block.replace("\r\n", "\n")
        content_norm = content.replace("\r\n", "\n")
        if search_norm in content_norm:
            content = content_norm.replace(search_norm, replace_block.replace("\r\n", "\n"), 1)
            continue

        # 3. Match with leading/trailing whitespace stripped if non-empty
        search_stripped = search_block.strip()
        if search_stripped and search_stripped in content:
            content = content.replace(search_stripped, replace_block.strip(), 1)
            continue

        # 4. Line-by-line whitespace-trimmed search
        search_lines = [line.strip() for line in search_block.splitlines() if line.strip()]
        content_lines = content.splitlines()
        found_start = -1

        for i in range(len(content_lines) - len(search_lines) + 1):
            window = [content_lines[i + j].strip() for j in range(len(search_lines))]
            if window == search_lines:
                found_start = i
                break

        if found_start != -1:
            replacement_lines = replace_block.splitlines()
            new_lines = (
                content_lines[:found_start]
                + replacement_lines
                + content_lines[found_start + len(search_lines):]
            )
            content = "\n".join(new_lines) + "\n"
            continue

        return False, f"SEARCH block #{idx} could not be matched in the target file."

    if not content.endswith("\n"):
        content += "\n"
    return True, content


def apply_edit(original_content: str, response_text: str, edit_format: str) -> tuple[bool, str]:
    """Universal dispatcher for model response editing.

    Args:
        original_content: Existing code in the target file.
        response_text: Raw output from the LLM.
        edit_format: 'whole' or 'diff'.

    Returns:
        (success, modified_code_or_error)
    """
    if edit_format.lower() == "diff":
        return apply_search_replace_diff(original_content, response_text)
    elif edit_format.lower() == "whole":
        code = extract_whole_file_code(response_text)
        return True, code
    else:
        raise ValueError(f"Unknown edit_format: '{edit_format}'. Expected 'whole' or 'diff'.")
