"""Generate golden chat-template renders with REAL `transformers`.

`src/chat_template.rs` claims byte parity with Hugging Face's
`apply_chat_template`. This script produces the reference side of that claim:
each model's own template rendered by `transformers` for conversation shapes an
agent harness actually sends. The Rust test `chat_template_goldens_match_transformers`
re-renders every case and asserts exact equality.

Regenerate after adding a model or case (needs the snapshots in the HF cache):

    uv run python apps/runtime-next/tests/fixtures/gen_chat_template_goldens.py
"""

from __future__ import annotations

import json
from pathlib import Path

import transformers
from transformers import AutoTokenizer

HUB = Path.home() / ".cache/huggingface/hub"
MODELS = {
    "qwen3.5-9b": "models--Qwen--Qwen3.5-9B",
    "mimo-v2.6-distill-qwen-9b": "models--XiaomiMiMo--MiMo-V2.6-Distill-Qwen-9B",
}

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city. Température in °C by default.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                "days": {"type": "integer", "minimum": 1},
            },
            "required": ["city"],
        },
    },
}
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_code",
        "description": "Search the repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {"type": "object"},
                "limit": {"type": "number"},
            },
            "required": ["query"],
        },
    },
}
CALL_WEATHER = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}},
}
CALL_SEARCH = {
    "id": "call_2",
    "type": "function",
    "function": {
        "name": "search_code",
        "arguments": {"query": "def main", "filters": {"lang": ["py", "rs"], "exact": True}, "limit": 0.07},
    },
}

CASES = [
    {"name": "plain_user", "messages": [{"role": "user", "content": "Say hi."}]},
    {"name": "system_user", "messages": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Explain uv in one line."}]},
    {"name": "tools_offered", "tools": [WEATHER_TOOL, SEARCH_TOOL],
     "messages": [{"role": "user", "content": "What's the weather in Paris?"}]},
    {"name": "system_and_tools", "tools": [WEATHER_TOOL],
     "messages": [{"role": "system", "content": "Be precise."},
                  {"role": "user", "content": "Weather in Oslo for 3 days?"}]},
    {"name": "multi_turn_with_reasoning", "messages": [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "<think>\nsimple addition\n</think>\n\n4"},
        {"role": "user", "content": "and 3+3?"}]},
    {"name": "tool_round_trip_null_content", "tools": [WEATHER_TOOL],
     "messages": [
         {"role": "user", "content": "Weather in Paris?"},
         {"role": "assistant", "content": None, "tool_calls": [CALL_WEATHER]},
         {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 18, \"sky\": \"clear\"}"}]},
    {"name": "tool_call_nested_typed_args", "tools": [SEARCH_TOOL],
     "messages": [
         {"role": "user", "content": "Find the entry point."},
         {"role": "assistant", "content": "Searching.", "tool_calls": [CALL_SEARCH]},
         {"role": "tool", "tool_call_id": "call_2", "content": "src/main.rs:17"},
         {"role": "user", "content": "Open it."}]},
    {"name": "thinking_disabled", "kwargs": {"enable_thinking": False},
     "messages": [{"role": "user", "content": "Say hi."}]},
    {"name": "unicode", "messages": [
        {"role": "system", "content": "Réponds en français."},
        {"role": "user", "content": "Café naïve 日本語 😀 \"quoted\" \\ back"}]},
    {"name": "full_conversation_no_generation_prompt", "add_generation_prompt": False,
     "messages": [
         {"role": "user", "content": "Install httpx."},
         {"role": "assistant", "content": "<think>\nThe project uses uv.\n</think>\n\nRun `uv add httpx`."}]},
]


def main() -> None:
    out: dict = {"transformers_version": transformers.__version__, "models": {}}
    for label, repo_dir in MODELS.items():
        snaps = sorted((HUB / repo_dir / "snapshots").glob("*"))
        if not snaps:
            print(f"skip {label}: not downloaded")
            continue
        tok = AutoTokenizer.from_pretrained(str(snaps[-1]))
        cases = []
        for case in CASES:
            entry = {"name": case["name"], "messages": case["messages"], "tools": case.get("tools"),
                     "kwargs": case.get("kwargs", {}),
                     "add_generation_prompt": case.get("add_generation_prompt", True)}
            try:
                entry["expected"] = tok.apply_chat_template(
                    case["messages"], tools=case.get("tools"), tokenize=False,
                    add_generation_prompt=entry["add_generation_prompt"], **entry["kwargs"])
            except Exception as exc:  # noqa: BLE001 - a template's own refusal is a valid golden
                entry["expected_error"] = f"{type(exc).__name__}: {exc}"
            cases.append(entry)
        out["models"][label] = {"repo_dir": repo_dir, "cases": cases}
        print(f"{label}: {len(cases)} cases, {sum('expected_error' in c for c in cases)} template errors")
    path = Path(__file__).with_name("chat_template_goldens.json")
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
