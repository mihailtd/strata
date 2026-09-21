"""Dumps the real aider-bench task prompts, rendered exactly as
`benchmark_mtp_head_aider_bench_acceptance.py` (mtp_head_folding/) renders
them for the real τ=3.52 acceptance measurement `docs/DECISIONS.md` §136
cites -- same system/user template, same 10 real task instructions -- so
the new Rust `runtime-next` serving-gate benchmark
(`apps/runtime-next/src/speculative.rs`,
`bench_real_speculative_vs_graphed_decode_on_real_aider_bench_tasks`) runs
on the SAME real prompts, not a re-invented substitute.

Deliberately does NOT load a model or touch the GPU -- this is pure text
rendering from real task files on disk, so it can run even while the GPU
is busy with other work. The Rust benchmark does its own tokenization and
chat-templating (already cross-validated elsewhere in this crate against
the real tokenizer), so this script dumps raw system/user TEXT, not token
ids -- keeping the two sides independent rather than cross-trusting a
second serialization format.

    uv run --env-file .env benchmarks/runtime/speculative/runtime_next_serving_gate/dump_aider_bench_prompts.py
"""

import json
import sys
from pathlib import Path

from runtime.canon import REPO_ROOT  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "aider_bench"))
from tasks import load_tasks  # noqa: E402

# Verbatim from benchmark_mtp_head_aider_bench_acceptance.py -- must match
# byte-for-byte, since the whole point is running the same real prompts
# the τ=3.52 acceptance number was measured against.
SYSTEM_PROMPT_WHOLE = (
    "You are an expert software engineer. Follow the user's instructions carefully.\n"
    "Implement the requested solution for `{target_file_name}`.\n"
    "Respond ONLY with the complete updated Python code inside a ```python ... ``` code block."
)
USER_PROMPT_WHOLE = (
    "Instructions:\n{instructions}\n\n"
    "Starting stub in `{target_file_name}`:\n"
    "```python\n{stub_code}\n```\n\n"
    "Please implement the full code for `{target_file_name}`."
)


def main():
    out_path = REPO_ROOT / "benchmarks" / "runtime" / "speculative" / "runtime_next_serving_gate" / "aider_bench_prompts.json"

    tasks = load_tasks()
    print(f"{len(tasks)} real aider-bench tasks")

    prompts = []
    for task in tasks:
        system_prompt = SYSTEM_PROMPT_WHOLE.format(target_file_name=task.target_file_name)
        user_prompt = USER_PROMPT_WHOLE.format(
            instructions=task.instructions, target_file_name=task.target_file_name, stub_code=task.stub_code
        )
        prompts.append({"name": task.name, "system_prompt": system_prompt, "user_prompt": user_prompt})
        print(f"  {task.name}: system {len(system_prompt)} chars, user {len(user_prompt)} chars")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"tasks": prompts}, indent=2))
    print(f"\nWrote {len(prompts)} real prompts to {out_path}")


if __name__ == "__main__":
    main()
