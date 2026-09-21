"""Dataset loader for HumanEval (164 coding benchmark problems)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent / "data" / "humaneval.jsonl"


@dataclass
class HumanEvalProblem:
    task_id: str
    prompt: str
    entry_point: str
    canonical_solution: str
    test: str

    @property
    def problem_index(self) -> int:
        """Extracts integer index from 'HumanEval/X'."""
        try:
            return int(self.task_id.split("/")[-1])
        except Exception:
            return -1


def load_humaneval_problems(cache_file: Path | None = None) -> list[HumanEvalProblem]:
    """Loads all 164 HumanEval problems from local cache, falling back to HuggingFace hub if needed."""
    file_path = cache_file or DATA_FILE
    problems: list[HumanEvalProblem] = []

    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                problems.append(
                    HumanEvalProblem(
                        task_id=data["task_id"],
                        prompt=data["prompt"],
                        entry_point=data["entry_point"],
                        canonical_solution=data["canonical_solution"],
                        test=data["test"],
                    )
                )
        return problems

    # Fallback to Hugging Face datasets
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        for row in ds:
            item = dict(row)
            f.write(json.dumps(item) + "\n")
            problems.append(
                HumanEvalProblem(
                    task_id=item["task_id"],
                    prompt=item["prompt"],
                    entry_point=item["entry_point"],
                    canonical_solution=item["canonical_solution"],
                    test=item["test"],
                )
            )

    return problems


def get_problem(task_id: str, problems: list[HumanEvalProblem] | None = None) -> HumanEvalProblem:
    """Gets a specific problem by task_id (e.g. 'HumanEval/0')."""
    problist = problems if problems is not None else load_humaneval_problems()
    for p in problist:
        if p.task_id == task_id or p.task_id == f"HumanEval/{task_id}":
            return p
    raise KeyError(f"HumanEval problem '{task_id}' not found.")
