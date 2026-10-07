"""Give every training record the `<think>` block the served model actually needs.

WHY
---
Every corpus under `apps/factory/data/` has zero reasoning traces, and the
trainer rendered records as `### Question: ... ### Answer: ...` -- a format the
served model never sees. At serving time Qwen3.5's template opens every answer
with `<think>\\n`, so the adapters learned "answer instantly and tersely" in a
position the model is never in, and measured -26 to -32 points of HumanEval
pass@1 (MEASURED_FINDINGS §12).

This script adds, per record, the private reasoning that precedes the answer:
understand the task, choose the tool or idiom AND SAY WHY FOR THIS TASK
(polars over pandas because..., uv over pip because..., a dataclass and a
function instead of a class because...), then plan. The opinion is one step in
the reasoning, never the whole of it -- a trace that only states a preference
teaches the model that thinking means stating a preference.

WHO WRITES THE THINKING
-----------------------
The base model itself, through `runtime-next`'s batch endpoint. It is shown the
question and the known-good answer and asked for the reasoning that leads there.
Traces written in a foreign voice (templates, or another model family) would
teach the adapter to imitate that voice -- more drift from the base, which is
the thing that cost correctness last time. Same family, same voice.

Honest limit: these are RATIONALIZED traces -- written knowing the answer. That
suits the goal (teaching *when and why* an opinion applies), but it is weaker
than traces from genuinely solving a problem. Context distillation with a
test-passing filter is the follow-up that fixes that; this is the cheap first
step that repairs the format and the missing reasoning slot.

Nothing is overwritten: output goes to `<input stem>_thinking.jsonl` with the
original record plus a `thinking` field and a `thinking_meta` block. Records
whose generated trace fails validation are written to a separate `_rejected`
file with the reason, never silently kept or silently dropped.

Usage (server must already be running -- single-engine rule):
    uv run python -m apps.factory.corpus.add_thinking apps/factory/data/python_modern/training_data_v6.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

RATIONALE_INSTRUCTIONS = """You are writing the private reasoning a senior practitioner -- engineer,
analyst or advisor, whichever the request calls for -- has BEFORE writing their answer.

Below is a user request and the answer that practitioner eventually wrote. Write only the
thinking that comes first, as if the answer does not exist yet:

1. What the request actually needs (inputs, constraints, edge cases).
2. Which library, tool or idiom fits, and WHY for this specific request -- name the
   realistic alternative and the concrete reason to prefer this one here (for example:
   polars over pandas because the file is large and the operations are columnar; uv over
   pip because it resolves and locks faster and keeps the environment reproducible; a
   frozen dataclass and a plain function instead of a class with methods because there is
   no mutable state to own).
3. A short plan of the steps.

Rules:
- First person, present tense, plain prose. 60 to 220 words.
- No code blocks. Mention function or module names inline if useful.
- Never refer to "the answer", "the reference", "the solution below", or to having been
  shown anything.
- If no tool choice is really at stake, say so briefly and focus on the problem instead of
  inventing a preference.
- Output ONLY the reasoning text.

<request>
{question}
</request>

<answer>
{answer}
</answer>"""

# Phrases that reveal the trace was written while looking at the answer. A trace
# containing them would teach the model to talk about an answer it cannot see.
LEAK_PATTERNS = re.compile(
    r"\b(the (given|provided|reference|above|below|final) (answer|solution|code)|"
    r"reference answer|as shown|the answer (above|below)|i was (given|shown))\b",
    re.IGNORECASE,
)
MIN_WORDS, MAX_WORDS = 40, 320


def split_record(rec: dict) -> tuple[str, str] | None:
    """Recovers (question, answer) from either corpus shape the trainer accepts."""
    if "messages" in rec and len(rec["messages"]) >= 2:
        return rec["messages"][0]["content"], rec["messages"][1]["content"]
    text = rec.get("text", "")
    marker = "\n\n### Answer:\n"
    if marker in text:
        head, answer = text.split(marker, 1)
        return head.removeprefix("### Question:\n"), answer
    return None


def validate(thinking: str, answer: str) -> str | None:
    """Returns a rejection reason, or None if the trace is usable."""
    if "```" in thinking:
        return "contains a code block (the plan must not pre-write the answer)"
    if "<think>" in thinking or "</think>" in thinking:
        return "contains think tags"
    if LEAK_PATTERNS.search(thinking):
        return "refers to having seen an answer"
    words = len(thinking.split())
    if words < MIN_WORDS:
        return f"too short ({words} words)"
    if words > MAX_WORDS:
        return f"too long ({words} words)"
    return None


def post_batch(
    url: str, questions_answers: list[tuple[str, str]], max_tokens: int, timeout: float, teacher_thinking: bool = False
) -> list[dict]:
    payload = {
        # Measured on the pilot: with its own reasoning ON, the 9B teacher spent
        # the whole 2048-token budget drafting the rationale inside <think> and
        # never emitted it (32/32 truncated, ~12 s/record). The rationale IS the
        # visible output here, so the template's own enable_thinking=false
        # branch is the right mode, not a shortcut.
        "chat_template_kwargs": {"enable_thinking": teacher_thinking},
        "batch": [
            {"messages": [{"role": "user", "content": RATIONALE_INSTRUCTIONS.format(question=q, answer=a)}]}
            for q, a in questions_answers
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    req = Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
    except HTTPError as exc:
        # The server's own message (e.g. "HIP error 2: out of memory" when the
        # batch's decode state does not fit) is the actionable part; a bare
        # "500" is not.
        raise RuntimeError(f"batch of {len(questions_answers)} failed: {exc.read().decode(errors='replace')}") from exc
    # The batch endpoint answers with `responses`, each carrying its `index`.
    return sorted(body["responses"], key=lambda r: r["index"])


def extract_rationale(content: str) -> str:
    """The rationale is the teacher's visible output.

    With teacher thinking ON the generation holds the teacher's own reasoning
    first, so the rationale is what follows `</think>`, and an unclosed think
    block means it never got there. With thinking OFF the template already
    closed the think block in the prompt, so the whole generation is the answer.
    """
    if "</think>" in content:
        return content.rsplit("</think>", 1)[1].strip()
    if "<think>" in content:
        return ""
    return content.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--url", default="http://127.0.0.1:8003/v1/chat/completions/batch")
    # Each slot holds its own decode state (534 MB measured at 9B). 9B leaves
    # ~6.5 GB free, so 16 slots OOM and 8 fit; smaller teachers can go higher.
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=768, help="a 220-word rationale is ~300 tokens")
    ap.add_argument("--limit", type=int, default=None, help="only the first N records per file (for pilots)")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="continue a file whose run crashed: skip as many usable records as already have a "
        "kept/rejected row (rows are written in order, whole batches at a time, so the count is exact)",
    )
    ap.add_argument(
        "--teacher-thinking",
        action="store_true",
        help="let the teacher reason before writing (off by default: it exhausts the budget)",
    )
    args = ap.parse_args()

    for path in args.inputs:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.limit:
            records = records[: args.limit]
        out_path = path.with_name(f"{path.stem}_thinking.jsonl")
        rej_path = path.with_name(f"{path.stem}_thinking_rejected.jsonl")
        kept = rejected = done_before = 0
        mode = "w"
        if args.resume and out_path.exists():
            kept = sum(1 for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip())
            rejected = sum(1 for line in rej_path.read_text(encoding="utf-8").splitlines() if line.strip()) \
                if rej_path.exists() else 0
            done_before, mode = kept + rejected, "a"
            print(f"{path.name}: resuming after {done_before} processed records", flush=True)
        t0 = time.perf_counter()
        with out_path.open(mode, encoding="utf-8") as out, rej_path.open(mode, encoding="utf-8") as rej:
            pending = [(i, split_record(r)) for i, r in enumerate(records)]
            for i, qa in pending:
                if qa is None and not done_before:
                    row = {"index": i, "reason": "no question/answer split", "record": records[i]}
                    rej.write(json.dumps(row) + "\n")
                    rejected += 1
            # Unsplittable records are written first, so a resumed count covers them too.
            n_unsplittable = sum(qa is None for _, qa in pending)
            usable = [(i, qa) for i, qa in pending if qa is not None][max(0, done_before - n_unsplittable):]
            for start in range(0, len(usable), args.batch_size):
                chunk = usable[start : start + args.batch_size]
                choices = post_batch(
                    args.url, [qa for _, qa in chunk], args.max_tokens, args.timeout, args.teacher_thinking
                )
                for (i, (_, answer)), choice in zip(chunk, choices, strict=True):
                    content = choice["message"]["content"]
                    thinking = extract_rationale(content)
                    reason = "generation truncated" if choice.get("finish_reason") == "length" else None
                    reason = reason or ("no rationale after </think>" if not thinking else validate(thinking, answer))
                    if reason:
                        row = {"index": i, "reason": reason, "raw": content, "record": records[i]}
                        rej.write(json.dumps(row) + "\n")
                        rejected += 1
                    else:
                        rec = dict(records[i])
                        rec["thinking"] = thinking
                        rec["thinking_meta"] = {
                            "method": "rationalized",
                            "generator": "runtime-next",
                            "words": len(thinking.split()),
                        }
                        out.write(json.dumps(rec) + "\n")
                        kept += 1
                done = start + len(chunk)
                rate = done / max(1e-9, time.perf_counter() - t0)
                where = f"{path.parent.name}/{path.name}"
                print(f"{where}: {done}/{len(usable)} kept={kept} rejected={rejected} ({rate:.2f} rec/s)", flush=True)
        print(f"WROTE {out_path} ({kept} kept) and {rej_path} ({rejected} rejected)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
