"""The trained format must be byte-identical to the served format.

The legacy adapters were trained on `### Question / ### Answer`, a position the
served model is never in. These tests pin the replacement against the REAL
Qwen3.5 tokenizer's own chat template -- not against a string someone believes
the template produces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

FACTORY = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FACTORY), str(FACTORY / "corpus")]

from add_thinking import extract_rationale, split_record, validate  # noqa: E402
from train_expert import load_qwen_chat_records  # noqa: E402

SNAPSHOTS = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots"


def _tokenizer():
    transformers = pytest.importorskip("transformers")
    snaps = sorted(SNAPSHOTS.glob("*")) if SNAPSHOTS.exists() else []
    if not snaps:
        pytest.skip("real Qwen3.5-4B snapshot not on disk")
    return transformers.AutoTokenizer.from_pretrained(str(snaps[-1]))


def _write(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "data.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_qwen_render_is_byte_identical_to_the_real_chat_template(tmp_path):
    tok = _tokenizer()
    q, a = "Install httpx for this project.", "Run `uv add httpx`."
    thinking = "The project is managed with uv, so I add it through uv to keep the lockfile in sync."
    records, skipped = load_qwen_chat_records(
        _write(tmp_path, [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}],
                           "thinking": thinking}])
    )
    assert skipped == 0 and len(records) == 1

    # Serving side: what the model is handed before it generates.
    user_only = [{"role": "user", "content": q}]
    served_prompt = tok.apply_chat_template(user_only, tokenize=False, add_generation_prompt=True)
    assert records[0]["prompt"] == served_prompt

    # Training target: the full conversation as the template renders it.
    full = tok.apply_chat_template(
        [{"role": "user", "content": q}, {"role": "assistant", "content": a, "reasoning_content": thinking}],
        tokenize=False,
    )
    assert records[0]["prompt"] + records[0]["completion"] == full.rstrip("\n")


def test_records_without_thinking_are_skipped_not_trained_empty(tmp_path):
    rows = [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
    records, skipped = load_qwen_chat_records(_write(tmp_path, rows))
    assert records == [] and skipped == 1


def test_legacy_text_records_are_split_and_rendered(tmp_path):
    rows = [{"text": "### Question:\nq?\n\n### Answer:\nans", "thinking": "reasoning here"}]
    records, _ = load_qwen_chat_records(_write(tmp_path, rows))
    assert records[0]["prompt"].startswith("<|im_start|>user\nq?<|im_end|>")
    assert records[0]["completion"] == "reasoning here\n</think>\n\nans<|im_end|>"


# --- add_thinking helpers ---

def test_split_record_handles_both_corpus_shapes():
    msg = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
    assert split_record(msg) == ("Q", "A")
    assert split_record({"text": "### Question:\nQ\n\n### Answer:\nA"}) == ("Q", "A")
    assert split_record({"text": "no marker"}) is None


def test_extract_rationale_takes_text_after_the_generators_own_think():
    assert extract_rationale("<think>\nplanning\n</think>\n\nI need polars here.") == "I need polars here."
    assert extract_rationale("<think>\nnever finished") == ""


def test_extract_rationale_with_teacher_thinking_off_is_the_whole_output():
    # The prompt already ends `<think>\n\n</think>\n\n`, so the generation has no tags.
    assert extract_rationale("  I need polars here.\n") == "I need polars here."


GOOD = ("The file is a few gigabytes and I only need grouped sums, so I reach for polars over pandas: "
        "its lazy scan pushes the filter down and runs the aggregation across all cores without loading "
        "everything into memory first. I read with scan_csv, filter, group by region, and collect once.")


def test_validate_accepts_a_real_reasoning_trace():
    assert validate(GOOD, "answer") is None


@pytest.mark.parametrize(
    ("text", "reason_fragment"),
    [
        ("```python\nx = 1\n```\n" + GOOD, "code block"),
        ("Looking at the reference answer, " + GOOD, "seen an answer"),
        ("As shown below, " + GOOD, "seen an answer"),
        ("Use polars.", "too short"),
        ("word " * 400, "too long"),
    ],
)
def test_validate_rejects_leaks_code_and_bad_lengths(text, reason_fragment):
    reason = validate(text, "answer")
    assert reason is not None and reason_fragment in reason
