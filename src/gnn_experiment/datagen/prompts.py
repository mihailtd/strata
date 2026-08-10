"""Prompt templates for the augmentoolkit-inspired doc-to-QA flow:
question generation -> answerability verification -> answer generation.
"""

from gnn_experiment.datagen.schema import Chunk

QUESTION_GEN_VERSION = "v1"
VERIFY_VERSION = "v1"
ANSWER_GEN_VERSION = "v1"


def question_gen_messages(chunk: Chunk, nudge: str = "") -> list[dict]:
    system = (
        "You are helping build a training dataset for a developer-focused AI assistant "
        "that is an expert in Astral's Python tooling: uv (package/project manager), "
        "ruff (linter/formatter), and ty (type checker)."
    )
    if nudge:
        system += f"\n\nAdditional guidance for this run: {nudge}"
    user = f"""Below is an excerpt from the official {chunk.tool} documentation
(source: {chunk.source_path}, section: {" > ".join(chunk.heading_path) or chunk.source_path}).

---
{chunk.text}
---

Write ONE realistic question a Python developer would type into a chat assistant
about {chunk.tool}, that this excerpt directly and completely answers. Prefer
concrete, practical questions over vague ones:
- GOOD: "How do I pin uv to a specific Python version in pyproject.toml?"
- GOOD: "What's the ruff equivalent of flake8's --select flag?"
- BAD: "What does this document describe?"
- BAD: "Summarize this section."

If migrating from a similar older tool (pip, pip-tools, virtualenv, flake8, black,
isort, mypy, pyright) is a natural framing for this content, you may phrase the
question that way.

Output ONLY the question text, nothing else."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def verify_messages(chunk: Chunk, question: str) -> list[dict]:
    system = (
        "You are a strict QA reviewer for a training dataset. Answer only YES or NO."
    )
    user = f"""Documentation excerpt:
---
{chunk.text}
---

Question: {question}

Can this question be answered COMPLETELY and ACCURATELY using ONLY the excerpt
above (no outside knowledge required)? Answer with exactly one word: YES or NO."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def answer_gen_messages(chunk: Chunk, question: str, nudge: str = "") -> list[dict]:
    system = (
        "You are an expert assistant for Astral's Python tooling (uv, ruff, ty). "
        "Answer questions accurately and concisely, grounded strictly in the provided "
        "documentation excerpt. Include concrete examples (CLI commands, config "
        "snippets) when the excerpt provides them."
    )
    if nudge:
        system += f"\n\nAdditional guidance for this run: {nudge}"
    user = f"""Documentation excerpt (source: {chunk.source_path}):
---
{chunk.text}
---

Question: {question}

Write a clear, direct answer as if responding to a developer in a chat. Do not
mention "the excerpt" or "the documentation" explicitly — answer as if you just
know this."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
