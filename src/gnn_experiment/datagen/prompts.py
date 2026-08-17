"""Prompt templates for the augmentoolkit-inspired doc-to-QA flow:
question generation -> answerability verification -> answer generation.

`domain_description` and `source_label` default to this project's original
use case (Astral's Python tooling docs) so existing configs/callers are
unaffected. Pass overrides (e.g. from a datagen config's `domain_description`
/ `source_label` fields) to point the same 3-stage flow at a different
corpus -- a book instead of tool docs, say -- without the prompts
misleadingly claiming "expert in Astral's Python tooling" over unrelated
content.
"""

from gnn_experiment.datagen.schema import Chunk

QUESTION_GEN_VERSION = "v1"
VERIFY_VERSION = "v1"
ANSWER_GEN_VERSION = "v1"

DEFAULT_DOMAIN_DESCRIPTION = (
    "Astral's Python tooling: uv (package/project manager), ruff (linter/formatter), and ty (type checker)"
)
DEFAULT_SOURCE_LABEL = "documentation"
DEFAULT_QUESTION_STYLE_GUIDANCE = """- GOOD: "How do I pin uv to a specific Python version in pyproject.toml?"
- GOOD: "What's the ruff equivalent of flake8's --select flag?"
- BAD: "What does this document describe?"
- BAD: "Summarize this section."

If migrating from a similar older tool (pip, pip-tools, virtualenv, flake8, black,
isort, mypy, pyright) is a natural framing for this content, you may phrase the
question that way."""


def question_gen_messages(
    chunk: Chunk,
    nudge: str = "",
    domain_description: str = DEFAULT_DOMAIN_DESCRIPTION,
    source_label: str = DEFAULT_SOURCE_LABEL,
    question_style_guidance: str = DEFAULT_QUESTION_STYLE_GUIDANCE,
) -> list[dict]:
    system = (
        "You are helping build a training dataset for a developer-focused AI assistant "
        f"that is an expert in {domain_description}."
    )
    if nudge:
        system += f"\n\nAdditional guidance for this run: {nudge}"
    user = f"""Below is an excerpt from the official {chunk.tool} {source_label}
(source: {chunk.source_path}, section: {" > ".join(chunk.heading_path) or chunk.source_path}).

---
{chunk.text}
---

Write ONE realistic question a Python developer would type into a chat assistant
about {chunk.tool}, that this excerpt directly and completely answers. Prefer
concrete, practical questions over vague ones:
{question_style_guidance}

Output ONLY the question text, nothing else."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def verify_messages(chunk: Chunk, question: str, source_label: str = DEFAULT_SOURCE_LABEL) -> list[dict]:
    system = "You are a strict QA reviewer for a training dataset. Answer only YES or NO."
    user = f"""{source_label.capitalize()} excerpt:
---
{chunk.text}
---

Question: {question}

Can this question be answered COMPLETELY and ACCURATELY using ONLY the excerpt
above (no outside knowledge required)? Answer with exactly one word: YES or NO."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def answer_gen_messages(
    chunk: Chunk,
    question: str,
    nudge: str = "",
    domain_description: str = DEFAULT_DOMAIN_DESCRIPTION,
    source_label: str = DEFAULT_SOURCE_LABEL,
) -> list[dict]:
    system = (
        f"You are an expert technical assistant for {domain_description}. "
        "Provide dense, authoritative, and direct technical answers grounded strictly in the provided "
        f"{source_label} excerpt. Include concrete examples (CLI commands, code, config "
        "snippets) when the excerpt provides them.\n\n"
        "STRICT STYLE CONSTRAINTS:\n"
        "- NEVER use greetings, pleasantries, or filler (DO NOT say 'Hello', 'Gladly', 'Sure', 'Certainly', 'Hey').\n"
        "- Start IMMEDIATELY with the direct technical answer or code block on the first line.\n"
        "- NEVER use closing sign-offs (DO NOT say 'Hope this helps', 'Let me know', 'Happy coding')."
    )
    if nudge:
        system += f"\n\nAdditional guidance for this run: {nudge}"
    user = f"""{source_label.capitalize()} excerpt (source: {chunk.source_path}):
---
{chunk.text}
---

Question: {question}

Write a direct, authoritative technical answer.
CRITICAL RULES:
1. NO greetings, filler words, or polite preambles (no 'Hello', 'Sure', 'Gladly', 'Here is how').
2. Start DIRECTLY with the answer, code, or command.
3. Do not mention 'the excerpt' or 'the document' — answer as domain ground truth.
4. End immediately after the technical explanation — zero closing fluff."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
