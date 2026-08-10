from dataclasses import dataclass


@dataclass
class Chunk:
    tool: str  # "uv" | "ruff" | "ty"
    source_path: str  # relative path within docs/, e.g. "concepts/dependencies.md"
    heading_path: list[str]
    text: str  # chunk content, with heading breadcrumb prepended
    char_count: int


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


@dataclass
class QARecord:
    chunk: Chunk
    question: str
    answer: str
    question_gen_version: str
    answer_gen_version: str

    def to_dict(self, text_field_fallback: str) -> dict:
        return {
            "messages": [
                {"role": "user", "content": self.question},
                {"role": "assistant", "content": self.answer},
            ],
            "meta": {
                "tool": self.chunk.tool,
                "source_path": self.chunk.source_path,
                "heading_path": self.chunk.heading_path,
                "question_gen_version": self.question_gen_version,
                "answer_gen_version": self.answer_gen_version,
            },
            "text": text_field_fallback,
        }
