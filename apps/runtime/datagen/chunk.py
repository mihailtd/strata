"""Header- and fence-aware markdown chunker.

Fenced code blocks come out of markdown-it-py's token stream as single atomic
`fence` tokens, so walking the token stream (rather than splitting on raw
text/regex) makes it structurally impossible to split a chunk in the middle
of a code block.
"""

from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt

from runtime.datagen.schema import Chunk

_MD = MarkdownIt("commonmark")


@dataclass
class _Block:
    text: str
    heading_path: list[str]


def _tokens_to_blocks(tokens) -> list[_Block]:
    """Walk the token stream, grouping content into blocks tagged with the
    heading path active at that point."""
    blocks: list[_Block] = []
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok.type == "heading_open":
            level = int(tok.tag[1])  # "h2" -> 2
            heading_text = tokens[i + 1].content if i + 1 < len(tokens) else ""
            heading_stack = [h for h in heading_stack if h[0] < level]
            heading_stack.append((level, heading_text))
            i += 3  # heading_open, inline, heading_close
            continue

        if tok.type == "fence":
            text = f"```{tok.info}\n{tok.content}```"
            blocks.append(_Block(text, [h[1] for h in heading_stack]))
            i += 1
            continue

        if tok.type == "paragraph_open":
            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            if inline is not None and inline.type == "inline":
                blocks.append(_Block(inline.content, [h[1] for h in heading_stack]))
            i += 3  # paragraph_open, inline, paragraph_close
            continue

        i += 1

    return blocks


def chunk_markdown_text(
    text: str,
    tool: str,
    source_path: str,
    chunk_size_chars: int = 2000,
    min_chars: int = 100,
    breadcrumb_label: str = "docs",
) -> list[Chunk]:
    """Header- and fence-aware chunking of already-in-memory markdown text.

    Shared by `chunk_markdown_file` (reads a .md file) and the EPUB path
    (converts each chapter's HTML to markdown first, see datagen/epub.py) --
    both just need to turn "markdown text + a source label" into `Chunk`s.
    """
    tokens = _MD.parse(text)
    blocks = _tokens_to_blocks(tokens)

    chunks: list[Chunk] = []
    current_text = ""
    current_heading_path: list[str] = []

    def flush():
        nonlocal current_text, current_heading_path
        stripped = current_text.strip()
        if len(stripped) >= min_chars:
            breadcrumb = f"# Context: {tool} {breadcrumb_label} > " + " > ".join(current_heading_path or [source_path])
            chunks.append(
                Chunk(
                    tool=tool,
                    source_path=source_path,
                    heading_path=list(current_heading_path),
                    text=f"{breadcrumb}\n\n{stripped}",
                    char_count=len(stripped),
                )
            )
        current_text = ""

    for block in blocks:
        if block.heading_path != current_heading_path and current_text:
            flush()
        current_heading_path = block.heading_path

        if current_text and len(current_text) + len(block.text) > chunk_size_chars:
            flush()

        current_text += block.text + "\n\n"

    flush()
    return chunks


def chunk_markdown_file(
    path: Path,
    tool: str,
    docs_root: Path,
    chunk_size_chars: int = 2000,
    min_chars: int = 100,
) -> list[Chunk]:
    text = path.read_text(errors="replace")
    source_path = str(path.relative_to(docs_root))
    return chunk_markdown_text(text, tool, source_path, chunk_size_chars, min_chars, breadcrumb_label="docs")


def chunk_all(docs_roots: dict[str, Path], chunk_size_chars: int = 2000, min_chars: int = 100) -> list[Chunk]:
    chunks: list[Chunk] = []
    for tool, docs_root in docs_roots.items():
        for md_path in sorted(docs_root.rglob("*.md")):
            chunks.extend(chunk_markdown_file(md_path, tool, docs_root, chunk_size_chars, min_chars))
    return chunks
