"""EPUB -> Chunk extraction: same downstream pipeline (question_gen -> verify
-> answer_gen -> dataset write) as the markdown-docs path in chunk.py/fetch.py,
just a different source. An EPUB chapter is XHTML, so the approach is:
read each spine item in reading order -> strip it to markdown (preserving
headings, so the existing header-aware chunker still works) -> hand off to
`chunk_markdown_text`, the same accumulation logic chunk_markdown_file uses.
"""

from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from markdownify import markdownify

from runtime.datagen.chunk import chunk_markdown_text
from runtime.datagen.schema import Chunk

# Chrome/boilerplate elements that show up on every chapter and would
# otherwise pollute every chunk's context (running headers, nav links, etc).
_STRIP_TAGS = ("nav", "header", "footer")


def _spine_items(book: epub.EpubBook) -> list[epub.EpubHtml]:
    """Spine items in reading order (book.get_items_of_type() does not
    guarantee order; the spine list is the actual reading sequence)."""
    items = []
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if item is not None and item.get_type() == ITEM_DOCUMENT:
            items.append(item)
    return items


def _chapter_to_markdown(item: epub.EpubHtml) -> str:
    # EPUB content documents are XHTML (well-formed XML), so parse as XML
    # rather than BeautifulSoup's HTML-recovery mode.
    soup = BeautifulSoup(item.get_content(), "xml")
    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    # heading_style="ATX" -> "# Heading" syntax, which is what chunk.py's
    # markdown-it token walk keys off of to track heading_path.
    return markdownify(str(soup), heading_style="ATX", strip=["a"])


def chunk_epub_file(
    path: str | Path,
    tool: str,
    chunk_size_chars: int = 2000,
    min_chars: int = 100,
) -> list[Chunk]:
    """Extract Chunks from one EPUB, one chapter (spine item) at a time."""
    path = Path(path)
    book = epub.read_epub(str(path))
    chunks: list[Chunk] = []
    for idx, item in enumerate(_spine_items(book)):
        md_text = _chapter_to_markdown(item)
        chapter_name = Path(item.get_name()).stem or f"chapter_{idx}"
        source_path = f"{path.stem}/{idx:03d}_{chapter_name}"
        chunks.extend(
            chunk_markdown_text(md_text, tool, source_path, chunk_size_chars, min_chars, breadcrumb_label="book")
        )
    return chunks


def chunk_all_epubs(epub_sources: dict[str, Path], chunk_size_chars: int = 2000, min_chars: int = 100) -> list[Chunk]:
    """Mirrors chunk_all()'s dict[str, Path] shape: {source_name: epub_file_path}."""
    chunks: list[Chunk] = []
    for name, epub_path in epub_sources.items():
        chunks.extend(chunk_epub_file(epub_path, name, chunk_size_chars, min_chars))
    return chunks
