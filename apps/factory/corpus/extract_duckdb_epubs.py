"""Extract text, code, and markdown from the 3 DuckDB EPUB books in apps/factory/data/duckdb/."""

import re
import zipfile

from bs4 import BeautifulSoup
from runtime_common.canon import REPO_ROOT  # noqa: E402

# REPO_ROOT comes from the installed package, never from __file__ arithmetic:
# `.parent.parent` silently resolves to the WRONG directory the moment a file
# is moved, and it broke all 31 scripts during the scripts/ reorg.
DUCKDB_DIR = REPO_ROOT / "apps/factory/data/duckdb"
RAW_OUT_DIR = DUCKDB_DIR / "raw_extracted"
RAW_OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_html(html_content: str) -> tuple[str, list[str]]:
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract code blocks
    code_blocks = []
    for pre in soup.find_all(["pre", "code"]):
        code_text = pre.get_text().strip()
        if len(code_text) > 10 and len(code_text.splitlines()) >= 1:
            code_blocks.append(code_text)

    # Extract clean text
    for s in soup(["script", "style", "nav"]):
        s.decompose()

    text = soup.get_text(separator="\n")
    # Clean whitespace
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    clean_text = "\n".join(lines)
    return clean_text, code_blocks


def extract_epubs():
    epubs = [f for f in DUCKDB_DIR.glob("*.epub")]
    print(f"Found {len(epubs)} EPUB files:")

    total_chapters = 0
    total_code_blocks = 0

    for epub_path in sorted(epubs):
        book_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", epub_path.stem).strip("_")
        book_dir = RAW_OUT_DIR / book_slug
        book_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(epub_path, "r") as z:
            html_files = [n for n in z.namelist() if n.endswith((".html", ".xhtml", ".htm")) and "nav" not in n.lower()]
            print(f"Extracting {epub_path.name}: {len(html_files)} documents...")

            book_text = []
            book_code = []

            for idx, hfile in enumerate(html_files):
                try:
                    content = z.read(hfile).decode("utf-8", errors="ignore")
                    text, codes = clean_html(content)
                    if len(text) > 200:
                        total_chapters += 1
                        total_code_blocks += len(codes)
                        book_text.append(f"# Chapter/Section: {hfile}\n\n{text}\n\n" + "=" * 60 + "\n")
                        book_code.extend(codes)
                except Exception as e:
                    print(f"  Warning: failed to parse {hfile}: {e}")

            (book_dir / "full_text.txt").write_text("\n".join(book_text), encoding="utf-8")
            (book_dir / "code_blocks.json").write_text(
                json_dump := __import__("json").dumps(book_code, indent=2), encoding="utf-8"
            )
            print(f"  -> Saved {len(book_text)} sections, {len(book_code)} code blocks in {book_dir.name}")

    print(f"\nTotal extracted: {total_chapters} sections, {total_code_blocks} code blocks across all 3 books.")


if __name__ == "__main__":
    extract_epubs()
