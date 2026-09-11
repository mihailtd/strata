"""Smoke tests for apps/runtime-ollama.

Verifies shell scripts exist, are executable, reference Ollama correctly,
and don't contain hardcoded performance numbers.
No live Ollama process is required.
"""

import re
import stat
from pathlib import Path

OLLAMA = Path(__file__).parent.parent


def test_run_server_sh_exists_and_executable() -> None:
    script = OLLAMA / "run_server.sh"
    assert script.exists(), "run_server.sh is missing"
    assert script.stat().st_mode & stat.S_IXUSR, "run_server.sh must be executable"


def test_setup_sh_exists() -> None:
    assert (OLLAMA / "setup.sh").exists(), "setup.sh is missing"


def test_run_server_references_ollama_command() -> None:
    source = (OLLAMA / "run_server.sh").read_text()
    assert "ollama" in source.lower(), "run_server.sh must invoke the ollama command"


def test_readme_exists_and_documents_port() -> None:
    readme = OLLAMA / "README.md"
    assert readme.exists(), "README.md is missing"
    text = readme.read_text()
    assert "11434" in text, "README must document Ollama's default port 11434"


def test_no_hardcoded_throughput_numbers() -> None:
    """Zero-Mock invariant: shell scripts must not hardcode tok/s or GB/s metrics."""
    for fname in ["run_server.sh", "setup.sh"]:
        path = OLLAMA / fname
        if path.exists():
            src = path.read_text()
            matches = re.findall(r"\d+\.\d+\s*(tok/s|GB/s)", src, re.IGNORECASE)
            assert not matches, (
                f"{fname} contains hardcoded metrics {matches} — "
                "all performance numbers must be measured live (Zero-Mock invariant)"
            )


def test_no_ollama_proxying_references() -> None:
    """runtime-ollama talks TO Ollama natively; it must not proxy another engine through Ollama."""
    source = (OLLAMA / "run_server.sh").read_text()
    assert "port 8000" not in source and "localhost:8000" not in source, \
        "runtime-ollama must not proxy to port 8000 (runtime-triton's port)"
