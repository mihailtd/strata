"""Smoke tests for apps/runtime-llama — standalone llama.cpp baseline.

Verifies the shell scripts exist and are correctly configured.
No llama.cpp build or model loading is triggered.
"""

import re
import stat
from pathlib import Path

LLAMA = Path(__file__).parent.parent


def test_run_server_sh_exists_and_executable() -> None:
    script = LLAMA / "run_server.sh"
    assert script.exists(), "run_server.sh is missing from runtime-llama"
    assert script.stat().st_mode & stat.S_IXUSR, "run_server.sh must be executable"


def test_setup_sh_exists() -> None:
    assert (LLAMA / "setup.sh").exists(), "setup.sh is missing"


def test_run_server_references_llama() -> None:
    source = (LLAMA / "run_server.sh").read_text()
    assert "llama" in source.lower(), "run_server.sh must invoke llama-server or similar"


def test_readme_exists_and_documents_port() -> None:
    readme = LLAMA / "README.md"
    assert readme.exists(), "README.md is missing"
    text = readme.read_text()
    assert "8001" in text, "README must document default port 8001"


def test_llama_cpp_subdir_exists() -> None:
    """llama.cpp source/build dir must be present (it was checked out as a submodule or clone)."""
    assert (LLAMA / "llama.cpp").is_dir(), "llama.cpp/ subdirectory is missing"


def test_no_hardcoded_throughput_numbers() -> None:
    """Zero-Mock invariant: no hardcoded tok/s metrics in shell scripts."""
    for fname in ["run_server.sh"]:
        path = LLAMA / fname
        if path.exists():
            src = path.read_text()
            matches = re.findall(r"\d+\.\d+\s*(tok/s|GB/s)", src, re.IGNORECASE)
            assert not matches, f"{fname} contains hardcoded metrics {matches} (Zero-Mock invariant)"


def test_no_triton_engine_references() -> None:
    """runtime-llama is a pure llama.cpp baseline; must not embed or call the Triton engine."""
    source = (LLAMA / "run_server.sh").read_text()
    assert "port 8000" not in source and "localhost:8000" not in source, (
        "runtime-llama must not reference port 8000 (runtime-triton's port)"
    )
