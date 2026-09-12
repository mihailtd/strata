"""Smoke tests for apps/runtime-vllm.

Verifies shell scripts and project files exist and are wired correctly.
No live vLLM process or GPU is required.
"""

import re
import stat
from pathlib import Path

VLLM = Path(__file__).parent.parent


def test_run_server_sh_exists_and_executable() -> None:
    script = VLLM / "run_server.sh"
    assert script.exists(), "run_server.sh is missing"
    assert script.stat().st_mode & stat.S_IXUSR, "run_server.sh must be executable"


def test_setup_sh_exists_and_executable() -> None:
    script = VLLM / "setup.sh"
    assert script.exists(), "setup.sh is missing"
    assert script.stat().st_mode & stat.S_IXUSR, "setup.sh must be executable"


def test_pyproject_pins_rocm_vllm() -> None:
    text = (VLLM / "pyproject.toml").read_text()
    assert "vllm" in text
    assert "rocm" in text.lower()


def test_run_server_references_vllm_command() -> None:
    source = (VLLM / "run_server.sh").read_text()
    assert "vllm serve" in source, "run_server.sh must invoke `vllm serve`"


def test_readme_exists_and_documents_port() -> None:
    readme = VLLM / "README.md"
    assert readme.exists(), "README.md is missing"
    assert "8004" in readme.read_text(), "README must document the default port 8004"


def test_no_hardcoded_throughput_numbers() -> None:
    """Zero-Mock invariant: shell scripts must not hardcode tok/s or GB/s metrics."""
    for fname in ["run_server.sh", "setup.sh"]:
        path = VLLM / fname
        if path.exists():
            src = path.read_text()
            matches = re.findall(r"\d+\.\d+\s*(tok/s|GB/s)", src, re.IGNORECASE)
            assert not matches, (
                f"{fname} contains hardcoded metrics {matches} — "
                "all performance numbers must be measured live (Zero-Mock invariant)"
            )


def test_no_port_collision_with_other_runtimes() -> None:
    source = (VLLM / "run_server.sh").read_text()
    for other_port in ("8000", "8001", "8002", "8003", "11434"):
        assert f"port {other_port}" not in source and f"localhost:{other_port}" not in source, (
            f"runtime-vllm must not hardcode another runtime's port ({other_port})"
        )
