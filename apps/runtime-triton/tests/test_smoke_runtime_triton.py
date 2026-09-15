"""Smoke tests for apps/runtime-triton.

Verifies the Triton 27B W4A16 server source is structurally sound and contains
all mandatory API routes. No GPU, model loading, or network calls are made.
"""

from pathlib import Path

SERVER = Path(__file__).parent.parent / "server.py"
README = Path(__file__).parent.parent / "README.md"
RUN_SH = Path(__file__).parent.parent / "run_server.sh"


def test_server_file_exists() -> None:
    assert SERVER.exists(), "server.py is missing from runtime-triton"


def test_run_server_sh_exists_and_executable() -> None:
    import stat

    assert RUN_SH.exists(), "run_server.sh is missing"
    assert RUN_SH.stat().st_mode & stat.S_IXUSR, "run_server.sh must be executable"


def test_server_has_openai_routes() -> None:
    source = SERVER.read_text()
    for route in [
        "/v1/chat/completions",
        "/v1/models",
        "/health",
        "/api/engine/status",
        "/api/engine/load",
        "/api/engine/unload",
    ]:
        assert route in source, f"Route {route!r} missing from server.py"


def test_server_uses_native_triton_engine() -> None:
    source = SERVER.read_text()
    assert "Native27BEngine" in source, "server.py must import Native27BEngine"


def test_server_does_not_proxy_ollama() -> None:
    """Zero-Mock invariant: Triton server must not proxy to Ollama."""
    source = SERVER.read_text()
    assert "ornith" not in source.lower(), "Must not reference Ornith proxy"
    # Eviction of Ollama on startup is OK; proxying is not
    assert "proxy" not in source.lower(), "Must not proxy to any external engine"


def test_server_default_port_is_8000() -> None:
    source = SERVER.read_text()
    assert '"8000"' in source or "'8000'" in source, "Default port must be 8000"


def test_server_is_single_tenant() -> None:
    """Dispatch queue must be present — no concurrent request handling."""
    source = SERVER.read_text()
    assert "asyncio.Queue" in source, "server.py must use asyncio.Queue for single-tenant dispatch"


def test_readme_exists_and_documents_port() -> None:
    assert README.exists(), "README.md is missing"
    text = README.read_text()
    assert "8000" in text, "README must document default port 8000"
    assert "A/B" in text or "A/B Testing" in text.replace("A/B", "A/B"), "README should document A/B testing protocol"


def test_no_hardcoded_performance_numbers() -> None:
    """Zero-Mock invariant: no hardcoded tok/s numbers anywhere in this engine."""
    import re

    source = SERVER.read_text()
    # Detect patterns like "20.9 tok/s" or "620.4 GB/s" hardcoded as string literals
    suspicious = re.findall(r'["\']\s*\d+\.\d+\s*(tok/s|GB/s)\s*["\']', source)
    assert not suspicious, f"Hardcoded metric strings found: {suspicious} (Zero-Mock invariant)"


def test_server_does_not_import_monolithic_runtime_engine() -> None:
    """Self-sufficiency invariant: runtime-triton must vendor its own engine,
    not reach back into apps/runtime for it. Checks every vendored .py file,
    not just server.py -- a lazy `from runtime.X import Y` inside a method
    body (e.g. native_27b_engine.py's init_syntax_drafter) previously slipped
    past a server.py-only check undetected."""
    for py_file in list(SERVER.parent.glob("*.py")) + list((SERVER.parent / "tests").glob("*.py")):
        if py_file.name == "test_smoke_runtime_triton.py":
            continue
        source = py_file.read_text()
        assert "from runtime.native_27b_engine" not in source, py_file.name
        assert "from runtime import" not in source, py_file.name
        assert "from runtime.canon" not in source, py_file.name
        assert "from runtime." not in source, py_file.name
    assert "runtime_common" in SERVER.read_text(), "gpu_preflight/canon must still come from runtime-common"


def test_engine_deps_are_vendored_locally() -> None:
    """The dependency closure of Native27BEngine and StateHandoff must be copied into
    this project, not imported from apps/runtime -- that's the whole point of
    self-sufficiency."""
    parent = SERVER.parent
    for f in (
        "native_27b_engine.py",
        "w4a16_loader.py",
        "triton_w4a16.py",
        "gguf_unpacker.py",
        "syntax_drafter.py",
        "adapter_stacker.py",
        "state_handoff_27b.py",
    ):
        assert (parent / f).exists(), f"{f} must be vendored locally in runtime-triton"


def test_project_has_own_pyproject() -> None:
    """runtime-triton must be an independently-installable uv project, not
    reliant on the root pyproject.toml's shared venv."""
    pyproject = SERVER.parent / "pyproject.toml"
    assert pyproject.exists(), "runtime-triton must have its own pyproject.toml"
    text = pyproject.read_text()
    assert 'name = "runtime-triton"' in text
    assert "runtime-common" in text, "must depend on runtime-common for gpu_preflight/canon"
