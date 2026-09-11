"""Smoke tests for apps/runtime-ipwf.

Verifies the IPWF server source is structurally sound and correctly separated
from the Triton 27B engine. No GPU, model loading, or network calls are made.
"""

from pathlib import Path

SERVER = Path(__file__).parent.parent / "server.py"
README = Path(__file__).parent.parent / "README.md"
RUN_SH = Path(__file__).parent.parent / "run_server.sh"


def test_server_file_exists() -> None:
    assert SERVER.exists(), "server.py is missing from runtime-ipwf"


def test_run_server_sh_exists_and_executable() -> None:
    import stat
    assert RUN_SH.exists(), "run_server.sh is missing"
    assert RUN_SH.stat().st_mode & stat.S_IXUSR, "run_server.sh must be executable"


def test_server_has_openai_routes() -> None:
    source = SERVER.read_text()
    for route in ["/v1/chat/completions", "/v1/models", "/health", "/api/engine/status",
                  "/api/engine/load", "/api/engine/unload"]:
        assert route in source, f"Route {route!r} missing from server.py"


def test_server_uses_ipwf_engine_components() -> None:
    source = SERVER.read_text()
    assert "WeightFoldingEngine" in source, "Must use WeightFoldingEngine"
    assert "FoldedCudaGraphDecoder" in source, "Must use FoldedCudaGraphDecoder"
    assert "FoldableExpert" in source, "Must use FoldableExpert"


def test_server_does_not_use_triton_27b_engine() -> None:
    """Separation invariant: IPWF server must not import Native27BEngine."""
    source = SERVER.read_text()
    assert "Native27BEngine" not in source, \
        "runtime-ipwf must NOT use Native27BEngine (belongs in runtime-triton)"


def test_server_default_port_is_8002() -> None:
    source = SERVER.read_text()
    assert '"8002"' in source or "'8002'" in source, \
        "Default port must be 8002 (not 8000 — avoid collision with runtime-triton)"


def test_server_is_single_tenant() -> None:
    source = SERVER.read_text()
    assert "asyncio.Queue" in source, "Must use asyncio.Queue for single-tenant dispatch"


def test_server_supports_9b_model() -> None:
    """IPWF engine must handle both 4B and 9B model loading."""
    source = SERVER.read_text()
    assert "9b" in source.lower() or "9B" in source, \
        "server.py must handle Qwen3.5-9B as well as 4B"


def test_server_has_riemannian_router() -> None:
    source = SERVER.read_text()
    assert "RiemannianTeamRouter" in source, "Must initialize RiemannianTeamRouter"


def test_server_has_notears_scheduler() -> None:
    source = SERVER.read_text()
    assert "NotearsCausalScheduler" in source, "Must initialize NotearsCausalScheduler"


def test_readme_exists_and_documents_port() -> None:
    assert README.exists(), "README.md is missing"
    text = README.read_text()
    assert "8002" in text, "README must document default port 8002"


def test_no_hardcoded_performance_numbers() -> None:
    import re
    source = SERVER.read_text()
    suspicious = re.findall(r'["\']\s*\d+\.\d+\s*(tok/s|GB/s)\s*["\']', source)
    assert not suspicious, f"Hardcoded metric strings found: {suspicious} (Zero-Mock invariant)"
