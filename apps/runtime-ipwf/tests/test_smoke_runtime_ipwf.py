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
    for route in [
        "/v1/chat/completions",
        "/v1/models",
        "/health",
        "/api/engine/status",
        "/api/engine/load",
        "/api/engine/unload",
    ]:
        assert route in source, f"Route {route!r} missing from server.py"


def test_server_uses_ipwf_engine_components() -> None:
    source = SERVER.read_text()
    assert "WeightFoldingEngine" in source, "Must use WeightFoldingEngine"
    assert "FoldedCudaGraphDecoder" in source, "Must use FoldedCudaGraphDecoder"
    assert "FoldableExpert" in source, "Must use FoldableExpert"


def test_server_does_not_use_triton_27b_engine() -> None:
    """Separation invariant: IPWF server must not import Native27BEngine."""
    source = SERVER.read_text()
    assert "Native27BEngine" not in source, "runtime-ipwf must NOT use Native27BEngine (belongs in runtime-triton)"


def test_server_default_port_is_8002() -> None:
    source = SERVER.read_text()
    assert '"8002"' in source or "'8002'" in source, (
        "Default port must be 8002 (not 8000 — avoid collision with runtime-triton)"
    )


def test_server_is_single_tenant() -> None:
    source = SERVER.read_text()
    assert "asyncio.Queue" in source, "Must use asyncio.Queue for single-tenant dispatch"


def test_server_supports_9b_model() -> None:
    """IPWF engine must handle both 4B and 9B model loading."""
    source = SERVER.read_text()
    assert "9b" in source.lower() or "9B" in source, "server.py must handle Qwen3.5-9B as well as 4B"


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


def test_engine_deps_are_vendored_locally() -> None:
    """Invariant: all 13 engine and routing modules are vendored locally in apps/runtime-ipwf."""
    expected_modules = [
        "bucketed_speculative.py",
        "cuda_graph.py",
        "cut_set_router.py",
        "dynamic_team_router.py",
        "fused_norm.py",
        "macd_speculation_circuit_breaker.py",
        "mtp_draft.py",
        "notears_causal_scheduler.py",
        "novel_peft.py",
        "range_statistic_gate.py",
        "riemannian_covariance.py",
        "state_handoff.py",
        "state_ring_buffer.py",
        "syntax_drafter.py",
        "tool_trace.py",
    ]
    for mod_name in expected_modules:
        mod_path = SERVER.parent / mod_name
        assert mod_path.exists(), f"Expected vendored module {mod_name} is missing from runtime-ipwf"


def test_does_not_import_monolithic_runtime_engine() -> None:
    """Self-sufficiency invariant: runtime-ipwf must vendor its own weight-folding/
    routing/speculation library, not reach back into apps/runtime for it. Checks
    every vendored .py file and test file, not just server.py, since a lazy
    `from runtime.X import Y` inside a method body would otherwise slip past
    a server.py-only check undetected."""
    # Check top-level files
    for py_file in SERVER.parent.glob("*.py"):
        source = py_file.read_text()
        assert "from runtime import" not in source, f"Found legacy import in {py_file.name}"
        assert "from runtime." not in source, f"Found legacy import in {py_file.name}"
        assert "import runtime." not in source, f"Found legacy import in {py_file.name}"

    # Check test files
    for py_file in (SERVER.parent / "tests").glob("*.py"):
        if py_file.name == "test_smoke_runtime_ipwf.py":
            continue
        source = py_file.read_text()
        assert "from runtime import" not in source, f"Found legacy import in {py_file.name}"
        assert "from runtime." not in source, f"Found legacy import in {py_file.name}"
        assert "import runtime." not in source, f"Found legacy import in {py_file.name}"

    assert "runtime_common" in SERVER.read_text(), "gpu_preflight/canon must still come from runtime-common"


def test_has_own_pyproject() -> None:
    """runtime-ipwf must be an independently-installable uv project, not
    reliant on the root pyproject.toml's shared venv."""
    pyproject = SERVER.parent / "pyproject.toml"
    assert pyproject.exists(), "runtime-ipwf must have its own pyproject.toml"
    text = pyproject.read_text()
    assert 'name = "runtime-ipwf"' in text
    assert "runtime-common" in text, "must depend on runtime-common for gpu_preflight/canon"
