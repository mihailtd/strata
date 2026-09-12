"""Smoke tests for apps/harness — the DSH evaluation harness.

Verifies file presence, YAML validity, and Python syntax of all harness
entry points. No live runtime connection is made.
"""

import ast
from pathlib import Path

import yaml

HARNESS = Path(__file__).parent.parent


def test_settings_yaml_valid() -> None:
    settings = HARNESS / "settings.yaml"
    assert settings.exists(), "settings.yaml is missing"
    with settings.open() as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), "settings.yaml must be a YAML mapping"


def test_settings_has_runtime_reference() -> None:
    settings = HARNESS / "settings.yaml"
    with settings.open() as f:
        data = yaml.safe_load(f)
    # DSH settings.yaml uses provider-namespaced keys like 'llm-deepseek', 'llm-pi-ai', etc.
    # At least one must embed a baseURL pointing to our runtime endpoints.
    llm_sections = {k: v for k, v in data.items() if k.startswith("llm-")}
    assert llm_sections, f"settings.yaml should contain at least one 'llm-*' provider section, got: {list(data.keys())}"
    # Verify at least one section points to a localhost runtime
    has_local_endpoint = any(
        "baseURL" in v and "127.0.0.1" in str(v.get("baseURL", ""))
        for v in llm_sections.values()
        if isinstance(v, dict)
    )
    assert has_local_endpoint, "At least one llm-* section must have baseURL pointing to 127.0.0.1 (local runtime)"


def test_coordinator_modules_syntax() -> None:
    module_names = (
        "orchestrator.py",
        "subagent.py",
        "planner.py",
        "types.py",
        "run_coordinator.py",
        "state_handoff_harness.py",
    )
    for name in module_names:
        src = HARNESS / "coordinator" / name
        assert src.exists(), f"coordinator/{name} is missing"
        ast.parse(src.read_text())


def test_state_compactor_syntax() -> None:
    src = HARNESS / "state_compactor.py"
    assert src.exists()
    ast.parse(src.read_text())


def test_test_harness_connection_syntax() -> None:
    src = HARNESS / "test_harness_connection.py"
    assert src.exists()
    ast.parse(src.read_text())


def test_start_dsh_sh_exists_and_executable() -> None:
    import stat

    script = HARNESS / "start_dsh.sh"
    assert script.exists(), "start_dsh.sh is missing"
    assert script.stat().st_mode & stat.S_IXUSR, "start_dsh.sh must be executable"


def test_readme_exists() -> None:
    readme = HARNESS / "README.md"
    assert readme.exists(), "README.md missing from harness"
    text = readme.read_text()
    assert len(text) > 100, "README is too short to be useful"
