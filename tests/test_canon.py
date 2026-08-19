"""Unit tests for gnn_experiment.canon constants, paths, and manifest stamping."""

import pytest
from pathlib import Path
from gnn_experiment.canon import (
    CANON,
    DOMAINS,
    REPO_ROOT,
    adapter_path,
)


def test_canon_invariants():
    """Verify canonical defaults are strictly pinned to preventing regression to legacy caps."""
    assert CANON.MAX_NEW_TOKENS == 2048
    assert CANON.ADAPTER_VERSION == "v4"
    assert CANON.LORA_RANK == 8
    assert CANON.LORA_ALPHA == 128
    assert CANON.BASE_MODEL == "Qwen/Qwen3.5-4B"
    assert CANON.GREEDY is True
    assert REPO_ROOT.exists()
    assert (REPO_ROOT / "src" / "gnn_experiment").exists()


@pytest.mark.parametrize(
    "domain,expected_stem",
    [
        ("astral", "m2_astral_r8a128_v4"),
        ("postgresql", "m2_postgresql_r8a128_v4"),
        ("duckdb", "m2_duckdb_r8a128_v4"),
        ("financial", "m2_financial_r8a128_v4"),
    ],
)
def test_adapter_path_valid_domains(domain: str, expected_stem: str):
    """Test resolution of canonical domain names."""
    path = adapter_path(domain)
    assert isinstance(path, Path)
    assert path.name == expected_stem
    assert path.parent == REPO_ROOT / "results" / "adapters"
    assert path.exists()


@pytest.mark.parametrize("invalid_domain", ["", "kubernetes", "unknown_domain", "redis", "docker"])
def test_adapter_path_invalid_domains_raise_valueerror(invalid_domain: str):
    """Unknown domains must raise explicit ValueError."""
    with pytest.raises(ValueError, match="unknown domain"):
        adapter_path(invalid_domain)


def test_adapter_path_missing_version_raises_filenotfound():
    """Requesting a nonexistent version must raise FileNotFoundError rather than silently falling back."""
    with pytest.raises(FileNotFoundError, match="canonical adapter missing"):
        adapter_path("astral", version="v999_nonexistent")


def test_canon_stamp_metadata():
    """Ensure benchmark result stamping produces all mandatory metadata keys."""
    stamp = CANON.stamp()
    assert isinstance(stamp, dict)
    assert stamp["MAX_NEW_TOKENS"] == 2048
    assert stamp["ADAPTER_VERSION"] == "v4"
    assert stamp["BASE_MODEL"] == "Qwen/Qwen3.5-4B"
    assert stamp["LORA_RANK"] == 8
    assert stamp["LORA_ALPHA"] == 128

