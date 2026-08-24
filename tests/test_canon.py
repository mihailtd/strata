"""Unit tests for runtime.canon constants, paths, and manifest stamping."""

import pytest
from pathlib import Path
from runtime.canon import (
    CANON,
    DOMAINS,
    REPO_ROOT,
    adapter_path,
)


def test_canon_invariants():
    """Verify canonical defaults are strictly pinned to preventing regression to legacy caps."""
    assert CANON.MAX_NEW_TOKENS == 2048
    assert CANON.ADAPTER_VERSION == "v7"
    assert CANON.LORA_RANK == 8
    assert CANON.LORA_ALPHA == 128
    assert CANON.BASE_MODEL == "Qwen/Qwen3.5-4B"
    assert CANON.GREEDY is True
    assert CANON.KV_CACHE_DTYPE == "bfloat16"
    assert CANON.ATTENTION_BACKEND == "sdpa"
    assert REPO_ROOT.exists()
    assert (REPO_ROOT / "src" / "runtime").exists()


@pytest.mark.parametrize(
    "domain,expected_stem",
    [
        ("astral", "m2_astral_r8a128_v7"),
        ("postgresql", "m2_postgresql_r8a128_v7"),
        ("duckdb", "m2_duckdb_r8a128_v7"),
        ("financial", "m2_financial_r8a128_v7"),
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
    assert stamp["ADAPTER_VERSION"] == "v7"
    assert stamp["BASE_MODEL"] == "Qwen/Qwen3.5-4B"
    assert stamp["LORA_RANK"] == 8
    assert stamp["LORA_ALPHA"] == 128
    assert stamp["KV_CACHE_DTYPE"] == "bfloat16"
    assert stamp["ATTENTION_BACKEND"] == "sdpa"


def test_validate_kv_cache_precision():
    """Test Action 1.1 KV cache validation rejects int4/fp4 and accepts valid types."""
    from runtime.canon import validate_kv_cache_precision

    assert validate_kv_cache_precision("bfloat16") == "bfloat16"
    assert validate_kv_cache_precision("float16") == "float16"
    assert validate_kv_cache_precision("int8") == "int8"

    with pytest.raises(ValueError, match="BANNED_PRECISION"):
        validate_kv_cache_precision("int4")

    with pytest.raises(ValueError, match="BANNED_PRECISION"):
        validate_kv_cache_precision("fp4")

    with pytest.raises(ValueError, match="BANNED_PRECISION"):
        validate_kv_cache_precision("q4_0")


def test_configure_deterministic_attention():
    """Test Action 2.3 deterministic attention configuration helper."""
    from runtime.canon import configure_deterministic_attention

    res = configure_deterministic_attention()
    assert isinstance(res, dict)
    assert "deterministic" in res


