"""Unit tests for check_canon.py AST audit rules, exemption paths, and token cap pattern matching."""

import pytest
from audit.check_canon import (
    ADAPTER_RE,
    CANON_ADAPTER_VERSION,
    CANON_MAX_NEW_TOKENS,
    TOKEN_RES,
    grades_quality,
    is_exempt,
)


def test_is_exempt_paths():
    """Verify exemption paths are accurately classified for latency/throughput vs quality benchmarks."""
    assert is_exempt("benchmarks/runtime/speculative/benchmark_mtp_speculative.py") is True
    assert is_exempt("benchmarks/runtime/memory/cuda_graph/benchmark_folded_cuda_graph.py") is True
    assert is_exempt("benchmarks/runtime/performance/batch_scaling/benchmark_batch_scaling.py") is True
    assert is_exempt("benchmarks/superseded/benchmark_speculative_decode.py") is True
    assert is_exempt("benchmarks/runtime/folding/benchmark_weight_folding.py") is True
    assert is_exempt("benchmarks/runtime/folding/evaluate_flash_norm_quality.py") is True

    # Quality and execution benchmarks must NOT be exempt
    assert is_exempt("benchmarks/multi_turn/multi_turn_execution_benchmark.py") is False
    assert is_exempt("benchmarks/applied_execution_gate.py") is False
    assert is_exempt("benchmarks/factory/duckdb_eval/benchmark_duckdb.py") is False


def test_grades_quality_heuristics():
    """Verify grades_quality detects evaluation and scoring markers in benchmark scripts."""
    quality_snippet = """
    def evaluate_results(output):
        score = score_one(q, output, MODERN_TERMS)
        return score
    """
    assert grades_quality(quality_snippet) is True

    execution_snippet = """
    proc = subprocess.run(["ruff", "check", tmp_path], capture_output=True)
    if proc.returncode == 0:
        return {"linter_score": 1.0}
    """
    assert grades_quality(execution_snippet) is True

    latency_only_snippet = """
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    model.forward(input_ids)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    """
    assert grades_quality(latency_only_snippet) is False


def test_token_cap_regex_detection():
    """Verify regex patterns catch illegal short decode budget literals (< 2048)."""
    samples = [
        ("ap.add_argument('--max-new-tokens', type=int, default=192)", 192),
        ("parser.add_argument(\"--max_new_tokens\", type=int, default=64)", 64),
        ("def generate_solution(prompt, max_new_tokens: int = 768):", 768),
        ("max_new_tokens = 448", 448),
    ]

    for line, expected_val in samples:
        matched_val = None
        for rx in TOKEN_RES:
            m = rx.search(line)
            if m:
                matched_val = int(m.group(1))
                break
        assert matched_val == expected_val
        assert matched_val < CANON_MAX_NEW_TOKENS


def test_legacy_adapter_version_regex_detection():
    """Verify ADAPTER_RE flags legacy adapter strings in quality benchmarks."""
    legacy_lines = [
        "adapter_path = 'results/adapters/m2_astral_r8a128_v2'",
        "path = adapters_dir / 'm2_postgresql_r8a128_v3'",
    ]
    for line in legacy_lines:
        m = ADAPTER_RE.search(line)
        assert m is not None
        domain, ver = m.group(1), m.group(2)
        assert domain in ("astral", "postgresql")
        assert ver != CANON_ADAPTER_VERSION

    # v7 is canonical now, so v6 and v4 are legacy strings the checker must flag.
    v7_line = "adapter_path = 'results/adapters/m2_duckdb_r8a128_v7'"
    m4 = ADAPTER_RE.search(v7_line)
    assert m4 is not None
    assert m4.group(2) == CANON_ADAPTER_VERSION
