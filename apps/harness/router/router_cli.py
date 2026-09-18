"""CLI bridge for the DSH moa-router-tool plugin.

Invoked by the JS plugin as a subprocess:
    uv run python -m harness.router.router_cli "<prompt>"

Outputs a single JSON object to stdout. Exits 0 on success, 1 on error.
No torch / GPU deps — only the regex routing math from DynamicMoARouter is used.
Adapter fusion is explicitly skipped so this stays fast and lightweight.
"""

from __future__ import annotations

import json
import re
import sys
import time

# ---------------------------------------------------------------------------
# Inline the routing math here so we never import adapter_stacker
# (which pulls torch, safetensors, etc.)
# ---------------------------------------------------------------------------

DOMAIN_PATTERNS = {
    "postgresql": [
        r"postgres",
        r"psql",
        r"asyncpg",
        r"pgvector",
        r"hnsw",
        r"cosine distance",
        r"vector\(",
        r"<=>",
    ],
    "python_web": [
        r"fastapi",
        r"lifespan",
        r"endpoint",
        r"router",
        r"pydantic",
        r"asynccontextmanager",
        r"starlette",
    ],
    "duckdb": [
        r"duckdb",
        r"parquet",
        r"qualify",
        r"olap",
        r"columnar",
        r"window function",
        r"percentile",
    ],
    "astral": [r"uv", r"ruff", r"pyproject\.toml", r"linter", r"formatter", r"workspace"],
    "python_modern": [
        r"pep\s*695",
        r"generics",
        r"type parameter",
        r"type alias",
        r"class\s+\w+\[T\]",
        r"def\s+\w+\[T\]",
    ],
    "financial_planning": [
        r"var",
        r"cvar",
        r"monte carlo",
        r"volatility",
        r"wealth",
        r"expected return",
        r"portfolio risk",
    ],
}

DOMAIN_PROMPTS = {
    "postgresql": "PostgreSQL 17 pgvector HNSW (<=>) & asyncpg parameterized pooling",
    "python_web": "FastAPI @asynccontextmanager lifespan & Pydantic v2 ConfigDict",
    "duckdb": "DuckDB vectorized SQL with native QUALIFY window analytics",
    "astral": "Astral uv workspace & strict [tool.ruff.lint] tables",
    "python_modern": "Python 3.12 PEP 695 type parameter syntax without legacy TypeVar",
    "financial_planning": "Vectorized numpy Value-at-Risk (VaR) and Conditional VaR (CVaR)",
}

# Maps (frozenset of active domains) → the pre-registered stacked model ID.
# Single-domain → single specialist. Multi-domain → pre-fused stack or auto router.
_STACK_MODEL_MAP: dict[frozenset[str], str] = {
    frozenset({"postgresql"}): "qwen3.8-27b-postgresql",
    frozenset({"python_web"}): "qwen3.8-27b-fastapi",
    frozenset({"duckdb"}): "qwen3.8-27b-duckdb",
    frozenset({"financial_planning"}): "qwen3.8-27b-financial",
    frozenset({"postgresql", "python_web"}): "ornith-1.5-stack-pg-web",
    frozenset({"postgresql", "duckdb", "python_web"}): "ornith-1.5-stack-pg-duck-web",
}

_FALLBACK_MULTI_MODEL = "qwen3.8-27b-auto"
_FALLBACK_GENERAL_MODEL = "qwen3.8:27b"


import urllib.error
import urllib.request


def route_via_service(
    prompt: str,
    url: str = "http://127.0.0.1:8100/route",
    timeout: float = 0.05,
) -> Optional[dict]:
    """Attempt fast neural decision routing via local Decision Service daemon."""
    try:
        data = json.dumps({"prompt": prompt}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                res = json.loads(resp.read().decode("utf-8"))
                res["engine"] = "neural_system_one"
                return res
    except Exception:
        pass
    return None


def route(prompt: str) -> dict:
    t0 = time.perf_counter()

    # Fast path: Check if local System One Decision Service is active
    neural_result = route_via_service(prompt)
    if neural_result is not None:
        return neural_result

    text = prompt.lower()

    scores: dict[str, float] = {}
    for domain, patterns in DOMAIN_PATTERNS.items():
        hits = sum(1 for pat in patterns if re.search(pat, text))
        if hits > 0:
            scores[domain] = float(hits)

    if not scores:
        return {
            "is_multi_expert": False,
            "experts": {"general": 1.0},
            "recommended_model_id": _FALLBACK_GENERAL_MODEL,
            "active_domains": [],
            "system_prompt_prefix": (
                "You are an expert autonomous software engineer writing clean modern Python code."
            ),
            "rationale": "No domain-specific patterns detected — using general engine.",
            "routing_latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    total = sum(scores.values())
    weights = {d: round(s / total, 3) for d, s in scores.items() if (s / total) >= 0.15}
    w_sum = sum(weights.values())
    weights = {d: round(w / w_sum, 3) for d, w in weights.items()}

    is_multi = len(weights) >= 2
    active_set = frozenset(weights.keys())

    model_id = _STACK_MODEL_MAP.get(active_set)
    if model_id is None:
        # Closest match: find the largest known subset
        best_overlap = 0
        for known_set, mid in _STACK_MODEL_MAP.items():
            overlap = len(known_set & active_set)
            if overlap > best_overlap and known_set <= active_set:
                best_overlap = overlap
                model_id = mid
        if model_id is None:
            model_id = _FALLBACK_MULTI_MODEL if is_multi else _FALLBACK_GENERAL_MODEL

    active_specs = [f"• {DOMAIN_PROMPTS[d]}" for d in weights]
    if is_multi:
        sys_prefix = (
            "You are an expert autonomous software engineer with "
            "simultaneous multi-domain mastery:\n"
            + "\n".join(active_specs)
            + "\nStrictly follow all modern conventions across all active domains."
        )
        detected_str = ", ".join(f"{d} (γ={w:.2f})" for d, w in sorted(weights.items(), key=lambda x: -x[1]))
        rationale = f"Multi-domain task detected: {detected_str}."
    else:
        domain = next(iter(weights))
        sys_prefix = (
            f"You are an expert software engineer specializing in {DOMAIN_PROMPTS[domain]}. "
            "Apply domain conventions precisely."
        )
        rationale = f"Single domain detected: {domain} (γ=1.00)."

    return {
        "is_multi_expert": is_multi,
        "experts": weights,
        "recommended_model_id": model_id,
        "active_domains": sorted(weights.keys()),
        "system_prompt_prefix": sys_prefix,
        "rationale": rationale,
        "routing_latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
    }


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: router_cli.py <prompt>"}))
        sys.exit(1)
    prompt = " ".join(sys.argv[1:])
    try:
        result = route(prompt)
        print(json.dumps(result))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
