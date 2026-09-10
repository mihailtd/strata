"""Dynamic Mixture-of-Adapters (MoA) Intelligent Router.

Analyzes multi-domain user instructions and autonomously:
1. Calculates domain mixture weights gamma_k across the 6 specialist domains.
2. If multiple domains are active (gamma_k >= 0.15 for K >= 2), dynamically fuses the
   required LoRA adapters into a unified multi-expert adapter in <30ms.
3. Injects the matched multi-expert instruction prefix and routes to the execution engine.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from src.runtime.adapter_stacker import DynamicAdapterStacker

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

DOMAIN_PATTERNS = {
    "postgresql": [r"postgres", r"psql", r"asyncpg", r"pgvector", r"hnsw", r"cosine distance", r"vector\(", r"<=>"],
    "python_web": [
        r"fastapi", r"lifespan", r"endpoint", r"router", r"pydantic", r"asynccontextmanager", r"starlette"
    ],
    "duckdb": [r"duckdb", r"parquet", r"qualify", r"olap", r"columnar", r"window function", r"percentile"],
    "astral": [r"uv", r"ruff", r"pyproject\.toml", r"linter", r"formatter", r"workspace"],
    "python_modern": [
        r"pep\s*695", r"generics", r"type parameter", r"type alias", r"class\s+\w+\[T\]", r"def\s+\w+\[T\]"
    ],
    "financial_planning": [
        r"var", r"cvar", r"monte carlo", r"volatility", r"wealth", r"expected return", r"portfolio risk"
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


class DynamicMoARouter:
    """Classifies multi-domain intents and fuses adapters on the fly."""

    def __init__(self, adapters_dir: Path = REPO_ROOT / "results" / "adapters"):
        self.stacker = DynamicAdapterStacker(adapters_dir)
        self.live_stack_dir = adapters_dir / "live_moa_stack"

    def route_and_stack(self, prompt: str) -> dict[str, Any]:
        """Analyzes prompt, determines active experts, and fuses them if multi-domain."""
        t0 = time.perf_counter()
        
        # 1. Compute domain match counts
        scores: dict[str, float] = {}
        text_lower = prompt.lower()

        for domain, patterns in DOMAIN_PATTERNS.items():
            hits = sum(1 for pat in patterns if re.search(pat, text_lower))
            if hits > 0:
                scores[domain] = float(hits)

        # 2. Determine if multi-domain
        if not scores:
            # Fallback to general
            return {
                "is_multi_expert": False,
                "experts": {"general": 1.0},
                "stacked_adapter_path": None,
                "system_prompt": "You are an expert autonomous software engineer writing clean modern Python code.",
                "routing_latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            }

        total_score = sum(scores.values())
        expert_weights = {d: round(s / total_score, 3) for d, s in scores.items() if (s / total_score) >= 0.15}

        # Renormalize
        w_sum = sum(expert_weights.values())
        expert_weights = {d: round(w / w_sum, 3) for d, w in expert_weights.items()}

        is_multi = len(expert_weights) >= 2
        fused_path = None

        if is_multi:
            # Fuse adapters in <30ms
            out_file = self.live_stack_dir / "adapter_model.safetensors"
            _, cfg = self.stacker.stack_adapters(expert_weights, output_path=out_file)
            fused_path = str(out_file)

        # 3. Construct dynamic system steering prompt
        active_specs = [f"• {DOMAIN_PROMPTS[d]}" for d in expert_weights]
        system_prompt = (
            "You are an expert autonomous software engineer with simultaneous multi-domain mastery:\n"
            + "\n".join(active_specs) + "\n"
            "Strictly follow all modern conventions across all active domains."
        )

        dt_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "is_multi_expert": is_multi,
            "experts": expert_weights,
            "stacked_adapter_path": fused_path,
            "system_prompt": system_prompt,
            "routing_latency_ms": round(dt_ms, 2),
        }

    def route_and_bind(self, prompt: str, engine: Any) -> dict[str, Any]:
        """Routes prompt and directly binds the optimal single or stacked LoRA into engine static VRAM."""
        route_res = self.route_and_stack(prompt)
        if route_res["is_multi_expert"]:
            experts = route_res["experts"]
            engine.set_active_stacked_lora({exp: 1.0 for exp in experts})
        else:
            single = next(iter(route_res["experts"]))
            if single != "general":
                engine.set_active_lora(single)
            else:
                engine.clear_loras()
        return route_res
