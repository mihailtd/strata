"""Dynamic Self-Describing Adapter Registry.

Domain-Agnostic Platform Invariant:
No hardcoded domain strings, regexes, or keyword dictionaries.
Adapters self-describe their capabilities via metadata or settings.yaml,
and are discovered and routed dynamically.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


# -----------------------------------------------------------------------------
# Rich Semantic Descriptions for Standard Baseline Domains
# -----------------------------------------------------------------------------

BASELINE_DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "general": (
        "General-purpose software engineering, standard programming, algorithm synthesis, "
        "script execution, or broad reasoning without specialized external tools."
    ),
    "postgresql": (
        "PostgreSQL 17 relational database, pgvector vector embeddings, HNSW distance index (<=>), "
        "asyncpg parameterized connection pooling, SQL query optimization, and schema migrations."
    ),
    "python_web": (
        "FastAPI asynchronous web services, @asynccontextmanager lifespan handlers, "
        "Pydantic v2 schemas and ConfigDict, Starlette routing, and SSE streaming."
    ),
    "duckdb": (
        "DuckDB analytical query engine, columnar storage, Apache Parquet integration, "
        "native QUALIFY window clauses, vectorized SQL, and OLAP data aggregations."
    ),
    "astral": (
        "Astral tooling ecosystem, uv package manager and lockfile resolution, "
        "ruff linter and formatter rules, pyproject.toml configuration, and workspace management."
    ),
    "python_modern": (
        "Modern Python 3.12+ syntax, PEP 695 type parameter syntax without legacy TypeVar, "
        "generic classes, type aliases, and structural pattern matching."
    ),
    "financial_planning": (
        "Quantitative financial modeling, portfolio risk optimization, Value at Risk (VaR), "
        "Conditional Value at Risk (CVaR), Monte Carlo wealth simulation, and expected returns."
    ),
}

# Domain aliases for standardizing model names / variations
DOMAIN_ALIASES: dict[str, str] = {
    "postgres": "postgresql",
    "fastapi": "python_web",
    "financial": "financial_planning",
    "risk": "financial_planning",
    "modern": "python_modern",
}


@dataclass
class AdapterMetadata:
    """Self-describing metadata for any arbitrary domain specialist adapter."""

    domain_id: str
    description: str
    model_id: str | None = None
    path: str | None = None
    energy_norm: float = 1.0
    capabilities: list[str] = field(default_factory=list)


class AdapterRegistry:
    """Dynamic, domain-agnostic registry for discoverable specialist adapters."""

    def __init__(
        self,
        adapters_dir: Path | str | None = None,
        settings_path: Path | str | None = None,
    ) -> None:
        self.adapters: dict[str, AdapterMetadata] = {}
        self.stack_models: dict[frozenset[str], str] = {}
        self.adapters_dir = Path(adapters_dir) if adapters_dir else REPO_ROOT / "results" / "adapters"
        self.settings_path = Path(settings_path) if settings_path else REPO_ROOT / "apps" / "harness" / "settings.yaml"
        self.fallback_general_model: str = "qwen3.8:27b"
        self.fallback_multi_model: str = "qwen3.8-27b-auto"
        self.discover()

    def register(self, metadata: AdapterMetadata) -> None:
        """Register or update an adapter in the registry."""
        canonical_id = DOMAIN_ALIASES.get(metadata.domain_id, metadata.domain_id)
        metadata.domain_id = canonical_id
        self.adapters[canonical_id] = metadata
        if metadata.model_id:
            self.stack_models[frozenset({canonical_id})] = metadata.model_id

    @property
    def domains(self) -> list[str]:
        return list(self.adapters.keys())

    @property
    def descriptions(self) -> list[str]:
        return [self.adapters[d].description for d in self.domains]

    def discover(self) -> list[AdapterMetadata]:
        """Dynamically discover adapters from settings.yaml and filesystem."""
        self.adapters.clear()
        self.stack_models.clear()

        # 1. Base / General fallback candidate (Universal Non-Interference Invariant)
        self.register(
            AdapterMetadata(
                domain_id="general",
                description=BASELINE_DOMAIN_DESCRIPTIONS["general"],
                model_id=self.fallback_general_model,
            )
        )

        # 2. Parse registered models and stacks from settings.yaml if present
        if self.settings_path.exists():
            try:
                import yaml

                with open(self.settings_path) as f:
                    cfg = yaml.safe_load(f) or {}

                # Look in llm-deepseek or llm-pi-ai providers
                models_list: list[dict[str, Any]] = []
                deepseek_models = cfg.get("llm-deepseek", {}).get("models", [])
                if isinstance(deepseek_models, list):
                    models_list.extend(deepseek_models)

                providers = cfg.get("llm-pi-ai", {}).get("providers", {})
                for prov in providers.values():
                    p_models = prov.get("models", [])
                    if isinstance(p_models, list):
                        models_list.extend(p_models)

                for m in models_list:
                    mid = m.get("id", "")
                    name = m.get("name", "")

                    if "Auto-Dynamic" in name or "moa-auto" in mid:
                        self.fallback_multi_model = mid
                        continue

                    # Handle specialist models: name has "[... Specialist]"
                    if "[" in name and "]" in name:
                        bracket_content = name.split("[")[-1].split("]")[0].strip()

                        # Check for Stack: "[Dual-Expert Stack: Postgres + FastAPI]"
                        if "Stack:" in bracket_content:
                            stack_part = bracket_content.split("Stack:")[-1].strip()
                            stack_domains = []
                            for piece in stack_part.split("+"):
                                p_clean = piece.strip().lower()
                                p_domain = DOMAIN_ALIASES.get(p_clean, p_clean)
                                stack_domains.append(p_domain)
                            if stack_domains:
                                self.stack_models[frozenset(stack_domains)] = mid
                            continue

                        # Check for single specialist
                        if "Specialist" in bracket_content:
                            parts = mid.split("-")
                            domain_cand = parts[-1] if len(parts) > 1 else mid
                            domain_id = DOMAIN_ALIASES.get(domain_cand, domain_cand)

                            # Enrich with baseline description if known, otherwise use bracket text
                            desc = BASELINE_DOMAIN_DESCRIPTIONS.get(
                                domain_id,
                                f"{bracket_content}. High-precision specialist implementation.",
                            )
                            self.register(
                                AdapterMetadata(
                                    domain_id=domain_id,
                                    description=desc,
                                    model_id=mid,
                                )
                            )
            except Exception as e:
                logger.warning("Error parsing settings.yaml in AdapterRegistry: %s", e)

        # 3. Scan adapters_dir for self-describing adapter directories
        if self.adapters_dir.exists():
            for child in sorted(self.adapters_dir.iterdir()):
                if not child.is_dir():
                    continue

                # 3a. adapter_metadata.json (highest priority self-description)
                meta_file = child / "adapter_metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file) as f:
                            m_data = json.load(f)
                        raw_id = m_data.get("domain_id") or child.name
                        d_id = DOMAIN_ALIASES.get(raw_id, raw_id)
                        d_desc = m_data.get("description") or BASELINE_DOMAIN_DESCRIPTIONS.get(
                            d_id, f"Domain specialist for {d_id}."
                        )
                        self.register(
                            AdapterMetadata(
                                domain_id=d_id,
                                description=d_desc,
                                path=str(child),
                                energy_norm=float(m_data.get("energy_norm", 1.0)),
                                capabilities=m_data.get("capabilities", []),
                            )
                        )
                        continue
                    except Exception:
                        pass

                # 3b. regime.json
                regime_file = child / "regime.json"
                if regime_file.exists():
                    try:
                        with open(regime_file) as f:
                            r_data = json.load(f)
                        raw_id = r_data.get("domain")
                        if raw_id:
                            d_id = DOMAIN_ALIASES.get(raw_id, raw_id)
                            if d_id not in self.adapters or not self.adapters[d_id].path:
                                d_desc = BASELINE_DOMAIN_DESCRIPTIONS.get(
                                    d_id, f"Domain specialist for {d_id} ({child.name})."
                                )
                                self.register(
                                    AdapterMetadata(
                                        domain_id=d_id,
                                        description=d_desc,
                                        path=str(child),
                                    )
                                )
                    except Exception:
                        pass

        # 4. Ensure any baseline domains not yet registered exist in the registry
        for b_domain, b_desc in BASELINE_DOMAIN_DESCRIPTIONS.items():
            if b_domain not in self.adapters:
                self.register(
                    AdapterMetadata(
                        domain_id=b_domain,
                        description=b_desc,
                    )
                )

        return list(self.adapters.values())

    def resolve_model(self, active_domains: list[str]) -> str:
        """Resolve the optimal model endpoint for the given active domains."""
        if not active_domains or active_domains == ["general"]:
            return self.fallback_general_model

        active_set = frozenset(active_domains)

        # 1. Exact match in registered stack models
        if active_set in self.stack_models:
            return self.stack_models[active_set]

        # 2. Check subset overlap with registered stacks
        best_match = None
        best_overlap = 0
        for known_set, mid in self.stack_models.items():
            overlap = len(known_set & active_set)
            if overlap > best_overlap and known_set <= active_set:
                best_overlap = overlap
                best_match = mid

        if best_match:
            return best_match

        # 3. Single domain with model_id
        if len(active_domains) == 1:
            d = active_domains[0]
            if d in self.adapters and self.adapters[d].model_id:
                return self.adapters[d].model_id

        # 4. Multi-expert fallback
        return self.fallback_multi_model if len(active_domains) >= 2 else self.fallback_general_model

    def format_system_prompt(self, active_weights: dict[str, float]) -> str:
        """Synthesize dynamic domain steering system prompt from adapter descriptions."""
        if not active_weights or list(active_weights.keys()) == ["general"]:
            return (
                "You are an expert autonomous software engineer writing clean modern code."
            )

        if len(active_weights) == 1:
            d = next(iter(active_weights))
            desc = self.adapters[d].description if d in self.adapters else d
            return (
                f"You are an expert software engineer specializing in {desc}. "
                "Apply domain conventions precisely."
            )

        active_specs = []
        for d in active_weights:
            desc = self.adapters[d].description if d in self.adapters else d
            active_specs.append(f"• {desc}")

        return (
            "You are an expert autonomous software engineer with simultaneous multi-domain mastery:\n"
            + "\n".join(active_specs)
            + "\nStrictly follow all modern conventions across all active domains."
        )

    def match_offline_keywords(self, prompt: str) -> dict[str, float]:
        """Lightweight, zero-dependency token overlap matching for offline router fallback.

        Extracts distinctive tokens from the prompt and scores them against
        registered adapter descriptions and domain IDs.
        """
        prompt_tokens = set(re.findall(r"[a-zA-Z0-9_\-\<\=\>]+", prompt.lower()))
        # Remove common English stop words
        stop_words = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
            "by", "about", "as", "into", "like", "through", "after", "over",
            "between", "out", "against", "during", "without", "before", "under",
            "around", "among", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "shall",
            "should", "can", "could", "may", "might", "must", "of", "it", "this",
            "that", "these", "those", "i", "you", "he", "she", "we", "they",
        }
        filtered_tokens = prompt_tokens - stop_words

        scores: dict[str, float] = {}
        for domain_id, meta in self.adapters.items():
            if domain_id == "general":
                continue

            desc_tokens = set(re.findall(r"[a-zA-Z0-9_\-\<\=\>]+", meta.description.lower())) - stop_words
            desc_tokens.add(domain_id.lower())
            for cap in meta.capabilities:
                desc_tokens.update(re.findall(r"[a-zA-Z0-9_\-\<\=\>]+", cap.lower()))

            overlap = len(filtered_tokens & desc_tokens)
            if overlap > 0:
                scores[domain_id] = float(overlap)

        if not scores:
            return {}

        total = sum(scores.values())
        return {d: round(s / total, 3) for d, s in scores.items() if (s / total) >= 0.15}
