"""System One Decision Service.

Local high-performance microservice serving non-autoregressive decision gates
(Choice, Noul, Score, and DSH domain routing) in sub-5ms latency.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from apps.factory.decision_engine.models import (
    HARNESS_DOMAIN_DESCRIPTIONS,
    HARNESS_DOMAINS,
    DecisionHeadConfig,
    ModernBertDecisionEngine,
    QwenDecisionEngine,
)

logger = logging.getLogger("decision_service")

# Global singleton state
_STATE: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "model_name": None,
    "ready": False,
}

# -----------------------------------------------------------------------------
# Mapping constants matching router_cli.py
# -----------------------------------------------------------------------------

DOMAIN_PROMPTS = {
    "postgresql": "PostgreSQL 17 pgvector HNSW (<=>) & asyncpg parameterized pooling",
    "python_web": "FastAPI @asynccontextmanager lifespan & Pydantic v2 ConfigDict",
    "duckdb": "DuckDB vectorized SQL with native QUALIFY window analytics",
    "astral": "Astral uv workspace & strict [tool.ruff.lint] tables",
    "python_modern": "Python 3.12 PEP 695 type parameter syntax without legacy TypeVar",
    "financial_planning": "Vectorized numpy Value-at-Risk (VaR) and Conditional VaR (CVaR)",
}

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


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------

class ChoiceRequest(BaseModel):
    prompt: str
    candidates: Optional[list[str]] = None
    candidate_labels: Optional[list[str]] = None
    temperature: Optional[float] = None


class ChoiceResponse(BaseModel):
    chosen_index: int
    chosen_label: Optional[str]
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float


class NoulRequest(BaseModel):
    prompt: str
    threshold: float = 0.5
    temperature: Optional[float] = None


class NoulResponse(BaseModel):
    decision: bool
    probability: float
    confidence: float
    latency_ms: float


class ScoreRequest(BaseModel):
    prompt: str


class ScoreResponse(BaseModel):
    score: float
    latency_ms: float


class RouteRequest(BaseModel):
    prompt: str


class RouteResponse(BaseModel):
    is_multi_expert: bool
    experts: dict[str, float]
    recommended_model_id: str
    active_domains: list[str]
    system_prompt_prefix: str
    rationale: str
    routing_latency_ms: float


# -----------------------------------------------------------------------------
# Model Initialization Helper
# -----------------------------------------------------------------------------

def load_engine(
    model_path: Optional[str] = None,
    model_type: str = "modernbert",
    model_name_or_path: str = "answerdotai/ModernBERT-base",
    device_str: Optional[str] = None,
) -> None:
    """Initialize the decision engine and tokenizer."""
    if device_str is not None:
        device = torch.device(device_str)
    elif torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info("Loading Decision Engine on %s...", device)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    head_cfg = DecisionHeadConfig()

    if model_type == "modernbert":
        model = ModernBertDecisionEngine(model_name_or_path=model_name_or_path, config=head_cfg)
    else:
        model = QwenDecisionEngine(model_name_or_path=model_name_or_path, config=head_cfg)

    if model_path and os.path.exists(model_path):
        logger.info("Loading trained checkpoint from %s", model_path)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    # Precompute candidate cache for the 6 DSH domains
    model.init_harness_cache(tokenizer, device)

    _STATE["model"] = model
    _STATE["tokenizer"] = tokenizer
    _STATE["device"] = device
    _STATE["model_name"] = model_name_or_path
    _STATE["ready"] = True
    logger.info("Decision Engine is ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lifecycle handler: auto-initialize engine on startup
    model_type = os.environ.get("DECISION_MODEL_TYPE", "modernbert")
    model_id = os.environ.get("DECISION_MODEL_ID", "answerdotai/ModernBERT-base")
    checkpoint = os.environ.get("DECISION_CHECKPOINT", "results/models/decision_engine_v1/decision_engine.pt")
    device_str = os.environ.get("DECISION_DEVICE", None)

    load_engine(
        model_path=checkpoint if os.path.exists(checkpoint) else None,
        model_type=model_type,
        model_name_or_path=model_id,
        device_str=device_str,
    )
    yield
    _STATE["ready"] = False


app = FastAPI(title="System One Decision Service", lifespan=lifespan)


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if _STATE["ready"] else "initializing",
        "device": str(_STATE["device"]),
        "model_name": _STATE["model_name"],
        "cached_domains": HARNESS_DOMAINS,
    }


@app.post("/decide/choice", response_model=ChoiceResponse)
def decide_choice(req: ChoiceRequest) -> ChoiceResponse:
    if not _STATE["ready"]:
        raise HTTPException(status_code=503, detail="Decision engine not ready")

    t0 = time.perf_counter()
    model: ModernBertDecisionEngine = _STATE["model"]
    tokenizer = _STATE["tokenizer"]
    device = _STATE["device"]

    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        if req.candidates is not None:
            # Dynamic candidates
            cand_tokens = tokenizer(req.candidates, padding=True, return_tensors="pt").to(device)
            cand_embs = model.get_context_embedding(
                cand_tokens["input_ids"], cand_tokens["attention_mask"]
            )
            labels = req.candidate_labels or [f"option_{i}" for i in range(len(req.candidates))]
            choice_out = model.choice_head(
                model.get_context_embedding(inputs["input_ids"], inputs["attention_mask"]),
                candidate_embeddings=cand_embs,
                candidate_labels=labels,
                temperature=req.temperature,
            )
        else:
            # Pre-cached fixed harness domains
            labels = HARNESS_DOMAINS
            choice_out = model.choice_head(
                model.get_context_embedding(inputs["input_ids"], inputs["attention_mask"]),
                temperature=req.temperature,
            )

    probs = choice_out.probabilities[0].tolist()
    prob_dict = {labels[i]: round(probs[i], 4) for i in range(len(labels))}

    chosen_idx = choice_out.chosen_index[0]
    chosen_lbl = labels[chosen_idx]
    confidence = choice_out.confidence[0]

    lat_ms = (time.perf_counter() - t0) * 1000.0

    return ChoiceResponse(
        chosen_index=chosen_idx,
        chosen_label=chosen_lbl,
        confidence=round(confidence, 4),
        probabilities=prob_dict,
        latency_ms=round(lat_ms, 2),
    )


@app.post("/decide/noul", response_model=NoulResponse)
def decide_noul(req: NoulRequest) -> NoulResponse:
    if not _STATE["ready"]:
        raise HTTPException(status_code=503, detail="Decision engine not ready")

    t0 = time.perf_counter()
    model: ModernBertDecisionEngine = _STATE["model"]
    tokenizer = _STATE["tokenizer"]
    device = _STATE["device"]

    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        pooled = model.get_context_embedding(inputs["input_ids"], inputs["attention_mask"])
        noul_out = model.noul_head(pooled, threshold=req.threshold, temperature=req.temperature)

    prob = float(noul_out.probability[0, 0].item())
    decision = noul_out.decision[0]
    confidence = noul_out.confidence[0]
    lat_ms = (time.perf_counter() - t0) * 1000.0

    return NoulResponse(
        decision=decision,
        probability=round(prob, 4),
        confidence=round(confidence, 4),
        latency_ms=round(lat_ms, 2),
    )


@app.post("/decide/score", response_model=ScoreResponse)
def decide_score(req: ScoreRequest) -> ScoreResponse:
    if not _STATE["ready"]:
        raise HTTPException(status_code=503, detail="Decision engine not ready")

    t0 = time.perf_counter()
    model: ModernBertDecisionEngine = _STATE["model"]
    tokenizer = _STATE["tokenizer"]
    device = _STATE["device"]

    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        pooled = model.get_context_embedding(inputs["input_ids"], inputs["attention_mask"])
        score_out = model.score_head(pooled)

    score_val = score_out.scores_list[0]
    lat_ms = (time.perf_counter() - t0) * 1000.0

    return ScoreResponse(
        score=round(score_val, 2),
        latency_ms=round(lat_ms, 2),
    )


@app.post("/route", response_model=RouteResponse)
def route_prompt(req: RouteRequest) -> RouteResponse:
    """Specialized neural routing endpoint for DSH specialist adapters."""
    if not _STATE["ready"]:
        raise HTTPException(status_code=503, detail="Decision engine not ready")

    t0 = time.perf_counter()
    model: ModernBertDecisionEngine = _STATE["model"]
    tokenizer = _STATE["tokenizer"]
    device = _STATE["device"]

    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        pooled = model.get_context_embedding(inputs["input_ids"], inputs["attention_mask"])
        choice_out = model.choice_head(pooled)

    probs = choice_out.probabilities[0].tolist()
    domains = HARNESS_DOMAINS

    # Retain all domains with probability >= 0.15 for multi-expert fusion
    active_weights = {domains[i]: round(probs[i], 3) for i in range(len(domains)) if probs[i] >= 0.15}

    if not active_weights:
        # If no single domain exceeds 0.15 threshold, fall back to general
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return RouteResponse(
            is_multi_expert=False,
            experts={"general": 1.0},
            recommended_model_id=_FALLBACK_GENERAL_MODEL,
            active_domains=[],
            system_prompt_prefix=(
                "You are an expert autonomous software engineer writing clean modern Python code."
            ),
            rationale="No specialist domain exceeded confidence threshold — using general engine.",
            routing_latency_ms=round(lat_ms, 2),
        )

    # Normalize weights
    total_w = sum(active_weights.values())
    weights = {d: round(w / total_w, 3) for d, w in active_weights.items()}

    is_multi = len(weights) >= 2
    active_set = frozenset(weights.keys())

    model_id = _STACK_MODEL_MAP.get(active_set)
    if model_id is None:
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
        rationale = f"Neural multi-domain detected: {detected_str}."
    else:
        domain = next(iter(weights))
        sys_prefix = (
            f"You are an expert software engineer specializing in {DOMAIN_PROMPTS[domain]}. "
            "Apply domain conventions precisely."
        )
        rationale = f"Neural single domain detected: {domain} (confidence={weights[domain]:.2f})."

    lat_ms = (time.perf_counter() - t0) * 1000.0

    return RouteResponse(
        is_multi_expert=is_multi,
        experts=weights,
        recommended_model_id=model_id,
        active_domains=sorted(weights.keys()),
        system_prompt_prefix=sys_prefix,
        rationale=rationale,
        routing_latency_ms=round(lat_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("DECISION_PORT", 8100))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
