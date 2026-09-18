"""System One Decision Service.

Local high-performance microservice serving non-autoregressive decision gates
(Choice, Noul, Score, and DSH domain routing) in sub-10ms latency.

Universal Domain-Agnostic Platform Invariant:
Zero hardcoded domain strings or technology regexes.
Adapters self-describe via metadata / settings.yaml and are discovered dynamically
by the AdapterRegistry.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if (REPO_ROOT / ".env").exists():
    load_dotenv(REPO_ROOT / ".env")

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "apps") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "apps"))

from apps.factory.decision_engine.models import (
    DecisionHeadConfig,
    ModernBertDecisionEngine,
    QwenDecisionEngine,
)
from apps.factory.decision_engine.registry import AdapterRegistry

logger = logging.getLogger("decision_service")

# Global singleton state
_STATE: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "device": None,
    "device_str": None,
    "model_name": None,
    "registry": None,
    "ready": False,
}


# -----------------------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------------------

class ChoiceRequest(BaseModel):
    prompt: str
    candidates: list[str] | None = None
    candidate_labels: list[str] | None = None
    temperature: float | None = None


class ChoiceResponse(BaseModel):
    chosen_index: int
    chosen_label: str | None
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float


class NoulRequest(BaseModel):
    prompt: str
    threshold: float = 0.5
    temperature: float | None = None


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
# Device Resolution Helper
# -----------------------------------------------------------------------------

def resolve_device(device_str: str | None = None) -> tuple[torch.device, str]:
    """Resolve target compute device (CPU, discrete GPU, NPU/MPS).

    Defaults to CPU unless explicitly requested, preventing VRAM contention
    with running LLM engines.
    """
    dev_str = device_str or os.environ.get("DECISION_DEVICE", "cpu").lower()

    if dev_str in ("cuda", "cuda:0", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda:0"), "cuda:0"
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu"), "cpu"

    if dev_str.startswith("cuda:"):
        try:
            dev = torch.device(dev_str)
            # Test simple allocation to avoid driver fault
            _ = torch.zeros(1, device=dev)
            return dev, dev_str
        except Exception as exc:
            logger.warning("Failed to initialize %s (%s). Falling back to CPU.", dev_str, exc)
            return torch.device("cpu"), "cpu"

    if dev_str == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"

    return torch.device("cpu"), "cpu"


# -----------------------------------------------------------------------------
# Model Initialization Helper
# -----------------------------------------------------------------------------

def load_engine(
    model_path: str | None = None,
    model_type: str = "modernbert",
    model_name_or_path: str = "answerdotai/ModernBERT-base",
    device_str: str | None = None,
) -> None:
    """Initialize the decision engine, tokenizer, and adapter registry."""
    device, resolved_str = resolve_device(device_str)
    logger.info("Initializing Decision Engine on device: %s (%s)", resolved_str, device)

    token = os.environ.get("HF_TOKEN")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=True,
            token=token,
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            token=token,
        )
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

    # Discover self-describing adapters dynamically
    registry = AdapterRegistry()
    model.init_from_registry(registry, tokenizer, device)

    _STATE["model"] = model
    _STATE["tokenizer"] = tokenizer
    _STATE["device"] = device
    _STATE["device_str"] = resolved_str
    _STATE["model_name"] = model_name_or_path
    _STATE["registry"] = registry
    _STATE["ready"] = True
    logger.info(
        "Decision Engine ready on %s with %d registered domains: %s",
        resolved_str,
        len(registry.domains),
        registry.domains,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lifecycle handler: auto-initialize engine on startup
    model_type = os.environ.get("DECISION_MODEL_TYPE", "modernbert")
    model_id = os.environ.get("DECISION_MODEL_ID", "answerdotai/ModernBERT-base")
    checkpoint = os.environ.get(
        "DECISION_CHECKPOINT",
        "results/models/decision_engine_v1/decision_engine.pt",
    )
    device_str = os.environ.get("DECISION_DEVICE", "cpu")

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
    registry: AdapterRegistry | None = _STATE.get("registry")
    dev_str = _STATE.get("device_str", "unknown")
    vram_mb = 0.0
    if dev_str.startswith("cuda") and torch.cuda.is_available():
        vram_mb = round(torch.cuda.memory_allocated(_STATE["device"]) / (1024 * 1024), 2)

    return {
        "status": "ok" if _STATE["ready"] else "initializing",
        "device": dev_str,
        "vram_mb": vram_mb,
        "model_name": _STATE.get("model_name"),
        "registered_domains": registry.domains if registry else [],
        "stack_models": (
            {", ".join(sorted(k)): v for k, v in registry.stack_models.items()}
            if registry
            else {}
        ),
    }


@app.post("/registry/reload")
def reload_registry() -> dict[str, Any]:
    """Hot-reload adapter registry without daemon restart."""
    if not _STATE["ready"]:
        raise HTTPException(status_code=503, detail="Decision engine not ready")

    registry: AdapterRegistry = _STATE["registry"]
    model: ModernBertDecisionEngine = _STATE["model"]
    tokenizer = _STATE["tokenizer"]
    device = _STATE["device"]

    registry.discover()
    model.init_from_registry(registry, tokenizer, device)

    return {
        "status": "reloaded",
        "domain_count": len(registry.domains),
        "domains": registry.domains,
        "stack_models": {", ".join(sorted(k)): v for k, v in registry.stack_models.items()},
    }


@app.post("/decide/choice", response_model=ChoiceResponse)
def decide_choice(req: ChoiceRequest) -> ChoiceResponse:
    """Dynamic candidate matching over user-provided or cached candidates."""
    if not _STATE["ready"]:
        raise HTTPException(status_code=503, detail="Decision engine not ready")

    t0 = time.perf_counter()
    model: ModernBertDecisionEngine = _STATE["model"]
    tokenizer = _STATE["tokenizer"]
    device = _STATE["device"]
    registry: AdapterRegistry = _STATE["registry"]

    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        if req.candidates is not None:
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
            labels = model.choice_head.cached_labels or registry.domains
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
    """Calibrated boolean verification (loop detection, test pass/fail check)."""
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
    """Continuous 0-100 rubric score."""
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
    """Universal multi-expert router with calibrated silence gating.

    Routes prompt to the optimal specialist team via dynamic candidate projections.
    """
    if not _STATE["ready"]:
        raise HTTPException(status_code=503, detail="Decision engine not ready")

    t0 = time.perf_counter()
    model: ModernBertDecisionEngine = _STATE["model"]
    tokenizer = _STATE["tokenizer"]
    device = _STATE["device"]
    registry: AdapterRegistry = _STATE["registry"]

    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        pooled = model.get_context_embedding(inputs["input_ids"], inputs["attention_mask"])
        choice_out = model.choice_head(pooled)

    probs = choice_out.probabilities[0].tolist()
    domains = model.choice_head.cached_labels or registry.domains

    chosen_idx = choice_out.chosen_index[0]
    chosen_label = domains[chosen_idx]
    top_confidence = choice_out.confidence[0]

    # Universal Silence Gating Invariant:
    # If general is chosen or confidence is below 0.20 (barely above uniform 1/K baseline),
    # fall back cleanly to base model
    if chosen_label == "general" or top_confidence < 0.20:
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return RouteResponse(
            is_multi_expert=False,
            experts={"general": 1.0},
            recommended_model_id=registry.fallback_general_model,
            active_domains=[],
            system_prompt_prefix=registry.format_system_prompt({"general": 1.0}),
            rationale=(
                f"Confidence below threshold ({top_confidence:.2f} < 0.20) or general intent "
                f"({chosen_label}) — using base engine without specialist adapter interference."
            ),
            routing_latency_ms=round(lat_ms, 2),
        )


    # Multi-expert fusion: retain specialist domains exceeding 0.15 threshold
    active_weights = {
        domains[i]: round(probs[i], 3)
        for i in range(len(domains))
        if domains[i] != "general" and probs[i] >= 0.15
    }

    if not active_weights:
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return RouteResponse(
            is_multi_expert=False,
            experts={"general": 1.0},
            recommended_model_id=registry.fallback_general_model,
            active_domains=[],
            system_prompt_prefix=registry.format_system_prompt({"general": 1.0}),
            rationale="No specialist domain exceeded 0.15 activation threshold — using base engine.",
            routing_latency_ms=round(lat_ms, 2),
        )

    # Physical Energy Norm Normalization (§60)
    # Scale each weight inversely by energy_norm to prevent loud adapters from dominating
    scaled_weights: dict[str, float] = {}
    for d, raw_w in active_weights.items():
        energy = registry.adapters[d].energy_norm if d in registry.adapters else 1.0
        scaled_weights[d] = raw_w / max(energy, 1e-4)

    total_scaled = sum(scaled_weights.values())
    normalized_weights = {d: round(w / total_scaled, 3) for d, w in scaled_weights.items()}

    is_multi = len(normalized_weights) >= 2
    sorted_domains = sorted(normalized_weights.keys())

    # Dynamically resolve target model endpoint and system prompt prefix from registry
    model_id = registry.resolve_model(sorted_domains)
    sys_prefix = registry.format_system_prompt(normalized_weights)

    if is_multi:
        detected_str = ", ".join(
            f"{d} (γ={w:.2f})"
            for d, w in sorted(normalized_weights.items(), key=lambda x: -x[1])
        )
        rationale = f"Neural multi-domain detected: {detected_str}."
    else:
        domain = next(iter(normalized_weights))
        rationale = f"Neural single domain detected: {domain} (confidence={normalized_weights[domain]:.2f})."

    lat_ms = (time.perf_counter() - t0) * 1000.0

    return RouteResponse(
        is_multi_expert=is_multi,
        experts=normalized_weights,
        recommended_model_id=model_id,
        active_domains=sorted_domains,
        system_prompt_prefix=sys_prefix,
        rationale=rationale,
        routing_latency_ms=round(lat_ms, 2),
    )


# -----------------------------------------------------------------------------
# Standalone CLI Entry Point
# -----------------------------------------------------------------------------

def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="System One Decision Service")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP to bind to")
    parser.add_argument("--port", type=int, default=8100, help="Port to bind to")
    parser.add_argument(
        "--device",
        default=os.environ.get("DECISION_DEVICE", "cpu"),
        help="Target device: 'cpu', 'cuda', 'cuda:0', 'mps'",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get(
            "DECISION_CHECKPOINT",
            "results/models/decision_engine_v1/decision_engine.pt",
        ),
        help="Path to trained decision engine checkpoint",
    )
    parser.add_argument(
        "--model-id",
        default="answerdotai/ModernBERT-base",
        help="Base encoder HuggingFace ID",
    )
    parser.add_argument(
        "--model-type",
        default="modernbert",
        choices=["modernbert", "qwen"],
        help="Model architecture",
    )
    args = parser.parse_args()

    # Pass configuration to lifespan via environment
    os.environ["DECISION_DEVICE"] = args.device
    os.environ["DECISION_CHECKPOINT"] = args.checkpoint
    os.environ["DECISION_MODEL_ID"] = args.model_id
    os.environ["DECISION_MODEL_TYPE"] = args.model_type

    print(f"Starting System One Decision Service on http://{args.host}:{args.port} [Device: {args.device}]...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
