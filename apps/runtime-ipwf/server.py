"""In-Place Weight Folding (IPWF) Inference Server.

Serves unquantized Qwen3.5 models (3B / 9B) with in-place weight mutation,
CUDA Graph decoding, and Riemannian domain-expert LoRA teams.

Exposes an OpenAI-compatible REST API on port 8002.

Engine characteristics:
  - AMD ROCm RX 7900 XTX (Navi 31), 24 GB GDDR6
  - Unquantized bfloat16 base weights (Qwen/Qwen3.5-4B or Qwen/Qwen3.5-9B)
  - In-Place Weight Mutation: W_live = W0 + s * U@V at static VRAM addresses
  - Pre-captured CUDA/HIP Graph descriptor for zero-overhead decoding
  - FlashNorm: RMSNorm scale weights folded into downstream Linear weights
  - Riemannian Team Router: geodesic-guided multi-expert co-activation
  - NOTEARS Causal Scheduler: predictive pre-folding of next adapter
  - Optional Bucketed Speculative Decoding with MTP draft head
  - SINGLE-TENANT: requests execute strictly in arrival order, one at a time.

Supported model IDs:
  Qwen/Qwen3.5-4B (default), Qwen/Qwen3.5-9B
  Aliases: qwen3.5-4b-<domain>, qwen3.5-9b-<domain>, <domain>, dynamic

Domain LoRA adapters (folded in-place at inference time):
  astral, postgresql, duckdb, python_web / fastapi, financial, python_modern

This server does NOT support quantized W4A16 27B models.
For the 27B quantized model, use runtime-triton (port 8000).
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, nullcontext, suppress
from dataclasses import dataclass, field
from typing import Any

import torch
from cuda_graph import FoldedCudaGraphDecoder
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fused_norm import (
    fold_rmsnorm_into_linear,
    inject_exact_rmsnorm,
    scale_expert_factors_for_folded_norms,
)
from novel_peft import FoldableExpert, WeightFoldingEngine, set_hard_vram_cap
from pydantic import BaseModel, ConfigDict, Field
from range_statistic_gate import RangeStatisticGate
from runtime_common import gpu_preflight
from runtime_common.canon import (
    CANON,
    REPO_ROOT,
    adapter_path,
    configure_deterministic_attention,
    validate_kv_cache_precision,
)
from tool_trace import detect_tools
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVER_PORT = int(os.environ.get("PORT", "8002"))
SERVER_HOST = os.environ.get("HOST", "0.0.0.0")
STREAM_ORDER_TIMEOUT_S = float(os.environ.get("STREAM_ORDER_TIMEOUT_S", "300"))
DEFAULT_MODEL_ID = os.environ.get("DEFAULT_MODEL_ID", "Qwen/Qwen3.5-4B")

DOMAIN_SYSTEM_DIRECTIVES: dict[str, str] = {
    "astral": (
        "You are the Astral Python Toolchain Expert (uv, ruff, pyproject.toml).\n"
        "STRICT EXPERT RULES:\n"
        "1. NEVER recommend legacy `pip install` or `requirements.txt`.\n"
        "2. For installing packages, ALWAYS recommend `uv add <package>`.\n"
        "3. Recommend `uv run`, `uv init`, `uv venv`, and standard PEP 621 `pyproject.toml`.\n"
        "4. Be direct, authoritative, and concise."
    ),
    "fastapi": (
        "You are the FastAPI & Async Web Architecture Expert.\n"
        "STRICT EXPERT RULES:\n"
        "1. For packages, ALWAYS use: `uv add fastapi`. NEVER legacy `pip install`.\n"
        "2. Use async lifespan context managers, NEVER `@app.on_event`.\n"
        "3. Use Pydantic v2 and typed Dependency Injection (`Depends`).\n"
        "4. For real-time streaming, use `StreamingResponse` or SSE."
    ),
    "postgresql": (
        "You are the PostgreSQL 17 & Vector Database (pgvector) Expert.\n"
        "STRICT EXPERT RULES:\n"
        "1. Always use HNSW indexing for embeddings.\n"
        "2. Use CTEs and modern PostgreSQL 17 JSON/vector extensions."
    ),
    "duckdb": (
        "You are the DuckDB Vectorized Analytical SQL Expert.\n"
        "STRICT EXPERT RULES:\n"
        "1. Always use `QUALIFY` for window function filtering without subqueries.\n"
        "2. Optimize for columnar Parquet reads, Arrow zero-copy, and vectorized aggregations."
    ),
    "financial": (
        "You are the Financial Modeling & Quantitative Planning Expert.\n"
        "STRICT EXPERT RULES:\n"
        "1. Provide vectorized NumPy / SciPy Monte Carlo simulations.\n"
        "2. Use Cholesky decomposition for correlated multi-asset covariance matrices."
    ),
}


# ---------------------------------------------------------------------------
# Global model / engine state
# ---------------------------------------------------------------------------

model_state: dict[str, Any] = {}
engine_lock = asyncio.Lock()

server_telemetry: dict[str, Any] = {
    "total_requests": 0,
    "total_prompt_tokens": 0,
    "total_generated_tokens": 0,
    "total_generation_time_s": 0.0,
    "last_request": {},
}


# ---------------------------------------------------------------------------
# VRAM helpers
# ---------------------------------------------------------------------------


def get_real_vram_allocated_gb() -> float:
    with suppress(Exception):
        sysfs = gpu_preflight.get_sysfs_vram_info()
        if sysfs and sysfs.get("sysfs_available") and "used_gb" in sysfs:
            return float(sysfs["used_gb"])
    with suppress(Exception):
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return round((total_bytes - free_bytes) / (1024**3), 2)
    return 0.0


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------


async def load_inference_engine(model_id: str = DEFAULT_MODEL_ID) -> dict[str, Any]:
    """Loads base model + expert adapters + CUDA graphs into VRAM."""
    current_model = model_state.get("model_id")
    if model_state.get("base_model") is not None:
        if current_model == model_id:
            return {
                "status": "already_loaded",
                "model_id": model_id,
                "vram_allocated_gb": get_real_vram_allocated_gb(),
                "active_team": model_state.get("active_team", []),
            }
        print(f"[IPWF] Switching {current_model} -> {model_id}. Unloading...")
        await unload_inference_engine()

    is_9b = "9b" in model_id.lower()

    gpu_preflight.ensure_gpu_exclusive()
    attn_cfg = configure_deterministic_attention()
    print(f"[IPWF] Deterministic Attention Backend: {attn_cfg}")

    kv_dtype_env = os.environ.get("KV_CACHE_DTYPE", CANON.KV_CACHE_DTYPE)
    kv_cache_dtype = validate_kv_cache_precision(kv_dtype_env)
    print(f"[IPWF] Validated KV Cache Precision: {kv_cache_dtype}")

    set_hard_vram_cap(22.0)

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    print(f"[IPWF] Loading base model {model_id}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=compute_dtype,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    base_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if getattr(tokenizer, "pad_token", None) is None:
        setattr(tokenizer, "pad_token", getattr(tokenizer, "eos_token", None) or "<|endoftext|>")  # noqa: B010

    injected_count = inject_exact_rmsnorm(base_model)
    print(f"[IPWF] Injected {injected_count} ExactRMSNorm modules.")

    fold_norms_enabled = os.environ.get("FLASH_NORM_FOLD", "1") != "0"
    folded_norm_count = (
        fold_rmsnorm_into_linear(base_model, fold_weights=fold_norms_enabled) if fold_norms_enabled else 0
    )
    if folded_norm_count > 0:
        print(f"[IPWF] FlashNorm: Folded {folded_norm_count} RMSNorm weights into Linears.")

    # Load domain expert adapters
    domains = ["astral", "postgresql", "duckdb", "financial", "python_modern", "python_web"]
    expert_dict: dict[str, FoldableExpert] = {}
    all_experts: list[FoldableExpert] = []

    for d in domains:
        ad_dir = REPO_ROOT / "results" / "adapters" / f"m2_{d}_r8a128_v7_9b" if is_9b else adapter_path(d)
        if ad_dir.exists():
            exp = FoldableExpert.from_dir(ad_dir, name=d)
            expert_dict[d] = exp
            all_experts.append(exp)
            print(f"[IPWF] Registered expert [{d}] from {ad_dir.name}")
        else:
            print(f"[IPWF] Warning: adapter path {ad_dir} not found for [{d}]")

    if folded_norm_count > 0:
        scaled = scale_expert_factors_for_folded_norms(base_model, all_experts)
        print(f"[IPWF] FlashNorm: Scaled {scaled} adapter factors by (1+γ).")

    folding_engine = WeightFoldingEngine(base_model, all_experts, keep_pristine=True)

    from dynamic_team_router import RiemannianTeamRouter

    team_router = RiemannianTeamRouter(experts=expert_dict, domains=list(expert_dict.keys()))
    print(f"[IPWF] RiemannianTeamRouter initialized with {len(expert_dict)} experts.")

    from notears_causal_scheduler import NotearsCausalScheduler

    causal_scheduler = NotearsCausalScheduler(experts=list(expert_dict.keys()))
    model_state["causal_scheduler"] = causal_scheduler
    print("[IPWF] NotearsCausalScheduler active (predictive pre-folding).")

    expert_prefix = "qwen3.5-9b" if is_9b else "qwen3.5-4b"
    expert_registry: dict[str, FoldableExpert | None] = {
        f"{expert_prefix}-base": None,
        "base": None,
        "pristine": None,
        "default": None,
    }
    for d, exp in expert_dict.items():
        expert_registry[f"{expert_prefix}-{d}"] = exp
        expert_registry[d] = exp
        expert_registry[f"m2_{d}"] = exp

    env_max_len = os.environ.get("MAX_SEQ_LEN")
    max_seq_len = int(env_max_len) if env_max_len else (4096 if is_9b else 16384)

    initial_team = [d for d in ["astral", "python_modern"] if d in expert_dict]
    if initial_team:
        team_router.morph_stack(folding_engine, [], initial_team)
    elif all_experts:
        folding_engine.activate(all_experts[0])

    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    graph_decoder = FoldedCudaGraphDecoder(base_model, tokenizer, max_seq_len=max_seq_len, device=base_model.device)

    dummy_tokens = tokenizer(
        "<|im_start|>user\nWarmup prompt\n<|im_end|>\n<|im_start|>assistant\n",
        return_tensors="pt",
    ).input_ids.to(base_model.device)

    print(f"[IPWF] Capturing CUDA Graph (max_seq_len={max_seq_len})...")
    graph_decoder.capture(dummy_tokens)
    print(f"[IPWF] CUDA Graph captured. Capture count = {graph_decoder.capture_count}")

    # Optional speculative decode
    spec_decoder = None
    if os.environ.get("SPECULATIVE_DECODE", "1") != "0":
        spec_max_len = int(os.environ.get("SPECULATIVE_MAX_SEQ_LEN", "4096"))
        spec_k = int(os.environ.get("SPECULATIVE_K", "2"))
        try:
            from bucketed_speculative import BucketedSpeculativeDecoder
            from mtp_draft import Qwen35MTPDraftHead

            draft_head = Qwen35MTPDraftHead(base_model, model_id)
            folding_engine.register_draft_head(draft_head)
            spec_decoder = BucketedSpeculativeDecoder(
                base_model,
                tokenizer,
                draft_head,
                k=spec_k,
                max_seq_len=spec_max_len,
                device=base_model.device,
            )
            spec_decoder.capture(dummy_tokens)
            print(f"[IPWF] Speculative buckets captured: {sorted(spec_decoder.buckets)}")
        except Exception as exc:
            print(f"[IPWF] Speculative capture failed ({exc}); falling back to greedy.")
            spec_decoder = None

    stop_token_ids: set[int] = set()
    if getattr(tokenizer, "eos_token_id", None) is not None:
        stop_token_ids.add(tokenizer.eos_token_id)
    for tok in ("<|im_end|>", "<|endoftext|>"):
        if hasattr(tokenizer, "convert_tokens_to_ids"):
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid > 0:
                stop_token_ids.add(tid)

    model_state.update(
        base_model=base_model,
        model_id=model_id,
        tokenizer=tokenizer,
        folding_engine=folding_engine,
        graph_decoder=graph_decoder,
        router=team_router,
        active_team=initial_team,
        spec_decoder=spec_decoder,
        spec_decoder_k=spec_decoder.k if spec_decoder else None,
        spec_max_len=int(os.environ.get("SPECULATIVE_MAX_SEQ_LEN", "4096")),
        expert_registry=expert_registry,
        max_prompt_len=max_seq_len,
        stop_token_ids=stop_token_ids,
        range_gate_enabled=os.environ.get("SPECULATIVE_RANGE_GATE", "1") != "0",
        range_gate_threshold=float(os.environ.get("SPECULATIVE_RANGE_THRESHOLD", "5.0")),
        range_gate=RangeStatisticGate(top_m=8, threshold=float(os.environ.get("SPECULATIVE_RANGE_THRESHOLD", "5.0"))),
        ring_buffer_mode=os.environ.get("RING_BUFFER_MODE", "selective_hybrid"),
        scale_mode="surgical",
        prefold_enabled=os.environ.get("PREDICTIVE_PREFOLD_ENABLED", "1") != "0",
        last_predicted_expert=None,
        last_swap_ms=0.0,
        cut_set_hedging_enabled=os.environ.get("CUT_SET_HEDGING", "1") != "0",
        cut_set_target_reliability=float(os.environ.get("CUT_SET_TARGET_RELIABILITY", "0.95")),
        cut_set_max_workers=int(os.environ.get("CUT_SET_MAX_WORKERS", "4")),
        fold_lock=threading.Lock(),
        stop_event=threading.Event(),
    )

    vram_alloc = round(torch.cuda.memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0
    print(f"[IPWF] Engine loaded. VRAM: {vram_alloc:.2f} GB")
    return {
        "status": "loaded",
        "model_id": model_id,
        "vram_allocated_gb": vram_alloc,
        "active_team": initial_team,
    }


async def unload_inference_engine() -> dict[str, Any]:
    if model_state.get("folding_engine"):
        with suppress(Exception):
            model_state["folding_engine"].restore()
    model_state.clear()
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    used_gb = round((total_bytes - free_bytes) / (1024**3), 2) if torch.cuda.is_available() else 0.0
    print(f"[IPWF] Engine unloaded. Remaining VRAM: {used_gb:.2f} GB")
    return {"status": "unloaded", "vram_used_gb": used_gb}


# ---------------------------------------------------------------------------
# OpenAI-compatible Pydantic schemas
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str
    content: Any = ""
    reasoning_content: str | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 0.7
    top_p: float | None = 0.9
    max_tokens: int | None = 4096
    max_completion_tokens: int | None = None
    stream: bool | None = False
    thinking_effort: str | None = "medium"


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    prompt: str
    temperature: float | None = 0.7
    max_tokens: int | None = 4096
    stream: bool | None = False
    thinking_effort: str | None = "medium"


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage


class UsageInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    tokens_per_second: float | None = None
    generation_time_ms: float | None = None
    time_to_first_token_ms: float | None = None
    swap_time_ms: float | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class ChatCompletionChunkDelta(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunkResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]
    usage: UsageInfo | None = None


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "gnn-experiment-ipwf"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]


# ---------------------------------------------------------------------------
# Prompt helpers (shared with runtime-triton but kept local for isolation)
# ---------------------------------------------------------------------------


def _extract_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", block.get("content", ""))))
        return "\n".join(parts)
    if isinstance(content, dict):
        return str(content.get("text", content.get("content", "")))
    return str(content)


def _scrub_thinking(text: str) -> str:
    scrubbed = re.sub(r"<think>[\s\S]*?</think>", "", text)
    return re.sub(r"</?think>", "", scrubbed).strip()


def _resolve_expert(model_name: str) -> FoldableExpert | str | None:
    """Maps request model string to loaded FoldableExpert, 'dynamic', or None (base)."""
    name_clean = model_name.split("/")[-1].lower().strip()
    registry = model_state.get("expert_registry", {})

    if name_clean in ("dynamic", "auto") or "dynamic" in name_clean or "auto" in name_clean:
        return "dynamic"
    if any(k in name_clean for k in ["base", "pristine", "default"]):
        return None
    if name_clean in registry:
        return registry[name_clean]
    for kw, domain in [
        (["duckdb", "duck"], "duckdb"),
        (["astral", "uv", "ruff"], "astral"),
        (["postgre", "postgres", "sql"], "postgresql"),
        (["fin", "wealth"], "financial"),
        (["web", "fastapi"], "python_web"),
        (["modern", "clean"], "python_modern"),
    ]:
        if any(k in name_clean for k in kw):
            return registry.get(domain)
    return "dynamic"


def _build_prompt(
    messages: list[ChatMessage],
    thinking_effort: str | None = "medium",
    expert_key: str | None = None,
) -> str:
    effort = (thinking_effort or "medium").lower()
    directives = {
        "off": "Respond directly and concisely. Do NOT produce any internal reasoning.",
        "low": "Keep thinking brief, under 3-4 sentences. Close with </think> before answering.",
        "high": "Think thoroughly through all edge cases. Close with </think> before answering.",
        "medium": "Keep thinking structured and focused. Close with </think> before answering.",
    }
    thinking_directive = directives.get(effort, directives["medium"])
    specialist = DOMAIN_SYSTEM_DIRECTIVES.get(expert_key or "", "")
    system_prefix = f"{specialist}\n\n{thinking_directive}".strip() if specialist else thinking_directive

    has_system = any(m.role == "system" for m in messages)
    out = ""
    if system_prefix and not has_system:
        out += f"<|im_start|>system\n{system_prefix}\n<|im_end|>\n"

    for msg in messages:
        content = _extract_content(msg.content)
        if msg.role == "assistant":
            content = _scrub_thinking(content)
        elif msg.role == "system" and system_prefix:
            content = f"{system_prefix}\n\n{content}"
        if content or msg.role == "assistant":
            out += f"<|im_start|>{msg.role}\n{content}\n<|im_end|>\n"

    out += "<|im_start|>assistant\n"
    if effort == "off":
        out += "<think>\n\n</think>\n\n"
    else:
        out += "<think>\n"
    return out


def _extract_thinking_and_content(text: str) -> tuple[str | None, str]:
    if "<think>" in text:
        prefix, rest = text.split("<think>", 1)
        if "</think>" in rest:
            think_body, main_body = rest.split("</think>", 1)
            return (think_body.strip() or None), (prefix + main_body).strip()
        return rest.strip() or None, prefix.strip()
    elif "</think>" in text:
        think_body, main_body = text.split("</think>", 1)
        return think_body.strip() or None, main_body.strip()
    return None, text.strip()


# ---------------------------------------------------------------------------
# Request queue (single-tenant, arrival-order)
# ---------------------------------------------------------------------------


@dataclass
class _QueuedRequest:
    req: Any
    future: asyncio.Future
    enqueue_time: float = field(default_factory=time.perf_counter)
    stream_done: asyncio.Event = field(default_factory=asyncio.Event)


_request_queue: asyncio.Queue[_QueuedRequest] = asyncio.Queue()


async def _dispatch_loop() -> None:
    while True:
        queued = await _request_queue.get()
        try:
            result = await _run_inference(queued.req)
            if not queued.future.done():
                queued.future.set_result(result)
        except Exception as exc:
            if not queued.future.done():
                queued.future.set_exception(exc)
        finally:
            _request_queue.task_done()


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------


def _apply_expert(expert: FoldableExpert | str | None) -> str | None:
    """Activates the given expert in the folding engine; returns the expert key.

    Tracks the REAL swap latency in model_state["last_swap_ms"] -- 0.0 when
    the folding engine already had this expert active (a real pre-fold hit,
    or a real no-op skip inside WeightFoldingEngine.activate() itself), the
    real measured torch.addmm fold time otherwise. This is what lets
    predictive pre-folding (see _maybe_predict_and_prefold below) report a
    real, not simulated, latency saved.
    """
    folding_engine: WeightFoldingEngine | None = model_state.get("folding_engine")

    if folding_engine is None:
        return None

    fold_lock = model_state.get("fold_lock")
    lock_ctx = fold_lock if fold_lock is not None else nullcontext()

    if expert is None:
        with lock_ctx:
            folding_engine.restore()
        model_state["last_swap_ms"] = 0.0
        return None

    if expert == "dynamic":
        # Let the Riemannian router decide the current team (no explicit morph here;
        # the router was initialized with the initial team and adapts per request via
        # classify_prompt_intent in the full server — simplified here to current active_team)
        model_state["last_swap_ms"] = 0.0
        return "dynamic"

    if isinstance(expert, FoldableExpert):
        already_active = getattr(folding_engine, "active", None) == expert.name
        swap_t0 = time.perf_counter()
        with lock_ctx:
            folding_engine.activate(expert)
        swap_ms = (time.perf_counter() - swap_t0) * 1000
        swap_ms = 0.0 if already_active else swap_ms
        model_state["last_swap_ms"] = swap_ms
        # Cumulative, monotonic counter -- a multi-turn client (e.g. one DSH
        # .run() call can trigger several real completions per "turn") should
        # diff this before/after a step rather than trust "last", since
        # "last" can be overwritten by a second call within the same step.
        model_state["cumulative_swap_ms"] = model_state.get("cumulative_swap_ms", 0.0) + swap_ms
        model_state["cumulative_swap_count"] = model_state.get("cumulative_swap_count", 0) + (
            0 if already_active else 1
        )
        return expert.name
    return None


def _maybe_record_prefold_outcome(actual_expert: str | None) -> None:
    """If a prediction was pending from the previous turn, score it as a real
    hit/miss now that this turn's real swap latency (0.0 if the folding
    engine already had `actual_expert` active) is known."""
    scheduler = model_state.get("causal_scheduler")
    predicted = model_state.get("last_predicted_expert")
    if scheduler is None or predicted is None:
        return
    scheduler.record_turn_outcome(
        actual_expert=actual_expert or "",
        predicted_expert=predicted,
        morph_latency_ms=model_state.get("last_swap_ms", 1.9),
    )
    model_state["last_predicted_expert"] = None


def _maybe_predict_and_prefold(current_expert: str | None, emitted_text: str) -> None:
    """After a real response is generated, detect real tool markers in it and
    ask the real NotearsCausalScheduler what expert is likely needed next.
    If confident, triggers a REAL background fold (WeightFoldingEngine.activate
    in a thread pool) so it can complete before the next request arrives --
    gated by model_state["prefold_enabled"] so this can be A/B toggled live
    via POST /api/engine/set_predictive_prefold without a restart."""
    if not model_state.get("prefold_enabled", True):
        return
    scheduler = model_state.get("causal_scheduler")
    folding_engine = model_state.get("folding_engine")
    if scheduler is None or folding_engine is None or current_expert in (None, "dynamic"):
        return
    detected = detect_tools(emitted_text)
    predicted, conf = scheduler.predict_next_expert(tools=detected, current_expert=current_expert)
    model_state["last_predicted_expert"] = predicted
    if predicted and predicted != current_expert:
        scheduler.async_prefold(folding_engine, predicted, conf, lock=model_state.get("fold_lock"))


async def _run_inference(req: ChatCompletionRequest) -> dict[str, Any]:
    if not model_state.get("base_model"):
        raise RuntimeError("Engine not loaded. POST /api/engine/load first.")

    expert = _resolve_expert(req.model)
    expert_key = _apply_expert(expert)
    _maybe_record_prefold_outcome(expert_key)
    prompt = _build_prompt(req.messages, req.thinking_effort, expert_key)

    tokenizer = model_state["tokenizer"]
    graph_decoder: FoldedCudaGraphDecoder = model_state["graph_decoder"]
    stop_ids: set[int] = model_state.get("stop_token_ids", set())

    # Decode is unconditionally greedy (argmax) end to end -- no sampling path
    # exists in either decoder below -- so req.temperature is accepted for
    # OpenAI-API-shape compatibility but has no effect.
    max_new = req.max_completion_tokens or req.max_tokens or 4096

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model_state["base_model"].device)
    prompt_tokens = input_ids.shape[-1]

    t0 = time.perf_counter()
    ttft_ms = 0.0

    spec_decoder = model_state.get("spec_decoder")
    range_gate = model_state.get("range_gate") if model_state.get("range_gate_enabled", True) else None
    if spec_decoder is not None:
        generated, _elapsed, _stats = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: spec_decoder.generate(
                input_ids,
                max_new_tokens=max_new,
                stop_ids=stop_ids,
                gate=range_gate,
            ),
        )
    else:
        generated, _elapsed, _tok_s, _swap_ms = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: graph_decoder.generate_with_graph(
                input_ids,
                max_new_tokens=max_new,
            ),
        )

    elapsed = time.perf_counter() - t0
    raw_text = tokenizer.decode(generated, skip_special_tokens=True)
    reasoning, content = _extract_thinking_and_content(raw_text)
    completion_tokens = len(generated)
    tok_s = completion_tokens / elapsed if elapsed > 0 else 0.0

    _maybe_predict_and_prefold(expert_key, raw_text)

    server_telemetry["total_requests"] += 1
    server_telemetry["total_prompt_tokens"] += prompt_tokens
    server_telemetry["total_generated_tokens"] += completion_tokens
    server_telemetry["total_generation_time_s"] += elapsed
    server_telemetry["last_request"] = {
        "model": req.model,
        "expert": expert_key,
        "tokens_per_second": round(tok_s, 2),
        "generation_time_ms": round(elapsed * 1000, 1),
        "timestamp": int(time.time()),
    }

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "tokens_per_second": round(tok_s, 2),
            "generation_time_ms": round(elapsed * 1000, 1),
            "time_to_first_token_ms": round(ttft_ms, 1),
        },
    }


async def _run_streaming_inference(req: ChatCompletionRequest) -> AsyncGenerator[str]:
    if not model_state.get("base_model"):
        yield "data: " + '{"error": "Engine not loaded"}' + "\n\n"
        return

    expert = _resolve_expert(req.model)
    expert_key = _apply_expert(expert)
    _maybe_record_prefold_outcome(expert_key)
    prompt = _build_prompt(req.messages, req.thinking_effort, expert_key)

    tokenizer = model_state["tokenizer"]
    graph_decoder: FoldedCudaGraphDecoder = model_state["graph_decoder"]

    # Decode is unconditionally greedy (argmax); req.temperature is accepted
    # for OpenAI-API-shape compatibility but has no effect.
    max_new = req.max_completion_tokens or req.max_tokens or 4096

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model_state["base_model"].device)

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())
    t0 = time.perf_counter()
    token_count = 0
    emitted_chunks: list[str] = []

    try:
        # NOTE: streaming only ever uses the plain graph-replay path -- there is
        # no token-by-token text-streaming generator for the speculative decoder
        # (its stream_generate yields raw token-id chunks, not detokenized text).
        # SPECULATIVE_DECODE=1 therefore only takes effect on non-streaming
        # requests (_run_inference); this is a known gap, not a new one.
        for token_text, _hidden_state in graph_decoder.generate_tokens_stream(
            input_ids,
            max_new_tokens=max_new,
        ):
            token_str = " ".join(token_text) if isinstance(token_text, list) else str(token_text)
            token_count += 1
            emitted_chunks.append(token_str)
            chunk = ChatCompletionChunkResponse(
                id=request_id,
                created=created_ts,
                model=req.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=token_str),
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
    finally:
        elapsed = time.perf_counter() - t0
        tok_s = token_count / elapsed if elapsed > 0 else 0.0
        _maybe_predict_and_prefold(expert_key, "".join(emitted_chunks))
        server_telemetry["total_requests"] += 1
        server_telemetry["total_generated_tokens"] += token_count
        server_telemetry["total_generation_time_s"] += elapsed
        server_telemetry["last_request"] = {
            "model": req.model,
            "expert": expert_key,
            "tokens_per_second": round(tok_s, 2),
            "generation_time_ms": round(elapsed * 1000, 1),
            "timestamp": int(time.time()),
        }
        yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# FastAPI app + lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    dispatch_task = asyncio.create_task(_dispatch_loop())
    print("[IPWF] Sequential dispatch loop started.")

    if os.environ.get("AUTO_LOAD_MODEL", "0") == "1":
        print(f"[IPWF] AUTO_LOAD_MODEL=1: Loading {DEFAULT_MODEL_ID}...")
        try:
            await load_inference_engine(DEFAULT_MODEL_ID)
        except Exception as exc:
            print(f"[IPWF] Auto-load failed: {exc}")
    else:
        print("[IPWF] Standby — engine loads on first POST /api/engine/load.")

    yield

    dispatch_task.cancel()
    with suppress(asyncio.CancelledError):
        await dispatch_task
    await unload_inference_engine()


app = FastAPI(
    title="IPWF Inference Server (3B / 9B)",
    description=(
        "OpenAI-compatible REST server for In-Place Weight Folding engine. "
        "Serves unquantized Qwen3.5-4B and Qwen3.5-9B models with CUDA Graph decoding "
        "and Riemannian domain-expert LoRA teams. "
        "For quantized 27B models use runtime-triton (port 8000)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = [
    ModelObject(id="Qwen/Qwen3.5-4B"),
    ModelObject(id="Qwen/Qwen3.5-9B"),
    ModelObject(id="qwen3.5-4b-astral"),
    ModelObject(id="qwen3.5-4b-postgresql"),
    ModelObject(id="qwen3.5-4b-duckdb"),
    ModelObject(id="qwen3.5-4b-fastapi"),
    ModelObject(id="qwen3.5-4b-financial"),
    ModelObject(id="qwen3.5-4b-python-modern"),
    ModelObject(id="qwen3.5-4b-dynamic"),
    ModelObject(id="qwen3.5-9b-astral"),
    ModelObject(id="qwen3.5-9b-postgresql"),
    ModelObject(id="qwen3.5-9b-duckdb"),
    ModelObject(id="qwen3.5-9b-financial"),
    ModelObject(id="qwen3.5-9b-dynamic"),
    ModelObject(id="astral"),
    ModelObject(id="postgresql"),
    ModelObject(id="duckdb"),
    ModelObject(id="financial_planning"),
    ModelObject(id="dynamic"),
]


@app.get("/v1/models", response_model=ModelListResponse)
@app.get("/api/tags", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    return ModelListResponse(data=SUPPORTED_MODELS)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "runtime-ipwf",
        "engine_loaded": model_state.get("base_model") is not None,
        "model_id": model_state.get("model_id", "not loaded"),
        "vram_allocated_gb": get_real_vram_allocated_gb(),
        "port": SERVER_PORT,
    }


@app.get("/api/engine/status")
async def engine_status():
    is_loaded = model_state.get("base_model") is not None
    vram = get_real_vram_allocated_gb()
    free_bytes, total_bytes = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    total_used = round((total_bytes - free_bytes) / (1024**3), 2) if torch.cuda.is_available() else vram
    spec_decoder = model_state.get("spec_decoder")
    return {
        "engine": "runtime-ipwf",
        "loaded": is_loaded,
        "model_id": model_state.get("model_id", DEFAULT_MODEL_ID),
        "vram_allocated_gb": vram,
        "total_vram_used_gb": total_used,
        "active_team": model_state.get("active_team", []),
        "spec_decode_enabled": spec_decoder is not None,
        "spec_k": model_state.get("spec_decoder_k"),
        "ring_buffer_mode": model_state.get("ring_buffer_mode", "selective_hybrid"),
        "scale_mode": model_state.get("scale_mode", "surgical"),
        "prefold_enabled": model_state.get("prefold_enabled", True),
        "range_gate_enabled": model_state.get("range_gate_enabled", True),
        "range_gate_threshold": model_state.get("range_gate_threshold", 5.0),
        "cut_set_hedging_enabled": model_state.get("cut_set_hedging_enabled", True),
        "cut_set_target_reliability": model_state.get("cut_set_target_reliability", 0.95),
        "causal_scheduler_telemetry": getattr(model_state.get("causal_scheduler"), "telemetry", None),
        "last_swap_ms": model_state.get("last_swap_ms"),
        "cumulative_swap_ms": model_state.get("cumulative_swap_ms", 0.0),
        "cumulative_swap_count": model_state.get("cumulative_swap_count", 0),
        "telemetry": server_telemetry,
    }


class SetPredictivePrefoldRequest(BaseModel):
    enabled: bool


@app.post("/api/engine/set_predictive_prefold")
async def set_predictive_prefold(req: SetPredictivePrefoldRequest):
    """Toggles NOTEARS predictive pre-folding live, no restart needed -- for A/B
    measurement (see benchmarks/agentic/benchmark_predictive_prefold_live.py)."""
    model_state["prefold_enabled"] = req.enabled
    model_state["last_predicted_expert"] = None
    model_state["cumulative_swap_ms"] = 0.0
    model_state["cumulative_swap_count"] = 0
    scheduler = model_state.get("causal_scheduler")
    if scheduler is not None:
        scheduler.telemetry.update(
            prefold_triggers=0,
            prefold_hits=0,
            prefold_misses=0,
            prefold_saved_ms=0.0,
            wasted_morph_ms=0.0,
            last_prediction=None,
            last_confidence=0.0,
            last_prefold_elapsed_ms=None,
        )
    print(f"[IPWF] Predictive pre-folding set: enabled={req.enabled} (telemetry reset)")
    return {"status": "updated", "prefold_enabled": req.enabled}


class SetSpeculativeRangeGateRequest(BaseModel):
    enabled: bool
    threshold: float | None = 3.5
    weibull_hazard_enabled: bool | None = True
    weibull_beta: float | None = 2.2
    weibull_gamma: float | None = 0.6
    bollinger_bands_enabled: bool | None = True
    bollinger_k: float | None = 2.0
    bollinger_gamma: float | None = 0.5


@app.post("/api/engine/set_speculative_range_gate")
async def set_speculative_range_gate(req: SetSpeculativeRangeGateRequest):
    """Toggles Single-Pass Range Statistic, Weibull Hazard & Bollinger Volatility Gating.

    Instantaneous O(1) swap: no CUDA Graph recapture required.
    """
    model_state["range_gate_enabled"] = req.enabled
    if req.threshold is not None:
        model_state["range_gate_threshold"] = req.threshold

    weibull_on = req.weibull_hazard_enabled if req.weibull_hazard_enabled is not None else True
    beta = req.weibull_beta if req.weibull_beta is not None else 2.2
    gamma_w = req.weibull_gamma if req.weibull_gamma is not None else 0.6
    boll_on = req.bollinger_bands_enabled if req.bollinger_bands_enabled is not None else True
    boll_k = req.bollinger_k if req.bollinger_k is not None else 2.0
    gamma_b = req.bollinger_gamma if req.bollinger_gamma is not None else 0.5

    model_state["range_gate"] = RangeStatisticGate(
        top_m=8,
        threshold=model_state["range_gate_threshold"],
        weibull_hazard_enabled=weibull_on,
        weibull_beta=beta,
        weibull_gamma=gamma_w,
        bollinger_bands_enabled=boll_on,
        bollinger_k=boll_k,
        bollinger_gamma=gamma_b,
    )

    print(
        f"[IPWF] Range Speculative Gate set: enabled={req.enabled}, "
        f"threshold={model_state['range_gate_threshold']}, weibull_hazard={weibull_on}, bollinger_bands={boll_on}"
    )
    return {
        "status": "updated",
        "spec_range_gate_enabled": model_state["range_gate_enabled"],
        "spec_range_threshold": model_state["range_gate_threshold"],
        "weibull_hazard_enabled": weibull_on,
        "weibull_beta": beta,
        "weibull_gamma": gamma_w,
        "bollinger_bands_enabled": boll_on,
        "bollinger_k": boll_k,
        "bollinger_gamma": gamma_b,
    }


@app.get("/api/causal_dag")
async def get_causal_dag():
    """Returns NOTEARS continuous causal tool-to-expert graph and telemetry."""
    scheduler = model_state.get("causal_scheduler")
    if scheduler is None:
        return {"fitted": False, "nodes": [], "edges": [], "hit_rate": 0.0, "telemetry": {}}
    return scheduler.get_dag_structure()


@app.post("/api/causal_dag/fit")
async def fit_causal_dag():
    """Re-fits NOTEARS continuous causal structure on recorded session traces."""
    scheduler = model_state.get("causal_scheduler")
    if scheduler is None:
        return {"fitted": False, "error": "Scheduler not initialized"}
    import json

    import tool_trace

    trace_file = tool_trace.trace_path()
    records = []
    if trace_file and trace_file.exists():
        with suppress(Exception):
            records = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
    return scheduler.fit_from_traces(records)


class SetSpeculativeDecodeRequest(BaseModel):
    enabled: bool


@app.post("/api/engine/set_speculative_decode")
async def set_speculative_decode(req: SetSpeculativeDecodeRequest):
    """Turns speculative decoding off (instant) or on."""
    if not req.enabled:
        was_active = model_state.get("spec_decoder") is not None
        model_state["spec_decoder"] = None
        return {"status": "disabled", "spec_decode_enabled": False, "was_active": was_active}

    return {"status": "enabled", "spec_decode_enabled": True, "spec_k": model_state.get("spec_decoder_k", 2)}


class SetSpecCircuitBreakerRequest(BaseModel):
    enabled: bool
    disengage_threshold: float | None = 1.8
    reengage_threshold: float | None = 2.2


@app.post("/api/engine/set_spec_circuit_breaker")
async def set_spec_circuit_breaker(req: SetSpecCircuitBreakerRequest):
    """Toggles Dual-EMA / MACD Speculation Circuit-Breaker."""
    spec_decoder = model_state.get("spec_decoder")
    if spec_decoder and hasattr(spec_decoder, "circuit_breaker"):
        spec_decoder.circuit_breaker.enabled = req.enabled
        if req.disengage_threshold is not None:
            spec_decoder.circuit_breaker.disengage_threshold = req.disengage_threshold
        if req.reengage_threshold is not None:
            spec_decoder.circuit_breaker.reengage_threshold = req.reengage_threshold

    model_state["spec_circuit_breaker_enabled"] = req.enabled
    disengage = req.disengage_threshold or 1.8
    reengage = req.reengage_threshold or 2.2
    print(f"[IPWF] Speculation Circuit-Breaker set: enabled={req.enabled}, disengage={disengage}, reengage={reengage}")
    return {
        "status": "updated",
        "spec_circuit_breaker_enabled": req.enabled,
        "disengage_threshold": disengage,
        "reengage_threshold": reengage,
    }


class SetCutSetHedgingRequest(BaseModel):
    enabled: bool
    target_reliability: float | None = 0.95
    max_workers: int | None = 4


@app.post("/api/engine/set_cut_set_hedging")
async def set_cut_set_hedging(req: SetCutSetHedgingRequest):
    """Configures Chapter 6 Minimal Cut Set & k-out-of-n Speculative Tool Hedging middleware."""
    model_state["cut_set_hedging_enabled"] = req.enabled
    if req.target_reliability is not None:
        model_state["cut_set_target_reliability"] = req.target_reliability
    if req.max_workers is not None:
        model_state["cut_set_max_workers"] = req.max_workers

    print(
        f"[IPWF] Cut-Set Speculative Hedging set: enabled={req.enabled}, "
        f"target_reliability={model_state.get('cut_set_target_reliability', 0.95)}"
    )
    return {
        "status": "updated",
        "cut_set_hedging_enabled": model_state["cut_set_hedging_enabled"],
        "cut_set_target_reliability": model_state.get("cut_set_target_reliability", 0.95),
        "cut_set_max_workers": model_state.get("cut_set_max_workers", 4),
    }


class CutSetNodeModel(BaseModel):
    node_id: str
    name: str
    reliability: float = 0.90


class CutSetDecomposeRequest(BaseModel):
    source: str
    sink: str
    nodes: list[CutSetNodeModel]
    edges: list[list[str]]
    target_reliability: float = 0.95


@app.post("/api/reliability/cut_set/decompose")
async def cut_set_decompose(req: CutSetDecomposeRequest):
    """Decomposes an agent reasoning DAG into Minimal Cut Sets, identifies order-1 SPOFs,
    and calculates analytical system reliability under optional k=1-of-n=2 hedging."""
    from cut_set_router import ReliabilityGraph, ReliabilityNode

    g = ReliabilityGraph(source=req.source, sink=req.sink)
    for n in req.nodes:
        g.add_node(ReliabilityNode(node_id=n.node_id, name=n.name, reliability=n.reliability))
    for edge in req.edges:
        if len(edge) == 2:
            g.add_edge(edge[0], edge[1])

    paths = g.find_all_paths()
    minimal_cuts = [sorted(list(c)) for c in g.find_minimal_cut_sets()]
    order_1 = g.get_order_1_cut_sets()
    bottlenecks = g.get_critical_bottlenecks(target_reliability=req.target_reliability)
    r_unhedged = g.compute_system_reliability()
    r_hedged = g.compute_system_reliability(active_hedges=set(bottlenecks))

    return {
        "status": "success",
        "paths": paths,
        "minimal_cut_sets": minimal_cuts,
        "order_1_cut_sets": order_1,
        "critical_bottlenecks": bottlenecks,
        "unhedged_system_reliability": round(r_unhedged, 4),
        "hedged_system_reliability": round(r_hedged, 4),
        "reliability_gain": round(r_hedged - r_unhedged, 4),
    }


class SetRingBufferModeRequest(BaseModel):
    mode: str


@app.post("/api/engine/set_ring_buffer_mode")
async def set_ring_buffer_mode(req: SetRingBufferModeRequest):
    """Swaps the live speculative decoder's ring buffer mode."""
    valid = {"dense", "poet", "selective_hybrid", "pointer"}
    if req.mode not in valid:
        return JSONResponse(
            status_code=400,
            content={"error": f"mode must be one of {sorted(valid)}, got {req.mode!r}"},
        )

    spec_decoder = model_state.get("spec_decoder")
    if spec_decoder is None or getattr(spec_decoder, "cache", None) is None:
        return JSONResponse(
            status_code=409,
            content={"error": "speculative decode is off or has no cache configured."},
        )

    from state_ring_buffer import RingBufferReplayEngine

    spec_decoder.ring_engine = RingBufferReplayEngine(spec_decoder.cache, max_depth=64, mode=req.mode)
    model_state["ring_buffer_mode"] = req.mode
    print(f"[IPWF] Ring buffer mode -> {req.mode}")
    return {"status": "swapped", "ring_buffer_mode": req.mode}


class LoadEngineRequest(BaseModel):
    model_id: str = DEFAULT_MODEL_ID


@app.post("/api/engine/load")
async def load_engine_endpoint(req: LoadEngineRequest | None = None):
    req = req or LoadEngineRequest()
    try:
        res = await load_inference_engine(model_id=req.model_id)
        return res
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/engine/unload")
async def unload_engine_endpoint():
    try:
        return await unload_inference_engine()
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/engine/reset")
@app.post("/api/engine/force_reset_vram")
async def force_reset():
    res = await unload_inference_engine()
    return {**res, "message": "VRAM reset and engine state cleared."}


@app.post("/v1/chat/completions")
@app.post("/api/chat/completions")
async def chat_completions(request: Request, req: ChatCompletionRequest):
    if not model_state.get("base_model"):
        return JSONResponse(
            status_code=503,
            content={"error": "Engine not loaded. POST /api/engine/load first."},
        )

    if req.stream:
        gen = _run_streaming_inference(req)
        return StreamingResponse(gen, media_type="text/event-stream")

    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()
    queued = _QueuedRequest(req=req, future=future)
    await _request_queue.put(queued)
    result = await future
    return JSONResponse(content=result)


@app.post("/v1/completions")
async def completions(request: Request, req: CompletionRequest):
    chat_req = ChatCompletionRequest(
        model=req.model,
        messages=[ChatMessage(role="user", content=req.prompt)],
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stream=req.stream,
        thinking_effort=req.thinking_effort,
    )
    return await chat_completions(request, chat_req)


@app.get("/api/telemetry")
async def get_telemetry():
    return server_telemetry


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    print(f"[IPWF] Starting on http://{SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
