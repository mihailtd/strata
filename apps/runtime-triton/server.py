"""Pure Native Triton 27B W4A16 Inference Server.

Serves a single fully-quantized 27B model (Qwen3.8:27B, GGUF W4A16) through
custom ROCm Triton GEMV kernels with in-memory LoRA swapping.

Exposes an OpenAI-compatible REST API on port 8000.

Engine characteristics:
  - AMD ROCm RX 7900 XTX (Navi 31), 24 GB GDDR6
  - 128-bit memory-coalesced GEMV, 620+ GB/s saturation
  - Fused SwiGLU in-register SiLU GEMV
  - Hybrid Qwen3.5 architecture: 48 Gated DeltaNet SSM + 16 Full Attention blocks
  - In-Register Mixture-of-Adapters (MoA) LoRA factor accumulation
  - Constant O(1) Gated DeltaNet recurrent state handoff
  - SINGLE-TENANT: requests execute strictly in arrival order, one at a time.

Supported model aliases (all route to the same W4A16 weights):
  qwen3.8:27b, qwen3.8:27b-triton, qwen3.8-27b, qwen3.8-27b-<domain>,
  qwen3.8-27b-auto, default, auto

Domain LoRA adapters (swapped in-memory at inference time):
  astral, postgresql, duckdb, fastapi / python_web, financial, python_modern

This server does NOT support unquantized (3B/9B) models.
For unquantized models, use runtime-ipwf (port 8002).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from runtime_common import gpu_preflight
from runtime_common.canon import configure_deterministic_attention
from native_27b_engine import Native27BEngine


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SERVER_PORT = int(os.environ.get("PORT", "8000"))
SERVER_HOST = os.environ.get("HOST", "0.0.0.0")
STREAM_ORDER_TIMEOUT_S = float(os.environ.get("STREAM_ORDER_TIMEOUT_S", "300"))

ADAPTER_MAP: dict[str, int] = {
    "astral": 0,
    "postgresql": 1,
    "duckdb": 2,
    "fastapi": 3,
    "python_web": 3,
    "financial": 4,
    "financial_planning": 4,
    "python_modern": 5,
}

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
# Global engine state
# ---------------------------------------------------------------------------

_engine: Native27BEngine | None = None
_tokenizer: Any = None

model_state: dict[str, Any] = {}

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
    """Reads live VRAM from amdgpu sysfs or PyTorch allocator."""
    try:
        sysfs = gpu_preflight.get_sysfs_vram_info()
        if sysfs and sysfs.get("sysfs_available") and "used_gb" in sysfs:
            return float(sysfs["used_gb"])
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return round((total_bytes - free_bytes) / (1024**3), 2)
    except Exception:
        pass
    return 0.0


def _evict_ollama_models() -> None:
    """Best-effort eviction of Ollama-resident models to free VRAM."""
    import urllib.request
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/ps")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            data = json.loads(resp.read())
            for m in data.get("models", []):
                name = m.get("name") or m.get("model")
                if name:
                    evict = urllib.request.Request(
                        "http://127.0.0.1:11434/api/generate",
                        data=json.dumps({"model": name, "keep_alive": 0}).encode(),
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(evict, timeout=1.0) as r:
                        r.read()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------

def _load_engine(num_layers: int = 64) -> Native27BEngine:
    """Loads the Native 27B Triton W4A16 engine into GPU VRAM."""
    global _engine, _tokenizer
    if _engine is not None:
        return _engine

    _evict_ollama_models()
    configure_deterministic_attention()

    print(f"[Triton-27B] Loading Native 27B Triton Engine ({num_layers} layers) into GPU...")
    engine = Native27BEngine(num_layers=num_layers)
    engine.load_from_cache()
    _engine = engine

    # Tokenizer: reuse Qwen3.5-9B snapshot (identical vocab)
    from transformers import AutoTokenizer
    snaps = list(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/*"))
    if snaps:
        _tokenizer = AutoTokenizer.from_pretrained(str(snaps[0]))
    else:
        _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

    vram = get_real_vram_allocated_gb()
    print(f"[Triton-27B] Engine loaded. VRAM: {vram:.2f} GB")
    model_state.update(
        loaded=True,
        model_id="qwen3.8:27b",
        vram_allocated_gb=vram,
        w4a16_enabled=True,
    )
    return _engine


def _unload_engine() -> None:
    """Unloads the engine and frees VRAM."""
    global _engine, _tokenizer
    if _engine is not None:
        print("[Triton-27B] Unloading engine to free VRAM...")
        _engine = None
        _tokenizer = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    model_state.clear()


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
    owned_by: str = "gnn-experiment-triton"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]


# ---------------------------------------------------------------------------
# Prompt helpers
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


def _classify_domain(messages: list[ChatMessage], model_name: str) -> str:
    m_lower = model_name.lower()
    for d in ADAPTER_MAP:
        if d in m_lower:
            return d
    texts = " ".join(
        _extract_content(m.content) for m in messages if m.role in ("user", "system")
    ).lower()
    if any(k in texts for k in ["uv", "ruff", "pyproject", "pip", "package", "packaging"]):
        return "astral"
    if any(k in texts for k in ["duckdb", "parquet", "olap", "arrow", "columnar"]):
        return "duckdb"
    if any(k in texts for k in ["postgres", "postgresql", "pgvector", "hnsw", "migration"]):
        return "postgresql"
    if any(k in texts for k in ["fastapi", "uvicorn", "endpoint", "sse", "pydantic"]):
        return "fastapi"
    if any(k in texts for k in ["portfolio", "wealth", "bond", "npv", "irr", "drawdown"]):
        return "financial"
    return "astral"


def _build_prompt(
    messages: list[ChatMessage],
    thinking_effort: str | None = "medium",
    domain: str | None = None,
) -> str:
    effort = (thinking_effort or "medium").lower()
    directives = {
        "off": "Respond directly and concisely. Do NOT produce any internal reasoning.",
        "low": "Keep thinking brief, under 3-4 sentences. Close with </think> before answering.",
        "high": "Think thoroughly through all edge cases. Close with </think> before answering.",
        "medium": "Keep thinking structured and focused. Close with </think> before answering.",
    }
    thinking_directive = directives.get(effort, directives["medium"])
    specialist = DOMAIN_SYSTEM_DIRECTIVES.get(domain or "", "")
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
# Request queue (single-tenant, arrival-order dispatch)
# ---------------------------------------------------------------------------

@dataclass
class _QueuedRequest:
    req: Any
    future: asyncio.Future
    enqueue_time: float = field(default_factory=time.perf_counter)
    stream_done: asyncio.Event = field(default_factory=asyncio.Event)


_request_queue: asyncio.Queue[_QueuedRequest] = asyncio.Queue()


async def _dispatch_loop() -> None:
    """Processes requests strictly in arrival order — one at a time."""
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

async def _run_inference(req: ChatCompletionRequest) -> dict[str, Any]:
    engine = _load_engine()
    domain = _classify_domain(req.messages, req.model)
    adapter_id = ADAPTER_MAP.get(domain)
    prompt = _build_prompt(req.messages, req.thinking_effort, domain)
    prompt_tokens = len(prompt.split())
    max_new = req.max_completion_tokens or req.max_tokens or 4096
    temperature = req.temperature if req.temperature is not None else 0.7

    t0 = time.perf_counter()
    raw_text = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: engine.generate(
            prompt=prompt,
            max_new_tokens=max_new,
            temperature=temperature,
            adapter_id=adapter_id,
        ),
    )
    elapsed = time.perf_counter() - t0

    reasoning, content = _extract_thinking_and_content(raw_text)
    completion_tokens = len(raw_text.split())
    tok_s = completion_tokens / elapsed if elapsed > 0 else 0.0

    server_telemetry["total_requests"] += 1
    server_telemetry["total_prompt_tokens"] += prompt_tokens
    server_telemetry["total_generated_tokens"] += completion_tokens
    server_telemetry["total_generation_time_s"] += elapsed
    server_telemetry["last_request"] = {
        "model": req.model,
        "domain": domain,
        "adapter_id": adapter_id,
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
        },
    }


async def _run_streaming_inference(
    req: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    engine = _load_engine()
    domain = _classify_domain(req.messages, req.model)
    adapter_id = ADAPTER_MAP.get(domain)
    prompt = _build_prompt(req.messages, req.thinking_effort, domain)
    max_new = req.max_completion_tokens or req.max_tokens or 4096
    temperature = req.temperature if req.temperature is not None else 0.7

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created_ts = int(time.time())
    t0 = time.perf_counter()
    token_count = 0

    try:
        for token_text in engine.stream_generate(
            prompt=prompt,
            max_new_tokens=max_new,
            temperature=temperature,
            adapter_id=adapter_id,
        ):
            token_count += 1
            chunk = ChatCompletionChunkResponse(
                id=request_id,
                created=created_ts,
                model=req.model,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=ChatCompletionChunkDelta(content=token_text),
                    )
                ],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
    finally:
        elapsed = time.perf_counter() - t0
        tok_s = token_count / elapsed if elapsed > 0 else 0.0
        server_telemetry["total_requests"] += 1
        server_telemetry["total_generated_tokens"] += token_count
        server_telemetry["total_generation_time_s"] += elapsed
        server_telemetry["last_request"] = {
            "model": req.model,
            "domain": domain,
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
    print("[Triton-27B] Sequential dispatch loop started.")
    if os.environ.get("AUTO_LOAD_MODEL", "0") == "1":
        try:
            _load_engine()
        except Exception as exc:
            print(f"[Triton-27B] Auto-load failed: {exc}")
    else:
        print("[Triton-27B] Standby — engine loads on first request.")
    yield
    dispatch_task.cancel()
    try:
        await dispatch_task
    except asyncio.CancelledError:
        pass
    _unload_engine()


app = FastAPI(
    title="Triton 27B W4A16 Inference Server",
    description=(
        "OpenAI-compatible REST server for the Native Triton 27B W4A16 engine "
        "(ROCm Navi 31 / RX 7900 XTX). Quantized 27B model only. "
        "For unquantized 3B/9B models use runtime-ipwf (port 8002)."
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
    ModelObject(id="qwen3.8:27b"),
    ModelObject(id="qwen3.8:27b-triton"),
    ModelObject(id="qwen3.8-27b-astral"),
    ModelObject(id="qwen3.8-27b-postgresql"),
    ModelObject(id="qwen3.8-27b-duckdb"),
    ModelObject(id="qwen3.8-27b-fastapi"),
    ModelObject(id="qwen3.8-27b-financial"),
    ModelObject(id="qwen3.8-27b-python-modern"),
    ModelObject(id="qwen3.8-27b-auto"),
]


@app.get("/v1/models", response_model=ModelListResponse)
@app.get("/api/tags", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    return ModelListResponse(data=SUPPORTED_MODELS)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "runtime-triton",
        "engine_loaded": _engine is not None,
        "vram_allocated_gb": get_real_vram_allocated_gb(),
        "port": SERVER_PORT,
    }


@app.get("/api/engine/status")
async def engine_status():
    vram = get_real_vram_allocated_gb()
    free_bytes, total_bytes = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    total_used = round((total_bytes - free_bytes) / (1024**3), 2) if torch.cuda.is_available() else vram
    return {
        "engine": "runtime-triton",
        "loaded": _engine is not None,
        "model_id": model_state.get("model_id", "qwen3.8:27b"),
        "vram_allocated_gb": vram,
        "total_vram_used_gb": total_used,
        "w4a16_enabled": True,
        "telemetry": server_telemetry,
    }


class LoadEngineRequest(BaseModel):
    num_layers: int = 64


@app.post("/api/engine/load")
async def load_engine_endpoint(req: LoadEngineRequest = Body(default_factory=LoadEngineRequest)):
    try:
        _load_engine(num_layers=req.num_layers)
        return {
            "status": "loaded",
            "model_id": "qwen3.8:27b",
            "vram_allocated_gb": get_real_vram_allocated_gb(),
            "w4a16_enabled": True,
        }
    except Exception as exc:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/engine/unload")
async def unload_engine_endpoint():
    _unload_engine()
    return {"status": "unloaded", "vram_allocated_gb": get_real_vram_allocated_gb()}


@app.post("/v1/chat/completions")
@app.post("/api/chat/completions")
async def chat_completions(request: Request, req: ChatCompletionRequest):
    if _engine is None:
        _load_engine()

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
    print(f"[Triton-27B] Starting on http://{SERVER_HOST}:{SERVER_PORT}")
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="info")
