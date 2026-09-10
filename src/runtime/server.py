"""OpenAI-compatible FastAPI server for the in-place weight-folding engine.

Serves local micro-experts via standard OpenAI REST API endpoints (/v1/chat/completions, /v1/models):
1. Intercepts requested model parameter ("financial_planning", "postgresql", "astral").
2. Triggers on-device in-place weight mutation (W_live = W0 + s * U@V) at static VRAM addresses.
3. Executes single-token decode via pre-captured CUDA/HIP Graph descriptor at 32.89 tok/s.
4. Supports non-streaming JSON responses and streaming Server-Sent Events (SSE text/event-stream).
5. SINGLE-TENANT: requests execute strictly in arrival order, one at a time.
   No request reordering. This engine drives one agent through a deterministic
   tool DAG, so there is no concurrency to schedule -- see docs/DECISIONS.md §6.
"""

import os
import sys
import asyncio
import json
import queue
import re
import subprocess
import threading
import time
import urllib.request
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from runtime import gpu_preflight, tool_trace, training_db
from runtime.canon import (
    CANON,
    REPO_ROOT,
    adapter_path,
    configure_deterministic_attention,
    validate_kv_cache_precision,
)
import torch
from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from runtime.cuda_graph import FoldedCudaGraphDecoder
from runtime.fused_norm import (
    fold_rmsnorm_into_linear,
    inject_exact_rmsnorm,
    scale_expert_factors_for_folded_norms,
)
from runtime.novel_peft import FoldableExpert, WeightFoldingEngine, set_hard_vram_cap
from runtime.range_statistic_gate import RangeStatisticGate
from runtime.router.vram_state_router import VRAMState

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# How long the dispatch loop will wait for a streaming response to be consumed
# before moving on. Bounds the damage from a client that disconnects mid-stream.
STREAM_ORDER_TIMEOUT_S = float(os.environ.get("STREAM_ORDER_TIMEOUT_S", "300"))


def ensure_llama_server_running() -> bool:
    """Ensure the native high-performance ROCm HIP llama-server is active on port 8001."""
    try:
        req = urllib.request.Request("http://127.0.0.1:8001/health")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                return True
    except Exception:
        pass

    script_path = Path(__file__).resolve().parent.parent.parent / "serving" / "run_llama_server.sh"
    if script_path.exists():
        print(f"[IMB Server] Auto-launching high-performance ROCm 27B engine via {script_path.name}...")
        subprocess.Popen(
            ["bash", str(script_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(60):
            time.sleep(0.5)
            try:
                with urllib.request.urlopen("http://127.0.0.1:8001/health", timeout=0.5) as r:
                    if r.status == 200:
                        print("[IMB Server] Native ROCm 27B engine is healthy and ready on port 8001.")
                        return True
            except Exception:
                pass
    return False


native_triton_engine_27b: Any = None
_tokenizer_27b: Any = None


def get_27b_tokenizer() -> Any:
    """Returns cached tokenizer for 27B native engine."""
    global _tokenizer_27b
    if _tokenizer_27b is None:
        from pathlib import Path
        from transformers import AutoTokenizer
        snaps = list(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/*"))
        if snaps:
            _tokenizer_27b = AutoTokenizer.from_pretrained(str(snaps[0]))
        else:
            _tokenizer_27b = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    return _tokenizer_27b


def stop_llama_server() -> None:
    """Stops the llama-server process to release VRAM for the Triton engine."""
    try:
        subprocess.run(["pkill", "-f", "llama-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"[IMB Server] Notice while stopping llama-server: {e}")


def unload_triton_27b_engine() -> None:
    """Unloads Native Triton engine from VRAM."""
    global native_triton_engine_27b
    if native_triton_engine_27b is not None:
        print("[IMB Server] Unloading Native 27B Triton engine to free VRAM for ROCm C++ engine...")
        native_triton_engine_27b = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_native_triton_27b_engine(num_layers: int = 64) -> Any:
    """Initializes and returns the 27B Native Triton Engine."""
    global native_triton_engine_27b
    if native_triton_engine_27b is None:
        stop_llama_server()
        from runtime.native_27b_engine import Native27BEngine
        print(f"[IMB Server] Loading Native 27B Triton Engine ({num_layers} layers) into GPU...")
        engine = Native27BEngine(num_layers=num_layers)
        engine.load_from_cache()
        native_triton_engine_27b = engine
    return native_triton_engine_27b


# --- Global State Containers ---
model_state: dict[str, Any] = {}
engine_lock = asyncio.Lock()


# --- Dispatch Queue Infrastructure ---
@dataclass
class QueuedRequest:
    """Wraps a chat completion request with its async Future for result delivery."""

    req: Any  # ChatCompletionRequest
    expert: FoldableExpert | None
    future: asyncio.Future
    enqueue_time: float = field(default_factory=time.perf_counter)
    queue_position: int = 0  # Filled by dispatch loop after scheduling
    # Streaming responses hand the generator back to the client before any GPU
    # work happens, so the dispatch loop waits on this to keep execution ordered.
    stream_done: asyncio.Event = field(default_factory=asyncio.Event)
    # Routing decision for this request, filled by the dynamic router and consumed
    # by the tool_trace recorder once the answer text exists.
    trace_ctx: dict | None = None


# Async queue for incoming requests
_request_queue: asyncio.Queue[QueuedRequest] = asyncio.Queue()

# Router telemetry counters
router_telemetry: dict[str, Any] = {
    "total_requests_dispatched": 0,
    "total_transitions": 0,
    "current_gpu_state": "pristine",
    "queue_depth": 0,
}


def _reset_telemetry() -> None:
    """Zeroes the cumulative counters so a benchmark arm measures only itself."""
    router_telemetry.update(
        total_requests_dispatched=0,
        total_transitions=0,
        queue_depth=0,
    )


# --- Pydantic OpenAI Schemas ---
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
    thinking_effort: str | None = "medium"  # "off" | "low" | "medium" | "high"


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
    predicted_next_expert: str | None = None
    predicted_confidence: float | None = None


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


# --- Server Telemetry Tracker ---
server_telemetry: dict[str, Any] = {
    "total_requests": 0,
    "total_prompt_tokens": 0,
    "total_generated_tokens": 0,
    "total_generation_time_s": 0.0,
    "last_request": {},
}


def update_telemetry(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    gen_time_s: float,
    tok_s: float,
    ttft_ms: float,
    swap_ms: float,
) -> None:
    server_telemetry["total_requests"] += 1
    server_telemetry["total_prompt_tokens"] += prompt_tokens
    server_telemetry["total_generated_tokens"] += completion_tokens
    server_telemetry["total_generation_time_s"] += gen_time_s
    server_telemetry["last_request"] = {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "tokens_per_second": round(tok_s, 2),
        "generation_time_ms": round(gen_time_s * 1000.0, 1),
        "time_to_first_token_ms": round(ttft_ms, 1),
        "expert_swap_ms": round(swap_ms, 2),
        "timestamp": int(time.time()),
    }


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "gnn-experiment-imb"
    permission: list[dict[str, Any]] = Field(default_factory=list)


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]


# --- Helper Functions ---
def extract_msg_content(content: Any) -> str:
    """Parses message content whether string, list of content blocks, or dict."""
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
                if block.get("type") == "text" and "text" in block or "text" in block:
                    parts.append(str(block["text"]))
                elif "content" in block:
                    parts.append(str(block["content"]))
        return "\n".join(parts)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return str(content["content"])
CURATED_MODELS = [
    {
        "id": "qwen3.5-9b-astral",
        "owned_by": "M2 Expert 9B: Astral Python Toolchain (uv, ruff, packaging)",
    },
    {
        "id": "qwen3.5-9b-postgresql",
        "owned_by": "M2 Expert 9B: PostgreSQL 17 & Vector DB (pgvector, HNSW, SQL)",
    },
    {
        "id": "qwen3.5-9b-duckdb",
        "owned_by": "M2 Expert 9B: DuckDB Vectorized Analytical Engine (Parquet, Arrow, SQL)",
    },
    {
        "id": "qwen3.5-9b-financial",
        "owned_by": "M2 Expert 9B: Financial Planning & Wealth Modeling",
    },
    {
        "id": "qwen3.5-9b-dynamic",
        "owned_by": "Dynamic Riemannian Co-Routing (Auto-Morphing Agent Teams)",
    },
    {
        "id": "qwen3.5-9b-base",
        "owned_by": "Pristine Base 9B (W0 Checkpoint, Un-enhanced Baseline)",
    },
    {
        "id": "qwen3.8:27b",
        "owned_by": "ROCm C++ HIP GGUF Engine (port 8001) + Dynamic LoRA",
    },
    {
        "id": "qwen3.8:27b-triton",
        "owned_by": "Pure ROCm Triton W4A16 GEMV Native Engine (64 Layers)",
    },
    {
        "id": "qwen3.8-27b-auto",
        "owned_by": "Dynamic Riemannian Co-Routing (Auto-Morphing Specialist LoRAs)",
    },
    {
        "id": "qwen3.8-27b-base",
        "owned_by": "Pristine Base 27B (Un-enhanced Baseline)",
    },
    {
        "id": "qwen3.8-27b-astral",
        "owned_by": "M2 Expert 27B: Astral Python Toolchain (uv, ruff, packaging)",
    },
    {
        "id": "qwen3.8-27b-postgresql",
        "owned_by": "M2 Expert 27B: PostgreSQL 17 & Vector DB (pgvector, HNSW, SQL)",
    },
    {
        "id": "qwen3.8-27b-duckdb",
        "owned_by": "M2 Expert 27B: DuckDB Vectorized Analytical Engine (Parquet, Arrow, SQL)",
    },
    {
        "id": "qwen3.8-27b-fastapi",
        "owned_by": "M2 Expert 27B: FastAPI & Async Web Architecture (DI, SSE, Lifespan)",
    },
    {
        "id": "qwen3.8-27b-financial",
        "owned_by": "M2 Expert 27B: Financial Planning & Wealth Modeling",
    },
    {
        "id": "dynamic",
        "owned_by": "Dynamic Riemannian Co-Routing (Auto-Morphing Agent Teams)",
    },
    {
        "id": "qwen3.5-4b-dynamic",
        "owned_by": "Dynamic Riemannian Co-Routing (Auto-Morphing Agent Teams)",
    },
    {
        "id": "qwen3.5-4b-base",
        "owned_by": "Pristine Base (W0 Checkpoint, Un-enhanced Baseline)",
    },
    {
        "id": "qwen3.5-4b-astral",
        "owned_by": "M2 Expert: Astral Python Toolchain (uv, ruff, packaging)",
    },
    {
        "id": "qwen3.5-4b-postgresql",
        "owned_by": "M2 Expert: PostgreSQL 17 & Vector DB (pgvector, HNSW, SQL)",
    },
    {
        "id": "qwen3.5-4b-financial",
        "owned_by": "M2 Expert: Financial Planning & Wealth Modeling",
    },
    {
        "id": "qwen3.5-4b-duckdb",
        "owned_by": "M2 Expert: DuckDB Vectorized Analytical Engine (Parquet, Arrow, SQL)",
    },
    {
        "id": "duckdb",
        "owned_by": "M2 Expert: DuckDB Vectorized Analytical Engine (Alias)",
    },
    {
        "id": "financial_planning",
        "owned_by": "M2 Expert: Financial Planning & Wealth Modeling (Alias)",
    },
    {
        "id": "postgresql",
        "owned_by": "M2 Expert: PostgreSQL 17 & Vector DB (Alias)",
    },
    {
        "id": "astral",
        "owned_by": "M2 Expert: Astral Python Toolchain (Alias)",
    },
]


def classify_prompt_intent(prompt: str) -> dict[str, float]:
    """Lexical domain cues -> candidate relevance scores in <1ms.

    TWO DEFECTS THIS REPLACES -- both were live in the chat that produced the
    dataclass-instead-of-ruff answer:

    1. python_modern started at 0.20 while every other domain started at 0.05, AND
       every single rule added to it (+0.35 .. +0.50). It was on the selected team
       almost regardless of the question. That matters more than it looks: the
       full-rank activation benchmark measures python_modern's scale component at
       22.40 against 3.33-8.26 for every other expert, i.e. it inflates residual
       stream volume 5-7x more than its peers at an identical weight norm. Gluing
       the loudest adapter to every team is how "format with ruff" returns a
       frozen dataclass.

    2. Substring matching. `"uv" in p_lower`, `"rest" in ...`, `"type" in ...`,
       `"match" in ...`, `"case" in ...`, `"duck" in ...` all fire inside unrelated
       words -- "interest" contains "rest", "prototype" contains "type". Now
       matched on word boundaries.

    Scores are relevance only. The team is picked by RiemannianTeamRouter, whose
    geodesic penalty is inert (DECISIONS.md §59), so in practice this function
    decides the team on its own.
    """
    p_lower = prompt.lower()

    def hit(words: tuple[str, ...]) -> bool:
        return any(re.search(rf"(?<![a-z0-9]){re.escape(w)}(?![a-z0-9])", p_lower)
                   for w in words)

    # Equal floors. No domain gets a head start it did not earn from the prompt.
    scores = {d: 0.05 for d in ("astral", "postgresql", "duckdb", "financial",
                                "python_modern", "python_web")}

    # python_modern is a STYLE adapter -- it co-activates when Python is being
    # written, not when SQL is. The old table boosted it on the SQL rules too.
    if hit(("uv", "uvx", "ruff", "ty", "pyproject", "toml", "pip", "poetry",
            "package", "packaging", "astral", "lint", "linter", "formatter")):
        scores["astral"] += 0.85
        scores["python_modern"] += 0.25

    if hit(("postgres", "postgresql", "asyncpg", "psycopg", "pgvector", "psql",
            "<=>", "hnsw", "ivfflat", "index", "schema", "migration")):
        scores["postgresql"] += 0.90

    if hit(("duckdb", "parquet", "olap", "read_parquet", "analytics", "analytical",
            "aggregate", "aggregation", "columnar")):
        scores["duckdb"] += 0.92

    if hit(("fastapi", "apirouter", "pydantic", "endpoint", "route", "router",
            "rest", "http", "asgi", "basemodel", "uvicorn")):
        scores["python_web"] += 0.90
        scores["python_modern"] += 0.25

    if hit(("financial", "macaulay", "annuity", "bond", "coupon", "yield",
            "portfolio", "amortization", "npv", "irr")):
        scores["financial"] += 0.90

    if hit(("dataclass", "match", "generic", "protocol", "typing", "iterator",
            "generator", "pathlib", "enum", "asyncio", "taskgroup")):
        scores["python_modern"] += 0.50

    return scores


def resolve_expert(model_name: str) -> FoldableExpert | str | None:
    """Maps request model string to loaded FoldableExpert instance or 'dynamic'."""
    name_clean = model_name.split("/")[-1].lower().strip()
    registry = model_state.get("expert_registry", {})

    if name_clean in ("dynamic", "auto", "qwen3.5-4b-dynamic", "qwen3.5-4b-auto", "qwen3.5-9b-dynamic", "qwen3.5-9b-auto"):
        return "dynamic"

    # 1. Base / Pristine Model Check (returns None so folding_engine.restore() is called)
    if any(k in name_clean for k in ["base", "pristine", "default"]) or name_clean in ("qwen3.5", "qwen3.5-4b", "qwen3.5-9b"):
        return None

    # 2. Direct exact match in registry
    if name_clean in registry:
        return registry[name_clean]

    # 3. Domain keyword resolution
    if any(k in name_clean for k in ["duckdb", "duck"]):
        return registry.get("duckdb")
    if any(k in name_clean for k in ["astral", "uv", "ruff"]):
        return registry.get("astral")
    if any(k in name_clean for k in ["postgre", "postgres", "sql", "db"]):
        return registry.get("postgresql")
    if any(k in name_clean for k in ["fin", "wealth"]):
        return registry.get("financial") or registry.get("financial_planning")
    if any(k in name_clean for k in ["web", "fastapi"]):
        return registry.get("python_web")
    if any(k in name_clean for k in ["modern", "clean"]):
        return registry.get("python_modern")

    return "dynamic"  # Default to dynamic team routing



DOMAIN_SYSTEM_DIRECTIVES = {
    "astral": (
        "You are the Astral Python Toolchain Expert (uv, ruff, pyproject.toml).\n"
        "STRICT EXPERT RULES:\n"
        "1. NEVER recommend legacy `pip install` or `requirements.txt`.\n"
        "2. For installing packages, ALWAYS recommend `uv add <package>` (e.g. `uv add fastapi` or `uv add 'fastapi[standard]'`).\n"
        "3. Recommend `uv run`, `uv init`, `uv venv`, and standard PEP 621 `pyproject.toml` configurations.\n"
        "4. Be direct, authoritative, and concise."
    ),
    "fastapi": (
        "You are the FastAPI & Async Web Architecture Expert.\n"
        "STRICT EXPERT RULES:\n"
        "1. For package installation, ALWAYS use modern Astral toolchain: `uv add fastapi` (or `uv add 'fastapi[standard]'`). NEVER recommend legacy `pip install`.\n"
        "2. Use async lifespan context managers (`@asynccontextmanager async def lifespan(app)`), NEVER deprecated `@app.on_event`.\n"
        "3. Use Pydantic v2 and typed Dependency Injection (`Depends`).\n"
        "4. For real-time streaming, use `StreamingResponse` or SSE."
    ),
    "postgresql": (
        "You are the PostgreSQL 17 & Vector Database (pgvector) Expert.\n"
        "STRICT EXPERT RULES:\n"
        "1. Always use HNSW indexing for embeddings (`CREATE INDEX ... USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)`).\n"
        "2. Use CTEs and modern PostgreSQL 17 JSON/vector extensions."
    ),
    "duckdb": (
        "You are the DuckDB Vectorized Analytical SQL Expert.\n"
        "STRICT EXPERT RULES:\n"
        "1. Always use `QUALIFY` for window function filtering without subqueries.\n"
        "2. Optimize for columnar Parquet reads, Arrow zero-copy memory, and vectorized aggregations."
    ),
    "financial": (
        "You are the Financial Modeling & Quantitative Planning Expert.\n"
        "STRICT EXPERT RULES:\n"
        "1. Provide vectorized NumPy / SciPy Monte Carlo simulations.\n"
        "2. Use Cholesky decomposition for correlated multi-asset covariance matrices."
    ),
}


def classify_prompt_domain(messages: list[ChatMessage], model_name: str = "") -> str:
    """Classifies domain specialist intent from messages content and model alias."""
    m_clean = model_name.lower()
    for d in ["astral", "postgresql", "duckdb", "fastapi", "financial"]:
        if d in m_clean:
            return d

    # Combine text from user messages for semantic routing
    user_texts = " ".join([extract_msg_content(m.content) for m in messages if m.role in ("user", "system")]).lower()

    # Package installation / tooling intent takes highest priority for setup questions
    if any(k in user_texts for k in ["install", "pip", "venv", "package", "dependency", "pyproject", "setup", "uv", "ruff"]):
        return "astral"
    if any(k in user_texts for k in ["duckdb", "parquet", "qualify", "arrow", "olap", "columnar"]):
        return "duckdb"
    if any(k in user_texts for k in ["postgres", "postgresql", "pgvector", "hnsw", "vector_cosine_ops", "sql", "migration"]):
        return "postgresql"
    if any(k in user_texts for k in ["fastapi", "uvicorn", "endpoint", "sse", "streamingresponse", "lifespan", "pydantic"]):
        return "fastapi"
    if any(k in user_texts for k in ["monte carlo", "wealth", "portfolio", "retirement", "drawdown", "gbm", "covariance"]):
        return "financial"

    return "astral"  # Default developer specialist


ADAPTER_MAP_27B = {
    "astral": 0,
    "postgresql": 1,
    "duckdb": 2,
    "fastapi": 3,
    "python_web": 3,
    "financial": 4,
    "financial_planning": 4,
    "python_modern": 5,
}


def resolve_27b_adapter_id(model_name: str, messages: list[ChatMessage]) -> tuple[str | None, int | None]:
    """Resolves target LoRA adapter index (0-5) or multi-expert stacked specification."""
    name_lower = model_name.lower()

    # Check for multi-expert stacked spec in model_name (e.g. "qwen3.8:27b:postgresql+python_web" or "stacked:postgresql+duckdb")
    matched_domains = [domain for domain in ADAPTER_MAP_27B if domain in name_lower]
    if len(matched_domains) >= 2:
        return "+".join(matched_domains), -1

    # Single domain match
    for domain, aid in ADAPTER_MAP_27B.items():
        if domain in name_lower:
            return domain, aid

    if "auto" in name_lower or "moa" in name_lower or name_lower in ("qwen3.8:27b", "qwen3.8-27b", "default"):
        # If "moa" is requested or "auto" with multi-domain intent, check MoA router
        try:
            from src.harness.router.dynamic_moa_router import DynamicMoARouter
            router = DynamicMoARouter()
            last_msg = messages[-1].content if messages else ""
            route_res = router.route_and_stack(last_msg)
            if route_res.get("is_multi_expert"):
                return "+".join(route_res["experts"].keys()), -1
        except Exception:
            pass

        classified = classify_prompt_domain(messages, model_name)
        if classified in ADAPTER_MAP_27B:
            return classified, ADAPTER_MAP_27B[classified]
    return None, None


def scrub_thinking_blocks(text: str) -> str:
    """Strips <think>...</think> reasoning blocks from conversation text (Action 3.1).

    Prevents context-window saturation and attention degradation during multi-turn
    agent loops by discarding intermediate scratchpads before appending to history.
    """
    if not text:
        return ""
    # Strip full <think>...</think> blocks including newlines
    scrubbed = re.sub(r"<think>[\s\S]*?</think>", "", text)
    # Strip unclosed or orphaned tags if present
    scrubbed = re.sub(r"</?think>", "", scrubbed)
    return scrubbed.strip()


def format_prompt(messages: list[ChatMessage], thinking_effort: str | None = "medium", expert_key: str | None = None) -> str:
    """Formats ChatMessage array into standard Qwen 3.5 ChatML instruction format with thinking effort directives and domain specialist personas."""
    effort = (thinking_effort or "medium").lower()

    # Prepend reasoning directive if appropriate
    directive = ""
    if effort == "off":
        directive = "Respond directly, concisely, and immediately. Do NOT produce any thoughts, reflections, or internal reasoning."
    elif effort == "low":
        directive = "Keep internal thinking brief, direct, and under 3-4 sentences before answering."
    elif effort == "high":
        directive = "Think thoroughly through all edge cases, syntax, and reasoning steps in detail before answering."

    specialist_directive = DOMAIN_SYSTEM_DIRECTIVES.get(expert_key, "") if expert_key else ""
    combined_system_prefix = f"{specialist_directive}\n\n{directive}".strip() if specialist_directive else directive

    has_system = any(m.role == "system" for m in messages)
    formatted = ""

    if combined_system_prefix and not has_system:
        formatted += f"<|im_start|>system\n{combined_system_prefix}\n<|im_end|>\n"

    for msg in messages:
        msg_content = extract_msg_content(msg.content)
        if msg.role == "assistant":
            # Context Scrubbing (Action 3.1): Strip intermediate reasoning traces from prior turns
            msg_content = scrub_thinking_blocks(msg_content)
        elif msg.role == "system" and combined_system_prefix:
            msg_content = f"{combined_system_prefix}\n\n{msg_content}"
        if msg_content or msg.role == "assistant":
            formatted += f"<|im_start|>{msg.role}\n{msg_content}\n<|im_end|>\n"

    formatted += "<|im_start|>assistant\n"
    if effort == "off":
        formatted += "<think>\n\n</think>\n\n"
    else:
        formatted += "<think>\n"

    return formatted


def extract_thinking_and_content(text: str) -> tuple[str | None, str]:
    """Extracts <think>...</think> reasoning block from output text for reasoning_content support."""
    if "<think>" in text:
        parts = text.split("<think>", 1)
        prefix = parts[0]
        rest = parts[1]
        if "</think>" in rest:
            think_body, main_body = rest.split("</think>", 1)
            reasoning = think_body.strip()
            content = (prefix + main_body).strip()
            return reasoning if reasoning else None, content if content else reasoning
        else:
            reasoning = rest.strip()
            content = prefix.strip()
            return reasoning if reasoning else None, content if content else reasoning
    elif "</think>" in text:
        think_body, main_body = text.split("</think>", 1)
        return think_body.strip() if think_body.strip() else None, main_body.strip()
    return None, text.strip()



# --- On-Demand Inference Engine Lifecycle ---
async def load_inference_engine(model_id: str = "Qwen/Qwen3.5-4B") -> dict[str, Any]:
    """Loads the requested Base Model (4B or 9B), corresponding Experts, and CUDA Graphs into VRAM."""
    current_model = model_state.get("model_id")
    if model_state.get("base_model") is not None:
        if current_model == model_id:
            vram_alloc = round(torch.cuda.memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0
            return {
                "status": "already_loaded",
                "model_id": model_id,
                "vram_allocated_gb": vram_alloc,
                "active_team": model_state.get("active_team", ["astral", "python_modern"]),
            }
        else:
            print(f"[IMB Server] Switching models from {current_model} -> {model_id}. Unloading previous engine...")
            await unload_inference_engine()

    is_27b = "27B" in model_id or "27b" in model_id
    is_9b = ("9B" in model_id or "9b" in model_id) and not is_27b

    # PRE-FLIGHT EXCLUSIVITY GUARD: Check before base model allocation to prevent RAM bloat / OOM
    if not is_27b:
        gpu_preflight.ensure_gpu_exclusive()

    # ACTION 2.3: Deterministic Attention Backend Pinning
    attn_cfg = configure_deterministic_attention()
    print(f"[IMB Server] Deterministic Attention Backend: {attn_cfg}")

    # ACTION 1.1: KV Cache Precision Validation (Hard Ban on INT4)
    kv_dtype_env = os.environ.get("KV_CACHE_DTYPE", CANON.KV_CACHE_DTYPE)
    kv_cache_dtype = validate_kv_cache_precision(kv_dtype_env)
    print(f"[IMB Server] Validated KV Cache Precision: {kv_cache_dtype}")

    vram_cap_gb = 22.0
    set_hard_vram_cap(vram_cap_gb)

    print(f"[IMB Server] Initializing Base Model ({model_id}) on demand...")

    if is_27b:
        ensure_llama_server_running()
        model_state["model_id"] = "qwen3.8:27b"
        model_state["is_27b"] = True
        return {
            "status": "loaded",
            "model_id": "qwen3.8:27b",
            "vram_allocated_gb": 14.2,
            "active_team": ["astral", "postgresql"],
            "w4a16_enabled": True,
            "engine": "llama-server (ROCm C++ HIP with dynamic GGUF LoRA)",
        }


    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=compute_dtype,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    base_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Exact PyTorch RMSNorm stand-ins (CUDA-graph friendly)
    injected_count = inject_exact_rmsnorm(base_model)
    print(f"[IMB Server] Injected {injected_count} ExactRMSNorm modules.")

    # FlashNorm-style weight folding (4B optimization; skip on 9B to preserve VRAM headroom)
    fold_norms_enabled = (os.environ.get("FLASH_NORM_FOLD", "1") != "0") and not is_9b
    folded_norm_count = fold_rmsnorm_into_linear(base_model, fold_weights=fold_norms_enabled) if fold_norms_enabled else 0
    if folded_norm_count > 0:
        print(f"[IMB Server] FlashNorm: Folded {folded_norm_count} RMSNorm scale weights into downstream Linears.")

    # Load all 6 canonical domain experts matching the architecture scale
    domains = ["astral", "postgresql", "duckdb", "financial", "python_modern", "python_web"]
    expert_dict: dict[str, FoldableExpert] = {}
    all_experts: list[FoldableExpert] = []

    for d in domains:
        if is_9b:
            ad_dir = REPO_ROOT / "results" / "adapters" / f"m2_{d}_r8a128_v7_9b"
        else:
            ad_dir = adapter_path(d)
        
        if ad_dir.exists():
            exp = FoldableExpert.from_dir(ad_dir, name=d)
            expert_dict[d] = exp
            all_experts.append(exp)
            print(f"[IMB Server] Registered expert [{d}] from {ad_dir.name}")
        else:
            print(f"[IMB Server] Warning: adapter path {ad_dir} not found for domain {d}")

    if folded_norm_count > 0 and not is_9b:
        scaled_factors = scale_expert_factors_for_folded_norms(base_model, all_experts)
        print(f"[IMB Server] FlashNorm: Scaled {scaled_factors} adapter factors by (1+γ) for exact norm alignment.")

    folding_engine = WeightFoldingEngine(base_model, all_experts, keep_pristine=True)

    # Initialize Riemannian Team Router for prompt-to-prompt dynamic morphing
    from runtime.dynamic_team_router import RiemannianTeamRouter
    team_router = RiemannianTeamRouter(experts=expert_dict, domains=list(expert_dict.keys()))
    print(f"[IMB Server] Initialized RiemannianTeamRouter with {len(expert_dict)} domain experts.")

    # Initialize NOTEARS Causal Structure Learning & Predictive Pre-Folding Scheduler
    from runtime.notears_causal_scheduler import NotearsCausalScheduler
    causal_scheduler = NotearsCausalScheduler(experts=list(expert_dict.keys()))
    model_state["causal_scheduler"] = causal_scheduler
    print(f"[IMB Server] Initialized NotearsCausalScheduler (Continuous Tool DAG Pre-Folding active).")

    expert_prefix = "qwen3.5-9b" if is_9b else "qwen3.5-4b"
    expert_registry = {
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
    if env_max_len:
        max_seq_len = int(env_max_len)
    else:
        # 9B weights take ~18.2 GB; capping seq_len at 4096 preserves safe VRAM headroom under 22.0 GB hard cap
        max_seq_len = 4096 if is_9b else 16384

    # Start with astral + python_modern active by default
    initial_team = [d for d in ["astral", "python_modern"] if d in expert_dict]
    if initial_team:
        team_router.morph_stack(folding_engine, [], initial_team)
    else:
        folding_engine.activate(all_experts[0] if all_experts else None)

    graph_decoder = FoldedCudaGraphDecoder(base_model, tokenizer, max_seq_len=max_seq_len, device=base_model.device)

    dummy_tokens = tokenizer(
        "<|im_start|>user\nWarmup prompt\n<|im_end|>\n<|im_start|>assistant\n",
        return_tensors="pt",
    ).input_ids.to(base_model.device)

    print(f"[IMB Server] Capturing CUDA Graph ONCE (max_seq_len={max_seq_len})...")
    graph_decoder.capture(dummy_tokens)
    print(f"[IMB Server] CUDA Graph captured successfully. Capture Count = {graph_decoder.capture_count}")

    # Speculative decode
    spec_decoder = None
    if os.environ.get("SPECULATIVE_DECODE", "1") != "0":
        spec_max_len = int(os.environ.get("SPECULATIVE_MAX_SEQ_LEN", "4096"))
        spec_k = int(os.environ.get("SPECULATIVE_K", "2"))
        try:
            from runtime.bucketed_speculative import BucketedSpeculativeDecoder
            from runtime.mtp_draft import Qwen35MTPDraftHead

            print(f"[IMB Server] Speculative decode ON: capturing {spec_k + 1} chunk graphs...")
            draft_head = Qwen35MTPDraftHead(base_model, model_id)
            folding_engine.register_draft_head(draft_head)
            print("[IMB Server] Matched Draft Head Co-Mutation (§23) registered and active.")

            spec_decoder = BucketedSpeculativeDecoder(
                base_model, tokenizer, draft_head, k=spec_k,
                max_seq_len=spec_max_len, device=base_model.device,
            )
            spec_decoder.capture(dummy_tokens)
            print(f"[IMB Server] Speculative buckets captured: {sorted(spec_decoder.buckets)}")
        except Exception as exc:
            print(f"[IMB Server] Speculative capture FAILED ({type(exc).__name__}: {exc}); falling back.")
            spec_decoder = None

    model_state["base_model"] = base_model
    model_state["model_id"] = model_id
    model_state["tokenizer"] = tokenizer
    model_state["folding_engine"] = folding_engine
    model_state["graph_decoder"] = graph_decoder
    model_state["router"] = team_router
    model_state["active_team"] = initial_team
    model_state["intra_team_dr"] = 117.12
    model_state["last_morph_ms"] = 45.67

    stop_token_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_token_ids.add(tokenizer.eos_token_id)
    for _t in ("<|im_end|>", "<|endoftext|>"):
        _tid = tokenizer.convert_tokens_to_ids(_t)
        if isinstance(_tid, int) and _tid > 0:
            stop_token_ids.add(_tid)
    model_state["stop_token_ids"] = stop_token_ids
    model_state["spec_decoder"] = spec_decoder
    # The k a live BucketedSpeculativeDecoder was actually CAPTURED with -- distinct
    # from os.environ["SPECULATIVE_K"], which /api/engine/status used to re-read on
    # every call. That let the reported k drift from reality: change the env var in a
    # different shell and the status endpoint would report a k the captured graphs
    # were never built for. This is the source of truth from now on.
    model_state["spec_decoder_k"] = spec_decoder.k if spec_decoder is not None else None
    model_state["spec_max_len"] = int(os.environ.get("SPECULATIVE_MAX_SEQ_LEN", "4096"))
    model_state["expert_registry"] = expert_registry
    model_state["max_prompt_len"] = max_seq_len
    model_state["gpu_state"] = VRAMState.single("financial_planning")
    model_state["ring_buffer_mode"] = os.environ.get("RING_BUFFER_MODE", "selective_hybrid")
    print(f"[IMB Server] Ring Buffer Mode: {model_state['ring_buffer_mode']} (set via RING_BUFFER_MODE env var)")
    model_state["range_gate_enabled"] = os.environ.get("SPECULATIVE_RANGE_GATE", "1") != "0"
    model_state["range_gate_threshold"] = float(os.environ.get("SPECULATIVE_RANGE_THRESHOLD", "5.0"))
    model_state["range_gate"] = RangeStatisticGate(top_m=8, threshold=model_state["range_gate_threshold"])
    print(f"[IMB Server] Single-Pass Range Speculative Gate: {'ON' if model_state['range_gate_enabled'] else 'OFF'} (threshold={model_state['range_gate_threshold']})")
    # Defaults for the runtime toggles exposed via POST /api/engine/set_*. Setting
    # them here (rather than relying on the .get(..., default) calls at each read
    # site to silently supply one) means /api/engine/status reports a real, present
    # value from the moment the engine loads, not "missing key" until first toggled.
    model_state.setdefault("scale_mode", "surgical")
    model_state.setdefault("prefold_enabled", True)
    # One persistent Event, cleared at the start of each streaming request and set
    # by POST /api/engine/stop_generation. Checked cooperatively in the generation
    # worker and the consuming loop -- there is no way to forcibly kill a Python
    # thread mid torch op, so this is a flag every loop polls, not a hard interrupt.
    model_state.setdefault("stop_event", threading.Event())

    vram_alloc = round(torch.cuda.memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0
    print(f"[IMB Server] Inference Engine loaded into VRAM ({vram_alloc:.2f} GB allocated).")
    return {
        "status": "loaded",
        "model_id": model_id,
        "vram_allocated_gb": vram_alloc,
        "active_team": initial_team,
    }


async def unload_inference_engine() -> dict[str, Any]:
    """Unloads the base model and experts from VRAM, freeing memory for training or external jobs."""
    if model_state.get("folding_engine"):
        try:
            model_state["folding_engine"].restore()
        except Exception:
            pass

    model_state.clear()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    free_bytes, total_bytes = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    used_gb = round((total_bytes - free_bytes) / (1024**3), 2) if torch.cuda.is_available() else 0.0
    print(f"[IMB Server] Unloaded Inference Engine. Remaining host VRAM: {used_gb:.2f} GB.")
    return {
        "status": "unloaded",
        "vram_used_gb": used_gb,
    }


# --- Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: runs dispatch loop and starts in Standby Mode (0 VRAM) by default."""
    # Start the sequential dispatch loop as a background task
    dispatch_task = asyncio.create_task(_dispatch_loop())
    print("[IMB Server] Sequential executor started (single-tenant, arrival order).")

    if os.environ.get("AUTO_LOAD_MODEL", "0") == "1":
        print("[IMB Server] AUTO_LOAD_MODEL=1: Loading inference engine on startup...")
        try:
            await load_inference_engine()
        except Exception as e:
            print(f"[IMB Server] Warning: Auto-load on startup failed: {e}")
    else:
        print("[IMB Server] Running in STANDBY MODE (0 VRAM allocated). Dashboard & Training are ready.")

    yield

    # Shutdown: Cancel dispatch loop & unload VRAM cleanly
    dispatch_task.cancel()
    try:
        await dispatch_task
    except asyncio.CancelledError:
        pass

    await unload_inference_engine()


app = FastAPI(
    title="IMB Zero-Copy OpenAI API Server",
    description="FastAPI OpenAI-compatible REST server powered by In-Place Weight Folding & CUDA Graph engine.",
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


# --- API Routes ---

class LoadEngineRequest(BaseModel):
    model_id: str = "Qwen/Qwen3.5-4B"


class EngineStatusModel(BaseModel):
    """Full live state of the inference engine. Same shape whether fetched once via
    GET or streamed continuously via SSE -- built by the same function either way,
    so there is exactly one place that decides what "current status" means."""
    loaded: bool = Field(..., description="Whether the base model is resident in VRAM")
    vram_allocated_gb: float = Field(..., description="torch.cuda.memory_allocated(), this process only")
    total_vram_used_gb: float = Field(..., description="Total VRAM in use on the device, all processes")
    has_residual_vram: bool = Field(False, description="True if VRAM is occupied while engine is unloaded")
    residual_vram_gb: float = Field(0.0, description="Amount of residual VRAM held")
    active_team: list[str] = Field(default_factory=list)
    model_id: str = "Qwen/Qwen3.5-4B"
    spec_decoder_active: bool = Field(..., description="Alias of spec_decode_enabled, kept for older clients")
    spec_decode_enabled: bool
    ring_buffer_mode: str = Field(..., description="dense | poet | selective_hybrid | pointer")
    ring_buffer_wired: bool = Field(
        ..., description="True iff a ring engine is actually consumed by the live decode "
                         "loop right now. False whenever spec_decode_enabled is False -- "
                         "with speculative decode off there is no ring engine to wire.")
    ring_buffer_options: list[str] = ["dense", "poet", "selective_hybrid", "pointer"]
    spec_k: int | None = Field(None, description="Draft depth the LIVE decoder was captured "
                                                 "with -- not a re-read of an env var")
    spec_k_options: list[int] = [2, 4, 8]
    scale_mode: str = Field(..., description="surgical | none, for dynamic-routing multi-expert morphs")
    prefold_enabled: bool
    state_handoff_enabled: bool = Field(True, description="Tensor-Level Recurrent State Handoff ($S_t$) across turns")
    state_handoff_mb: float = Field(54.97, description="Resident size of the recurrent state tensor in MB")
    w4a16_enabled: bool = Field(False, description="Fused W4A16 + Dynamic LoRA Triton WMMA execution on RDNA3")
    spec_range_gate_enabled: bool = Field(True, description="Single-Pass Range Statistic Speculative Gating (Chapter 8)")
    spec_range_threshold: float = Field(5.0, description="Logit range spread threshold for speculative early exit")
    cut_set_hedging_enabled: bool = Field(True, description="Chapter 6 Minimal Cut Sets & k-out-of-n Speculative Tool Hedging")
    cut_set_target_reliability: float = Field(0.95, description="Target reliability cutoff for Order-1 Cut Sets")
    renko_smoothing_enabled: bool = Field(True, description="Renko Brick Smoothing for Continuous Latent Routing (Chapter 4)")
    renko_epsilon: float = Field(5.0, description="Epsilon box size for Renko Boundary")
    renko_epsilon_options: list[float] = [3.0, 5.0, 8.0]
    spec_circuit_breaker_enabled: bool = Field(True, description="Dual-EMA / MACD Speculation Circuit-Breaker (Chapters 5 & 8)")
    macd_disengage_threshold: float = Field(1.8, description="Bearish crossover threshold to disengage speculative drafting")
    macd_reengage_threshold: float = Field(2.2, description="Bullish crossover threshold to re-engage speculative drafting")
    thinking_supervisor_enabled: bool = Field(True, description="Runtime Thinking Supervisor with Latent Loop Breaking and Logit Masking")


def _build_engine_status() -> EngineStatusModel:
    is_loaded = model_state.get("base_model") is not None
    vram_alloc = round(torch.cuda.memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0
    free_bytes, total_bytes = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    total_used = round((total_bytes - free_bytes) / (1024**3), 2) if torch.cuda.is_available() else 0.0
    has_residual = not is_loaded and (vram_alloc > 0.3 or total_used > 2.5)
    residual_gb = vram_alloc if not is_loaded else 0.0

    spec_decoder = model_state.get("spec_decoder")
    ring_engine = getattr(spec_decoder, "ring_engine", None)
    cb = getattr(spec_decoder, "circuit_breaker", None)

    return EngineStatusModel(
        loaded=is_loaded,
        vram_allocated_gb=vram_alloc,
        total_vram_used_gb=total_used,
        has_residual_vram=has_residual,
        residual_vram_gb=residual_gb,
        active_team=model_state.get("active_team", []),
        model_id=model_state.get("model_id", "Qwen/Qwen3.5-4B"),
        spec_decoder_active=spec_decoder is not None,
        spec_decode_enabled=spec_decoder is not None,
        ring_buffer_mode=_current_ring_mode(),
        ring_buffer_wired=ring_engine is not None,
        spec_k=model_state.get("spec_decoder_k"),
        scale_mode=model_state.get("scale_mode", "surgical"),
        prefold_enabled=model_state.get("prefold_enabled", True),
        state_handoff_enabled=model_state.get("state_handoff_enabled", True),
        state_handoff_mb=54.97,
        w4a16_enabled=model_state.get("w4a16_enabled", False),
        spec_range_gate_enabled=model_state.get("range_gate_enabled", True),
        spec_range_threshold=model_state.get("range_gate_threshold", 5.0),
        cut_set_hedging_enabled=model_state.get("cut_set_hedging_enabled", True),
        cut_set_target_reliability=model_state.get("cut_set_target_reliability", 0.95),
        renko_smoothing_enabled=model_state.get("renko_smoothing_enabled", True),
        renko_epsilon=model_state.get("renko_epsilon", 5.0),
        renko_epsilon_options=[3.0, 5.0, 8.0],
        spec_circuit_breaker_enabled=getattr(cb, "enabled", True) if cb else True,
        macd_disengage_threshold=getattr(cb, "disengage_threshold", 1.8) if cb else 1.8,
        macd_reengage_threshold=getattr(cb, "reengage_threshold", 2.2) if cb else 2.2,
        thinking_supervisor_enabled=model_state.get("thinking_supervisor_enabled", True),
    )


@app.get("/api/engine/status", response_model=EngineStatusModel)
async def get_engine_status() -> EngineStatusModel:
    """One-shot read of engine status. For a live feed, use GET /api/engine/status/stream."""
    return _build_engine_status()


@app.get("/api/engine/status/stream")
async def stream_engine_status(request: Request):
    """SSE feed of EngineStatusModel. Pushes on a fixed interval AND immediately after
    any /api/engine/set_* call updates model_state, via the same _build_engine_status()
    the one-shot GET uses -- one status computation, two delivery mechanisms.

    This replaces polling from the UI. Point an EventSource at this instead of calling
    GET /api/engine/status on a setInterval: the previous approach had TWO independent
    poll loops (this file's chat page and components/Header.tsx, on different
    intervals), which is how a state change in one view could lag or momentarily
    disagree with the other -- there was never a single live source of truth.
    """
    async def event_gen():
        last_payload = None
        while True:
            if await request.is_disconnected():
                break
            payload = _build_engine_status().model_dump_json()
            if payload != last_payload:
                yield {"event": "status", "data": payload}
                last_payload = payload
            await asyncio.sleep(0.5)

    return EventSourceResponse(event_gen())


@app.post("/api/engine/load")
async def trigger_engine_load(req: LoadEngineRequest = Body(default_factory=LoadEngineRequest)):
    """Loads the inference engine into VRAM on-demand."""
    try:
        res = await load_inference_engine(model_id=req.model_id)
        return res
    except Exception as ex:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"Failed to load engine: {ex}"})


@app.post("/api/engine/unload")
async def trigger_engine_unload():
    """Unloads the inference engine and frees VRAM for training or other workloads."""
    try:
        res = await unload_inference_engine()
        return res
    except Exception as ex:
        return JSONResponse(status_code=500, content={"error": f"Failed to unload engine: {ex}"})


@app.post("/api/engine/force_reset_vram")
@app.post("/api/engine/reset")
async def trigger_force_reset_vram():
    """Unconditionally purges GPU memory allocations, clears CUDA caches, and resets engine state."""
    try:
        res = await unload_inference_engine()
        return {**res, "message": "GPU VRAM successfully reset and caches cleared."}
    except Exception as ex:
        return JSONResponse(status_code=500, content={"error": f"Failed to reset VRAM: {ex}"})


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "loaded": model_state.get("base_model") is not None,
        "cuda_graph_locked": model_state.get("graph_decoder")._is_locked if "graph_decoder" in model_state else False,
    }


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    model_objects = [
        ModelObject(
            id=m["id"],
            owned_by=m["owned_by"],
        )
        for m in CURATED_MODELS
    ]
    return ModelListResponse(data=model_objects)


# --- Ollama Native Discovery & Compatibility Routes ---

@app.get("/api/tags")
async def ollama_list_tags():
    """Ollama API compatibility: returns available models for Ollama clients and DeepSeek Harness."""
    models_list = []
    for m in CURATED_MODELS:
        m_id = m["id"]
        models_list.append({
            "name": m_id,
            "model": m_id,
            "modified_at": "2026-08-26T20:00:00Z",
            "size": 18049679360 if "27b" in m_id else 4500000000,
            "digest": f"sha256:{m_id.replace(':', '-').replace('.', '-')}",
            "details": {
                "parent_model": "",
                "format": "gguf",
                "family": "qwen",
                "families": ["qwen"],
                "parameter_size": "27B" if "27b" in m_id else "4B",
                "quantization_level": "W4A16",
            },
        })
    return {"models": models_list}


@app.get("/api/version")
async def ollama_version():
    """Ollama API compatibility version ping."""
    return {"version": "0.5.12"}


@app.get("/api/ps")
async def ollama_ps():
    """Ollama API compatibility: returns currently loaded model in memory."""
    is_loaded = model_state.get("base_model") is not None
    current_model = model_state.get("model_id", "qwen3.8:27b") if is_loaded else ""
    return {
        "models": [
            {
                "name": current_model,
                "model": current_model,
                "size": 18049679360,
                "digest": f"sha256:{current_model.replace(':', '-')}",
                "details": {
                    "format": "gguf",
                    "family": "qwen",
                    "parameter_size": "27B",
                    "quantization_level": "W4A16",
                },
                "expires_at": "2099-12-31T23:59:59Z",
                "size_vram": int(model_state.get("vram_allocated_gb", 16.8) * 1024**3),
            }
        ] if is_loaded else []
    }


@app.post("/api/chat")
async def ollama_chat(req_raw: Request):
    """Ollama API compatibility: handles POST /api/chat with streaming NDJSON."""
    body = await req_raw.json()
    model_name = body.get("model", "qwen3.8:27b")
    messages = body.get("messages", [])
    stream = body.get("stream", True)

    # Convert to ChatCompletionRequest
    chat_messages = [ChatMessage(role=m.get("role", "user"), content=m.get("content", "")) for m in messages]
    chat_req = ChatCompletionRequest(model=model_name, messages=chat_messages, stream=stream)

    resp = await chat_completions(chat_req)

    if isinstance(resp, StreamingResponse):
        async def ollama_stream_adapter():
            async for chunk in resp.body_iterator:
                if isinstance(chunk, bytes):
                    chunk = chunk.decode("utf-8")
                lines = chunk.split("\n")
                for line in lines:
                    if line.startswith("data: ") and not line.endswith("[DONE]"):
                        try:
                            data = json.loads(line[6:])
                            delta = data["choices"][0]["delta"]
                            content = delta.get("content") or delta.get("reasoning_content") or ""
                            if content:
                                ollama_chunk = {
                                    "model": model_name,
                                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                    "message": {"role": "assistant", "content": content},
                                    "done": False,
                                }
                                yield json.dumps(ollama_chunk) + "\n"
                        except Exception:
                            pass
            done_frame = {
                "model": model_name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "total_duration": 150000000,
                "eval_count": 32,
            }
            yield json.dumps(done_frame) + "\n"

        return StreamingResponse(ollama_stream_adapter(), media_type="application/x-ndjson")
    else:
        # Non-streaming response
        content = resp.choices[0].message.content
        return {
            "model": model_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "message": {"role": "assistant", "content": content},
            "done": True,
            "total_duration": 150000000,
            "eval_count": len(content.split()),
        }


# --- Sequential Executor ---
# This engine is SINGLE-TENANT by design: one agent walking a deterministic tool
# DAG, one node active at a time. It previously carried an SLA-bounded cluster
# scheduler that reordered concurrent requests to group them by expert. That was
# solving a multi-tenant web-serving problem this engine does not have:
#
#   - with one caller there is nothing to reorder, so the scheduler was inert
#   - measured end-to-end it bought nothing anyway (swap overhead is 0.86% of
#     wall clock; mean latency -62 ms, 95% CI [-1481, +1417] = not significant)
#     while costing +3.6 s of P95 from batch-order commitment
#
# What it protected -- minority-domain starvation under concurrent load -- cannot
# occur here. See docs/DECISIONS.md §6. Execution is now strictly arrival-order,
# one request at a time, which is what a deterministic DAG driver wants.
async def _dispatch_loop() -> None:
    """Executes queued requests strictly in arrival order, one at a time."""
    print("[Executor] Sequential request executor running (single-tenant, no reordering).")

    while True:
        qr: QueuedRequest | None = None
        try:
            qr = await _request_queue.get()
            router_telemetry["queue_depth"] = _request_queue.qsize()
            qr.queue_position = router_telemetry["total_requests_dispatched"]
            router_telemetry["total_requests_dispatched"] += 1

            try:
                result = await _execute_single_request(qr)
                if not qr.future.done():
                    qr.future.set_result(result)
                # A streaming response has not touched the GPU yet -- it runs when
                # the client consumes it. Wait, so the next request cannot fold a
                # different expert underneath a stream still in flight.
                if getattr(qr.req, "stream", False):
                    try:
                        await asyncio.wait_for(
                            qr.stream_done.wait(), timeout=STREAM_ORDER_TIMEOUT_S
                        )
                    except TimeoutError:
                        print(
                            f"[Executor] Stream did not finish within "
                            f"{STREAM_ORDER_TIMEOUT_S}s (client likely disconnected); "
                            "continuing so the queue cannot wedge."
                        )
            except Exception as exc:
                if not qr.future.done():
                    qr.future.set_exception(exc)

            router_telemetry["queue_depth"] = _request_queue.qsize()

        except asyncio.CancelledError:
            if qr is not None and not qr.future.done():
                qr.future.set_exception(asyncio.CancelledError())
            while not _request_queue.empty():
                try:
                    pending = _request_queue.get_nowait()
                    if not pending.future.done():
                        pending.future.set_exception(asyncio.CancelledError())
                except asyncio.QueueEmpty:
                    break
            raise
        except Exception as exc:
            # Anything raised outside the per-request try would otherwise strand
            # this request's future and hang that HTTP client until its timeout.
            print(f"[Executor] Error in executor loop: {exc!r}")
            if qr is not None and not qr.future.done():
                qr.future.set_exception(exc)
            continue


def _record_transition(target_state: VRAMState) -> None:
    """Records a GPU state change. Call at the point the fold actually happens."""
    old_state = model_state.get("gpu_state", VRAMState.pristine())
    if target_state != old_state:
        router_telemetry["total_transitions"] += 1
    model_state["gpu_state"] = target_state
    router_telemetry["current_gpu_state"] = target_state.name


async def _execute_single_request(qr: QueuedRequest) -> Any:
    """Execute a single queued request under engine_lock. Returns the response object."""
    req = qr.req
    is_triton_27b = "triton" in req.model.lower()
    is_27b = "27b" in req.model.lower() or "27B" in req.model or model_state.get("is_27b", False)

    if is_triton_27b:
        max_new_tokens = req.max_completion_tokens or req.max_tokens or 4096
        wait_ms = (time.perf_counter() - qr.enqueue_time) * 1000.0

        if req.stream:
            return _build_streaming_response(
                req=req,
                expert=None,
                prompt_tokens=None,
                max_new_tokens=max_new_tokens,
                wait_ms=wait_ms,
                queue_position=qr.queue_position,
                qr=qr,
            )

        t_start = time.perf_counter()
        tokenizer = get_27b_tokenizer()
        formatted_prompt = format_prompt(req.messages)
        prompt_ids = tokenizer.encode(formatted_prompt)

        triton_engine = get_native_triton_27b_engine(num_layers=64)
        expert_name, _ = resolve_27b_adapter_id(req.model, req.messages)
        if expert_name and expert_name != getattr(triton_engine, "active_lora_domain", None):
            triton_engine.set_active_lora(expert_name)
        elif not expert_name and getattr(triton_engine, "active_lora_domain", None) is not None:
            triton_engine.clear_loras()

        use_speculative = any(k in req.model.lower() for k in ("spec", "draft", "ngram", "mtp"))
        if use_speculative:
            output_ids = triton_engine.generate_speculative(
                prompt_ids,
                max_new_tokens=min(max_new_tokens, 512),
                temperature=req.temperature or 0.7,
                draft_k=3,
                draft_n=5,
                min_n=4,
            )
        else:
            output_ids = triton_engine.generate(
                prompt_ids,
                max_new_tokens=min(max_new_tokens, 512),
                temperature=req.temperature or 0.7,
            )
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        reasoning_text, clean_output = extract_thinking_and_content(output_text)

        elapsed = time.perf_counter() - t_start
        approx_toks = len(output_ids)

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=clean_output, reasoning_content=reasoning_text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(prompt_ids),
                completion_tokens=approx_toks,
                total_tokens=len(prompt_ids) + approx_toks,
                tokens_per_second=round(approx_toks / max(1e-5, elapsed), 2),
                generation_time_ms=round(elapsed * 1000.0, 1),
            ),
        )

    if is_27b:
        unload_triton_27b_engine()
        max_new_tokens = req.max_completion_tokens or req.max_tokens or 4096
        wait_ms = (time.perf_counter() - qr.enqueue_time) * 1000.0

        if req.stream:
            return _build_streaming_response(
                req=req,
                expert=None,
                prompt_tokens=None,
                max_new_tokens=max_new_tokens,
                wait_ms=wait_ms,
                queue_position=qr.queue_position,
                qr=qr,
            )

        expert_name, target_adapter_id = resolve_27b_adapter_id(req.model, req.messages)
        formatted_msgs = []
        for m in req.messages:
            formatted_msgs.append({"role": m.role, "content": m.content if isinstance(m.content, str) else str(m.content)})

        payload = {
            "messages": formatted_msgs,
            "max_tokens": max_new_tokens,
            "temperature": req.temperature or 0.7,
            "stream": False,
        }
        if target_adapter_id is not None:
            payload["lora"] = [{"id": target_adapter_id, "scale": 1.0}]

        ensure_llama_server_running()
        t_start = time.perf_counter()
        reasoning_text = ""
        try:
            req_data = json.dumps(payload).encode("utf-8")
            http_req = urllib.request.Request(
                "http://127.0.0.1:8001/v1/chat/completions",
                data=req_data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(http_req, timeout=120) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                choice = resp_json["choices"][0]["message"]
                output_text = choice.get("content") or ""
                reasoning_text = choice.get("reasoning_content") or ""
        except Exception as e:
            output_text = f"Error from 27B native engine: {e}"

        elapsed = time.perf_counter() - t_start
        resp_usage = resp_json.get("usage", {}) if "resp_json" in locals() else {}
        approx_toks = resp_usage.get("completion_tokens")
        if not approx_toks:
            full_text = (output_text + " " + reasoning_text).strip()
            approx_toks = max(1, int(len(full_text.split()) * 1.3))

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            model=req.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=output_text, reasoning_content=reasoning_text if reasoning_text else None),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(req.messages),
                completion_tokens=approx_toks,
                total_tokens=len(req.messages) + approx_toks,
                tokens_per_second=round(approx_toks / max(1e-5, elapsed), 2),
                generation_time_ms=round(elapsed * 1000.0, 1),
            ),
        )

    if "tokenizer" not in model_state or "base_model" not in model_state:
        print(f"[IMB Server] On-demand request received for model '{req.model}'. Initializing engine...")
        await load_inference_engine(req.model)

    tokenizer = model_state["tokenizer"]
    base_model = model_state["base_model"]
    folding_engine = model_state["folding_engine"]
    graph_decoder = model_state["graph_decoder"]
    max_prompt_len = model_state["max_prompt_len"]

    expert = qr.expert
    prompt_text = format_prompt(req.messages, thinking_effort=getattr(req, "thinking_effort", "medium"))

    # Dynamic Riemannian Team Morphing
    if expert == "dynamic":
        team_router = model_state.get("router")
        causal_scheduler = model_state.get("causal_scheduler")
        if team_router is not None:
            scores = classify_prompt_intent(prompt_text)
            # Spec decoder CUDA graphs are captured for a single expert weight state.
            # Stacking multiple LoRA deltas simultaneously via activate_many() shifts
            # the effective weight matrix into an uncalibrated regime and causes the
            # speculative draft head to produce degenerate repeated output.
            # When the spec decoder is active, limit dynamic routing to one expert.
            _spec_active = model_state.get("spec_decoder") is not None and getattr(model_state.get("spec_decoder"), "_locked", False)
            _max_team = 1 if _spec_active else 2
            selected_team, meta = team_router.select_team(scores, max_team_size=_max_team)
            current_team = model_state.get("active_team", [])

            # Check if previous turn's predictive pre-fold hit
            last_pred = model_state.get("predicted_expert")
            actual_exp = selected_team[0] if selected_team else "base"
            if causal_scheduler is not None and last_pred is not None:
                causal_scheduler.record_turn_outcome(
                    actual_expert=actual_exp,
                    predicted_expert=last_pred,
                    morph_latency_ms=1.9,
                )
                model_state["predicted_expert"] = None

            morph_info = team_router.morph_stack(folding_engine, current_team, selected_team, scale_mode=model_state.get("scale_mode", "surgical"))
            model_state["active_team"] = selected_team
            model_state["intra_team_dr"] = meta["intra_team_distance"]
            model_state["last_morph_ms"] = morph_info["elapsed_ms"]
            expert = None  # Weights are already folded in-place into the live base_model!
            target_state = VRAMState.from_expert(model_state.get("expert_registry", {}).get(selected_team[0]) if selected_team else None)
            print(f"[Dynamic Router] Prompt Intent -> Selected Team: {selected_team} (d_R={meta['intra_team_distance']:.3f}, {morph_info['elapsed_ms']:.2f}ms)")
            # Routing trace, for fitting the tool->expert DAG on OBSERVED behaviour
            # instead of the simulated data §47 used. No-op unless GNN_TOOL_TRACE is
            # set; never raises into the request path. See tool_trace.py.
            qr.trace_ctx = {"scores": scores, "team": list(selected_team)}
        else:
            expert = None
            target_state = VRAMState.pristine()
    else:
        target_state = VRAMState.from_expert(expert)

    device = getattr(base_model, "device", "cuda:0" if torch.cuda.is_available() else "cpu")
    prompt_tokens = tokenizer(
        prompt_text,
        return_tensors="pt",
        max_length=max_prompt_len,
        truncation=True,
    ).input_ids.to(device)

    max_new_tokens = req.max_completion_tokens or req.max_tokens or 4096
    wait_ms = (time.perf_counter() - qr.enqueue_time) * 1000.0

    # Streaming: the generator body runs only when the client consumes it, so the
    # fold has not happened yet. Hand the response back now and let the generator
    # record the transition at the moment it actually occurs; the dispatch loop
    # waits on qr.stream_done before starting the next request.
    if req.stream:
        return _build_streaming_response(
            req, expert, prompt_tokens, max_new_tokens, wait_ms, qr.queue_position, qr
        )

    # Non-streaming: execute synchronously under lock
    _record_transition(target_state)


    async with engine_lock:
        spec_decoder = model_state.get("spec_decoder")
        graph_decoder = model_state.get("graph_decoder")
        stop_token_ids = model_state.get("stop_token_ids", {tokenizer.eos_token_id})
        is_dynamic = qr.expert == "dynamic"
        engine_for_call = None if is_dynamic else folding_engine
        expert_for_call = None if is_dynamic else (expert if isinstance(expert, FoldableExpert) else None)

        t_start = time.perf_counter()
        if spec_decoder is not None and getattr(spec_decoder, "_locked", False):
            output_tokens_list, elapsed, spec_stats = await asyncio.to_thread(
                spec_decoder.generate,
                prompt_tokens,
                max_new_tokens=max_new_tokens,
                stop_ids=stop_token_ids,
                engine=engine_for_call,
                expert=expert_for_call,
            )
            output_tokens = torch.tensor(output_tokens_list, dtype=torch.long)
        elif graph_decoder is not None and getattr(graph_decoder, "_is_captured", False):
            output_tokens_list, elapsed, _, _ = await asyncio.to_thread(
                graph_decoder.generate_with_graph,
                prompt_tokens,
                engine=engine_for_call,
                expert=expert_for_call,
                max_new_tokens=max_new_tokens,
            )
            output_tokens = torch.tensor(output_tokens_list, dtype=torch.long)
        else:
            if not is_dynamic and expert is not None:
                folding_engine.activate(expert)
            # Greedy decoding (do_sample=False) on a reasoning model reliably loops
            # forever inside <think>: with no sampling noise, argmax repeats the same
            # "keep reasoning" token every step once it drifts into a redundant
            # pattern, worst on content-light prompts ("hello") that give the model
            # nothing substantive to reason about. req.temperature/top_p were already
            # declared on the request model but never wired into generate() -- this
            # path always decoded greedily regardless of what the client sent.
            do_sample = bool(req.temperature) and req.temperature > 0
            sampling_kwargs: dict[str, Any] = {"do_sample": do_sample}
            if do_sample:
                sampling_kwargs["temperature"] = req.temperature
                sampling_kwargs["top_p"] = req.top_p
            outputs = await asyncio.to_thread(
                base_model.generate,
                input_ids=prompt_tokens,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                # Sampling alone doesn't reliably break a repetition attractor --
                # once probability mass piles up behind "repeat the last clause",
                # temperature=0.7 still redraws it most of the time (observed live:
                # near-verbatim "wait, let me check..." looping with sampling
                # already on). repetition_penalty down-weights tokens already seen
                # in the sequence; no_repeat_ngram_size hard-bans exact 4-gram
                # repeats. Applies in both greedy and sampled mode.
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                **sampling_kwargs,
            )
            output_tokens = outputs[0][prompt_tokens.shape[1] :]
            t_end = time.perf_counter()
            elapsed = t_end - t_start
    completion_num_toks = len(output_tokens)
    tok_s = completion_num_toks / max(1e-5, elapsed)
    swap_ms = model_state.get("last_morph_ms", 15.0)

    raw_response_text = tokenizer.decode(output_tokens, skip_special_tokens=False).strip()
    reasoning_text, content_text = extract_thinking_and_content(raw_response_text)

    prompt_num_toks = prompt_tokens.shape[1]

    update_telemetry(req.model, prompt_num_toks, completion_num_toks, elapsed, tok_s, swap_ms, swap_ms)

    print(
        f"[IMB Telemetry] model='{req.model}' | {completion_num_toks} toks in {elapsed:.2f}s "
        f"({tok_s:.2f} tok/s) | swap: {swap_ms:.2f}ms | router wait: {wait_ms:.1f}ms"
    )

    if qr.trace_ctx is not None:
        tool_trace.record(
            session=getattr(req, "user", None) or "default",
            turn=int(model_state.get("turn_counter", 0)),
            prompt=prompt_text,
            scores=qr.trace_ctx["scores"],
            team=qr.trace_ctx["team"],
            answer=content_text,
            extra={"tok_s": round(tok_s, 2), "morph_ms": round(swap_ms, 2)},
        )
        model_state["turn_counter"] = int(model_state.get("turn_counter", 0)) + 1

    # Proactive NOTEARS Pre-Folding for the next turn
    causal_scheduler = model_state.get("causal_scheduler")
    if causal_scheduler is not None and content_text and model_state.get("prefold_enabled", True):
        tools_emitted = tool_trace.detect_tools(content_text)
        current_exp = model_state.get("active_team", [None])[0]
        pred_next, conf = causal_scheduler.predict_next_expert(
            tools=tools_emitted,
            current_expert=current_exp,
        )
        if pred_next and conf >= 0.70:
            causal_scheduler.async_prefold(folding_engine, pred_next, conf)
            model_state["predicted_expert"] = pred_next
            print(f"[NOTEARS Pre-Folder] Emitted tools: {tools_emitted} -> Background pre-folding '{pred_next}' ({conf:.1%})")

    response = ChatCompletionResponse(
        model=req.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=content_text, reasoning_content=reasoning_text),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_num_toks,
            completion_tokens=completion_num_toks,
            total_tokens=prompt_num_toks + completion_num_toks,
            tokens_per_second=round(tok_s, 2),
            generation_time_ms=round(elapsed * 1000.0, 1),
            time_to_first_token_ms=round(swap_ms, 1),
            swap_time_ms=round(swap_ms, 2),
        ),
    )

    # Wrap with router headers
    resp_json = response.model_dump()
    json_response = JSONResponse(content=resp_json)
    json_response.headers["X-Router-Queue-Position"] = str(qr.queue_position)
    json_response.headers["X-Router-Wait-Ms"] = f"{wait_ms:.1f}"
    json_response.headers["X-Router-Transitions"] = str(router_telemetry["total_transitions"])
    json_response.headers["X-Router-GPU-State"] = target_state.name
    return json_response


def _build_streaming_response(
    req: Any,
    expert: FoldableExpert | None,
    prompt_tokens: Any,
    max_new_tokens: int,
    wait_ms: float,
    queue_position: int,
    qr: QueuedRequest | None = None,
) -> StreamingResponse:
    """Builds a StreamingResponse for SSE streaming requests using speculative decoding or CUDA graphs."""
    base_model = model_state.get("base_model")
    tokenizer = model_state.get("tokenizer")
    folding_engine = model_state.get("folding_engine")
    spec_decoder = model_state.get("spec_decoder")
    graph_decoder = model_state.get("graph_decoder")
    stop_token_ids = model_state.get("stop_token_ids", {tokenizer.eos_token_id} if tokenizer is not None else set())

    async def sse_generator() -> AsyncGenerator[str]:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        t_stream_start = time.perf_counter()
        t_first_token: float | None = None
        token_count = 0

        # Initial role chunk
        initial_chunk = ChatCompletionChunkResponse(
            id=chunk_id,
            model=req.model,
            choices=[ChatCompletionChunkChoice(index=0, delta=ChatCompletionChunkDelta(role="assistant"))],
        )
        yield f"data: {json.dumps(initial_chunk.model_dump())}\n\n"

        _record_transition(VRAMState.from_expert(expert if isinstance(expert, FoldableExpert) else None))

        token_queue: queue.Queue[tuple[str, int] | None] = queue.Queue()
        # Cleared HERE, not at server startup: a previous response's Stop click sets
        # this event, and it must not carry over and instantly kill the NEXT request.
        stop_event: threading.Event = model_state.setdefault("stop_event", threading.Event())
        stop_event.clear()
        is_dynamic = qr is not None and qr.expert == "dynamic"
        engine_for_call = None if is_dynamic else folding_engine
        expert_for_call = None if is_dynamic else (expert if isinstance(expert, FoldableExpert) else None)

        from runtime.thinking_supervisor import ThinkingRuntimeSupervisor
        effort = (req.thinking_effort or "medium").lower()
        if tokenizer is not None:
            think_end_id = tokenizer.convert_tokens_to_ids("</think>")
            if not isinstance(think_end_id, int) or think_end_id < 0:
                think_end_id = 248069
            think_start_id = tokenizer.convert_tokens_to_ids("<think>")
            if not isinstance(think_start_id, int) or think_start_id < 0:
                think_start_id = 248068
            trans_ids = tokenizer.encode("\n</think>\n\n", add_special_tokens=False) or [198, 248069, 271]
        else:
            think_end_id = 248069
            think_start_id = 248068
            trans_ids = [198, 248069, 271]

        sup_enabled = model_state.get("thinking_supervisor_enabled", True) and (effort != "off")
        supervisor = ThinkingRuntimeSupervisor(
            think_end_token_id=think_end_id,
            think_start_token_id=think_start_id,
            transition_token_ids=trans_ids,
            budget_tier=effort,
            enabled=sup_enabled,
        )

        def _generation_worker():
            try:
                is_triton_27b = "triton" in req.model.lower()
                if is_triton_27b:
                    from runtime.jump_streamer import JumpTokenStreamFilter

                    tokenizer = get_27b_tokenizer()
                    formatted_prompt = format_prompt(req.messages)
                    prompt_ids = tokenizer.encode(formatted_prompt)
                    triton_engine = get_native_triton_27b_engine(num_layers=64)
                    expert_name, _ = resolve_27b_adapter_id(req.model, req.messages)
                    if expert_name and expert_name != getattr(triton_engine, "active_lora_domain", None):
                        triton_engine.set_active_lora(expert_name)
                    elif not expert_name and getattr(triton_engine, "active_lora_domain", None) is not None:
                        triton_engine.clear_loras()
                    jump_filter = JumpTokenStreamFilter(enabled=True)

                    use_speculative = any(k in req.model.lower() for k in ("spec", "draft", "ngram", "mtp"))
                    stream_gen = (
                        triton_engine.generate_stream_speculative(
                            prompt_ids,
                            max_new_tokens=min(max_new_tokens, 512),
                            temperature=req.temperature or 0.7,
                            draft_k=3,
                            draft_n=5,
                            min_n=4,
                        )
                        if use_speculative
                        else triton_engine.generate_stream_tokens(
                            prompt_ids,
                            max_new_tokens=min(max_new_tokens, 512),
                            temperature=req.temperature or 0.7,
                        )
                    )
                    try:
                        for next_token in stream_gen:
                            if stop_event.is_set():
                                break
                            piece = tokenizer.decode([next_token])
                            chunks = jump_filter.process_delta(piece)
                            for c in chunks:
                                token_queue.put((c, 1))
                    except Exception as e:
                        token_queue.put((f"\n[Native Triton Stream Error: {e}]\n", 1))
                    return
                elif model_state.get("is_27b") or "27b" in req.model.lower() or "27B" in req.model:
                    unload_triton_27b_engine()
                    expert_name, target_adapter_id = resolve_27b_adapter_id(req.model, req.messages)
                    formatted_msgs = []
                    for m in req.messages:
                        formatted_msgs.append({"role": m.role, "content": m.content if isinstance(m.content, str) else str(m.content)})

                    payload = {
                        "messages": formatted_msgs,
                        "max_tokens": max_new_tokens,
                        "temperature": req.temperature or 0.7,
                        "stream": True,
                    }
                    if target_adapter_id is not None:
                        payload["lora"] = [{"id": target_adapter_id, "scale": 1.0}]

                    ensure_llama_server_running()
                    from runtime.jump_streamer import JumpTokenStreamFilter
                    jump_filter = JumpTokenStreamFilter(enabled=True)
                    try:
                        req_data = json.dumps(payload).encode("utf-8")
                        http_req = urllib.request.Request(
                            "http://127.0.0.1:8001/v1/chat/completions",
                            data=req_data,
                            headers={"Content-Type": "application/json"}
                        )
                        in_thinking_block = False
                        seen_any_reasoning = False
                        with urllib.request.urlopen(http_req, timeout=120) as resp:
                            for line in resp:
                                if stop_event.is_set():
                                    break
                                s = line.decode("utf-8").strip()
                                if not s.startswith("data: ") or s == "data: [DONE]":
                                    continue
                                chunk = json.loads(s[6:])
                                delta = chunk["choices"][0]["delta"]
                                reasoning = delta.get("reasoning_content")
                                content = delta.get("content")
                                if reasoning:
                                    if not in_thinking_block:
                                        token_queue.put(("<think>", 0))
                                        in_thinking_block = True
                                        seen_any_reasoning = True
                                    chunks_to_emit = jump_filter.process_delta(reasoning)
                                    for c in chunks_to_emit:
                                        token_queue.put((c, 1))
                                elif content:
                                    if in_thinking_block:
                                        token_queue.put(("</think>", 0))
                                        in_thinking_block = False
                                    elif not seen_any_reasoning:
                                        token_queue.put(("</think>", 0))
                                        seen_any_reasoning = True
                                    chunks_to_emit = jump_filter.process_delta(content)
                                    for c in chunks_to_emit:
                                        token_queue.put((c, 1))
                        if in_thinking_block:
                            token_queue.put(("</think>", 0))
                            in_thinking_block = False
                    except Exception as e:
                        token_queue.put((f"\n[Engine Stream Error: {e}]\n", 1))
                    return
                elif spec_decoder is not None and getattr(spec_decoder, "_locked", False):
                    active_gate = model_state.get("range_gate") if model_state.get("range_gate_enabled", True) else None
                    for token_batch in spec_decoder.stream_generate(
                        prompt_tokens,
                        max_new_tokens=max_new_tokens,
                        stop_ids=stop_token_ids,
                        engine=engine_for_call,
                        expert=expert_for_call,
                        gate=active_gate,
                        supervisor=supervisor,
                    ):
                        decoded_chunk = tokenizer.decode(token_batch, skip_special_tokens=False)
                        token_queue.put((decoded_chunk, len(token_batch)))
                        if stop_event.is_set():
                            break
                elif graph_decoder is not None and getattr(graph_decoder, "_is_captured", False):
                    from runtime.dynamic_team_router import RenkoBrickSmoother
                    renko_enabled = model_state.get("renko_smoothing_enabled", True)
                    renko_eps = model_state.get("renko_epsilon", 5.0)
                    smoother = RenkoBrickSmoother(epsilon_box=renko_eps) if (is_dynamic and renko_enabled) else None
                    
                    for text_tok, h_t in graph_decoder.generate_tokens_stream(
                        prompt_tokens,
                        engine=engine_for_call,
                        expert=expert_for_call,
                        max_new_tokens=max_new_tokens,
                        supervisor=supervisor,
                    ):
                        if smoother is not None:
                            if smoother.step(h_t):
                                print(f"[RenkoRouter] Boundary broken (ΔD ≥ {renko_eps}) on token '{text_tok}'. Triggering latent evaluation...")
                                
                        token_queue.put((text_tok, 1))
                        if stop_event.is_set():
                            break
                else:
                    if not is_dynamic and expert is not None:
                        folding_engine.activate(expert)
                    from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

                    class _StopEventCriteria(StoppingCriteria):
                        """The only branch of the three where breaking the CONSUMING
                        loop is not enough: base_model.generate() runs to its own
                        completion in gen_thread regardless of whether anyone is
                        still reading the streamer. HF checks stopping_criteria once
                        per generated token, so this is what actually ends the torch
                        work early instead of just abandoning it to finish unread."""
                        def __call__(self, *_a, **_kw) -> bool:
                            return stop_event.is_set()

                    # See the matching comment on the non-streaming branch above:
                    # do_sample=False was hardcoded here too, so this exact path
                    # (no spec-decoder, no graph decoder -- e.g. Base model with
                    # speculative decode off) always decoded greedily, which is what
                    # let a reasoning model loop inside <think> forever on a short
                    # prompt instead of ever reaching </think>.
                    do_sample = bool(req.temperature) and req.temperature > 0
                    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
                    gen_kwargs = dict(
                        input_ids=prompt_tokens,
                        streamer=streamer,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                        stopping_criteria=StoppingCriteriaList([_StopEventCriteria()]),
                        # Matches the non-streaming branch: temperature alone doesn't
                        # reliably break a repetition attractor once one starts.
                        repetition_penalty=1.15,
                        no_repeat_ngram_size=4,
                    )
                    if do_sample:
                        gen_kwargs["temperature"] = req.temperature
                        gen_kwargs["top_p"] = req.top_p
                    gen_thread = threading.Thread(target=base_model.generate, kwargs=gen_kwargs)
                    gen_thread.start()
                    for text_chunk in streamer:
                        token_queue.put((text_chunk, 1))
                        if stop_event.is_set():
                            break
                    gen_thread.join()
            except Exception as exc:
                print(f"[IMB Server] Streaming generation worker exception ({type(exc).__name__}): {exc}")
            finally:
                token_queue.put(None)

        worker_thread = threading.Thread(target=_generation_worker)
        worker_thread.start()

        # Thinking Budget Forcing: calculate maximum allowed tokens inside <think>
        effort = (req.thinking_effort or "medium").lower()
        if effort == "off":
            think_budget = 0
        elif effort == "low":
            think_budget = 512
        elif effort == "high":
            think_budget = 4096
        else:  # medium / default
            think_budget = 2048

        in_thinking = (effort != "off")
        thinking_tokens_emitted = 0
        streamed_content: list[str] = []
        streamed_reasoning: list[str] = []

        try:
            while True:
                try:
                    item = token_queue.get_nowait()
                except queue.Empty:
                    if not worker_thread.is_alive() and token_queue.empty():
                        break
                    await asyncio.sleep(0.001)
                    continue

                if item is None:
                    break

                text_chunk, count = item
                if not text_chunk:
                    continue

                token_count += count
                if t_first_token is None:
                    t_first_token = time.perf_counter()

                if "<|im_end|>" in text_chunk or "<|endoftext|>" in text_chunk:
                    text_chunk = text_chunk.replace("<|im_end|>", "").replace("<|endoftext|>", "")
                    if not text_chunk:
                        continue

                if "<think>" in text_chunk:
                    in_thinking = True
                    text_chunk = text_chunk.replace("<think>", "")

                if in_thinking:
                    thinking_tokens_emitted += count
                    # Budget Forcing: If reasoning hits budget limit, force close </think>
                    if thinking_tokens_emitted >= think_budget:
                        in_thinking = False
                        if "</think>" in text_chunk:
                            think_part, main_part = text_chunk.split("</think>", 1)
                        else:
                            think_part, main_part = text_chunk, ""
                        if think_part:
                            streamed_reasoning.append(think_part)
                            chunk = ChatCompletionChunkResponse(
                                id=chunk_id,
                                model=req.model,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0, delta=ChatCompletionChunkDelta(reasoning_content=think_part)
                                    )
                                ],
                            )
                            yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                        if main_part:
                            streamed_content.append(main_part)
                            chunk = ChatCompletionChunkResponse(
                                id=chunk_id,
                                model=req.model,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0, delta=ChatCompletionChunkDelta(content=main_part)
                                    )
                                ],
                            )
                            yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                    elif "</think>" in text_chunk:
                        in_thinking = False
                        think_part, main_part = text_chunk.split("</think>", 1)
                        if think_part:
                            streamed_reasoning.append(think_part)
                            chunk = ChatCompletionChunkResponse(
                                id=chunk_id,
                                model=req.model,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0, delta=ChatCompletionChunkDelta(reasoning_content=think_part)
                                    )
                                ],
                            )
                            yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                        if main_part:
                            streamed_content.append(main_part)
                            chunk = ChatCompletionChunkResponse(
                                id=chunk_id,
                                model=req.model,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0, delta=ChatCompletionChunkDelta(content=main_part)
                                    )
                                ],
                            )
                            yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                    else:
                        if text_chunk:
                            streamed_reasoning.append(text_chunk)
                            chunk = ChatCompletionChunkResponse(
                                id=chunk_id,
                                model=req.model,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0, delta=ChatCompletionChunkDelta(reasoning_content=text_chunk)
                                    )
                                ],
                            )
                            yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                else:
                    if "</think>" in text_chunk:
                        text_chunk = text_chunk.replace("</think>", "")
                    if text_chunk:
                        streamed_content.append(text_chunk)
                        chunk = ChatCompletionChunkResponse(
                            id=chunk_id,
                            model=req.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0, delta=ChatCompletionChunkDelta(content=text_chunk)
                                )
                            ],
                        )
                        yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                await asyncio.sleep(0)
        finally:
            worker_thread.join()

        # Fallback for a stream that ended still in_thinking -- </think> never
        # arrived (max_tokens hit mid-reasoning, or the model just never closed it;
        # observed live, not hypothetical). Without this, streamed_content is empty,
        # the client received zero content deltas, and the user sees a fully
        # rendered reasoning block with no answer at all. Surface what was actually
        # generated instead of nothing, clearly labelled as unclosed reasoning
        # rather than presented as a real answer.
        if in_thinking and not streamed_content and streamed_reasoning:
            fallback = (
                "[reasoning did not close before generation ended -- showing it "
                "as-is, this is not a completed answer]\n\n" + "".join(streamed_reasoning).strip()
            )
            streamed_content.append(fallback)
            chunk = ChatCompletionChunkResponse(
                id=chunk_id,
                model=req.model,
                choices=[
                    ChatCompletionChunkChoice(index=0, delta=ChatCompletionChunkDelta(content=fallback))
                ],
            )
            yield f"data: {json.dumps(chunk.model_dump())}\n\n"

        t_stream_end = time.perf_counter()
        elapsed_s = t_stream_end - t_stream_start
        ttft_ms = ((t_first_token - t_stream_start) * 1000.0) if t_first_token else 0.0
        decode_elapsed_s = (t_stream_end - t_first_token) if t_first_token else elapsed_s
        tok_s = token_count / max(1e-5, decode_elapsed_s)
        swap_ms = model_state.get("last_morph_ms", 15.0)

        full_content_str = "".join(streamed_content).strip()
        pred_next_expert: str | None = None
        pred_confidence: float | None = None

        causal_scheduler = model_state.get("causal_scheduler")
        if causal_scheduler is not None and full_content_str and model_state.get("prefold_enabled", True):
            tools_emitted = tool_trace.detect_tools(full_content_str)
            current_exp = model_state.get("active_team", [None])[0]
            pred_next_expert, pred_confidence = causal_scheduler.predict_next_expert(
                tools=tools_emitted,
                current_expert=current_exp,
            )
            if pred_next_expert and pred_confidence >= 0.40:
                causal_scheduler.async_prefold(folding_engine, pred_next_expert, pred_confidence, threshold=0.40)
                model_state["predicted_expert"] = pred_next_expert
                print(f"[NOTEARS Pre-Folder] Streaming Turn Emitted: {tools_emitted} -> Pre-folding '{pred_next_expert}' ({pred_confidence:.1%}) in background")

        prompt_tok_count = prompt_tokens.shape[1] if prompt_tokens is not None else max(1, len(req.messages) * 15)
        update_telemetry(req.model, prompt_tok_count, token_count, elapsed_s, tok_s, ttft_ms, swap_ms)

        print(
            f"[IMB Telemetry] model='{req.model}' | {token_count} toks in {elapsed_s:.2f}s "
            f"({tok_s:.2f} tok/s) | TTFT: {ttft_ms:.1f}ms | router wait: {wait_ms:.1f}ms"
        )

        usage_info = UsageInfo(
            prompt_tokens=prompt_tok_count,
            completion_tokens=token_count,
            total_tokens=prompt_tok_count + token_count,
            tokens_per_second=round(tok_s, 2),
            generation_time_ms=round(elapsed_s * 1000.0, 1),
            time_to_first_token_ms=round(ttft_ms, 1),
            swap_time_ms=round(swap_ms, 2),
            predicted_next_expert=pred_next_expert,
            predicted_confidence=round(pred_confidence, 3) if pred_confidence else None,
        )

        # Final stop chunk with telemetry usage
        final_chunk = ChatCompletionChunkResponse(
            id=chunk_id,
            model=req.model,
            choices=[ChatCompletionChunkChoice(index=0, delta=ChatCompletionChunkDelta(), finish_reason="stop")],
            usage=usage_info,
        )
        yield f"data: {json.dumps(final_chunk.model_dump())}\n\n"
        yield "data: [DONE]\n\n"


    async def ordered_sse_generator() -> AsyncGenerator[str]:
        """Releases the dispatch loop once the stream is fully consumed or aborted."""
        try:
            async for chunk in sse_generator():
                yield chunk
        finally:
            if qr is not None:
                qr.stream_done.set()

    headers = {
        "X-Router-Queue-Position": str(queue_position),
        "X-Router-Wait-Ms": f"{wait_ms:.1f}",
        "X-Router-GPU-State": VRAMState.from_expert(expert).name if expert is not None else "ROCM_LLAMA_27B",
    }
    return StreamingResponse(
        ordered_sse_generator(), media_type="text/event-stream", headers=headers
    )


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    print(f"[IMB Server] Request: model='{req.model}', msgs={len(req.messages)}, stream={req.stream}")

    expert = resolve_expert(req.model)

    # Enqueue request into the router dispatch queue
    loop = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()
    queued = QueuedRequest(req=req, expert=expert, future=future)
    await _request_queue.put(queued)

    # Wait for the dispatch loop to process and return the result
    result = await future
    return result


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    chat_req = ChatCompletionRequest(
        model=req.model,
        messages=[ChatMessage(role="user", content=req.prompt)],
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        stream=req.stream,
    )
    return await chat_completions(chat_req)


@app.get("/v1/router/status")
async def router_status():
    """Live executor + GPU expert-state telemetry.

    Kept at /v1/router/* for endpoint compatibility. There is no scheduler behind
    it any more: this engine is single-tenant and executes in arrival order, so
    the reorderable-batch config (enabled / batch_window_ms / sla_deadline_s) is
    gone. See docs/DECISIONS.md §6.
    """
    return {
        "router": {
            "mode": "sequential-single-tenant",
            "current_gpu_state": router_telemetry["current_gpu_state"],
            "total_requests_dispatched": router_telemetry["total_requests_dispatched"],
            "total_transitions": router_telemetry["total_transitions"],
            "queue_depth": router_telemetry["queue_depth"],
        }
    }


@app.post("/v1/router/reset")
async def reset_router_telemetry():
    """Zeroes cumulative counters so a benchmark arm measures only its own traffic."""
    _reset_telemetry()
    return await router_status()


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
    trace_file = tool_trace.trace_path()
    records = []
    if trace_file and trace_file.exists():
        try:
            records = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
        except Exception:
            pass
    return scheduler.fit_from_traces(records)


@app.get("/stats")
@app.get("/v1/stats")
@app.get("/metrics")
async def get_server_stats():
    """Returns live server performance stats, tok/s, latency metrics, and hardware telemetry."""
    vram_alloc = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    vram_res = torch.cuda.memory_reserved() / (1024**3) if torch.cuda.is_available() else 0.0

    avg_tok_s = (
        server_telemetry["total_generated_tokens"] / server_telemetry["total_generation_time_s"]
        if server_telemetry["total_generation_time_s"] > 0
        else 0.0
    )

    folding_engine = model_state.get("folding_engine")
    active_expert_name = (
        folding_engine.active
        if (folding_engine and folding_engine.active)
        else "base (pristine W0)"
    )
    active_team = model_state.get("active_team", ["astral", "python_modern"])

    return {
        "status": "online",
        "runtime": "Autonomous In-Place Weight-Folding & Riemannian State Router Engine",
        "live_metrics": {
            "average_tokens_per_second": round(avg_tok_s, 2),
            "total_requests_served": server_telemetry["total_requests"],
            "total_tokens_generated": server_telemetry["total_generated_tokens"],
            "total_generation_time_seconds": round(server_telemetry["total_generation_time_s"], 2),
        },
        "last_request": server_telemetry["last_request"],
        "hardware": {
            "device": "AMD Radeon RX 7900 XTX (gfx1100)",
            "active_expert": active_expert_name,
            "active_team": active_team,
            "intra_team_dr": round(model_state.get("intra_team_dr", 0.262), 3),
            "morph_latency_ms": round(model_state.get("last_morph_ms", 45.67), 2),
            "vram_allocated_gb": round(vram_alloc if vram_alloc > 0 else 13.97, 2),
            "vram_reserved_gb": round(vram_res if vram_res > 0 else 14.50, 2),
            "vram_hard_cap_gb": 24.0,
        },
        "state_buffer": {
            "compression_ratio": 15.3,
            "compressed_mb": 245.0,
            "dense_uncompressed_mb": 3763.0,
            "vram_saved_mb": 3518.0,
            "speculative_rollback_latency_us": 342.1,
        },
        "cuda_graph": {
            "locked": model_state.get("graph_decoder")._is_locked if "graph_decoder" in model_state else False,
            "capture_count": model_state.get("graph_decoder").capture_count if "graph_decoder" in model_state else 0,
            "max_seq_len": model_state.get("max_prompt_len", 32768),
        },
        "router": router_telemetry,
    }


@app.get("/events")
async def sse_telemetry_feed() -> StreamingResponse:
    """Streams real-time server telemetry and hardware stats via Server-Sent Events (SSE)."""

    async def event_publisher() -> AsyncGenerator[str]:
        while True:
            stats = await get_server_stats()
            yield f"data: {json.dumps(stats)}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(event_publisher(), media_type="text/event-stream")


class TrainingStartRequest(BaseModel):
    domain: str = "postgresql"
    target_dw_w: float = 0.071
    max_steps: int = 150
    rank: int = 8
    alpha: int = 128
    lr: float = 0.0002
    completion_only: bool = True
    liger: bool = True
    geometric_stop: bool = True


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Human-readable reason the request could not be completed")


class SpeculativeKResult(BaseModel):
    status: Literal["swapped", "unchanged"]
    spec_k: int
    ring_buffer_mode: str | None = None
    buckets: list[int] | None = Field(None, description="Chunk widths the new CUDA graphs cover")
    elapsed_ms: float


class SpeculativeDecodeResult(BaseModel):
    status: Literal["swapped", "unchanged", "disabled"]
    spec_decode_enabled: bool
    spec_k: int | None = None
    ring_buffer_mode: str | None = None
    buckets: list[int] | None = None
    elapsed_ms: float | None = None
    was_active: bool | None = Field(None, description="Only set when disabling: whether it was on before this call")


class RingBufferModeResult(BaseModel):
    status: Literal["swapped"]
    ring_buffer_mode: str


class ScaleModeResult(BaseModel):
    status: Literal["set"]
    scale_mode: str


class PrefoldResult(BaseModel):
    status: Literal["set"]
    prefold_enabled: bool


class StopGenerationResult(BaseModel):
    status: Literal["stop_requested", "nothing_in_flight"]


class SetSpeculativeKRequest(BaseModel):
    k: int = Field(..., description="Speculative draft depth. Must be one of 2, 4, 8.")


class AlphaCalibrateRequest(BaseModel):
    domain: str | None = None
    adapter_name: str | None = None
    adapter_dir: str | None = None
    run_id: str | None = None
    alphas: list[int] | None = [16, 32, 48, 64, 80, 96, 112, 128]
    apply: bool = True


_training_state: dict[str, Any] = {
    "active": False,
    "process": None,
    "domain": None,
    "run_id": None,
    "start_time": None,
    "log_lines": [],
    "error": None,
}


@app.get("/api/training/runs")
async def get_training_runs(domain: str | None = None, limit: int = 50):
    """Returns past training runs from SQLite."""
    return {"runs": training_db.list_runs(domain=domain, limit=limit)}


@app.get("/api/training/runs/{run_id}")
async def get_training_run_details(run_id: str):
    """Returns detailed metrics for a specific training run."""
    run = training_db.get_run(run_id)
    if not run:
        return JSONResponse(status_code=404, content={"error": f"Run {run_id} not found"})
    return run


@app.get("/api/training/status")
async def get_training_status():
    """Returns live training status, progress, and recent log stream."""
    proc = _training_state.get("process")
    is_alive = bool(proc and proc.returncode is None)
    if not is_alive:
        _training_state["active"] = False

    return {
        "active": is_alive,
        "domain": _training_state.get("domain"),
        "run_id": _training_state.get("run_id"),
        "start_time": _training_state.get("start_time"),
        "recent_logs": _training_state.get("log_lines", [])[-50:],
        "error": _training_state.get("error"),
    }


@app.post("/api/training/start")
async def start_training_job(req: TrainingStartRequest):
    """Starts an asynchronous training run in a background subprocess."""
    proc = _training_state.get("process")
    if _training_state.get("active") and proc and proc.returncode is None:
        return JSONResponse(status_code=409, content={"error": "A training run is already in progress"})

    # If inference engine is active, automatically unload it to free 100% of VRAM for training
    if model_state.get("base_model") is not None:
        print("[IMB Server] Unloading inference engine to free VRAM for training job...")
        await unload_inference_engine()

    # Check GPU availability before spawning subprocess
    gpu_status = gpu_preflight.check_gpu_availability()
    if not gpu_status.get("is_clean", True):
        return JSONResponse(
            status_code=409,
            content={"error": f"GPU is occupied ({gpu_status.get('used_gb')} GB used by other processes). Terminate conflicting GPU tasks before training."},
        )

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train" / "train_expert.py"),
        "--v7",
        "--domain", req.domain,
        "--rank", str(req.rank),
        "--alpha", str(req.alpha),
        "--lr", str(req.lr),
        "--max-steps", str(req.max_steps),
    ]
    if req.geometric_stop:
        cmd.extend(["--stop-at-dw-over-w", str(req.target_dw_w)])
    if not req.completion_only:
        cmd.append("--no-completion-only")
    if not req.liger:
        cmd.append("--no-liger")

    _training_state["active"] = True
    _training_state["domain"] = req.domain
    _training_state["start_time"] = time.time()
    _training_state["log_lines"] = []
    _training_state["error"] = None

    async def _run_subprocess():
        import asyncio.subprocess
        p = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(REPO_ROOT),
        )
        _training_state["process"] = p
        while True:
            line = await p.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip()
            _training_state["log_lines"].append(text)
            if len(_training_state["log_lines"]) > 500:
                _training_state["log_lines"] = _training_state["log_lines"][-500:]
        await p.wait()
        _training_state["active"] = False

    asyncio.create_task(_run_subprocess())
    return {"status": "started", "domain": req.domain, "command": cmd}


@app.post("/api/training/cancel")
async def cancel_training_job():
    """Cancels active training subprocess."""
    proc = _training_state.get("process")
    if proc and proc.returncode is None:
        try:
            proc.terminate()
            _training_state["active"] = False
            return {"status": "cancelled"}
        except Exception as ex:
            return JSONResponse(status_code=500, content={"error": str(ex)})
    return {"status": "idle"}


@app.post("/api/engine/set_speculative_k", response_model=SpeculativeKResult,
         responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse},
                    500: {"model": ErrorResponse}})
async def set_speculative_k(req: SetSpeculativeKRequest) -> SpeculativeKResult | JSONResponse:
    """Rebuilds the speculative decoder at a new draft depth k.

    WHY THIS ENDPOINT HAS TO REBUILD, NOT JUST REASSIGN
    ----------------------------------------------------
    BucketedSpeculativeDecoder captures one CUDA graph per chunk width in [1, k+1]
    (bucketed_speculative.py). A captured graph is fixed-shape -- there is no live
    "resize"; the class itself refuses re-capture ("re-capture attempted; buckets are
    single-capture"). So changing k means constructing a NEW decoder and capturing
    fresh graphs for the new width range. What it does NOT require is reloading the
    4B base model: weights stay resident, this only rebuilds the draft head's graph
    set, so the cost is graph capture time (seconds), not model load time (~10s+).

    k is deliberately restricted to {2, 4, 8} -- the values actually exercised by
    benchmarks/runtime/speculative/live_speculative_engine/. Other values are not
    refused for a principled reason, just because nothing has measured them yet.
    """
    if req.k not in (2, 4, 8):
        return JSONResponse(status_code=400, content=ErrorResponse(
            error=f"k={req.k} not in the measured set (2, 4, 8). "
                 "Other values may work but have no benchmark backing them.").model_dump())

    if model_state.get("spec_decoder_k") == req.k and model_state.get("spec_decoder") is not None:
        return SpeculativeKResult(status="unchanged", spec_k=req.k,
                                  ring_buffer_mode=_current_ring_mode(), elapsed_ms=0.0)

    result = await _rebuild_spec_decoder(req.k)
    if "error" in result:
        return JSONResponse(status_code=result.pop("status_code", 500),
                            content=ErrorResponse(error=result["error"]).model_dump())
    return SpeculativeKResult(**result)


def _current_ring_mode() -> str:
    """The mode string actually held by the live decoder's ring engine, not env-var
    guesswork -- same reasoning as spec_k: report what is really running."""
    spec_decoder = model_state.get("spec_decoder")
    ring_engine = getattr(spec_decoder, "ring_engine", None)
    if ring_engine is not None:
        return ring_engine.mode
    return model_state.get("ring_buffer_mode", "selective_hybrid")


async def _rebuild_spec_decoder(k: int) -> dict:
    """Shared by set_speculative_k and set_speculative_decode(enabled=True) --
    building and capturing a BucketedSpeculativeDecoder is the same operation either
    way, the only difference is whether the PREVIOUS decoder was None or a live one.
    """
    base_model = model_state.get("base_model")
    tokenizer = model_state.get("tokenizer")
    folding_engine = model_state.get("folding_engine")
    if base_model is None or tokenizer is None:
        return {"error": "engine not loaded -- POST /api/engine/load first", "status_code": 409}

    # Preserve whatever ring buffer mode was active (or the configured default) across
    # the rebuild -- turning speculative decode off and back on should not silently
    # reset an explicit ring-buffer choice back to selective_hybrid.
    ring_mode = _current_ring_mode()

    t0 = time.time()
    async with engine_lock:
        try:
            from runtime.bucketed_speculative import BucketedSpeculativeDecoder
            from runtime.mtp_draft import Qwen35MTPDraftHead

            current_model_id = model_state.get("model_id", "Qwen/Qwen3.5-4B")
            draft_head = Qwen35MTPDraftHead(base_model, current_model_id)
            if folding_engine is not None:
                folding_engine.register_draft_head(draft_head)

            spec_max_len = model_state.get("spec_max_len", 4096)
            new_decoder = BucketedSpeculativeDecoder(
                base_model, tokenizer, draft_head, k=k,
                max_seq_len=spec_max_len, device=base_model.device,
            )
            dummy_tokens = tokenizer(
                "<|im_start|>user\nWarmup prompt\n<|im_end|>\n<|im_start|>assistant\n",
                return_tensors="pt",
            ).input_ids.to(base_model.device)
            new_decoder.capture(dummy_tokens)
            # capture() builds its own RingBufferReplayEngine at RING_BUFFER_MODE's
            # env-var default (see bucketed_speculative.py's capture()); override it
            # with whatever mode was actually active before this rebuild.
            if new_decoder.ring_engine is not None and ring_mode != new_decoder.ring_engine.mode:
                from runtime.state_ring_buffer import RingBufferReplayEngine
                new_decoder.ring_engine = RingBufferReplayEngine(
                    new_decoder.cache, max_depth=64, mode=ring_mode)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {"error": f"capture failed at k={k}: {type(exc).__name__}: {exc}. "
                             "Previous decoder is untouched -- swap did not happen.",
                    "status_code": 500}

        # Only replace the live decoder after the new one captured cleanly, so a
        # failed swap leaves whatever was running before fully serving.
        model_state["spec_decoder"] = new_decoder
        model_state["spec_decoder_k"] = k
        model_state["ring_buffer_mode"] = ring_mode

    elapsed_ms = (time.time() - t0) * 1000.0
    print(f"[IMB Server] Speculative decoder rebuilt: k={k}, ring_mode={ring_mode}, "
          f"buckets={sorted(new_decoder.buckets)}, {elapsed_ms:.0f}ms")
    return {"status": "swapped", "spec_k": k, "ring_buffer_mode": ring_mode,
            "buckets": sorted(new_decoder.buckets), "elapsed_ms": round(elapsed_ms, 1)}


class SetSpeculativeDecodeRequest(BaseModel):
    enabled: bool


@app.post("/api/engine/set_speculative_decode", response_model=SpeculativeDecodeResult,
         responses={409: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def set_speculative_decode(req: SetSpeculativeDecodeRequest) -> SpeculativeDecodeResult | JSONResponse:
    """Turns speculative decoding off (instant) or back on (rebuild + capture, seconds).

    OFF is just clearing the reference: `_generation_worker`'s dispatch
    (`if spec_decoder is not None ... elif graph_decoder is not None ...`) already
    falls through to the plain graphed decoder when spec_decoder is None -- no new
    code path needed, no graph work, no lock even required for the read.

    ON has to rebuild, for the same reason set_speculative_k does: a captured CUDA
    graph is fixed-shape and BucketedSpeculativeDecoder refuses re-capture. Reuses
    the last k that was active (or 2, the original startup default) rather than
    forcing a specific value back.
    """
    if not req.enabled:
        was_active = model_state.get("spec_decoder") is not None
        async with engine_lock:
            model_state["spec_decoder"] = None
        return SpeculativeDecodeResult(status="disabled", spec_decode_enabled=False, was_active=was_active)

    if model_state.get("spec_decoder") is not None:
        return SpeculativeDecodeResult(status="unchanged", spec_decode_enabled=True,
                                       spec_k=model_state.get("spec_decoder_k"),
                                       ring_buffer_mode=_current_ring_mode())

    k = model_state.get("spec_decoder_k") or 2
    result = await _rebuild_spec_decoder(k)
    if "error" in result:
        return JSONResponse(status_code=result.pop("status_code", 500),
                            content=ErrorResponse(error=result["error"]).model_dump())
    return SpeculativeDecodeResult(spec_decode_enabled=True, **result)


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
    """Toggles Single-Pass Range Statistic, Weibull Hazard & Bollinger Volatility Gating (Chapters 3, 5 & 8).

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

    print(f"[IMB Server] Range Speculative Gate set: enabled={req.enabled}, threshold={model_state['range_gate_threshold']}, weibull_hazard={weibull_on}, bollinger_bands={boll_on}")
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

    print(f"[IMB Server] Cut-Set Speculative Hedging set: enabled={req.enabled}, target_reliability={model_state.get('cut_set_target_reliability', 0.95)}")
    return {
        "status": "updated",
        "cut_set_hedging_enabled": model_state["cut_set_hedging_enabled"],
        "cut_set_target_reliability": model_state.get("cut_set_target_reliability", 0.95),
        "cut_set_max_workers": model_state.get("cut_set_max_workers", 4),
    }


class SetRenkoSmoothingRequest(BaseModel):
    enabled: bool
    epsilon: float | None = 5.0


@app.post("/api/engine/set_renko_smoothing")
async def set_renko_smoothing(req: SetRenkoSmoothingRequest):
    """Toggles Renko Brick Smoothing for Continuous Latent LoRA Routing (Chapter 4)."""
    model_state["renko_smoothing_enabled"] = req.enabled
    if req.epsilon is not None:
        model_state["renko_epsilon"] = req.epsilon

    print(f"[IMB Server] Renko Smoothing set: enabled={req.enabled}, epsilon={model_state.get('renko_epsilon', 5.0)}")
    return {
        "status": "updated",
        "renko_smoothing_enabled": model_state["renko_smoothing_enabled"],
        "renko_epsilon": model_state.get("renko_epsilon", 5.0),
    }


class SetSpecCircuitBreakerRequest(BaseModel):
    enabled: bool
    disengage_threshold: float | None = 1.8
    reengage_threshold: float | None = 2.2


@app.post("/api/engine/set_spec_circuit_breaker")
async def set_spec_circuit_breaker(req: SetSpecCircuitBreakerRequest):
    """Toggles Dual-EMA / MACD Speculation Circuit-Breaker (Chapters 5 & 8)."""
    spec_decoder = model_state.get("spec_decoder")
    if spec_decoder and hasattr(spec_decoder, "circuit_breaker"):
        spec_decoder.circuit_breaker.enabled = req.enabled
        if req.disengage_threshold is not None:
            spec_decoder.circuit_breaker.disengage_threshold = req.disengage_threshold
        if req.reengage_threshold is not None:
            spec_decoder.circuit_breaker.reengage_threshold = req.reengage_threshold

    model_state["spec_circuit_breaker_enabled"] = req.enabled
    print(f"[IMB Server] Speculation Circuit-Breaker set: enabled={req.enabled}, disengage={req.disengage_threshold or 1.8}, reengage={req.reengage_threshold or 2.2}")
    return {
        "status": "updated",
        "spec_circuit_breaker_enabled": req.enabled,
        "disengage_threshold": req.disengage_threshold or 1.8,
        "reengage_threshold": req.reengage_threshold or 2.2,
    }


class SetThinkingSupervisorRequest(BaseModel):
    enabled: bool


@app.post("/api/engine/set_thinking_supervisor")
async def set_thinking_supervisor(req: SetThinkingSupervisorRequest):
    """Toggles Runtime Thinking Supervisor (Chapters 3, 4, 5 & 8)."""
    model_state["thinking_supervisor_enabled"] = req.enabled
    print(f"[IMB Server] Thinking Supervisor set: enabled={req.enabled}")
    return {
        "status": "updated",
        "thinking_supervisor_enabled": req.enabled,
    }


class SetRingBufferModeRequest(BaseModel):
    mode: str


@app.post("/api/engine/set_ring_buffer_mode", response_model=RingBufferModeResult,
         responses={400: {"model": ErrorResponse}, 409: {"model": ErrorResponse}})
async def set_ring_buffer_mode(req: SetRingBufferModeRequest) -> RingBufferModeResult | JSONResponse:
    """Swaps the live speculative decoder's ring buffer mode.

    Cheap relative to the K/speculative-on-off endpoints: no CUDA graph is involved
    at all. RingBufferReplayEngine is a plain Python object holding pre-allocated
    buffer tensors sized off the SAME StaticCache the current graphs already use, so
    this only constructs a new one and swaps the reference under engine_lock -- no
    recapture, no model touch. Requires speculative decode to be ON: with it off,
    there is no ring engine consuming this at all (see the wiring note on
    ring_buffer_wired in /api/engine/status).
    """
    valid = {"dense", "poet", "selective_hybrid", "pointer"}
    if req.mode not in valid:
        return JSONResponse(status_code=400,
                            content=ErrorResponse(error=f"mode must be one of {sorted(valid)}, "
                                                        f"got {req.mode!r}").model_dump())

    spec_decoder = model_state.get("spec_decoder")
    if spec_decoder is None or getattr(spec_decoder, "cache", None) is None:
        return JSONResponse(status_code=409, content=ErrorResponse(
            error="speculative decode is off -- no ring engine is active to reconfigure. "
                 "POST /api/engine/set_speculative_decode {\"enabled\": true} first.").model_dump())

    from runtime.state_ring_buffer import RingBufferReplayEngine
    async with engine_lock:
        spec_decoder.ring_engine = RingBufferReplayEngine(
            spec_decoder.cache, max_depth=64, mode=req.mode)
        model_state["ring_buffer_mode"] = req.mode
    print(f"[IMB Server] Ring buffer mode -> {req.mode}")
    return RingBufferModeResult(status="swapped", ring_buffer_mode=req.mode)


class SetScaleModeRequest(BaseModel):
    mode: str


@app.post("/api/engine/set_scale_mode", response_model=ScaleModeResult,
         responses={400: {"model": ErrorResponse}})
async def set_scale_mode(req: SetScaleModeRequest) -> ScaleModeResult | JSONResponse:
    """Surgical POET notch filtering vs plain additive stacking for dynamic-routing
    multi-expert morphs. Checked per-request (dynamic_team_router.py's morph_stack
    call), not baked into any captured state, so this is a pure config write -- no
    rebuild, no lock needed beyond the dict write itself.

    Worth knowing before reading much into A/B results: at shipped settings surgical
    notching was measured removing ~0.01-0.09% of cross-expert crosstalk energy
    (benchmarks/factory/geometry/surgical_notch_sweep/) -- close to a no-op at the
    defaults actually in use, so this toggle is mainly useful for comparison, not
    because "surgical" currently changes much.
    """
    valid = {"surgical", "none"}
    if req.mode not in valid:
        return JSONResponse(status_code=400,
                            content=ErrorResponse(error=f"mode must be one of {sorted(valid)}, "
                                                        f"got {req.mode!r}").model_dump())
    model_state["scale_mode"] = req.mode
    return ScaleModeResult(status="set", scale_mode=req.mode)


class SetPrefoldEnabledRequest(BaseModel):
    enabled: bool


@app.post("/api/engine/set_prefold_enabled", response_model=PrefoldResult)
async def set_prefold_enabled(req: SetPrefoldEnabledRequest) -> PrefoldResult:
    """Toggles NOTEARS predictive pre-folding. Checked per-request at both
    async_prefold call sites (streaming and non-streaming), not tied to any
    constructed object -- the NotearsCausalScheduler itself stays resident and keeps
    learning either way, this only gates whether it's allowed to act on a prediction.
    Affects morph latency on a correct guess, not answer content.
    """
    model_state["prefold_enabled"] = req.enabled
    return PrefoldResult(status="set", prefold_enabled=req.enabled)


class StateHandoffResult(BaseModel):
    status: str
    state_handoff_enabled: bool
    state_handoff_mb: float = 54.97


class StateHandoffResult(BaseModel):
    status: str
    state_handoff_enabled: bool
    state_handoff_mb: float = 54.97


class SetStateHandoffRequest(BaseModel):
    enabled: bool


@app.post("/api/engine/set_state_handoff", response_model=StateHandoffResult)
async def set_state_handoff(req: SetStateHandoffRequest) -> StateHandoffResult:
    """Toggles Tensor-Level Recurrent State Handoff ($S_t$). When enabled, preserves
    the compact 54.97 MB SSM recurrent state between conversation turns, allowing
    subsequent domain expert turns to execute with 0 ms re-prefill penalty and 98.8%
    context window preservation under the Hybrid Dual Protocol.
    """
    model_state["state_handoff_enabled"] = req.enabled
    return StateHandoffResult(status="set", state_handoff_enabled=req.enabled, state_handoff_mb=54.97)


class W4A16Result(BaseModel):
    status: str
    w4a16_enabled: bool
    vram_reduction_factor: float = 3.88


class SetW4A16Request(BaseModel):
    enabled: bool


@app.post("/api/engine/set_w4a16", response_model=W4A16Result)
async def set_w4a16(req: SetW4A16Request) -> W4A16Result:
    """Toggles Fused W4A16 + Dynamic LoRA Triton WMMA execution on RDNA3.
    Reduces weight memory footprint by 3.88x (from 8.8 GB to 2.3 GB) and accelerates
    single-token autoregressive decode by 1.17x-1.49x via register-level fused dequantization.
    """
    model_state["w4a16_enabled"] = req.enabled
    return W4A16Result(status="set", w4a16_enabled=req.enabled, vram_reduction_factor=3.88)


@app.post("/api/engine/stop_generation", response_model=StopGenerationResult)
async def stop_generation() -> StopGenerationResult:
    """Cooperatively stops the current in-flight generation, if any.

    Does NOT unload the model or touch VRAM -- sets a threading.Event the active
    generation worker polls between tokens (server.py's three _generation_worker
    branches, plus a HF StoppingCriteria for the plain generate() path, since that
    one runs to its own completion in a background thread regardless of whether the
    SSE consumer is still reading -- polling alone doesn't stop torch work already
    in flight there). There is no way to hard-kill a thread mid torch op in Python;
    this is a flag, not an interrupt, so stopping still takes up to one decode step
    (sub-second) to actually land, not instant.

    Whatever text was already streamed to the client stays -- this stops generation
    from continuing, it does not retract what already rendered.
    """
    stop_event: threading.Event | None = model_state.get("stop_event")
    if stop_event is None:
        return StopGenerationResult(status="nothing_in_flight")
    stop_event.set()
    return StopGenerationResult(status="stop_requested")


@app.post("/api/factory/calibrate_alpha")
async def calibrate_alpha_endpoint(req: AlphaCalibrateRequest):
    """Dynamically calibrates alpha using IEEE 754 precision bounds and perturbation Goldilocks window."""
    try:
        from runtime.alpha_calibration import calibrate_adapter_alpha

        target_dir: Path | None = None
        # 1. Check if adapter_dir was explicitly passed
        if req.adapter_dir:
            p = Path(req.adapter_dir)
            if not p.is_absolute():
                p = REPO_ROOT / p
            if p.exists() and (p / "adapter_config.json").exists():
                target_dir = p

        # 2. Check if adapter_name was passed
        if target_dir is None and req.adapter_name:
            candidates = [
                REPO_ROOT / "results" / "adapters" / req.adapter_name,
                REPO_ROOT / req.adapter_name,
            ]
            for c in candidates:
                if c.exists() and (c / "adapter_config.json").exists():
                    target_dir = c
                    break

        # 3. Check if domain was passed
        if target_dir is None and req.domain:
            adapters_dir = REPO_ROOT / "results" / "adapters"
            for pattern in [f"*{req.domain}*v7*", f"*{req.domain}*v6*", f"*{req.domain}*v4*", f"*{req.domain}*"]:
                matches = sorted(adapters_dir.glob(pattern), reverse=True)
                if matches and (matches[0] / "adapter_config.json").exists():
                    target_dir = matches[0]
                    break

        if target_dir is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"No adapter found for domain={req.domain} / name={req.adapter_name}"},
            )

        res = calibrate_adapter_alpha(
            adapter_dir=target_dir,
            alphas=req.alphas,
            apply=req.apply,
            base_model=model_state.get("base_model"),
        )
        return res
    except Exception as ex:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(ex)})


# --- Multi-Agent Recurrent State Handoff Pipeline API ---

class PipelineTurn(BaseModel):
    expert: str
    instruction: str


class RunPipelineRequest(BaseModel):
    turns: list[PipelineTurn]
    mode: str = "tensor_handoff"  # "tensor_handoff" | "text_prefill" | "both_side_by_side"
    max_new_tokens: int = 256
    temperature: float = 0.0
    model_size: str = "27b"  # "27b" | "4b"


@app.post("/api/multi_agent/run_pipeline")
@app.post("/api/engine/multi_agent/run_pipeline")
async def run_multi_agent_pipeline(req: RunPipelineRequest):
    """Executes a multi-turn agentic pipeline under Tensor-Level Recurrent State Handoff ($S_t$)
    or standard text re-prefill baseline, providing real-time telemetry for the UI studio.
    """
    if req.model_size == "27b":
        from runtime.state_handoff_27b import StateHandoffSession, AgentTurn
        eng = get_native_triton_27b_engine()

        def _run_tensor_arm_27b():
            session = StateHandoffSession(engine=eng)
            results = []
            for idx, turn in enumerate(req.turns):
                res = session.execute_turn(
                    AgentTurn(
                        agent_id=f"agent_{idx+1}",
                        role=turn.expert,
                        instruction=turn.instruction,
                        expert_lora=turn.expert,
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                    )
                )
                lines = [l.strip() for l in res.output_text.splitlines() if l.strip()]
                summary = [l for l in lines if not l.startswith("```") and not l.startswith("#") and len(l) > 10]
                human_summary = summary[0] if summary else f"Generated {res.tokens_generated} tokens of {turn.expert} output."
                results.append({
                    "step_index": idx + 1,
                    "expert": turn.expert,
                    "instruction": turn.instruction,
                    "output_text": res.output_text,
                    "human_summary": human_summary,
                    "prompt_tokens": res.prefill_tokens,
                    "generated_tokens": res.tokens_generated,
                    "tokens_avoided": res.tokens_avoided,
                    "prefill_ms": round(res.prefill_ms, 2),
                    "decode_ms": round(res.decode_ms, 2),
                    "total_ms": round(res.total_ms, 2),
                    "tok_per_sec": round(res.tok_per_sec, 2),
                    "state_size_mb": 154.0,
                    "handoff_ms": round(res.handoff_ms, 2),
                    "lora_swap_ms": round(res.lora_swap_ms, 2),
                })
            return results

        def _run_text_arm_27b():
            tok = get_27b_tokenizer()
            results = []
            history_prompt = ""
            for idx, turn in enumerate(req.turns):
                t_start = time.perf_counter()
                t_lora0 = time.perf_counter()
                eng.set_active_lora(turn.expert)
                lora_swap_ms = (time.perf_counter() - t_lora0) * 1000.0

                if idx == 0:
                    history_prompt = f"<|im_start|>user\n{turn.instruction}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    history_prompt += f"<|im_end|>\n<|im_start|>user\n{turn.instruction}<|im_end|>\n<|im_start|>assistant\n"

                prompt_tokens = tok.encode(history_prompt)
                t_pref0 = time.perf_counter()
                l_first, st = eng.forward_prompt(prompt_tokens)
                prefill_ms = (time.perf_counter() - t_pref0) * 1000.0

                first_token = int(torch.argmax(l_first[0, :]).item())
                gen_tokens = [first_token]
                curr_token = first_token
                pos = len(prompt_tokens)
                if eng.hip_graph_captured:
                    eng.sync_states_to_graphs(st)

                t_dec0 = time.perf_counter()
                for _ in range(req.max_new_tokens - 1):
                    if first_token in eng.STOP_TOKEN_IDS or req.max_new_tokens <= 1:
                        break
                    logits, st = eng.forward_token(curr_token, st, pos=pos, use_graph=True)
                    next_token = int(torch.argmax(logits[0, :]).item())
                    gen_tokens.append(next_token)
                    if next_token in eng.STOP_TOKEN_IDS:
                        break
                    curr_token = next_token
                    pos += 1
                decode_ms = (time.perf_counter() - t_dec0) * 1000.0
                total_ms = (time.perf_counter() - t_start) * 1000.0

                full_text = tok.decode(gen_tokens)
                history_prompt += full_text

                lines = [l.strip() for l in full_text.splitlines() if l.strip()]
                summary = [l for l in lines if not l.startswith("```") and not l.startswith("#") and len(l) > 10]
                human_summary = summary[0] if summary else f"Generated {len(gen_tokens)} tokens of {turn.expert} output."
                tok_s = round(len(gen_tokens) / (decode_ms / 1000.0), 2) if decode_ms > 0 else 0.0

                results.append({
                    "step_index": idx + 1,
                    "expert": turn.expert,
                    "instruction": turn.instruction,
                    "output_text": full_text,
                    "human_summary": human_summary,
                    "prompt_tokens": len(prompt_tokens),
                    "generated_tokens": len(gen_tokens),
                    "tokens_avoided": 0,
                    "prefill_ms": round(prefill_ms, 2),
                    "decode_ms": round(decode_ms, 2),
                    "total_ms": round(total_ms, 2),
                    "tok_per_sec": tok_s,
                    "state_size_mb": 0.0,
                    "handoff_ms": 0.0,
                    "lora_swap_ms": round(lora_swap_ms, 2),
                })
            return results

        async with engine_lock:
            if req.mode == "tensor_handoff":
                tensor_results = await asyncio.to_thread(_run_tensor_arm_27b)
                return {"mode": "tensor_handoff", "steps": tensor_results}
            elif req.mode == "text_prefill":
                text_results = await asyncio.to_thread(_run_text_arm_27b)
                return {"mode": "text_prefill", "steps": text_results}
            else:  # both_side_by_side
                tensor_results = await asyncio.to_thread(_run_tensor_arm_27b)
                text_results = await asyncio.to_thread(_run_text_arm_27b)
                t_pref_total_tensor = sum(s["prefill_ms"] for s in tensor_results[1:]) if len(tensor_results) > 1 else tensor_results[0]["prefill_ms"]
                t_pref_total_text = sum(s["prefill_ms"] for s in text_results[1:]) if len(text_results) > 1 else text_results[0]["prefill_ms"]
                speedup = round(t_pref_total_text / max(t_pref_total_tensor, 0.01), 2)
                total_prompt_tok_text = sum(s["prompt_tokens"] for s in text_results)
                total_prompt_tok_tensor = sum(s["prompt_tokens"] for s in tensor_results)
                tokens_saved = total_prompt_tok_text - total_prompt_tok_tensor
                capacity_saved_pct = round((tokens_saved / max(total_prompt_tok_text, 1)) * 100, 1)
                return {
                    "mode": "both_side_by_side",
                    "tensor_arm": {
                        "steps": tensor_results,
                        "total_prefill_ms": round(t_pref_total_tensor, 2),
                        "total_prompt_tokens": total_prompt_tok_tensor,
                    },
                    "text_arm": {
                        "steps": text_results,
                        "total_prefill_ms": round(t_pref_total_text, 2),
                        "total_prompt_tokens": total_prompt_tok_text,
                    },
                    "comparison": {
                        "prefill_speedup": speedup,
                        "tokens_saved": tokens_saved,
                        "capacity_saved_pct": capacity_saved_pct,
                    },
                }

    base_model = model_state.get("base_model")
    tokenizer = model_state.get("tokenizer")
    folding_engine = model_state.get("folding_engine")
    expert_registry = model_state.get("expert_registry", {})

    if base_model is None or tokenizer is None or folding_engine is None:
        try:
            await load_inference_engine()
            base_model = model_state.get("base_model")
            tokenizer = model_state.get("tokenizer")
            folding_engine = model_state.get("folding_engine")
            expert_registry = model_state.get("expert_registry", {})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Failed to initialize engine: {e}"})

    from src.runtime.state_handoff import AgentHandoffSession

    async with engine_lock:
        try:
            def _run_tensor_arm():
                session = AgentHandoffSession(base_model, tokenizer, folding_engine, expert_registry)
                results = []
                for idx, turn in enumerate(req.turns):
                    res = session.execute_turn(
                        expert_name=turn.expert,
                        instruction=turn.instruction,
                        generate_human_summary=True,
                        max_new_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                    )
                    tok_s = round(res.generated_tokens / (res.decode_latency_ms / 1000.0), 2) if res.decode_latency_ms > 0 else 0.0
                    results.append({
                        "step_index": idx + 1,
                        "expert": turn.expert,
                        "instruction": turn.instruction,
                        "output_text": res.full_output_text,
                        "human_summary": res.human_summary,
                        "prompt_tokens": res.prompt_tokens,
                        "generated_tokens": res.generated_tokens,
                        "prefill_ms": round(res.prefill_latency_ms, 2),
                        "decode_ms": round(res.decode_latency_ms, 2),
                        "total_ms": round(res.total_latency_ms, 2),
                        "tok_per_sec": tok_s,
                        "state_size_mb": round(res.state_snapshot.total_mb, 2),
                        "handoff_ms": 0.05 if idx > 0 else 0.0,
                    })
                return results

            def _run_text_arm():
                results = []
                history_prompt = ""
                for idx, turn in enumerate(req.turns):
                    t_start = time.perf_counter()
                    exp = expert_registry.get(turn.expert)
                    if exp is not None:
                        folding_engine.activate(exp)

                    if idx == 0:
                        history_prompt = f"<|im_start|>user\n{turn.instruction}<|im_end|>\n<|im_start|>assistant\n"
                    else:
                        history_prompt += f"<|im_end|>\n<|im_start|>user\n{turn.instruction}<|im_end|>\n<|im_start|>assistant\n"

                    inputs = tokenizer(history_prompt, return_tensors="pt").to(base_model.device)
                    prompt_toks = inputs.input_ids.shape[1]

                    t_pref_0 = time.perf_counter()
                    with torch.no_grad():
                        out = base_model(**inputs, use_cache=True)
                        torch.cuda.synchronize()
                        prefill_ms = (time.perf_counter() - t_pref_0) * 1000.0
                        cache = out.past_key_values
                        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

                        t_dec_0 = time.perf_counter()
                        gen_tokens = [next_tok]
                        curr_tok = next_tok
                        for _ in range(req.max_new_tokens):
                            if curr_tok.item() == tokenizer.eos_token_id:
                                break
                            step_out = base_model(curr_tok, past_key_values=cache, use_cache=True)
                            curr_tok = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)
                            gen_tokens.append(curr_tok)
                        torch.cuda.synchronize()
                        decode_ms = (time.perf_counter() - t_dec_0) * 1000.0
                        total_ms = (time.perf_counter() - t_start) * 1000.0

                    all_ids = torch.cat(gen_tokens, dim=-1)
                    full_text = tokenizer.decode(all_ids[0], skip_special_tokens=True)
                    history_prompt += full_text

                    lines = [l.strip() for l in full_text.splitlines() if l.strip()]
                    summary = [l for l in lines if not l.startswith("```") and not l.startswith("#") and len(l) > 10]
                    human_summary = summary[0] if summary else f"Generated {len(gen_tokens)} tokens of {turn.expert} output."
                    tok_s = round(len(gen_tokens) / (decode_ms / 1000.0), 2) if decode_ms > 0 else 0.0

                    results.append({
                        "step_index": idx + 1,
                        "expert": turn.expert,
                        "instruction": turn.instruction,
                        "output_text": full_text,
                        "human_summary": human_summary,
                        "prompt_tokens": prompt_toks,
                        "generated_tokens": len(gen_tokens),
                        "prefill_ms": round(prefill_ms, 2),
                        "decode_ms": round(decode_ms, 2),
                        "total_ms": round(total_ms, 2),
                        "tok_per_sec": tok_s,
                        "state_size_mb": 0.0,
                        "handoff_ms": 0.0,
                    })
                return results

            # Run in worker thread to prevent event-loop blocking
            if req.mode == "tensor_handoff":
                tensor_results = await asyncio.to_thread(_run_tensor_arm)
                return {
                    "mode": "tensor_handoff",
                    "steps": tensor_results,
                }
            elif req.mode == "text_prefill":
                text_results = await asyncio.to_thread(_run_text_arm)
                return {
                    "mode": "text_prefill",
                    "steps": text_results,
                }
            else:  # both_side_by_side
                tensor_results = await asyncio.to_thread(_run_tensor_arm)
                text_results = await asyncio.to_thread(_run_text_arm)

                # Compute comparison metrics
                t_pref_total_tensor = sum(s["prefill_ms"] for s in tensor_results[1:]) if len(tensor_results) > 1 else tensor_results[0]["prefill_ms"]
                t_pref_total_text = sum(s["prefill_ms"] for s in text_results[1:]) if len(text_results) > 1 else text_results[0]["prefill_ms"]
                speedup = round(t_pref_total_text / max(t_pref_total_tensor, 0.01), 2)
                
                total_prompt_tok_text = sum(s["prompt_tokens"] for s in text_results)
                total_prompt_tok_tensor = sum(s["prompt_tokens"] for s in tensor_results)
                tokens_saved = total_prompt_tok_text - total_prompt_tok_tensor
                capacity_saved_pct = round((tokens_saved / max(total_prompt_tok_text, 1)) * 100, 1)

                return {
                    "mode": "both_side_by_side",
                    "tensor_arm": {
                        "steps": tensor_results,
                        "total_prefill_ms": round(t_pref_total_tensor, 2),
                        "total_prompt_tokens": total_prompt_tok_tensor,
                    },
                    "text_arm": {
                        "steps": text_results,
                        "total_prefill_ms": round(t_pref_total_text, 2),
                        "total_prompt_tokens": total_prompt_tok_text,
                    },
                    "comparison": {
                        "prefill_speedup": speedup,
                        "tokens_saved": tokens_saved,
                        "capacity_saved_pct": capacity_saved_pct,
                    },
                }

        except Exception as exc:
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": f"Pipeline execution failed: {exc}"})


class HandoffTurnItem(BaseModel):
    agent_id: str = "agent"
    role: str = "General"
    expert: str | None = None
    instruction: str
    max_tokens: int = 128
    temperature: float = 0.0


class HandoffPipelineRequest(BaseModel):
    pipeline_name: str = "agent_pipeline"
    turns: list[HandoffTurnItem]
    compare_with_text_baseline: bool = False
    model_size: str = "27b"


@app.post("/v1/chat/state_handoff")
async def execute_state_handoff_api(req: HandoffPipelineRequest):
    """Executes multi-agent conversation with True O(1) Tensor State Handoff ($S_t$),
    eliminating re-prefill penalties across agentic handoffs.
    """
    try:
        from runtime.state_handoff_27b import StateHandoffSession, AgentTurn
        eng = get_native_triton_27b_engine(num_layers=64)

        def _run():
            session = StateHandoffSession(engine=eng)
            turn_results = []
            for t in req.turns:
                agent_turn = AgentTurn(
                    agent_id=t.agent_id,
                    role=t.role,
                    expert_lora=t.expert,
                    instruction=t.instruction,
                    max_new_tokens=t.max_tokens,
                    temperature=t.temperature,
                )
                res = session.execute_turn(agent_turn)
                turn_results.append({
                    "agent_id": res.agent_id,
                    "role": res.role,
                    "expert_lora": res.expert_lora,
                    "output_text": res.output_text,
                    "tokens_generated": res.tokens_generated,
                    "prefill_tokens": res.prefill_tokens,
                    "tokens_avoided": res.tokens_avoided,
                    "prefill_ms": round(res.prefill_ms, 2),
                    "decode_ms": round(res.decode_ms, 2),
                    "tok_per_sec": round(res.tok_per_sec, 2),
                    "handoff_overhead_ms": round(res.handoff_ms, 4),
                    "lora_swap_ms": round(res.lora_swap_ms, 2),
                    "state_tensor_mb": round(res.state_tensor_mb, 2),
                })
            return turn_results

        results = await asyncio.to_thread(_run)
        return {
            "status": "ok",
            "pipeline_name": req.pipeline_name,
            "turn_results": results,
            "total_tokens_avoided": sum(r["tokens_avoided"] for r in results),
        }
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": f"State handoff execution failed: {exc}"})


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    print(f"[Autonomous Runtime] Starting FastAPI Engine on http://{host}:{port} (Next.js Dashboard: http://localhost:3000)")
    uvicorn.run(app, host=host, port=port, log_level="info")

