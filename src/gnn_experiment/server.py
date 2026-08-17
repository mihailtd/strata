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

# Automatically ensure ROCm HSA runtime is preloaded for AMD Radeon RX 7900 XTX
rocm_hsa_lib = "/opt/rocm-7.2.0/lib/libhsa-runtime64.so"
if os.path.exists(rocm_hsa_lib) and rocm_hsa_lib not in os.environ.get("LD_PRELOAD", ""):
    current_preload = os.environ.get("LD_PRELOAD", "")
    os.environ["LD_PRELOAD"] = f"{rocm_hsa_lib}:{current_preload}".strip(":")
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from gnn_experiment.cuda_graph import FoldedCudaGraphDecoder
from gnn_experiment.dashboard import DASHBOARD_HTML
from gnn_experiment.fused_norm import (
    fold_rmsnorm_into_linear,
    inject_exact_rmsnorm,
    scale_expert_factors_for_folded_norms,
)
from gnn_experiment.novel_peft import FoldableExpert, WeightFoldingEngine, set_hard_vram_cap
from gnn_experiment.router.vram_state_router import VRAMState

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# How long the dispatch loop will wait for a streaming response to be consumed
# before moving on. Bounds the damage from a client that disconnects mid-stream.
STREAM_ORDER_TIMEOUT_S = float(os.environ.get("STREAM_ORDER_TIMEOUT_S", "300"))

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


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    prompt: str
    temperature: float | None = 0.7
    max_tokens: int | None = 4096
    stream: bool | None = False


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


def resolve_expert(model_name: str) -> FoldableExpert | None:
    """Maps request model string to loaded FoldableExpert instance.

    Handles curated names (qwen3.5-4b-*) as well as friendly domain aliases.
    """
    name_clean = model_name.split("/")[-1].lower().strip()
    registry = model_state.get("expert_registry", {})

    # 1. Base / Pristine Model Check (returns None so folding_engine.restore() is called)
    if any(k in name_clean for k in ["base", "pristine", "default"]) or name_clean in ("qwen3.5", "qwen3.5-4b"):
        return None

    # 2. Direct exact match in registry
    if name_clean in registry:
        return registry[name_clean]

    # 3. Domain keyword resolution
    if any(k in name_clean for k in ["astral", "python", "uv", "ruff"]):
        return registry.get("astral")
    if any(k in name_clean for k in ["postgre", "postgres", "sql", "db"]):
        return registry.get("postgresql")
    if any(k in name_clean for k in ["fin", "wealth"]):
        return registry.get("financial_planning")

    return None


def format_prompt(messages: list[ChatMessage]) -> str:
    """Formats ChatMessage array into standard Qwen 3.5 ChatML instruction format."""
    formatted = ""
    for msg in messages:
        msg_content = extract_msg_content(msg.content)
        formatted += f"<|im_start|>{msg.role}\n{msg_content}\n<|im_end|>\n"
    formatted += "<|im_start|>assistant\n"
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
            return reasoning if reasoning else None, content
        else:
            return rest.strip(), prefix.strip()
    return None, text.strip()


# --- Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: initializes GPU base model, experts, AITER ops, and CUDA Graph once."""
    vram_cap_gb = 22.0
    set_hard_vram_cap(vram_cap_gb)

    model_id = "Qwen/Qwen3.5-4B"
    print(f"[IMB Server] Initializing Base Model ({model_id})...")

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=compute_dtype,
        device_map={"": 0} if torch.cuda.is_available() else "auto",
        trust_remote_code=True,
    )
    base_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Exact PyTorch RMSNorm stand-ins (CUDA-graph friendly). NOT a speedup --
    # measured slower-or-equal to ATen; AITER's kernels are unusable here
    # (silently return zeros at hidden=2560 on gfx1100). See fused_norm.py.
    injected_count = inject_exact_rmsnorm(base_model)
    print(f"[IMB Server] Injected {injected_count} ExactRMSNorm modules.")

    # FlashNorm-style weight folding: eliminate RMSNorm kernel launch overhead by
    # folding layer scales into downstream Linear projection weights.
    fold_norms_enabled = os.environ.get("FLASH_NORM_FOLD", "1") != "0"
    folded_norm_count = fold_rmsnorm_into_linear(base_model, fold_weights=fold_norms_enabled)
    if folded_norm_count > 0:
        print(f"[IMB Server] FlashNorm: Folded {folded_norm_count} RMSNorm scale weights into downstream Linears.")

    # Load factor experts into host memory
    # Alpha-sweep winners. Each is the best of five alphas measured against base
    # under the corrected (stop_strings) harness -- see README "Measured Findings".
    # The peak is domain-specific, so these are NOT all the same alpha:
    #     astral      a64  60.20% vs base 12.20%  (+47.99pp)
    #     postgresql  a64  74.67% vs base 49.67%  (+25.00pp)
    #     financial   a32  83.33% vs base 78.33%   (+5.00pp)
    # The previous financial adapter (financial_planning_krona_dora) was trained on
    # a dataset of 940 copies of ONE templated prompt and scored 33.3% -- below base.
    financial_dir = REPO_ROOT / "results" / "adapters" / "m2_financial_r8a128"
    postgres_dir = REPO_ROOT / "results" / "adapters" / "m2_postgresql_r8a128"
    astral_dir = REPO_ROOT / "results" / "adapters" / "m2_astral_r8a128"

    exp_fin = FoldableExpert.from_dir(financial_dir, "financial_planning")
    exp_pg = FoldableExpert.from_dir(postgres_dir, "postgresql")
    exp_astral = FoldableExpert.from_dir(astral_dir, "astral")

    if folded_norm_count > 0:
        scaled_factors = scale_expert_factors_for_folded_norms(base_model, [exp_fin, exp_pg, exp_astral])
        print(f"[IMB Server] FlashNorm: Scaled {scaled_factors} adapter factors by (1+γ) for exact norm alignment.")

    folding_engine = WeightFoldingEngine(base_model, [exp_fin, exp_pg, exp_astral], keep_pristine=True)

    expert_registry = {
        "qwen3.5-4b-base": None,
        "qwen3.5-4b-astral": exp_astral,
        "qwen3.5-4b-postgresql": exp_pg,
        "qwen3.5-4b-financial": exp_fin,
        "base": None,
        "astral": exp_astral,
        "postgresql": exp_pg,
        "financial_planning": exp_fin,
    }

    # No scheduler is constructed. Expert state is tracked as a plain VRAMState
    # (see _record_transition); the measured transition-cost model lives on in
    # router/vram_state_router.py as documented, tested physics -- it is what
    # retired the APSP router -- but nothing in the serving path consumes it now
    # that requests are not reordered. See docs/DECISIONS.md §6.

    # Warmup & Capture CUDA Graph ONCE
    # Default to 32768 tokens (32K context) taking ~11.8 GB VRAM total.
    # Can be overridden via MAX_SEQ_LEN env var (e.g. MAX_SEQ_LEN=4096 or 16384 or 32768)

    env_max_len = os.environ.get("MAX_SEQ_LEN")
    if env_max_len:
        max_seq_len = int(env_max_len)
    else:
        text_config = getattr(base_model.config, "text_config", base_model.config)
        max_seq_len = getattr(text_config, "max_position_embeddings", 32768)
        max_seq_len = min(max_seq_len, 32768)

    folding_engine.activate(exp_fin)
    graph_decoder = FoldedCudaGraphDecoder(base_model, tokenizer, max_seq_len=max_seq_len, device=base_model.device)

    dummy_tokens = tokenizer(
        "<|im_start|>user\nWarmup prompt\n<|im_end|>\n<|im_start|>assistant\n",
        return_tensors="pt",
    ).input_ids.to(base_model.device)

    print(f"[IMB Server] Capturing CUDA Graph ONCE (max_seq_len={max_seq_len})...")
    graph_decoder.capture(dummy_tokens)
    print(f"[IMB Server] CUDA Graph captured successfully. Capture Count = {graph_decoder.capture_count}")

    model_state["base_model"] = base_model
    model_state["tokenizer"] = tokenizer
    model_state["folding_engine"] = folding_engine
    model_state["graph_decoder"] = graph_decoder
    model_state["expert_registry"] = expert_registry
    model_state["max_prompt_len"] = max_seq_len
    model_state["gpu_state"] = VRAMState.single("financial_planning")  # matches final activate above

    # Start the dispatch loop as a background task
    dispatch_task = asyncio.create_task(_dispatch_loop())
    print("[IMB Server] Sequential executor started (single-tenant, arrival order).")

    yield

    # Cancel dispatch loop
    dispatch_task.cancel()
    try:
        await dispatch_task
    except asyncio.CancelledError:
        pass

    print("[IMB Server] Shutting down. Restoring pristine W0 base weights...")
    folding_engine.restore()
    model_state.clear()

    # Force Garbage Collection & Clear GPU VRAM Cache
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[IMB Server] Restored pristine W0 base weights and cleaned up VRAM memory successfully.")


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
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
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
    tokenizer = model_state["tokenizer"]
    base_model = model_state["base_model"]
    folding_engine = model_state["folding_engine"]
    graph_decoder = model_state["graph_decoder"]
    max_prompt_len = model_state["max_prompt_len"]

    expert = qr.expert
    prompt_text = format_prompt(req.messages)

    prompt_tokens = tokenizer(
        prompt_text,
        return_tensors="pt",
        max_length=max_prompt_len,
        truncation=True,
    ).input_ids.to(base_model.device)

    max_new_tokens = req.max_completion_tokens or req.max_tokens or 4096
    wait_ms = (time.perf_counter() - qr.enqueue_time) * 1000.0
    target_state = VRAMState.from_expert(expert)

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
        output_tokens, elapsed, tok_s, swap_ms = await asyncio.to_thread(
            graph_decoder.generate_with_graph,
            prompt_tokens,
            engine=folding_engine,
            expert=expert,
            max_new_tokens=max_new_tokens,
        )

    raw_response_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
    reasoning_text, content_text = extract_thinking_and_content(raw_response_text)

    prompt_num_toks = prompt_tokens.shape[1]
    completion_num_toks = len(output_tokens)

    update_telemetry(req.model, prompt_num_toks, completion_num_toks, elapsed, tok_s, swap_ms, swap_ms)

    print(
        f"[IMB Telemetry] model='{req.model}' | {completion_num_toks} toks in {elapsed:.2f}s "
        f"({tok_s:.2f} tok/s) | swap: {swap_ms:.2f}ms | router wait: {wait_ms:.1f}ms"
    )

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
    """Builds a StreamingResponse for SSE streaming requests."""
    graph_decoder = model_state["graph_decoder"]
    folding_engine = model_state["folding_engine"]

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

        in_thinking = False

        # The fold happens here, not when the dispatch loop handed this back.
        _record_transition(VRAMState.from_expert(expert))

        async with engine_lock:
            for token_piece in graph_decoder.generate_tokens_stream(
                prompt_tokens, engine=folding_engine, expert=expert, max_new_tokens=max_new_tokens
            ):
                token_count += 1
                if t_first_token is None:
                    t_first_token = time.perf_counter()

                if "<think>" in token_piece:
                    in_thinking = True
                    token_piece = token_piece.replace("<think>", "")

                if in_thinking:
                    if "</think>" in token_piece:
                        in_thinking = False
                        think_part, main_part = token_piece.split("</think>", 1)
                        if think_part:
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
                        if token_piece:
                            chunk = ChatCompletionChunkResponse(
                                id=chunk_id,
                                model=req.model,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0, delta=ChatCompletionChunkDelta(reasoning_content=token_piece)
                                    )
                                ],
                            )
                            yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                else:
                    if token_piece:
                        chunk = ChatCompletionChunkResponse(
                            id=chunk_id,
                            model=req.model,
                            choices=[
                                ChatCompletionChunkChoice(
                                    index=0, delta=ChatCompletionChunkDelta(content=token_piece)
                                )
                            ],
                        )
                        yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                await asyncio.sleep(0)  # Yield ASGI event loop frame

        t_stream_end = time.perf_counter()
        elapsed_s = t_stream_end - t_stream_start
        ttft_ms = ((t_first_token - t_stream_start) * 1000.0) if t_first_token else 0.0
        tok_s = token_count / elapsed_s if elapsed_s > 0 else 0.0

        prompt_num_toks = prompt_tokens.shape[1]
        update_telemetry(req.model, prompt_num_toks, token_count, elapsed_s, tok_s, ttft_ms, 0.0)

        print(
            f"[IMB Telemetry] model='{req.model}' | {token_count} toks in {elapsed_s:.2f}s "
            f"({tok_s:.2f} tok/s) | TTFT: {ttft_ms:.1f}ms | router wait: {wait_ms:.1f}ms"
        )

        usage_info = UsageInfo(
            prompt_tokens=prompt_num_toks,
            completion_tokens=token_count,
            total_tokens=prompt_num_toks + token_count,
            tokens_per_second=round(tok_s, 2),
            generation_time_ms=round(elapsed_s * 1000.0, 1),
            time_to_first_token_ms=round(ttft_ms, 1),
            swap_time_ms=0.0,
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
        "X-Router-GPU-State": VRAMState.from_expert(expert).name,
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

    return {
        "status": "online",
        "runtime": "Autonomous In-Place Weight-Folding & VRAM State Router Engine",
        "live_metrics": {
            "average_tokens_per_second": round(avg_tok_s, 2),
            "total_requests_served": server_telemetry["total_requests"],
            "total_tokens_generated": server_telemetry["total_generated_tokens"],
            "total_generation_time_seconds": round(server_telemetry["total_generation_time_s"], 2),
        },
        "last_request": server_telemetry["last_request"],
        "hardware": {
            "device": "AMD ROCm GPU (gfx1100)",
            "active_expert": active_expert_name,
            "vram_allocated_gb": round(vram_alloc, 2),
            "vram_reserved_gb": round(vram_res, 2),
            "vram_hard_cap_gb": 22.0,
        },
        "cuda_graph": {
            "locked": model_state.get("graph_decoder")._is_locked if "graph_decoder" in model_state else False,
            "capture_count": model_state.get("graph_decoder").capture_count if "graph_decoder" in model_state else 0,
            "max_seq_len": model_state.get("max_prompt_len", 32768),
        },
        "router": router_telemetry,
    }


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the real-time dark-mode GPU telemetry and VRAM state router dashboard."""
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/events")
async def sse_telemetry_feed() -> StreamingResponse:
    """Streams real-time server telemetry and hardware stats via Server-Sent Events (SSE)."""

    async def event_publisher() -> AsyncGenerator[str]:
        while True:
            stats = await get_server_stats()
            yield f"data: {json.dumps(stats)}\n\n"
            await asyncio.sleep(1.0)

    return StreamingResponse(event_publisher(), media_type="text/event-stream")
