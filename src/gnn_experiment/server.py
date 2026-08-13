"""OpenAI-compatible FastAPI server for the in-place weight-folding engine.

Serves local micro-experts via standard OpenAI REST API endpoints (/v1/chat/completions, /v1/models):
1. Intercepts requested model parameter ("financial_planning", "postgresql", "astral").
2. Triggers on-device in-place weight mutation (W_live = W0 + s * U@V) at static VRAM addresses.
3. Executes single-token decode via pre-captured CUDA/HIP Graph descriptor at 32.89 tok/s.
4. Supports non-streaming JSON responses and streaming Server-Sent Events (SSE text/event-stream).
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from gnn_experiment.cuda_graph import FoldedCudaGraphDecoder
from gnn_experiment.fused_norm import inject_exact_rmsnorm
from gnn_experiment.novel_peft import FoldableExpert, WeightFoldingEngine, set_hard_vram_cap

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Global State Containers ---
model_state: dict[str, Any] = {}
engine_lock = asyncio.Lock()


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
    max_tokens: int | None = 64
    stream: bool | None = False


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str
    prompt: str
    temperature: float | None = 0.7
    max_tokens: int | None = 64
    stream: bool | None = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


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
    return str(content)


def resolve_expert(model_name: str) -> FoldableExpert | None:
    """Maps request model string to loaded FoldableExpert instance.

    Strips provider prefixes (e.g. openai/postgresql -> postgresql).
    """
    name_clean = model_name.split("/")[-1].lower().strip()
    registry = model_state.get("expert_registry", {})
    if name_clean in registry:
        return registry[name_clean]
    for key, expert in registry.items():
        if key in name_clean or name_clean in key:
            return expert
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

    # Load factor experts into host memory
    # Alpha-sweep winners. Each is the best of five alphas measured against base
    # under the corrected (stop_strings) harness -- see README "Measured Findings".
    # The peak is domain-specific, so these are NOT all the same alpha:
    #     astral      a64  60.20% vs base 12.20%  (+47.99pp)
    #     postgresql  a64  74.67% vs base 49.67%  (+25.00pp)
    #     financial   a32  83.33% vs base 78.33%   (+5.00pp)
    # The previous financial adapter (financial_planning_krona_dora) was trained on
    # a dataset of 940 copies of ONE templated prompt and scored 33.3% -- below base.
    financial_dir = REPO_ROOT / "results" / "adapters" / "fin_sweep_a32"
    postgres_dir = REPO_ROOT / "results" / "adapters" / "pg_sweep_a64"
    astral_dir = REPO_ROOT / "results" / "adapters" / "astral_sweep_a64"

    exp_fin = FoldableExpert.from_dir(financial_dir, "financial_planning")
    exp_pg = FoldableExpert.from_dir(postgres_dir, "postgresql")
    exp_astral = FoldableExpert.from_dir(astral_dir, "astral")

    folding_engine = WeightFoldingEngine(base_model, [exp_fin, exp_pg, exp_astral], keep_pristine=True)

    expert_registry = {
        "base": None,
        "qwen3.5": None,
        "financial_planning": exp_fin,
        "financial": exp_fin,
        "fin": exp_fin,
        "postgresql": exp_pg,
        "postgres": exp_pg,
        "astral": exp_astral,
    }

    # Warmup & Capture CUDA Graph ONCE
    # Default to 8192 tokens (8K context) taking ~11.1 GB VRAM total.
    # Can be overridden via MAX_SEQ_LEN env var (e.g. MAX_SEQ_LEN=4096 or 16384 or 32768)
    import os

    env_max_len = os.environ.get("MAX_SEQ_LEN")
    if env_max_len:
        max_seq_len = int(env_max_len)
    else:
        text_config = getattr(base_model.config, "text_config", base_model.config)
        max_seq_len = getattr(text_config, "max_position_embeddings", 8192)
        max_seq_len = min(max_seq_len, 8192)

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

    yield

    print("[IMB Server] Shutting down. Restoring pristine W0 base weights...")
    folding_engine.restore()
    model_state.clear()

    # Force Garbage Collection & Clear GPU VRAM Cache
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
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
    registry = model_state.get("expert_registry", {})
    unique_models = list(dict.fromkeys(registry.keys()))
    prefixed_models = [f"openai/{m}" for m in unique_models]
    all_models = unique_models + prefixed_models
    model_objects = [ModelObject(id=m) for m in all_models]
    return ModelListResponse(data=model_objects)


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    tokenizer = model_state["tokenizer"]
    base_model = model_state["base_model"]
    folding_engine = model_state["folding_engine"]
    graph_decoder = model_state["graph_decoder"]
    max_prompt_len = model_state["max_prompt_len"]

    print(f"[IMB Server] Request: model='{req.model}', msgs={len(req.messages)}, stream={req.stream}")

    expert = resolve_expert(req.model)
    prompt_text = format_prompt(req.messages)

    # Encode prompt exact without artificial padding to preserve full context window
    prompt_tokens = tokenizer(
        prompt_text,
        return_tensors="pt",
        max_length=max_prompt_len,
        truncation=True,
    ).input_ids.to(base_model.device)

    max_new_tokens = req.max_tokens or 512

    if req.stream:

        async def sse_generator() -> AsyncGenerator[str]:
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

            # Initial role chunk
            initial_chunk = ChatCompletionChunkResponse(
                id=chunk_id,
                model=req.model,
                choices=[ChatCompletionChunkChoice(index=0, delta=ChatCompletionChunkDelta(role="assistant"))],
            )
            yield f"data: {json.dumps(initial_chunk.model_dump())}\n\n"

            in_thinking = False

            async with engine_lock:
                for token_piece in graph_decoder.generate_tokens_stream(
                    prompt_tokens, engine=folding_engine, expert=expert, max_new_tokens=max_new_tokens
                ):
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

            # Final stop chunk
            final_chunk = ChatCompletionChunkResponse(
                id=chunk_id,
                model=req.model,
                choices=[ChatCompletionChunkChoice(index=0, delta=ChatCompletionChunkDelta(), finish_reason="stop")],
            )
            yield f"data: {json.dumps(final_chunk.model_dump())}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    # Non-streaming Response
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

    return ChatCompletionResponse(
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
        ),
    )


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
