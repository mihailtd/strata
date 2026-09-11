"""Thin OpenAI-compatible client for the local llama-server instance."""

import asyncio
import contextlib
import json
import os
import random
import signal
import subprocess
import time

import httpx

from runtime.datagen.schema import ChatResult


def stop_llama_server(port: int, grace_s: float = 5.0) -> bool:
    """Finds and stops the llama-server process bound to `port` (SIGTERM,
    then SIGKILL after grace_s if it hasn't exited) — frees its VRAM.
    Returns True if a process was found and signaled, False if none was
    running. Matches by port in the command line, so it only touches the
    server we were actually talking to, not other llama-server instances.
    """
    pattern = rf"llama-server.*--port\s+{port}\b"

    def _find_pids() -> list[int]:
        result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        return [int(p) for p in result.stdout.split() if p.strip()]

    pids = _find_pids()
    if not pids:
        return False

    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)

    deadline = time.time() + grace_s
    while time.time() < deadline:
        if not _find_pids():
            return True
        time.sleep(0.5)

    for pid in _find_pids():
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)
    return True


def _sse_payload(line: str) -> dict | None:
    """Parses one `data: {...}` SSE line into its JSON chunk. Returns None
    for lines to skip -- blank keepalive lines and the terminal `[DONE]`."""
    if not line.startswith("data: "):
        return None
    payload = line[len("data: ") :]
    if payload.strip() == "[DONE]":
        return None
    return json.loads(payload)


class LocalLLMClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model: str = "models/Qwen3.5-4B-Q8_0.gguf",
        timeout: float = 120.0,
    ):
        self.model = model
        # max_keepalive_connections=0: don't pool idle connections for reuse.
        # Measured directly against this server: a shared keep-alive pool
        # reproducibly hits "Server disconnected without sending a response"
        # on ~25-40% of requests (confirmed via llama-server's own --log-file
        # showing zero errors/disconnects server-side for the exact same
        # run -- every task it received completed cleanly, so the failure is
        # httpx reusing a pooled connection the server already closed after
        # an idle gap). On loopback the cost of a fresh connection per
        # request is sub-millisecond, so there's no real tradeoff here.
        # `retries=1` is a backstop for any other transient connection error.
        limits = httpx.Limits(max_keepalive_connections=0)
        self._client = httpx.Client(
            base_url=base_url, timeout=timeout, limits=limits, transport=httpx.HTTPTransport(retries=1)
        )
        # Separate async client for achat() -- concurrent datagen callers hit
        # this one; ping()/shutdown_server() stay on the sync client since
        # they're one-off setup/teardown calls, not the hot path.
        self._aclient = httpx.AsyncClient(
            base_url=base_url, timeout=timeout, limits=limits, transport=httpx.AsyncHTTPTransport(retries=1)
        )

    def _request_json(self, messages: list[dict], temperature: float, max_tokens: int) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            # llama-server (OpenAI-compatible) reports usage in the final
            # streamed chunk when asked -- if a server doesn't support this
            # it's simply absent, usage stays 0 in the result below.
            "stream_options": {"include_usage": True},
            # Qwen3.5 is a reasoning model — without this it burns the whole
            # max_tokens budget on <think> content and never reaches the
            # actual answer, leaving `content` empty.
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        span_name: str = "llm_chat",
    ) -> ChatResult:
        start = time.perf_counter()
        ttft: float | None = None
        text_parts: list[str] = []
        usage: dict = {}
        with self._client.stream(
            "POST", "/v1/chat/completions", json=self._request_json(messages, temperature, max_tokens)
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                chunk = _sse_payload(line)
                if chunk is None:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    piece = choices[0].get("delta", {}).get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        text_parts.append(piece)
                if chunk.get("usage"):
                    usage = chunk["usage"]
        elapsed = time.perf_counter() - start

        return ChatResult(
            text="".join(text_parts),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_s=elapsed,
            ttft_s=ttft if ttft is not None else elapsed,
        )

    async def achat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        span_name: str = "llm_chat",
    ) -> ChatResult:
        """Async twin of chat() -- lets a caller run many of these concurrently
        (e.g. asyncio.gather/Semaphore) so independent requests actually reach
        llama-server's parallel slots instead of queuing one at a time behind
        a single blocking client.

        The random jitter below is load-bearing, not cosmetic: several new
        connections opened in the same event-loop tick reliably trip "Server
        disconnected without sending a response" on this machine (confirmed
        via an isolated repro -- 5-40% failure rate with zero jitter vs 0/80
        with it, independent of llama-server's own config -- same failure
        with keep-alive disabled, more HTTP threads, a smaller ctx-size, and
        a freshly restarted server, so it's a connection-burst race rather
        than an application-level setting). It has to live here rather than
        only at each worker's first call, since within one chunk's 3
        sequential stages every stage opens its own new connection -- two
        concurrent chunks can still collide the moment their stages happen
        to line up mid-run, not just at startup.
        """
        await asyncio.sleep(random.uniform(0, 0.15))
        start = time.perf_counter()
        ttft: float | None = None
        text_parts: list[str] = []
        usage: dict = {}
        async with self._aclient.stream(
            "POST", "/v1/chat/completions", json=self._request_json(messages, temperature, max_tokens)
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                chunk = _sse_payload(line)
                if chunk is None:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    piece = choices[0].get("delta", {}).get("content")
                    if piece:
                        if ttft is None:
                            ttft = time.perf_counter() - start
                        text_parts.append(piece)
                if chunk.get("usage"):
                    usage = chunk["usage"]
        elapsed = time.perf_counter() - start

        return ChatResult(
            text="".join(text_parts),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            latency_s=elapsed,
            ttft_s=ttft if ttft is not None else elapsed,
        )

    def ping(self) -> bool:
        try:
            response = self._client.get("/v1/models", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def shutdown_server(self) -> bool:
        """Stops the llama-server process this client talks to, freeing its
        VRAM. Returns True if a server was found and stopped."""
        port = self._client.base_url.port
        return stop_llama_server(port) if port else False

    def close(self):
        self._client.close()

    async def aclose(self):
        await self._aclient.aclose()
