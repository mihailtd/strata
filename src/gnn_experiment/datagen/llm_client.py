"""Thin OpenAI-compatible client for the local llama-server instance.

Every call is wrapped in an MLflow trace span (visible in the MLflow UI's
Traces tab) so each individual LLM call — prompt, response, latency, token
usage — is inspectable, not just the per-run aggregate metrics.
"""

import os
import signal
import subprocess
import time

import httpx
import mlflow

from gnn_experiment.datagen.schema import ChatResult


def stop_llama_server(port: int, grace_s: float = 5.0) -> bool:
    """Finds and stops the llama-server process bound to `port` (SIGTERM,
    then SIGKILL after grace_s if it hasn't exited) — frees its VRAM.
    Returns True if a process was found and signaled, False if none was
    running. Matches by port in the command line, so it only touches the
    server we were actually talking to, not other llama-server instances.
    """
    pattern = rf"llama-server.*--port\s+{port}\b"

    def _find_pids() -> list[int]:
        result = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True
        )
        return [int(p) for p in result.stdout.split() if p.strip()]

    pids = _find_pids()
    if not pids:
        return False

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.time() + grace_s
    while time.time() < deadline:
        if not _find_pids():
            return True
        time.sleep(0.5)

    for pid in _find_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return True


class LocalLLMClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model: str = "models/Qwen3.5-4B-Q8_0.gguf",
        timeout: float = 120.0,
    ):
        self.model = model
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        span_name: str = "llm_chat",
    ) -> ChatResult:
        with mlflow.start_span(name=span_name, span_type="LLM") as span:
            span.set_inputs(
                {
                    "messages": messages,
                    "model": self.model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )

            start = time.perf_counter()
            response = self._client.post(
                "/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    # Qwen3.5 is a reasoning model — without this it burns the
                    # whole max_tokens budget on <think> content and never
                    # reaches the actual answer, leaving `content` empty.
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            response.raise_for_status()
            elapsed = time.perf_counter() - start
            data = response.json()
            usage = data.get("usage", {})

            result = ChatResult(
                text=data["choices"][0]["message"]["content"],
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                latency_s=elapsed,
            )

            span.set_outputs({"text": result.text})
            span.set_attributes(
                {
                    "latency_s": elapsed,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "model": self.model,
                }
            )
            return result

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
