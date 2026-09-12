"""Test Suite for DeepSeek Harness Connection & LoRA Model Aliases."""

import json
import time
import urllib.request


def test_list_models():
    print("Testing GET /v1/models...")
    req = urllib.request.Request("http://localhost:8000/v1/models")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        models = [m["id"] for m in data.get("data", [])]
        print(f"✅ Found {len(models)} models: {models[:6]}")
        assert "qwen3.8:27b" in models
        assert "qwen3.8-27b-postgresql" in models
        assert "qwen3.8-27b-duckdb" in models


def test_chat_completions_with_lora_alias(model_id: str = "qwen3.8-27b-postgresql"):
    print(f"\nTesting POST /v1/chat/completions with model='{model_id}'...")
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "How do I create an HNSW cosine index in pgvector?"}],
        "stream": True,
        "max_tokens": 128,
    }
    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:8000/v1/chat/completions",
        data=data_bytes,
        headers={"Content-Type": "application/json"},
    )

    t_start = time.perf_counter()
    chunks = []
    with urllib.request.urlopen(req, timeout=30) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if line_str.startswith("data: ") and not line_str.endswith("[DONE]"):
                try:
                    chunk = json.loads(line_str[6:])
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content") or delta.get("reasoning_content") or ""
                    chunks.append(content)
                except Exception:
                    pass

    elapsed = time.perf_counter() - t_start
    full_text = "".join(chunks)
    print(f"✅ Received {len(chunks)} SSE chunks in {elapsed:.2f}s")
    print(f"Snippet: {full_text[:120]}...")
    assert len(chunks) > 0


if __name__ == "__main__":
    test_list_models()
    test_chat_completions_with_lora_alias("qwen3.8:27b")
    test_chat_completions_with_lora_alias("qwen3.8-27b-postgresql")
    print("\n🎉 ALL DEEPSEEK HARNESS CONNECTION TESTS PASSED!")
