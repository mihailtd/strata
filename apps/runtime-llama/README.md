# Standalone `llama.cpp` Runtime (`runtime-llama`)

A cleanly isolated, standalone serving runtime running upstream `llama.cpp` compiled with native AMD ROCm / HIP support (`gfx1100`, AMD Radeon RX 7900 XTX).

---

## Architecture & Responsibilities

- **Scope**: Serves GGUF models directly via native C++ `llama-server`.
- **Isolation**: Completely separated from `apps/runtime` (Python/Triton) and `apps/runtime-next` (Rust).
- **Default Port**: `8001` (OpenAI-compatible `/v1/chat/completions`).

---

## Quick Start

### 1. Build / Compile (One-time)
```bash
cd apps/runtime-llama
./setup.sh
```

### 2. Start Server
```bash
cd apps/runtime-llama
./run_server.sh
```

By default, this:
- Loads the 27B GGUF model into GPU VRAM (`-ngl 99`).
- Enables FlashAttention (`-fa on`) and Q8_0 KV cache (`-ctk q8_0 -ctv q8_0`).
- Activates native MTP and N-Gram speculative decoding (`--spec-type draft-mtp,ngram-mod`).
- Pre-loads all domain specialist LoRA adapters (`astral`, `postgresql`, `duckdb`, `python_web`, `financial`, `python_modern`).

---

## Endpoints

- **Health check**: `GET http://127.0.0.1:8001/health`
- **OpenAI Chat**: `POST http://127.0.0.1:8001/v1/chat/completions`
