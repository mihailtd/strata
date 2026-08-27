# 🔌 DeepSeek Harness (dsh) Integration Guide

This directory provides the configuration and integration layer to plug our **Supercharged 27B Native Triton W4A16 & Specialist LoRA Engine** into **DeepSeek Harness (`dsh`)**.

---

## 🎯 Quickstart: Launching the Harness

### 1. Ensure our Python Runtime is Running
Our server runs on port `8000`:
```bash
uv run --env-file .env python -m runtime.server
```

### 2. Launch DeepSeek Harness Web UI
```bash
bash src/harness/start_dsh.sh
```
Or directly via `npx`:
```bash
npx -y @deepseek-ai/dsh web --port 4000
```

### 3. Connect Model in the Web UI
Open **`http://localhost:4000`** in your browser:
1. Click the **Gear icon (Settings) &rarr; Models**.
2. Click **Add Custom Provider / Model**:
   - **Provider**: `OpenAI Compatible`
   - **Base URL**: `http://localhost:8000/v1`
   - **API Key**: `sk-local` (any non-empty string)
   - **Model ID**: `qwen3.8:27b` *(or pick any specialist alias like `qwen3.8-27b-postgresql` or `qwen3.8-27b-duckdb`)*.
3. Start chatting! The harness will now execute all tool interactions and reasoning traces through our **Native 27B Speculative LoRA Engine**.

---

## 🔄 How LoRA Switching Works via OpenAI-Compatible API

Because standard agent harnesses communicate via the OpenAI REST specification (`/v1/chat/completions`), LoRA switching is handled seamlessly through **three coordinated mechanisms**:

### 1. Auto-Dynamic Semantic Routing (`model="qwen3.8:27b"` or `model="qwen3.8-27b-auto"`)
* When the agent sends a prompt or tool output, the server runs our **Ledoit-Wolf semantic routing classifier in $<30\mu\text{s}$**.
* The server dynamically hot-activates the optimal specialist adapter (e.g. `postgresql` or `duckdb`) directly in VRAM before emitting the first generated token.

### 2. Virtual Model Aliases (Deterministic Specialist Pinning)
You can pin a specific domain specialist in the Harness dropdown:
* `qwen3.8-27b-postgresql` &rarr; Locks into PostgreSQL 17 & pgvector HNSW specialist.
* `qwen3.8-27b-duckdb` &rarr; Locks into DuckDB Parquet OLAP specialist.
* `qwen3.8-27b-astral` &rarr; Locks into Astral `uv` / `ruff` toolchain specialist.
* `qwen3.8-27b-fastapi` &rarr; Locks into FastAPI async web architecture specialist.
* `qwen3.8-27b-financial` &rarr; Locks into Monte Carlo wealth modeling specialist.
* `qwen3.8-27b-base` &rarr; Restores pristine base weights (zero LoRA adapters).

---

## ⚖️ Architectural Analysis: What is Preserved vs What is Lost?

| Feature Dimension | Preserved over OpenAI API? | How It Works / Architectural Trade-off |
| :--- | :---: | :--- |
| **Native ROCm Acceleration (`gfx1100`)** | ✅ **100% Preserved** | Native HIP compilation with Flash Attention offloading 100% of layers to 24GB GDDR6 VRAM. |
| **Hybrid Speculative Decoding (MTP + N-Gram)** | ✅ **100% Preserved** | Uses Qwen MTP draft heads + N-gram lookahead streaming at **$\approx 47.4\text{ tok/s}$** on 27B. |
| **Collapsible Reasoning (`<think>`)** | ✅ **100% Preserved** | Streamed as reasoning chunks or XML tags rendered in the DSH UI. |
| **Dynamic Specialist Intent Routing** | ✅ **100% Preserved** | Sub-millisecond classifier dynamically injects domain specialist expertise (PostgreSQL 17, DuckDB, Astral, FastAPI). |
| **Multi-Turn Context Prefix Caching** | ✅ **100% Preserved** | Automatically matches prompt prefixes across turns, reducing warm TTFT from $330\text{ ms}$ to $<25\text{ ms}$. |
| **Continuous Multi-LoRA Blending** | ⚠️ **Mapped to Query String** | Continuous domain blending can be specified as `model="qwen3.8-27b:postgres=0.6,duckdb=0.4"`. |

