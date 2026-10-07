# ⚡ Strata

**Strata** is an ultra-high-throughput, zero-allocation native Rust serving engine built specifically for **Qwen 3.5 hybrid architectures** (Gated DeltaNet linear attention + full attention) running bare-metal on **AMD ROCm / HIP** hardware.

Designed from first principles to eliminate host overhead, memory fragmentation, and runtime jitter, Strata executes pure GPU compute passes via hand-curated HIP kernels and hipBLASLt without Python or PyTorch runtime baggage.

---

## 🏛️ Architecture Highlights

- **Hybrid Interleaved Execution**: Full native support for Qwen 3.5's hybrid layout (recurrent Gated DeltaNet layers interleaved with full multi-head attention every 4th layer).
- **Zero-Allocation Hot Path**: Scratch tensors and KV states are allocated once at startup. Decoding steps perform zero `hipMalloc` / `hipFree` calls.
- **Hand-Curated HIP Kernels**: Custom device kernels for RMSNorm, RoPE, SwiGLU, Causal Conv1D update, GDN recurrent state update, and W4A16 GEMM/GEMV.
- **OpenAI-Compatible Serving**: Drop-in HTTP server (`GET /health`, `GET /v1/models`, `POST /v1/chat/completions`) supporting both non-streaming JSON and streaming Server-Sent Events (SSE).
- **Tool Calling & Agentic Support**: Parses both structured JSON and Qwen XML tool calls (`<tool_call>...`), supporting parallel and multi-function calling.
- **Byte-Identical Chat Templates**: Built-in Jinja template engine rendering the model's exact `chat_template.jinja` matching Hugging Face `transformers` outputs.
- **In-Place Weight Folding (IPWF)**: Dynamic LoRA adapter activation without duplicating base weights or thrashing VRAM.

---

## ⚙️ Hardware & System Requirements

- **GPU**: AMD Radeon RX 7900 XTX / 7900 XT (`gfx1100`), AMD Instinct MI200 (`gfx90a`), MI300X (`gfx942`).
- **OS**: Linux (Ubuntu 22.04 / 24.04, Arch Linux).
- **ROCm**: Version 6.0 or newer (`libamdhip64`, `hipblas`, `hipblaslt`).
- **Toolchain**: Rust 2024 edition (`rustc`, `cargo`), `hipcc`.

On Arch Linux:
```bash
sudo pacman -S rocm-hip-sdk hipblas hipblaslt
```

On Ubuntu / Debian:
```bash
sudo apt-get install rocm-hip-sdk hipblas-dev hipblaslt-dev
```

---

## 🚀 Building & Running

### 1. Build from Source
Select the model size feature flag matching your target checkpoint (`qwen35_0_8b`, `qwen35_2b`, `qwen35_4b`, `qwen35_9b`, or `qwen35_27b`):

```bash
# Target Qwen 3.5 4B (default)
cargo build --release --features qwen35_4b

# Target Qwen 3.5 9B
cargo build --release --features qwen35_9b
```

To target a specific AMD GPU architecture, set `RUNTIME_NEXT_OFFLOAD_ARCH` or `ROCM_TARGET`:
```bash
RUNTIME_NEXT_OFFLOAD_ARCH=gfx1100 cargo build --release --features qwen35_4b
```

The build produces two artifacts:
- Binary: `target/release/strata`
- Device Kernel Library: `libruntime_next_kernels.so` (linked automatically via `$ORIGIN`)

### 2. Launch the Server
Ensure your model checkpoint is available in your Hugging Face cache (`~/.cache/huggingface/hub/`) or specify the directory explicitly:

```bash
# Launch with default Hugging Face cache lookup on port 8003
./target/release/strata --port 8003

# Or specify a custom model checkpoint directory:
RUNTIME_NEXT_MODEL_DIR=/path/to/model/safetensors ./target/release/strata --port 8003
```

### 3. Query the Engine
Strata is fully compatible with any OpenAI client:

```bash
curl http://127.0.0.1:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:4b-rust",
    "messages": [{"role": "user", "content": "Write a fast fibonacci function in Rust."}],
    "temperature": 0.7,
    "stream": true
  }'
```

---

## 🧪 Testing

Run smoke and verification tests (single-threaded to prevent GPU context collision):

```bash
cargo test --bin strata -- --test-threads=1
```

---

## 📜 License

Licensed under the Apache License, Version 2.0.

