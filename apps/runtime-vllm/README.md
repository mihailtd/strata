# Standalone vLLM Baseline Runtime (`runtime-vllm`)

A cleanly isolated, standalone baseline harness running upstream **vLLM** with AMD ROCm hardware acceleration (`gfx1100`, AMD Radeon RX 7900 XTX).

---

## Architecture & Responsibilities

- **Scope**: External SOTA / commodity baseline for comparative A/B evaluation against this repo's own engines (`apps/runtime`, `apps/runtime-triton`, `apps/runtime-ipwf`).
- **Isolation**: Own `pyproject.toml`/`uv.lock`/`.venv` — vLLM's ROCm wheel trio (`torch==2.12.0+rocm7.14.0`, `vllm` from the `vllm-rdna` index, `flash-attn`) conflicts with every other project's torch pin in this repo (root: `torch>=2.13.0`; `apps/factory`: `torch==2.11.0+rocm7.2`; `apps/runtime-triton`: `torch>=2.13.0` from a different index). This project exists specifically so that conflict never has to be resolved.
- **Default Port**: `8004` (OpenAI-compatible `/v1/chat/completions`, vLLM's built-in server).
- **Single-Engine Rule**: never run this concurrently with another runtime on the same GPU — this hardware has one GPU. Run one, benchmark it, stop it, then benchmark the next (`apps/runtime-common`'s `gpu_preflight` guard enforces this at process level).

---

## Quick Start

### 1. Sync the environment
```bash
cd apps/runtime-vllm
./setup.sh
```

### 2. Start the server
```bash
cd apps/runtime-vllm
MODEL=Qwen/Qwen3.5-9B ./run_server.sh
```
`MODEL` accepts any HuggingFace repo id or local checkpoint path vLLM understands. This sets the ROCm environment flags (`HSA_OVERRIDE_GFX_VERSION=11.0.0`, single-GPU visibility) and execs `vllm serve`.

### 3. Stop the server
```bash
./run_server.sh stop
```

---

## Why this exists

vLLM here is a commodity-baseline comparator, not a production serving path — this repo's own engines (`apps/runtime-triton` for the 27B W4A16 model, `apps/runtime-ipwf` for 3B/9B in-place weight folding) are what's actually being developed and measured. vLLM's job is to give an honest, independently-maintained upstream number to compare against, the same role `apps/runtime-ollama` and `apps/runtime-llama` play for llama.cpp/Ollama-based baselines.

Moved out of the old root-level `serving/` folder (pre-Moon-restructuring layout) unchanged except for its name and location — same dependency pins, same ROCm indexes.
