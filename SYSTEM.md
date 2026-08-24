# System Details (Native CachyOS Linux Workstation)

Snapshot updated 2026-08-24. This document serves as the authoritative reference for the development environment, compute stack, and runtime toolchain across the `gnn-experiment` repository.

---

## 1. Operating System & Host Environment

- **OS**: CachyOS Linux (Arch-based rolling release)
- **Kernel**: `7.2.0-1-cachyos` (x86_64, `PREEMPT_DYNAMIC`, optimized BORE/EEVDF scheduler)
- **Architecture**: Native bare-metal Linux workstation — **no WSL2, no virtualization overhead, no Direct3D translation layers, and no guest memory/CPU caps**.
- **User & Groups**: User `mihai` with active membership in `render`, `video`, `wheel`, `kvm`, and `storage` groups for direct hardware access.

---

## 2. Hardware Architecture

- **CPU**: AMD Ryzen 9 9900X (12 Cores / 24 Threads, Zen 5 architecture)
  - All 24 logical threads available natively with full boost clock performance.
- **System Memory (RAM)**: 64 GB DDR5 (~60 GiB usable physical RAM)
  - Configured with 60 GiB fast NVMe swap space for high-concurrency datagen and large dataset caching.
- **Storage**: 2.0 TB PCIe Gen4 NVMe SSD (`/dev/nvme1n1p2`, ~1.8 TB free space). Direct native Linux I/O performance (no 9P/VHDX bottlenecks).

---

## 3. GPU & AMD Compute Stack

### Primary Compute Hardware
- **Card**: AMD Radeon RX 7900 XTX (Navi 31 / RDNA3 architecture, `gfx1100` target)
- **VRAM**: **24 GB GDDR6** dedicated VRAM
- **Integrated Graphics**: AMD Granite Ridge Radeon Graphics (2 GB shared)

### Native Driver & Device Interfaces
- **Direct KFD Node**: `/dev/kfd` (Kernel Fusion Driver) natively present with full topology access.
- **Direct DRM Render Nodes**: `/dev/dri/renderD128`, `/dev/dri/renderD129` (card0, card1).
- **Direct Hardware Compute**: Native Linux amdgpu kernel driver. All WSL-specific DXCore shims, Direct3D translation wrappers, and `LD_PRELOAD` workarounds have been completely eliminated.

### ROCm Installation
- **ROCm Version**: **ROCm 7.2.4** installed natively at `/opt/rocm` (includes `rocminfo`, `hipcc`, `llvm/bin/clang`, `llvm/bin/clang++`).
- **GPU Agent Detection**: System `rocminfo` and PyTorch natively identify the `gfx1100` compute agent without requiring `HSA_OVERRIDE_GFX_VERSION`.

### GPU Preflight Verification
To verify hardware compute status and run a live 4096×4096 GPU matmul:
```bash
uv run python scripts/audit/check_gpu.py
```
Expected output:
- `torch.cuda.is_available(): True`
- `device count: 2`
- `[0] AMD Ryzen 9 9900X / AMD Radeon RX 7900 XTX — 24.0 GB Total`
- Clean execution with 0 leftover GPU resident processes.

---

## 4. Language Runtimes & Toolchains

### Python & uv
- **Python Version**: Python 3.14.7 (managed via `uv 0.12.5`).
- **Package Manager**: `uv` is used for all Python workspace resolution, dependency locking (`uv.lock`), and virtual environment management (`.venv`).
- **Interpreter Isolation**: Native Linux x86_64 interpreter (`sys_platform == 'linux'`).

### JavaScript / Web Toolchain
- **Node.js**: Node v26.7.0 (managed via `fnm`).
- **Package Managers**: `pnpm 11.22.0` (primary for `src/dashboard`), `npm 11.19.0`.

### Compilers & Build Tools
- **C/C++**: Clang 21 (`/usr/bin/clang`, `/usr/bin/clang++`), GCC 15 (`/usr/bin/gcc`, `/usr/bin/g++`), Make.
- **HIP Compiler**: HIP-Clang (`/opt/rocm/bin/hipcc`).
- **Git**: Git 2.53.0 on native Linux filesystem (`core.fileMode = false` configured, author `Mihai Farcas <farcasmihai91@gmail.com>`).

---

## 5. Repository Subprojects & Isolated Environments

The codebase uses cleanly isolated environments for distinct dependency constraints:

### 1. Root Engine (`runtime`)
- **Manifest**: `pyproject.toml`, `uv.lock`
- **Virtualenv**: `/.venv`
- **Purpose**: Autonomous Runtime Engine with dynamic weight folding, speculative decoding, Riemannian dynamic routing, data generation, and evaluation suites.
- **Key Dependencies**:
  - `torch==2.13.0+rocm7.2`, `triton-rocm==3.7.1`, `torchvision==0.28.0+rocm7.2` (from `https://download.pytorch.org/whl/rocm7.2`)
  - `vllm==0.23.1.dev1+rocm7.14.0.g9ddef7117.d20260715`, `flash-attn==2.8.3` (from `https://rocm.frameworks.amd.com/whl-multi-arch/vllm-rdna`)
  - `transformers>=5.14.1`, `peft>=0.20.0`, `accelerate>=1.14.0`, `bitsandbytes>=0.50.0`
  - `liger-kernel>=0.8.1`, `py-pglite[extensions]>=0.5.3`, `psycopg>=3.3.4`, `pgvector>=0.5.0`
- **Test Suite**:
  ```bash
  uv run pytest tests/
  ```

### 2. Training Engine (`training/`)
- **Manifest**: `training/pyproject.toml`, `training/uv.lock`
- **Virtualenv**: `training/.venv`
- **Purpose**: Isolated fine-tuning harness for `unsloth` and Unsloth Studio. Kept isolated because `unsloth` pins `torch<2.12.0`.
- **Key Dependencies**:
  - `torch==2.11.0+rocm7.2`, `triton-rocm==3.6.0`, `torchvision==0.26.0+rocm7.2`
  - `unsloth[amd,studio]`, `gguf>=0.19.0`, `av>=18.0.0`
- **Sync Command**:
  ```bash
  cd training && uv sync
  ```

### 3. Serving Engine (`serving/`)
- **Manifest**: `serving/pyproject.toml`, `serving/uv.lock`
- **Virtualenv**: `serving/.venv`
- **Purpose**: Isolated serving environment for OpenAI-compatible endpoints and native llama.cpp HIP builds on `gfx1100`.

### 4. Interactive Web Dashboard (`src/dashboard/`)
- **Manifest**: `src/dashboard/package.json`, `src/dashboard/pnpm-lock.yaml`
- **Node Modules**: `src/dashboard/node_modules`
- **Purpose**: Next.js 16 web application for real-time telemetry, interactive DAG visualization, adapter training goldilocks curves, and chat interface.
- **Tech Stack**:
  - Next.js 16.3.1 (Turbopack), React 19.2.8, TailwindCSS v4
  - Apache ECharts 6.1.0, `@xyflow/react` 12.11.3, KaTeX math typesetting, MDX documentation loader
- **Development & Build Commands**:
  ```bash
  cd src/dashboard
  pnpm dev     # Launch development server on http://localhost:3000
  pnpm build   # Optimized production Turbopack build
  ```

---

## 6. Authentication & Environment Configuration

- **`.env`** (Repository root, gitignored):
  ```bash
  HF_TOKEN=hf_...  # HuggingFace token for model access & gated weights
  ```
- **Note on `LD_PRELOAD`**: The previous WSL workaround `LD_PRELOAD=/opt/rocm-7.2.0/lib/libhsa-runtime64.so` is **not required** on CachyOS native Linux. Native KFD driver integration handles all HSA runtime communications automatically.
