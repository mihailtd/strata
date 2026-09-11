#!/usr/bin/env bash
# ==============================================================================
# One-time setup: clone and build llama.cpp with HIP/ROCm support for gfx1100
# (AMD Radeon RX 7900 XTX).
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export PATH="$HOME/.local/bin:$PATH"
uv tool install cmake >/dev/null 2>&1 || true
uv tool install ninja >/dev/null 2>&1 || true

if [[ ! -d llama.cpp ]]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
fi

cd llama.cpp
cmake -B build -G Ninja \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx1100 \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j"$(nproc)"

echo
echo "✅ Build complete. Binary located at: ${SCRIPT_DIR}/llama.cpp/build/bin/llama-server"
