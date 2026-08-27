#!/usr/bin/env bash
# One-time setup: clone and build llama.cpp with HIP/ROCm support for gfx1100
# (RX 7900 XTX). Run from the serving/ directory. No sudo needed — cmake and
# ninja are fetched as uv tools, not system packages.
#
#   cd serving && bash setup_llamacpp.sh

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
uv tool install cmake >/dev/null
uv tool install ninja >/dev/null

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
echo "Done. Binary at: llama.cpp/build/bin/llama-server"
