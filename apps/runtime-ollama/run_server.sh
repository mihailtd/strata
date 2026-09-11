#!/usr/bin/env bash
# ==============================================================================
# Standalone Ollama Baseline Server Launcher (AMD ROCm / RX 7900 XTX gfx1100)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT=11434
HOST="127.0.0.1"

# Handle "stop" argument to cleanly evict and terminate
if [[ "${1:-}" == "stop" ]]; then
    echo "[runtime-ollama] Stopping Ollama server and freeing VRAM..."
    # Attempt gentle model eviction first
    curl -s -X POST "http://${HOST}:${PORT}/api/generate" -d '{"model":"", "keep_alive":0}' >/dev/null 2>&1 || true
    # Kill any standalone running ollama serve process
    pkill -f "ollama serve" || true
    echo "[runtime-ollama] Ollama stopped."
    exit 0
fi

# Ensure /usr/bin/ollama exists
OLLAMA_BIN="$(command -v ollama || true)"
if [[ -z "${OLLAMA_BIN}" ]]; then
    echo "ERROR: 'ollama' binary not found on PATH. Install ollama or check /usr/bin/ollama." >&2
    exit 1
fi

# Set AMD ROCm and RDNA3 environment variables
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_HOST="${HOST}:${PORT}"
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0
export OLLAMA_NUM_PARALLEL=1

echo "=============================================================================="
echo " 🦙 STARTING STANDALONE OLLAMA BASELINE RUNTIME"
echo "    Binary: ${OLLAMA_BIN}"
echo "    Host / Port: http://${HOST}:${PORT}"
echo "    GPU: AMD Radeon RX 7900 XTX (gfx1100, HSA_OVERRIDE_GFX_VERSION=11.0.0)"
echo "    Flash Attention: Enabled"
echo "    KV Cache: Q8_0"
echo "=============================================================================="

# Check if port is already active
if curl -s "http://${HOST}:${PORT}/api/version" >/dev/null 2>&1; then
    echo "[runtime-ollama] Ollama server is already running on http://${HOST}:${PORT}."
    echo "[runtime-ollama] Active models in VRAM:"
    ollama ps || true
    exit 0
fi

# Execute ollama serve
exec "${OLLAMA_BIN}" serve
