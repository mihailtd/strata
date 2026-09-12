#!/usr/bin/env bash
# ==============================================================================
# Standalone vLLM Baseline Server Launcher (AMD ROCm / RX 7900 XTX gfx1100)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PORT="${PORT:-8004}"
HOST="${HOST:-127.0.0.1}"
MODEL="${MODEL:?Set MODEL to a HuggingFace repo id or local checkpoint path, e.g. MODEL=Qwen/Qwen3.5-9B ./run_server.sh}"

if [[ "${1:-}" == "stop" ]]; then
  echo "[runtime-vllm] Stopping vLLM server..."
  pkill -f "vllm serve" || true
  echo "[runtime-vllm] Stopped."
  exit 0
fi

export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0

echo "=============================================================================="
echo " 🚀 STARTING STANDALONE VLLM BASELINE RUNTIME"
echo "    Model: ${MODEL}"
echo "    Host / Port: http://${HOST}:${PORT}"
echo "    GPU: AMD Radeon RX 7900 XTX (gfx1100, HSA_OVERRIDE_GFX_VERSION=11.0.0)"
echo "=============================================================================="

exec uv run vllm serve "${MODEL}" --host "${HOST}" --port "${PORT}"
