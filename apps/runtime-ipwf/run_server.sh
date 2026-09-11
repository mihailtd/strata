#!/usr/bin/env bash
# run_server.sh — Start the In-Place Weight Folding (IPWF) Inference Engine
#
# Usage:
#   ./apps/runtime-ipwf/run_server.sh [OPTIONS]
#
# Environment variables (all optional):
#   PORT=8002                    HTTP port (default: 8002)
#   HOST=0.0.0.0                 Bind address (default: 0.0.0.0)
#   DEFAULT_MODEL_ID             Which base model to load (default: Qwen/Qwen3.5-4B)
#                                Options: Qwen/Qwen3.5-4B | Qwen/Qwen3.5-9B
#   AUTO_LOAD_MODEL=1            Pre-load the engine on startup (default: 0 = standby)
#   SPECULATIVE_DECODE=1         Enable bucketed speculative decoding (default: 1)
#   SPECULATIVE_K=2              Speculative draft depth (default: 2)
#   SPECULATIVE_MAX_SEQ_LEN=4096 Max sequence length for speculative buckets (default: 4096)
#   FLASH_NORM_FOLD=1            Enable FlashNorm weight folding for 4B (default: 1)
#   MAX_SEQ_LEN                  Override max sequence length (default: 16384 for 4B, 4096 for 9B)
#   RING_BUFFER_MODE             Ring buffer strategy: dense|poet|selective_hybrid|pointer
#   KV_CACHE_DTYPE               KV cache dtype (bfloat16 required; INT4 is banned)
#
# A/B Testing Note:
#   This engine occupies the full 24 GB VRAM of the RX 7900 XTX.
#   NEVER start this alongside runtime-triton or runtime-llama.
#   Sequential protocol: stop one engine, then start the other.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/apps:$REPO_ROOT"

# Defaults
export PORT="${PORT:-8002}"
export HOST="${HOST:-0.0.0.0}"
export DEFAULT_MODEL_ID="${DEFAULT_MODEL_ID:-Qwen/Qwen3.5-4B}"
export AUTO_LOAD_MODEL="${AUTO_LOAD_MODEL:-0}"
export SPECULATIVE_DECODE="${SPECULATIVE_DECODE:-1}"
export SPECULATIVE_K="${SPECULATIVE_K:-2}"
export FLASH_NORM_FOLD="${FLASH_NORM_FOLD:-1}"

echo "========================================"
echo "  IPWF Engine (In-Place Weight Folding)"
echo "  Port:  $PORT"
echo "  Model: $DEFAULT_MODEL_ID (unquantized bfloat16)"
echo "  GPU:   AMD RX 7900 XTX (ROCm)"
echo "  Spec:  SPECULATIVE_DECODE=$SPECULATIVE_DECODE  k=$SPECULATIVE_K"
echo "========================================"

if [[ "${AUTO_LOAD_MODEL}" == "1" ]]; then
    echo "  Mode:  PRELOAD (loads on startup)"
else
    echo "  Mode:  STANDBY (loads on first POST /api/engine/load)"
fi
echo ""

cd "$REPO_ROOT"
uv run python "$SCRIPT_DIR/server.py"
