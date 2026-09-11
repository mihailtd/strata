#!/usr/bin/env bash
# run_server.sh — Start the Native Triton 27B W4A16 Inference Engine
#
# Usage:
#   ./apps/runtime-triton/run_server.sh [OPTIONS]
#
# Environment variables (all optional):
#   PORT=8000               HTTP port (default: 8000)
#   HOST=0.0.0.0            Bind address (default: 0.0.0.0)
#   AUTO_LOAD_MODEL=1       Pre-load the engine on startup (default: 0 = standby)
#   STREAM_ORDER_TIMEOUT_S  Max seconds to wait for a streaming client (default: 300)
#
# A/B Testing Note:
#   This engine occupies the full 24 GB VRAM of the RX 7900 XTX.
#   NEVER start this alongside runtime-ipwf or runtime-llama.
#   Sequential protocol: stop one engine, then start the other.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONUNBUFFERED=1
# No PYTHONPATH hack needed: this is now a fully independent uv project with
# its own venv (own pyproject.toml, native_27b_engine.py + deps vendored
# locally, only runtime-common pulled in as a path dependency). Running
# `python server.py` puts the script's own directory on sys.path[0], which is
# all the flat sibling imports (native_27b_engine, w4a16_loader, etc.) need.

# Defaults
export PORT="${PORT:-8000}"
export HOST="${HOST:-0.0.0.0}"
export AUTO_LOAD_MODEL="${AUTO_LOAD_MODEL:-0}"

echo "========================================"
echo "  Triton 27B W4A16 Engine"
echo "  Port:  $PORT"
echo "  Model: Qwen3.8:27B (GGUF W4A16)"
echo "  GPU:   AMD RX 7900 XTX (ROCm)"
echo "========================================"

if [[ "${AUTO_LOAD_MODEL}" == "1" ]]; then
    echo "  Mode:  PRELOAD (loads on startup)"
else
    echo "  Mode:  STANDBY (loads on first request)"
fi
echo ""

cd "$SCRIPT_DIR"
uv run python server.py
