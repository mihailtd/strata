#!/usr/bin/env bash
# ==============================================================================
# Setup & Verification for runtime-vllm
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

echo "=============================================================================="
echo " 🔍 SYNCING VLLM/ROCM ENVIRONMENT"
echo "=============================================================================="
uv sync

echo ""
echo "--- vLLM version ---"
uv run python -c "import vllm; print(vllm.__version__)"

echo ""
echo "✅ Setup check complete. To launch the server, run: MODEL=<repo-id> ./run_server.sh"
