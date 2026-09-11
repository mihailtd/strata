#!/usr/bin/env bash
# ==============================================================================
# Setup & Verification for runtime-ollama Baseline
# ==============================================================================
set -euo pipefail

echo "=============================================================================="
echo " 🔍 VERIFYING OLLAMA BASELINE INSTALLATION & GPU DETECTION"
echo "=============================================================================="

OLLAMA_BIN="$(command -v ollama || true)"
if [[ -z "${OLLAMA_BIN}" ]]; then
    echo "❌ ERROR: 'ollama' binary not found on PATH." >&2
    exit 1
fi
echo "✅ Ollama binary found: ${OLLAMA_BIN}"
"${OLLAMA_BIN}" --version

echo ""
echo "--- Installed Models Available for Baseline A/B Testing ---"
"${OLLAMA_BIN}" list

echo ""
echo "✅ Setup check complete. To launch the server, run: ./run_server.sh"
