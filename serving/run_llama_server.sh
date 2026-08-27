#!/usr/bin/env bash
# ==============================================================================
# High-Performance Native ROCm HIP Server for 27B on RX 7900 XTX (gfx1100)
# Matches & Exceeds Ollama Defaults with MTP Speculation + N-Gram Lookahead
# ==============================================================================

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/var/lib/ollama/blobs/sha256-f5f1dd8920d417aac2718b0bda3403da274301efdd6760b4f0f4b864ff2ad57d}"
PORT="${PORT:-8001}"
HOST="${HOST:-127.0.0.1}"
CTX_SIZE="${CTX_SIZE:-16384}"
THREADS="${THREADS:-8}"
SLOTS="${SLOTS:-1}"
BATCH_SIZE="${BATCH_SIZE:-512}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/llama.cpp/build/bin"

if [[ ! -f "${BIN_DIR}/llama-server" ]]; then
  echo "Error: llama-server binary not found at ${BIN_DIR}/llama-server"
  echo "Please build it first with: bash ${SCRIPT_DIR}/setup_llamacpp.sh"
  exit 1
fi

export LD_LIBRARY_PATH="${BIN_DIR}:${LD_LIBRARY_PATH:-}"

echo "================================================================================"
echo "🚀 Starting High-Performance ROCm 27B Server with MTP + N-Gram Speculation"
echo "================================================================================"
echo "Model:       ${MODEL_PATH}"
echo "Endpoint:    http://${HOST}:${PORT}"
echo "Context:     ${CTX_SIZE}"
echo "Threads:     ${THREADS}"
echo "Batch Size:  ${BATCH_SIZE}"
echo "Speculation: draft-mtp,ngram-mod (Max Draft: 4, Max N-Gram: 4)"
echo "================================================================================"

exec "${BIN_DIR}/llama-server" \
  --model "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  -ngl 99 \
  -fa on \
  -c "${CTX_SIZE}" \
  -b "${BATCH_SIZE}" \
  -ub "${BATCH_SIZE}" \
  -t "${THREADS}" \
  -np "${SLOTS}" \
  --spec-type draft-mtp,ngram-mod \
  --spec-draft-n-max 4 \
  --spec-ngram-mod-n-max 4 \
  --spec-draft-backend-sampling \
  --no-webui
