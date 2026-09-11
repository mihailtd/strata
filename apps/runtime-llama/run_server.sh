#!/usr/bin/env bash
# ==============================================================================
# High-Performance Native ROCm HIP Server (llama.cpp) for 27B on RX 7900 XTX
# Standalone Runtime: apps/runtime-llama
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
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ ! -f "${BIN_DIR}/llama-server" ]]; then
  echo "Error: llama-server binary not found at ${BIN_DIR}/llama-server"
  echo "Please build it first with: bash ${SCRIPT_DIR}/setup.sh"
  exit 1
fi

export LD_LIBRARY_PATH="${BIN_DIR}:${LD_LIBRARY_PATH:-}"

echo "================================================================================"
echo "🚀 Starting Standalone ROCm llama.cpp Server (port ${PORT})"
echo "================================================================================"
echo "Model:       ${MODEL_PATH}"
echo "Endpoint:    http://${HOST}:${PORT}"
echo "Context:     ${CTX_SIZE}"
echo "Threads:     ${THREADS}"
echo "Batch Size:  ${BATCH_SIZE}"
echo "Speculation: draft-mtp,ngram-mod (Max Draft: 4, Max N-Gram: 4)"
echo "================================================================================"

ADAPTERS_DIR="${PROJECT_ROOT}/results/adapters"
LORA_ARGS=()
if [[ -f "${ADAPTERS_DIR}/astral_27b.gguf" ]]; then
  LORA_LIST="${ADAPTERS_DIR}/astral_27b.gguf,${ADAPTERS_DIR}/postgresql_27b.gguf,${ADAPTERS_DIR}/duckdb_27b.gguf,${ADAPTERS_DIR}/python_web_27b.gguf,${ADAPTERS_DIR}/financial_27b.gguf,${ADAPTERS_DIR}/python_modern_27b.gguf"
  LORA_ARGS=(
    --lora "${LORA_LIST}"
    --lora-init-without-apply
  )
  echo "LoRAs loaded: astral(0), postgresql(1), duckdb(2), python_web(3), financial(4), python_modern(5)"
fi

exec "${BIN_DIR}/llama-server" \
  --model "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  -ngl 99 \
  -fa on \
  -ctk q8_0 \
  -ctv q8_0 \
  -c "${CTX_SIZE}" \
  -b "${BATCH_SIZE}" \
  -ub "${BATCH_SIZE}" \
  -t "${THREADS}" \
  -np "${SLOTS}" \
  --spec-type draft-mtp,ngram-mod \
  --spec-draft-n-max 4 \
  --spec-ngram-mod-n-max 4 \
  --spec-draft-backend-sampling \
  "${LORA_ARGS[@]}" \
  --no-webui
