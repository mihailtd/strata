#!/usr/bin/env bash
# Launcher for DeepSeek Harness (dsh) connected to local Supercharged 27B Triton Engine

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/settings.yaml"
SERVER_URL="http://localhost:8000/v1/models"

echo "=========================================================================="
echo "🚀 DEEPSEEK HARNESS (dsh) LAUNCHER"
echo "=========================================================================="
echo "Checking server status at ${SERVER_URL}..."

if curl -s -f "${SERVER_URL}" > /dev/null; then
    echo "✅ Local Supercharged 27B Server is online on port 8000!"
else
    echo "⚠️ Local server not responding on port 8000."
    echo "   Starting runtime server in background..."
    uv run --env-file .env python -m runtime.server &
    sleep 3
fi

PORT=4000

# 1. Sync configuration to DeepSeek Harness home directory
mkdir -p "${HOME}/.dsh"
cp "${SCRIPT_DIR}/settings.yaml" "${HOME}/.dsh/settings.yaml"
echo "✅ Synchronized settings to ${HOME}/.dsh/settings.yaml"

# 2. Free port 4000 if previously occupied
if command -v fuser > /dev/null 2>&1; then
    fuser -k "${PORT}/tcp" > /dev/null 2>&1 || true
fi

echo "Launching DeepSeek Harness Web UI on http://localhost:${PORT}..."
echo "Models configured:"
echo "  • qwen3.8:27b (Auto-Dynamic Specialist LoRA Router)"
echo "  • qwen3.8-27b-postgresql (PostgreSQL 17 & Vector Search Expert)"
echo "  • qwen3.8-27b-duckdb (DuckDB Parquet OLAP Expert)"
echo "  • qwen3.8-27b-astral (Astral uv & ruff Toolchain Expert)"
echo "  • qwen3.8-27b-fastapi (FastAPI Async Expert)"
echo "  • qwen3.8-27b-financial (Monte Carlo Wealth Expert)"
echo "=========================================================================="

if command -v npx > /dev/null; then
    npx -y @deepseek-ai/dsh web --port "${PORT}"
else
    echo "Error: Node.js / npx is required to launch DeepSeek Harness."
    exit 1
fi
