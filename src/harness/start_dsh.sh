#!/usr/bin/env bash
# Launcher for DeepSeek Harness (dsh) connected to local ROCm 35B MoE Engine & 6-Domain LoRAs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/settings.yaml"

echo "=========================================================================="
echo "🚀 DEEPSEEK HARNESS (dsh) LAUNCHER"
echo "=========================================================================="

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
echo "  • ornith-1.5:35b (Ornith 1.5 35B MoE - Primary Agent Engine)"
echo "  • ornith-1.5:32k (Ornith 1.5 35B MoE - 32k Context Zero-Spill)"
echo "  • qwen3.6:35b    (Qwen 3.6 35B MoE)"
echo "=========================================================================="

if command -v dsh > /dev/null; then
    dsh web --port "${PORT}"
elif command -v npx > /dev/null; then
    npx -y @deepseek-ai/dsh web --port "${PORT}"
else
    echo "Error: dsh or npx is required to launch DeepSeek Harness."
    exit 1
fi
