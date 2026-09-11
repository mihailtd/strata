#!/usr/bin/env bash
# Launcher for DeepSeek Harness (dsh) connected to local ROCm 35B MoE Engine & 6-Domain LoRAs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/settings.yaml"

echo "=========================================================================="
echo "🚀 DEEPSEEK HARNESS (dsh) LAUNCHER — v0.1.5-rc.2"
echo "=========================================================================="

PORT=4000

# 1. Sync configuration and plugin patch layer to DeepSeek Harness
mkdir -p "${HOME}/.dsh"
cp "${SCRIPT_DIR}/settings.yaml" "${HOME}/.dsh/settings.yaml"
cp "${SCRIPT_DIR}/cordis.patch.yml" "${HOME}/.dsh/cordis.patch.yml"
rm -f "${HOME}/.dsh/profiles/web/cordis.patch.yml"
echo "✅ Synchronized settings to ${HOME}/.dsh/settings.yaml"
echo "✅ Synchronized plugin patch layer to ${HOME}/.dsh/cordis.patch.yml"

# 3. Ensure all local plugins can resolve @deepseek-ai/* packages.
# DSH loads plugins from absolute paths; Node ESM resolution walks up from the
# plugin file's directory. Symlinking node_modules → ~/.dsh/profiles/node_modules
# (where DSH installs its packages) lets plugins import dsh-tools, cordis, etc.
DSH_NM="${HOME}/.dsh/profiles/node_modules"
for plugin_dir in "${SCRIPT_DIR}/plugins"/*/; do
    if [ ! -e "${plugin_dir}node_modules" ]; then
        ln -sfn "${DSH_NM}" "${plugin_dir}node_modules"
        echo "✅ Linked node_modules for $(basename ${plugin_dir})"
    fi
done

# 2. Free port 4000 if previously occupied
if command -v fuser > /dev/null 2>&1; then
    fuser -k "${PORT}/tcp" > /dev/null 2>&1 || true
fi

echo "Launching DeepSeek Harness Web UI on http://localhost:${PORT}..."
echo "Configured Models & Specialist LoRAs (Port 8000 Native ROCm Engine):"
echo "  • qwen3.8:27b (Primary W4A16 Triton + MTP Speculation Engine)"
echo "  • qwen3.8-27b-auto (Auto-Dynamic Riemannian LoRA Router)"
echo "  • qwen3.8-27b-postgresql (PostgreSQL 17 & pgvector HNSW)"
echo "  • qwen3.8-27b-duckdb (DuckDB Parquet OLAP Analytics)"
echo "  • qwen3.8-27b-fastapi (FastAPI Async Web Specialist)"
echo "  • qwen3.8-27b-astral (Astral uv & ruff Toolchain)"
echo "  • qwen3.8-27b-financial (Financial Planning & Risk Specialist)"
echo "Active Plugins:"
echo "  • tool-subagent:    ENABLED (Native DSH Subagent Delegation)"
echo "  • tool-code-verify: MOUNTED (Language-Agnostic Test & Lint Verification)"
echo "  • tool-moa-router:  MOUNTED (Dynamic MoA Expert Routing — route_prompt tool)"
echo "  • infinite-agent:   MOUNTED (Loop Detection + Context Budget + Delegation)"
echo "Agent Instructions:"
echo "  • AGENTS.md loaded from project root (expert routing protocol, loop rules)"
echo "=========================================================================="

if command -v dsh > /dev/null; then
    dsh web --port "${PORT}"
elif command -v npx > /dev/null; then
    npx -y @deepseek-ai/dsh web --port "${PORT}"
else
    echo "Error: dsh or npx is required to launch DeepSeek Harness."
    exit 1
fi


