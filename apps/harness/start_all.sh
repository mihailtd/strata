#!/usr/bin/env bash
# ==============================================================================
# Unified Stack Launcher for DeepSeek Harness (DSH) + runtime-next + System One
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=========================================================================="
echo "🚀 STARTING FULL AUTONOMOUS STACK (DSH + runtime-next + System One)"
echo "=========================================================================="

PIDS=()

cleanup() {
    trap - SIGINT SIGTERM
    echo ""
    echo "🛑 Shutting down stack services..."
    for pid in "${PIDS[@]}"; do
        if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
            kill "$pid" >/dev/null 2>&1 || true
        fi
    done
    wait 2>/dev/null || true
    echo "✅ All services stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM

# 1. Source .env and enforce offline mode for local HF cache
if [ -f "${WORKSPACE_ROOT}/.env" ]; then
    set -a
    source "${WORKSPACE_ROOT}/.env"
    set +a
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 2. Clean up ports 4000 (DSH), 8003 (runtime-next), 8100 (decision_service)
if command -v fuser >/dev/null 2>&1; then
    fuser -k 4000/tcp >/dev/null 2>&1 || true
    fuser -k 8003/tcp >/dev/null 2>&1 || true
    fuser -k 8100/tcp >/dev/null 2>&1 || true
fi

# 3. Start System One Decision Service (Port 8100 on CPU)
echo "▶ [1/3] Starting System One Decision Service (port 8100 on CPU)..."
uv --directory "${WORKSPACE_ROOT}" run python -m apps.harness.router.decision_service --port 8100 --device cpu &
PID_DECISION=$!
PIDS+=($PID_DECISION)

# 3. Start runtime-next (Port 8003 on RX 7900 XTX)
echo "▶ [2/3] Starting runtime-next (port 8003 on RX 7900 XTX)..."
PORT=8003 "${WORKSPACE_ROOT}/apps/runtime-next/target/release/runtime-next" &
PID_RUNTIME=$!
PIDS+=($PID_RUNTIME)

# 4. Wait for backends to be fully initialized before launching DSH
echo "Waiting for System One Decision Service on port 8100..."
while ! curl -s "http://127.0.0.1:8100/health" >/dev/null 2>&1; do
    if ! kill -0 "$PID_DECISION" >/dev/null 2>&1; then
        echo "❌ System One Decision Service exited prematurely!"
        exit 1
    fi
    sleep 0.5
done
echo "✅ System One Decision Service ready on port 8100."

echo "Waiting for runtime-next 9B (loading ~18GB unquantized weights into GPU VRAM, ~7s)..."
while ! curl -s "http://127.0.0.1:8003/health" >/dev/null 2>&1; do
    if ! kill -0 "$PID_RUNTIME" >/dev/null 2>&1; then
        echo "❌ runtime-next process exited prematurely!"
        exit 1
    fi
    sleep 0.5
done
echo "✅ runtime-next 9B engine ready on port 8003."

# 5. Launch DeepSeek Harness Web UI (Port 4000)
echo "▶ [3/3] Launching DeepSeek Harness Web UI (http://localhost:4000)..."
"${SCRIPT_DIR}/start_dsh.sh" &
PID_DSH=$!
PIDS+=($PID_DSH)

echo "=========================================================================="
echo "✅ All 3 services active! Web UI: http://localhost:4000"
echo "Press Ctrl+C to stop all services simultaneously."
echo "=========================================================================="

wait "$PID_DSH"
