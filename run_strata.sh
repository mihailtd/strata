#!/usr/bin/env bash
# ==============================================================================
# Strata 1-Command Golden Path Launcher
# ==============================================================================
# Builds and starts Strata (native Rust + AMD ROCm/HIP) with hardware preflight.
# Usage:
#   ./run_strata.sh [--port 8003] [--feature qwen35_4b]
# ==============================================================================
set -e

PORT="8003"
FEATURE="qwen35_4b"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --port) PORT="$2"; shift ;;
        --feature) FEATURE="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "=========================================================="
echo "⚡ Starting Strata (AMD ROCm / HIP Native Inference Engine)"
echo "=========================================================="

# Check ROCm toolchain
if ! command -v rocminfo &> /dev/null; then
    echo "⚠️  Warning: rocminfo not found in PATH. Checking /opt/rocm..."
    if [ -d "/opt/rocm" ]; then
        export PATH="/opt/rocm/bin:$PATH"
        export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"
    else
        echo "❌ Error: ROCm not detected. Please install ROCm 6.0+."
        exit 1
    fi
fi

# Detect AMD GPU Architecture
DETECTED_ARCH=$(rocminfo 2>/dev/null | grep -m1 "Name:" | grep -o "gfx[0-9]*" || echo "gfx1100")
export ROCM_TARGET="${ROCM_TARGET:-$DETECTED_ARCH}"
export RUNTIME_NEXT_OFFLOAD_ARCH="$ROCM_TARGET"

echo " Detected GPU Architecture: $ROCM_TARGET"
echo " Model Feature Selection:   $FEATURE"
echo " HTTP Server Port:          $PORT"
echo "=========================================================="

cd "$(dirname "$0")/apps/runtime-next"
cargo run --release --no-default-features --features "$FEATURE" -- --port "$PORT"
