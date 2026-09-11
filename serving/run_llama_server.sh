#!/usr/bin/env bash
# Backwards-compatibility wrapper: redirects to standalone runtime in apps/runtime-llama
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/../apps/runtime-llama/run_server.sh" "$@"
