#!/usr/bin/env bash
set -euo pipefail
cd /home/mihai/Projects/gnn-experiment
out=$(rocm-smi --showpids 2>&1); [[ $out == *"No KFD PIDs currently running"* ]] || { echo "GPU busy" >&2; exit 3; }
models/runtime_next_bins_current/runtime-next-qwen35_9b --port 8003 > results/benchmarks/v9_pipeline/server_9b.log 2>&1 &
S=$!; trap 'kill $S 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do curl -sf http://127.0.0.1:8003/health >/dev/null && break; sleep 1; done
python3 evals/factory/compare_adapter_generations.py --size 9b --old v7 --new v9 \
  --domains agentic_coding,python_modern,python_web,astral,duckdb,postgresql,financial \
  --old-humaneval-domains agentic_coding,python_modern >> results/benchmarks/v9_pipeline/compare_9b.log 2>&1
echo "[$(date +%T)] 9b eval done"
