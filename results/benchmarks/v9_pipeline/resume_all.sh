#!/usr/bin/env bash
# Resume everything that was paused on 2026-09-24. Every stage skips what is
# already saved (eval arms, trained adapters, BFCL categories), so it is safe to
# stop and re-run at any point. Run detached:
#   setsid nohup bash results/benchmarks/v9_pipeline/resume_all.sh > results/benchmarks/v9_pipeline/resume_all.log 2>&1 < /dev/null & disown
set -euo pipefail
cd /home/mihai/Projects/gnn-experiment
gpu_free() { local out; out=$(rocm-smi --showpids 2>&1); [[ $out == *"No KFD PIDs currently running"* ]]; }
gpu_free || { echo "GPU busy -- stop the other engine first" >&2; exit 3; }

echo "[$(date +%T)] 1/3 finish 9B v9 evaluation (financial + old-arm HumanEval for all domains)"
models/runtime_next_bins_current/runtime-next-qwen35_9b --port 8003 > results/benchmarks/v9_pipeline/server_9b.log 2>&1 &
S=$!; trap 'kill $S 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do curl -sf http://127.0.0.1:8003/health >/dev/null && break; sleep 1; done
python3 evals/factory/compare_adapter_generations.py --size 9b --old v7 --new v9 \
  --domains agentic_coding,python_modern,python_web,astral,duckdb,postgresql,financial \
  >> results/benchmarks/v9_pipeline/compare_9b.log 2>&1
kill $S; wait $S 2>/dev/null || true; trap - EXIT
for _ in $(seq 1 60); do gpu_free && break; sleep 1; done

echo "[$(date +%T)] 2/3 v10 at 4B"
VERSION=v10 OLD=v9 STOP=0.036 BASE_FROM=results/benchmarks/adapter_generations_v7_vs_v9_4b.json \
  bash apps/factory/run_adapter_pipeline.sh 4b agentic_coding python_modern python_web financial_planning astral duckdb postgresql

echo "[$(date +%T)] 3/3 MiMo"
NO_WAIT=1 bash results/benchmarks/v9_pipeline/then_mimo.sh
echo "[$(date +%T)] ALL DONE"
