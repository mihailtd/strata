#!/usr/bin/env bash
set -euo pipefail
cd /home/mihai/Projects/gnn-experiment
until grep -q "9b eval done" results/benchmarks/v9_pipeline/resume_9b.log 2>/dev/null; do
  pgrep -f "resume_9b_eval.sh" >/dev/null || { grep -q "9b eval done" results/benchmarks/v9_pipeline/resume_9b.log || { echo "9B eval ended without finishing" >&2; exit 1; }; }
  sleep 30
done
sleep 10
echo "[$(date +%T)] starting v10 4b"
VERSION=v10 OLD=v9 STOP=0.036 BASE_FROM=results/benchmarks/adapter_generations_v7_vs_v9_4b.json \
  bash apps/factory/run_adapter_pipeline.sh 4b agentic_coding python_modern python_web financial_planning astral duckdb postgresql
echo "[$(date +%T)] v10 4b done"
