#!/usr/bin/env bash
set -euo pipefail
cd /home/mihai/Projects/gnn-experiment
echo "[$(date +%T)] waiting for corpus generator (pid 90562)"
while kill -0 90562 2>/dev/null; do sleep 20; done
grep -c "^WROTE" results/benchmarks/corpus_v7_thinking_generation_rest2.log | grep -qx 5 \
  || { echo "generator did not write all 5 remaining corpora -- see corpus_v7_thinking_generation_rest2.log" >&2; exit 1; }
python3 apps/factory/corpus/assemble_corpus_v7.py python_modern python_web financial_planning astral duckdb postgresql
echo "[$(date +%T)] stopping 9B teacher"
kill $(ps -eo pid,comm | awk '$2 ~ /^runtime-next/ {print $1}') 2>/dev/null || true
ALL=(agentic_coding python_modern python_web financial_planning astral duckdb postgresql)
for spec in "4b ${ALL[*]}" "9b ${ALL[*]}" "2b python_modern agentic_coding" "0_8b python_modern agentic_coding"; do
  echo "[$(date +%T)] ===== pipeline $spec"
  bash apps/factory/run_v9_pipeline.sh $spec
done
echo "[$(date +%T)] ALL SIZES DONE"
