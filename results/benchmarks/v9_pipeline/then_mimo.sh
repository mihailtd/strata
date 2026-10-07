#!/usr/bin/env bash
set -euo pipefail
cd /home/mihai/Projects/gnn-experiment
MIMO="/home/mihai/.cache/huggingface/hub/models--XiaomiMiMo--MiMo-V2.6-Distill-Qwen-9B/snapshots/2367e865d009c13ac81713a2878291d33ab28177/"
BIN=models/runtime_next_bins_current/runtime-next-qwen35_9b
L=results/benchmarks/mimo
mkdir -p $L
gpu_free() { local out; out=$(rocm-smi --showpids 2>&1); [[ $out == *"No KFD PIDs currently running"* ]]; }
wait_free() { for _ in $(seq 1 120); do gpu_free && return 0; sleep 1; done; echo "GPU busy" >&2; exit 3; }
if [ -z "${NO_WAIT:-}" ]; then until grep -qE "v10 4b done|ended without|Error|exit" results/benchmarks/v9_pipeline/then_v10_4b.log 2>/dev/null && ! pgrep -f then_v10_4b.sh >/dev/null; do sleep 30; done; fi
echo "[$(date +%T)] v10 finished; starting MiMo"

serve() { wait_free; RUNTIME_NEXT_MODEL_DIR="$1" $BIN --port 8003 > "$2" 2>&1 & SRV=$!;
          for _ in $(seq 1 240); do curl -sf http://127.0.0.1:8003/health >/dev/null && return 0; sleep 1; done; echo "server down" >&2; exit 5; }
stop() { kill $SRV 2>/dev/null || true; wait $SRV 2>/dev/null || true; }

# 1. smoke: own template + stop set in the log, one chat, one tool call
serve "$MIMO" $L/server_smoke.log
grep "chat template" $L/server_smoke.log
curl -s http://127.0.0.1:8003/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"What is 15% of 240?"}],"max_tokens":1024,"temperature":0}' > $L/smoke_chat.json
curl -s http://127.0.0.1:8003/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"What is the weather in Paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":1024,"temperature":0}' > $L/smoke_tool.json
python3 -c "import json;[print(f, json.load(open('$L/'+f))['choices'][0]['finish_reason'], json.dumps(json.load(open('$L/'+f))['choices'][0]['message'].get('tool_calls'))[:200]) for f in ('smoke_chat.json','smoke_tool.json')]"

# 2. HumanEval on MiMo itself (same prompts/budget as the adapter evals)
python3 evals/factory/compare_adapter_generations.py --size 9b --base-only --out results/benchmarks/humaneval_mimo_v2.6_distill_9b.json > $L/humaneval.log 2>&1
stop

# 3. BFCL, MiMo and Qwen3.5-9B, both under their own (HF-parity) templates
RUNTIME_NEXT_MODEL_DIR="$MIMO" python3 -m benchmarks.bfcl.run_size_sweep --bin-dir models/runtime_next_bins_current --sizes qwen35_9b --tag mimo_v2.6 > $L/bfcl_mimo.log 2>&1
python3 -m benchmarks.bfcl.run_size_sweep --bin-dir models/runtime_next_bins_current --sizes qwen35_9b --tag hftemplate > $L/bfcl_qwen9b_hftemplate.log 2>&1
echo "[$(date +%T)] MiMo done"
