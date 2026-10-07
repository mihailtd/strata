#!/usr/bin/env bash
# Adapter v9 pipeline for ONE base-model size: train every listed domain on
# corpus v7 in the served chat format, then compare base / shipped / v9 through
# runtime-next. Single-engine rule enforced: the GPU must be free before each
# stage, and every stage leaves it free.
#
# Each size reproduces the regime its SHIPPED adapters were trained with, so the
# v9 difference stays attributable to the corpus + format change alone:
#   4b    stop at first ||dW||/||W|| check >= 0.071, cap 900   (as the 4B v7s)
#   9b    stop 0.075, cap 150, bs1 x ga4, VRAM cap 23 GB       (train_all_9b_experts.py)
#   2b/0_8b  150 fixed steps, geometry stop off                (their v7/v8 regime.json)
#
# Every trained adapter must end with ||dW||/||W|| > 0.01 as recorded in its
# regime.json -- a guard against shipping an untrained adapter. (Six 9B v7s record
# ~7e-6 there, but their weights measure normal: that telemetry was wrong, see the
# v9 CHANGELOG retraction. A low value here means "check the weights", not "null".)
#
#   bash apps/factory/run_v9_pipeline.sh 4b agentic_coding python_modern ...
set -euo pipefail
cd "$(dirname "$0")/../.."

SIZE="$1"; shift
DOMAINS=("$@")
LOGDIR=results/benchmarks/v9_pipeline
mkdir -p "$LOGDIR"

case "$SIZE" in
  4b)   MODEL=Qwen/Qwen3.5-4B;   SUFFIX="";      BIN=qwen35_4b
        FLAGS=(--stop-at-dw-over-w 0.071 --max-steps 900 --gradient-checkpointing) ;;
  9b)   MODEL=Qwen/Qwen3.5-9B;   SUFFIX="_9b";   BIN=qwen35_9b
        FLAGS=(--stop-at-dw-over-w 0.075 --max-steps 150 --batch-size 1 --grad-accum 4
               --gradient-checkpointing --vram-cap-gb 23.0) ;;
  2b)   MODEL=Qwen/Qwen3.5-2B;   SUFFIX="_2b";   BIN=qwen35_2b
        FLAGS=(--stop-at-dw-over-w 0 --max-steps 150 --gradient-checkpointing) ;;
  0_8b) MODEL=Qwen/Qwen3.5-0.8B; SUFFIX="_0_8b"; BIN=qwen35_0_8b
        FLAGS=(--stop-at-dw-over-w 0 --max-steps 150 --gradient-checkpointing) ;;
  *) echo "unknown size $SIZE" >&2; exit 2 ;;
esac

# Capture first: under `set -o pipefail`, `rocm-smi | grep -q` reports failure on a
# MATCH -- grep exits early, rocm-smi takes SIGPIPE, pipefail propagates it.
gpu_free() { local out; out=$(rocm-smi --showpids 2>&1); [[ $out == *"No KFD PIDs currently running"* ]]; }
require_gpu_free() {
  for _ in $(seq 1 60); do gpu_free && return 0; sleep 1; done
  echo "GPU still busy -- refusing to start $1" >&2; rocm-smi --showpids >&2; exit 3
}
stem() { [ "$1" = financial_planning ] && echo financial || echo "$1"; }

# ---- train -------------------------------------------------------------------
for d in "${DOMAINS[@]}"; do
  out="results/adapters/m2_$(stem "$d")_r8a128_v9${SUFFIX}"
  if [ -f "$out/adapter_model.safetensors" ]; then echo "== $out exists, skipping training"; continue; fi
  require_gpu_free "training $d"
  echo "== train $SIZE $d -> $out"
  uv run python apps/factory/train_expert.py --domain "$d" --v9 --chat-format qwen --model-id "$MODEL" \
    --rank 8 --alpha 128 --lr 2e-4 --skip-alpha-calibration --out "$out" "${FLAGS[@]}" \
    > "$LOGDIR/train_${SIZE}_${d}.log" 2>&1
  dw=$(python3 -c "import json;print(json.load(open('$out/regime.json'))['merge_precision']['dw_over_w'])")
  echo "   final ||dW||/||W|| = $dw"
  python3 -c "import sys; sys.exit(0 if float('$dw') > 0.01 else 1)" \
    || { echo "NULL ADAPTER: $out finished at ||dW||/||W|| = $dw -- stopping the pipeline" >&2; exit 4; }
done

# ---- evaluate ------------------------------------------------------------------
require_gpu_free "the $SIZE eval server"
models/runtime_next_bins_current/runtime-next-$BIN --port 8003 > "$LOGDIR/server_${SIZE}.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do curl -sf http://127.0.0.1:8003/health >/dev/null && break; sleep 1; done
curl -sf http://127.0.0.1:8003/health >/dev/null || { echo "server did not come up" >&2; exit 5; }

eval_domains=$(for d in "${DOMAINS[@]}"; do stem "$d"; done | paste -sd,)
python3 evals/factory/compare_adapter_generations.py --size "$SIZE" --old v7 --new v9 --domains "$eval_domains" \
  > "$LOGDIR/compare_${SIZE}.log" 2>&1
kill $SERVER; wait $SERVER 2>/dev/null || true
trap - EXIT
require_gpu_free "exit"
echo "== $SIZE done"
