#!/bin/bash
set -euo pipefail

source /data/Albus/miniconda3/etc/profile.d/conda.sh
conda activate agentsheet310

echo "Cleaning up old Stage2 v2 processes..."
pkill -9 -f stage2_gtn_v2_updated.py || true
sleep 2

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="stage2_gtn_v2_${CURRENT_TIME}"
OUTPUT_DIR="outputs/stage2_gtn_v2"
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

MODEL_NAME="${MODEL_NAME:-/root/sheetagentresearch/proactivesheetagent/local_models/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594}"
DATA_DIR="${DATA_DIR:-data}"
GTN_LAYERS="${GTN_LAYERS:-2}"
GTN_CHANNELS="${GTN_CHANNELS:-4}"
NEG_RATIO="${NEG_RATIO:-0.5}"
LAMBDA_NODE="${LAMBDA_NODE:-0.2}"
POS_WEIGHT="${POS_WEIGHT:-2.0}"

if [[ -n "${STAGE1_CKPT:-}" ]]; then
  RESOLVED_STAGE1_CKPT="$STAGE1_CKPT"
elif [[ -f "best_model/classifier.pt" ]]; then
  RESOLVED_STAGE1_CKPT="best_model/classifier.pt"
elif [[ -f "final_model/classifier.pt" ]]; then
  RESOLVED_STAGE1_CKPT="final_model/classifier.pt"
else
  echo "ERROR: cannot find Stage1 checkpoint."
  exit 1
fi

GPU_COUNT=$(python - <<'PY'
import os
print(len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if x.strip()]))
PY
)

mkdir -p "${LOG_DIR}" "runs/stage2_gtn_v2"

nohup torchrun --nproc_per_node="${GPU_COUNT}" stage2_gtn_v2_updated.py \
  --run-name "$RUN_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --data-dir "$DATA_DIR" \
  --features-file "${DATA_DIR}/sheets.json" \
  --model-name "$MODEL_NAME" \
  --stage1-checkpoint "$RESOLVED_STAGE1_CKPT" \
  --freeze-backbone \
  --use-tensorboard \
  --tensorboard-logdir runs/stage2_gtn_v2 \
  --num-epochs 50 \
  --batch-size 8 \
  --learning-rate 2e-5 \
  --max-length 256 \
  --max-query-length 64 \
  --max-workspace-size 10 \
  --max-header-texts 12 \
  --embedding-strategy cls \
  --normalize-embeddings \
  --num-gcn-layers 1 \
  --gtn-layers "$GTN_LAYERS" \
  --gtn-channels "$GTN_CHANNELS" \
  --negative-ratio "$NEG_RATIO" \
  --tau 0.07 \
  --lambda-align 0.10 \
  --lambda-node "$LAMBDA_NODE" \
  --pos-weight "$POS_WEIGHT" \
  >> "$LOG_FILE" 2>&1 &

PID=$!
echo "Stage2 GTN v2 training started!"
echo "PID (torchrun): $PID"
echo "Log file: $LOG_FILE"
