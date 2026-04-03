#!/bin/bash
set -euo pipefail

# Recommended placement:
# repo_root/train_stage2_ddp.sh
#
# Usage:
#   bash train_stage2_ddp.sh gtn_lite
#   bash train_stage2_ddp.sh full_gtn
#
# Optional env overrides:
#   CUDA_VISIBLE_DEVICES=0,1 bash train_stage2_ddp.sh gtn_lite
#   STAGE1_CKPT=/path/to/classifier.pt bash train_stage2_ddp.sh full_gtn

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sheetagent

echo "Cleaning up old Stage2 processes..."
pkill -9 -f stage2_gtn.py || true
sleep 2

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

GRAPH_MODE=${1:-gtn_lite}
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="stage2_${GRAPH_MODE}_${CURRENT_TIME}.log"
RUN_NAME="${GRAPH_MODE}_${CURRENT_TIME}"

MODEL_NAME="${MODEL_NAME:-local_models/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594}"
DATA_DIR="${DATA_DIR:-data}"

# Prefer the new Stage1 output path produced by the current Bi-Encoder trainer.
if [[ -n "${STAGE1_CKPT:-}" ]]; then
  RESOLVED_STAGE1_CKPT="$STAGE1_CKPT"
elif [[ -f "outputs/stage1_biencoder/best_model/classifier.pt" ]]; then
  RESOLVED_STAGE1_CKPT="outputs/stage1_biencoder/best_model/classifier.pt"
elif [[ -f "best_model/classifier.pt" ]]; then
  RESOLVED_STAGE1_CKPT="best_model/classifier.pt"
else
  echo "ERROR: cannot find Stage1 checkpoint."
  echo "Expected one of:"
  echo "  outputs/stage1_biencoder/best_model/classifier.pt"
  echo "  best_model/classifier.pt"
  echo "Or pass it explicitly:"
  echo "  STAGE1_CKPT=/path/to/classifier.pt bash train_stage2_ddp.sh ${GRAPH_MODE}"
  exit 1
fi

if [[ ! -f "${DATA_DIR}/nway_train.json" ]]; then
  echo "ERROR: missing ${DATA_DIR}/nway_train.json"
  exit 1
fi

if [[ ! -f "${DATA_DIR}/nway_eval.json" ]]; then
  echo "ERROR: missing ${DATA_DIR}/nway_eval.json"
  exit 1
fi

GPU_COUNT=$(python - <<'PY'
import os
print(len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if x.strip()]))
PY
)

{
  echo "=============================="
  echo "Run started at $(date)"
  echo "Graph mode: ${GRAPH_MODE}"
  echo "Log file: ${LOG_FILE}"
  echo "Run name: ${RUN_NAME}"
  echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
  echo "Data dir: ${DATA_DIR}"
  echo "Stage1 checkpoint: ${RESOLVED_STAGE1_CKPT}"
  echo "=============================="
} >> "$LOG_FILE"

nohup torchrun --nproc_per_node="${GPU_COUNT}" stage2_gtn.py \
  --graph-mode "$GRAPH_MODE" \
  --run-name "$RUN_NAME" \
  --data-dir "$DATA_DIR" \
  --model-name "$MODEL_NAME" \
  --stage1-checkpoint "$RESOLVED_STAGE1_CKPT" \
  --use-tensorboard \
  --tensorboard-logdir runs/stage2_gtn \
  --num-epochs 20 \
  --batch-size 4 \
  --learning-rate 2e-5 \
  --max-length 256 \
  --max-query-length 64 \
  --max-workspace-size 10 \
  --max-header-texts 12 \
  --include-shape-feature \
  --include-source-feature \
  --embedding-strategy cls \
  --normalize-embeddings \
  --num-gat-layers 1 \
  --tau 0.07 \
  --lambda-align 0.10 \
  --lambda-subgraph 0.05 \
  --bce-weight 1.0 \
  >> "$LOG_FILE" 2>&1 &

PID=$!

echo "Stage2 ${GRAPH_MODE} training started!"
echo "PID (torchrun): $PID"
echo "Log file created: $LOG_FILE"
echo "Stage1 checkpoint: $RESOLVED_STAGE1_CKPT"
echo "Check progress: tail -f $LOG_FILE"
