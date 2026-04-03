#!/bin/bash
set -euo pipefail

# Recommended placement:
# repo_root/train_ddp.sh

source /root/miniconda3/etc/profile.d/conda.sh
conda activate sheetagent

echo "Cleaning up old biencoder training processes..."
pkill -9 -f biencoder_model.py || true
sleep 2

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

GPU_COUNT=$(python - <<'PY'
import os
print(len([x for x in os.environ.get("CUDA_VISIBLE_DEVICES","").split(",") if x.strip()]))
PY
)

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="training_ddp_${CURRENT_TIME}.log"
RUN_NAME="stage1-biencoder_${CURRENT_TIME}"

echo "==============================" >> "$LOG_FILE"
echo "Run started at $(date)" >> "$LOG_FILE"
echo "Log file: $LOG_FILE" >> "$LOG_FILE"
echo "Mode: Distributed Data Parallel (DDP)" >> "$LOG_FILE"
echo "GPUs: $CUDA_VISIBLE_DEVICES" >> "$LOG_FILE"
echo "==============================" >> "$LOG_FILE"

nohup torchrun --nproc_per_node="${GPU_COUNT}" biencoder_model.py \
  --data-dir data \
  --output-dir outputs/stage1_biencoder \
  --run-name "$RUN_NAME" \
  --use-tensorboard \
  --tensorboard-logdir runs/stage1_biencoder \
  --num-epochs 50 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --max-length 256 \
  --embedding-strategy cls \
  --include-shape-feature \
  --include-source-feature \
  >> "$LOG_FILE" 2>&1 &

PID=$!

echo "Stage1 Bi-Encoder DDP training started!"
echo "PID (torchrun): $PID"
echo "Log file: $LOG_FILE"
echo "Run name: $RUN_NAME"
echo "Check progress: tail -f $LOG_FILE"
