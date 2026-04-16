#!/bin/bash
# =============================================================================
# Training script for Defocus Deblurring on DPDD dataset
#
# Usage:
#   bash scripts/train_dpdd.sh          # Train from scratch (2 GPUs)
#   bash scripts/train_dpdd.sh resume   # Resume from latest checkpoint
# =============================================================================

set -euo pipefail

# ---- Configuration ----
export CUDA_VISIBLE_DEVICES=1
NUM_GPUS=1
MAIN_PORT=4562
CONFIG=configs/train/train_dpdd.yaml
EXP_DIR=exp_dpdd_defocus
RESUME_PATH="/raid/danielchen/defous-deblur/DeblurDiff/exp_dpdd_defocus/run_20260416_162830/checkpoints/step_0004000_20260416_191326.pt"

# ---- Pre-flight checks ----
echo "============================================"
echo "  Defocus Deblurring Training (DPDD)"
echo "============================================"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} devices)"
echo "Config:      ${CONFIG}"
echo "Exp dir:     ${EXP_DIR}"
echo "--------------------------------------------"

# Check SD weights exist
SD_WEIGHTS=checkpoints/v2-1_512-ema-pruned.safetensors
if [[ ! -f "$SD_WEIGHTS" ]]; then
    echo "ERROR: SD weights not found at $SD_WEIGHTS"
    exit 1
fi

# Check dataset
DATASET_ROOT=/raid/danielchen/defous-deblur/dd_dp_dataset_png
if [[ ! -d "$DATASET_ROOT/train_c/source" ]]; then
    echo "ERROR: DPDD dataset not found at $DATASET_ROOT"
    exit 1
fi

echo "SD weights:  OK ($SD_WEIGHTS)"
echo "Dataset:     OK ($DATASET_ROOT)"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo "============================================"
echo ""

# ---- Launch training ----
accelerate launch \
    --num_processes ${NUM_GPUS} \
    --main_process_port ${MAIN_PORT} \
    train_dpdd.py \
    --config ${CONFIG} \
    --resume ${RESUME_PATH}

echo ""
echo "Training complete!"
echo "Checkpoints saved to: ${EXP_DIR}/checkpoints/"
echo "TensorBoard logs:     ${EXP_DIR}/"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir ${EXP_DIR} --port 6006"
