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
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
MAIN_PORT=4562
CONFIG=configs/train/train_dpdd.yaml
EXP_DIR=exp_dpdd_defocus

# ---- Resume from checkpoint (optional) ----
RESUME_FLAG=""
if [[ "${1:-}" == "resume" ]]; then
    # Find latest checkpoint
    LATEST_CKPT=$(ls -1t "${EXP_DIR}/checkpoints/"*.pt 2>/dev/null | head -n1)
    if [[ -n "$LATEST_CKPT" ]]; then
        echo "Resuming from: $LATEST_CKPT"
        # Temporarily patch config to set resume path
        # (the config's train.resume field controls this)
        RESUME_FLAG="--resume $LATEST_CKPT"
    else
        echo "No checkpoint found in ${EXP_DIR}/checkpoints/, training from scratch."
    fi
fi

# ---- Pre-flight checks ----
echo "============================================"
echo "  Defocus Deblurring Training (DPDD)"
echo "============================================"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS} devices)"
echo "Config:      ${CONFIG}"
echo "Exp dir:     ${EXP_DIR}"
echo "--------------------------------------------"

# Check SD weights exist
SD_WEIGHTS=checkpoints/v2-1_768-ema-pruned.safetensors
if [[ ! -f "$SD_WEIGHTS" ]]; then
    echo "ERROR: SD weights not found at $SD_WEIGHTS"
    exit 1
fi

# Check dataset
DATASET_ROOT=/ssd1/chingheng/Deblurring-Dataset/dd_dp_dataset_png
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
    --config ${CONFIG}

echo ""
echo "Training complete!"
echo "Checkpoints saved to: ${EXP_DIR}/checkpoints/"
echo "TensorBoard logs:     ${EXP_DIR}/"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir ${EXP_DIR} --port 6006"
