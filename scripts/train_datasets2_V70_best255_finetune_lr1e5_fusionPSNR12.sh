#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-datasets2_V70_best255_finetune_lr1e5_fusionPSNR12_bs3_ep430_gpu1_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME
export GPU="${GPU:-1}"
export BATCH_SIZE="${BATCH_SIZE:-3}"
export NUM_THREADS="${NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/train_datasets2_V69_V3textureResidual_currentDecom.sh" \
  --continue_train \
  --epoch best_val \
  --epoch_count 256 \
  --lr 0.000010 \
  --niter 430 \
  --niter_decay 0 \
  --lambda_fusion_l1 1.50 \
  --lambda_fusion_psnr 12.00 \
  --lambda_fusion_ssim 6.00 \
  "$@"
