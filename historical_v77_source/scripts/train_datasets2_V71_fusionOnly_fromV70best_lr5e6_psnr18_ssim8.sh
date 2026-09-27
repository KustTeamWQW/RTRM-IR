#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-datasets2_V71_fusionOnly_fromV70best_lr5e6_psnr18_ssim8_bs3_ep520_gpu1_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME
export GPU="${GPU:-1}"
export BATCH_SIZE="${BATCH_SIZE:-3}"
export NUM_THREADS="${NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/train_datasets2_V69_V3textureResidual_currentDecom.sh" \
  --continue_train \
  --epoch best_val \
  --epoch_count 431 \
  --lr 0.000005 \
  --niter 520 \
  --niter_decay 0 \
  --fusion_only_finetune 1 \
  --fusion_only_zero_aux 1 \
  --lambda_fusion_l1 2.00 \
  --lambda_fusion_psnr 18.00 \
  --lambda_fusion_ssim 8.00 \
  "$@"
