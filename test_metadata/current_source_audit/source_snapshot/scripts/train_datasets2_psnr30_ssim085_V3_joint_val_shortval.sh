#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/jpc/jpc/VIFPM}"

export EXP_NAME="${EXP_NAME:-datasets2_psnr30_ssim085_V3_joint_val_shortval_$(date +%Y%m%d_%H%M%S)}"
export EVAL_LABELS="${EVAL_LABELS:-latest best best_val}"

exec bash "$ROOT_DIR/scripts/train_datasets2_psnr30_ssim085_V2_joint_shortval.sh" \
  --val_during_train 1 \
  --val_phase val \
  --val_freq "${VAL_FREQ:-2}" \
  --val_max_images "${VAL_MAX_IMAGES:-64}" \
  --val_save_label best_val \
  --val_ssim_weight "${VAL_SSIM_WEIGHT:-30.0}" \
  "$@"
