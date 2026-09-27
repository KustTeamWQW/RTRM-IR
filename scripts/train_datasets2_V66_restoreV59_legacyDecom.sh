#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-datasets2_V66_restoreV59_legacyDecom_V3texture_keepV58fusion_bs3_ep380_gpu1_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/train_datasets2_V63_TirDirectInternal_noInternalSplit.sh" \
  --decom_mode legacy \
  --legacy_radiation_from_decom 0 \
  --texture_residual_ir_focus_gain 0.18 \
  --texture_residual_gain 0.16 \
  --texture_ir_skip 0.08 \
  --texture_lcn_blend 0.42 \
  --texture_vis_supervision_mode gradtex \
  --lambda_texture_gt 0.75 \
  --lambda_texture_vis 0.04 \
  --lambda_texture_step 0.005 \
  --lambda_texture_distill 0.01 \
  --lambda_texture_lowfreq 0.01 \
  --lambda_ir_hf 0.02 \
  --lambda_detail_gt 0.02 \
  --lambda_decom 0.10 \
  --lambda_radiation 0.25 \
  --niter 380 \
  --niter_decay 0 \
  --val_during_train 1 \
  --val_freq 1 \
  --val_max_images 64 \
  --val_save_label best_val \
  "$@"
