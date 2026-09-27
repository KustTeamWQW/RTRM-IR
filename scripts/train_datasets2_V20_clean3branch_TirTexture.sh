#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-datasets2_V20_clean3branch_TirTexture_noMax_fusion3_val_bs3_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/train_datasets2_V12_signedres_stepind_shortval.sh" \
  --texture_residual_max_delta 0.0 \
  --texture_residual_gain 0.30 \
  --texture_residual_gain_per_step 0 \
  --texture_residual_steps 2 \
  --texture_ir_skip 0.00 \
  --fusion_detach_texture_epochs 0 \
  --val_during_train 1 \
  --val_freq 1 \
  --val_max_images 64 \
  --val_save_label best_val \
  --lambda_texture_gt 0.16875 \
  --lambda_texture_vis 0.16000 \
  --lambda_texture_step 0.150 \
  --lambda_texture_ag 0.00 \
  --lambda_texture_distill 0.00 \
  --lambda_texture_ms_grad 0.00 \
  --lambda_texture_contrast 0.00 \
  --lambda_texture_lowfreq 0.00 \
  --lambda_ir_hf 0.000 \
  --lambda_detail_gt 0.00 \
  --lambda_decom 0.10 \
  --lambda_radiation 0.20 \
  --radiation_grad_weight 0.00 \
  --lambda_fusion_l1 1.20 \
  --lambda_fusion_ssim 6.00 \
  --lambda_fusion_psnr 8.50 \
  --lambda_fusion_charb 0.00 \
  --lambda_fusion_grad 0.000 \
  --lambda_fusion_structure 0.00 \
  --lambda_fusion_lap 0.00 \
  --lambda_fusion_lowfreq 0.00 \
  --lambda_fusion_stats 0.00 \
  --lambda_fusion_residual_gt 0.00 \
  --lambda_fusion_brightness_gt 0.00 \
  "$@"
