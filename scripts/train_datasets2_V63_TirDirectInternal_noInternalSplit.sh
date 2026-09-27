#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-datasets2_V63_TirDirectInternal_noInternalSplit_V3texture_keepV58fusion_bs3_ep380_gpu1_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME
export GPU="${GPU:-1}"
export BATCH_SIZE="${BATCH_SIZE:-3}"
export NUM_THREADS="${NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/train_datasets2_V20_clean3branch_TirTexture.sh" \
  --decom_mode ir_radiation_texture_ref \
  --decom_radiation_ksize 31 \
  --decom_radiation_iters 2 \
  --decom_guided_ksize 15 \
  --decom_guided_eps 0.01 \
  --gt_radiation_ksize 31 \
  --gt_radiation_raw_blend 0.20 \
  --ir_radiation_raw_blend 0.00 \
  --texture_domain_norm_enabled 0 \
  --niter 380 \
  --niter_decay 0 \
  --texture_residual_gain 0.16 \
  --texture_residual_max_delta 0.0 \
  --texture_residual_steps 2 \
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
  --radiation_grad_weight 0.00 \
  --lambda_fusion_I 1.80 \
  --lambda_fusion_l1 1.20 \
  --lambda_fusion_ssim 6.00 \
  --lambda_fusion_psnr 8.50 \
  --lambda_fusion_charb 0.00 \
  --lambda_fusion_grad 0.00 \
  --lambda_fusion_structure 0.00 \
  --lambda_fusion_lap 0.00 \
  --lambda_fusion_lowfreq 0.00 \
  --lambda_fusion_stats 0.00 \
  --lambda_fusion_residual_gt 0.00 \
  --lambda_fusion_brightness_gt 0.00 \
  --val_during_train 1 \
  --val_freq 1 \
  --val_max_images 64 \
  --val_save_label best_val \
  "$@"
