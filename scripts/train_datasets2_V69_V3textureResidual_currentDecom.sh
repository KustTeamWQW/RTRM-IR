#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-datasets2_V69_V3textureResidual_currentDecom_keepRadFusion_bs3_ep380_gpu1_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME
export GPU="${GPU:-1}"
export BATCH_SIZE="${BATCH_SIZE:-3}"
export NUM_THREADS="${NUM_THREADS:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/train_datasets2_V63_TirDirectInternal_noInternalSplit.sh" \
  --texture_residual_gain 0.16 \
  --texture_residual_max_delta 0.35 \
  --texture_residual_gain_per_step 0 \
  --texture_residual_steps 2 \
  --texture_residual_step_scales 1.00,1.00 \
  --texture_residual_step_loss_weights 1.00,1.00 \
  --texture_residual_ir_focus_gain 0.18 \
  --texture_ir_skip 0.08 \
  --texture_lcn_blend 0.42 \
  --texture_vis_supervision_mode gradtex \
  --lambda_texture_gt 0.75 \
  --lambda_texture_vis 0.04 \
  --lambda_texture_step 0.005 \
  "$@"
