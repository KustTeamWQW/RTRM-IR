#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-datasets2_V28_irGuidedSep_gain010_max1_gt035_vis005_step005_bs3_ep200_gpu1_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/train_datasets2_V20_clean3branch_TirTexture.sh" \
  --decom_mode ir_guided_ref \
  --decom_guided_ksize 15 \
  --decom_guided_eps 0.010 \
  --niter 200 \
  --niter_decay 0 \
  --texture_residual_max_delta 1.0 \
  --texture_residual_gain 0.10 \
  --lambda_texture_gt 0.35000 \
  --lambda_texture_vis 0.05000 \
  --lambda_texture_step 0.050 \
  --lambda_radiation 0.25 \
  "$@"
