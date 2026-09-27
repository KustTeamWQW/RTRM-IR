#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-datasets2_V26_RtexToTvis_gain020_vis008_step008_bs3_ep200_gpu4_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/train_datasets2_V20_clean3branch_TirTexture.sh" \
  --niter 200 \
  --niter_decay 0 \
  --texture_residual_gain 0.20 \
  --lambda_texture_vis 0.08000 \
  --lambda_texture_step 0.080 \
  "$@"
