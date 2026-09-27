#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-datasets2_V40_textureScale_k11_raw025_fusionWarm3_lr1e5_clip03_bs3_ep200_gpu1_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/train_datasets2_V30_irRadTexSep_noMax_gain010_gt035_vis005_step005.sh" \
  --lr 0.000010 \
  --grad_clip_norm_g 0.30 \
  --texture_bootstrap_epochs 3 \
  --texture_bootstrap_fusion_scale 0.10 \
  --decom_radiation_ksize 11 \
  --decom_radiation_iters 1 \
  --ir_radiation_raw_blend 0.25 \
  --gt_radiation_ksize 11 \
  --gt_radiation_raw_blend 0.25 \
  "$@"
