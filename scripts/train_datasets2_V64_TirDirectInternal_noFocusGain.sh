#!/usr/bin/env bash
set -euo pipefail

EXP_NAME="${EXP_NAME:-datasets2_V64_TirDirectInternal_noFocusGain_V3texture_keepV58fusion_bs3_ep380_gpu1_$(date +%Y%m%d_%H%M%S)}"
export EXP_NAME

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$SCRIPT_DIR/train_datasets2_V63_TirDirectInternal_noInternalSplit.sh" \
  --texture_residual_ir_focus_gain 0.00 \
  "$@"
